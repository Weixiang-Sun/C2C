"""
Bidirectional Universal Cache Training Script

Trains BidirectionalUniversalProjector with 4-path cross training:
  A2B (primary), B2A (reverse), A2A (reconstruction), B2B (reconstruction)
where A=teacher (Qwen2.5-0.5B) and B=base (Qwen3-0.6B).

Shared decoders force encoder Z-space alignment.
"""

import gc
import torch
import torch.nn as nn
import random
import types
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_scheduler
from transformers.cache_utils import DynamicCache
from torch.optim import AdamW
from tqdm import tqdm
import os
import json
import yaml
import argparse
import shutil
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import math
import contextlib

from rosetta.model.projector import create_projector, save_projector, BidirectionalUniversalProjector
from rosetta.train.dataset_adapters import ChatDataset, RosettaDataCollator, create_dataset
from rosetta.train.model_utils import last_aligned_sources, k_nearest_sources
from rosetta.utils.evaluate import set_default_chat_template


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def enable_full_determinism():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass


def broadcast_decision_from_rank0(decision: bool, distributed: bool, device: str, rank: int) -> bool:
    if not distributed:
        return decision
    if rank == 0:
        tensor_flag = torch.tensor([1 if decision else 0], device=device, dtype=torch.int)
    else:
        tensor_flag = torch.empty(1, device=device, dtype=torch.int)
    dist.broadcast(tensor_flag, src=0)
    return bool(tensor_flag.item())


def freeze_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def load_config(config_path: str) -> Dict[str, Any]:
    file_ext = os.path.splitext(config_path)[1].lower()
    with open(config_path, "r") as f:
        if file_ext == ".json":
            config = json.load(f)
        elif file_ext in [".yaml", ".yml"]:
            config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")
    return config


def clone_kv_cache(kv_cache: DynamicCache) -> DynamicCache:
    new_cache = DynamicCache()
    for k, v in zip(kv_cache.key_cache, kv_cache.value_cache):
        new_cache.key_cache.append(k.clone().detach())
        new_cache.value_cache.append(v.clone().detach())
    return new_cache


def _monkeypatch_qwen3_attention_forward(attn_module, new_k_cache, new_v_cache):
    """Monkeypatch Qwen3Attention.forward to inject fused KV cache (has q_norm/k_norm)."""
    from transformers.models.qwen3.modeling_qwen3 import (
        apply_rotary_pos_emb,
        eager_attention_forward,
        ALL_ATTENTION_FUNCTIONS,
    )
    orig_forward = attn_module.forward

    def patched_forward(self, hidden_states, position_embeddings, attention_mask,
                        past_key_value=None, cache_position=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if new_k_cache is not None and new_v_cache is not None:
            if key_states.shape == new_k_cache.shape:
                key_states = new_k_cache
            if value_states.shape == new_v_cache.shape:
                value_states = new_v_cache

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                attention_interface = eager_attention_forward
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, sliding_window=self.sliding_window, **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    attn_module.forward = types.MethodType(patched_forward, attn_module)
    return orig_forward


def _monkeypatch_qwen2_attention_forward(attn_module, new_k_cache, new_v_cache):
    """Monkeypatch Qwen2Attention.forward to inject fused KV cache (no q_norm/k_norm)."""
    from transformers.models.qwen2.modeling_qwen2 import (
        apply_rotary_pos_emb,
        eager_attention_forward,
        ALL_ATTENTION_FUNCTIONS,
    )
    orig_forward = attn_module.forward

    def patched_forward(self, hidden_states, position_embeddings, attention_mask,
                        past_key_value=None, cache_position=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if new_k_cache is not None and new_v_cache is not None:
            if key_states.shape == new_k_cache.shape:
                key_states = new_k_cache
            if value_states.shape == new_v_cache.shape:
                value_states = new_v_cache

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                attention_interface = eager_attention_forward
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    attn_module.forward = types.MethodType(patched_forward, attn_module)
    return orig_forward


def monkeypatch_attention(model, layer_idx, new_k, new_v):
    """Auto-detect model type and apply the correct monkeypatch."""
    attn = model.model.layers[layer_idx].self_attn
    if hasattr(attn, 'k_norm'):
        return _monkeypatch_qwen3_attention_forward(attn, new_k, new_v)
    else:
        return _monkeypatch_qwen2_attention_forward(attn, new_k, new_v)


def remove_hooks(hook_handlers):
    for attn, orig_forward in hook_handlers:
        attn.forward = orig_forward


def build_fused_cache_and_hooks(
    projectors: nn.ModuleList,
    source_kv: DynamicCache,
    target_kv: DynamicCache,
    target_model,
    layer_mapping: Dict[int, List[Tuple[int, int]]],
    direction: str,
    seq_start: int,
    seq_end: int,
):
    """
    Apply projectors to build fused KV cache and monkeypatch target model attention.

    Args:
        projectors: nn.ModuleList of BidirectionalUniversalProjector
        source_kv: KV cache from source model
        target_kv: KV cache from target model
        target_model: the model that will do the forward pass
        layer_mapping: {target_layer_idx: [(source_layer_idx, projector_idx), ...]}
        direction: one of "A2B", "B2A", "A2A", "B2B"
        seq_start, seq_end: token range to project

    Returns:
        hook_handlers: list of (attn_module, orig_forward) for cleanup
        fused_kv: DynamicCache with projected values
    """
    fused_kv = clone_kv_cache(target_kv)
    hook_handlers = []

    for target_layer_idx, src_proj_pairs in layer_mapping.items():
        source_layer_idx, projector_idx = src_proj_pairs[0]  # K=1

        # Extract source and target KV slices
        src_k = source_kv.key_cache[source_layer_idx][:, :, seq_start:seq_end, :]
        src_v = source_kv.value_cache[source_layer_idx][:, :, seq_start:seq_end, :]
        tgt_k = target_kv.key_cache[target_layer_idx][:, :, seq_start:seq_end, :]
        tgt_v = target_kv.value_cache[target_layer_idx][:, :, seq_start:seq_end, :]

        # Project using the correct projector (always indexed by B layer)
        proj_k, proj_v = projectors[projector_idx].forward(
            (src_k, src_v), (tgt_k, tgt_v), direction=direction
        )

        # Update fused cache
        fused_kv.key_cache[target_layer_idx][:, :, seq_start:seq_end, :] = proj_k
        fused_kv.value_cache[target_layer_idx][:, :, seq_start:seq_end, :] = proj_v

    # Monkeypatch all layers
    num_layers = target_model.config.num_hidden_layers
    for i in range(num_layers):
        new_k = fused_kv.key_cache[i][:, :, seq_start:seq_end, :]
        new_v = fused_kv.value_cache[i][:, :, seq_start:seq_end, :]
        orig_forward = monkeypatch_attention(target_model, i, new_k, new_v)
        attn = target_model.model.layers[i].self_attn
        hook_handlers.append((attn, orig_forward))

    return hook_handlers, fused_kv


def build_mapping_with_projector_idx(
    forward_mapping: Dict[int, List[int]],
) -> Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, List[Tuple[int, int]]]]:
    """
    Build explicit (source_layer, projector_idx) mappings for both directions.

    Forward mapping input: {B_layer: [A_layer, ...]}  (from last_aligned_sources)
    Projectors are indexed by B_layer.

    Returns:
        forward_with_idx: {B_layer: [(A_layer, projector_idx=B_layer), ...]}
            Used for A2B and B2B paths (target=B)
        reverse_with_idx: {A_layer: [(B_layer, projector_idx=B_layer), ...]}
            Used for B2A and A2A paths (target=A)
    """
    forward_with_idx: Dict[int, List[Tuple[int, int]]] = {}
    reverse_with_idx: Dict[int, List[Tuple[int, int]]] = {}

    for b_layer, a_layers in forward_mapping.items():
        projector_idx = b_layer  # projectors are indexed by B layer
        forward_with_idx[b_layer] = [(a_layer, projector_idx) for a_layer in a_layers]
        for a_layer in a_layers:
            if a_layer not in reverse_with_idx:
                reverse_with_idx[a_layer] = []
            reverse_with_idx[a_layer].append((b_layer, projector_idx))

    return forward_with_idx, reverse_with_idx


def main():
    parser = argparse.ArgumentParser(description="Bidirectional Universal Cache Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_config = cfg["model"]
    training_config = cfg["training"]
    output_config = cfg["output"]
    data_config = cfg["data"]

    set_seed(training_config["seed"])
    enable_full_determinism()

    timestamped_output_dir = output_config["output_dir"]
    os.makedirs(timestamped_output_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(timestamped_output_dir, "config.json"))

    # Distributed setup
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank = 0
        local_rank = 0
        device = training_config.get("device", "cuda")
    is_main_process = rank == 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{output_config['wandb_config']['run_name']}_{timestamp}"
    if is_main_process:
        wandb.init(
            project=output_config["wandb_config"]["project"],
            name=run_name,
            config=cfg,
            mode=output_config["wandb_config"]["mode"],
            entity=output_config["wandb_config"].get("entity", None),
        )

    if is_main_process:
        print(f"Outputs will be saved to: {timestamped_output_dir}")

    # Load models
    dtype = torch.bfloat16
    # B = base model (Qwen3-0.6B), A = teacher model (Qwen2.5-0.5B)
    model_B = AutoModelForCausalLM.from_pretrained(
        model_config["base_model"], torch_dtype=dtype,
        attn_implementation=model_config.get("attn_implementation", None)
    ).to(device)
    model_A = AutoModelForCausalLM.from_pretrained(
        model_config["teacher_model"], torch_dtype=dtype,
        attn_implementation=model_config.get("attn_implementation", None)
    ).to(device)

    freeze_model(model_A)
    freeze_model(model_B)
    model_A.eval()
    model_B.eval()

    # Model dimensions
    dim_A = int(model_A.model.layers[0].self_attn.k_proj.out_features / model_A.config.num_key_value_heads)
    dim_B = int(model_B.model.layers[0].self_attn.k_proj.out_features / model_B.config.num_key_value_heads)
    heads_A = model_A.config.num_key_value_heads
    heads_B = model_B.config.num_key_value_heads
    num_layers_A = model_A.config.num_hidden_layers
    num_layers_B = model_B.config.num_hidden_layers

    if is_main_process:
        print(f"Model A ({model_config['teacher_model']}): {num_layers_A} layers, {heads_A} KV heads, head_dim={dim_A}")
        print(f"Model B ({model_config['base_model']}): {num_layers_B} layers, {heads_B} KV heads, head_dim={dim_B}")

    # Create projectors: one per target layer of B (for A2B direction)
    # These same projectors handle all 4 directions since BidirectionalUniversalProjector
    # has encoders/decoders for both sides
    projector_config = model_config["projector"]
    projector_params = projector_config["params"].copy()
    projector_params["dtype"] = dtype
    num_projectors = num_layers_B

    projector_list = nn.ModuleList()
    for _ in range(num_projectors):
        proj = create_projector(
            projector_config["type"],
            source_dim=dim_A,
            target_dim=dim_B,
            source_num_heads=heads_A,
            target_num_heads=heads_B,
            **projector_params,
        )
        projector_list.append(proj)
    projector_list = projector_list.to(device=device, dtype=dtype)

    # Layer mappings
    # Raw mapping: B_layer -> [A_layer] (from last_aligned_sources)
    raw_mapping = last_aligned_sources(num_layers_B, num_layers_A, k=1)
    # Build explicit (source_layer, projector_idx) mappings for both directions
    forward_mapping, reverse_mapping = build_mapping_with_projector_idx(raw_mapping)

    if is_main_process:
        print(f"Forward mapping (B_layer -> [(A_layer, proj_idx)]): {forward_mapping}")
        print(f"Reverse mapping (A_layer -> [(B_layer, proj_idx)]): {reverse_mapping}")

    # Loss weights
    loss_weights = model_config.get("loss_weights", {"AB": 1.0, "BA": 0.5, "AA": 0.3, "BB": 0.3})
    total_weight = sum(loss_weights.values())
    if is_main_process:
        print(f"Loss weights: {loss_weights}, total: {total_weight}")

    # Tokenizers
    tokenizer_B = AutoTokenizer.from_pretrained(model_config["base_model"])
    if tokenizer_B.pad_token is None:
        tokenizer_B.pad_token = tokenizer_B.eos_token
        tokenizer_B.pad_token_id = tokenizer_B.eos_token_id
    set_default_chat_template(tokenizer_B, model_config["base_model"])

    tokenizer_A = AutoTokenizer.from_pretrained(model_config["teacher_model"])
    if tokenizer_A.pad_token is None:
        tokenizer_A.pad_token = tokenizer_A.eos_token
        tokenizer_A.pad_token_id = tokenizer_A.eos_token_id
    set_default_chat_template(tokenizer_A, model_config["teacher_model"])

    # Dataset
    instruct_ds = create_dataset(dataset_type=data_config["type"], **data_config["kwargs"])
    full_dataset = ChatDataset(instruct_ds, tokenizer_B)

    train_size = int(data_config["train_ratio"] * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(full_dataset, [train_size, eval_size])

    per_device_batch_size = training_config["per_device_train_batch_size"]
    grad_accum_steps = training_config.get("gradient_accumulation_steps", 1)

    collator = RosettaDataCollator(
        slm_tokenizer=tokenizer_B,
        llm_tokenizer=tokenizer_A,
        max_length=training_config.get("max_length", 2048),
        do_alignment=False,
    )

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, seed=training_config["seed"]
        )
    else:
        train_sampler = None

    def _worker_init_fn(worker_id):
        np.random.seed(training_config["seed"] + worker_id)
        random.seed(training_config["seed"] + worker_id)

    train_loader = DataLoader(
        train_dataset, batch_size=per_device_batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        collate_fn=collator, worker_init_fn=_worker_init_fn,
    )

    # Wrap projectors for DDP
    if distributed:
        projector_list = DistributedDataParallel(
            projector_list, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True,
        )

    total_params = sum(p.numel() for p in projector_list.parameters())
    trainable_params = sum(p.numel() for p in projector_list.parameters() if p.requires_grad)
    if is_main_process:
        print(f"Total projector parameters: {total_params:,}")
        print(f"Trainable projector parameters: {trainable_params:,}")

    updates_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_steps = updates_per_epoch * training_config["num_epochs"]

    # Optimizer
    lr = training_config["learning_rate"]
    gate_params = []
    other_params = []
    for name, param in projector_list.named_parameters():
        if param.requires_grad:
            if "gate" in name:
                gate_params.append(param)
            else:
                other_params.append(param)

    optimizer = AdamW([
        {"params": gate_params, "lr": lr},
        {"params": other_params, "lr": lr},
    ], weight_decay=training_config["weight_decay"])

    scheduler = get_scheduler(
        training_config["scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=int(training_config["warmup_ratio"] * total_steps),
        num_training_steps=total_steps,
    )

    # Training loop
    if is_main_process:
        print(f"Starting training: {total_steps} steps, {training_config['num_epochs']} epochs")

    global_step = 0
    optimizer.zero_grad()
    projectors_ref = projector_list.module if isinstance(projector_list, DistributedDataParallel) else projector_list

    for epoch in range(training_config["num_epochs"]):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        projector_list.train()
        epoch_loss = 0.0
        progress_bar = tqdm(total=updates_per_epoch, desc=f"Epoch {epoch+1}", disable=not is_main_process)

        macro_step = 0
        accum_loss = 0.0
        micro_count = 0

        for batch_idx, batch in enumerate(train_loader):
            is_accum_step = ((batch_idx + 1) % grad_accum_steps) != 0
            sync_ctx = projector_list.no_sync() if distributed and hasattr(projector_list, "no_sync") and is_accum_step else contextlib.nullcontext()

            with sync_ctx:
                loss = bidir_train_step(
                    batch, model_A, model_B, projectors_ref,
                    tokenizer_A, tokenizer_B,
                    forward_mapping, reverse_mapping,
                    loss_weights, total_weight,
                    device, training_config.get("max_length", 2048),
                )
                true_loss = loss.detach().item()
                scaled_loss = loss / grad_accum_steps
                scaled_loss.backward()

            epoch_loss += true_loss
            accum_loss += true_loss
            micro_count += 1

            did_step = (not is_accum_step) or (batch_idx + 1 == len(train_loader))
            if did_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in projector_list.parameters() if p.requires_grad],
                    max_norm=training_config["max_grad_norm"],
                )
                grad_norm_value = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                macro_step += 1

                # Update temperatures
                for proj in projectors_ref:
                    if hasattr(proj, 'update_temperature'):
                        proj.update_temperature(global_step)

            if is_main_process and did_step:
                avg_loss = accum_loss / max(1, micro_count)
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })
                progress_bar.update(1)

                wandb.log({
                    "train/loss": avg_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm_value,
                    "train/epoch": epoch + (macro_step / updates_per_epoch),
                }, step=global_step)
                accum_loss = 0.0
                micro_count = 0

            # Checkpointing
            if did_step:
                want_save = (global_step % output_config["save_steps"] == 0)
                want_save = broadcast_decision_from_rank0(want_save, distributed, device, rank)
                if want_save and is_main_process:
                    checkpoint_dir = os.path.join(timestamped_output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    for i, proj in enumerate(projectors_ref):
                        torch.save(proj.state_dict(), os.path.join(checkpoint_dir, f"projector_{i}.pt"))
                        save_projector(proj, os.path.join(checkpoint_dir, f"projector_{i}.json"))
                    # Save forward mapping as projector_config.json (compatible with eval)
                    projector_config_dict = {
                        "0": {  # target_model_idx (base=B)
                            "1": {}  # source_model_idx (teacher=A)
                        }
                    }
                    for tgt_layer, src_proj_pairs in forward_mapping.items():
                        src_layer, proj_idx = src_proj_pairs[0]
                        projector_config_dict["0"]["1"][str(tgt_layer)] = [
                            [src_layer, proj_idx]
                        ]
                    with open(os.path.join(checkpoint_dir, "projector_config.json"), "w") as f:
                        json.dump(projector_config_dict, f)
                    torch.save({
                        "step": global_step, "epoch": epoch,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    }, os.path.join(checkpoint_dir, "training_state.pt"))
                    print(f"\nCheckpoint saved at step {global_step}")

        progress_bar.close()

    # Save final
    if is_main_process:
        final_dir = os.path.join(timestamped_output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        for i, proj in enumerate(projectors_ref):
            torch.save(proj.state_dict(), os.path.join(final_dir, f"projector_{i}.pt"))
            save_projector(proj, os.path.join(final_dir, f"projector_{i}.json"))
        projector_config_dict = {"0": {"1": {}}}
        for tgt_layer, src_layers in forward_mapping.items():
            projector_config_dict["0"]["1"][str(tgt_layer)] = [[src_layers[0], tgt_layer]]
        with open(os.path.join(final_dir, "projector_config.json"), "w") as f:
            json.dump(projector_config_dict, f)
        print("Training completed!")
        wandb.finish()

    if distributed:
        dist.destroy_process_group()


def bidir_train_step(
    batch: Dict[str, Any],
    model_A,  # teacher (Qwen2.5-0.5B)
    model_B,  # base (Qwen3-0.6B)
    projectors: nn.ModuleList,  # indexed by B layer
    tokenizer_A,
    tokenizer_B,
    forward_mapping: Dict[int, List[Tuple[int, int]]],  # B_layer -> [(A_layer, proj_idx)]
    reverse_mapping: Dict[int, List[Tuple[int, int]]],  # A_layer -> [(B_layer, proj_idx)]
    loss_weights: Dict[str, float],
    total_weight: float,
    device: str,
    max_length: int,
) -> torch.Tensor:
    """
    Single training step with 4 directional losses.
    """
    # Parse batch (from RosettaDataCollator)
    if isinstance(batch["input_ids"], list):
        input_ids_list = [ids.to(device) for ids in batch["input_ids"]]
        attention_mask_list = [m.to(device) for m in batch["attention_mask"]]
    else:
        input_ids_list = [batch["input_ids"].to(device)]
        attention_mask_list = [batch["attention_mask"].to(device)]

    labels_B = batch["labels"].to(device)
    kv_cache_index = [x.to(device) for x in batch["kv_cache_index"]]

    # Use base model (B) tokenized input for B, re-tokenize for A if needed
    ids_B = input_ids_list[0]
    mask_B = attention_mask_list[0]

    # Find instruction boundary from kv_cache_index
    # kv_cache_index[0] is the instruction section, kv_cache_index[1] is response
    instr_len = kv_cache_index[0].shape[1] if len(kv_cache_index) > 0 else 0
    resp_len = kv_cache_index[1].shape[1] if len(kv_cache_index) > 1 else ids_B.shape[1] - instr_len

    # For model A, use teacher tokenization if available
    if len(input_ids_list) > 1:
        ids_A = input_ids_list[1]
        mask_A = attention_mask_list[1]
    else:
        # Fallback: re-tokenize with model A's tokenizer
        # Decode from B's tokens and re-encode with A's tokenizer
        ids_A = ids_B
        mask_A = mask_B

    B_batch = ids_B.shape[0]
    seq_len_B = ids_B.shape[1]
    seq_len_A = ids_A.shape[1]

    # Step 1: Extract instruction-portion KV caches
    instr_ids_B = ids_B[:, :instr_len]
    instr_mask_B = mask_B[:, :instr_len]
    instr_ids_A = ids_A[:, :min(instr_len, seq_len_A)]
    instr_mask_A = mask_A[:, :min(instr_len, seq_len_A)]

    with torch.no_grad():
        out_A = model_A(input_ids=instr_ids_A, attention_mask=instr_mask_A, use_cache=True)
        KV_A = out_A.past_key_values
        out_B = model_B(input_ids=instr_ids_B, attention_mask=instr_mask_B, use_cache=True)
        KV_B = out_B.past_key_values

    # Detach KV caches so they're reusable across paths
    KV_A = clone_kv_cache(KV_A)
    KV_B = clone_kv_cache(KV_B)

    instr_len_A = instr_ids_A.shape[1]
    instr_len_B = instr_ids_B.shape[1]

    # Labels for model B (base): mask instruction tokens
    labels_B_full = labels_B.clone()

    # Labels for model A (teacher as target): use the teacher tokenized version
    # For B->A and A->A paths, teacher generates based on its own tokenization
    if len(input_ids_list) > 1:
        # Teacher tokenized labels: mask instruction portion
        labels_A_full = ids_A.clone()
        labels_A_full[:, :instr_len_A] = -100
    else:
        labels_A_full = labels_B_full.clone()

    # Response portion IDs
    resp_ids_B = ids_B[:, instr_len:]
    resp_mask_B = mask_B[:, :seq_len_B]  # Full mask for causal attention
    resp_ids_A = ids_A[:, instr_len_A:]
    resp_mask_A = mask_A[:, :seq_len_A]

    # Step 2: 4 loss paths
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    paths = []
    if loss_weights.get("AB", 0) > 0:
        paths.append(("A2B", KV_A, KV_B, model_B, resp_ids_B, resp_mask_B, labels_B_full[:, instr_len:],
                       forward_mapping, loss_weights["AB"]))
    if loss_weights.get("BA", 0) > 0:
        paths.append(("B2A", KV_B, KV_A, model_A, resp_ids_A, resp_mask_A, labels_A_full[:, instr_len_A:],
                       reverse_mapping, loss_weights["BA"]))
    if loss_weights.get("AA", 0) > 0:
        paths.append(("A2A", KV_A, KV_A, model_A, resp_ids_A, resp_mask_A, labels_A_full[:, instr_len_A:],
                       reverse_mapping, loss_weights["AA"]))
    if loss_weights.get("BB", 0) > 0:
        paths.append(("B2B", KV_B, KV_B, model_B, resp_ids_B, resp_mask_B, labels_B_full[:, instr_len:],
                       forward_mapping, loss_weights["BB"]))

    for direction, src_kv, tgt_kv, target_model, resp_ids, resp_mask, resp_labels, mapping, weight in paths:
        if resp_ids.shape[1] == 0:
            continue

        # Build fused cache and monkeypatch
        src_seq_len = src_kv.key_cache[0].shape[2]
        tgt_seq_len = tgt_kv.key_cache[0].shape[2]

        hook_handlers, fused_kv = build_fused_cache_and_hooks(
            projectors, src_kv, tgt_kv, target_model,
            mapping, direction, 0, tgt_seq_len,
        )

        try:
            # Forward pass with fused cache
            outputs = target_model(
                input_ids=resp_ids,
                attention_mask=resp_mask,
                past_key_values=fused_kv,
                labels=resp_labels,
                use_cache=False,
            )
            path_loss = outputs.loss
            if path_loss is not None:
                total_loss = total_loss + (weight / total_weight) * path_loss
        finally:
            remove_hooks(hook_handlers)

        # Free intermediate tensors
        del fused_kv, hook_handlers
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return total_loss


if __name__ == "__main__":
    main()
