"""
Cache Debate Experiment (Idea 1 — Identical Model Pair)

Two independent model instances exchange KV caches across debate rounds.

Protocol per example:
  Round 0:
    - A and B independently encode the question  → cache_A, cache_B (identical initially)
    - Bidirectional exchange: A ← proj(cache_B), B ← proj(cache_A)
    - A generates intermediate r_A (sampling, T=0.7)
    - B generates intermediate r_B (sampling, T=0.7)  →  r_A ≠ r_B due to stochasticity
  Round 1 .. D-1:
    - A encodes [question + r_B], B encodes [question + r_A]  →  caches now differ
    - Exchange ONLY the question-prefix portion of the caches (keeps shapes aligned)
    - Each generates next intermediate response
  Final:
    - Exchange question-prefix caches one more time
    - A generates the final answer (greedy) from its enriched context

Usage:
    python script/playground/debate_experiment.py \
        --checkpoint_dir 0.6B_identical_C2C/final \
        --base_model Qwen/Qwen3-0.6B \
        --benchmark arc \
        --debate_rounds 2 \
        --num_samples 200

    # Single exchange (no intermediate text, structurally same as iterative round 1)
    python script/playground/debate_experiment.py \
        --checkpoint_dir 0.6B_identical_C2C/final \
        --base_model Qwen/Qwen3-0.6B \
        --benchmark arc \
        --debate_rounds 1
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from rosetta.model.projector import load_projector
from rosetta.model.wrapper import clone_kv_cache, RosettaModel
from rosetta.utils.evaluate import set_default_chat_template


# ---------------------------------------------------------------------------
# Model loading (reuses projector infrastructure from RosettaModel)
# ---------------------------------------------------------------------------

def load_assets(base_model_path: str, checkpoint_dir: str, device: torch.device):
    """Load model, tokenizer, projector list, and projector config."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    set_default_chat_template(tokenizer, base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map={"": device}
    ).eval()

    num_projectors = len([f for f in os.listdir(checkpoint_dir) if re.match(r"projector_\d+\.pt", f)])
    if num_projectors == 0:
        raise FileNotFoundError(f"No projector_*.pt files in {checkpoint_dir}")

    projector_list = []
    for t in range(num_projectors):
        proj = load_projector(os.path.join(checkpoint_dir, f"projector_{t}.json")).to(device)
        pt_path = os.path.join(checkpoint_dir, f"projector_{t}.pt")
        if os.path.exists(pt_path):
            proj.load_state_dict(torch.load(pt_path, map_location=device), strict=False)
        proj.eval()
        projector_list.append(proj)

    # Load projector_config.json via a throwaway RosettaModel
    dummy = RosettaModel(model_list=[model, model], base_model_idx=0,
                         projector_list=projector_list).to(device).eval()
    dummy.load_projector_config(os.path.join(checkpoint_dir, "projector_config.json"))
    # projector_dict[target_model=0][source_model=1][target_layer] = [(src_layer, proj_idx), ...]
    proj_cfg = dummy.projector_dict[0][1]

    return model, tokenizer, projector_list, proj_cfg


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_kv_cache(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> DynamicCache:
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    cache = out.past_key_values
    if not isinstance(cache, DynamicCache):
        cache = DynamicCache.from_legacy_cache(cache)
    return cache


def apply_projection(
    source_cache: DynamicCache,
    target_cache: DynamicCache,
    projector_list,
    proj_cfg: Dict,
    seq_len: int,
) -> DynamicCache:
    """
    Project source_cache into target_cache's space for the first seq_len positions.
    Returns a new DynamicCache (target_cache outside seq_len is preserved as-is).
    """
    new_cache = clone_kv_cache(target_cache)
    for target_layer_idx, entries in proj_cfg.items():
        for source_layer_idx, projector_idx in entries:
            sk = source_cache.key_cache[source_layer_idx][:, :, :seq_len, :]
            sv = source_cache.value_cache[source_layer_idx][:, :, :seq_len, :]
            tk = target_cache.key_cache[target_layer_idx][:, :, :seq_len, :]
            tv = target_cache.value_cache[target_layer_idx][:, :, :seq_len, :]
            pk, pv = projector_list[projector_idx].forward((sk, sv), (tk, tv))
            new_cache.key_cache[target_layer_idx][:, :, :seq_len, :] = pk
            new_cache.value_cache[target_layer_idx][:, :, :seq_len, :] = pv
    return new_cache


@torch.no_grad()
def generate_from_cache(
    model,
    input_ids: torch.Tensor,
    full_cache: DynamicCache,
    max_new_tokens: int,
    tokenizer,
    device: torch.device,
    do_sample: bool = False,
    temperature: float = 0.7,
) -> str:
    """
    Generate tokens starting from a pre-built full cache.

    Strategy: trim cache to (seq_len - 1), re-run the last prompt token to get
    correct logits under the fused cache, then decode greedily or with sampling.
    """
    seq_len = input_ids.shape[1]

    # Trim cache to positions 0..seq_len-2
    trimmed = DynamicCache()
    for k, v in zip(full_cache.key_cache, full_cache.value_cache):
        trimmed.key_cache.append(k[:, :, :seq_len - 1, :].clone())
        trimmed.value_cache.append(v[:, :, :seq_len - 1, :].clone())

    last_token = input_ids[:, -1:]
    pos_ids = torch.tensor([[seq_len - 1]], device=device)

    out = model(input_ids=last_token, past_key_values=trimmed,
                use_cache=True, position_ids=pos_ids)
    current_logits = out.logits[:, -1, :]
    current_cache = out.past_key_values

    generated = []
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)

    for _ in range(max_new_tokens):
        if do_sample:
            probs = torch.softmax(current_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_token = current_logits.argmax(dim=-1)

        generated.append(next_token.item())
        if next_token.item() in eos_ids:
            break

        out = model(input_ids=next_token.unsqueeze(-1),
                    past_key_values=current_cache, use_cache=True)
        current_logits = out.logits[:, -1, :]
        current_cache = out.past_key_values

    return tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def build_arc_prompt(tokenizer, question: str, choices: Dict) -> str:
    opts = "\n".join(f"{label}. {text}" for label, text in zip(choices["label"], choices["text"]))
    content = (f"Answer the following multiple-choice question with a single letter.\n\n"
               f"{question}\n{opts}\n\nAnswer:")
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def build_debate_continuation(tokenizer, question: str, choices: Dict, other_response: str) -> str:
    """Prompt for re-encoding after seeing the other debater's intermediate response."""
    opts = "\n".join(f"{label}. {text}" for label, text in zip(choices["label"], choices["text"]))
    content = (f"Answer the following multiple-choice question with a single letter.\n\n"
               f"{question}\n{opts}\n\n"
               f"Another model suggests: {other_response.strip()}\n\n"
               f"Your answer:")
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def extract_arc_answer(text: str, valid_labels: List[str]) -> str:
    for char in reversed(text.strip()):
        if char in valid_labels:
            return char
    return "X"


# ---------------------------------------------------------------------------
# Debate runner
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_debate(
    model,
    projector_list,
    proj_cfg: Dict,
    tokenizer,
    question_prompt: str,
    question: str,
    choices: Dict,
    debate_rounds: int,
    intermediate_tokens: int,
    final_tokens: int,
    device: torch.device,
) -> str:
    """
    Run the full multi-round cache debate for one example. Returns A's final answer string.
    """
    enc = tokenizer(question_prompt, return_tensors="pt").to(device)
    question_ids = enc["input_ids"]
    question_mask = enc["attention_mask"]
    question_len = question_ids.shape[1]

    # Round 0: both encode the same question → identical caches
    cache_a = get_kv_cache(model, question_ids, question_mask)
    cache_b = clone_kv_cache(cache_a)

    # Current context length for each debater (starts at question_len)
    len_a = question_len
    len_b = question_len

    for round_idx in range(debate_rounds):
        is_final = (round_idx == debate_rounds - 1)

        # Exchange question-prefix portion of caches (always question_len tokens)
        fused_a = apply_projection(cache_b, cache_a, projector_list, proj_cfg, question_len)
        fused_b = apply_projection(cache_a, cache_b, projector_list, proj_cfg, question_len)

        if is_final:
            # Use fused_a with A's full context for final generation
            # Replace question-prefix in cache_a with fused version
            final_cache = clone_kv_cache(cache_a)
            for layer_idx in range(len(final_cache.key_cache)):
                final_cache.key_cache[layer_idx][:, :, :question_len, :] = \
                    fused_a.key_cache[layer_idx][:, :, :question_len, :]
                final_cache.value_cache[layer_idx][:, :, :question_len, :] = \
                    fused_a.value_cache[layer_idx][:, :, :question_len, :]

            # For final generation, use the current input_ids for A
            current_ids_a = question_ids  # will use cache trimmed to question_len-1
            answer = generate_from_cache(model, current_ids_a, final_cache,
                                         final_tokens, tokenizer, device, do_sample=False)
            return answer

        # Generate intermediate responses with sampling (so A and B diverge)
        r_a = generate_from_cache(model, question_ids, fused_a,
                                   intermediate_tokens, tokenizer, device,
                                   do_sample=True, temperature=0.7)
        r_b = generate_from_cache(model, question_ids, fused_b,
                                   intermediate_tokens, tokenizer, device,
                                   do_sample=True, temperature=0.7)

        # Re-encode: A reads [question + r_B], B reads [question + r_A]
        prompt_a_new = build_debate_continuation(tokenizer, question, choices, r_b)
        prompt_b_new = build_debate_continuation(tokenizer, question, choices, r_a)

        enc_a = tokenizer(prompt_a_new, return_tensors="pt").to(device)
        enc_b = tokenizer(prompt_b_new, return_tensors="pt").to(device)

        cache_a = get_kv_cache(model, enc_a["input_ids"], enc_a["attention_mask"])
        cache_b = get_kv_cache(model, enc_b["input_ids"], enc_b["attention_mask"])

        # Update lengths (for next round's final generation reference)
        len_a = enc_a["input_ids"].shape[1]
        len_b = enc_b["input_ids"].shape[1]

        # Update question_ids to A's new full prompt for final generation next round
        question_ids = enc_a["input_ids"]

    # Should not reach here if debate_rounds >= 1
    return ""


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_arc(
    model,
    projector_list,
    proj_cfg,
    tokenizer,
    dataset,
    debate_rounds: int,
    intermediate_tokens: int,
    device: torch.device,
) -> float:
    correct = 0
    total = 0

    for example in tqdm(dataset, desc=f"ARC-C (debate_rounds={debate_rounds})"):
        question = example["question"]
        choices = example["choices"]
        true_answer = example["answerKey"]
        valid_labels = choices["label"]

        question_prompt = build_arc_prompt(tokenizer, question, choices)

        answer_text = run_debate(
            model=model,
            projector_list=projector_list,
            proj_cfg=proj_cfg,
            tokenizer=tokenizer,
            question_prompt=question_prompt,
            question=question,
            choices=choices,
            debate_rounds=debate_rounds,
            intermediate_tokens=intermediate_tokens,
            final_tokens=10,
            device=device,
        )

        pred = extract_arc_answer(answer_text, valid_labels)
        correct += int(pred == true_answer)
        total += 1

    return correct / total * 100 if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--base_model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--benchmark", choices=["arc"], default="arc",
                        help="Only ARC-C supported for now")
    parser.add_argument("--debate_rounds", type=int, default=2,
                        help="Number of debate rounds. 1 = single exchange (no intermediate text).")
    parser.add_argument("--intermediate_tokens", type=int, default=30,
                        help="Max tokens for each intermediate debate response")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Debate rounds: {args.debate_rounds}")
    if args.debate_rounds == 1:
        print("  Note: rounds=1 means single exchange with no intermediate text.")
        print("  With identical models + same input, this is equivalent to iterative round 1.")

    model, tokenizer, projector_list, proj_cfg = load_assets(
        args.base_model, args.checkpoint_dir, device
    )

    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    if args.num_samples:
        ds = ds.select(range(min(args.num_samples, len(ds))))

    acc = evaluate_arc(model, projector_list, proj_cfg, tokenizer, ds,
                       args.debate_rounds, args.intermediate_tokens, device)

    print(f"\n{'='*50}")
    print(f"Cache Debate Results — ARC-C")
    print(f"{'='*50}")
    print(f"Debate rounds : {args.debate_rounds}")
    print(f"Accuracy      : {acc:.2f}%")
    print(f"{'='*50}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"config": vars(args), "accuracy": acc}, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
