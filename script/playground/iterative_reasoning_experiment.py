"""
Iterative Cache Reasoning Experiment (Idea 1 — Identical Model Pair)

Tests whether applying the C2C projector iteratively (K rounds) progressively improves accuracy.
Each round uses the current fused cache as both source and target for further self-refinement.

Usage:
    # Single checkpoint, sweep rounds 1-4 on GSM8K
    python script/playground/iterative_reasoning_experiment.py \
        --checkpoint_dir local/checkpoints/0.6B_identical_C2C/final \
        --base_model Qwen/Qwen3-0.6B \
        --teacher_model Qwen/Qwen3-0.6B \
        --benchmark gsm8k \
        --max_rounds 4 \
        --num_samples 200

    # ARC-C (faster, good for quick sanity check)
    python script/playground/iterative_reasoning_experiment.py \
        --checkpoint_dir local/checkpoints/0.6B_identical_C2C/final \
        --base_model Qwen/Qwen3-0.6B \
        --teacher_model Qwen/Qwen3-0.6B \
        --benchmark arc \
        --max_rounds 4
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from rosetta.model.projector import load_projector
from rosetta.model.wrapper import RosettaModel
from rosetta.utils.evaluate import set_default_chat_template
from rosetta.utils.matheval import GSM8KEvaluator


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(base_model_path: str, teacher_model_path: str, checkpoint_dir: str,
               device: torch.device) -> Tuple[RosettaModel, AutoTokenizer]:
    """Load RosettaModel with trained identical-pair (or any) projectors."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    set_default_chat_template(tokenizer, base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map={"": device}
    ).eval()

    if teacher_model_path == base_model_path:
        # Identical pair: share the same model object to save memory
        teacher_model = base_model
    else:
        teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_model_path, torch_dtype=torch.bfloat16, device_map={"": device}
        ).eval()

    # Load projectors
    num_projectors = len([f for f in os.listdir(checkpoint_dir) if re.match(r"projector_\d+\.pt", f)])
    if num_projectors == 0:
        raise FileNotFoundError(f"No projector_*.pt files found in {checkpoint_dir}")

    projector_list = []
    for t in range(num_projectors):
        json_cfg = os.path.join(checkpoint_dir, f"projector_{t}.json")
        proj = load_projector(json_cfg).to(device)
        pt_path = os.path.join(checkpoint_dir, f"projector_{t}.pt")
        if os.path.exists(pt_path):
            proj.load_state_dict(torch.load(pt_path, map_location=device), strict=False)
        proj.eval()
        projector_list.append(proj)

    rosetta_model = RosettaModel(
        model_list=[base_model, teacher_model],
        base_model_idx=0,
        projector_list=projector_list,
    ).to(device).eval()

    proj_cfg_path = os.path.join(checkpoint_dir, "projector_config.json")
    rosetta_model.load_projector_config(proj_cfg_path)

    return rosetta_model, tokenizer


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def load_gsm8k(num_samples: Optional[int] = None):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if num_samples:
        ds = ds.select(range(min(num_samples, len(ds))))
    return ds


def load_arc(num_samples: Optional[int] = None):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    if num_samples:
        ds = ds.select(range(min(num_samples, len(ds))))
    return ds


def build_gsm8k_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def build_arc_prompt(tokenizer, question: str, choices: Dict) -> str:
    opts = "\n".join(f"{label}. {text}" for label, text in zip(choices["label"], choices["text"]))
    content = f"Answer the following multiple-choice question with a single letter.\n\n{question}\n{opts}\n\nAnswer:"
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def extract_arc_answer(text: str, valid_labels: List[str]) -> str:
    for char in reversed(text):
        if char in valid_labels:
            return char
    return "X"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_generate(rosetta_model: RosettaModel, tokenizer: AutoTokenizer, input_ids: torch.Tensor,
                 attention_mask: torch.Tensor, max_new_tokens: int, iterative_rounds: int,
                 device: torch.device) -> str:
    """Run generation for a single example with a given number of iterative rounds."""
    seq_len = input_ids.shape[1]
    # Section 0: all tokens use C2C fusion (sharer_mask=1)
    instruction_idx = torch.tensor([1, 0], dtype=torch.long).repeat(seq_len - 1, 1).unsqueeze(0).to(device)
    # Section 1: final token / generation step (no projection)
    label_idx = torch.tensor([-1, 0], dtype=torch.long).repeat(1, 1).unsqueeze(0).to(device)
    kv_cache_index = [instruction_idx, label_idx]

    output_ids = rosetta_model.generate(
        kv_cache_index=kv_cache_index,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        iterative_rounds=iterative_rounds,
    )
    # Decode only the generated part
    generated = output_ids[0, seq_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


@torch.no_grad()
def evaluate_gsm8k(rosetta_model: RosettaModel, tokenizer: AutoTokenizer, dataset,
                   max_rounds: int, device: torch.device) -> Dict[int, float]:
    evaluator = GSM8KEvaluator()
    results = {r: {"correct": 0, "total": 0} for r in range(1, max_rounds + 1)}

    for example in tqdm(dataset, desc="GSM8K"):
        prompt = build_gsm8k_prompt(tokenizer, example["question"])
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        for rounds in range(1, max_rounds + 1):
            output_text = run_generate(
                rosetta_model, tokenizer, input_ids, attention_mask,
                max_new_tokens=512, iterative_rounds=rounds, device=device
            )
            is_correct, _ = evaluator.rule_judge(output_text, example["answer"])
            results[rounds]["correct"] += int(is_correct)
            results[rounds]["total"] += 1

    return {r: v["correct"] / v["total"] * 100 for r, v in results.items()}


@torch.no_grad()
def evaluate_arc(rosetta_model: RosettaModel, tokenizer: AutoTokenizer, dataset,
                 max_rounds: int, device: torch.device) -> Dict[int, float]:
    results = {r: {"correct": 0, "total": 0} for r in range(1, max_rounds + 1)}

    for example in tqdm(dataset, desc="ARC-C"):
        prompt = build_arc_prompt(tokenizer, example["question"], example["choices"])
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        valid_labels = example["choices"]["label"]
        true_answer = example["answerKey"]

        for rounds in range(1, max_rounds + 1):
            output_text = run_generate(
                rosetta_model, tokenizer, input_ids, attention_mask,
                max_new_tokens=10, iterative_rounds=rounds, device=device
            )
            pred = extract_arc_answer(output_text.strip(), valid_labels)
            results[rounds]["correct"] += int(pred == true_answer)
            results[rounds]["total"] += 1

    return {r: v["correct"] / v["total"] * 100 for r, v in results.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True, help="Path to trained projector checkpoint dir")
    parser.add_argument("--base_model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--teacher_model", default="Qwen/Qwen3-0.6B",
                        help="Set equal to base_model for identical-pair experiment")
    parser.add_argument("--benchmark", choices=["gsm8k", "arc"], default="arc")
    parser.add_argument("--max_rounds", type=int, default=4,
                        help="Maximum number of iterative projection rounds to test")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit dataset size for quick runs")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to this JSON file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: base={args.base_model}, teacher={args.teacher_model}")
    print(f"Identical pair: {args.base_model == args.teacher_model}")
    print(f"Benchmark: {args.benchmark}, max_rounds: {args.max_rounds}")
    print()

    # Load model
    rosetta_model, tokenizer = load_model(
        args.base_model, args.teacher_model, args.checkpoint_dir, device
    )

    # Load dataset
    if args.benchmark == "gsm8k":
        dataset = load_gsm8k(args.num_samples)
        acc_by_rounds = evaluate_gsm8k(rosetta_model, tokenizer, dataset, args.max_rounds, device)
    else:
        dataset = load_arc(args.num_samples)
        acc_by_rounds = evaluate_arc(rosetta_model, tokenizer, dataset, args.max_rounds, device)

    # Print results table
    print("\n" + "="*50)
    print(f"Iterative Cache Reasoning Results — {args.benchmark.upper()}")
    print("="*50)
    print(f"{'Rounds':<10} {'Accuracy (%)':<15}")
    print("-"*25)
    for rounds, acc in sorted(acc_by_rounds.items()):
        marker = " <-- baseline" if rounds == 1 else ""
        print(f"{rounds:<10} {acc:.2f}{marker}")
    print("="*50)

    # Delta vs round 1
    base_acc = acc_by_rounds[1]
    print(f"\nDelta vs Round 1 (standard C2C):")
    for rounds in range(2, args.max_rounds + 1):
        delta = acc_by_rounds[rounds] - base_acc
        print(f"  Round {rounds}: {delta:+.2f}%")

    # Save results
    if args.output:
        out = {
            "config": vars(args),
            "accuracy_by_rounds": acc_by_rounds,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
