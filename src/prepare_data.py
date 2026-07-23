"""Prepare the French instruction dataset for LoRA fine-tuning.

Downloads a French instruction dataset from the Hugging Face Hub, formats
each example with the model's chat template, and writes train/eval JSONL
splits to data/.

Usage:
    python -m src.prepare_data [--n-samples 3000] [--eval-ratio 0.05]
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_ID = "jpacifico/French-Alpaca-dataset-Instruct-55K"


def format_example(example: dict) -> dict:
    instruction = example["instruction"].strip()
    context = (example.get("input") or "").strip()
    user_message = f"{instruction}\n\n{context}" if context else instruction
    return {
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": example["output"].strip()},
        ]
    }


def prepare(n_samples: int, eval_ratio: float, seed: int = 42) -> tuple[Path, Path]:
    dataset = load_dataset(DATASET_ID, split="train")
    dataset = dataset.shuffle(seed=seed).select(range(n_samples))
    formatted = [format_example(ex) for ex in dataset]

    n_eval = max(1, int(len(formatted) * eval_ratio))
    eval_set, train_set = formatted[:n_eval], formatted[n_eval:]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "train.jsonl"
    eval_path = DATA_DIR / "eval.jsonl"
    for path, rows in [(train_path, train_set), (eval_path, eval_set)]:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✓ {len(train_set)} train / {len(eval_set)} eval examples written to {DATA_DIR}")
    return train_path, eval_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    args = parser.parse_args()
    prepare(args.n_samples, args.eval_ratio)
