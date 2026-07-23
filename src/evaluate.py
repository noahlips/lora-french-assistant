"""Compare the base model and the LoRA-tuned model.

Measures perplexity on the held-out eval split for both models, and prints
side-by-side generations on a few French prompts.

Usage:
    python -m src.evaluate [--n-eval 50]
"""

import argparse
import json
import math
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.train_lora import MAX_LENGTH, MODEL_ID, OUTPUT_DIR

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_PROMPTS = [
    "Explique la différence entre apprentissage supervisé et non supervisé.",
    "Rédige un court e-mail pour reporter une réunion à demain.",
    "Quels sont les avantages du télétravail ?",
]


@torch.no_grad()
def perplexity(model, tokenizer, texts: list[str]) -> float:
    losses = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        loss = model(**inputs, labels=inputs["input_ids"]).loss
        losses.append(loss.item())
    return math.exp(sum(losses) / len(losses))


@torch.no_grad()
def generate(model, tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-eval", type=int, default=50)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    eval_rows = [
        json.loads(line)
        for line in (BASE_DIR / "data" / "eval.jsonl").read_text(encoding="utf-8").splitlines()
    ][: args.n_eval]
    eval_texts = [
        tokenizer.apply_chat_template(row["messages"], tokenize=False) for row in eval_rows
    ]

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    base_ppl = perplexity(base_model, tokenizer, eval_texts)
    print(f"Base model perplexity:  {base_ppl:.2f}")

    tuned_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    tuned_ppl = perplexity(tuned_model, tokenizer, eval_texts)
    print(f"LoRA model perplexity:  {tuned_ppl:.2f}")

    for prompt in SAMPLE_PROMPTS:
        print(f"\n=== {prompt}")
        print(f"[LoRA] {generate(tuned_model, tokenizer, prompt)}")


if __name__ == "__main__":
    main()
