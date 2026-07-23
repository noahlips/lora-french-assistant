"""LoRA fine-tuning of a small instruct model on French instructions.

Trains low-rank adapters (PEFT) on top of a frozen Qwen2.5-0.5B-Instruct.
Runs on CPU (slow but functional) or any CUDA GPU. Only the adapters are
saved (~a few MB), never the base weights.

Usage:
    python -m src.train_lora                     # full config
    python -m src.train_lora --max-steps 20      # smoke test
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = BASE_DIR / "outputs" / "lora-adapter"
MAX_LENGTH = 512


def load_tokenized_datasets(tokenizer):
    files = {
        "train": str(BASE_DIR / "data" / "train.jsonl"),
        "eval": str(BASE_DIR / "data" / "eval.jsonl"),
    }
    raw = load_dataset("json", data_files=files)

    def tokenize(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return tokenizer(text, truncation=True, max_length=MAX_LENGTH)

    return raw.map(tokenize, remove_columns=["messages"])


def build_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-steps", type=int, default=-1, help="override for smoke tests")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    datasets = load_tokenized_datasets(tokenizer)
    model = build_model()

    training_args = TrainingArguments(
        output_dir=str(BASE_DIR / "runs"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✓ LoRA adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
