# LoRA French Assistant

Parameter-efficient fine-tuning (LoRA) of **Qwen2.5-0.5B-Instruct** on a French instruction dataset, with a before/after evaluation comparing the base model and the tuned adapters.

The goal of this repo is to demonstrate the *complete* PEFT workflow — dataset preparation with chat templates, adapter training, and measurable evaluation — not to chase benchmark scores with a small model.

## Why LoRA?

Full fine-tuning of even a 0.5B model updates ~500M parameters. LoRA freezes the base model and trains low-rank decomposition matrices injected into the attention projections (`q/k/v/o_proj`) — here **~2M trainable parameters (<1%)**, producing an adapter of a few MB that can be shared and stacked independently of the base weights.

## Pipeline

```
prepare_data.py                train_lora.py                    evaluate.py
HF dataset (French-Alpaca) →   PEFT LoRA r=16, α=32        →    perplexity base vs tuned
chat-template JSONL splits     bf16 on GPU / fp32 on CPU        + side-by-side generations
```

## Usage

```bash
pip install -r requirements.txt

python -m src.prepare_data --n-samples 3000   # download + format dataset
python -m src.train_lora                      # train adapters (GPU recommended)
python -m src.train_lora --max-steps 20       # CPU smoke test
python -m src.evaluate                        # perplexity before/after + samples
```

## Configuration

| Hyperparameter | Value |
|----------------|-------|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| LR / schedule | 2e-4, cosine, 3% warmup |
| Effective batch size | 16 (4 × 4 grad. accumulation) |

## Results

_To be published after the full training run (perplexity table + qualitative comparison)._
