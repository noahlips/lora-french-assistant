# LoRA French Assistant

Parameter-efficient fine-tuning (LoRA) of **Qwen2.5-0.5B-Instruct** on a French instruction dataset, with a before/after evaluation comparing the base model and the tuned adapters.

The goal of this repo is to demonstrate the *complete* PEFT workflow (dataset preparation with chat templates, adapter training, and measurable evaluation) rather than to chase benchmark scores with a small model.

## Results

Perplexity on 50 held-out examples, after 1 epoch (179 optimizer steps, ~5 h on CPU):

| Model | Perplexity |
|-------|------------|
| Qwen2.5-0.5B-Instruct (base) | 19.75 |
| + LoRA adapter | **3.10** |

Training loss fell from 2.14 to ~1.21 (mean 1.30). Only **2,162,688 parameters** were trained out of 496,195,456 (**0.44 %**), producing an 8.6 MB adapter.

### Reading these numbers honestly

A 6.4x perplexity drop looks dramatic, but it must be interpreted carefully: the evaluation split comes from the *same* dataset as the training data, so most of the gain reflects adaptation to French-Alpaca's format and register rather than a genuine capability gain. Perplexity measures how predictable the reference answer is, not whether the answer is useful.

The qualitative outputs confirm this. Asked to contrast supervised and unsupervised learning, the tuned model produces fluent, well-formed French that is **factually wrong** (it describes both paradigms identically). Asked to draft an email, it returns a placeholder instead of the email. Only the most generic prompt (advantages of remote work) yields a usable answer.

The honest conclusion: the adapter successfully taught a 0.5B model the *shape* of a French instruction response, and could not teach it knowledge or reasoning it never had. That is the expected outcome at this scale, and it is why perplexity alone is an insufficient evaluation.

## Why LoRA?

Full fine-tuning of even a 0.5B model updates ~500M parameters. LoRA freezes the base model and trains low-rank decomposition matrices injected into the attention projections (`q/k/v/o_proj`): here ~2M trainable parameters, producing an adapter of a few MB that can be shared and stacked independently of the base weights.

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
python -m src.train_lora --epochs 1           # train adapters (GPU strongly recommended)
python -m src.train_lora --max-steps 20       # CPU smoke test
python -m src.evaluate --n-eval 50            # perplexity before/after + samples
```

## Configuration

| Hyperparameter | Value |
|----------------|-------|
| Base model | Qwen2.5-0.5B-Instruct |
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| LR / schedule | 2e-4, cosine, 3% warmup |
| Effective batch size | 16 (4 × 4 grad. accumulation) |
| Dataset | French-Alpaca, 2850 train / 150 eval |

## What would improve this

- Evaluate on a French benchmark the model was not tuned on, so the score measures capability rather than format adaptation
- Compare several LoRA ranks to find where the accuracy/size trade-off sits
- Start from a 7B base model, where the adapter has real capacity to work with
