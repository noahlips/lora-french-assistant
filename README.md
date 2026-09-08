# LoRA fine-tuning on French instructions

I wanted to try fine-tuning a language model myself instead of just calling one through an API. This repo is the result: a LoRA adapter trained on top of Qwen2.5-0.5B-Instruct, using a French instruction dataset, and evaluated against the base model.

Everything ran on CPU, since I don't have a GPU. One epoch took about 5 hours.

## Setup

| | |
|---|---|
| Base model | Qwen2.5-0.5B-Instruct |
| Dataset | French-Alpaca, 2850 train / 150 eval |
| LoRA | r=16, alpha=32, dropout 0.05 |
| Adapted layers | q_proj, k_proj, v_proj, o_proj |
| Trained parameters | 2.16M out of 496M (0.44%) |
| Adapter size | 8.6 MB |

Training used Adam at 2e-4 with a cosine schedule, batch size 4 with 4 gradient accumulation steps.

## What came out of it

Perplexity on the held-out split, before and after:

| | Perplexity |
|---|---|
| Base model | 19.75 |
| With LoRA adapter | 3.10 |

Training loss went from 2.14 down to about 1.21 over 179 steps.

## Why that number is misleading

A 6x drop in perplexity looks great until you notice the eval split comes from the same dataset as the training data. So the model mostly learned what a French-Alpaca answer looks like, not anything new about the world.

The generated samples make this obvious. Asked to explain the difference between supervised and unsupervised learning, the tuned model writes clean French and gets it wrong: it describes both the same way. Asked to write a short email, it returns a placeholder instead of an actual email. The only decent answer was to the vaguest question.

So the honest conclusion is that LoRA taught a 0.5B model the shape of a French instruction answer, and could not give it knowledge it never had. That is what should happen at this size, and it is a good reminder that perplexity on its own does not tell you whether a model is useful.

## Running it

```bash
pip install -r requirements.txt

python -m src.prepare_data --n-samples 3000
python -m src.train_lora --epochs 1        # ~5h on CPU
python -m src.evaluate --n-eval 50
```

`python -m src.train_lora --max-steps 20` runs a quick smoke test if you just want to check the pipeline works.

## If I redo this

The obvious next step is evaluating on a French benchmark the model was never trained on, so the score measures something real. After that, comparing a few LoRA ranks, and starting from a 7B model where the adapter actually has room to work.
