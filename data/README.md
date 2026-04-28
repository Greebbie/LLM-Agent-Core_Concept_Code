# Datasets

This folder contains focused, file-based datasets for both training routes.

## Focused Task
- Short Chinese knowledge Q&A (single-sentence answers)
- Emphasis on concise, factual responses
- DPO preference pairs favor structured "要点" style (definition + usage) over unstructured text

## Evaluation
- eval_zh_extended.jsonl
- 120 cases for App8 / Bonus_B style batch evaluation:
  HR, Tech, Product, MCP, OOD, Direct, and Chinese common-knowledge sanity checks.
- The tiny 10-case file under `assets/enterprise_5days/` is kept for fast classroom demos.

## Custom GPT Pretrain Corpus
- pretrain_corpus_zh.txt
- custom_pretrain_corpus.txt
- `pretrain_corpus_zh.txt` is the recommended corpus for Ch7 and Custom GPT pretraining.
  Generate it with:

  ```bash
  python data/_download_pretrain_corpus.py
  ```

- The downloader prefers a small streamed subset of Chinese Wikipedia because
  its modern expository style matches the tutorial prompts better than literary
  text. If that source is unavailable, it falls back to public-domain Project
  Gutenberg classics.
- `custom_pretrain_corpus.txt` is kept as a tiny smoke-test fallback only.

## Custom GPT (CPU-friendly, Chinese)
- custom_sft_train.jsonl / custom_sft_val.jsonl / custom_sft_test.jsonl
- custom_dpo_train.jsonl / custom_dpo_val.jsonl / custom_dpo_test.jsonl

Sizes per split:
- train: 8000
- val: 1000
- test: 1000

## GPT-2 Route (GPU, Chinese)
- gpt2_sft_train.jsonl / gpt2_sft_val.jsonl / gpt2_sft_test.jsonl

Sizes per split:
- train: 20000
- val: 2000
- test: 2000

## Archived (not referenced by current notebooks)
- data/_archive/gpt2_dpo_train.jsonl / data/_archive/gpt2_dpo_val.jsonl / data/_archive/gpt2_dpo_test.jsonl

## Format
SFT JSONL:
{ "instruction": "...", "response": "...", "category": "...", "metric": "...", "expected": "..." }

DPO JSONL:
{ "prompt": "...", "chosen": "...", "rejected": "...", "category": "..." }
