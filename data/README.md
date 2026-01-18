# Datasets

This folder contains focused, file-based datasets for both training routes.

## Focused Task
- Short Chinese knowledge Q&A (single-sentence answers)
- Emphasis on concise, factual responses
- DPO preference pairs favor structured "要点" style (definition + usage) over unstructured text

## Custom GPT Pretrain Corpus
- custom_pretrain_corpus.txt
- General Chinese text for pretraining (kept distinct from the SFT/DPO task)

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
