# Datasets

This folder contains focused, file-based datasets for both training routes.

## Focused Task
- Short Chinese knowledge Q&A (single-sentence answers)
- Emphasis on concise, factual responses
- SFT splits are controlled classroom data. They are useful for showing
  instruction-following and loss masking, but the normal validation/test sets
  intentionally stay in-domain and share templates with training.
- DPO preference pairs now make the preference explicit: the prompt asks for a
  two-point "definition + usage" answer. Rejected answers are incomplete,
  wrong-concept, vague, or format-violating, so the chosen answer is not only a
  prettier surface form.

## Evaluation
- eval_zh_extended.jsonl
- 120 cases for App8 / Bonus_B style batch evaluation.
  HR, Tech, Product, MCP, OOD, Direct, and Chinese common-knowledge sanity checks.
- A small companion file `eval_dataset.jsonl` (≈10 cases) is provided for App8 fast demos.

## Custom GPT Pretrain Corpus
- pretrain_corpus_zh.txt
- custom_pretrain_corpus.txt
- `pretrain_corpus_zh.txt` is the recommended corpus for Ch7 and Custom GPT pretraining.
- It is included directly in this repository so the notebooks do not need to
  download or generate data before class.
- The corpus is open Chinese expository text selected for next-token
  pretraining demonstrations. It is large enough to show meaningful loss and
  perplexity movement in Ch7 / Custom_02 while still remaining practical for a
  course repository.
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

## Data Meaning Notes
- Pretraining data (`pretrain_corpus_zh.txt`) is the only route using real
  open text by default. The lesson goal is next-token language modeling and
  perplexity reduction, not factual QA ability.
- GPT-2 SFT (`gpt2_sft_*`) is a semantic classification drill over e-commerce
  customer intents. It has real label signal, but only 18 normalized feedback
  templates, so high accuracy mainly means the model learned the controlled
  task.
- Custom GPT SFT (`custom_sft_*`) is a compact concept memorization /
  instruction-format drill. It is intentionally not a real benchmark.
- Custom GPT DPO (`custom_dpo_*`) is now an instruction-preference drill:
  "return definition + usage in two bullet-like points." It teaches DPO
  mechanics and preference direction, not open-ended human preference modeling.
