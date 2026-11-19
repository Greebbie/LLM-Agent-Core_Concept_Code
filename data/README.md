# Datasets

## Training tasks
- Custom GPT: turn a library return note into a handover record. For a damaged book, both “缺页：修书台，补页” and “缺页：补页，修书台” have the correct contents. DPO pairs keep the same characters and prefer the action before the desk.
- GPT-2 SFT and LoRA: classify Chinese customer tickets into six queues. Ch8 generates a category and reason; Ch9 generates the category alone.
- Ch10 constructs its own exhibit notices inside the notebook. For each question, one rejected answer names the wrong location and another adds an unnecessary courtesy phrase to the correct location.

## Evaluation
- `eval_zh_extended.jsonl` contains 120 App8 cases covering HR, Tech, Product, MCP, OOD, Direct, and Chinese common knowledge.
- A small companion file `eval_dataset.jsonl` (13 cases) is provided for App8 fast demos. In cases with `required_facts`, each group must match at least one listed alternative in the answer itself.

## Custom GPT Pretrain Corpus
Ch7 and Custom GPT 02 both read `pretrain_corpus_zh.txt`, a collection of Chinese expository text included in this repository. `custom_pretrain_corpus.txt` is a smaller example.

## Custom GPT (CPU-friendly, Chinese)
- custom_sft_train.jsonl / custom_sft_val.jsonl / custom_sft_test.jsonl
- custom_dpo_train.jsonl / custom_dpo_val.jsonl / custom_dpo_test.jsonl

Sizes per split:
- train: 384
- val: 96
- test: 96

## GPT-2 Route (GPU, Chinese)
- gpt2_sft_train.jsonl / gpt2_sft_val.jsonl / gpt2_sft_test.jsonl

Sizes per split:
- train: 216
- val: 12
- test: 18

## Archived (not referenced by current notebooks)
- data/_archive/gpt2_dpo_train.jsonl / data/_archive/gpt2_dpo_val.jsonl / data/_archive/gpt2_dpo_test.jsonl

## Format
SFT JSONL:
`{ "instruction": "...", "response": "...", "category": "...", "metric": "...", "expected": "..." }`

DPO JSONL:
`{ "prompt": "...", "chosen": "...", "rejected": "...", "category": "..." }`

## Train and test splits
- Pretraining data (`pretrain_corpus_zh.txt`) is the only route using real
  open text by default. The lesson goal is next-token language modeling and
  perplexity reduction, not factual QA ability.
- GPT-2 SFT (`gpt2_sft_*`) is a semantic classification drill over e-commerce
  customer intents. Each training class combines six clues with six openings. Validation and test use separately written messages. The evidence field quotes the relevant words in each ticket. Compare these with the generated reason; a correct category alone does not establish that the reason is grounded.
- Custom GPT SFT and DPO share the same prompts within each split. The SFT response gives one accepted order; three quarters of training responses put the desk first. The DPO chosen answer and the SFT `expected` field put the action first.
- The 96 test records combine three held-out descriptions (one per condition), eight book titles and four return locations. Each record names its condition explicitly. The separate three-pair paraphrase exercise in Custom_05 replaces that word while keeping the underlying condition unchanged.
