"""图书馆交接实验共用的输入格式、贪心解码与验收规则。"""
import json
from pathlib import Path

import torch


def records(root, stage, split):
    with (Path(root) / "data" / f"custom_{stage}_{split}.jsonl").open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prompt_text(note):
    return f"<|user|>{note}<|assistant|>"


def reply(model, tokenizer, note):
    ids = tokenizer.encode(prompt_text(note), add_eos=False)
    device = next(model.parameters()).device
    prompt = torch.tensor([ids], device=device)
    output = model.generate(prompt, max_new_tokens=16, do_sample=False, top_k=None, top_p=None)
    return tokenizer.decode(output[0, len(ids):].tolist())


def evaluate_replies(model, tokenizer, cases):
    content_hits = action_first = 0
    examples = []
    for case in cases:
        answer = reply(model, tokenizer, case["instruction"])
        preferred = case["expected"]
        status, fields = preferred.split("：")
        alternate = status + "：" + "，".join(reversed(fields.split("，")))
        content_hits += answer in (preferred, alternate)
        action_first += answer == preferred
        examples.append(answer)
    return content_hits, action_first, examples
