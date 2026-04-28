"""Build a small modern Chinese corpus for pretraining demos.

The original demo corpus is intentionally tiny and useful only for smoke tests.
This script builds data/pretrain_corpus_zh.txt from real Chinese text while
keeping the file small enough for classroom notebooks.

Default source:
    Hugging Face datasets: wikimedia/wikipedia, config 20231101.zh

Fallback source:
    Public-domain Chinese classics from Project Gutenberg. The fallback keeps
    the notebook runnable when the Hugging Face dataset cannot be reached, but
    Wikipedia is preferred because its modern expository style better matches
    the tutorial prompts.
"""
from __future__ import annotations

import os
import re
import sys
import urllib.request
from collections.abc import Iterable
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "pretrain_corpus_zh.txt"
TARGET_BYTES = int(os.environ.get("PRETRAIN_CORPUS_TARGET_BYTES", "6000000"))
SOURCE = os.environ.get("PRETRAIN_CORPUS_SOURCE", "wiki").lower()
WIKI_CONFIG = os.environ.get("PRETRAIN_WIKI_CONFIG", "20231101.zh")

GUTENBERG_SOURCES = [
    ("西游记", "https://www.gutenberg.org/ebooks/23962.txt.utf-8"),
    ("红楼梦", "https://www.gutenberg.org/ebooks/24264.txt.utf-8"),
    ("三国演义", "https://www.gutenberg.org/ebooks/23950.txt.utf-8"),
]


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk / len(text)


def normalize_line(text: str) -> str | None:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("[]()（）")
    if len(text) < 24:
        return None
    if cjk_ratio(text) < 0.35:
        return None
    if text.startswith(("参见", "外部链接", "参考文献", "分类:")):
        return None
    return text


def iter_wikipedia_lines() -> Iterable[str]:
    from datasets import load_dataset

    print(f"Streaming wikimedia/wikipedia:{WIKI_CONFIG}")
    dataset = load_dataset(
        "wikimedia/wikipedia",
        WIKI_CONFIG,
        split="train",
        streaming=True,
    )

    for item in dataset:
        title = normalize_line(item.get("title", ""))
        if title:
            yield f"标题：{title}"

        text = item.get("text", "")
        for raw in re.split(r"\n+", text):
            line = normalize_line(raw)
            if line:
                yield line


def fetch_text(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "LLM-Agent-Core-Course/1.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="ignore")


def strip_gutenberg_boilerplate(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text)
    end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*", text)
    if start:
        text = text[start.end() :]
    if end:
        text = text[: end.start()]
    return text


def iter_gutenberg_lines() -> Iterable[str]:
    for title, url in GUTENBERG_SOURCES:
        print(f"Downloading fallback source {title}: {url}")
        text = strip_gutenberg_boilerplate(fetch_text(url))
        for raw in text.splitlines():
            line = normalize_line(raw)
            if line:
                yield line


def write_corpus(lines: Iterable[str]) -> tuple[int, int]:
    collected: list[str] = []
    total_bytes = 0

    for line in lines:
        encoded_len = len((line + "\n").encode("utf-8"))
        collected.append(line)
        total_bytes += encoded_len
        if total_bytes >= TARGET_BYTES:
            break

    if not collected:
        raise RuntimeError("No corpus lines collected; check data source/network access.")

    OUT_PATH.write_text("\n".join(collected) + "\n", encoding="utf-8")
    return len(collected), OUT_PATH.stat().st_size


def main() -> None:
    if SOURCE not in {"wiki", "wikipedia", "gutenberg"}:
        raise ValueError("PRETRAIN_CORPUS_SOURCE must be 'wiki' or 'gutenberg'.")

    try:
        if SOURCE in {"wiki", "wikipedia"}:
            line_count, size = write_corpus(iter_wikipedia_lines())
        else:
            line_count, size = write_corpus(iter_gutenberg_lines())
    except Exception as exc:
        if SOURCE == "gutenberg":
            raise
        print(f"Wikipedia source failed: {exc}", file=sys.stderr)
        print("Falling back to Project Gutenberg public-domain classics.", file=sys.stderr)
        line_count, size = write_corpus(iter_gutenberg_lines())

    print(f"Wrote {OUT_PATH} ({line_count:,} lines, {size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
