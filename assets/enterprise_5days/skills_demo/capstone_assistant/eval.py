"""Evaluation helpers for enterprise-knowledge-assistant Skill."""
from __future__ import annotations
import json
from pathlib import Path
from pipeline import upgraded_pipeline  # type: ignore


def batch_eval(dataset_path: str | Path) -> dict:
    """Run pipeline on all eval cases. Return success_rate + per_category."""
    items = [json.loads(line) for line in open(dataset_path, encoding="utf-8")]
    results = []
    cat_correct, cat_total = {}, {}
    for it in items:
        out = upgraded_pipeline(it["query"])
        answer = out.get("answer", "")
        ok = any(kw.lower() in answer.lower() for kw in it["expected_keywords"])
        cat = it["category"]
        cat_total[cat] = cat_total.get(cat, 0) + 1
        if ok:
            cat_correct[cat] = cat_correct.get(cat, 0) + 1
        results.append({"query": it["query"], "ok": ok, "cat": cat, "answer": answer[:80]})
    success = sum(1 for r in results if r["ok"])
    return {
        "success_rate": success / len(items) if items else 0.0,
        "per_category": {c: cat_correct.get(c, 0) / cat_total[c] for c in cat_total},
        "details": results,
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "reference/eval_cases.jsonl"
    result = batch_eval(path)
    print(f"Success rate: {result['success_rate']:.0%}")
    for c, acc in result["per_category"].items():
        print(f"  {c}: {acc:.0%}")
