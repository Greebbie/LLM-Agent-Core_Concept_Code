"""Helper script invoked by code-review skill.

Runs ruff + mypy and returns structured findings.
"""
from __future__ import annotations
import subprocess
from pathlib import Path


def run_checks(target: str | Path) -> dict:
    """Run ruff + mypy on `target` (file or directory). Returns:
        {"ruff": {"ok": bool, "issues": [...]},
         "mypy": {"ok": bool, "issues": [...]}}
    """
    target = str(target)
    out = {}
    # ruff
    try:
        result = subprocess.run(
            ["ruff", "check", target, "--output-format=concise"],
            capture_output=True, text=True, timeout=30,
        )
        out["ruff"] = {
            "ok": result.returncode == 0,
            "issues": result.stdout.splitlines() if result.stdout else [],
        }
    except FileNotFoundError:
        out["ruff"] = {"ok": False, "issues": ["ruff not installed (pip install ruff)"]}
    except Exception as e:
        out["ruff"] = {"ok": False, "issues": [f"error: {e}"]}
    # mypy
    try:
        result = subprocess.run(
            ["mypy", "--no-error-summary", target],
            capture_output=True, text=True, timeout=60,
        )
        out["mypy"] = {
            "ok": result.returncode == 0,
            "issues": result.stdout.splitlines() if result.stdout else [],
        }
    except FileNotFoundError:
        out["mypy"] = {"ok": False, "issues": ["mypy not installed (pip install mypy)"]}
    except Exception as e:
        out["mypy"] = {"ok": False, "issues": [f"error: {e}"]}
    return out


if __name__ == "__main__":
    import sys, json
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    print(json.dumps(run_checks(target), indent=2))
