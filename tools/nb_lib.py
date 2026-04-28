"""Notebook helper library for enterprise_ver2 transformations.

Pure stdlib. Operates on .ipynb (JSON) files.
"""
from __future__ import annotations
import json
import re
import uuid
from pathlib import Path
from typing import Iterable


def load_nb(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_nb(nb: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")


def cell_source(cell: dict) -> str:
    """Return the source as a single string regardless of list/str storage."""
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def set_cell_source(cell: dict, text: str) -> None:
    """Store source as a list of lines (jupyter convention preserves diffs)."""
    lines = text.splitlines(keepends=True)
    cell["source"] = lines if lines else [""]


def make_md(text: str, tags: Iterable[str] = ()) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:12],
        "metadata": {"tags": list(tags)} if tags else {},
        "source": text.splitlines(keepends=True) or [""],
    }


def make_code(text: str, tags: Iterable[str] = ()) -> dict:
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:12],
        "execution_count": None,
        "metadata": {"tags": list(tags)} if tags else {},
        "outputs": [],
        "source": text.splitlines(keepends=True) or [""],
    }


def add_tag(cell: dict, tag: str) -> None:
    meta = cell.setdefault("metadata", {})
    tags = meta.setdefault("tags", [])
    if tag not in tags:
        tags.append(tag)


def find_cells(nb: dict, predicate) -> list[tuple[int, dict]]:
    return [(i, c) for i, c in enumerate(nb["cells"]) if predicate(c)]


def find_by_first_line(nb: dict, regex: str, cell_type: str = "code") -> list[tuple[int, dict]]:
    rx = re.compile(regex)
    out = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != cell_type:
            continue
        first = cell_source(c).splitlines()[0] if cell_source(c) else ""
        if rx.search(first):
            out.append((i, c))
    return out


def insert_after(nb: dict, idx: int, *cells: dict) -> int:
    """Insert cells after position idx. Returns new position of last inserted cell."""
    nb["cells"][idx + 1:idx + 1] = list(cells)
    return idx + len(cells)


def insert_before(nb: dict, idx: int, *cells: dict) -> int:
    nb["cells"][idx:idx] = list(cells)
    return idx + len(cells) - 1


def delete_cells(nb: dict, indices: Iterable[int]) -> None:
    """Delete by ABSOLUTE original indices. Pass them in any order."""
    keep = [c for i, c in enumerate(nb["cells"]) if i not in set(indices)]
    nb["cells"] = keep


def replace_in_source(cell: dict, old: str, new: str) -> bool:
    """Replace literal text in cell source. Returns True if changed."""
    src = cell_source(cell)
    if old not in src:
        return False
    set_cell_source(cell, src.replace(old, new))
    return True


def replace_regex_in_source(cell: dict, pattern: str, repl: str) -> bool:
    src = cell_source(cell)
    new, n = re.subn(pattern, repl, src)
    if n == 0:
        return False
    set_cell_source(cell, new)
    return True


# ============================================================
# Path-fix cell — added near top of every notebook
# ============================================================
PATH_FIX_CODE = '''# ── 课程环境就位（自动定位课程根目录，让 utils/data/fonts 路径无论从 instructor/ 还是 student/ 都能用）──
import os, sys
_cur = os.path.abspath("")
_root = None
for _candidate in [_cur] + [os.path.dirname(_cur), os.path.dirname(os.path.dirname(_cur))]:
    if all(os.path.isdir(os.path.join(_candidate, d)) for d in ("utils", "data")):
        _root = _candidate
        break
if _root is None:
    raise RuntimeError("找不到课程根目录（应包含 utils/ 与 data/）。请确认 notebook 位于 enterprise_ver2/instructor 或 enterprise_ver2/student 下。")
os.chdir(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)
if os.path.join(_root, "utils") not in sys.path:
    sys.path.insert(0, os.path.join(_root, "utils"))
print(f"📂 课程根目录：{_root}")
'''


def make_path_fix_cell() -> dict:
    return make_code(PATH_FIX_CODE, tags=["setup", "instructor_and_student"])


# ============================================================
# 「📋 讲课提示」envelope generator
# ============================================================
def make_lecture_note(
    title: str,
    duration_min: int,
    opener: str,
    key_points: list[str],
    misconceptions: list[str] = (),
    interaction: str = "",
    if_short_on_time: str = "",
) -> dict:
    parts = [
        "> 📋 **讲课提示** *(此区块仅讲师版含，学员版自动剥离)*",
        f"> ",
        f"> **本节：** {title}　|　**时长：** {duration_min} min",
        f"> ",
        f"> **开场：** {opener}",
        f"> ",
        "> **重点强调：**",
    ]
    for kp in key_points:
        parts.append(f"> - {kp}")
    if misconceptions:
        parts.append(">")
        parts.append("> **常见误解：**")
        for m in misconceptions:
            parts.append(f"> - {m}")
    if interaction:
        parts.append(">")
        parts.append(f"> **互动设计：** {interaction}")
    if if_short_on_time:
        parts.append(">")
        parts.append(f"> **时间紧时：** {if_short_on_time}")
    return make_md("\n".join(parts), tags=["instructor_only", "lecture_note"])


# ============================================================
# Mark exercise cells with `fillin` tag based on common patterns
# ============================================================
EXERCISE_MARKERS = re.compile(
    r"^\s*#\s*"
    r"(?:======+\s*)?"
    r"(?:练习\s*\d+|Exercise\s*\d+|Ex\s*\d+|Checkpoint\s*\d+|Mini-?Project|"
    r"练习\s*[一二三四五六七八九十]+|实操|动手|你来实现)"
)


def autotag_exercises(nb: dict) -> int:
    """Tag exercise code cells with `fillin`. Returns count tagged."""
    count = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        first = cell_source(c).splitlines()[0] if cell_source(c) else ""
        if EXERCISE_MARKERS.match(first):
            add_tag(c, "fillin")
            count += 1
    return count


# ============================================================
# Cost / monetary phrase scanner — detect cells that should be reviewed
# ============================================================
COST_PHRASES = re.compile(
    r"(?:节省¥|月省|年节省|节约 ?¥|"
    r"成本计算器|成本估算|"
    r"API 调用 vs 自部署|自部署 vs API|"
    r"商业决策分析|"
    r"年节约|月节约)"
)


def scan_cost_cells(nb: dict) -> list[tuple[int, str, str]]:
    out = []
    for i, c in enumerate(nb["cells"]):
        src = cell_source(c)
        if COST_PHRASES.search(src):
            first = src.splitlines()[0][:80] if src else ""
            out.append((i, c["cell_type"], first))
    return out


if __name__ == "__main__":
    print("nb_lib loaded OK")
