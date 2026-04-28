"""Derive student/ notebooks from instructor/ notebooks.

Rules:
- Cell with tag `instructor_only` → DELETE entirely (the 📋 讲课提示 cells)
- Cell with tag `fillin` → blank out implementation + clear outputs:
    * Keep the leading comment block (`# 练习 N：...` or `# Checkpoint N:`)
    * If `# ↓↓↓ 你的代码` / `# ↑↑↑ 你的代码` markers present, blank only between them
    * Otherwise insert a `# TODO: 完成此练习。讲师版有完整答案。\npass` stub
- All other cells → keep as-is (including outputs)

Run from repo root:
    python tools/derive_student.py
"""
from __future__ import annotations
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_lib import load_nb, save_nb, cell_source, set_cell_source

INSTRUCTOR = Path("assets/enterprise_ver2/instructor")
STUDENT = Path("assets/enterprise_ver2/student")

# Match ANY「↓↓↓ ... ↓↓↓ ... ↑↑↑ ... ↑↑↑」frame (greedy across cell — supports multiple blocks per cell, e.g. 【基础】+【进阶】)
# Group 1: open marker line (incl trailing \n)
# Group 2: body (between markers)
# Group 3: close marker line (incl leading whitespace + trailing \n)
CODE_FRAME = re.compile(
    r"(#[ \t]*↓↓↓[^\n]*?↓↓↓[^\n]*\n)(.*?)([ \t]*#[ \t]*↑↑↑[^\n]*?↑↑↑[^\n]*\n?)",
    flags=re.DOTALL,
)


def _stub_for(open_marker_line: str, indent: str) -> str:
    """Return a TODO stub appropriate for the marker (基础 vs 进阶 vs 默认)."""
    if "【基础】" in open_marker_line or "基础" in open_marker_line:
        label = "【基础】"
    elif "【进阶】" in open_marker_line or "进阶" in open_marker_line:
        label = "【进阶】"
    else:
        label = ""
    if label:
        return (
            f"{indent}# TODO: {label}完成此处\n"
            f"{indent}# 卡壳时可对比 instructor/ 同名 notebook\n"
            f"{indent}raise NotImplementedError(\"{label}未实现\")\n"
        )
    return (
        f"{indent}# TODO: 完成本练习\n"
        f"{indent}# 提示：讲师版含完整答案；如卡壳可对比 instructor/ 版本\n"
        f"{indent}pass\n"
    )


def blank_fillin_source(src: str) -> str:
    """Return a learner-facing version of an exercise cell.
    Replaces EVERY ↓↓↓...↑↑↑ block (supports both single-block legacy cells
    and 【基础】+【进阶】 dual-block Batch 5 cells)."""
    matches = list(CODE_FRAME.finditer(src))
    if matches:
        out_parts = []
        last_end = 0
        for m in matches:
            out_parts.append(src[last_end:m.start()])
            open_marker = m.group(1)
            body = m.group(2)
            close_marker = m.group(3)
            # Detect indent of body's first non-blank line
            body_lines = [ln for ln in body.splitlines() if ln.strip()]
            indent = ""
            if body_lines:
                stripped = body_lines[0].lstrip()
                indent = body_lines[0][: len(body_lines[0]) - len(stripped)]
            stub = _stub_for(open_marker, indent)
            out_parts.append(open_marker + stub + close_marker)
            last_end = m.end()
        out_parts.append(src[last_end:])
        return "".join(out_parts)

    # Strategy B: keep leading comment block (consecutive `#` lines), then stub
    lines = src.splitlines(keepends=True)
    leading = []
    rest_start = 0
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if stripped.startswith("#") or stripped == "":
            leading.append(ln)
            rest_start = i + 1
        else:
            break
    head_text = "".join(leading)
    if not head_text.endswith("\n"):
        head_text += "\n"
    return (
        head_text
        + "\n"
        + "# TODO: 完成此练习。讲师版含完整答案。\n"
        + "# 写完跑下面的 verify() (若提供) 自动判分。\n"
        + "pass\n"
    )


def derive_one(src_path: Path, dst_path: Path) -> dict:
    nb = load_nb(src_path)
    new_cells = []
    n_removed_lecture = 0
    n_blanked = 0
    n_kept = 0

    for cell in nb["cells"]:
        tags = cell.get("metadata", {}).get("tags", [])
        if "instructor_only" in tags:
            n_removed_lecture += 1
            continue
        if "fillin" in tags and cell["cell_type"] == "code":
            new_src = blank_fillin_source(cell_source(cell))
            new_cell = {
                "cell_type": "code",
                "id": cell.get("id", ""),
                "execution_count": None,
                "metadata": cell.get("metadata", {}),
                "outputs": [],  # clear outputs for fillin
                "source": new_src.splitlines(keepends=True) or [""],
            }
            new_cells.append(new_cell)
            n_blanked += 1
            continue
        # Plain pass-through (with outputs preserved if present)
        new_cells.append(cell)
        n_kept += 1

    nb["cells"] = new_cells
    save_nb(nb, dst_path)
    return {"removed_lecture": n_removed_lecture, "blanked": n_blanked, "kept": n_kept}


def main():
    STUDENT.mkdir(parents=True, exist_ok=True)
    notebooks = sorted(INSTRUCTOR.glob("*.ipynb"))
    print(f"Deriving {len(notebooks)} notebooks from instructor/ → student/\n")
    for src in notebooks:
        dst = STUDENT / src.name
        stats = derive_one(src, dst)
        print(f"  {src.name}")
        print(f"    removed (lecture):  {stats['removed_lecture']}")
        print(f"    blanked (fillin):   {stats['blanked']}")
        print(f"    kept (pass-thru):   {stats['kept']}")

    print("\nSpot-check sample:")
    sample_path = STUDENT / "Day1_上午_从文本到向量.ipynb"
    if sample_path.exists():
        nb = load_nb(sample_path)
        fillin_count = sum(
            1 for c in nb["cells"]
            if c["cell_type"] == "code"
            and "TODO: 完成" in cell_source(c)
        )
        instructor_only_count = sum(
            1 for c in nb["cells"]
            if "instructor_only" in c.get("metadata", {}).get("tags", [])
        )
        print(f"  {sample_path.name}: fillin stubs = {fillin_count}, "
              f"residual instructor_only = {instructor_only_count} (should be 0)")


if __name__ == "__main__":
    main()
