"""Shared plotting helpers for course notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


CJK_FONT_FILENAME = "NotoSansCJKsc-Regular.otf"
SYSTEM_CJK_FONTS = [
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]


def _candidate_font_paths() -> list[Path]:
    """Return likely locations for the bundled CJK font.

    Notebooks may be executed from the repo root, from a chapter directory, or
    from an extracted enterprise package. Searching upward keeps the font setup
    stable across those launch locations.
    """

    candidates: list[Path] = []
    roots = [Path.cwd().resolve(), Path(__file__).resolve().parents[1]]
    for root in list(roots):
        roots.extend(root.parents)

    seen: set[Path] = set()
    for root in roots:
        for rel in (
            Path("assets") / "fonts" / CJK_FONT_FILENAME,
            Path("fonts") / CJK_FONT_FILENAME,
        ):
            path = (root / rel).resolve()
            if path not in seen:
                candidates.append(path)
                seen.add(path)
    return candidates


def find_bundled_cjk_font() -> Optional[Path]:
    """Find the bundled Noto Sans CJK font if it is available."""

    for path in _candidate_font_paths():
        if path.exists():
            return path
    return None


def setup_chinese_matplotlib(*, dpi: int = 120) -> Optional[str]:
    """Configure Matplotlib so Chinese labels render in notebooks.

    Returns the detected bundled font family name when found, otherwise None.
    """

    import matplotlib as mpl
    from matplotlib import font_manager as fm

    font_names = list(SYSTEM_CJK_FONTS)
    bundled = find_bundled_cjk_font()
    bundled_name: Optional[str] = None

    if bundled is not None:
        fm.fontManager.addfont(str(bundled))
        bundled_name = fm.FontProperties(fname=str(bundled)).get_name()
        font_names.insert(0, bundled_name)

    deduped: list[str] = []
    for name in font_names:
        if name and name not in deduped:
            deduped.append(name)

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = deduped
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["figure.dpi"] = dpi
    return bundled_name
