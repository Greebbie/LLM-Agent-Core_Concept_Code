"""Keep ``assets/enterprise_5days/utils/`` in sync with the root ``utils/``.

Why this script exists
----------------------
We want a single source of truth for shared backend modules at the repo's
root ``utils/`` directory. Two strategies are supported, in order:

1. **Symlink** (preferred) — ``assets/enterprise_5days/utils/`` is a symbolic
   link to ``../../utils``. Zero drift, one source. Linux/macOS get this for
   free. Windows needs Developer Mode, an Administrator shell, or
   ``git config --global core.symlinks true`` set before clone.

2. **Copy fallback** — when symlink creation isn't permitted, the script
   refreshes a regular directory copy and drops a ``.utils_is_copy`` marker
   file inside it. Drop the marker (or re-run with privileges) once Developer
   Mode is on and the next run will upgrade to a symlink automatically.

The script is idempotent: re-running it either confirms the symlink is
correct, or refreshes the copy from the canonical source.

Usage
-----
    python tools/restore_utils_symlink.py            # link or sync
    python tools/restore_utils_symlink.py --check    # report state, do nothing
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "utils"
TARGET = REPO_ROOT / "assets" / "enterprise_5days" / "utils"
COPY_MARKER = ".utils_is_copy"


def report_state() -> str:
    if not TARGET.exists():
        return "missing"
    if TARGET.is_symlink():
        try:
            resolved = TARGET.resolve()
        except OSError:
            return "broken-symlink"
        if resolved == SOURCE.resolve():
            return "ok-symlink"
        return f"symlink-but-points-elsewhere ({resolved})"
    if TARGET.is_dir():
        if (TARGET / COPY_MARKER).exists():
            return "copy-fallback"
        return "directory-copy"
    return "unexpected-file"


def can_create_symlinks() -> bool:
    """Probe whether this OS will accept a directory symlink at TARGET.

    The probe creates a throwaway link inside ``TARGET.parent`` rather than
    the system temp dir, so it exercises the same drive and same filesystem
    as the real operation. Some virtualized or container environments accept
    same-drive symlinks but block cross-drive ones; probing on TEMP (usually
    on C:) when the repo lives on D: would falsely report failure.
    """
    if not TARGET.parent.is_dir():
        return False
    test_link = TARGET.parent / f".probe_{uuid.uuid4().hex}"
    try:
        os.symlink(SOURCE, test_link, target_is_directory=True)
    except OSError:
        return False
    finally:
        try:
            test_link.unlink()
        except OSError:
            pass
    return True


def remove_existing(target: Path) -> None:
    """Remove ``target`` whether it's a symlink, a directory, or a file."""
    if target.is_symlink() or target.is_file():
        target.unlink()
        return
    if target.is_dir():
        shutil.rmtree(target)


def install_symlink() -> None:
    if TARGET.exists() or TARGET.is_symlink():
        remove_existing(TARGET)
    rel_source = os.path.relpath(SOURCE, TARGET.parent)
    os.symlink(rel_source, TARGET, target_is_directory=True)


def install_copy() -> None:
    """Refresh the directory copy from the canonical source."""
    if TARGET.exists() or TARGET.is_symlink():
        remove_existing(TARGET)
    shutil.copytree(SOURCE, TARGET, ignore=shutil.ignore_patterns("__pycache__"))
    marker = TARGET / COPY_MARKER
    marker.write_text(
        "This directory is a copy of the repo root utils/.\n"
        "Update via:  python tools/restore_utils_symlink.py\n"
        "Delete this marker (and rerun the script with Developer Mode on)\n"
        "to upgrade to a symlink and avoid future drift.\n",
        encoding="utf-8",
    )


def install() -> int:
    state = report_state()
    print(f"current state: {state}")
    if not SOURCE.is_dir():
        print(f"ERROR: canonical source missing: {SOURCE}")
        return 2
    if state == "ok-symlink":
        print("nothing to do — symlink already points at the canonical utils/.")
        return 0
    if can_create_symlinks():
        install_symlink()
        rel = os.path.relpath(SOURCE, TARGET.parent)
        print(f"installed symlink: {TARGET} -> {rel}")
        return 0
    install_copy()
    print(
        f"refreshed directory copy at {TARGET} (symlink unavailable on this OS).\n"
        f"  marker file: {TARGET / COPY_MARKER}\n"
        "  to upgrade later: enable Windows Developer Mode (or run an Admin\n"
        "  shell) and rerun this script."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report state without modifying anything",
    )
    args = parser.parse_args()
    if args.check:
        print(report_state())
        return 0
    return install()


if __name__ == "__main__":
    sys.exit(main())
