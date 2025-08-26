#!/usr/bin/env python3
"""
Execute notebooks and save executed copies with embedded outputs.

Behavior:
- Optionally patch notebooks for synthetic data fallbacks first.
- Executes notebooks using nbclient.
- Saves executed notebooks as `<name>-executed.ipynb` alongside originals.

Usage:
  python run_notebooks.py [--patch] [NOTEBOOK ...]

If no notebooks are listed, executes all `.ipynb` files in the directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
import uuid


def _normalize_cell_ids(nb):
    for cell in nb.get("cells", []):
        if not isinstance(cell, dict):
            continue
        if "id" not in cell:
            cell["id"] = uuid.uuid4().hex


def execute_notebook(ipynb: Path, timeout: int = 600) -> Path:
    nb = nbformat.read(ipynb, as_version=4)
    _normalize_cell_ids(nb)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", allow_errors=False)
    client.execute()
    out = ipynb.with_name(ipynb.stem + "-executed.ipynb")
    nbformat.write(nb, out)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebooks", nargs="*", help="Specific notebooks to run")
    ap.add_argument("--patch", action="store_true", help="Patch notebooks to add data fallbacks before execution")
    ap.add_argument("--timeout", type=int, default=900, help="Cell execution timeout (s)")
    args = ap.parse_args(argv)

    if args.patch:
        patcher = Path(__file__).with_name("patch_notebooks_for_fallbacks.py")
        if patcher.exists():
            import runpy
            print("Patching notebooks for data fallbacks...")
            runpy.run_path(str(patcher))
        else:
            print("patch_notebooks_for_fallbacks.py not found; skipping patch step.")

    if args.notebooks:
        targets = [Path(n) for n in args.notebooks]
    else:
        targets = [p for p in Path(".").glob("*.ipynb") if not p.name.endswith("-executed.ipynb")]

    if not targets:
        print("No notebooks found.")
        return 0

    for nb in targets:
        try:
            print(f"Executing: {nb.name}")
            out = execute_notebook(nb, timeout=args.timeout)
            print(f"  -> Wrote {out.name}")
        except Exception as e:
            import traceback
            print(f"FAILED: {nb.name}: {e}")
            traceback.print_exc()
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
