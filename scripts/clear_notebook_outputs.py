#!/usr/bin/env python3
"""Clear notebook execution artifacts without requiring Jupyter."""

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parents[1],
        help="directory to scan recursively (default: repository root)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report execution artifacts without writing; exit 1 if found",
    )
    args = parser.parse_args()
    root = args.directory.resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")

    changed = 0
    failed = 0
    for directory, folders, files in os.walk(root):
        # Avoid virtual environments, hidden caches, and symlinked directories.
        folders[:] = sorted(
            name
            for name in folders
            if not name.startswith(".") and name not in {"venv", "env", "node_modules"}
        )
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix != ".ipynb" or path.is_symlink():
                continue
            try:
                original = path.read_text(encoding="utf-8")
                notebook = json.loads(original)
                if notebook.get("nbformat") != 4:
                    raise ValueError("expected notebook format 4")
                before = json.dumps(notebook)
                notebook.get("metadata", {}).pop("widgets", None)
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        cell["outputs"] = []
                        cell["execution_count"] = None
                        metadata = cell.get("metadata", {})
                        for key in ("execution", "ExecuteTime"):
                            metadata.pop(key, None)
                if json.dumps(notebook) == before:
                    continue
                if not args.check:
                    path.write_text(
                        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8",
                    )
                changed += 1
                verb = "Would clear" if args.check else "Cleared"
                print(f"{verb} {path.relative_to(root)}")
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                print(f"Error: {path.relative_to(root)}: {error}")
                failed += 1

    status = "need clearing" if args.check else "cleared"
    print(f"{changed} notebook(s) {status}; {failed} error(s).")
    return int(bool(failed or (args.check and changed)))


if __name__ == "__main__":
    raise SystemExit(main())
