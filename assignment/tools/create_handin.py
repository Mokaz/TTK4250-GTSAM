"""Create the zip file to hand in on Canvas.

Run from the root of your handout::

    python create_handin.py

It collects the source files you edited plus the tuning configs you used, and
writes ``handin_<group>.zip`` next to this script. Nothing else is included --
run directories and figures are yours to keep, and the report is uploaded
separately as a pdf.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

INCLUDE_GLOBS = [
    "src/graphslam/**/*.py",
    "configs/*.yaml",
]

EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".venv", "runs", "figures"}


def collect() -> list[Path]:
    files: list[Path] = []
    for pattern in INCLUDE_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file() and not EXCLUDE_PARTS.intersection(path.parts):
                files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", default="group", help="your group number, e.g. 17")
    args = parser.parse_args()

    files = collect()
    if not files:
        raise SystemExit(f"Nothing to hand in -- run this from the handout root ({ROOT}).")

    archive = ROOT / f"handin_{args.group}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(ROOT))

    print(f"Wrote {archive.name} with {len(files)} files:")
    for path in files:
        print(f"  {path.relative_to(ROOT)}")
    print("\nUpload this zip under 'Group Assignment 2 code' on Canvas.")


if __name__ == "__main__":
    main()
