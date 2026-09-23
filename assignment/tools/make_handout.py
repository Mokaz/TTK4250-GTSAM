"""Turn this reference solution into the student handout and the LaTeX snippets.

Run from the assignment root::

    python tools/make_handout.py --out ../handout
    python tools/make_handout.py --latex ../group-assignment-2-gtsam/py

Every graded function in ``src/graphslam`` is written as::

    # TODO(a): one line saying what to do
    # BEGIN SOLUTION
    <the reference implementation>
    # END SOLUTION

``--out`` writes a copy of the tree with each such block replaced by a
``raise NotImplementedError`` stub, which is what the students receive.
``--latex`` writes ``<task>/def.tex`` and ``<task>/solu.tex`` snippets for each
block, so the assignment PDF and the code can never drift apart -- the same
mechanism the EKF-SLAM handout used.

Keeping one source of truth matters more than it sounds: the failure mode of a
hand-maintained skeleton is that it quietly stops matching the solution the
tests were written against, and nobody notices until the students do.
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

BEGIN = "# BEGIN SOLUTION"
END = "# END SOLUTION"
TODO = re.compile(r"^(?P<indent>\s*)# TODO\((?P<task>[^)]+)\):\s*(?P<text>.*)$")

SOURCE_ROOT = Path(__file__).resolve().parents[1]

# Files copied into the handout verbatim, with no solution blocks inside.
ALWAYS_COPY = ["configs", "tests", "docs", "README.md", "pyproject.toml"]


@dataclass
class SolutionBlock:
    task: str
    instruction: str
    indent: str
    qualified_name: str
    signature: list[str]
    body: list[str]


def _signature(lines: list[str], index: int) -> list[str]:
    """The ``def`` header of the function enclosing ``index``, possibly multi-line."""
    for i in range(index, -1, -1):
        if re.match(r"^\s*def\s+\w+", lines[i]):
            end = i
            while end < len(lines) and not lines[end].rstrip().endswith(":"):
                end += 1
            header = lines[i : end + 1]
            indent = len(header[0]) - len(header[0].lstrip())
            return [line[indent:] if line.startswith(" " * indent) else line for line in header]
    return []


def _qualified_name(lines: list[str], index: int) -> str:
    """Walk backwards to the enclosing def, and its enclosing class if any."""
    function = None
    for i in range(index, -1, -1):
        match = re.match(r"^(\s*)def\s+(\w+)", lines[i])
        if match and function is None:
            function = (len(match.group(1)), match.group(2))
            continue
        class_match = re.match(r"^(\s*)class\s+(\w+)", lines[i])
        if class_match and function and len(class_match.group(1)) < function[0]:
            return f"{class_match.group(2)}.{function[1]}"
    return function[1] if function else "unknown"


def parse(path: Path) -> tuple[list[str], list[SolutionBlock]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[SolutionBlock] = []

    index = 0
    while index < len(lines):
        if lines[index].strip() == BEGIN:
            start = index
            end = start + 1
            while end < len(lines) and lines[end].strip() != END:
                end += 1
            if end == len(lines):
                raise SystemExit(f"{path}: unterminated {BEGIN} at line {start + 1}")

            todo = TODO.match(lines[start - 1]) if start > 0 else None
            if todo is None:
                raise SystemExit(
                    f"{path}: the line before {BEGIN} (line {start}) must be a "
                    f'"# TODO(<task>): <instruction>" comment'
                )

            blocks.append(
                SolutionBlock(
                    task=todo.group("task"),
                    instruction=todo.group("text"),
                    indent=todo.group("indent"),
                    qualified_name=_qualified_name(lines, start),
                    signature=_signature(lines, start),
                    body=lines[start + 1 : end],
                )
            )
            index = end + 1
            continue
        index += 1

    return lines, blocks


def strip(path: Path) -> str:
    """Return the student version of one file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []

    index = 0
    while index < len(lines):
        if lines[index].strip() == BEGIN:
            end = index + 1
            while end < len(lines) and lines[end].strip() != END:
                end += 1

            todo = TODO.match(lines[index - 1])
            indent = todo.group("indent") if todo else ""
            task = todo.group("task") if todo else "?"

            out.append(f'{indent}raise NotImplementedError("Task 1 ({task})")')
            index = end + 1
            continue

        out.append(lines[index])
        index += 1

    return "\n".join(out) + "\n"


def write_handout(destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    for name in ALWAYS_COPY:
        source = SOURCE_ROOT / name
        if not source.exists():
            continue
        if source.is_dir():
            shutil.copytree(
                source, destination / name, ignore=shutil.ignore_patterns("__pycache__")
            )
        else:
            shutil.copy2(source, destination / name)

    stripped = 0
    for source in sorted((SOURCE_ROOT / "src").rglob("*.py")):
        if "__pycache__" in source.parts:
            continue
        target = destination / source.relative_to(SOURCE_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = strip(source)
        target.write_text(text, encoding="utf-8")
        stripped += text.count("raise NotImplementedError")

    shutil.copy2(Path(__file__).parent / "create_handin.py", destination / "create_handin.py")

    # The data set is large and lives at the repository root, not inside the
    # assignment tree; copy whichever copy we can find.
    for candidate in (SOURCE_ROOT / "data", SOURCE_ROOT.parent / "data"):
        if candidate.is_dir():
            shutil.copytree(candidate, destination / "data")
            break
    else:
        print("  warning: no data/ directory found -- add it to the handout by hand")

    print(f"handout written to {destination} ({stripped} stubs)")


def write_latex(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    written = 0

    for source in sorted((SOURCE_ROOT / "src").rglob("*.py")):
        if "__pycache__" in source.parts:
            continue
        _, blocks = parse(source)

        for block in blocks:
            folder = destination / block.task
            folder.mkdir(parents=True, exist_ok=True)

            module = source.stem
            (folder / "def.tex").write_text(
                f"\\pythoninline{{{module}.{block.qualified_name}}}\\unskip\n",
                encoding="utf-8",
            )

            # Re-indent the body by one level under the reproduced signature, so
            # the snippet in the PDF is a complete, copy-pasteable function.
            dedented = [
                line[len(block.indent) :] if line.startswith(block.indent) else line
                for line in block.body
            ]
            reindented = [("    " + line) if line.strip() else line for line in dedented]
            snippet = block.signature + reindented

            (folder / "solu.tex").write_text(
                "\\begin{pythoncode}\n" + "\n".join(snippet) + "\n\\end{pythoncode}\n",
                encoding="utf-8",
            )
            written += 1

    print(f"{written} LaTeX snippet pairs written to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="write the stripped student handout here")
    parser.add_argument("--latex", type=Path, help="write def.tex/solu.tex snippets here")
    parser.add_argument("--list", action="store_true", help="list the graded blocks and exit")
    args = parser.parse_args()

    if args.list or not (args.out or args.latex):
        for source in sorted((SOURCE_ROOT / "src").rglob("*.py")):
            if "__pycache__" in source.parts:
                continue
            _, blocks = parse(source)
            for block in blocks:
                print(
                    f"  ({block.task})  {source.relative_to(SOURCE_ROOT)}"
                    f"::{block.qualified_name}  -- {block.instruction}"
                )
        return

    if args.out:
        write_handout(args.out.resolve())
    if args.latex:
        write_latex(args.latex.resolve())


if __name__ == "__main__":
    main()
