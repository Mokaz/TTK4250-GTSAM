"""Every graded test is tagged with its part, so ``pytest -m c3`` runs (c3)'s tests."""

from __future__ import annotations

from pathlib import Path

PARTS = {"a", "b", "c1", "c2", "c3", "d", "e", "f", "g", "h"}
GRADED_FILES = {"test_preprocessing.py", "test_factor_graph.py", "test_landmark_manager.py"}


def test_every_graded_test_is_tagged_with_exactly_one_part(request) -> None:
    problems = []
    for item in request.session.items:
        markers = {marker.name for marker in item.iter_markers()}
        parts = markers & PARTS
        in_graded_file = Path(str(item.fspath)).name in GRADED_FILES
        if len(parts) > 1:
            problems.append(f"{item.nodeid}: tagged with several parts {sorted(parts)}")
        elif in_graded_file and not parts and "given" not in markers:
            problems.append(f"{item.nodeid}: not tagged with its part (e.g. @pytest.mark.c3)")

    assert not problems, "\n".join(problems)
