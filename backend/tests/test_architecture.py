"""NFR-06: no circular dependencies between packages, verified by static
analysis — not just asserted by the architecture diagram (CLAUDE.md §6).

The contracts themselves live in `.importlinter` at the repo root, along
with the note explaining why `src.serving.api` and `src.orchestration` are
named as separate layers rather than treating "src.serving" as one block
(api.py is the composition root that wires orchestration in; orchestration
in turn depends on src.serving.inference specifically, not api.py).
"""
from __future__ import annotations

from pathlib import Path

from importlinter.cli import EXIT_STATUS_SUCCESS, lint_imports

CONFIG_PATH = Path(__file__).resolve().parents[1] / ".importlinter"


def test_no_circular_dependencies_between_packages():
    result = lint_imports(config_filename=str(CONFIG_PATH), no_cache=True, no_logo=True)
    assert result == EXIT_STATUS_SUCCESS
