"""NFR-06: no circular dependencies between packages, verified by static
analysis — not just asserted by the architecture diagram (CLAUDE.md §6).

The contracts themselves live in `.importlinter` at the repo root, along
with the note explaining why `src.serving.api` and `src.orchestration` are
named as separate layers rather than treating "src.serving" as one block
(api.py is the composition root that wires orchestration in; orchestration
in turn depends on src.serving.inference specifically, not api.py).
"""
from __future__ import annotations

import logging
from pathlib import Path

from importlinter.cli import EXIT_STATUS_SUCCESS, lint_imports

CONFIG_PATH = Path(__file__).resolve().parents[1] / ".importlinter"


def test_no_circular_dependencies_between_packages():
    # lint_imports() calls logging.config.dictConfig(), whose default
    # disable_existing_loggers=True silences every logger created so far —
    # every src.* module logger imported at collection time — for the rest
    # of the session, so any later caplog-based test captured nothing.
    loggers = [lg for lg in logging.root.manager.loggerDict.values() if isinstance(lg, logging.Logger)]
    was_disabled = {lg: lg.disabled for lg in loggers}
    try:
        result = lint_imports(config_filename=str(CONFIG_PATH), no_cache=True, no_logo=True)
    finally:
        for lg, disabled in was_disabled.items():
            lg.disabled = disabled
    assert result == EXIT_STATUS_SUCCESS
