"""Preset simulation windows over stored market history.

Each window was picked because it shows something; that is what makes it a
demonstration and not a result. What the model decides, and whether the gate
promotes or rejects, is computed live each time and not scripted.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

# Calendar days of history the sandbox starts with: 730 for training, the
# evaluation window before that, and indicator/z-score warm-up before that.
HISTORY_DAYS = 1100


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    description: str
    start: date
    days: int  # trading days available to step through
    crash_first_retrain: bool = False

    @property
    def history_start(self) -> date:
        return self.start - timedelta(days=HISTORY_DAYS)

    @property
    def end(self) -> date:
        return self.start + timedelta(days=self.days * 2)


SCENARIOS: dict[str, Scenario] = {
    s.id: s
    for s in (
        Scenario(
            "volatility", "Volatility shock (2020)",
            "Real data through the February 2020 crash. VIX crosses 35: the fail-safe takes over "
            "and retraining is deferred until it subsides.",
            date(2020, 2, 18), 25,
        ),
        Scenario(
            "decay", "Decay, retraining and the gate (2022)",
            "Real data. The loop checks its model each day, retrains when the rolling Sharpe falls "
            "below target, and the acceptance gate promotes or rejects the candidate.",
            date(2022, 6, 1), 25,
        ),
        Scenario(
            "crash", "Retraining failure (2022)",
            "The same window with an injected fault: the first retrain after start-up crashes. "
            "The current model keeps serving and the next day tries again.",
            date(2022, 6, 1), 25, crash_first_retrain=True,
        ),
    )
}
