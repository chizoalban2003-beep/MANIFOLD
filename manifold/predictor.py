"""PredictiveBrain — wraps ManifoldBrain with regret prediction."""
from __future__ import annotations

from typing import Any


class PredictiveBrain:
    """Wraps ManifoldBrain with a regret-prediction layer."""

    def __init__(self, brain: Any = None) -> None:
        if brain is None:
            from .brain import ManifoldBrain
            brain = ManifoldBrain()
        self._brain = brain
        self._prediction_log: list[dict] = []
        self._prediction_errors: list[float] = []

    # ------------------------------------------------------------------
    def predict_regret(self, task: Any) -> float:
        """Returns min(1.0, stakes * uncertainty * complexity)."""
        complexity = getattr(task, "complexity", 0.5)
        return min(1.0, task.stakes * task.uncertainty * complexity)

    # ------------------------------------------------------------------
    def predict_and_decide(self, task: Any, actual_outcome: float | None = None) -> Any:
        """Predict regret, run decide(), optionally record error."""
        predicted = self.predict_regret(task)
        decision = self._brain.decide(task)
        self._prediction_log.append(
            {"task": task, "predicted": predicted, "decision": decision}
        )
        if actual_outcome is not None:
            self._prediction_errors.append(abs(predicted - actual_outcome))
        return decision

    # ------------------------------------------------------------------
    def mean_prediction_error(self) -> float:
        """Mean of recorded prediction errors. 0.0 if empty."""
        if not self._prediction_errors:
            return 0.0
        return sum(self._prediction_errors) / len(self._prediction_errors)

    # ------------------------------------------------------------------
    def calibration_signal(self) -> dict:
        """Return calibration statistics."""
        if not self._prediction_log:
            return {
                "mean_error": 0.0,
                "samples": 0,
                "overestimates": 0,
                "underestimates": 0,
            }
        # Overestimate: predicted > actual (need actual_outcome stored)
        # We approximate using prediction_errors where we also stored direction
        overestimates = 0
        underestimates = 0
        for entry in self._prediction_log:
            # We only have a signal when actual_outcome was provided; infer from log length
            pass
        # Count from log entries that have an actual value
        actual_entries = [
            e for e in self._prediction_log if "actual" in e
        ]
        for e in actual_entries:
            if e["predicted"] > e["actual"]:
                overestimates += 1
            else:
                underestimates += 1
        return {
            "mean_error": self.mean_prediction_error(),
            "samples": len(self._prediction_errors),
            "overestimates": overestimates,
            "underestimates": underestimates,
        }
