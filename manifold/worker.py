"""ManifoldWorker — background learning loop for automatic self-improvement."""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Any


class ManifoldWorker:
    """Runs periodic learning tasks in a background daemon thread."""

    def __init__(
        self,
        pipeline: Any,
        db: Any = None,
        interval_seconds: int = 86400,
    ) -> None:
        self._pipeline = pipeline
        self._db = db
        self.interval_seconds = interval_seconds
        self._running: bool = False
        self._thread: threading.Thread | None = None
        self._last_run: float | None = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the background learning loop as a daemon thread."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Signal the background loop to stop."""
        self._running = False

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while self._running:
            time.sleep(self.interval_seconds)
            if self._running:
                self._run_once()

    # ------------------------------------------------------------------
    def _run_once(self) -> dict:
        """Execute all three learning tasks and return a summary dict."""
        errors: list[str] = []
        calibration_result = "skipped (no db)"
        rules_promoted = 0

        # Task 1 — calibration
        if self._db is not None:
            try:
                from manifold.calibrator import calibration_report
                calibration_result = asyncio.run(calibration_report(self._db))
            except Exception as exc:  # noqa: BLE001
                calibration_result = f"error: {exc}"
                errors.append(str(exc))

        # Task 2 — consolidation
        try:
            prediction_log = getattr(
                getattr(self._pipeline, "_predictor", None),
                "_prediction_log",
                [],
            )
            outcome_log = []
            for entry in prediction_log:
                task = entry.get("task")
                decision = entry.get("decision")
                if task is None or decision is None:
                    continue
                action = getattr(decision, "action", "verify")
                success = action not in ("refuse", "stop")
                outcome_log.append(
                    {
                        "domain": getattr(task, "domain", "general"),
                        "action": action,
                        "stakes": getattr(task, "stakes", 0.5),
                        "success": success,
                    }
                )
            newly_promoted = self._pipeline.nightly_consolidation(outcome_log)
            rules_promoted = len(newly_promoted)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

        # Task 3 — cognitive map flush
        try:
            prediction_log = getattr(
                getattr(self._pipeline, "_predictor", None),
                "_prediction_log",
                [],
            )
            for entry in prediction_log:
                if "actual_outcome" not in entry:
                    continue
                row = entry.get("row")
                col = entry.get("col")
                if row is None or col is None:
                    continue
                task = entry.get("task")
                decision = entry.get("decision")
                if task is None or decision is None:
                    continue
                action = getattr(decision, "action", "verify")
                success = action not in ("refuse", "stop")
                risk_score = getattr(decision, "risk_score", 0.5)
                self._pipeline.record_outcome(row, col, action, success, risk_score)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

        self._last_run = time.time()
        return {
            "calibration": calibration_result,
            "rules_promoted": rules_promoted,
            "timestamp": self._last_run,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    def status(self) -> dict:
        """Return the current worker status."""
        return {
            "running": self._running,
            "last_run": self._last_run,
            "interval_seconds": self.interval_seconds,
        }
