"""manifold/experiments/calibrated_policy.py — Per-domain threshold calibration."""

from __future__ import annotations

import random


class ThresholdCalibrator:
    """Fits per-domain decision thresholds from outcome history.

    Uses Youden's J statistic (sensitivity + specificity - 1) to find
    the threshold that maximises the sum of TPR and TNR.  Falls back to
    0.7 when no outcome data is available.
    """

    _DEFAULT_THRESHOLD = 0.7

    def __init__(self) -> None:
        # domain → list of (risk_score, ground_truth_harmful)
        self._outcomes: dict[str, list[tuple[float, bool]]] = {}

    def record_outcome(self, domain: str, risk_score: float, was_harmful: bool) -> None:
        self._outcomes.setdefault(domain, []).append((risk_score, was_harmful))

    def load_from_vault(self, vault_dir: str | None = None) -> None:
        """Load historical outcomes from a vault directory (no-op for stubs)."""

    def _youden_threshold(self, records: list[tuple[float, bool]]) -> float:
        """Return threshold maximising Youden's J from a list of (score, label)."""
        if not records:
            return self._DEFAULT_THRESHOLD
        thresholds = sorted({r[0] for r in records})
        best_j = -1.0
        best_t = self._DEFAULT_THRESHOLD
        for t in thresholds:
            tp = sum(1 for s, h in records if s > t and h)
            fp = sum(1 for s, h in records if s > t and not h)
            tn = sum(1 for s, h in records if s <= t and not h)
            fn = sum(1 for s, h in records if s <= t and h)
            tpr = tp / max(tp + fn, 1)
            tnr = tn / max(tn + fp, 1)
            j = tpr + tnr - 1.0
            if j > best_j:
                best_j = j
                best_t = t
        return best_t

    def calibration_report(
        self,
        domains: list[str] | None = None,
        seed: int = 42,
    ) -> dict[str, dict]:
        """Return per-domain calibration report.

        For domains without real outcome data, synthetic data is generated
        using the same distribution as EXP-D's simulation.
        """
        rng = random.Random(seed)
        target_domains = domains or list(self._outcomes.keys()) or [
            "finance", "healthcare", "legal", "devops", "general"
        ]
        report: dict[str, dict] = {}
        for domain in target_domains:
            records = list(self._outcomes.get(domain, []))
            if not records:
                # Generate synthetic calibration data
                adversarial_rate = {
                    "finance": 0.15, "healthcare": 0.08, "legal": 0.10,
                    "devops": 0.20, "general": 0.12,
                }.get(domain, 0.12)
                for _ in range(200):
                    harmful = rng.random() < adversarial_rate
                    if harmful:
                        score = min(1.0, max(0.0, rng.gauss(0.78, 0.12)))
                    else:
                        score = min(1.0, max(0.0, rng.gauss(0.38, 0.18)))
                    records.append((score, harmful))
            threshold = self._youden_threshold(records)
            report[domain] = {
                "calibrated_threshold": round(threshold, 4),
                "n_outcomes": len(records),
                "domain": domain,
            }
        return report
