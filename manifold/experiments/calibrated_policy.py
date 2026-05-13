"""EXP4 — Calibrated Policy Thresholds.

Tests whether data-driven threshold calibration from vault outcomes
improves governance precision vs the current hand-tuned thresholds.
No new dependencies.  Uses synthetic or actual vault data.
"""

from __future__ import annotations

import math
import random


# ---------------------------------------------------------------------------
# ThresholdCalibrator
# ---------------------------------------------------------------------------

class ThresholdCalibrator:
    """Calibrates governance thresholds using Platt scaling and vault data."""

    def __init__(self) -> None:
        self._outcomes: list[dict] = []

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_outcomes_from_vault(self, vault_path: str | None = None) -> list[dict]:
        """Read NDJSON outcome logs from the vault directory.

        Each record should contain:
            ``risk_score``, ``stakes``, ``action``, ``was_escalated``,
            ``was_correct_to_escalate`` (bool), ``domain``.

        Returns
        -------
        list of outcome dicts (may be empty if vault is absent).
        """
        import json
        import os

        outcomes: list[dict] = []
        search_dirs: list[str] = []

        if vault_path is not None:
            search_dirs.append(vault_path)
        else:
            default = os.path.expanduser("~/.manifold/vault")
            search_dirs.append(default)

        for directory in search_dirs:
            if not os.path.isdir(directory):
                continue
            for fname in os.listdir(directory):
                if not fname.endswith(".jsonl"):
                    continue
                try:
                    with open(os.path.join(directory, fname), encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                                if "risk_score" in rec:
                                    outcomes.append(rec)
                            except json.JSONDecodeError:
                                pass
                except OSError:
                    pass

        self._outcomes = outcomes
        return outcomes

    # ------------------------------------------------------------------
    # Platt scaling (Newton's method on logistic loss, 50 iterations)
    # ------------------------------------------------------------------

    def platt_scale(
        self,
        scores: list,
        labels: list,
    ) -> tuple:
        """Fit logistic regression P(y=1|s) = sigmoid(a*s + b).

        Uses Newton's method on the logistic log-likelihood for 50 iterations.

        Returns
        -------
        (a, b) — Platt scaling parameters.
        """
        if not scores or not labels:
            return (1.0, 0.0)

        a, b = 0.0, 0.0

        for _ in range(50):
            grad_a = 0.0
            grad_b = 0.0
            h_aa = 0.0
            h_bb = 0.0
            h_ab = 0.0

            for s, y in zip(scores, labels):
                z = a * s + b
                # Numerically stable sigmoid
                if z >= 0:
                    p = 1.0 / (1.0 + math.exp(-z))
                else:
                    ez = math.exp(z)
                    p = ez / (1.0 + ez)

                p = max(1e-12, min(1.0 - 1e-12, p))
                residual = float(y) - p
                w = p * (1.0 - p)

                grad_a += residual * s
                grad_b += residual
                h_aa -= w * s * s
                h_bb -= w
                h_ab -= w * s

            det = h_aa * h_bb - h_ab * h_ab
            if abs(det) < 1e-14:
                break  # Hessian is singular — stop

            # Newton step for maximising log-likelihood: θ -= H^{-1} * grad
            da = (h_bb * grad_a - h_ab * grad_b) / det
            db = (-h_ab * grad_a + h_aa * grad_b) / det
            a -= da
            b -= db

        return (a, b)

    # ------------------------------------------------------------------
    # Calibrate threshold
    # ------------------------------------------------------------------

    def calibrate_threshold(
        self,
        domain: str = "all",
        target_precision: float = 0.90,
    ) -> float:
        """Find threshold T such that precision(T) >= target_precision.

        Parameters
        ----------
        domain:
            Filter outcomes to this domain, or ``"all"`` for all domains.
        target_precision:
            Minimum acceptable precision at the returned threshold.

        Returns
        -------
        Calibrated threshold in [0.0, 1.0].
        """
        outcomes = self._outcomes
        if domain != "all":
            outcomes = [o for o in outcomes if o.get("domain") == domain]

        if len(outcomes) < 10:
            # Insufficient data — return a conservative default
            return 0.75

        scores = [float(o.get("risk_score", 0.5)) for o in outcomes]
        labels = [1 if o.get("was_correct_to_escalate", False) else 0 for o in outcomes]

        a, b = self.platt_scale(scores, labels)

        # Scan thresholds to find where precision >= target
        best_t = 1.0
        for t_int in range(100, -1, -1):
            t = t_int / 100.0
            tp = sum(1 for s, lb in zip(scores, labels) if s >= t and lb == 1)
            fp = sum(1 for s, lb in zip(scores, labels) if s >= t and lb == 0)
            if tp + fp == 0:
                continue
            precision = tp / (tp + fp)
            if precision >= target_precision:
                best_t = t
                break

        return round(max(0.0, min(1.0, best_t)), 4)

    # ------------------------------------------------------------------
    # Calibration report
    # ------------------------------------------------------------------

    def calibration_report(self) -> dict:
        """Return per-domain calibration summary.

        Returns a dict with domain-level analysis or
        ``{status: 'insufficient_data', minimum_events_needed: 100}``
        when the vault is empty.
        """
        if len(self._outcomes) < 100:
            return {
                "status": "insufficient_data",
                "minimum_events_needed": 100,
            }

        domains = list({o.get("domain", "general") for o in self._outcomes})
        report: dict = {}
        for d in domains:
            dom_outcomes = [o for o in self._outcomes if o.get("domain", "general") == d]
            calibrated = self.calibrate_threshold(domain=d)
            report[d] = {
                "current_threshold": 0.7,
                "calibrated_threshold": calibrated,
                "expected_precision_improvement": round(calibrated - 0.7, 4),
                "sample_size": len(dom_outcomes),
            }
        return {"status": "ok", "domains": report, "total_events": len(self._outcomes)}

    # ------------------------------------------------------------------
    # Expected Calibration Error
    # ------------------------------------------------------------------

    def expected_calibration_error(self) -> float:
        """Compute ECE = Σ |confidence - accuracy| per bin (10 bins).

        Measures how well-calibrated the current risk scores are.
        Lower = better.  Returns 0.0 if no outcome data.
        """
        if not self._outcomes:
            return 0.0

        n_bins = 10
        bin_total = [0] * n_bins
        bin_correct = [0] * n_bins
        bin_conf_sum = [0.0] * n_bins

        for o in self._outcomes:
            score = float(o.get("risk_score", 0.5))
            correct = 1 if o.get("was_correct_to_escalate", False) else 0
            b = min(int(score * n_bins), n_bins - 1)
            bin_total[b] += 1
            bin_correct[b] += correct
            bin_conf_sum[b] += score

        n = len(self._outcomes)
        ece = 0.0
        for b in range(n_bins):
            if bin_total[b] == 0:
                continue
            conf = bin_conf_sum[b] / bin_total[b]
            acc = bin_correct[b] / bin_total[b]
            ece += abs(conf - acc) * bin_total[b] / n

        return round(ece, 6)


# ---------------------------------------------------------------------------
# Benchmark (uses synthetic data)
# ---------------------------------------------------------------------------

def run_calibration_benchmark() -> dict:
    """Run calibration on synthetic outcome data and return calibration metrics.

    Returns
    -------
    dict with keys matching ``calibration_report()`` output plus ECE.
    """
    # Generate synthetic outcomes
    rng = random.Random(2024)
    cal = ThresholdCalibrator()
    outcomes = []
    for _ in range(150):
        risk = rng.uniform(0.3, 1.0)
        correct = risk > 0.65 + rng.gauss(0, 0.05)
        outcomes.append({
            "risk_score": risk,
            "stakes": rng.uniform(0.4, 1.0),
            "action": "escalate" if risk > 0.7 else "proceed",
            "was_escalated": risk > 0.7,
            "was_correct_to_escalate": bool(correct),
            "domain": rng.choice(["finance", "medical", "legal"]),
        })
    cal._outcomes = outcomes
    report = cal.calibration_report()
    report["ece"] = cal.expected_calibration_error()
    return report
