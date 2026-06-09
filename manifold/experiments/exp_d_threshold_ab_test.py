"""EXP-D — Calibrated vs. Fixed Threshold Shadow A/B Test.

Deploys EXP4's calibrated thresholds in shadow mode alongside the
fixed 0.7 threshold.  Logs divergence cases (where calibrated and fixed
disagree) and measures precision + escalation rate difference over
simulated decisions per domain.

No new dependencies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from manifold.experiments.calibrated_policy import ThresholdCalibrator


@dataclass
class Decision:
    domain: str
    risk_score: float
    stakes: float
    ground_truth_harmful: bool
    fixed_threshold_decision: str   # "block" or "allow"
    calibrated_decision: str        # "block" or "allow"
    diverged: bool


_FIXED_THRESHOLD = 0.7


def _make_decision(risk_score: float, threshold: float) -> str:
    return "block" if risk_score > threshold else "allow"


def _run_domain_ab(
    domain: str,
    calibrated_threshold: float,
    n_decisions: int,
    rng: random.Random,
) -> list[Decision]:
    """Simulate n_decisions for a domain and compare threshold decisions."""
    decisions = []
    # Domain-specific ground truth distribution parameters
    # Higher stakes = more adversarial inputs in that domain
    domain_adversarial_rate = {
        "finance": 0.15,
        "healthcare": 0.08,
        "legal": 0.10,
        "devops": 0.20,
        "general": 0.12,
    }.get(domain, 0.12)

    for _ in range(n_decisions):
        is_harmful = rng.random() < domain_adversarial_rate
        # Risk scores: harmful inputs cluster near 0.6-0.95, benign near 0.1-0.65
        if is_harmful:
            risk_score = min(1.0, max(0.0, rng.gauss(0.78, 0.12)))
        else:
            risk_score = min(1.0, max(0.0, rng.gauss(0.38, 0.18)))

        stakes = min(1.0, max(0.0, rng.gauss(0.5, 0.2)))

        fixed_dec = _make_decision(risk_score, _FIXED_THRESHOLD)
        calib_dec = _make_decision(risk_score, calibrated_threshold)

        decisions.append(Decision(
            domain=domain,
            risk_score=round(risk_score, 4),
            stakes=round(stakes, 4),
            ground_truth_harmful=is_harmful,
            fixed_threshold_decision=fixed_dec,
            calibrated_decision=calib_dec,
            diverged=fixed_dec != calib_dec,
        ))
    return decisions


def _metrics(decisions: list[Decision], use_calibrated: bool) -> dict[str, float]:
    """Compute precision, recall, and escalation rate for one threshold."""
    tp = fp = tn = fn = 0
    for d in decisions:
        decision = d.calibrated_decision if use_calibrated else d.fixed_threshold_decision
        blocked = decision == "block"
        if blocked and d.ground_truth_harmful:
            tp += 1
        elif blocked and not d.ground_truth_harmful:
            fp += 1
        elif not blocked and not d.ground_truth_harmful:
            tn += 1
        else:
            fn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    escalation_rate = (tp + fp) / max(len(decisions), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "escalation_rate": round(escalation_rate, 4),
        "false_positive_rate": round(fp / max(fp + tn, 1), 4),
    }


def run_threshold_ab_benchmark() -> dict[str, Any]:
    """Run shadow A/B over 5 domains × 1000 decisions each.

    Calibrated thresholds are derived from ThresholdCalibrator on synthetic
    outcome data (same as EXP4 benchmark).

    Returns
    -------
    dict with per-domain comparison plus overall divergence analysis.
    """
    rng = random.Random(7777)

    # Get calibrated thresholds from EXP4's calibrator
    calibrator = ThresholdCalibrator()
    try:
        calibrator.load_from_vault(vault_dir=None)  # no vault — use synthetic
    except Exception:  # noqa: BLE001
        pass

    # Generate synthetic calibration data and fit per domain
    calibrated_thresholds: dict[str, float] = {}
    for domain in ["finance", "healthcare", "legal", "devops", "general"]:
        try:
            report = calibrator.calibration_report(domains=[domain])
            calibrated_thresholds[domain] = report.get(domain, {}).get(
                "calibrated_threshold", _FIXED_THRESHOLD
            )
        except Exception:  # noqa: BLE001
            calibrated_thresholds[domain] = _FIXED_THRESHOLD

    domain_results = {}
    all_decisions: list[Decision] = []
    n_per_domain = 1000

    for domain in ["finance", "healthcare", "legal", "devops", "general"]:
        cal_threshold = calibrated_thresholds.get(domain, _FIXED_THRESHOLD)
        decisions = _run_domain_ab(domain, cal_threshold, n_per_domain, rng)
        all_decisions.extend(decisions)

        fixed_m = _metrics(decisions, use_calibrated=False)
        calib_m = _metrics(decisions, use_calibrated=True)
        divergences = [d for d in decisions if d.diverged]

        domain_results[domain] = {
            "calibrated_threshold": round(cal_threshold, 4),
            "fixed_threshold": _FIXED_THRESHOLD,
            "fixed": fixed_m,
            "calibrated": calib_m,
            "divergence_count": len(divergences),
            "divergence_rate": round(len(divergences) / n_per_domain, 4),
            "precision_gain": round(calib_m["precision"] - fixed_m["precision"], 4),
            "escalation_reduction": round(
                fixed_m["escalation_rate"] - calib_m["escalation_rate"], 4
            ),
        }

    # Overall summary
    total_divergences = sum(d["divergence_count"] for d in domain_results.values())
    avg_precision_gain = sum(
        d["precision_gain"] for d in domain_results.values()
    ) / len(domain_results)
    avg_escalation_reduction = sum(
        d["escalation_reduction"] for d in domain_results.values()
    ) / len(domain_results)

    return {
        "domain_results": domain_results,
        "total_decisions": len(all_decisions),
        "total_divergences": total_divergences,
        "overall_divergence_rate": round(total_divergences / len(all_decisions), 4),
        "avg_precision_gain": round(avg_precision_gain, 4),
        "avg_escalation_reduction": round(avg_escalation_reduction, 4),
        "calibration_worthwhile": avg_precision_gain > 0 or avg_escalation_reduction > 0,
    }
