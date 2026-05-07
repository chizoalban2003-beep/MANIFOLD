"""Phase 67: Self-Calibrating Learning Loop.

Reads domain stats from ManifoldDB and nudges escalation thresholds
when actual escalation rates diverge from targets.
"""
from __future__ import annotations

from dataclasses import dataclass

from manifold.policy import DOMAIN_TEMPLATES

# ---------------------------------------------------------------------------
# Target escalation rates per domain
# ---------------------------------------------------------------------------

TARGET_ESCALATION_RATES: dict[str, float] = {
    "general": 0.08,
    "healthcare": 0.18,
    "finance": 0.12,
    "devops": 0.06,
    "legal": 0.15,
    "infrastructure": 0.25,
    "trading": 0.10,
    "supply_chain": 0.07,
}

MAX_ADJUSTMENT = 0.03
MIN_SAMPLES = 50
LEARNING_RATE = 0.15

# ---------------------------------------------------------------------------
# CalibrationResult
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    domain: str
    old_threshold: float
    new_threshold: float
    actual_rate: float
    target_rate: float
    samples: int
    adjusted: bool
    reason: str

# ---------------------------------------------------------------------------
# Core calibration logic
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))

async def calibrate_domain(db, domain: str, lookback_hours: int = 24) -> CalibrationResult:
    """Calibrate escalation threshold for a single domain."""
    stats = await db.get_domain_stats(domain)
    total_tasks = stats.get("total_tasks", 0)
    total_escalations = stats.get("total_escalations", 0)
    target_rate = TARGET_ESCALATION_RATES.get(domain, 0.08)
    tpl = DOMAIN_TEMPLATES.get(domain, DOMAIN_TEMPLATES.get("general", {}))
    old_threshold = tpl.get("escalation_threshold", 0.35) if tpl else 0.35

    if total_tasks < MIN_SAMPLES:
        return CalibrationResult(
            domain=domain,
            old_threshold=old_threshold,
            new_threshold=old_threshold,
            actual_rate=0.0,
            target_rate=target_rate,
            samples=total_tasks,
            adjusted=False,
            reason=f"Insufficient samples ({total_tasks} < {MIN_SAMPLES})",
        )

    actual_rate = total_escalations / total_tasks
    error = actual_rate - target_rate
    raw_adjustment = LEARNING_RATE * error
    adjustment = _clamp(raw_adjustment, -MAX_ADJUSTMENT, MAX_ADJUSTMENT)
    new_threshold = _clamp(old_threshold + adjustment, 0.05, 0.95)

    adjusted = abs(new_threshold - old_threshold) >= 0.001
    if adjusted:
        if domain in DOMAIN_TEMPLATES and isinstance(DOMAIN_TEMPLATES[domain], dict):
            DOMAIN_TEMPLATES[domain]["escalation_threshold"] = new_threshold
        reason = f"Adjusted {old_threshold:.4f} → {new_threshold:.4f} (actual={actual_rate:.3f}, target={target_rate:.3f})"
    else:
        reason = f"No adjustment needed (actual={actual_rate:.3f}, target={target_rate:.3f})"

    return CalibrationResult(
        domain=domain,
        old_threshold=old_threshold,
        new_threshold=new_threshold,
        actual_rate=actual_rate,
        target_rate=target_rate,
        samples=total_tasks,
        adjusted=adjusted,
        reason=reason,
    )

async def run_calibration(db) -> list[CalibrationResult]:
    """Run calibrate_domain for all known domains and return all results."""
    results = []
    for domain in TARGET_ESCALATION_RATES:
        result = await calibrate_domain(db, domain)
        results.append(result)
    return results

async def calibration_report(db) -> str:
    """Return a human-readable text report of all calibration results."""
    results = await run_calibration(db)
    lines = ["MANIFOLD Calibration Report", "=" * 40]
    for r in results:
        status = "ADJUSTED" if r.adjusted else "unchanged"
        lines.append(
            f"[{r.domain}] {status}: threshold={r.new_threshold:.4f}, "
            f"actual_rate={r.actual_rate:.3f}, target={r.target_rate:.3f}, "
            f"samples={r.samples} — {r.reason}"
        )
    return "\n".join(lines)
