"""manifold/governance_reporter.py — Plain English governance summaries.

GovernanceReporter reads from in-process stats and pipeline data to produce
natural language reports for managers.  Zero external dependencies.
"""

from __future__ import annotations

from datetime import date
from typing import Any


class GovernanceReporter:
    """Generates plain English governance summaries for MANIFOLD.

    All methods read from the live pipeline stats accessible via
    the in-process module globals so they work without a separate DB
    connection (though they also check the vault when available).

    Parameters
    ----------
    pipeline:
        The live :class:`ManifoldPipeline` instance (optional).
    rule_engine:
        The live :class:`PolicyRuleEngine` instance (optional).
    """

    def __init__(
        self,
        pipeline: Any = None,
        rule_engine: Any = None,
    ) -> None:
        self._pipeline = pipeline
        self._rule_engine = rule_engine

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_stats(self) -> dict:
        """Pull live stats from server globals if available."""
        try:
            import manifold.server as _srv  # noqa: PLC0415
            return {
                "total_tasks": getattr(_srv, "_TASK_COUNT", 0),
                "escalated": getattr(_srv, "_ESCALATION_COUNT", 0),
                "refused": getattr(_srv, "_REFUSAL_COUNT", 0),
            }
        except Exception:  # noqa: BLE001
            return {"total_tasks": 0, "escalated": 0, "refused": 0}

    def _get_prediction_log(self) -> list[dict]:
        """Return the pipeline prediction log if accessible."""
        pipeline = self._pipeline
        if pipeline is None:
            try:
                import manifold.server as _srv  # noqa: PLC0415
                pipeline = getattr(_srv, "_pipeline", None)
            except Exception:  # noqa: BLE001
                pass
        if pipeline is None:
            return []
        try:
            return list(getattr(pipeline._predictor, "_prediction_log", []))
        except Exception:  # noqa: BLE001
            return []

    def _domain_counts(self, log: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in log:
            task = entry.get("task")
            domain = getattr(task, "domain", "general") if task else "general"
            counts[domain] = counts.get(domain, 0) + 1
        return counts

    def _top_domains(self, log: list[dict], n: int = 3) -> list[str]:
        counts = self._domain_counts(log)
        return sorted(counts, key=lambda d: counts[d], reverse=True)[:n]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def daily_summary(self, org_id: str, target_date: date | None = None) -> str:
        """Return a plain English summary of one day's governance activity."""
        if target_date is None:
            target_date = date.today()

        stats = self._get_stats()
        log = self._get_prediction_log()
        total = stats["total_tasks"]
        escalated = stats["escalated"]
        refused = stats["refused"]
        permitted = max(0, total - escalated - refused)
        top_domains = self._top_domains(log)

        # Policy fire counts
        rule_engine = self._rule_engine
        if rule_engine is None:
            try:
                import manifold.server as _srv  # noqa: PLC0415
                rule_engine = getattr(_srv, "_RULE_ENGINE", None)
            except Exception:  # noqa: BLE001
                pass

        rule_summary = ""
        if rule_engine is not None:
            try:
                rules = rule_engine.rules_for_org(org_id)
                if rules:
                    rule_names = [r.name for r in rules[:3]]
                    rule_summary = (
                        f"  The most active policies are: {', '.join(rule_names)}."
                    )
            except Exception:  # noqa: BLE001
                pass

        # Improvement suggestion
        suggestion = self._suggest_improvement(stats, log)

        lines = [
            f"MANIFOLD Daily Governance Summary — {target_date.isoformat()} — Org: {org_id}",
            "",
            f"Tasks processed:   {total}",
            f"  Permitted:       {permitted}",
            f"  Escalated:       {escalated}",
            f"  Refused:         {refused}",
        ]
        if top_domains:
            lines.append(f"Most active domains: {', '.join(top_domains)}.")
        if rule_summary:
            lines.append(rule_summary)
        if suggestion:
            lines.append("")
            lines.append(f"Suggestion: {suggestion}")

        return "\n".join(lines)

    def weekly_digest(self, org_id: str) -> str:
        """Return a plain English summary of the last 7 days."""
        stats = self._get_stats()
        log = self._get_prediction_log()
        total = stats["total_tasks"]
        escalated = stats["escalated"]
        refused = stats["refused"]
        permitted = max(0, total - escalated - refused)
        top_domains = self._top_domains(log)

        escalation_rate = (escalated / total * 100) if total else 0.0
        refusal_rate = (refused / total * 100) if total else 0.0

        suggestion = self._suggest_improvement(stats, log)

        lines = [
            f"MANIFOLD Weekly Governance Digest — Org: {org_id}",
            f"Period: last 7 days  (up to {date.today().isoformat()})",
            "",
            f"Total tasks:      {total}",
            f"  Permitted:      {permitted}  ({100 - escalation_rate - refusal_rate:.1f}%)",
            f"  Escalated:      {escalated}  ({escalation_rate:.1f}%)",
            f"  Refused:        {refused}  ({refusal_rate:.1f}%)",
        ]
        if top_domains:
            lines.append(f"Top active domains: {', '.join(top_domains)}.")
        if suggestion:
            lines.append("")
            lines.append(f"Policy recommendation: {suggestion}")

        return "\n".join(lines)

    def explain_escalation(self, event_id: str) -> str:
        """Return a plain English explanation of why a task was escalated."""
        log = self._get_prediction_log()

        # Try to find matching event in the log
        for entry in reversed(log):
            decision = entry.get("decision")
            task = entry.get("task")
            if decision is None:
                continue
            action = getattr(decision, "action", "")
            if action not in ("escalate", "require_approval"):
                continue
            prompt = getattr(task, "prompt", "")[:80] if task else ""
            domain = getattr(task, "domain", "general") if task else "general"
            risk = float(getattr(decision, "risk_score", 0.0))

            return (
                f"Event: {event_id}\n"
                f"This task was escalated because it triggered MANIFOLD's escalation policy.\n"
                f"Domain: {domain}\n"
                f"Prompt excerpt: \"{prompt}\"\n"
                f"Risk score: {risk:.2f}\n\n"
                f"To prevent unnecessary escalations:\n"
                f"  1. Review the PolicyRule conditions for domain '{domain}'.\n"
                f"  2. Lower the escalation threshold if the risk score ({risk:.2f}) "
                f"is acceptable for this domain.\n"
                f"  3. Add a higher-priority 'allow' rule for trusted prompt patterns."
            )

        return (
            f"Event '{event_id}' not found in the current prediction log.  "
            f"The log may have been rotated.  "
            f"Check the MANIFOLD audit trail for historical records."
        )

    def simulate(self, org_id: str, policy_change_dict: dict) -> str:
        """Describe what would have changed if a new policy had been in place."""
        log = self._get_prediction_log()

        try:
            from manifold.policy_translator import PolicyTranslator  # noqa: PLC0415
            translator = PolicyTranslator(org_id=org_id)
            new_rule = translator.validate_rule({**policy_change_dict, "org_id": org_id})
        except Exception as exc:  # noqa: BLE001
            return f"Could not validate the proposed policy change: {exc}"

        # Simulate against the prediction log
        would_match = 0
        for entry in log:
            task = entry.get("task")
            if task is None:
                continue
            context = {
                "domain": getattr(task, "domain", "general"),
                "stakes": float(getattr(task, "stakes", 0.5)),
                "risk_score": float(getattr(task, "risk_score", 0.5)),
                "prompt": getattr(task, "prompt", ""),
                "org_id": org_id,
                "tools_used": getattr(task, "tools_used", []) or [],
            }
            if new_rule.matches(context):
                would_match += 1

        total = len(log)
        lines = [
            f"Simulation: org={org_id}, policy='{new_rule.name}'",
            f"Analysed {total} tasks from the prediction log.",
            f"  {would_match} tasks would have matched the new rule (action: {new_rule.action}).",
        ]
        if total > 0:
            pct = would_match / total * 100
            lines.append(f"  Impact: {pct:.1f}% of recent traffic would be affected.")
            if new_rule.action in ("refuse", "escalate"):
                lines.append(
                    f"  Warning: this rule would block or escalate {would_match} tasks. "
                    f"Review carefully before applying."
                )
            elif new_rule.action == "allow":
                lines.append(
                    f"  This rule would permit {would_match} tasks that may currently be blocked."
                )
        lines.append("\nTo apply this policy, use POST /rules or POST /llm/chat.")
        return "\n".join(lines)

    # ------------------------------------------------------------------

    def _suggest_improvement(self, stats: dict, log: list[dict]) -> str:
        """Return one plain English policy improvement suggestion."""
        total = stats.get("total_tasks", 0)
        escalated = stats.get("escalated", 0)
        refused = stats.get("refused", 0)

        if total == 0:
            return "No tasks have been processed yet.  Deploy MANIFOLD to start collecting governance data."

        escalation_rate = escalated / total
        refusal_rate = refused / total

        if escalation_rate > 0.3:
            return (
                f"Escalation rate is high ({escalation_rate:.0%}).  "
                f"Consider adding domain-specific 'allow' rules for trusted patterns "
                f"to reduce operational noise."
            )
        if refusal_rate > 0.2:
            return (
                f"Refusal rate is elevated ({refusal_rate:.0%}).  "
                f"Review your highest-priority 'refuse' rules — some may be too broad."
            )
        domain_counts = self._domain_counts(log)
        if domain_counts:
            top = max(domain_counts, key=lambda d: domain_counts[d])
            return (
                f"Domain '{top}' accounts for the most activity.  "
                f"Consider applying an industry compliance preset (HIPAA, GDPR, SOX, or ISO27001) "
                f"if governance in that domain is not yet formalised."
            )
        return (
            "Governance looks healthy.  "
            "Consider running 'POST /rules/preset' with a compliance preset "
            "to codify regulatory obligations."
        )
