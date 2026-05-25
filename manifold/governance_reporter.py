"""manifold/governance_reporter.py — Plain English governance summaries.

GovernanceReporter reads from in-process stats and pipeline data to produce
natural language reports for managers.  Zero external dependencies.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any


# ---------------------------------------------------------------------------
# Industry vocabulary mapping (domain, action) → human description per type
# ---------------------------------------------------------------------------

INDUSTRY_VOCAB: dict[tuple[str, str], dict[str, str]] = {
    # Healthcare / medical
    ("healthcare", "dispense_meds"): {
        "developer": "dispense_meds",
        "doctor": "dispense medication (standing order required)",
        "executive": "medication dispensing",
        "non_technical": "give out medicine",
    },
    ("healthcare", "update_record"): {
        "developer": "update_patient_record",
        "doctor": "update clinical record",
        "executive": "patient record update",
        "non_technical": "update patient info",
    },
    # Legal
    ("legal", "process_docs"): {
        "developer": "process_docs",
        "lawyer": "document review",
        "executive": "document processing",
        "non_technical": "review documents",
    },
    ("legal", "file_motion"): {
        "developer": "file_motion",
        "lawyer": "file motion with court",
        "executive": "legal filing",
        "non_technical": "submit legal paperwork",
    },
    # Finance
    ("finance", "execute_trade"): {
        "developer": "execute_trade",
        "trader": "trade execution",
        "executive": "trade authorisation",
        "non_technical": "make a trade",
    },
    ("finance", "transfer_funds"): {
        "developer": "transfer_funds",
        "trader": "fund transfer",
        "executive": "fund transfer approval",
        "non_technical": "move money",
    },
}


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

    # ------------------------------------------------------------------
    # Progressive disclosure — generate_escalation_message
    # ------------------------------------------------------------------

    @staticmethod
    def _risk_level(risk_score: float) -> str:
        if risk_score >= 0.75:
            return "high"
        if risk_score >= 0.45:
            return "medium"
        return "low"

    @staticmethod
    def _vocab(domain: str, action: str, user_type: str) -> str:
        """Look up the human-friendly description for (domain, action, user_type)."""
        entry = INDUSTRY_VOCAB.get((domain.lower(), action.lower()))
        if entry:
            return entry.get(user_type, entry.get("executive", action))
        return action

    def generate_escalation_message(
        self,
        escalation: dict,
        user_type: str,
    ) -> str:
        """Generate a message tailored to the recipient's user type.

        Parameters
        ----------
        escalation:
            Dict with keys: action, domain, risk_score, agent_id, agent_name
            (optional), vault_id (optional), crna_values (optional),
            policy_rule_id (optional).
        user_type:
            One of: "developer" | "executive" | "doctor" | "lawyer" |
            "trader" | "non_technical"

        Returns
        -------
        str
            A formatted message appropriate for the user type.
        """
        action = str(escalation.get("action", "unknown"))
        domain = str(escalation.get("domain", "general"))
        risk_score = float(escalation.get("risk_score", 0.5))
        agent_id = str(escalation.get("agent_id", "an agent"))
        agent_name = str(escalation.get("agent_name", agent_id))
        vault_id = str(escalation.get("vault_id", ""))
        crna_values = escalation.get("crna_values", {})
        policy_rule_id = str(escalation.get("policy_rule_id", ""))
        risk_label = self._risk_level(risk_score)

        # ---- Developer ----
        if user_type == "developer":
            payload: dict[str, Any] = {
                "risk_score": round(risk_score, 4),
                "agent_id": agent_id,
                "action": action,
                "domain": domain,
            }
            if vault_id:
                payload["vault_id"] = vault_id
            if crna_values:
                payload["crna_values"] = crna_values
            if policy_rule_id:
                payload["policy_rule_id"] = policy_rule_id
            return json.dumps(payload, indent=2)

        # ---- Executive ----
        if user_type == "executive":
            human_action = self._vocab(domain, action, "executive")
            return (
                f"Agent '{agent_name}' is requesting authorisation to perform "
                f"'{human_action}' in the {domain} domain. "
                f"Risk level: {risk_label}. Please Approve or Deny."
            )

        # ---- Doctor ----
        if user_type == "doctor":
            human_action = self._vocab(domain, action, "doctor")
            patient_room = escalation.get("patient_room", "unknown room")
            medication = escalation.get("medication", "prescribed medication")
            dose = escalation.get("dose", "standard dose")
            standing_order = escalation.get("standing_order", "SO-PENDING")
            return (
                f"Clinical action required: pharmacy robot '{agent_name}' "
                f"requests to {human_action} ({medication}, {dose}) "
                f"for patient in {patient_room}. "
                f"Standing order reference: {standing_order}. "
                f"Risk level: {risk_label}. Approve or Deny."
            )

        # ---- Lawyer ----
        if user_type == "lawyer":
            human_action = self._vocab(domain, action, "lawyer")
            matter_number = escalation.get("matter_number", "MATTER-UNKNOWN")
            privilege_flag = escalation.get("privilege_flag", False)
            review_stage = escalation.get("review_stage", "initial review")
            priv_text = "⚠ Privileged document" if privilege_flag else "Non-privileged"
            return (
                f"Legal workflow: agent '{agent_name}' requests to perform "
                f"'{human_action}' ({review_stage}). "
                f"Matter: {matter_number}. {priv_text}. "
                f"Risk level: {risk_label}. Approve or Deny."
            )

        # ---- Trader ----
        if user_type == "trader":
            human_action = self._vocab(domain, action, "trader")
            instrument = escalation.get("instrument", "unspecified instrument")
            notional = escalation.get("notional_value", "N/A")
            desk = escalation.get("desk", "trading desk")
            reg_flag = escalation.get("regulatory_flag", False)
            risk_pct = f"{risk_score * 100:.0f}th percentile"
            reg_text = " ⚠ Regulatory review required." if reg_flag else ""
            return (
                f"Trade authorisation: '{agent_name}' requests {human_action} "
                f"for {instrument} (notional: {notional}) on {desk}. "
                f"Risk: {risk_pct}.{reg_text} Approve or Deny."
            )

        # ---- Non-technical ----
        # Maximum 2 short sentences, zero jargon, emoji
        human_action_simple = self._vocab(domain, action, "non_technical") or action.replace("_", " ")
        return (
            f"Your {agent_name} wants to {human_action_simple}. "
            f"Is that OK? 👍 Yes  👎 No"
        )
