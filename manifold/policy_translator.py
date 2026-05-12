"""manifold/policy_translator.py — Validates LLM-generated policy dicts.

PolicyTranslator ensures that any policy rule produced by an LLM or external
source is schema-valid before it is applied to the live MANIFOLD infrastructure.

It also provides:
- Natural language vocabulary mapping (patient → healthcare, etc.)
- Industry compliance presets: HIPAA, GDPR, SOX, ISO 27001
"""

from __future__ import annotations

import uuid

from manifold.policy_rules import PolicyRule

# ---------------------------------------------------------------------------
# Valid values
# ---------------------------------------------------------------------------

VALID_CONDITIONS = frozenset({
    "domain",
    "domain_in",
    "stakes_gt",
    "stakes_lt",
    "risk_gt",
    "prompt_contains",
    "prompt_regex",
    "org_id",
    "tool_used",
})

VALID_ACTIONS = frozenset({
    "allow",
    "refuse",
    "escalate",
    "audit",
    "throttle",
    "redact",
    "shadow_test",
    "require_approval",
    "notify",
    "log_only",
    "quarantine",
    "rate_limit",
    "sandbox",
})

VALID_DOMAINS = frozenset({
    "healthcare",
    "finance",
    "devops",
    "legal",
    "infrastructure",
    "trading",
    "supply_chain",
    "general",
})

# ---------------------------------------------------------------------------
# Vocabulary map — natural language → MANIFOLD domain
# ---------------------------------------------------------------------------

_VOCAB_MAP: dict[str, str] = {
    # healthcare
    "patient": "healthcare",
    "phi": "healthcare",
    "medical": "healthcare",
    "medical record": "healthcare",
    "clinical": "healthcare",
    "hospital": "healthcare",
    "ehr": "healthcare",
    "hipaa": "healthcare",
    # finance
    "payroll": "finance",
    "salary": "finance",
    "compensation": "finance",
    "financial": "finance",
    "banking": "finance",
    "payment": "finance",
    "accounting": "finance",
    "invoice": "finance",
    # legal
    "contract": "legal",
    "clause": "legal",
    "jurisdiction": "legal",
    "compliance": "legal",
    "regulation": "legal",
    "litigation": "legal",
    "agreement": "legal",
    # devops
    "deploy": "devops",
    "deployment": "devops",
    "pipeline": "devops",
    "ci": "devops",
    "cd": "devops",
    "production": "devops",
    "build": "devops",
    "release": "devops",
    # infrastructure
    "server": "infrastructure",
    "network": "infrastructure",
    "firewall": "infrastructure",
    "database": "infrastructure",
    "cloud": "infrastructure",
    "storage": "infrastructure",
    # trading
    "trade": "trading",
    "stock": "trading",
    "market": "trading",
    "portfolio": "trading",
    "option": "trading",
    "equity": "trading",
    # supply_chain
    "supply": "supply_chain",
    "logistics": "supply_chain",
    "inventory": "supply_chain",
    "vendor": "supply_chain",
    "procurement": "supply_chain",
}


def translate_domain(term: str) -> str:
    """Map a natural language term to a MANIFOLD domain string.

    Returns the original term lowercased if no mapping exists.
    """
    key = term.strip().lower()
    return _VOCAB_MAP.get(key, key)


# ---------------------------------------------------------------------------
# PolicyTranslator
# ---------------------------------------------------------------------------

class PolicyTranslator:
    """Validates and normalises policy rule dicts produced by LLMs.

    Parameters
    ----------
    org_id:
        The MANIFOLD org to use as fallback when the dict has none.
    """

    def __init__(self, org_id: str = "default") -> None:
        self.org_id = org_id

    def validate_rule(self, data: dict) -> PolicyRule:
        """Validate *data* and return a :class:`PolicyRule`.

        Raises
        ------
        ValueError
            With a plain English message describing what is wrong.
        """
        if not isinstance(data, dict):
            raise ValueError("Policy rule must be a JSON object (dict).")

        # Required fields
        name = data.get("name") or data.get("rule_name") or ""
        if not name:
            raise ValueError(
                "Policy rule is missing a 'name' field.  "
                "Please give the rule a short descriptive name."
            )

        # Action
        action = str(data.get("action", "allow")).lower().strip()
        if action not in VALID_ACTIONS:
            raise ValueError(
                f"'{action}' is not a valid MANIFOLD action.  "
                f"Valid values are: {', '.join(sorted(VALID_ACTIONS))}."
            )

        # Priority
        try:
            priority = int(data.get("priority", 50))
        except (TypeError, ValueError):
            raise ValueError(
                f"Priority must be an integer between 0 and 100, "
                f"got '{data.get('priority')}'."
            )
        if not 0 <= priority <= 100:
            raise ValueError(
                f"Priority must be between 0 and 100, got {priority}."
            )

        # Conditions
        conditions: dict = data.get("conditions") or {}
        if not isinstance(conditions, dict):
            raise ValueError(
                "Conditions must be a JSON object.  "
                "Example: {\"domain\": \"healthcare\", \"risk_gt\": 0.7}"
            )
        invalid_keys = set(conditions) - VALID_CONDITIONS
        if invalid_keys:
            raise ValueError(
                f"Unknown condition key(s): {', '.join(sorted(invalid_keys))}.  "
                f"Valid keys are: {', '.join(sorted(VALID_CONDITIONS))}."
            )

        # Normalise domain value in conditions using vocab map
        if "domain" in conditions:
            conditions["domain"] = translate_domain(str(conditions["domain"]))
        if "domain_in" in conditions:
            conditions["domain_in"] = [
                translate_domain(str(d)) for d in conditions["domain_in"]
            ]

        # Validate numeric condition values
        for float_key in ("stakes_gt", "stakes_lt", "risk_gt"):
            if float_key in conditions:
                try:
                    conditions[float_key] = float(conditions[float_key])
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Condition '{float_key}' must be a float between 0 and 1, "
                        f"got '{conditions[float_key]}'."
                    )
                if not 0.0 <= conditions[float_key] <= 1.0:
                    raise ValueError(
                        f"Condition '{float_key}' must be between 0.0 and 1.0, "
                        f"got {conditions[float_key]}."
                    )

        rule_id = str(data.get("rule_id") or uuid.uuid4())
        org_id = str(data.get("org_id") or self.org_id)
        enabled = bool(data.get("enabled", True))

        return PolicyRule(
            rule_id=rule_id,
            org_id=org_id,
            name=name,
            conditions=conditions,
            action=action,
            priority=priority,
            enabled=enabled,
        )

    # ------------------------------------------------------------------
    # Industry compliance presets
    # ------------------------------------------------------------------

    @classmethod
    def hipaa_preset(cls, org_id: str = "default") -> list[PolicyRule]:
        """Return HIPAA-aligned MANIFOLD PolicyRule objects."""
        rules = [
            PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=org_id,
                name="HIPAA §164.502 — Refuse PHI disclosure without authorisation",
                conditions={"domain": "healthcare", "prompt_contains": "patient"},
                action="require_approval",
                priority=90,
            ),
            PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=org_id,
                name="HIPAA §164.312 — Audit all healthcare data access",
                conditions={"domain": "healthcare"},
                action="audit",
                priority=80,
            ),
            PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=org_id,
                name="HIPAA §164.308 — Escalate high-risk healthcare operations",
                conditions={"domain": "healthcare", "risk_gt": 0.7},
                action="escalate",
                priority=95,
            ),
            PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=org_id,
                name="HIPAA §164.514 — Redact PHI from logs",
                conditions={"domain": "healthcare", "prompt_contains": "ssn"},
                action="redact",
                priority=100,
            ),
            PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=org_id,
                name="HIPAA §164.530 — Require approval for bulk healthcare exports",
                conditions={"domain": "healthcare", "stakes_gt": 0.8},
                action="require_approval",
                priority=85,
            ),
        ]
        return rules

    @classmethod
    def gdpr_preset(cls, org_id: str = "default") -> list[PolicyRule]:
        """Return GDPR-aligned MANIFOLD PolicyRule objects."""
        rules = [
            PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=org_id,
                name="GDPR Art.5 — Refuse processing of personal data without lawful basis",
                conditions={"prompt_contains": "personal data"},
                action="require_approval",
                priority=90,
            ),
            PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=org_id,
                name="GDPR Art.17 — Right to erasure — refuse retention tasks",
                conditions={"prompt_contains": "delete my data"},
                action="require_approval",
                priority=95,
            ),
            PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=org_id,
                name="GDPR Art.22 — Flag automated decisions about individuals",
                conditions={"prompt_contains": "decision about"},
                action="audit",
                priority=80,
            ),
            PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=org_id,
                name="GDPR Art.33 — Escalate potential data breaches",
                conditions={"prompt_contains": "breach", "risk_gt": 0.6},
                action="escalate",
                priority=100,
            ),
            PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=org_id,
                name="GDPR Art.25 — Privacy by design: redact PII in output",
                conditions={"prompt_contains": "email"},
                action="redact",
                priority=70,
            ),
            PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=org_id,
                name="GDPR Art.30 — Log all data processing activities",
                conditions={"domain_in": ["healthcare", "finance", "legal"]},
                action="audit",
                priority=60,
            ),
        ]
        return rules

    @classmethod
    def sox_preset(cls, org_id: str = "default") -> list[PolicyRule]:
        """Return SOX-aligned MANIFOLD PolicyRule objects."""
        rules = [
            PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=org_id,
                name="SOX §302 — Require approval for financial statements",
                conditions={"domain": "finance", "prompt_contains": "financial statement"},
                action="require_approval",
                priority=95,
            ),
            PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=org_id,
                name="SOX §404 — Audit all financial control changes",
                conditions={"domain": "finance"},
                action="audit",
                priority=80,
            ),
            PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=org_id,
                name="SOX §802 — Refuse deletion of financial records",
                conditions={"domain": "finance", "prompt_contains": "delete"},
                action="refuse",
                priority=100,
            ),
            PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=org_id,
                name="SOX §906 — Escalate high-stakes financial operations",
                conditions={"domain": "finance", "stakes_gt": 0.75},
                action="escalate",
                priority=90,
            ),
            PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=org_id,
                name="SOX §301 — Sandbox untested financial automation",
                conditions={"domain": "finance", "risk_gt": 0.65},
                action="sandbox",
                priority=70,
            ),
            PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=org_id,
                name="SOX §201 — Refuse auditor-independence violations",
                conditions={"domain": "finance", "prompt_contains": "audit"},
                action="require_approval",
                priority=85,
            ),
        ]
        return rules

    @classmethod
    def iso27001_preset(cls, org_id: str = "default") -> list[PolicyRule]:
        """Return ISO 27001-aligned MANIFOLD PolicyRule objects."""
        rules = [
            PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=org_id,
                name="ISO 27001 A.9.4 — Refuse unauthorised system access",
                conditions={"prompt_contains": "access control"},
                action="require_approval",
                priority=90,
            ),
            PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=org_id,
                name="ISO 27001 A.12.4 — Log all privileged infrastructure actions",
                conditions={"domain": "infrastructure"},
                action="audit",
                priority=75,
            ),
            PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=org_id,
                name="ISO 27001 A.14.2 — Sandbox untested deployment code",
                conditions={"domain": "devops", "risk_gt": 0.6},
                action="sandbox",
                priority=80,
            ),
            PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=org_id,
                name="ISO 27001 A.16.1 — Escalate security incident keywords",
                conditions={"prompt_contains": "incident"},
                action="escalate",
                priority=95,
            ),
            PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=org_id,
                name="ISO 27001 A.18.1 — Audit regulatory compliance queries",
                conditions={"domain": "legal"},
                action="audit",
                priority=70,
            ),
            PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=org_id,
                name="ISO 27001 A.8.2 — Redact classified data in responses",
                conditions={"prompt_contains": "confidential"},
                action="redact",
                priority=85,
            ),
            PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=org_id,
                name="ISO 27001 A.6.1 — Notify on high-risk cross-domain tasks",
                conditions={"risk_gt": 0.8},
                action="notify",
                priority=65,
            ),
        ]
        return rules

    @classmethod
    def available_presets(cls) -> list[str]:
        """Return list of available preset names."""
        return ["hipaa", "gdpr", "sox", "iso27001"]

    @classmethod
    def apply_preset(cls, name: str, org_id: str = "default") -> list[PolicyRule]:
        """Return rules for the named preset."""
        name = name.lower().strip()
        methods = {
            "hipaa": cls.hipaa_preset,
            "gdpr": cls.gdpr_preset,
            "sox": cls.sox_preset,
            "iso27001": cls.iso27001_preset,
        }
        if name not in methods:
            raise ValueError(
                f"Unknown preset '{name}'.  "
                f"Available: {', '.join(methods)}."
            )
        return methods[name](org_id=org_id)
