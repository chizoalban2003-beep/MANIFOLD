"""Phase 18: Policy-as-Code — the MANIFOLD "Manifesto".

As organisations adopt MANIFOLD, compliance officers need a way to
**version-control their moral compass**.  Hard-coding thresholds into Python
is fine for developers, but not for audit trails.

Phase 18 formalises the outputs of ``AutoRuleDiscovery`` (Phase 12) and the
``InterceptorConfig`` (Phase 13) into a portable **``manifold.yaml``** file.

Key concepts
------------
``PolicyDomain``
    Declarative configuration for a single business domain: stakes,
    risk tolerance, coordination tax cap, and fallback strategy.

``ManifoldPolicy``
    The top-level policy document.  Wraps one or more ``PolicyDomain``
    entries plus global interceptor settings.  Can be serialised to YAML /
    dict and loaded back.

``RuleDiff``
    When ``PenaltyOptimizer`` proposes a penalty change, a ``RuleDiff``
    records it as a reviewable delta — essentially a "Pull Request" against
    the policy file.

``PolicyExporter``
    Bridges live MANIFOLD runtime state (``AutoRuleDiscovery``,
    ``InterceptorConfig``) → ``ManifoldPolicy`` object → YAML string.

``PolicyLoader``
    Reads a YAML string or dict → ``ManifoldPolicy`` object.

Example
-------
::

    from manifold.policy import PolicyExporter, PolicyLoader

    exporter = PolicyExporter(discovery=auto_discovery, config=interceptor_cfg)
    policy = exporter.export(domains=["finance", "legal"])
    yaml_str = policy.to_yaml()

    # Later — reload and apply
    policy2 = PolicyLoader.from_yaml(yaml_str)
    cfg = policy2.interceptor_config()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .autodiscovery import AutoRuleDiscovery, PenaltyProposal
from .interceptor import InterceptorConfig


# ---------------------------------------------------------------------------
# PolicyDomain
# ---------------------------------------------------------------------------


@dataclass
class PolicyDomain:
    """Declarative governance configuration for a single business domain.

    Parameters
    ----------
    name:
        Domain identifier (e.g. ``"finance"``, ``"legal"``).
    stakes:
        Default task stakes for this domain [0, 1].
    risk_tolerance:
        Maximum acceptable risk score before veto [0, 1].
    coordination_tax_cap:
        Maximum fraction of utility that coordination overhead may consume.
    fallback_strategy:
        What to do when a call is vetoed: ``"hitl"``, ``"fallback"``, or
        ``"refuse"``.
    min_tool_reliability:
        Minimum tool reliability required for a call to proceed.
    notes:
        Free-text governance annotation.

    Example
    -------
    ::

        domain = PolicyDomain(
            name="finance",
            stakes=0.9,
            risk_tolerance=0.3,
            coordination_tax_cap=0.15,
            fallback_strategy="hitl",
        )
    """

    name: str
    stakes: float = 0.5
    risk_tolerance: float = 0.45
    coordination_tax_cap: float = 0.20
    fallback_strategy: str = "hitl"
    min_tool_reliability: float = 0.70
    notes: str = ""
    escalation_threshold: float = 0.35
    refusal_threshold: float = 0.2
    verification_cost: float = 0.15
    penalty_scale: float = 1.5
    allowed_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (YAML-serialisable)."""
        return {
            "name": self.name,
            "stakes": round(self.stakes, 4),
            "risk_tolerance": round(self.risk_tolerance, 4),
            "coordination_tax_cap": round(self.coordination_tax_cap, 4),
            "fallback_strategy": self.fallback_strategy,
            "min_tool_reliability": round(self.min_tool_reliability, 4),
            "notes": self.notes,
            "escalation_threshold": round(self.escalation_threshold, 4),
            "refusal_threshold": round(self.refusal_threshold, 4),
            "verification_cost": round(self.verification_cost, 4),
            "penalty_scale": round(self.penalty_scale, 4),
            "allowed_actions": list(self.allowed_actions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicyDomain":
        """Deserialise from a plain dict."""
        return cls(
            name=str(data["name"]),
            stakes=float(data.get("stakes", 0.5)),
            risk_tolerance=float(data.get("risk_tolerance", 0.45)),
            coordination_tax_cap=float(data.get("coordination_tax_cap", 0.20)),
            fallback_strategy=str(data.get("fallback_strategy", "hitl")),
            min_tool_reliability=float(data.get("min_tool_reliability", 0.70)),
            notes=str(data.get("notes", "")),
            escalation_threshold=float(data.get("escalation_threshold", 0.35)),
            refusal_threshold=float(data.get("refusal_threshold", 0.2)),
            verification_cost=float(data.get("verification_cost", 0.15)),
            penalty_scale=float(data.get("penalty_scale", 1.5)),
            allowed_actions=list(data.get("allowed_actions", [])),
        )


# ---------------------------------------------------------------------------
# RuleDiff
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleDiff:
    """A proposed change to the policy, derived from ``PenaltyOptimizer``.

    Think of this as a "Pull Request" against the ``manifold.yaml``.  The
    diff records the old and new values for review.  A compliance officer
    can approve or reject the diff before it is applied.

    Attributes
    ----------
    rule_name:
        The rule being updated.
    trigger:
        The trigger string.
    domain:
        Domain this rule belongs to.
    field:
        The policy field being changed (currently always ``"risk_tolerance"``
        or ``"stakes"``; extended in future phases).
    old_value:
        Current value.
    new_value:
        Proposed value.
    rationale:
        Human-readable explanation from ``PenaltyOptimizer``.
    confidence:
        How confident the optimizer is [0, 1].
    status:
        ``"pending"``, ``"approved"``, or ``"rejected"``.
    """

    rule_name: str
    trigger: str
    domain: str
    field: str
    old_value: float
    new_value: float
    rationale: str
    confidence: float
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "rule_name": self.rule_name,
            "trigger": self.trigger,
            "domain": self.domain,
            "field": self.field,
            "old_value": round(self.old_value, 4),
            "new_value": round(self.new_value, 4),
            "rationale": self.rationale,
            "confidence": round(self.confidence, 4),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuleDiff":
        """Deserialise from a plain dict."""
        return cls(
            rule_name=str(data["rule_name"]),
            trigger=str(data["trigger"]),
            domain=str(data.get("domain", "general")),
            field=str(data.get("field", "risk_tolerance")),
            old_value=float(data["old_value"]),
            new_value=float(data["new_value"]),
            rationale=str(data.get("rationale", "")),
            confidence=float(data.get("confidence", 0.5)),
            status=str(data.get("status", "pending")),
        )

    def apply(self, policy: "ManifoldPolicy") -> "ManifoldPolicy":
        """Apply this diff to a ``ManifoldPolicy`` and return a new policy.

        Only applies diffs with ``status == "approved"``.

        Parameters
        ----------
        policy:
            The policy to update.

        Returns
        -------
        ManifoldPolicy
            A new policy instance with the diff applied (or unchanged if
            status is not ``"approved"``).
        """
        if self.status != "approved":
            return policy

        domains = {d.name: d for d in policy.domains}
        domain = domains.get(self.domain)
        if domain is None:
            return policy

        # Build updated domain
        kwargs = domain.to_dict()
        kwargs[self.field] = self.new_value
        updated_domain = PolicyDomain.from_dict(kwargs)
        domains[self.domain] = updated_domain

        return ManifoldPolicy(
            version=policy.version,
            description=policy.description,
            domains=list(domains.values()),
            global_veto_threshold=policy.global_veto_threshold,
            global_fallback_strategy=policy.global_fallback_strategy,
            pending_diffs=[d for d in policy.pending_diffs if d is not self],
        )


# ---------------------------------------------------------------------------
# ManifoldPolicy
# ---------------------------------------------------------------------------

# Domain-template presets — reusable "constitutions"
DOMAIN_TEMPLATES: dict[str, dict[str, Any]] = {
    "finance": {
        "stakes": 0.9,
        "risk_tolerance": 0.30,
        "coordination_tax_cap": 0.10,
        "fallback_strategy": "hitl",
        "min_tool_reliability": 0.85,
        "notes": "High-stakes financial operations; minimal risk tolerance.",
        "escalation_threshold": 0.4,
        "refusal_threshold": 0.25,
        "verification_cost": 0.15,
        "penalty_scale": 1.8,
        "allowed_actions": ["answer", "clarify", "verify", "retrieve", "use_tool", "escalate", "refuse"],
    },
    "legal": {
        "stakes": 0.85,
        "risk_tolerance": 0.25,
        "coordination_tax_cap": 0.12,
        "fallback_strategy": "hitl",
        "min_tool_reliability": 0.88,
        "notes": "Legal and compliance domain; strict veto on uncertainty.",
        "escalation_threshold": 0.35,
        "refusal_threshold": 0.2,
        "verification_cost": 0.20,
        "penalty_scale": 2.0,
        "allowed_actions": ["answer", "clarify", "verify", "retrieve", "escalate", "refuse"],
    },
    "creative": {
        "stakes": 0.3,
        "risk_tolerance": 0.70,
        "coordination_tax_cap": 0.35,
        "fallback_strategy": "fallback",
        "min_tool_reliability": 0.60,
        "notes": "Creative domain; high tolerance for experimental tools.",
    },
    "support": {
        "stakes": 0.5,
        "risk_tolerance": 0.50,
        "coordination_tax_cap": 0.20,
        "fallback_strategy": "hitl",
        "min_tool_reliability": 0.75,
        "notes": "Customer support; balance speed and safety.",
    },
    "research": {
        "stakes": 0.4,
        "risk_tolerance": 0.60,
        "coordination_tax_cap": 0.30,
        "fallback_strategy": "fallback",
        "min_tool_reliability": 0.65,
        "notes": "Research and exploration; accept higher uncertainty.",
    },
    "medical": {
        "stakes": 0.95,
        "risk_tolerance": 0.15,
        "coordination_tax_cap": 0.05,
        "fallback_strategy": "refuse",
        "min_tool_reliability": 0.92,
        "notes": "Clinical domain; near-zero risk tolerance.",
    },
    "general": {
        "stakes": 0.5,
        "risk_tolerance": 0.45,
        "coordination_tax_cap": 0.20,
        "fallback_strategy": "hitl",
        "min_tool_reliability": 0.70,
        "notes": "Default general-purpose policy.",
    },
    # ---------------------------------------------------------------------------
    # Priority 6 — Extended domain templates
    # ---------------------------------------------------------------------------
    "healthcare": {
        "stakes": 0.95,
        "risk_tolerance": 0.15,
        "coordination_tax_cap": 0.05,
        "fallback_strategy": "hitl",
        "min_tool_reliability": 0.90,
        "notes": "Healthcare domain; escalate early — stakes are life.",
        "escalation_threshold": 0.3,
        "refusal_threshold": 0.15,
        "verification_cost": 0.25,
        "penalty_scale": 2.5,
        "allowed_actions": ["answer", "clarify", "verify", "retrieve", "escalate", "refuse"],
    },
    "cybersecurity": {
        "stakes": 0.95,
        "risk_tolerance": 0.10,
        "coordination_tax_cap": 0.05,
        "fallback_strategy": "refuse",
        "min_tool_reliability": 0.92,
        "notes": "Cybersecurity domain; maximum verification, minimal tolerance.",
        "escalation_threshold": 0.25,
        "refusal_threshold": 0.1,
        "verification_cost": 0.30,
        "penalty_scale": 3.0,
        "allowed_actions": ["verify", "clarify", "retrieve", "escalate", "refuse"],
    },
    "ecommerce": {
        "stakes": 0.5,
        "risk_tolerance": 0.45,
        "coordination_tax_cap": 0.18,
        "fallback_strategy": "hitl",
        "min_tool_reliability": 0.72,
        "notes": "E-commerce domain; balance speed and safety for order fulfilment.",
        "escalation_threshold": 0.6,
        "refusal_threshold": 0.45,
        "verification_cost": 0.08,
        "penalty_scale": 1.2,
        "allowed_actions": ["answer", "clarify", "use_tool", "verify", "escalate"],
    },
}


@dataclass
class ManifoldPolicy:
    """The top-level MANIFOLD policy document.

    A ``ManifoldPolicy`` encodes all governance decisions in a
    format that can be serialised to YAML, committed to version control,
    reviewed by compliance officers, and loaded at runtime.

    Parameters
    ----------
    version:
        Semantic version string (e.g. ``"1.0.0"``).
    description:
        Free-text description of the policy's purpose.
    domains:
        List of domain-level configurations.
    global_veto_threshold:
        Fallback risk veto threshold applied when no domain-specific config
        matches.  Default: ``0.45``.
    global_fallback_strategy:
        Fallback strategy when no domain-specific config matches.
        Default: ``"hitl"``.
    pending_diffs:
        Unreviewed ``RuleDiff`` objects proposed by the ``PenaltyOptimizer``.

    Example
    -------
    ::

        policy = ManifoldPolicy.from_template("finance")
        yaml_str = policy.to_yaml()
        print(yaml_str)
    """

    version: str = "1.0.0"
    description: str = "MANIFOLD governance policy"
    domains: list[PolicyDomain] = field(default_factory=list)
    global_veto_threshold: float = 0.45
    global_fallback_strategy: str = "hitl"
    pending_diffs: list[RuleDiff] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_template(cls, template_name: str, **kwargs: Any) -> "ManifoldPolicy":
        """Build a policy pre-seeded with a named domain template.

        Parameters
        ----------
        template_name:
            One of ``"finance"``, ``"legal"``, ``"creative"``, ``"support"``,
            ``"research"``, ``"medical"``, ``"general"``.
        **kwargs:
            Override any ``ManifoldPolicy`` constructor arguments.

        Returns
        -------
        ManifoldPolicy

        Raises
        ------
        ValueError
            If *template_name* is not a known template.
        """
        if template_name not in DOMAIN_TEMPLATES:
            known = list(DOMAIN_TEMPLATES.keys())
            raise ValueError(
                f"Unknown template {template_name!r}. Known: {known}"
            )
        tpl = DOMAIN_TEMPLATES[template_name].copy()
        tpl["name"] = template_name
        domain = PolicyDomain.from_dict(tpl)
        return cls(domains=[domain], **kwargs)

    @classmethod
    def default(cls) -> "ManifoldPolicy":
        """Return a minimal default policy covering the ``"general"`` domain."""
        return cls.from_template("general")

    # ------------------------------------------------------------------
    # Domain lookup
    # ------------------------------------------------------------------

    def domain(self, name: str) -> PolicyDomain | None:
        """Return the ``PolicyDomain`` for *name*, or ``None`` if not found."""
        for d in self.domains:
            if d.name == name:
                return d
        return None

    def domain_names(self) -> list[str]:
        """Return all configured domain names."""
        return [d.name for d in self.domains]

    def interceptor_config(self, domain_name: str | None = None) -> InterceptorConfig:
        """Build an ``InterceptorConfig`` from the policy.

        Uses the domain-specific settings when *domain_name* is given and
        a matching ``PolicyDomain`` exists; otherwise falls back to the
        global settings.

        Parameters
        ----------
        domain_name:
            Optional domain to look up.

        Returns
        -------
        InterceptorConfig
        """
        domain = self.domain(domain_name) if domain_name else None
        if domain is not None:
            return InterceptorConfig(
                risk_veto_threshold=domain.risk_tolerance,
                redirect_strategy=domain.fallback_strategy,
                fallback_min_reliability=domain.min_tool_reliability,
            )
        return InterceptorConfig(
            risk_veto_threshold=self.global_veto_threshold,
            redirect_strategy=self.global_fallback_strategy,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (ready for YAML/JSON serialisation)."""
        return {
            "version": self.version,
            "description": self.description,
            "global_veto_threshold": round(self.global_veto_threshold, 4),
            "global_fallback_strategy": self.global_fallback_strategy,
            "domains": [d.to_dict() for d in self.domains],
            "pending_diffs": [diff.to_dict() for diff in self.pending_diffs],
        }

    def to_yaml(self) -> str:
        """Serialise to a YAML-formatted string (no PyYAML dependency).

        Uses a hand-written minimal YAML serialiser that handles dicts,
        lists, strings, ints, and floats — sufficient for ``manifold.yaml``.

        Returns
        -------
        str
            A valid YAML document.
        """
        return _dict_to_yaml(self.to_dict())

    def to_json(self) -> str:
        """Serialise to a JSON string.

        Returns
        -------
        str
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManifoldPolicy":
        """Deserialise from a plain dict.

        Parameters
        ----------
        data:
            Dict as produced by ``to_dict()``.

        Returns
        -------
        ManifoldPolicy
        """
        domains = [PolicyDomain.from_dict(d) for d in data.get("domains", [])]
        diffs = [RuleDiff.from_dict(d) for d in data.get("pending_diffs", [])]
        return cls(
            version=str(data.get("version", "1.0.0")),
            description=str(data.get("description", "")),
            domains=domains,
            global_veto_threshold=float(data.get("global_veto_threshold", 0.45)),
            global_fallback_strategy=str(data.get("global_fallback_strategy", "hitl")),
            pending_diffs=diffs,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ManifoldPolicy":
        """Deserialise from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    # ------------------------------------------------------------------
    # Diff management
    # ------------------------------------------------------------------

    def add_diff(self, diff: RuleDiff) -> None:
        """Add a ``RuleDiff`` to the pending queue."""
        self.pending_diffs.append(diff)

    def approve_diff(self, rule_name: str) -> "ManifoldPolicy":
        """Approve the first pending diff for *rule_name* and apply it.

        Parameters
        ----------
        rule_name:
            Rule name to approve.

        Returns
        -------
        ManifoldPolicy
            Updated policy with the diff applied.
        """
        for i, diff in enumerate(self.pending_diffs):
            if diff.rule_name == rule_name and diff.status == "pending":
                approved = RuleDiff(
                    rule_name=diff.rule_name,
                    trigger=diff.trigger,
                    domain=diff.domain,
                    field=diff.field,
                    old_value=diff.old_value,
                    new_value=diff.new_value,
                    rationale=diff.rationale,
                    confidence=diff.confidence,
                    status="approved",
                )
                # Replace in pending list then apply
                self.pending_diffs[i] = approved
                return approved.apply(self)
        return self  # nothing to approve

    def reject_diff(self, rule_name: str) -> None:
        """Mark the first pending diff for *rule_name* as rejected (in-place)."""
        for i, diff in enumerate(self.pending_diffs):
            if diff.rule_name == rule_name and diff.status == "pending":
                self.pending_diffs[i] = RuleDiff(
                    rule_name=diff.rule_name,
                    trigger=diff.trigger,
                    domain=diff.domain,
                    field=diff.field,
                    old_value=diff.old_value,
                    new_value=diff.new_value,
                    rationale=diff.rationale,
                    confidence=diff.confidence,
                    status="rejected",
                )
                return


# ---------------------------------------------------------------------------
# PolicyExporter
# ---------------------------------------------------------------------------


@dataclass
class PolicyExporter:
    """Bridges live MANIFOLD state → ``ManifoldPolicy`` → YAML.

    Parameters
    ----------
    discovery:
        ``AutoRuleDiscovery`` instance with accumulated observations.
    config:
        ``InterceptorConfig`` to embed as global defaults.
    version:
        Policy version string.

    Example
    -------
    ::

        exporter = PolicyExporter(discovery=auto_discovery, config=cfg)
        policy = exporter.export(domains=["finance"])
        print(policy.to_yaml())
    """

    discovery: AutoRuleDiscovery
    config: InterceptorConfig = field(default_factory=InterceptorConfig)
    version: str = "1.0.0"

    def export(
        self,
        domains: list[str] | None = None,
        description: str = "Auto-exported MANIFOLD policy",
    ) -> ManifoldPolicy:
        """Export the current runtime state as a ``ManifoldPolicy``.

        Parameters
        ----------
        domains:
            List of domain names to include.  Each known domain from
            ``PolicySynthesizer`` is included automatically; additionally
            any domains listed here will be included (using template defaults
            if no synthesised data is available).
        description:
            Policy description string.

        Returns
        -------
        ManifoldPolicy
        """
        all_domains: set[str] = set(self.discovery.synthesizer.known_domains())
        if domains:
            all_domains.update(domains)
        if not all_domains:
            all_domains.add("general")

        policy_domains: list[PolicyDomain] = []
        for domain_name in sorted(all_domains):
            tpl = DOMAIN_TEMPLATES.get(domain_name, DOMAIN_TEMPLATES["general"]).copy()
            tpl["name"] = domain_name

            # Incorporate synthesised templates if available
            templates = self.discovery.suggest_policy_templates(domain_name)
            if templates:
                best = templates[0]
                tpl["coordination_tax_cap"] = min(
                    round(best.avg_coordination_tax + 0.05, 4),
                    tpl.get("coordination_tax_cap", 0.20),  # type: ignore[arg-type]
                )
                tpl["notes"] = (
                    f"Auto-derived from {best.occurrence_count} decompositions. "
                    + tpl.get("notes", "")  # type: ignore[arg-type]
                )

            policy_domains.append(PolicyDomain.from_dict(tpl))

        # Convert PenaltyProposals → RuleDiffs
        proposals = self.discovery.suggest_penalty_updates()
        diffs = _proposals_to_diffs(proposals, domain="general")

        return ManifoldPolicy(
            version=self.version,
            description=description,
            domains=policy_domains,
            global_veto_threshold=self.config.risk_veto_threshold,
            global_fallback_strategy=self.config.redirect_strategy,
            pending_diffs=diffs,
        )


# ---------------------------------------------------------------------------
# PolicyLoader
# ---------------------------------------------------------------------------


class PolicyLoader:
    """Deserialises a ``ManifoldPolicy`` from YAML or JSON.

    Example
    -------
    ::

        policy = PolicyLoader.from_yaml(open("manifold.yaml").read())
        policy = PolicyLoader.from_json(open("manifold.json").read())
    """

    @staticmethod
    def from_yaml(yaml_str: str) -> ManifoldPolicy:
        """Parse a YAML string produced by ``ManifoldPolicy.to_yaml()``.

        Uses a hand-written parser compatible with the subset of YAML
        produced by ``to_yaml()``.  No PyYAML dependency required.

        Parameters
        ----------
        yaml_str:
            YAML document string.

        Returns
        -------
        ManifoldPolicy
        """
        data = _yaml_to_dict(yaml_str)
        return ManifoldPolicy.from_dict(data)

    @staticmethod
    def from_json(json_str: str) -> ManifoldPolicy:
        """Parse a JSON string produced by ``ManifoldPolicy.to_json()``."""
        return ManifoldPolicy.from_json(json_str)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ManifoldPolicy:
        """Load from a plain dict."""
        return ManifoldPolicy.from_dict(data)


# ---------------------------------------------------------------------------
# Internal YAML helpers (zero-dependency)
# ---------------------------------------------------------------------------


def _dict_to_yaml(obj: Any, indent: int = 0) -> str:  # noqa: ANN001
    """Minimal dict → YAML serialiser (no PyYAML required).

    Handles: dict, list, str, int, float, bool, None.
    Produces block-style YAML compatible with PyYAML's safe_load.
    """
    pad = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return "{}\n"
        lines: list[str] = []
        for k, v in obj.items():
            if isinstance(v, list) and not v:
                lines.append(f"{pad}{k}: []")
            elif isinstance(v, dict) and not v:
                lines.append(f"{pad}{k}: {{}}")
            elif isinstance(v, (dict, list)):
                lines.append(f"{pad}{k}:")
                lines.append(_dict_to_yaml(v, indent + 1).rstrip("\n"))
            else:
                lines.append(f"{pad}{k}: {_scalar_yaml(v)}")
        return "\n".join(lines) + "\n"
    if isinstance(obj, list):
        if not obj:
            return f"{pad}[]\n"
        lines = []
        for item in obj:
            if isinstance(item, dict):
                first = True
                for k, v in item.items():
                    prefix = f"{pad}- " if first else f"{pad}  "
                    first = False
                    if isinstance(v, list) and not v:
                        lines.append(f"{prefix}{k}: []")
                    elif isinstance(v, dict) and not v:
                        lines.append(f"{prefix}{k}: {{}}")
                    elif isinstance(v, (dict, list)):
                        lines.append(f"{prefix}{k}:")
                        lines.append(_dict_to_yaml(v, indent + 2).rstrip("\n"))
                    else:
                        lines.append(f"{prefix}{k}: {_scalar_yaml(v)}")
            else:
                lines.append(f"{pad}- {_scalar_yaml(item)}")
        return "\n".join(lines) + "\n"
    return f"{pad}{_scalar_yaml(obj)}\n"


def _scalar_yaml(v: Any) -> str:  # noqa: ANN001
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v) if isinstance(v, float) else str(v)
    # String — quote if contains special chars
    s = str(v)
    if any(c in s for c in ":#{}[]|>&*!,\n\"'") or s.strip() != s or s == "":
        # Use double-quote style with minimal escaping
        escaped = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    return s


def _yaml_to_dict(yaml_str: str) -> dict[str, Any]:
    """Minimal YAML → dict parser for the subset produced by ``_dict_to_yaml``.

    Supports: top-level dict, nested dicts, lists of dicts, scalar values.
    Uses JSON for complex nested structures when PyYAML is unavailable.
    """
    # Try PyYAML first (available in many environments)
    try:
        import yaml  # type: ignore[import-untyped]
        result = yaml.safe_load(yaml_str)
        return result if isinstance(result, dict) else {}
    except ImportError:
        pass

    # Fallback: parse the simplified block-YAML produced by _dict_to_yaml
    return _parse_block_yaml(yaml_str)


def _parse_block_yaml(yaml_str: str) -> dict[str, Any]:  # noqa: C901
    """Very small subset block-YAML parser — handles what _dict_to_yaml emits."""
    lines = yaml_str.splitlines()
    # Remove empty lines, build (indent, content) pairs
    parsed: list[tuple[int, str]] = []
    for raw in lines:
        stripped = raw.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(stripped)
        parsed.append((indent, stripped))

    def parse_value(s: str) -> Any:
        s = s.strip()
        if s in ("null", "~", ""):
            return None
        if s == "true":
            return True
        if s == "false":
            return False
        # Quoted string
        if (s.startswith('"') and s.endswith('"')) or (
            s.startswith("'") and s.endswith("'")
        ):
            return s[1:-1].replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")
        # Number
        try:
            if "." in s:
                return float(s)
            return int(s)
        except ValueError:
            return s

    def parse_block(
        items: list[tuple[int, str]], start: int, base_indent: int
    ) -> tuple[Any, int]:
        """Parse a block starting at *start* with *base_indent*."""
        if start >= len(items):
            return {}, start

        _, first_line = items[start]

        # List block?
        if first_line.startswith("- "):
            result_list: list[Any] = []
            i = start
            while i < len(items):
                ind, line = items[i]
                if ind < base_indent:
                    break
                if not line.startswith("- "):
                    break
                item_content = line[2:].strip()
                if ":" in item_content and not item_content.startswith('"'):
                    # Dict item starting on same line
                    d: dict[str, Any] = {}
                    k, _, v = item_content.partition(":")
                    k = k.strip()
                    v = v.strip()
                    if v:
                        d[k] = parse_value(v)
                    else:
                        # Sub-dict
                        sub, i = parse_block(items, i + 1, ind + 2)
                        d[k] = sub
                        result_list.append(d)
                        continue
                    # Collect remaining k:v at same indent+2
                    j = i + 1
                    item_indent = ind + 2
                    while j < len(items):
                        jind, jline = items[j]
                        if jind < item_indent:
                            break
                        if ":" in jline and not jline.startswith("- "):
                            jk, _, jv = jline.partition(":")
                            jk = jk.strip()
                            jv = jv.strip()
                            if jv:
                                d[jk] = parse_value(jv)
                            j += 1
                        else:
                            break
                    i = j
                    result_list.append(d)
                else:
                    result_list.append(parse_value(item_content))
                    i += 1
            return result_list, i

        # Dict block
        result_dict: dict[str, Any] = {}
        i = start
        while i < len(items):
            ind, line = items[i]
            if ind < base_indent:
                break
            if line.startswith("- "):
                break
            if ":" not in line:
                i += 1
                continue
            k, _, v = line.partition(":")
            k = k.strip()
            v = v.strip()
            if v:
                result_dict[k] = parse_value(v)
                i += 1
            else:
                # Next lines are the value (nested block)
                i += 1
                if i < len(items):
                    child_indent = items[i][0]
                    child, i = parse_block(items, i, child_indent)
                    result_dict[k] = child
                else:
                    result_dict[k] = None
        return result_dict, i

    if not parsed:
        return {}
    root, _ = parse_block(parsed, 0, 0)
    return root if isinstance(root, dict) else {}


# ---------------------------------------------------------------------------
# Helper: PenaltyProposal → RuleDiff
# ---------------------------------------------------------------------------


def _proposals_to_diffs(
    proposals: list[PenaltyProposal], domain: str = "general"
) -> list[RuleDiff]:
    """Convert ``PenaltyProposal`` list to ``RuleDiff`` list."""
    diffs: list[RuleDiff] = []
    for p in proposals:
        # Map penalty increase → tighten risk_tolerance
        # Map penalty decrease → relax risk_tolerance
        direction = -1.0 if p.delta > 0 else 1.0
        current_rt = 0.45  # default fallback
        proposed_rt = max(0.05, min(0.95, current_rt + direction * abs(p.delta) * 0.05))
        diffs.append(
            RuleDiff(
                rule_name=p.rule_name,
                trigger=p.trigger,
                domain=domain,
                field="risk_tolerance",
                old_value=round(current_rt, 4),
                new_value=round(proposed_rt, 4),
                rationale=p.rationale,
                confidence=p.confidence,
                status="pending",
            )
        )
    return diffs
