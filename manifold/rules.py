"""Rule compiler and adaptive rule definitions for MANIFOLD."""

from __future__ import annotations

from dataclasses import dataclass
import re


RULE_RE = re.compile(
    r"^\s*if\s+(?P<rule>[a-zA-Z_][a-zA-Z0-9_]*)\s+then\s+"
    r"(?P<sign>[+-])?\s*£?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<rest>.*)$"
)
KV_RE = re.compile(r"@(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)=(?P<value>[^\s]+)")


@dataclass(frozen=True)
class RuleDefinition:
    """One rule with adaptive tuning parameters."""

    name: str
    penalty: float
    target_rate: float = 0.20
    alpha: float = 1.0
    min_penalty: float = 0.5
    max_penalty: float = 40.0


def _as_float(raw: str, *, key: str, line: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid @{key} value in rule line: {line!r}") from exc


def compile_rulebook(text: str) -> tuple[RuleDefinition, ...]:
    """Compile DSL text into RuleDefinition objects.

    Supported line format:
        if late_delivery then -£8.20 @target=0.15 @alpha=1.2

    Notes:
    - A leading '-' or '+' before currency is accepted.
    - Penalty is always stored as a positive magnitude.
    - Empty lines and lines starting with '#' are ignored.
    """

    rules: list[RuleDefinition] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = RULE_RE.match(line)
        if not match:
            raise ValueError(
                "Rule line does not match expected format "
                f"'if <rule> then -£<value> ...': {line!r}"
            )

        name = match.group("rule")
        penalty = abs(float(match.group("value")))
        opts: dict[str, str] = {}
        for kv in KV_RE.finditer(match.group("rest")):
            opts[kv.group("key").lower()] = kv.group("value")

        target_rate = _as_float(opts["target"], key="target", line=line) if "target" in opts else 0.20
        alpha = _as_float(opts["alpha"], key="alpha", line=line) if "alpha" in opts else 1.0
        min_penalty = _as_float(opts["min"], key="min", line=line) if "min" in opts else 0.5
        max_penalty = _as_float(opts["max"], key="max", line=line) if "max" in opts else 40.0

        rules.append(
            RuleDefinition(
                name=name,
                penalty=penalty,
                target_rate=target_rate,
                alpha=alpha,
                min_penalty=min_penalty,
                max_penalty=max_penalty,
            )
        )

    if not rules:
        raise ValueError("Rulebook is empty after parsing.")
    return tuple(rules)
