"""Phase 4: PromptEncoder — maps raw prompt text to BrainTask features.

The encoder bridges the gap between unstructured natural language prompts
and the structured ``BrainTask`` feature vector consumed by ``ManifoldBrain``.

Architecture
------------
Two-stage pipeline:

1. **Keyword extractor** — rule-based patterns map surface cues in the prompt
   to initial feature estimates (complexity, stakes, uncertainty,
   source_confidence).  This stage needs *no training data* and works out
   of the box for any English prompt.

2. **EMA correction layer** — thin delta corrections, one per feature per
   ``domain``, that are updated whenever a ``PriceAdapter`` or
   ``AssetAdapter`` signals that a prediction was wrong.  As real traffic
   accumulates the extractor self-calibrates: if prompts in a given domain
   consistently turn out to be more expensive than estimated, the encoder
   learns to raise its complexity estimate for that domain.

This is the meta-learning loop described in the Phase 4 roadmap: the encoder
learns to predict the economics that MANIFOLD will later discover, without
requiring human feature engineering or a large ML model.

Key classes
-----------
``EncoderCorrection``
    Mutable EMA state for a single (domain, feature) pair.
``PromptFeatures``
    Frozen snapshot of encoder output for one prompt.
``PromptEncoder``
    Main entry point: ``encode(prompt, domain)`` and ``update_from_*`` methods.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .brain import BrainTask
from .trustrouter import clamp01


# ---------------------------------------------------------------------------
# Keyword signal tables
# ---------------------------------------------------------------------------

# Each entry: (regex_pattern, weight) where weight ∈ [-1, 1].
# Matched weights are summed and scaled to [0, 1] via _sum_to_signal().

_COMPLEXITY_SIGNALS: list[tuple[str, float]] = [
    (r"\b(comprehensive|exhaustive|thorough|detailed|in-depth)\b", 0.30),
    (r"\b(analyze|analyse|evaluate|compare|contrast|synthesize)\b", 0.25),
    (r"\b(report|research|survey|study|investigation)\b", 0.20),
    (r"\b(step[- ]by[- ]step|walkthrough|breakdown|explain)\b", 0.15),
    (r"\b(simple|quick|brief|short|basic|easy)\b", -0.30),
    (r"\b(yes[/ ]no|binary|one[- ]word|just tell me)\b", -0.25),
    (r"\b(what is|who is|when is|where is)\b", -0.20),
    (r"\b(how to|how do i|can you|please)\b", 0.05),
    (r"\b(multi[- ]?step|pipeline|workflow|architecture)\b", 0.25),
    (r"\b(write a|build a|create a|design a)\b", 0.10),
]

_STAKES_SIGNALS: list[tuple[str, float]] = [
    (r"\b(critical|urgent|production|legal|medical|financial|compliance)\b", 0.35),
    (r"\b(customer|client|user|public|stakeholder)\b", 0.20),
    (r"\b(revenue|profit|cost|budget|payment|invoice|contract)\b", 0.25),
    (r"\b(security|vulnerability|breach|exploit|credential)\b", 0.30),
    (r"\b(test|experiment|prototype|draft|demo|sandbox|toy)\b", -0.25),
    (r"\b(internal|personal|informal|casual|fun)\b", -0.20),
    (r"\b(outage|incident|failure|disaster|emergency)\b", 0.40),
    (r"\b(example|sample|practice|tutorial|learning)\b", -0.15),
]

_UNCERTAINTY_SIGNALS: list[tuple[str, float]] = [
    (r"\b(maybe|might|could|possibly|perhaps|not sure|uncertain)\b", 0.30),
    (r"\b(estimate|guess|approximate|rough|ballpark)\b", 0.25),
    (r"\b(unclear|ambiguous|vague|confused|unsure|don't know)\b", 0.35),
    (r"\b(multiple|various|several|options|alternatives|tradeoffs)\b", 0.15),
    (r"\b(definitely|certainly|exactly|precisely|confirmed|known)\b", -0.30),
    (r"\b(always|never|constant|fixed|guaranteed)\b", -0.25),
    (r"\b(random|stochastic|probabilistic|noisy)\b", 0.20),
]

_SOURCE_CONFIDENCE_SIGNALS: list[tuple[str, float]] = [
    # High-confidence sources raise source_confidence
    (r"\b(official|documented|spec|specification|standard|rfc)\b", 0.30),
    (r"\b(verified|confirmed|tested|proven|cited|referenced)\b", 0.25),
    (r"\b(database|structured|tabular|csv|json)\b", 0.20),
    # Low-confidence signals lower it
    (r"\b(rumor|hearsay|forum|reddit|twitter|blog|anecdotal)\b", -0.30),
    (r"\b(old|outdated|deprecated|legacy|obsolete)\b", -0.20),
    (r"\b(unknown|unverified|unconfirmed|speculative)\b", -0.25),
    (r"\b(user report|said|claimed|mentioned|heard)\b", -0.10),
]

# Tool-relevance signals — how likely is a tool needed?
_TOOL_RELEVANCE_SIGNALS: list[tuple[str, float]] = [
    (r"\b(search|find|look up|retrieve|fetch|query|lookup)\b", 0.35),
    (r"\b(calculate|compute|run|execute|code|script|api)\b", 0.40),
    (r"\b(translate|convert|format|parse|encode|decode)\b", 0.30),
    (r"\b(email|send|post|submit|upload|download)\b", 0.25),
    (r"\b(summarize|extract|classify|detect)\b", 0.20),
    (r"\b(think|consider|what do you|opinion|recommend)\b", -0.15),
    (r"\b(explain|describe|define|tell me)\b", -0.10),
]


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def _sum_to_signal(
    text: str,
    signals: list[tuple[str, float]],
    baseline: float = 0.5,
) -> float:
    """Sum regex pattern weights from *text* and map result to [0, 1].

    Parameters
    ----------
    text:
        Lower-cased prompt text.
    signals:
        List of ``(pattern, weight)`` tuples.  Patterns are compiled on demand
        (Python caches ``re.search`` under the hood for simple patterns).
    baseline:
        Starting value before any signals are applied.  Defaults to 0.5.
    """
    score = 0.0
    for pattern, weight in signals:
        if re.search(pattern, text, re.IGNORECASE):
            score += weight
    # Sigmoid-like mapping: baseline ± score/2, clamped to [0, 1]
    return clamp01(baseline + score * 0.5)


def _length_complexity(text: str) -> float:
    """Estimate complexity contribution from prompt word count.

    Very short prompts (<5 words) → low complexity bonus.
    Long prompts (>50 words) → high complexity bonus.
    """
    words = len(text.split())
    if words <= 5:
        return -0.10
    if words <= 15:
        return 0.0
    if words <= 40:
        return 0.05
    if words <= 80:
        return 0.10
    return 0.15


# ---------------------------------------------------------------------------
# Encoder data structures
# ---------------------------------------------------------------------------


@dataclass
class EncoderCorrection:
    """EMA state for a single (domain, feature) correction pair.

    Attributes
    ----------
    delta:
        Exponential moving average of (realized − estimated) for this feature.
        Positive values mean the encoder consistently underestimates.
    n_updates:
        Number of EMA updates applied so far.
    """

    delta: float = 0.0
    n_updates: int = 0


@dataclass(frozen=True)
class PromptFeatures:
    """The feature vector produced by ``PromptEncoder.encode``.

    All values are in [0, 1] except ``prompt`` which is the original string.

    Attributes
    ----------
    prompt:
        Original prompt text (preserved for logging / debugging).
    domain:
        Domain string passed to the encoder.
    complexity:
        Estimated task complexity.
    stakes:
        Estimated task stakes.
    uncertainty:
        Estimated task uncertainty.
    source_confidence:
        Estimated source confidence.
    tool_relevance:
        Estimated tool relevance.
    raw_complexity:
        Pre-correction complexity (useful for diagnosing encoder drift).
    raw_stakes:
        Pre-correction stakes.
    """

    prompt: str
    domain: str
    complexity: float
    stakes: float
    uncertainty: float
    source_confidence: float
    tool_relevance: float
    raw_complexity: float
    raw_stakes: float

    def to_brain_task(self, **overrides: object) -> BrainTask:
        """Convert to a ``BrainTask``, optionally overriding specific fields.

        Parameters
        ----------
        **overrides:
            Any ``BrainTask`` field to override.  Useful for injecting
            ``time_pressure``, ``safety_sensitivity``, ``user_patience``, etc.
            that the encoder doesn't currently infer from text.

        Example
        -------
        ::

            features = encoder.encode("search EU compliance rules", "legal")
            task = features.to_brain_task(time_pressure=0.7, safety_sensitivity=0.9)
        """
        kwargs: dict[str, object] = {
            "prompt": self.prompt,
            "domain": self.domain,
            "complexity": self.complexity,
            "stakes": self.stakes,
            "uncertainty": self.uncertainty,
            "source_confidence": self.source_confidence,
            "tool_relevance": self.tool_relevance,
        }
        kwargs.update(overrides)
        return BrainTask(**kwargs)  # type: ignore[arg-type]


@dataclass
class PromptEncoder:
    """Lightweight prompt-to-features encoder with EMA self-calibration.

    ``PromptEncoder`` does not require a trained model or external dependencies.
    It uses keyword signals for initial feature estimation and an EMA correction
    layer that updates whenever the MANIFOLD ecosystem discovers that its
    predictions were wrong (via ``PriceAdapter`` or ``AssetAdapter`` deltas).

    Parameters
    ----------
    lr:
        EMA learning rate for domain-level corrections.  Defaults to 0.12.
    complexity_baseline:
        Initial prior for complexity before any signals are applied.
    stakes_baseline:
        Initial prior for stakes.
    uncertainty_baseline:
        Initial prior for uncertainty.
    source_confidence_baseline:
        Initial prior for source confidence.
    tool_relevance_baseline:
        Initial prior for tool relevance.

    Example
    -------
    ::

        encoder = PromptEncoder()
        features = encoder.encode("search EU compliance rules", domain="legal")
        task = features.to_brain_task(time_pressure=0.6)
        brain.decide(task)

        # After outcome:
        encoder.update_from_price_delta("legal", cost_delta=0.18, risk_delta=0.05)
        # → complexity and risk estimates for "legal" domain rise.
    """

    lr: float = 0.12
    complexity_baseline: float = 0.50
    stakes_baseline: float = 0.40
    uncertainty_baseline: float = 0.45
    source_confidence_baseline: float = 0.65
    tool_relevance_baseline: float = 0.40

    _corrections: dict[str, dict[str, EncoderCorrection]] = field(
        default_factory=dict, init=False, repr=False
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, prompt: str, domain: str = "general") -> PromptFeatures:
        """Extract features from *prompt* and apply domain-level corrections.

        Parameters
        ----------
        prompt:
            Raw natural-language prompt from the user.
        domain:
            Task domain for applying domain-specific EMA corrections.

        Returns
        -------
        PromptFeatures
            Feature vector with corrected and raw values.
        """
        text = prompt.lower()

        raw_complexity = clamp01(
            _sum_to_signal(text, _COMPLEXITY_SIGNALS, self.complexity_baseline)
            + _length_complexity(text)
        )
        raw_stakes = _sum_to_signal(text, _STAKES_SIGNALS, self.stakes_baseline)
        uncertainty = _sum_to_signal(text, _UNCERTAINTY_SIGNALS, self.uncertainty_baseline)
        source_confidence = _sum_to_signal(text, _SOURCE_CONFIDENCE_SIGNALS, self.source_confidence_baseline)
        tool_relevance = _sum_to_signal(text, _TOOL_RELEVANCE_SIGNALS, self.tool_relevance_baseline)

        # Apply EMA corrections from prior domain experience
        dc = self._get_correction(domain, "complexity")
        ds = self._get_correction(domain, "stakes")
        du = self._get_correction(domain, "uncertainty")
        dsc = self._get_correction(domain, "source_confidence")
        dtr = self._get_correction(domain, "tool_relevance")

        return PromptFeatures(
            prompt=prompt,
            domain=domain,
            complexity=clamp01(raw_complexity + dc.delta),
            stakes=clamp01(raw_stakes + ds.delta),
            uncertainty=clamp01(uncertainty + du.delta),
            source_confidence=clamp01(source_confidence + dsc.delta),
            tool_relevance=clamp01(tool_relevance + dtr.delta),
            raw_complexity=raw_complexity,
            raw_stakes=raw_stakes,
        )

    def update_from_price_delta(
        self,
        domain: str,
        cost_delta: float,
        risk_delta: float = 0.0,
    ) -> None:
        """Update complexity and risk corrections from a ``PriceAdapter`` signal.

        Call this after a ``BrainOutcome`` is observed and ``PriceAdapter``
        has computed the cost/risk gaps.  The encoder infers: if MANIFOLD
        paid *more* than expected, the prompt was probably *more complex* than
        estimated; if the tool failed with higher risk than expected, the
        uncertainty estimate should be raised.

        Parameters
        ----------
        domain:
            The task domain for this observation.
        cost_delta:
            ``PriceAdapter.price_corrections()[tool].cost_delta`` — positive
            means the tool cost more than stated → raise complexity.
        risk_delta:
            ``PriceAdapter.price_corrections()[tool].risk_delta`` — positive
            means more risk than stated → raise uncertainty.
        """
        self._ema_update(domain, "complexity", cost_delta * 0.5)
        if risk_delta:
            self._ema_update(domain, "uncertainty", risk_delta * 0.5)

    def update_from_asset_delta(
        self,
        domain: str,
        action: str,
        asset_delta: float,
    ) -> None:
        """Update stakes correction from an ``AssetAdapter`` signal.

        Call this after an asset gap is observed.  If users consistently
        reject outcomes for a given domain (asset_delta < 0), the encoder
        infers that stakes were higher than estimated.

        Parameters
        ----------
        domain:
            The task domain.
        action:
            The ``BrainAction`` string (unused currently; reserved for
            action-specific encoder fine-tuning in Phase 5).
        asset_delta:
            ``AssetAdapter.asset_corrections()[action].asset_delta`` — negative
            means users were more dissatisfied than expected → raise stakes.
        """
        # Negative asset_delta → stakes were likely underestimated.
        self._ema_update(domain, "stakes", -asset_delta * 0.5)

    def corrections(self) -> dict[str, dict[str, EncoderCorrection]]:
        """Return a copy of all EMA corrections keyed by domain → feature.

        Useful for inspecting what the encoder has learned about each domain.
        """
        return {d: dict(features) for d, features in self._corrections.items()}

    def reset_domain(self, domain: str) -> None:
        """Discard all learned corrections for *domain*.

        Useful when a domain's characteristics change and accumulated history
        is no longer representative.
        """
        self._corrections.pop(domain, None)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_correction(self, domain: str, feature: str) -> EncoderCorrection:
        return self._corrections.get(domain, {}).get(feature, EncoderCorrection())

    def _ema_update(self, domain: str, feature: str, gap: float) -> None:
        domain_map = self._corrections.setdefault(domain, {})
        corr = domain_map.setdefault(feature, EncoderCorrection())
        corr.delta = corr.delta * (1.0 - self.lr) + gap * self.lr
        corr.n_updates += 1
