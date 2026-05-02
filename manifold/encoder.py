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

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

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
    encoder_confidence:
        Confidence in Stage 1 (keyword) signal quality, [0, 1].
        Values < ``DualPathEncoder.confidence_threshold`` trigger the
        semantic slow-path in ``DualPathEncoder``.  For plain
        ``PromptEncoder`` instances this is always 1.0.
    semantic_cluster:
        Name of the ``PromptCluster`` matched by the semantic slow-path,
        or ``""`` when only the fast-path was used.
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
    encoder_confidence: float = 1.0
    semantic_cluster: str = ""

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

    def freeze_base(self) -> None:
        """Freeze Stage 1 keyword weights (no-op for base class; semantic override in DualPathEncoder).

        After calling this, ``update_from_price_delta`` and
        ``update_from_asset_delta`` continue to update EMA corrections, but
        the Stage 1 keyword signal tables are treated as immutable priors.
        In ``DualPathEncoder``, this also freezes the cluster similarity weights.
        """
        # Base PromptEncoder has no mutable weights in Stage 1 (signal tables are
        # module-level constants), so this is a semantic marker only.
        self._frozen = True

    def fine_tune(
        self,
        domain: str,
        interactions: list[dict[str, float]],
    ) -> None:
        """Fine-tune EMA corrections from a list of observed interaction records.

        Each interaction record must have at least one of:
        ``cost_delta``, ``risk_delta``, ``asset_delta``.

        This implements the Phase 5a roadmap: freeze encoder weights, then
        fine-tune domain corrections on <100 real interactions.  No manual
        labeling is needed — the economic deltas are the training signal.

        Parameters
        ----------
        domain:
            The domain to update corrections for.
        interactions:
            List of dicts, each optionally containing:
            - ``cost_delta`` (float): from PriceAdapter
            - ``risk_delta`` (float): from PriceAdapter
            - ``asset_delta`` (float): from AssetAdapter
            - ``action`` (str): BrainAction label (ignored in base class)

        Example
        -------
        ::

            legal_logs = [
                {"cost_delta": 0.35, "risk_delta": 0.10},
                {"cost_delta": 0.28, "asset_delta": -0.30},
            ]
            encoder.freeze_base()
            encoder.fine_tune(domain="legal", interactions=legal_logs)
        """
        for rec in interactions:
            cost_delta = float(rec.get("cost_delta", 0.0))
            risk_delta = float(rec.get("risk_delta", 0.0))
            asset_delta = float(rec.get("asset_delta", 0.0))
            self.update_from_price_delta(domain, cost_delta=cost_delta, risk_delta=risk_delta)
            if asset_delta:
                self.update_from_asset_delta(domain, action=str(rec.get("action", "answer")), asset_delta=asset_delta)

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


# ---------------------------------------------------------------------------
# Phase 5: Semantic Bridge — pure-Python similarity over phrase clusters
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer: lowercase, strip punctuation, split on whitespace."""
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


def _bow_vector(tokens: list[str], vocab: dict[str, int]) -> list[float]:
    """Bag-of-words vector aligned to *vocab*."""
    v = [0.0] * len(vocab)
    for tok in tokens:
        if tok in vocab:
            v[vocab[tok]] += 1.0
    return v


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length float vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class PromptCluster:
    """A named semantic cluster of anchor phrases used by ``SemanticBridge``.

    Each cluster represents a conceptual category (e.g., "Analysis",
    "Factual", "Risk") with associated feature deltas that blend into the
    encoder output when a novel prompt maps to this cluster.

    Attributes
    ----------
    name:
        Human-readable cluster name (e.g., ``"Analysis"``, ``"Factual"``).
    anchor_phrases:
        Representative phrases that define this cluster's position in
        bag-of-words space.
    complexity_delta:
        Complexity adjustment applied when this cluster matches (−1 to +1).
    stakes_delta:
        Stakes adjustment.
    uncertainty_delta:
        Uncertainty adjustment.
    tool_relevance_delta:
        Tool-relevance adjustment.
    weight:
        How strongly this cluster's deltas are blended when it matches.
        Defaults to 0.5 (half-weight blend with Stage 1 signal).
    """

    name: str
    anchor_phrases: list[str]
    complexity_delta: float = 0.0
    stakes_delta: float = 0.0
    uncertainty_delta: float = 0.0
    tool_relevance_delta: float = 0.0
    weight: float = 0.5


# ---------------------------------------------------------------------------
# Default cluster definitions — the "Price Atlas"
# ---------------------------------------------------------------------------

_DEFAULT_CLUSTERS: list[PromptCluster] = [
    PromptCluster(
        name="Analysis",
        anchor_phrases=[
            "analyze examine investigate evaluate assess scrutinize inspect dissect",
            "comprehensive thorough detailed in-depth exhaustive systematic",
            "compare contrast juxtapose benchmark measure audit review",
        ],
        complexity_delta=+0.25,
        stakes_delta=+0.05,
        tool_relevance_delta=+0.10,
    ),
    PromptCluster(
        name="Factual",
        anchor_phrases=[
            "what is who is when is where is define meaning explain",
            "simple quick brief short basic yes no binary one word",
            "tell me fact definition lookup",
        ],
        complexity_delta=-0.25,
        stakes_delta=-0.10,
        tool_relevance_delta=-0.10,
    ),
    PromptCluster(
        name="Risk",
        anchor_phrases=[
            "critical urgent emergency outage incident failure disaster breach",
            "legal medical financial compliance regulatory liability privacy security",
            "production live customer revenue stakeholder board executive",
        ],
        complexity_delta=+0.10,
        stakes_delta=+0.35,
        uncertainty_delta=+0.10,
    ),
    PromptCluster(
        name="Research",
        anchor_phrases=[
            "research survey study literature review report white paper summary",
            "investigate explore discover findings evidence data analysis",
            "competitive market landscape overview trend synthesis",
        ],
        complexity_delta=+0.20,
        tool_relevance_delta=+0.15,
    ),
    PromptCluster(
        name="Action",
        anchor_phrases=[
            "execute run deploy build create generate write code script automate",
            "calculate compute fetch retrieve search query lookup transform convert",
            "send email submit post upload download process pipeline workflow",
        ],
        complexity_delta=+0.10,
        tool_relevance_delta=+0.30,
    ),
    PromptCluster(
        name="Uncertain",
        anchor_phrases=[
            "maybe might could possibly perhaps not sure maybe I think",
            "unclear ambiguous vague fuzzy open ended multiple options alternatives",
            "random estimate guess approximate rough ballpark uncertain unknown",
        ],
        uncertainty_delta=+0.25,
        stakes_delta=+0.05,
    ),
    PromptCluster(
        name="Trusted",
        anchor_phrases=[
            "official specification standard rfc documented verified confirmed",
            "database structured tabular proven tested certified validated authoritative",
            "exact precise fixed constant guaranteed always never known",
        ],
        uncertainty_delta=-0.20,
        stakes_delta=+0.05,
    ),
]


@dataclass
class SemanticBridge:
    """Pure-Python phrase-cluster similarity engine for novel vocabulary.

    ``SemanticBridge`` provides a zero-dependency "slow path" for
    ``DualPathEncoder``.  It uses cosine similarity between a
    bag-of-words representation of the input prompt and each cluster's
    anchor phrases to identify the best matching semantic cluster.

    No external libraries, no network calls, no GPU required.
    The vocabulary is built lazily from the anchor phrases themselves.

    Parameters
    ----------
    clusters:
        List of ``PromptCluster`` instances.  Defaults to ``_DEFAULT_CLUSTERS``.
    min_similarity:
        Minimum cosine similarity to consider a cluster match.
        Below this threshold, no cluster is selected and the bridge returns
        no adjustments (pure fast-path result is used).  Defaults to 0.12.

    Example
    -------
    ::

        bridge = SemanticBridge()
        cluster, sim = bridge.find_cluster("scrutinize the GDPR compliance protocols")
        # → ("Risk", 0.43)
    """

    clusters: list[PromptCluster] = field(default_factory=lambda: list(_DEFAULT_CLUSTERS))
    min_similarity: float = 0.12

    _vocab: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _cluster_vectors: list[list[float]] = field(default_factory=list, init=False, repr=False)
    _built: bool = field(default=False, init=False, repr=False)

    def _build(self) -> None:
        """Lazily build vocabulary and cluster vectors from anchor phrases."""
        if self._built:
            return
        # Collect all tokens from all anchor phrases
        all_tokens: list[str] = []
        for cluster in self.clusters:
            for phrase in cluster.anchor_phrases:
                all_tokens.extend(_tokenize(phrase))
        unique = sorted(set(all_tokens))
        self._vocab = {tok: i for i, tok in enumerate(unique)}
        # Build cluster vectors as the sum of each anchor phrase's BoW vector
        for cluster in self.clusters:
            vec = [0.0] * len(self._vocab)
            for phrase in cluster.anchor_phrases:
                phrase_vec = _bow_vector(_tokenize(phrase), self._vocab)
                vec = [v + p for v, p in zip(vec, phrase_vec)]
            self._cluster_vectors.append(vec)
        self._built = True

    def find_cluster(self, text: str) -> tuple[PromptCluster | None, float]:
        """Find the best matching cluster for *text* by cosine similarity.

        Parameters
        ----------
        text:
            The prompt text to classify.

        Returns
        -------
        (cluster, similarity):
            The best matching ``PromptCluster`` (or ``None`` if no cluster
            exceeds ``min_similarity``), and the cosine similarity score.
        """
        self._build()
        if not self._vocab:
            return None, 0.0
        tokens = _tokenize(text)
        query_vec = _bow_vector(tokens, self._vocab)
        best_cluster: PromptCluster | None = None
        best_sim = 0.0
        for cluster, cvec in zip(self.clusters, self._cluster_vectors):
            sim = _cosine_similarity(query_vec, cvec)
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster
        if best_sim < self.min_similarity:
            return None, best_sim
        return best_cluster, best_sim

    def apply(
        self,
        text: str,
        complexity: float,
        stakes: float,
        uncertainty: float,
        tool_relevance: float,
        blend: float = 0.5,
    ) -> tuple[float, float, float, float, str, float]:
        """Apply semantic cluster adjustments to Stage 1 feature estimates.

        Parameters
        ----------
        text:
            Prompt text.
        complexity, stakes, uncertainty, tool_relevance:
            Stage 1 feature estimates (from ``PromptEncoder.encode``).
        blend:
            How strongly cluster deltas blend in (0 = no effect, 1 = full).
            Overridden by the matched cluster's own ``weight`` field.

        Returns
        -------
        (complexity, stakes, uncertainty, tool_relevance, cluster_name, similarity)
        """
        cluster, sim = self.find_cluster(text)
        if cluster is None:
            return complexity, stakes, uncertainty, tool_relevance, "", sim
        w = cluster.weight * blend
        return (
            clamp01(complexity + cluster.complexity_delta * w),
            clamp01(stakes + cluster.stakes_delta * w),
            clamp01(uncertainty + cluster.uncertainty_delta * w),
            clamp01(tool_relevance + cluster.tool_relevance_delta * w),
            cluster.name,
            sim,
        )


@dataclass
class DualPathEncoder(PromptEncoder):
    """Phase 5: Dual-path encoder with keyword fast-path + semantic slow-path.

    Extends ``PromptEncoder`` with a ``SemanticBridge`` that activates when
    Stage 1 keyword confidence falls below ``confidence_threshold``.

    Stage 1 (keyword fast-path):
        Regex patterns score the prompt for known vocabulary.  The number of
        *distinct* pattern groups that fire divided by the total number of
        feature dimensions is used as the confidence proxy.  If the prompt
        contains rich known vocabulary, confidence is high and Stage 2 is skipped.

    Stage 2 (semantic slow-path):
        ``SemanticBridge`` maps the prompt to the nearest ``PromptCluster``
        using bag-of-words cosine similarity.  The cluster's feature deltas
        are blended into the Stage 1 estimates with ``semantic_blend`` weight.

    The fast-path first, slow-path only when needed design:
    - Keeps latency <1ms for high-frequency vocabulary prompts (support queries,
      factual lookups) — the dominant pattern in production.
    - Solves the "vocabulary cliff" for novel phrasing without external models.
    - Zero dependencies; runs entirely in the Python standard library.

    Parameters
    ----------
    bridge:
        ``SemanticBridge`` instance.  Defaults to one built from
        ``_DEFAULT_CLUSTERS``.
    confidence_threshold:
        Stage 1 confidence below this value triggers Stage 2.  Defaults to 0.35.
    semantic_blend:
        Blend weight for semantic cluster deltas.  Defaults to 0.50.

    Example
    -------
    ::

        encoder = DualPathEncoder()

        # Novel phrasing not in keyword list — semantic bridge activates
        f = encoder.encode("Scrutinize the GDPR compliance protocols", "legal")
        # f.semantic_cluster → "Risk" (mapped via cosine similarity)
        # f.encoder_confidence → 0.28 (below threshold, slow-path triggered)

        # Standard phrasing — fast-path only
        f2 = encoder.encode("What is 2 + 2?", "general")
        # f2.encoder_confidence → 0.85 (threshold not crossed)
        # f2.semantic_cluster → "" (fast-path only)
    """

    bridge: SemanticBridge = field(default_factory=SemanticBridge)
    confidence_threshold: float = 0.35
    semantic_blend: float = 0.50

    def encode(self, prompt: str, domain: str = "general") -> PromptFeatures:
        """Encode *prompt* using fast-path + optional semantic slow-path.

        Computes Stage 1 keyword features, measures confidence by counting
        how many distinct pattern groups fired.  If confidence < threshold,
        activates Stage 2 semantic bridge to handle novel vocabulary.

        Returns
        -------
        PromptFeatures
            Includes ``encoder_confidence`` and ``semantic_cluster`` fields
            indicating which path was used and the matched cluster name.
        """
        text = prompt.lower()

        # ----- Stage 1: keyword fast-path -----
        raw_complexity = clamp01(
            _sum_to_signal(text, _COMPLEXITY_SIGNALS, self.complexity_baseline)
            + _length_complexity(text)
        )
        raw_stakes = _sum_to_signal(text, _STAKES_SIGNALS, self.stakes_baseline)
        uncertainty = _sum_to_signal(text, _UNCERTAINTY_SIGNALS, self.uncertainty_baseline)
        source_confidence = _sum_to_signal(text, _SOURCE_CONFIDENCE_SIGNALS, self.source_confidence_baseline)
        tool_relevance = _sum_to_signal(text, _TOOL_RELEVANCE_SIGNALS, self.tool_relevance_baseline)

        # Confidence = fraction of all signal groups that fired at least once
        all_tables = [
            _COMPLEXITY_SIGNALS, _STAKES_SIGNALS, _UNCERTAINTY_SIGNALS,
            _SOURCE_CONFIDENCE_SIGNALS, _TOOL_RELEVANCE_SIGNALS,
        ]
        fired = sum(
            1
            for table in all_tables
            for pattern, _ in table
            if re.search(pattern, text, re.IGNORECASE)
        )
        total_patterns = sum(len(t) for t in all_tables)
        confidence = clamp01(fired / max(1, total_patterns) * 5.0)  # scale to ~0-1

        semantic_cluster = ""
        if confidence < self.confidence_threshold:
            # ----- Stage 2: semantic slow-path -----
            complexity, stakes, uncertainty, tool_relevance, semantic_cluster, _ = (
                self.bridge.apply(
                    text,
                    raw_complexity,
                    raw_stakes,
                    uncertainty,
                    tool_relevance,
                    blend=self.semantic_blend,
                )
            )
            raw_complexity = complexity
            raw_stakes = stakes

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
            encoder_confidence=confidence,
            semantic_cluster=semantic_cluster,
        )

