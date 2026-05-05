"""Phase 48: Meta-Prompt Evolution — A/B Testing for LLM System Prompts.

MANIFOLD self-optimises its system prompts by running evolutionary A/B tests
between a "Champion" prompt and a mutated "Challenger" prompt.

Architecture
------------
1. :class:`PromptGenome` — dataclass holding a system-prompt template and its
   accumulated performance statistics on MANIFOLD's 4-vector grid
   ``[cost, risk, neutrality, asset]``.
2. :class:`ABTestingEngine` — randomly routes tasks to either the Champion or
   a Challenger and promotes the Challenger to Champion if it outperforms by
   at least :attr:`ABTestingEngine.promotion_threshold` over
   :attr:`ABTestingEngine.min_trials` trials.

Key classes
-----------
``PromptGenome``
    Prompt template + rolling grid-score vector + trial counters.
``ABTestingEngine``
    Champion/Challenger A/B engine with automatic promotion logic.

Promotion rule
--------------
After the Challenger has accumulated at least ``min_trials`` results, if::

    challenger.success_rate > champion.success_rate + promotion_threshold

the Challenger is promoted to Champion and a fresh Challenger is generated.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# PromptGenome
# ---------------------------------------------------------------------------


@dataclass
class PromptGenome:
    """A system-prompt template with historical performance tracking.

    The 4-vector grid mirrors MANIFOLD's canonical scoring axes:
    ``[cost, risk, neutrality, asset]``.

    Parameters
    ----------
    prompt_id:
        Unique identifier for this genome.
    template:
        The system-prompt text that will be injected into agent requests.
    grid_scores:
        Running average of outcomes on each grid axis.  Initialised to
        ``[0.0, 0.0, 0.0, 0.0]``.

    Example
    -------
    ::

        genome = PromptGenome(prompt_id="base", template="You are a helpful agent.")
        genome.record_outcome(success=True, grid_delta=[0.9, 0.1, 0.8, 0.7])
        assert genome.trial_count == 1
        assert genome.success_rate == 1.0
    """

    prompt_id: str
    template: str
    grid_scores: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    trial_count: int = 0
    success_count: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def success_rate(self) -> float:
        """Overall success rate in ``[0.0, 1.0]`` (0.0 if no trials yet)."""
        if self.trial_count == 0:
            return 0.0
        return self.success_count / self.trial_count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        success: bool,
        grid_delta: list[float] | None = None,
    ) -> None:
        """Record one task outcome and update running statistics.

        Parameters
        ----------
        success:
            Whether the task was considered successful.
        grid_delta:
            Optional 4-vector ``[cost, risk, neutrality, asset]`` score for
            this trial.  If provided, each axis is updated as a running
            average.  Values outside ``[0.0, 1.0]`` are accepted but not
            clamped here — the caller controls the scale.
        """
        self.trial_count += 1
        if success:
            self.success_count += 1

        if grid_delta is not None:
            if len(grid_delta) != 4:
                raise ValueError(
                    f"grid_delta must have exactly 4 elements, got {len(grid_delta)}"
                )
            n = self.trial_count
            for i, delta in enumerate(grid_delta):
                # Welford-style running mean
                self.grid_scores[i] += (delta - self.grid_scores[i]) / n

    def mutate(
        self,
        mutation_id: str,
        *,
        seed: int | None = None,
        noise_words: int = 1,
    ) -> "PromptGenome":
        """Return a mutated copy of this genome to be used as a Challenger.

        The mutation strategy appends a version suffix to one or more
        randomly chosen words in the template.  Grid scores are inherited
        from the parent so the Challenger starts with the same prior.

        Parameters
        ----------
        mutation_id:
            ``prompt_id`` for the new genome.
        seed:
            Optional RNG seed for reproducibility.
        noise_words:
            Number of words to mutate.  Clamped to ``[1, len(words)]``.
        """
        rng = random.Random(seed)
        words = self.template.split()
        if words:
            n_mutate = max(1, min(noise_words, len(words)))
            indices = rng.sample(range(len(words)), n_mutate)
            words_copy = list(words)
            for idx in indices:
                words_copy[idx] = words_copy[idx] + f"_v{rng.randint(2, 99)}"
            new_template = " ".join(words_copy)
        else:
            new_template = self.template

        return PromptGenome(
            prompt_id=mutation_id,
            template=new_template,
            grid_scores=list(self.grid_scores),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "prompt_id": self.prompt_id,
            "template": self.template,
            "grid_scores": list(self.grid_scores),
            "trial_count": self.trial_count,
            "success_count": self.success_count,
            "success_rate": round(self.success_rate, 6),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptGenome":
        """Reconstruct a :class:`PromptGenome` from a serialised dict."""
        return cls(
            prompt_id=str(data["prompt_id"]),
            template=str(data["template"]),
            grid_scores=list(data.get("grid_scores", [0.0, 0.0, 0.0, 0.0])),
            trial_count=int(data.get("trial_count", 0)),
            success_count=int(data.get("success_count", 0)),
        )


# ---------------------------------------------------------------------------
# ABTestingEngine
# ---------------------------------------------------------------------------


@dataclass
class ABTestingEngine:
    """Champion / Challenger A/B testing engine for prompt evolution.

    Parameters
    ----------
    champion:
        The current best-performing :class:`PromptGenome`.
    promotion_threshold:
        Minimum improvement in success-rate required for promotion.
        Default: ``0.05`` (5 percentage points).
    min_trials:
        Minimum number of Challenger trials before promotion is evaluated.
        Default: ``100``.
    seed:
        Optional RNG seed for the selection coin-flip.

    Example
    -------
    ::

        champion = PromptGenome(prompt_id="champ", template="Act as an assistant.")
        engine = ABTestingEngine(champion=champion, min_trials=3)

        for _ in range(5):
            genome = engine.select()
            engine.record_outcome(genome, success=True)

        print(engine.summary())
    """

    champion: PromptGenome
    promotion_threshold: float = 0.05
    min_trials: int = 100
    seed: int | None = None

    _challenger: PromptGenome | None = field(default=None, init=False, repr=False)
    _promotions: int = field(default=0, init=False, repr=False)
    _created_at: float = field(
        default_factory=time.time, init=False, repr=False
    )
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_challenger(self) -> PromptGenome:
        """Lazily create a Challenger if none exists."""
        if self._challenger is None:
            self._challenger = self.champion.mutate(
                f"{self.champion.prompt_id}_challenger",
                seed=self._rng.randint(0, 2**31),
            )
        return self._challenger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(self) -> PromptGenome:
        """Randomly select Champion or Challenger for the next task.

        The selection is a fair 50/50 coin-flip so both genomes receive
        statistically comparable exposure.

        Returns
        -------
        PromptGenome
            The selected genome whose :meth:`~PromptGenome.record_outcome`
            should be called once the task result is known.
        """
        challenger = self._ensure_challenger()
        return self.champion if self._rng.random() < 0.5 else challenger

    def record_outcome(
        self,
        genome: PromptGenome,
        *,
        success: bool,
        grid_delta: list[float] | None = None,
    ) -> bool:
        """Record a task outcome on *genome* and potentially promote Challenger.

        Parameters
        ----------
        genome:
            The genome returned by :meth:`select` for this task.
        success:
            Whether the task succeeded.
        grid_delta:
            Optional 4-vector score delta to pass through to
            :meth:`~PromptGenome.record_outcome`.

        Returns
        -------
        bool
            ``True`` if a promotion occurred as a result of this call.
        """
        genome.record_outcome(success, grid_delta)
        return self._maybe_promote()

    def force_promote(self) -> bool:
        """Manually trigger a promotion check regardless of trial count.

        Returns
        -------
        bool
            ``True`` if the Challenger was promoted.
        """
        return self._maybe_promote(force=True)

    def _maybe_promote(self, *, force: bool = False) -> bool:
        """Evaluate promotion criteria and promote Challenger if met."""
        challenger = self._challenger
        if challenger is None:
            return False

        if not force and challenger.trial_count < self.min_trials:
            return False

        if challenger.success_rate > self.champion.success_rate + self.promotion_threshold:
            self.champion = challenger
            self._challenger = None
            self._promotions += 1
            return True

        return False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def promotions(self) -> int:
        """Total number of Challenger promotions that have occurred."""
        return self._promotions

    @property
    def challenger(self) -> PromptGenome | None:
        """The current Challenger genome, or ``None`` if not yet created."""
        return self._challenger

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the engine's state.

        Returns
        -------
        dict
            Keys: ``"champion"``, ``"challenger"`` (or ``None``),
            ``"promotions"``, ``"promotion_threshold"``, ``"min_trials"``.
        """
        return {
            "champion": self.champion.to_dict(),
            "challenger": self._challenger.to_dict() if self._challenger else None,
            "promotions": self._promotions,
            "promotion_threshold": self.promotion_threshold,
            "min_trials": self.min_trials,
        }
