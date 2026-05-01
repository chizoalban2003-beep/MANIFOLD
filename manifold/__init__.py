"""MANIFOLD simulation package."""

from .simulation import (
    ACTION_ADVANCE,
    ACTION_DETOUR_RECHARGE,
    ACTION_PAUSE_RECHARGE,
    AgentGenome,
    GenerationMetrics,
    LAYER_INFO,
    LAYER_PHYSICAL,
    LAYER_SOCIAL,
    ManifoldConfig,
    PhaseConfig,
    SimulationResult,
    run_manifold,
    summarize_result,
)
from .rules import RuleDefinition, compile_rulebook

__all__ = [
    "AgentGenome",
    "ACTION_ADVANCE",
    "ACTION_DETOUR_RECHARGE",
    "ACTION_PAUSE_RECHARGE",
    "LAYER_INFO",
    "LAYER_PHYSICAL",
    "LAYER_SOCIAL",
    "GenerationMetrics",
    "ManifoldConfig",
    "PhaseConfig",
    "RuleDefinition",
    "SimulationResult",
    "compile_rulebook",
    "run_manifold",
    "summarize_result",
]
