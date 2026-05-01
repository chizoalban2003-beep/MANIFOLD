"""Project MANIFOLD simulation package."""

from .simulation import (
    GenerationSummary,
    LifeResult,
    ManifoldExperiment,
    SimulationConfig,
    VectorGenome,
    run_experiment,
    transfer_population,
)
from .gridmapper import (
    AgentPopulation,
    DynamicTarget,
    GridOptimizationResult,
    GridWorld,
    Rule,
)
from .social import (
    CellVector,
    PolicyAudit,
    SocialConfig,
    SocialGenerationSummary,
    SocialGenome,
    SocialManifoldExperiment,
    compile_policy_audit,
    config_for_preset,
    recommended_prices,
    run_social_experiment,
)

__all__ = [
    "AgentPopulation",
    "CellVector",
    "DynamicTarget",
    "GenerationSummary",
    "GridOptimizationResult",
    "GridWorld",
    "LifeResult",
    "ManifoldExperiment",
    "PolicyAudit",
    "Rule",
    "SimulationConfig",
    "SocialConfig",
    "SocialGenerationSummary",
    "SocialGenome",
    "SocialManifoldExperiment",
    "VectorGenome",
    "compile_policy_audit",
    "config_for_preset",
    "recommended_prices",
    "run_experiment",
    "run_social_experiment",
    "transfer_population",
]
