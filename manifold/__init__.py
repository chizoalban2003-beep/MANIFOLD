"""MANIFOLD — The Universal Governance OS for Digital and Physical Intelligence.

Submodules that are not yet fully implemented are imported with a guard
so the package remains importable at all times.
"""

__version__ = "2.9.1"

# ---------------------------------------------------------------------------
# Implemented modules (always available)
# ---------------------------------------------------------------------------

try:
    from .brain import (
        AssetAdapter,
        BrainConfig,
        BrainDecision,
        BrainMemory,
        BrainOutcome,
        BrainTask,
        DecompositionPlan,
        GossipNote,
        HierarchicalBrain,
        HierarchicalDecision,
        LearnedPrices,
        ManifoldBrain,
        PriceAdapter,
        ScoutRecord,
        SubTaskSpec,
        ToolProfile,
        attribute_to_tool,
        classify_user_signal,
        decide_task,
        default_tools,
    )
except (ImportError, AttributeError):
    pass

try:
    from .connector import (
        ConnectorRegistry,
        ShadowModeWrapper,
        ToolConnector,
        ToolConnectorResult,
        VirtualRegret,
    )
except (ImportError, AttributeError):
    pass

try:
    from .adversarial import (
        AdversarialPricingDetector,
        AuditTrigger,
        NashEquilibriumGate,
        ReputationLaunderingDetector,
        ToolOutcomeWindow,
    )
except (ImportError, AttributeError):
    pass

try:
    from .autodiscovery import (
        AutoRuleDiscovery,
        DecompositionTemplate,
        PenaltyOptimizer,
        PenaltyProposal,
        PolicySynthesizer,
        RuleObservation,
    )
except (ImportError, AttributeError):
    pass

try:
    from .brainbench import (
        BrainBenchmarkReport,
        BrainLabelledTask,
        BrainPolicyScore,
        load_brain_tasks_csv,
        run_brain_benchmark,
        sample_brain_tasks,
    )
except (ImportError, AttributeError):
    pass

try:
    from .adapters import (
        ManifoldCallbackHandler,
        ManifoldOpenAIWrapper,
    )
except (ImportError, AttributeError):
    pass

try:
    from .b2b import (
        AgentEconomyLedger,
        B2BRouteResult,
        B2BRouter,
        EconomyEntry,
        HandshakeResult,
        OrgPolicy,
        PolicyHandshake,
    )
except (ImportError, AttributeError):
    pass

try:
    from .crypto import (
        GossipSigner,
        OrgPolicySigner,
        PolicySigningKey,
        SignatureVerificationError,
        SignedGossipNote,
        SignedOrgPolicy,
        VerifiedPolicyHandshake,
    )
except (ImportError, AttributeError):
    pass

# ---------------------------------------------------------------------------
# In-progress modules — guarded until fully implemented
# ---------------------------------------------------------------------------

try:
    from .simulation import (
        GenerationSummary,
        LifeResult,
        ManifoldExperiment,
        SimulationConfig,
        VectorGenome,
        run_experiment,
        transfer_population,
    )
except (ImportError, AttributeError):
    pass

try:
    from .trustrouter import (
        DialogueTask,
        TrustLearningMemory,
        TrustRouter,
        TrustRouterConfig,
        TrustRouterDecision,
        route_task,
    )
except (ImportError, AttributeError):
    pass

try:
    from .live import GossipBus, HierarchicalLiveBrain, LiveBrain
except (ImportError, AttributeError):
    pass

try:
    from .encoder import (
        DualPathEncoder,
        EncoderCorrection,
        PromptCluster,
        PromptEncoder,
        PromptFeatures,
        SemanticBridge,
    )
except (ImportError, AttributeError):
    pass

try:
    from .transfer import ReputationRegistry, WarmStartConfig, warm_start_memory
except (ImportError, AttributeError):
    pass

try:
    from .hitl import HITLConfig, HITLGate, HITLRecord, TeacherSpike
except (ImportError, AttributeError):
    pass

try:
    from .federation import (
        FederatedGossipBridge,
        FederatedGossipPacket,
        GlobalReputationLedger,
        OrgReputationSnapshot,
        cold_start_from_ledger,
    )
except (ImportError, AttributeError):
    pass

try:
    from .interceptor import (
        ActiveInterceptor,
        InterceptorConfig,
        InterceptorVeto,
        InterceptResult,
        shield,
    )
except (ImportError, AttributeError):
    pass

try:
    from .trustaudit import (
        TrustAuditConfig,
        TrustAuditFinding,
        TrustAuditReport,
        format_trust_audit_report,
        run_support_trust_audit,
        sample_support_tasks,
    )
except (ImportError, AttributeError):
    pass

try:
    from .research import (
        ResearchFinding,
        ResearchReport,
        run_asset_learning_suite,
        run_dual_encoder_suite,
        run_encoder_suite,
        run_gossip_hierarchical_suite,
        run_gossip_research_suite,
        run_hierarchical_suite,
        run_price_learning_suite,
        run_research_suite,
        run_server_telemetry_suite,
        run_warm_start_suite,
    )
except (ImportError, AttributeError):
    pass

try:
    from .trustbench import (
        BenchmarkReport,
        LabelledTask,
        PolicyScore,
        load_labelled_tasks_csv,
        run_trust_benchmark,
        sample_trust_tasks,
    )
except (ImportError, AttributeError):
    pass

try:
    from .gridmapper import (
        AgentPopulation,
        DynamicTarget,
        GridOptimizationResult,
        GridWorld,
        Rule,
    )
except (ImportError, AttributeError):
    pass

try:
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
except (ImportError, AttributeError):
    pass

try:
    from .hub import CommunityBaseline, ReputationHub
except (ImportError, AttributeError):
    pass

try:
    from .recruiter import MarketplaceListing, RecruitmentResult, SovereignRecruiter
except (ImportError, AttributeError):
    pass

try:
    from .policy import (
        DOMAIN_TEMPLATES,
        ManifoldPolicy,
        PolicyDomain,
        PolicyExporter,
        PolicyLoader,
        RuleDiff,
    )
except (ImportError, AttributeError):
    pass

try:
    from .gitops import (
        CIRiskDelta,
        CIRiskReport,
        ManifoldCICheck,
        AutonomousPRProposal,
        PRDraft,
        generate_github_action,
    )
except (ImportError, AttributeError):
    pass

try:
    from .fleet import (
        B2BEconomySnapshot,
        CIBuildHistory,
        CIBuildRecord,
        FleetDashboardData,
        FleetPanelRenderer,
    )
except (ImportError, AttributeError):
    pass

try:
    from .polyglot import (
        ManifoldOpenAPISpec,
        generate_openapi_spec,
        spec_to_json,
        spec_to_yaml,
    )
except (ImportError, AttributeError):
    pass

try:
    from .privacy import PrivacyConfig, PrivacyGuard
except (ImportError, AttributeError):
    pass

try:
    from .replay import ReplayReport, StateRehydrator, VirtualExecution
except (ImportError, AttributeError):
    pass

try:
    from .verify import PolicyConflict, PolicyVerifier, VerificationResult
except (ImportError, AttributeError):
    pass

try:
    from .mapreduce import (
        Chunk,
        ChunkDistributor,
        ChunkResult,
        JobResult,
        JobTracker,
        MapReduceJob,
        SwarmAggregator,
    )
except (ImportError, AttributeError):
    pass

try:
    from .zkp import PolicyCommitment, ZKPVerifier, ZKProof
except (ImportError, AttributeError):
    pass

try:
    from .registry import (
        PublishResult,
        RegistryEntry,
        SwarmRegistry,
        ToolEndorsement,
        ToolManifest,
    )
except (ImportError, AttributeError):
    pass

try:
    from .temporal import (
        BranchResult,
        CollapseResult,
        ForkSpec,
        ParallelTimeline,
        StateForker,
        TimelineCollapse,
    )
except (ImportError, AttributeError):
    pass

try:
    from .rosetta import (
        EgressResult,
        EgressTranslator,
        ForeignPayloadIngress,
        FrameworkSchema,
        IngressResult,
    )
except (ImportError, AttributeError):
    pass

try:
    from .singularity import (
        ASTMutator,
        MutationProposal,
        MutationStrategy,
        OptimizationResult,
        SandboxedTestRunner,
        STRATEGIES,
    )
except (ImportError, AttributeError):
    pass


__all__ = [
    # brain
    "AssetAdapter", "BrainConfig", "BrainDecision", "BrainMemory",
    "BrainOutcome", "BrainTask", "DecompositionPlan", "GossipNote",
    "HierarchicalBrain", "HierarchicalDecision", "LearnedPrices",
    "ManifoldBrain", "PriceAdapter", "ScoutRecord", "SubTaskSpec",
    "ToolProfile", "attribute_to_tool", "classify_user_signal",
    "decide_task", "default_tools",
    # connector
    "ConnectorRegistry", "ShadowModeWrapper", "ToolConnector",
    "ToolConnectorResult", "VirtualRegret",
    # adversarial
    "AdversarialPricingDetector", "AuditTrigger", "NashEquilibriumGate",
    "ReputationLaunderingDetector", "ToolOutcomeWindow",
    # autodiscovery
    "AutoRuleDiscovery", "DecompositionTemplate", "PenaltyOptimizer",
    "PenaltyProposal", "PolicySynthesizer", "RuleObservation",
    # brainbench
    "BrainBenchmarkReport", "BrainLabelledTask", "BrainPolicyScore",
    "load_brain_tasks_csv", "run_brain_benchmark", "sample_brain_tasks",
    # adapters
    "ManifoldCallbackHandler", "ManifoldOpenAIWrapper",
    # b2b
    "AgentEconomyLedger", "B2BRouteResult", "B2BRouter", "EconomyEntry",
    "HandshakeResult", "OrgPolicy", "PolicyHandshake",
    # crypto
    "GossipSigner", "OrgPolicySigner", "PolicySigningKey",
    "SignatureVerificationError", "SignedGossipNote", "SignedOrgPolicy",
    "VerifiedPolicyHandshake",
    # simulation
    "GenerationSummary", "LifeResult", "ManifoldExperiment", "SimulationConfig",
    "VectorGenome", "run_experiment", "transfer_population",
    # trustrouter
    "DialogueTask", "TrustLearningMemory", "TrustRouter", "TrustRouterConfig",
    "TrustRouterDecision", "route_task",
    # live
    "GossipBus", "HierarchicalLiveBrain", "LiveBrain",
    # encoder
    "DualPathEncoder", "EncoderCorrection", "PromptCluster", "PromptEncoder",
    "PromptFeatures", "SemanticBridge",
    # transfer
    "ReputationRegistry", "WarmStartConfig", "warm_start_memory",
    # hitl
    "HITLConfig", "HITLGate", "HITLRecord", "TeacherSpike",
    # federation
    "FederatedGossipBridge", "FederatedGossipPacket", "GlobalReputationLedger",
    "OrgReputationSnapshot", "cold_start_from_ledger",
    # interceptor
    "ActiveInterceptor", "InterceptorConfig", "InterceptorVeto",
    "InterceptResult", "shield",
    # trustaudit
    "TrustAuditConfig", "TrustAuditFinding", "TrustAuditReport",
    "format_trust_audit_report", "run_support_trust_audit", "sample_support_tasks",
    # research
    "ResearchFinding", "ResearchReport", "run_asset_learning_suite",
    "run_dual_encoder_suite", "run_encoder_suite", "run_gossip_hierarchical_suite",
    "run_gossip_research_suite", "run_hierarchical_suite", "run_price_learning_suite",
    "run_research_suite", "run_server_telemetry_suite", "run_warm_start_suite",
    # trustbench
    "BenchmarkReport", "LabelledTask", "PolicyScore", "load_labelled_tasks_csv",
    "run_trust_benchmark", "sample_trust_tasks",
    # gridmapper
    "AgentPopulation", "DynamicTarget", "GridOptimizationResult", "GridWorld", "Rule",
    # social
    "CellVector", "PolicyAudit", "SocialConfig", "SocialGenerationSummary",
    "SocialGenome", "SocialManifoldExperiment", "compile_policy_audit",
    "config_for_preset", "recommended_prices", "run_social_experiment",
    # hub
    "CommunityBaseline", "ReputationHub",
    # recruiter
    "MarketplaceListing", "RecruitmentResult", "SovereignRecruiter",
    # policy
    "DOMAIN_TEMPLATES", "ManifoldPolicy", "PolicyDomain", "PolicyExporter",
    "PolicyLoader", "RuleDiff",
    # gitops
    "CIRiskDelta", "CIRiskReport", "ManifoldCICheck", "AutonomousPRProposal",
    "PRDraft", "generate_github_action",
    # fleet
    "B2BEconomySnapshot", "CIBuildHistory", "CIBuildRecord",
    "FleetDashboardData", "FleetPanelRenderer",
    # polyglot
    "ManifoldOpenAPISpec", "generate_openapi_spec", "spec_to_json", "spec_to_yaml",
    # privacy
    "PrivacyConfig", "PrivacyGuard",
    # replay
    "ReplayReport", "StateRehydrator", "VirtualExecution",
    # verify
    "PolicyConflict", "PolicyVerifier", "VerificationResult",
    # mapreduce
    "Chunk", "ChunkDistributor", "ChunkResult", "JobResult", "JobTracker",
    "MapReduceJob", "SwarmAggregator",
    # zkp
    "PolicyCommitment", "ZKPVerifier", "ZKProof",
    # registry
    "PublishResult", "RegistryEntry", "SwarmRegistry", "ToolEndorsement", "ToolManifest",
    # temporal
    "BranchResult", "CollapseResult", "ForkSpec", "ParallelTimeline",
    "StateForker", "TimelineCollapse",
    # rosetta
    "EgressResult", "EgressTranslator", "ForeignPayloadIngress",
    "FrameworkSchema", "IngressResult",
    # singularity
    "ASTMutator", "MutationProposal", "MutationStrategy", "OptimizationResult",
    "SandboxedTestRunner", "STRATEGIES",
]
