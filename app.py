"""Streamlit dashboard for Project MANIFOLD."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import streamlit as st

from manifold import (
    AgentPopulation,
    BrainConfig,
    BrainTask,
    DialogueTask,
    GridWorld,
    ManifoldBrain,
    TrustRouterConfig,
    ManifoldExperiment,
    SimulationConfig,
    TrustRouter,
    default_tools,
    run_research_suite,
    run_brain_benchmark,
    run_support_trust_audit,
    run_trust_benchmark,
    sample_brain_tasks,
    sample_trust_tasks,
)
from manifold.research import format_research_report
from manifold.trustaudit import format_trust_audit_report
from manifold.social import (
    SocialConfig,
    compile_policy_audit,
    config_for_preset,
    run_social_experiment,
)
from manifold.adversarial import AdversarialPricingDetector
from manifold.autodiscovery import PenaltyOptimizer, PolicySynthesizer
from manifold import (
    ActiveInterceptor,
    AutoRuleDiscovery,
    ConnectorRegistry,
    FederatedGossipBridge,
    GlobalReputationLedger,
    HITLConfig,
    HITLGate,
    InterceptorConfig,
    OrgReputationSnapshot,
    ShadowModeWrapper,
    ToolConnector,
    ToolProfile,
)


st.set_page_config(
    page_title="Project MANIFOLD",
    page_icon="M",
    layout="wide",
)


@st.cache_data(show_spinner="Running MANIFOLD path experiment...")
def run_path_cached(config: SimulationConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    experiment = ManifoldExperiment(config)
    history = experiment.run()
    rows = []
    for item in history:
        row = asdict(item)
        row.update(
            {
                "body": item.niche_counts["Body"],
                "planners": item.niche_counts["Planner"],
                "hybrids": item.niche_counts["Hybrid"],
            }
        )
        del row["niche_counts"]
        rows.append(row)
    overlays = {
        "teacher_spikes": experiment.teacher_spikes,
        "pheromone": experiment.pheromone,
        "teacher_strengths": experiment.teacher_strengths,
    }
    return pd.DataFrame(rows), overlays


@st.cache_data(show_spinner="Running MANIFOLD social experiment...")
def run_social_cached(config: SocialConfig) -> pd.DataFrame:
    rows = []
    for item in run_social_experiment(config):
        row = asdict(item)
        row.update(
            {
                "scouts": item.niche_counts["Scout"],
                "verifiers": item.niche_counts["Verifier"],
                "deceivers": item.niche_counts["Deceiver"],
                "gossips": item.niche_counts["Gossip"],
                "pragmatists": item.niche_counts["Pragmatist"],
            }
        )
        del row["niche_counts"]
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Running GridMapper OS optimization...")
def run_gridmapper_cached(
    size: int,
    generations: int,
    population_size: int,
    seed: int,
    data_path: str,
    target_specs: tuple[str, ...],
    rule_specs: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, object]]:
    world = GridWorld(size=size, seed=seed)
    if data_path:
        world.load_from_csv(data_path)
    for spec in target_specs:
        target_id, row, col, asset, *rest = spec.split(",")
        world.add_dynamic_targets(
            [
                {
                    "id": target_id.strip(),
                    "pos": (int(row), int(col)),
                    "asset": float(asset),
                    "moves": rest[0].strip() if rest else "static",
                }
            ]
        )
    for spec in rule_specs:
        name, penalty, trigger = spec.split(",")
        world.add_rule(name.strip(), float(penalty), trigger.strip())
    if not world.targets:
        center = size // 2
        world.add_dynamic_targets(
            [{"id": "default_target", "pos": (center, center), "asset": 1.0}]
        )

    result = AgentPopulation(seed=str(seed), n=population_size).optimize(
        world, generations=generations
    )
    rows = []
    for item in result.history:
        row = asdict(item)
        row.update(
            {
                "scouts": item.niche_counts["Scout"],
                "verifiers": item.niche_counts["Verifier"],
                "deceivers": item.niche_counts["Deceiver"],
                "gossips": item.niche_counts["Gossip"],
                "pragmatists": item.niche_counts["Pragmatist"],
            }
        )
        del row["niche_counts"]
        rows.append(row)
    metadata = {
        "verification": result.verification,
        "gossip": result.gossip,
        "reputation_cap": result.reputation_cap,
        "robustness": result.audit.robustness_score,
        "rule_penalty_budget": result.rule_penalty_budget,
        "targets": result.target_snapshots.get(len(result.history) - 1, ()),
    }
    return pd.DataFrame(rows), metadata


st.title("Project MANIFOLD")
st.caption("A priced-action engine for evolving social intelligence on vector grids")
st.markdown(
    "MANIFOLD keeps the name: the project is now a manifold of problem spaces. "
    "The subtitle changes from routing to social rules that evolve from economics."
)

with st.sidebar:
    st.header("Experiment controls")
    mode = st.radio(
        "Engine",
        [
            "Research probes",
            "Trust Audit",
            "MANIFOLD Brain",
            "BrainBench",
            "TrustRouter",
            "TrustBench",
            "GridMapper OS",
            "Social intelligence",
            "Path / teacher",
            "Shadow Mode",
        ],
        horizontal=True,
    )
    population_size = st.slider(
        "Population",
        12,
        240,
        180 if mode != "Path / teacher" else 52,
        step=4,
    )
    generations = st.slider(
        "Generations",
        5,
        500,
        40
        if mode in {"MANIFOLD Brain", "TrustRouter"}
        else 80
        if mode == "GridMapper OS"
        else 120
        if mode == "Social intelligence"
        else 200,
        step=5,
    )
    seed = st.number_input("Seed", value=2500 if mode != "Path / teacher" else 13, step=1)

if mode == "Research probes":
    report = run_research_suite(seed=int(seed))
    st.subheader("Bounded research probes")
    for finding in report.findings:
        st.metric(
            finding.name,
            f"{finding.metric:.3f}",
            "PASS" if finding.passed else "WARN",
        )
        st.write(finding.interpretation)
    with st.expander("Honest research summary"):
        st.text(format_research_report(report))
elif mode == "Trust Audit":
    report = run_support_trust_audit()
    st.subheader("Customer support trust audit")
    cols = st.columns(len(report.findings))
    for col, finding in zip(cols, report.findings):
        col.metric(finding.name, f"{finding.improvement:.0%}")
        col.caption(f"baseline={finding.baseline_cost:.2f}, manifold={finding.manifold_cost:.2f}")
    st.subheader("Recommendations")
    for recommendation in report.recommendations:
        st.write("- " + recommendation)
    with st.expander("Audit detail"):
        st.text(format_trust_audit_report(report))
elif mode == "BrainBench":
    config = BrainConfig(
        generations=generations,
        population_size=population_size,
        grid_size=5,
        seed=int(seed),
    )
    report = run_brain_benchmark(sample_brain_tasks(), config)
    rows = [asdict(score) for score in report.scores]
    scores = pd.DataFrame(rows).sort_values("utility", ascending=False)
    st.subheader("BrainBench: agentic policy comparison")
    cols = st.columns(3)
    cols[0].metric("Best policy", report.best_policy)
    cols[1].metric("Brain rank", f"#{report.brain_rank}")
    brain_score = scores[scores["name"] == "manifold_brain"].iloc[0]
    cols[2].metric("Brain utility", f"{brain_score.utility:.3f}")
    st.dataframe(scores, use_container_width=True)
    st.subheader("Benchmark recommendations")
    for recommendation in report.recommendations:
        st.write("- " + recommendation)
elif mode == "MANIFOLD Brain":
    with st.sidebar:
        grid_size = st.select_slider("Grid size", options=[5, 11, 21], value=11)
        prompt = st.text_area("Task", value="Use the best available tool to solve this task safely.")
        domain = st.text_input("Domain", value="general")
        uncertainty = st.slider("Uncertainty", 0.0, 1.0, 0.5, step=0.05)
        complexity = st.slider("Complexity", 0.0, 1.0, 0.6, step=0.05)
        stakes = st.slider("Stakes", 0.0, 1.0, 0.6, step=0.05)
        source_confidence = st.slider("Source confidence", 0.0, 1.0, 0.7, step=0.05)
        tool_relevance = st.slider("Tool relevance", 0.0, 1.0, 0.7, step=0.05)
        time_pressure = st.slider("Time pressure", 0.0, 1.0, 0.4, step=0.05)
        safety_sensitivity = st.slider("Safety sensitivity", 0.0, 1.0, 0.2, step=0.05)
        collaboration_value = st.slider("Collaboration value", 0.0, 1.0, 0.3, step=0.05)
        user_patience = st.slider("User patience", 0.0, 1.0, 0.7, step=0.05)
        dynamic_goal = st.toggle("Dynamic goal", value=False)
    brain = ManifoldBrain(
        BrainConfig(
            generations=generations,
            population_size=population_size,
            grid_size=grid_size,
            seed=int(seed),
        ),
        default_tools(),
    )
    decision = brain.decide(
        BrainTask(
            prompt=prompt,
            domain=domain,
            uncertainty=uncertainty,
            complexity=complexity,
            stakes=stakes,
            source_confidence=source_confidence,
            tool_relevance=tool_relevance,
            time_pressure=time_pressure,
            safety_sensitivity=safety_sensitivity,
            collaboration_value=collaboration_value,
            user_patience=user_patience,
            dynamic_goal=dynamic_goal,
        )
    )
    history = pd.DataFrame([asdict(item) for item in decision.result.history])
    cols = st.columns(6)
    cols[0].metric("Action", decision.action)
    cols[1].metric("Tool", decision.selected_tool or "none")
    cols[2].metric("Confidence", f"{decision.confidence:.0%}")
    cols[3].metric("Risk", f"{decision.risk_score:.0%}")
    cols[4].metric("Utility", f"{decision.expected_utility:.2f}")
    cols[5].metric("Robustness", f"{decision.robustness_score:.2f}")
    for note in decision.notes:
        st.write("- " + note)
    st.subheader("Brain policy trace")
    st.line_chart(
        history.set_index("generation")[
            ["average_verification", "average_gossip", "average_predation_threshold"]
        ]
    )
    with st.expander("Raw Brain generation data"):
        st.dataframe(history, use_container_width=True)
elif mode == "TrustBench":
    config = TrustRouterConfig(
        generations=generations,
        population_size=population_size,
        grid_size=5,
        seed=int(seed),
    )
    report = run_trust_benchmark(sample_trust_tasks(), config)
    rows = [asdict(score) for score in report.scores]
    scores = pd.DataFrame(rows).sort_values("utility", ascending=False)
    st.subheader("TrustBench: policy comparison")
    cols = st.columns(3)
    cols[0].metric("Best policy", report.best_policy)
    cols[1].metric("TrustRouter rank", f"#{report.trustrouter_rank}")
    trustrouter_score = scores[scores["name"] == "trustrouter"].iloc[0]
    cols[2].metric("TrustRouter utility", f"{trustrouter_score.utility:.3f}")
    st.dataframe(scores, use_container_width=True)
    st.subheader("Benchmark recommendations")
    for recommendation in report.recommendations:
        st.write("- " + recommendation)
elif mode == "TrustRouter":
    with st.sidebar:
        grid_size = st.select_slider("Grid size", options=[5, 11, 21], value=11)
        prompt = st.text_area("Prompt / task", value="The user asks an ambiguous support question.")
        domain = st.text_input("Domain", value="support")
        uncertainty = st.slider("Uncertainty", 0.0, 1.0, 0.6, step=0.05)
        complexity = st.slider("Complexity", 0.0, 1.0, 0.5, step=0.05)
        stakes = st.slider("Stakes", 0.0, 1.0, 0.6, step=0.05)
        source_confidence = st.slider("Source confidence", 0.0, 1.0, 0.5, step=0.05)
        user_patience = st.slider("User patience", 0.0, 1.0, 0.7, step=0.05)
        safety_sensitivity = st.slider("Safety sensitivity", 0.0, 1.0, 0.2, step=0.05)
        dynamic_intent = st.toggle("Dynamic user intent", value=False)
    router = TrustRouter(
        TrustRouterConfig(
            generations=generations,
            population_size=population_size,
            grid_size=grid_size,
            seed=int(seed),
        )
    )
    decision = router.route(
        DialogueTask(
            prompt=prompt,
            domain=domain,
            uncertainty=uncertainty,
            complexity=complexity,
            stakes=stakes,
            source_confidence=source_confidence,
            user_patience=user_patience,
            safety_sensitivity=safety_sensitivity,
            dynamic_intent=dynamic_intent,
        )
    )
    history = pd.DataFrame([asdict(item) for item in decision.result.history])
    cols = st.columns(6)
    cols[0].metric("Action", decision.action)
    cols[1].metric("Confidence", f"{decision.confidence:.0%}")
    cols[2].metric("Risk", f"{decision.risk_score:.0%}")
    cols[3].metric("Verification", f"{decision.recommended_verification_rate:.0%}")
    cols[4].metric("Rep cap", f"{decision.reputation_cap:.0%}")
    cols[5].metric("Robustness", f"{decision.robustness_score:.2f}")

    st.subheader("Action thresholds")
    threshold_cols = st.columns(4)
    threshold_cols[0].metric("Clarify", f"{decision.clarification_threshold:.0%}")
    threshold_cols[1].metric("Retrieve", f"{decision.retrieval_threshold:.0%}")
    threshold_cols[2].metric("Verify", f"{decision.verification_threshold:.0%}")
    threshold_cols[3].metric("Escalate", f"{decision.escalation_threshold:.0%}")
    for note in decision.notes:
        st.write("- " + note)

    st.subheader("TrustRouter learning trace")
    st.line_chart(
        history.set_index("generation")[
            ["average_verification", "average_gossip", "average_predation_threshold"]
        ]
    )
    with st.expander("Raw TrustRouter generation data"):
        st.dataframe(history, use_container_width=True)
elif mode == "GridMapper OS":
    with st.sidebar:
        grid_size = st.select_slider("Grid size", options=[5, 11, 21, 31], value=11)
        data_path = st.text_input("CSV grid path", value="")
        targets_text = st.text_area(
            "Targets: id,row,col,asset[,moves]",
            value="order_1,5,5,10,static",
        )
        rules_text = st.text_area(
            "Rules: name,penalty,trigger",
            value="miss_target,1.0,miss_target\ntrusted_lie,0.5,trusted_lie",
        )
    target_specs = tuple(
        line.strip() for line in targets_text.splitlines() if line.strip()
    )
    rule_specs = tuple(line.strip() for line in rules_text.splitlines() if line.strip())
    history, metadata = run_gridmapper_cached(
        grid_size,
        generations,
        population_size,
        int(seed),
        data_path,
        target_specs,
        rule_specs,
    )
    latest = history.iloc[-1]

    cols = st.columns(6)
    cols[0].metric("Fitness", f"{latest.average_fitness:.2f}")
    cols[1].metric("Verification", f"{metadata['verification']:.0%}")
    cols[2].metric("Gossip", f"{metadata['gossip']:.0%}")
    cols[3].metric("Rep cap", f"{metadata['reputation_cap']:.0%}")
    cols[4].metric("Robustness", f"{metadata['robustness']:.2f}")
    cols[5].metric("Rule budget", f"{metadata['rule_penalty_budget']:.2f}")
    st.caption(f"Final target positions: {metadata['targets']}")

    left, right = st.columns(2)
    with left:
        st.subheader("Policy evolution")
        st.line_chart(
            history.set_index("generation")[
                ["average_verification", "average_gossip", "average_predation_threshold"]
            ]
        )
    with right:
        st.subheader("Trust and monopoly")
        st.line_chart(
            history.set_index("generation")[
                ["lie_rate", "trusted_lie_rate", "top_source_share", "monopoly_pressure"]
            ]
        )

    st.subheader("Niches")
    st.area_chart(
        history.set_index("generation")[
            ["scouts", "verifiers", "deceivers", "gossips", "pragmatists"]
        ]
    )

    with st.expander("Raw GridMapper generation data"):
        st.dataframe(history, use_container_width=True)
elif mode == "Social intelligence":
    with st.sidebar:
        preset = st.selectbox("Problem preset", ["trust", "birmingham", "misinformation", "compute"])
        grid_size = st.select_slider("Grid size", options=[11, 21, 31], value=31)
        data_path = st.text_input("CSV grid path", value="")
    if preset == "trust":
        config = SocialConfig(
            population_size=population_size,
            generations=generations,
            seed=int(seed),
            grid_size=grid_size,
            preset=preset,
            data_path=data_path or None,
        )
    else:
        config = config_for_preset(
            preset,
            generations=generations,
            population_size=population_size,
            seed=int(seed),
        )
        if data_path:
            config = SocialConfig(
                population_size=population_size,
                generations=generations,
                seed=int(seed),
                grid_size=grid_size,
                preset=preset,
                signal_cost=config.signal_cost,
                verification_cost=config.verification_cost,
                false_trust_penalty=config.false_trust_penalty,
                detected_lie_penalty=config.detected_lie_penalty,
                data_path=data_path,
            )
    history = run_social_cached(config)
    latest = history.iloc[-1]
    audit = compile_policy_audit(
        [
            type(
                "Summary",
                (),
                {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {"scouts", "verifiers", "deceivers", "gossips", "pragmatists"}
                },
            )()
            for row in history.to_dict("records")
        ],
        config,
    )

    cols = st.columns(6)
    cols[0].metric("Fitness", f"{latest.average_fitness:.2f}")
    cols[1].metric("Deception", f"{latest.average_deception:.0%}")
    cols[2].metric("Verification", f"{latest.average_verification:.0%}")
    cols[3].metric("Gossip", f"{latest.average_gossip:.0%}")
    cols[4].metric("Memory", f"{latest.average_memory_ticks:.0f} ticks")
    cols[5].metric("Diversity", f"{latest.diversity:.2f}")

    st.subheader("Compiled policy recommendations")
    policy_cols = st.columns(5)
    policy_cols[0].metric("Verify above lie p", f"{audit.verification_threshold:.0%}")
    policy_cols[1].metric("Target verification", f"{audit.recommended_verification_rate:.0%}")
    policy_cols[2].metric("Target gossip", f"{audit.recommended_gossip_rate:.0%}")
    policy_cols[3].metric("Predation cap", f"{audit.recommended_predation_threshold:.0%}")
    policy_cols[4].metric("Robustness", f"{audit.robustness_score:.2f}")
    st.caption(f"Forgiveness window: {audit.recommended_forgiveness_window} ticks")
    st.caption("Monopoly controls: " + ", ".join(audit.monopoly_controls))

    left, right = st.columns(2)
    with left:
        st.subheader("Social genes")
        st.line_chart(
            history.set_index("generation")[[
                "average_deception",
                "average_verification",
                "average_gossip",
                "average_predation_threshold",
            ]]
        )
    with right:
        st.subheader("Trust economy")
        st.line_chart(
            history.set_index("generation")[[
                "lie_rate",
                "verification_rate",
                "predatory_scout_rate",
                "trusted_lie_rate",
                "honest_correlation",
            ]]
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Reputation dynamics")
        st.line_chart(history.set_index("generation")[["blacklist_rate", "forgiveness_rate"]])
    with right:
        st.subheader("Verification market concentration")
        st.line_chart(
            history.set_index("generation")[
                [
                    "top_source_share",
                    "source_hhi",
                    "monopoly_pressure",
                    "predatory_scout_rate",
                ]
            ]
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Niches")
        st.area_chart(
            history.set_index("generation")[
                ["scouts", "verifiers", "deceivers", "gossips", "pragmatists"]
            ]
        )
    with right:
        st.subheader("Audit notes")
        for note in audit.notes:
            st.write("- " + note)

    with st.expander("Raw social generation data"):
        st.dataframe(history, use_container_width=True)
elif mode == "Path / teacher":
    with st.sidebar:
        grid_size = st.select_slider("Grid size", options=[11, 21, 31], value=11)
        teacher_mode = st.selectbox(
            "Teacher mode", ["periodic", "reactive", "random", "adversarial", "multi"]
        )
        energy_max = st.slider("Energy battery", 4.0, 20.0, 8.0, step=1.0)
        recharge_enabled = st.toggle("Chargers", value=True)
        communication_enabled = st.toggle("2-bit communication", value=False)
    config = SimulationConfig(
        population_size=population_size,
        generations=generations,
        seed=int(seed),
        grid_size=grid_size,
        teacher_mode=teacher_mode,
        energy_max=energy_max,
        recharge_enabled=recharge_enabled,
        communication_enabled=communication_enabled,
    )
    history, overlays = run_path_cached(config)
    latest = history.iloc[-1]

    metric_cols = st.columns(6)
    metric_cols[0].metric("Survival", f"{latest.survival_rate:.0%}")
    metric_cols[1].metric("Average regret", f"{latest.average_regret:.2f}")
    metric_cols[2].metric("Energy spent", f"{latest.average_energy_spent:.2f}")
    metric_cols[3].metric("Charger visits", f"{latest.average_recharge_visits:.2f}")
    metric_cols[4].metric("max_r", f"{latest.average_max_risk:.2f}")
    metric_cols[5].metric("Aversion", f"{latest.average_energy_aversion:.2f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Survival, regret, and waste")
        st.line_chart(
            history.set_index("generation")[["survival_rate", "average_regret", "average_energy_spent"]]
        )
    with right:
        st.subheader("Phylogeny vs ontogeny")
        st.line_chart(
            history.set_index("generation")[["average_max_risk", "average_energy_aversion"]]
        )

    with st.expander("Current overlays and teacher strengths"):
        st.write("Teacher spikes", overlays["teacher_spikes"] or "none")
        st.write("Death pheromones", overlays["pheromone"] or "none")
        st.write("Teacher strengths", overlays["teacher_strengths"])

    with st.expander("Raw path generation data"):
        st.dataframe(history, use_container_width=True)

elif mode == "Shadow Mode":
    import random as _random

    st.subheader("Shadow Mode — Phases 8-14 Live Dashboard")
    st.caption(
        "Runs MANIFOLD in parallel with a naive ReAct agent over a synthetic customer-support stream. "
        "No production traffic is touched. All results are computed in shadow (observation-only) mode."
    )

    with st.sidebar:
        n_tasks = st.slider("Stream size (tasks)", 50, 500, 120, step=10)
        sm_seed = st.number_input("Shadow seed", value=2500, step=1)
        failure_start = st.slider("Honey-pot failure after N calls", 5, 50, 15, step=1)
        veto_threshold = st.slider("Interceptor veto threshold", 0.20, 0.80, 0.40, step=0.05)

    @st.cache_data(show_spinner="Running MANIFOLD shadow simulation...")
    def _run_shadow(n: int, seed: int, fs: int, veto_thresh: float) -> dict:
        rng = _random.Random(seed)
        tool_names = ["support_kb", "crm_lookup", "billing_api", "ticket_system", "email_sender"]
        _PROMPTS_LOCAL = [
            "My invoice is wrong — I was charged twice.",
            "I cannot log into my account after the reset.",
            "The product arrived damaged.",
            "Can you explain the difference between the plans?",
            "I want to cancel my subscription.",
            "The app crashes when I open settings.",
            "Where is my refund?",
            "I need to change the delivery address.",
            "My promo code is not applying.",
            "I need a copy of my last 12 invoices.",
        ]
        _DOMAINS_LOCAL = ["billing", "technical", "returns", "general", "escalation"]
        _NAIVE_ACTIONS_LOCAL = ["auto_resolve", "route_to_faq", "send_template_email", "escalate", "ignore"]

        # Phase 10 — federated cold start
        ledger = GlobalReputationLedger(min_orgs_required=2)
        for org_id in ["org_alpha", "org_beta", "org_gamma"]:
            rates = {t: (rng.uniform(0.6, 0.95), rng.randint(20, 100)) for t in tool_names}
            snap = OrgReputationSnapshot(org_id=org_id, rates=rates)
            ledger.ingest_snapshot(snap)

        # Phase 8 — registry + shadow wrapper
        registry = ConnectorRegistry()
        call_counts: dict[str, list[int]] = {t: [0] for t in tool_names}

        def _make_fn(name: str, limit: int | None) -> object:
            cc = call_counts[name]
            def fn(q: str) -> dict:
                cc[0] += 1
                if limit is not None and cc[0] > limit:
                    raise ConnectionError(f"{name} unavailable")
                return {"ok": q[:10]}
            return fn

        for t in tool_names:
            lim = fs if t == "billing_api" else None
            reliability = ledger.global_rate(t) or 0.80
            profile = ToolProfile(t, cost=0.1, latency=0.1, reliability=reliability, risk=0.1, asset=0.7)
            registry.register(ToolConnector(name=t, fn=_make_fn(t, lim), profile=profile))

        brain_cfg = BrainConfig(generations=20, population_size=40, grid_size=5, seed=seed)
        brain = ManifoldBrain(brain_cfg, registry.tool_profiles())
        wrapper = ShadowModeWrapper(brain=brain)

        # Phase 9 — HITL gate
        hitl_gate = HITLGate(config=HITLConfig(risk_stakes_threshold=0.55))

        # Phase 11 — adversarial detector
        adv_detector = AdversarialPricingDetector(warm_up_size=max(5, fs // 2), post_window_size=30, drop_threshold=0.35)

        # Phase 12 — auto discovery
        discovery = AutoRuleDiscovery(
            optimizer=PenaltyOptimizer(min_observations=5),
            synthesizer=PolicySynthesizer(min_occurrences=3),
        )

        # Phase 13 — active interceptor
        interceptor = ActiveInterceptor(
            registry=registry,
            brain=brain,
            config=InterceptorConfig(
                risk_veto_threshold=veto_thresh,
                redirect_strategy="hitl",
            ),
        )

        # tracking
        hitl_count = 0
        tool_failures = 0
        high_risk_manifold = 0
        high_risk_naive = 0
        first_failure_idx: int | None = None
        inoculation_speed = -1
        billing_inoculated = False
        ledger_rates: dict[str, list[float]] = {t: [] for t in tool_names}
        # Trust ROI tracking
        gross_exposure = 0.0
        regret_avoided = 0.0

        for i in range(n):
            domain = rng.choice(_DOMAINS_LOCAL)
            prompt = rng.choice(_PROMPTS_LOCAL)
            stakes = rng.uniform(0.4, 0.95) if domain == "escalation" else rng.uniform(0.1, 0.65)
            task = BrainTask(
                prompt=f"[#{i}] {prompt}", domain=domain,
                uncertainty=rng.uniform(0.1, 0.8), complexity=rng.uniform(0.2, 0.7),
                stakes=stakes, source_confidence=rng.uniform(0.4, 0.9),
                tool_relevance=rng.uniform(0.5, 1.0), time_pressure=rng.uniform(0.1, 0.6),
                safety_sensitivity=0.7 if domain == "escalation" else rng.uniform(0.0, 0.4),
                collaboration_value=rng.uniform(0.1, 0.5), user_patience=rng.uniform(0.3, 0.9),
            )
            naive_action = rng.choice(_NAIVE_ACTIONS_LOCAL) if stakes <= 0.75 or rng.random() < 0.45 else "escalate"
            vr = wrapper.observe(task, actual_action=naive_action)

            if hitl_gate.should_escalate(task, vr.manifold_decision):
                hitl_count += 1
            if task.stakes > 0.75:
                if vr.manifold_action == "escalate":
                    high_risk_manifold += 1
                if naive_action == "escalate":
                    high_risk_naive += 1

            tool_name = rng.choice(tool_names)
            connector = registry.get(tool_name)
            result = connector.call(task.prompt[:40])
            outcome = result.to_brain_outcome()
            adv_detector.record(tool_name, result.success)

            if not result.success:
                tool_failures += 1
                if tool_name == "billing_api" and first_failure_idx is None:
                    first_failure_idx = i
                if first_failure_idx is not None and not billing_inoculated:
                    if i - first_failure_idx >= 3:
                        inoculation_speed = i - first_failure_idx
                        billing_inoculated = True
                discovery.observe_rule_event(tool_name, "tool_failure", outcome, penalty=1.0)

            # Phase 13: interceptor pre-flight (non-blocking, log only)
            try:
                ir = interceptor.intercept(task, tool_name)
                profile = registry.get(tool_name)
                risk_exposure = task.stakes * (profile.refreshed_profile().risk if profile else 0.1)
                gross_exposure += risk_exposure
                if not ir.permitted:
                    regret_avoided += risk_exposure
            except KeyError:
                pass

            for t in tool_names:
                c = registry.get(t)
                ledger_rates[t].append(c.observed_reliability())

        shadow_rpt = wrapper.shadow_report()
        interceptor_summary = interceptor.summary()
        return {
            "shadow": shadow_rpt,
            "hitl": hitl_count,
            "tool_failures": tool_failures,
            "high_risk_manifold": high_risk_manifold,
            "high_risk_naive": high_risk_naive,
            "inoculation_speed": inoculation_speed,
            "first_failure_idx": first_failure_idx,
            "adversarial_suspects": adv_detector.suspects(),
            "penalty_proposals": [
                {
                    "rule_name": p.rule_name, "trigger": p.trigger,
                    "current": p.current_penalty, "proposed": p.proposed_penalty,
                    "delta": p.delta, "confidence": p.confidence, "rationale": p.rationale,
                }
                for p in discovery.suggest_penalty_updates()
            ],
            "ledger_rates": {t: v[-1] if v else 1.0 for t, v in ledger_rates.items()},
            "ledger_series": ledger_rates,
            "status": discovery.status(),
            "interceptor": interceptor_summary,
            "gross_exposure": round(gross_exposure, 4),
            "regret_avoided": round(regret_avoided, 4),
        }

    data = _run_shadow(n_tasks, int(sm_seed), failure_start, veto_threshold)
    shadow = data["shadow"]

    # ── Top-level metrics ──────────────────────────────────────────────
    st.subheader("Observation summary")
    m0, m1, m2, m3, m4 = st.columns(5)
    m0.metric("Tasks observed", shadow["total_observations"])
    m1.metric("MANIFOLD disagreements", shadow["total_disagreements"])
    m2.metric("Disagreement rate", f"{shadow['disagreement_rate']:.1%}")
    m3.metric("HITL triggers (Ph 9)", data["hitl"])
    m4.metric("Tool failures (Ph 8)", data["tool_failures"])

    # ── ROI signals ───────────────────────────────────────────────────
    st.subheader("ROI signals — virtual regret saved")
    r0, r1, r2, r3 = st.columns(4)
    r0.metric("High-risk escalations (MANIFOLD)", data["high_risk_manifold"])
    r1.metric("High-risk escalations (naive agent)", data["high_risk_naive"])
    saved = max(0, data["high_risk_manifold"] - data["high_risk_naive"])
    r2.metric("Virtual regret saved", saved, delta=f"+{saved}" if saved > 0 else None)
    inoculation = (
        f"{data['inoculation_speed']} tasks"
        if data["inoculation_speed"] >= 0
        else "n/a"
    )
    r3.metric("Gossip inoculation speed (Ph 10)", inoculation)

    # ── Phase 10: Global Reputation Ledger ────────────────────────────
    st.subheader("Phase 10 — Global Reputation Ledger")
    st.caption("Final observed reliability per tool (blended from federated gossip).")
    ledger_df = pd.DataFrame(
        [{"tool": t, "observed_reliability": round(v, 3)} for t, v in data["ledger_rates"].items()]
    ).sort_values("observed_reliability")
    st.bar_chart(ledger_df.set_index("tool")["observed_reliability"])

    # ── Phase 11: Adversarial suspects ───────────────────────────────
    st.subheader("Phase 11 — Adversarial Pricing Detector")
    suspects = data["adversarial_suspects"]
    if suspects:
        for s in suspects:
            st.error(
                f"⚠️ **HONEY-POT DETECTED**: `{s['tool_name']}`  |  "
                f"warm-up={s['warm_up_rate']:.0%}  post={s['post_rate']:.0%}  "
                f"drop={s['drop']:.0%}"
            )
    else:
        st.success("✓ No adversarial honey-pot tools detected")

    # ── Phase 12: Penalty proposals ──────────────────────────────────
    st.subheader("Phase 12 — AutoRuleDiscovery: Penalty Proposals")
    proposals = data["penalty_proposals"]
    if proposals:
        for p in proposals:
            direction = "↑" if p["delta"] > 0 else "↓"
            st.info(
                f"{direction} **{p['rule_name']}** / `{p['trigger']}` | "
                f"current={p['current']:.3f} → proposed={p['proposed']:.3f}  "
                f"(Δ={p['delta']:+.3f}, conf={p['confidence']:.0%})  \n"
                f"_{p['rationale']}_"
            )
    else:
        status = data["status"]
        st.caption(
            f"No proposals yet — triggers observed: {status['known_triggers']}  |  "
            f"min_observations threshold: {status.get('min_observations', 5)}"
        )

    # ── Phase 13: Active Interceptor ─────────────────────────────────
    st.subheader("Phase 13 — Active Interceptor")
    ic = data.get("interceptor", {})
    if ic:
        i0, i1, i2, i3, i4 = st.columns(5)
        i0.metric("Calls evaluated", ic.get("total_calls", 0))
        i1.metric("Permitted", ic.get("permitted", 0))
        i2.metric("Vetoed", ic.get("vetoed", 0))
        i3.metric("Veto rate", f"{ic.get('veto_rate', 0):.1%}")
        i4.metric("Redirected to HITL", ic.get("redirected_to_hitl", 0))
        st.caption(f"Avg risk score: {ic.get('avg_risk_score', 0):.3f}  |  Veto threshold: {veto_threshold:.2f}")
    else:
        st.caption("(interceptor not active)")

    # ── Phase 16: Trust ROI Economic Impact View ──────────────────────
    st.subheader("Phase 16 — Trust ROI: Economic Impact View")
    st.caption(
        "The metrics below quantify the **financial value** MANIFOLD provides. "
        "Gross Exposure = total potential damage if tools ran unmonitored. "
        "Regret Avoided = savings from vetoed high-risk calls."
    )
    gross = data.get("gross_exposure", 0.0)
    avoided = data.get("regret_avoided", 0.0)
    veto_rate = ic.get("veto_rate", 0.0) if ic else 0.0
    hitl_rate = data["hitl"] / max(1, shadow["total_observations"])
    gossip_speed = data["inoculation_speed"]

    e0, e1, e2, e3, e4 = st.columns(5)
    e0.metric(
        "Gross Exposure",
        f"{gross:.3f}",
        help="Σ(Risk × Stakes) — total potential damage if unmonitored.",
    )
    e1.metric(
        "Regret Avoided",
        f"{avoided:.3f}",
        delta=f"+{avoided:.3f}" if avoided > 0 else None,
        help="Exposure saved by veto — the 'Value Shield' metric.",
    )
    shield_pct = avoided / gross if gross > 0 else 0.0
    e2.metric(
        "Shield Efficiency",
        f"{shield_pct:.1%}",
        help="Regret Avoided / Gross Exposure — how much exposure MANIFOLD eliminated.",
    )
    e3.metric(
        "Gossip Lead-Time",
        f"{gossip_speed} tasks" if gossip_speed >= 0 else "n/a",
        help="Tasks between first tool failure and global inoculation.",
    )
    e4.metric(
        "Human Efficiency",
        f"{hitl_rate:.1%}",
        help="Fraction of tasks that needed HITL — lower is better post-automation.",
    )

    # ── Top disagreements ─────────────────────────────────────────────
    st.subheader("Top MANIFOLD ↔ Naive disagreement pairs")
    top = shadow["top_disagreement_actions"]
    if top:
        disc_df = pd.DataFrame(top, columns=["Naive action", "MANIFOLD action"])
        st.dataframe(disc_df, use_container_width=True)
    else:
        st.caption("All actions agreed.")

