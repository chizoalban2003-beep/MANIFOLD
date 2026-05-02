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
    run_trust_benchmark,
    sample_brain_tasks,
    sample_trust_tasks,
)
from manifold.research import format_research_report
from manifold.social import (
    SocialConfig,
    compile_policy_audit,
    config_for_preset,
    run_social_experiment,
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
            "MANIFOLD Brain",
            "BrainBench",
            "TrustRouter",
            "TrustBench",
            "GridMapper OS",
            "Social intelligence",
            "Path / teacher",
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
else:
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
