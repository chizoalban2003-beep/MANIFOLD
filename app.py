"""Streamlit dashboard for Project MANIFOLD."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import streamlit as st

from manifold import ManifoldExperiment, SimulationConfig
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


st.title("Project MANIFOLD")
st.caption("A priced-action engine for evolving social intelligence on vector grids")
st.markdown(
    "MANIFOLD keeps the name: the project is now a manifold of problem spaces. "
    "The subtitle changes from routing to social rules that evolve from economics."
)

with st.sidebar:
    st.header("Experiment controls")
    mode = st.radio("Engine", ["Social intelligence", "Path / teacher"], horizontal=True)
    population_size = st.slider("Population", 12, 240, 180 if mode == "Social intelligence" else 52, step=4)
    generations = st.slider("Generations", 5, 500, 120 if mode == "Social intelligence" else 200, step=5)
    seed = st.number_input("Seed", value=2500 if mode == "Social intelligence" else 13, step=1)

if mode == "Social intelligence":
    with st.sidebar:
        preset = st.selectbox("Problem preset", ["trust", "birmingham", "misinformation", "compute"])
        grid_size = st.select_slider("Grid size", options=[11, 21, 31], value=31)
    if preset == "trust":
        config = SocialConfig(
            population_size=population_size,
            generations=generations,
            seed=int(seed),
            grid_size=grid_size,
            preset=preset,
        )
    else:
        config = config_for_preset(
            preset,
            generations=generations,
            population_size=population_size,
            seed=int(seed),
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
    policy_cols[3].metric("Forgiveness", f"{audit.recommended_forgiveness_window} ticks")
    policy_cols[4].metric("Robustness", f"{audit.robustness_score:.2f}")
    st.caption("Monopoly controls: " + ", ".join(audit.monopoly_controls))

    left, right = st.columns(2)
    with left:
        st.subheader("Social genes")
        st.line_chart(
            history.set_index("generation")[[
                "average_deception",
                "average_verification",
                "average_gossip",
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
