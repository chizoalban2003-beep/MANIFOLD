"""Streamlit dashboard for Project MANIFOLD."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import streamlit as st

from manifold import ManifoldExperiment, SimulationConfig


st.set_page_config(
    page_title="Project MANIFOLD",
    page_icon="M",
    layout="wide",
)


@st.cache_data(show_spinner="Running MANIFOLD experiment...")
def run_cached(config: SimulationConfig) -> tuple[pd.DataFrame, dict[str, object]]:
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


st.title("Project MANIFOLD")
st.caption("Multi-Agent Non-stationary Framework for Ontogenetic Learning and Dynamic valuation")
st.markdown(
    "MANIFOLD tests whether intelligence emerges when agents must budget finite "
    "energy against terrain risk, teacher spikes, and waste."
)

with st.sidebar:
    st.header("Experiment controls")
    population_size = st.slider("Population", 12, 120, 52, step=4)
    generations = st.slider("Generations", 10, 300, 200, step=10)
    seed = st.number_input("Seed", value=13, step=1)
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

history, overlays = run_cached(config)
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

left, right = st.columns(2)
with left:
    st.subheader("Niche ecology")
    st.area_chart(history.set_index("generation")[["body", "planners", "hybrids"]])

with right:
    st.subheader("Communication and deception")
    st.line_chart(
        history.set_index("generation")[["signal_spike_correlation", "lie_rate"]]
    )

st.subheader("Planning pressure")
st.line_chart(history.set_index("generation")[["average_recharge_visits", "diversity", "teacher_strength"]])

teacher_events = history[history["teacher_mutated"]]
if not teacher_events.empty:
    st.info(
        "Teacher spikes occurred at generations: "
        + ", ".join(str(int(value)) for value in teacher_events["generation"].head(20))
        + ("..." if len(teacher_events) > 20 else "")
    )

with st.expander("Current overlays and teacher strengths"):
    st.write("Teacher spikes", overlays["teacher_spikes"] or "none")
    st.write("Death pheromones", overlays["pheromone"] or "none")
    st.write("Teacher strengths", overlays["teacher_strengths"])

with st.expander("Raw generation data"):
    st.dataframe(history, use_container_width=True)
