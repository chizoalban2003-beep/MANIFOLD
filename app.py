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
def run_cached(config: SimulationConfig) -> tuple[pd.DataFrame, dict[str, dict[tuple[int, int], float]]]:
    experiment = ManifoldExperiment(config)
    history = experiment.run()
    rows = []
    for item in history:
        row = asdict(item)
        row.update(
            {
                "tanks": item.niche_counts["Tank"],
                "scouts": item.niche_counts["Scout"],
                "hybrids": item.niche_counts["Hybrid"],
            }
        )
        del row["niche_counts"]
        rows.append(row)
    overlays = {
        "teacher_spikes": experiment.teacher_spikes,
        "pheromone": experiment.pheromone,
    }
    return pd.DataFrame(rows), overlays


st.title("Project MANIFOLD")
st.caption("Multi-Agent Non-stationary Framework for Ontogenetic Learning and Dynamic valuation")

with st.sidebar:
    st.header("Experiment controls")
    population_size = st.slider("Population", 12, 80, 36, step=4)
    generations = st.slider("Generations", 10, 160, 60, step=5)
    seed = st.number_input("Seed", value=13, step=1)
    teacher_interval = st.slider("Teacher interval", 5, 40, 15)
    flicker_period = st.slider("Flicker period", 2, 20, 8)
    energy_max = st.slider("Energy battery", 5.0, 60.0, 30.0, step=1.0)
    recharge_enabled = st.toggle("Recharge sub-targets", value=True)
    recharge_amount = st.slider("Recharge amount", 1.0, 30.0, 12.0, step=1.0)

config = SimulationConfig(
    population_size=population_size,
    generations=generations,
    seed=int(seed),
    teacher_interval=teacher_interval,
    flicker_period=flicker_period,
    energy_max=energy_max,
    recharge_enabled=recharge_enabled,
    recharge_amount=recharge_amount,
)

history, overlays = run_cached(config)
latest = history.iloc[-1]

metric_cols = st.columns(5)
metric_cols[0].metric("Average regret", f"{latest.average_regret:.2f}")
metric_cols[1].metric("Best regret", f"{latest.best_regret:.2f}")
metric_cols[2].metric("Diversity", f"{latest.diversity:.2f}")
metric_cols[3].metric("Energy spent", f"{latest.average_energy_spent:.2f}")
metric_cols[4].metric("Flicker risk", f"{latest.flicker_risk:.1f}")

left, right = st.columns(2)
with left:
    st.subheader("Regret and energy load")
    st.line_chart(
        history.set_index("generation")[
            ["average_regret", "best_regret", "average_energy_spent"]
        ]
    )

with right:
    st.subheader("Niche ecology")
    st.area_chart(history.set_index("generation")[["tanks", "scouts", "hybrids"]])

st.subheader("Diversity under non-stationarity")
st.line_chart(history.set_index("generation")[["diversity", "average_energy_remaining"]])

teacher_events = history[history["teacher_mutated"]]
if not teacher_events.empty:
    st.info(
        "Bored Teacher mutations occurred at generations: "
        + ", ".join(str(int(value)) for value in teacher_events["generation"])
    )

with st.expander("Current risk overlays"):
    st.write("Teacher spikes", overlays["teacher_spikes"] or "none")
    st.write("Death pheromones", overlays["pheromone"] or "none")

with st.expander("Raw generation data"):
    st.dataframe(history, use_container_width=True)
