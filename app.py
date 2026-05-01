from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from manifold import ManifoldSimulation, extract_teacher_events, reshape_cells


st.set_page_config(page_title="Project MANIFOLD", page_icon="MAN", layout="wide")


def run_simulation(scenario: str, seed: int, generations: int, population_size: int, mutation_sigma: float) -> dict:
    simulation = ManifoldSimulation(
        seed=seed,
        scenario=scenario,
        population_size=None if population_size == 0 else population_size,
        mutation_sigma=mutation_sigma,
    )
    return simulation.run(generations=generations)


def history_frame(results: dict) -> pd.DataFrame:
    frame = pd.DataFrame(results["history"])
    frame["teacher_flag"] = frame["teacher_event"].notna()
    return frame


def population_frame(results: dict) -> pd.DataFrame:
    return pd.DataFrame(results["population"])


def route_frame(results: dict) -> pd.DataFrame:
    return pd.DataFrame(results["routes"])


st.title("Project MANIFOLD")
st.caption(
    "Multi-Agent Non-stationary Framework for Ontogenetic Learning and Dynamic valuation"
)

with st.sidebar:
    st.header("Simulation controls")
    scenario = st.selectbox(
        "Scenario",
        options=["geometry", "emergent", "ontogeny"],
        format_func=lambda value: {
            "geometry": "V1 geometry discovery",
            "emergent": "V3 emergent adaptation",
            "ontogeny": "Phase 5 ontogeny",
        }[value],
    )
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=11, step=1)
    generations = st.slider("Generations", min_value=20, max_value=180, value=80, step=5)
    population_size = st.slider(
        "Population size override (0 = scenario default)",
        min_value=0,
        max_value=60,
        value=0,
        step=1,
    )
    mutation_sigma = st.slider("Mutation sigma", min_value=0.01, max_value=0.12, value=0.05, step=0.01)
    run_button = st.button("Run MANIFOLD", type="primary")

if "results" not in st.session_state or run_button:
    st.session_state["results"] = run_simulation(
        scenario=scenario,
        seed=int(seed),
        generations=int(generations),
        population_size=int(population_size),
        mutation_sigma=float(mutation_sigma),
    )

results = st.session_state["results"]
summary = results["summary"]
history = history_frame(results)
population = population_frame(results)
routes = route_frame(results)
teacher_events = pd.DataFrame(extract_teacher_events(results["history"]))
geometry_matrix = reshape_cells(results["geometry_weights"])
final_cell_matrix = reshape_cells(results["history"][-1]["cell_loads"])

st.markdown(
    """
    MANIFOLD turns the classic 3x3 grid into a moving target. Agents are seeded across a broad
    physics range, learn route preferences inside their lifetimes, and reproduce under regret-based
    selection. In the ontogeny scenario, each vector also carries a finite energy battery that can
    be spent on temporary armor instead of accepting a costly detour.
    """
)

metric_columns = st.columns(5)
metric_columns[0].metric("Start avg regret", f"{summary['start_avg_regret']:.2f}")
metric_columns[1].metric("End avg regret", f"{summary['end_avg_regret']:.2f}")
metric_columns[2].metric("Best regret", f"{summary['best_regret']:.2f}")
metric_columns[3].metric("Final diversity", f"{summary['final_diversity']:.2f}")
metric_columns[4].metric("Teacher events", int(summary["teacher_events"]))

energy_columns = st.columns(3)
energy_columns[0].metric("Final avg energy", f"{summary['final_avg_energy']:.2f}")
energy_columns[1].metric("Peak avg energy", f"{summary['peak_avg_energy']:.2f}")
energy_columns[2].metric("Cumulative avg energy", f"{summary['cumulative_avg_energy']:.2f}")

left, right = st.columns([1.6, 1.0])

with left:
    st.subheader("Population dynamics")
    regret_fig = px.line(
        history,
        x="generation",
        y=["avg_regret", "best_regret", "diversity", "avg_energy_used"],
        markers=False,
    )
    regret_fig.update_layout(legend_title_text="Signal", height=420)
    st.plotly_chart(regret_fig, use_container_width=True)

    st.subheader("Teacher, flicker, and mortality")
    stress_fig = px.line(
        history,
        x="generation",
        y=["deaths", "flicker_risk", "active_spikes", "avg_unresolved_risk"],
    )
    stress_fig.update_layout(legend_title_text="Signal", height=380)
    st.plotly_chart(stress_fig, use_container_width=True)

with right:
    st.subheader("Geometry prior vs final traffic")
    geom_fig = px.imshow(
        geometry_matrix,
        text_auto=".2f",
        aspect="equal",
        color_continuous_scale="Blues",
        title="Static route-membership weights",
    )
    geom_fig.update_xaxes(showticklabels=False)
    geom_fig.update_yaxes(showticklabels=False)
    st.plotly_chart(geom_fig, use_container_width=True)

    traffic_fig = px.imshow(
        final_cell_matrix,
        text_auto=".1f",
        aspect="equal",
        color_continuous_scale="Oranges",
        title="Final generation cell traffic",
    )
    traffic_fig.update_xaxes(showticklabels=False)
    traffic_fig.update_yaxes(showticklabels=False)
    st.plotly_chart(traffic_fig, use_container_width=True)

bottom_left, bottom_right = st.columns(2)

with bottom_left:
    st.subheader("Final population")
    if not population.empty:
        scatter = px.scatter(
            population,
            x="risk_multiplier",
            y="max_risk",
            color="niche",
            size="energy_policy",
            hover_data=["armor_efficiency", "learning_rate", "explore_rate"],
        )
        scatter.update_layout(height=380)
        st.plotly_chart(scatter, use_container_width=True)
    st.dataframe(population, use_container_width=True, hide_index=True)

with bottom_right:
    st.subheader("Route design")
    st.dataframe(routes, use_container_width=True, hide_index=True)
    if not teacher_events.empty:
        st.subheader("Teacher events")
        st.dataframe(teacher_events, use_container_width=True, hide_index=True)
    else:
        st.info("No teacher spikes were triggered in this run.")

st.subheader("Interpretation")
scenario_notes = {
    "Static geometry": (
        "Agents recover the classic center-heavy prior because the middle cell participates in more routes."
    ),
    "Emergent adaptation": (
        "Population-level regret can plateau, triggering the bored teacher to spike dominant corridors."
    ),
    "Ontogeny energy budget": (
        "Vectors must decide whether to burn finite energy on armor now or conserve budget for future spikes."
    ),
}
st.write(scenario_notes.get(results["scenario"], ""))

