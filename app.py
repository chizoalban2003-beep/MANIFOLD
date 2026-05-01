"""Streamlit dashboard for MANIFOLD production telemetry."""

from __future__ import annotations

import json

import streamlit as st

from manifold.simulation import ManifoldConfig, run_manifold, summarize_result


st.set_page_config(page_title="Project MANIFOLD", layout="wide")

st.title("Project MANIFOLD")
st.caption(
    "Multi-Agent Non-stationary Framework for Ontogenetic Learning and Dynamic valuation"
)

seed = st.sidebar.number_input("Seed", min_value=0, max_value=100_000, value=7, step=1)
show_explainability = st.sidebar.checkbox("Show explainability samples", value=True)
show_confidence = st.sidebar.checkbox("Show confidence distribution", value=True)
run_clicked = st.sidebar.button("Run simulation", type="primary")

if run_clicked:
    with st.spinner("Simulating..."):
        result = run_manifold(config=ManifoldConfig(seed=int(seed)))
        summary = summarize_result(result)

    st.subheader("Summary")
    st.json(summary)

    st.subheader("Recent generations")
    rows = [metric.__dict__ for metric in result.metrics[-10:]]
    st.dataframe(rows, use_container_width=True)

    last = result.metrics[-1]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predator spawn rate", f"{last.predator_spawn_rate:.3f}")
    with col2:
        st.metric("Max reputation", f"{last.max_reputation:.3f}")
    with col3:
        st.metric("Memory revenue (avg)", f"{last.average_memory_revenue:.3f}")

    st.subheader("Predator dashboard")
    pred_rows = [
        {
            "generation": metric.generation,
            "phase": metric.phase,
            "predator_spawn_rate": metric.predator_spawn_rate,
            "max_reputation": metric.max_reputation,
            "death_rate": metric.death_rate,
        }
        for metric in result.metrics
    ]
    st.dataframe(pred_rows, use_container_width=True)

    if show_confidence:
        st.subheader("Decision confidence distribution")
        st.json(last.confidence_distribution)

    if show_explainability:
        st.subheader("Explainability samples")
        for sample in last.explain_samples:
            st.code(sample)

    with st.expander("Full summary JSON"):
        st.code(json.dumps(summary, indent=2), language="json")
