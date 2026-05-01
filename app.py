"""Minimal Streamlit app for MANIFOLD telemetry inspection."""

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

    with st.expander("Full summary JSON"):
        st.code(json.dumps(summary, indent=2), language="json")
