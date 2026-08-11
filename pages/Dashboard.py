"""
pages/Dashboard.py
-------------------
Aggregate analytics dashboard over all logged patient screenings.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.logger import load_logs, summary_stats

st.set_page_config(page_title="Dashboard | Heart Attack Prediction Agent", page_icon="📊", layout="wide")

st.title("📊 Monitoring Dashboard")
st.caption("Aggregate view across all patient screenings performed with this agent.")

stats = summary_stats()
df = load_logs()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Screenings", stats["total_patients_screened"])
col2.metric("High Risk", stats["high_risk_count"])
col3.metric("Moderate Risk", stats["moderate_risk_count"])
col4.metric("Low Risk", stats["low_risk_count"])
col5.metric("Avg Risk %", f"{stats['avg_risk_percent']}%")

st.markdown("---")

if df.empty:
    st.info("No screenings logged yet. Go to the main page to run a prediction.")
    st.stop()

# ---------- Risk distribution ----------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Risk Level Distribution")
    risk_counts = df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]
    color_map = {"High Risk": "#D64545", "Moderate Risk": "#E0A93E", "Low Risk": "#3E9E5E"}
    fig = px.pie(risk_counts, names="Risk Level", values="Count",
                 color="Risk Level", color_discrete_map=color_map, hole=0.45)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Screenings Over Time")
    df_time = df.copy()
    df_time["date"] = df_time["timestamp"].dt.date
    daily = df_time.groupby("date").size().reset_index(name="count")
    fig2 = px.line(daily, x="date", y="count", markers=True)
    fig2.update_layout(xaxis_title="Date", yaxis_title="Screenings")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

c3, c4 = st.columns(2)

with c3:
    st.subheader("Age vs Risk Probability")
    fig3 = px.scatter(
        df, x="age", y="risk_percent", color="risk_level",
        color_discrete_map={"High Risk": "#D64545", "Moderate Risk": "#E0A93E", "Low Risk": "#3E9E5E"},
        hover_data=["patient_name"],
        labels={"age": "Age", "risk_percent": "Risk %"}
    )
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    st.subheader("Average Clinical Values by Risk Level")
    numeric_cols = ["trestbps", "chol", "thalach", "oldpeak"]
    avg_df = df.groupby("risk_level")[numeric_cols].mean().reset_index()
    avg_df_melted = avg_df.melt(id_vars="risk_level", var_name="metric", value_name="value")
    fig4 = px.bar(avg_df_melted, x="metric", y="value", color="risk_level", barmode="group",
                  color_discrete_map={"High Risk": "#D64545", "Moderate Risk": "#E0A93E", "Low Risk": "#3E9E5E"})
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")
st.subheader("Recent Screenings")
recent = df.sort_values("timestamp", ascending=False).head(10)
st.dataframe(
    recent[["timestamp", "patient_name", "age", "sex", "risk_level", "risk_percent"]],
    use_container_width=True,
    hide_index=True,
)
