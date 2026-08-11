"""
pages/Patient_History.py
-------------------------
Browse, search, and manage past patient screening records.
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.logger import load_logs, delete_log, get_patient_history

st.set_page_config(page_title="Patient History | Heart Attack Prediction Agent", page_icon="🗂️", layout="wide")

st.title("🗂️ Patient History")
st.caption("Search and review all previously logged screenings.")

df = load_logs()

if df.empty:
    st.info("No screenings logged yet. Go to the main page to run a prediction.")
    st.stop()

# ---------- Search / filter ----------
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    search_name = st.text_input("🔎 Search by patient name")
with col2:
    risk_filter = st.multiselect("Risk Level", options=["High Risk", "Moderate Risk", "Low Risk"])
with col3:
    sort_order = st.selectbox("Sort by", options=["Most Recent", "Oldest", "Highest Risk", "Lowest Risk"])

filtered = df.copy()
if search_name:
    filtered = filtered[filtered["patient_name"].str.contains(search_name, case=False, na=False)]
if risk_filter:
    filtered = filtered[filtered["risk_level"].isin(risk_filter)]

if sort_order == "Most Recent":
    filtered = filtered.sort_values("timestamp", ascending=False)
elif sort_order == "Oldest":
    filtered = filtered.sort_values("timestamp", ascending=True)
elif sort_order == "Highest Risk":
    filtered = filtered.sort_values("risk_percent", ascending=False)
else:
    filtered = filtered.sort_values("risk_percent", ascending=True)

st.markdown(f"**{len(filtered)}** record(s) found")

st.dataframe(
    filtered[[
    "log_id",
    "timestamp",
    "patient_name",
    "age",
    "sex",
    "risk_level",
    "risk_percent",
    "model_name"
    ]],
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# ---------- Individual patient trend ----------
st.subheader("Patient Trend Lookup")
unique_names = sorted(df["patient_name"].dropna().unique().tolist())
selected_patient = st.selectbox("Select a patient to view their history", options=["-- Select --"] + unique_names)

if selected_patient != "-- Select --":
    history = get_patient_history(selected_patient)
    st.write(f"### History for {selected_patient}")
    st.dataframe(
        history[["timestamp", "age", "trestbps", "chol", "thalach", "risk_level", "risk_percent"]],
        use_container_width=True,
        hide_index=True,
    )
    if len(history) > 1:
        st.line_chart(history.sort_values("timestamp").set_index("timestamp")["risk_percent"])

st.markdown("---")

# ---------- Delete records ----------
with st.expander("🗑️ Delete a record"):
    log_id_to_delete = st.text_input("Enter Log ID to delete")
    if st.button("Delete Record"):
        if log_id_to_delete:
            success = delete_log(log_id_to_delete.strip())
            if success:
                st.success(f"Deleted record {log_id_to_delete}. Refresh to see changes.")
            else:
                st.error("Log ID not found.")
        else:
            st.warning("Please enter a Log ID.")

# ---------- Export ----------
st.download_button(
    "⬇️ Export all filtered records as CSV",
    data=filtered.to_csv(index=False),
    file_name="patient_history_export.csv",
    mime="text/csv",
)
