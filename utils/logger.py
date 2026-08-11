"""
utils/logger.py
----------------
Handles reading/writing the patient monitoring log (logs/monitoring_logs.csv).
Every prediction made in the app gets appended here so it can be reviewed
later in the Patient History page or exported.
"""

import os
import uuid
import pandas as pd
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_PATH = os.path.join(LOG_DIR, "monitoring_logs.csv")

LOG_COLUMNS = [
    "log_id", "timestamp", "patient_name",
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
    "prediction", "risk_percent", "risk_level", "model_name", "notes",
]


def _ensure_log_file():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_PATH):
        pd.DataFrame(columns=LOG_COLUMNS).to_csv(LOG_PATH, index=False)


def log_prediction(patient: dict, result: dict, patient_name: str = "Unnamed", notes: str = "") -> str:
    """Append a single prediction event to the monitoring log. Returns the log_id."""
    _ensure_log_file()

    log_id = str(uuid.uuid4())[:8]
    entry = {
        "log_id": log_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_name": patient_name,
        **{col: patient.get(col) for col in [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]},
        "prediction": result.get("prediction"),
        "risk_percent": result.get("risk_percent"),
        "risk_level": result.get("risk_level"),
        "model_name": result.get("model_name"),
        "notes": notes,
    }

    df = pd.DataFrame([entry])
    df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    return log_id


def load_logs() -> pd.DataFrame:
    """Return the full monitoring log as a DataFrame (empty if none yet)."""
    _ensure_log_file()
    df = pd.read_csv(LOG_PATH)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def get_patient_history(patient_name: str) -> pd.DataFrame:
    """Return all logged records for a specific patient name, most recent first."""
    df = load_logs()
    if df.empty:
        return df
    history = df[df["patient_name"].str.lower() == patient_name.lower()]
    return history.sort_values("timestamp", ascending=False)


def delete_log(log_id: str) -> bool:
    """Remove a single log entry by id. Returns True if a row was deleted."""
    df = load_logs()
    if df.empty or log_id not in df["log_id"].values:
        return False
    df = df[df["log_id"] != log_id]
    df.to_csv(LOG_PATH, index=False)
    return True


def clear_logs():
    """Wipe all monitoring logs (keeps header)."""
    _ensure_log_file()
    pd.DataFrame(columns=LOG_COLUMNS).to_csv(LOG_PATH, index=False)


def summary_stats() -> dict:
    """Quick aggregate stats for the dashboard."""
    df = load_logs()
    if df.empty:
        return {
            "total_patients_screened": 0, "high_risk_count": 0,
            "moderate_risk_count": 0, "low_risk_count": 0, "avg_risk_percent": 0,
        }
    return {
        "total_patients_screened": len(df),
        "high_risk_count": int((df["risk_level"] == "High Risk").sum()),
        "moderate_risk_count": int((df["risk_level"] == "Moderate Risk").sum()),
        "low_risk_count": int((df["risk_level"] == "Low Risk").sum()),
        "avg_risk_percent": round(df["risk_percent"].mean(), 1),
    }
