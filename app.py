"""
app.py
------
Main Streamlit application: Heart Attack Prediction Agent.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd

from utils.predictor import get_predictor, FEATURE_LABELS
from utils.logger import log_prediction
from utils.report_generator import generate_pdf_report

st.set_page_config(
    page_title="Heart Attack Prediction Agent",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
.risk-card {
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}
.risk-high { background-color: #FDEDED; border-left: 6px solid #D64545; }
.risk-moderate { background-color: #FFF6E5; border-left: 6px solid #E0A93E; }
.risk-low { background-color: #EAF7EE; border-left: 6px solid #3E9E5E; }
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
logo_path = os.path.join("assets", "logo.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=120)
st.sidebar.title("❤️ Heart Attack Prediction Agent")
st.sidebar.markdown(
    "Use the form to enter a patient's clinical values and get an "
    "instant ML-based heart attack risk assessment."
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Pages**\n"
    "- Dashboard — aggregate analytics\n"
    "- Patient History — past screenings\n"
    "- Reports — download PDF reports\n\n"
    "Use the page selector in the top-left sidebar navigation."
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ Educational tool only. Not a substitute for professional "
    "medical diagnosis. Always consult a physician."
)

# ---------- Load model ----------
try:
    predictor = get_predictor()
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Run `python train_model.py` in your terminal, then reload this page.")
    st.stop()

# ---------- Header ----------
st.title("Heart Attack Prediction Agent")
st.markdown("Enter the patient's clinical parameters below.")

# ---------- Input Form ----------
with st.form("patient_form"):
    patient_name = st.text_input("Patient Name", value="")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
        cp = st.selectbox(
            "Chest Pain Type", options=[
                ("Typical Angina (0)", 0), ("Atypical Angina (1)", 1),
                ("Non-anginal Pain (2)", 2), ("Asymptomatic (3)", 3)
            ], format_func=lambda x: x[0]
        )[1]
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=130)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

    with col2:
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=650, value=230)
        restecg = st.selectbox(
            "Resting ECG Results", options=[
                ("Normal (0)", 0), ("ST-T Abnormality (1)", 1), ("LV Hypertrophy (2)", 2)
            ], format_func=lambda x: x[0]
        )[1]
        thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

    with col3:
        oldpeak = st.number_input("ST Depression (Exercise vs Rest)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment", options=[
                ("Upsloping (0)", 0), ("Flat (1)", 1), ("Downsloping (2)", 2)
            ], format_func=lambda x: x[0]
        )[1]
        ca = st.selectbox("Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3, 4])
        thal = st.selectbox(
            "Thalassemia", options=[
                ("Normal (1)", 1), ("Fixed Defect (2)", 2), ("Reversible Defect (3)", 3)
            ], format_func=lambda x: x[0]
        )[1]

    notes = st.text_area("Clinician Notes (optional)", placeholder="Any additional observations...")

    submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)

# ---------- Prediction ----------
if submitted:
    patient = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }

    result = predictor.predict(patient)
    display_name = patient_name.strip() if patient_name.strip() else "Unnamed Patient"

    # Log automatically
    log_id = log_prediction(patient, result, patient_name=display_name, notes=notes)

    st.markdown("---")
    st.subheader("Prediction Result")

    risk_class = {
    "CRITICAL RISK": "risk-high",
    "HIGH RISK": "risk-high",
    "MODERATE RISK": "risk-moderate",
    "LOW RISK": "risk-low",
    "High Risk": "risk-high",
    "Moderate Risk": "risk-moderate",
    "Low Risk": "risk-low"
}.get(result["risk_level"], "risk-low")
    st.markdown(f"""
    <div class="risk-card {risk_class}">
        <h2 style="margin:0;">{result['risk_level']}</h2>
        <p style="font-size:1.1rem; margin:0.3rem 0 0 0;">
            Predicted probability of heart disease: <b>{result['risk_percent']}%</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        st.metric("Risk Probability", f"{result['risk_percent']}%")
       

    with colB:
        st.markdown("**Top Contributing Factors**")
        if result["top_factors"]:
            for f in result["top_factors"]:
                st.write(f"- **{f['feature']}**: {f['value']} (importance: {f['importance']})")
        else:
            st.write("Feature importance not available for this model.")

    st.info(
        "This prediction is generated by a machine learning model and is "
        "**not a medical diagnosis**. Please consult a qualified healthcare "
        "professional for clinical decisions."
    )

    # Report generation
    st.markdown("---")
    st.subheader("Generate Report")
    if st.button("📄 Generate PDF Report"):
        with st.spinner("Generating report..."):
            pdf_path = generate_pdf_report(patient, result, patient_name=display_name, notes=notes)
        with open(pdf_path, "rb") as f:
            st.download_button(
                "⬇️ Download PDF Report",
                data=f.read(),
                file_name=os.path.basename(pdf_path),
                mime="application/pdf",
            )
        st.success(f"Report generated and saved to {pdf_path}")

    st.caption(f"This screening was logged (ID: {log_id}). View it anytime in the Patient History page.")
