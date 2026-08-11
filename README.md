# ❤️ Heart Attack Prediction Agent

A Streamlit-based AI agent that predicts heart attack / heart disease risk
from clinical parameters, logs every screening, and generates downloadable
PDF patient reports.

## Features
- **Risk prediction** — enter a patient's clinical values and get an instant
  ML-based risk score (Low / Moderate / High) with top contributing factors.
- **Model comparison + training** — `train_model.py` trains and compares
  RandomForest, GradientBoosting, and Logistic Regression, and saves the best one.
- **Monitoring logs** — every prediction is automatically appended to
  `logs/monitoring_logs.csv`.
- **Patient history** — search, filter, and trend past screenings per patient.
- **PDF reports** — generate a professional PDF report per screening,
  tracked in `reports/patient_reports.csv`.
- **Dashboard** — aggregate charts: risk distribution, screenings over time,
  age vs. risk, average clinical values by risk level.

## Project Structure
```
Heart Attack Prediction Agent/
├── app.py                     # Main Streamlit app (prediction form)
├── data/heart.csv             # Training dataset
├── models/heart_model.pkl     # Trained model bundle (model + scaler)
├── logs/monitoring_logs.csv   # Every screening ever run
├── reports/patient_reports.csv# Index of generated PDF reports
├── utils/
│   ├── predictor.py           # Model loading + inference
│   ├── report_generator.py    # PDF report creation
│   └── logger.py              # CSV log read/write helpers
├── pages/
│   ├── Dashboard.py           # Analytics dashboard
│   ├── Patient_History.py     # Search/browse past screenings
│   └── Reports.py             # Browse/download PDF reports
├── assets/logo.png            # Sidebar logo
├── generate_sample_data.py    # Creates a synthetic heart.csv if you don't have one
├── train_model.py             # Trains & saves the model
└── requirements.txt
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Get a dataset
# Option A: use the included synthetic dataset generator
python generate_sample_data.py

# Option B: use the real UCI Heart Disease dataset
# Download it and save as data/heart.csv with these columns:
# age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
# oldpeak, slope, ca, thal, target

# 3. Train the model
python train_model.py

# 4. Launch the app
streamlit run app.py
```

The app opens with the prediction form. Use the sidebar page navigator to
switch to Dashboard, Patient History, or Reports.

## Dataset Columns Reference

| Column   | Meaning |
|----------|---------|
| age      | Age in years |
| sex      | 1 = male, 0 = female |
| cp       | Chest pain type (0-3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol     | Serum cholesterol (mg/dl) |
| fbs      | Fasting blood sugar > 120 mg/dl (1/0) |
| restecg  | Resting ECG results (0-2) |
| thalach  | Max heart rate achieved |
| exang    | Exercise induced angina (1/0) |
| oldpeak  | ST depression induced by exercise |
| slope    | Slope of peak exercise ST segment (0-2) |
| ca       | Major vessels colored by fluoroscopy (0-4) |
| thal     | 1 = normal, 2 = fixed defect, 3 = reversible defect |
| target   | 1 = disease present, 0 = no disease (training label only) |

## Retraining with your own data
Replace `data/heart.csv` with your own dataset using the same column names,
then re-run `python train_model.py`. It automatically compares three model
types and keeps whichever scores best on ROC-AUC.

## Disclaimer
This tool is for **educational and informational purposes only**. It is not
a medical device and does not provide medical diagnoses. Always consult a
qualified healthcare professional for medical decisions.
