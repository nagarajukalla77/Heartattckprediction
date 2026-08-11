"""
utils/predictor.py
"""

import os
import pickle
import pandas as pd

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "heart_model.pkl"
)

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Serum Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG",
    "thalach": "Maximum Heart Rate",
    "exang": "Exercise Induced Angina",
    "oldpeak": "ST Depression",
    "slope": "Slope",
    "ca": "Major Vessels",
    "thal": "Thalassemia",
}


class HeartRiskPredictor:

    def __init__(self, model_path=MODEL_PATH):

        self.model_path = model_path
        self.bundle = None
        self.model = None
        self.scaler = None
        self.feature_columns = FEATURE_COLUMNS
        self.model_name = None
        self.metrics = None
        self.feature_importance = {}

        self._load()

    def _load(self):

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        with open(self.model_path, "rb") as f:
            self.bundle = pickle.load(f)

        self.model = self.bundle["model"]
        self.scaler = self.bundle["scaler"]

        self.feature_columns = self.bundle.get(
            "feature_columns",
            FEATURE_COLUMNS
        )

        self.model_name = self.bundle.get(
            "model_name",
            "Heart Prediction Model"
        )

        self.metrics = self.bundle.get(
            "metrics",
            {}
        )

        self.feature_importance = self.bundle.get(
            "feature_importance",
            {}
        )

    def _prepare_input(self, patient):

        row = {
            col: patient.get(col)
            for col in self.feature_columns
        }

        missing = [
            k for k, v in row.items()
            if v is None
        ]

        if missing:
            raise ValueError(
                f"Missing fields: {missing}"
            )

        return pd.DataFrame(
            [row],
            columns=self.feature_columns
        )

    def predict(self, patient):

        df = self._prepare_input(patient)

        X_scaled = self.scaler.transform(df)

        pred = int(
            self.model.predict(X_scaled)[0]
        )

        proba = float(
            self.model.predict_proba(X_scaled)[0][1]
        )

        risk_level, risk_color = self._risk_level(
            proba,
            patient
        )

        return {
            "prediction": pred,
            "probability": round(proba, 4),
            "risk_percent": round(proba * 100, 1),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "top_factors": self._top_factors(patient),
        }

    def predict_batch(self, df_patients):

        X = df_patients[self.feature_columns]

        X_scaled = self.scaler.transform(X)

        preds = self.model.predict(X_scaled)

        probas = self.model.predict_proba(X_scaled)[:, 1]

        output = df_patients.copy()

        output["prediction"] = preds
        output["risk_percent"] = (probas * 100).round(1)

        output["risk_level"] = [
            self._risk_level(
                p,
                row.to_dict()
            )[0]
            for p, (_, row)
            in zip(probas, df_patients.iterrows())
        ]

        return output

    def _risk_level(self, proba, patient):

        age = patient.get("age", 0)
        chol = patient.get("chol", 0)
        bp = patient.get("trestbps", 0)
        heart_rate = patient.get("thalach", 0)
        exang = patient.get("exang", 0)

        # CRITICAL CONDITIONS

        if chol >= 500:
            return "CRITICAL RISK", "darkred"

        if bp >= 180:
            return "CRITICAL RISK", "darkred"

        if age >= 75 and proba >= 0.60:
            return "CRITICAL RISK", "darkred"

        if heart_rate < 90 and age > 60:
            return "CRITICAL RISK", "darkred"

        # HIGH RISK

        if proba >= 0.80:
            return "HIGH RISK", "red"

        if chol >= 350:
            return "HIGH RISK", "red"

        if bp >= 160:
            return "HIGH RISK", "red"

        if exang == 1:
            return "HIGH RISK", "red"

        # MODERATE RISK

        if proba >= 0.50:
            return "MODERATE RISK", "orange"

        if chol >= 240:
            return "MODERATE RISK", "orange"

        if bp >= 140:
            return "MODERATE RISK", "orange"

        # LOW RISK

        return "LOW RISK", "green"

    def _top_factors(self, patient, n=3):

        if not self.feature_importance:
            return []

        ranked = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        factors = []

        for feat, importance in ranked[:n]:

            factors.append(
                {
                    "feature": FEATURE_LABELS.get(
                        feat,
                        feat
                    ),
                    "value": patient.get(feat),
                    "importance": round(
                        importance,
                        4
                    ),
                }
            )

        return factors


_predictor_instance = None


def get_predictor():

    global _predictor_instance

    if _predictor_instance is None:
        _predictor_instance = HeartRiskPredictor()

    return _predictor_instance