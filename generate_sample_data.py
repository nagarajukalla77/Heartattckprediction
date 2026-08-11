"""
generate_sample_data.py
------------------------
Creates a synthetic heart.csv with the same schema as the classic
UCI Heart Disease (Cleveland) dataset so the rest of the project
(train_model.py, app.py, etc.) runs immediately out of the box.

If you have the real UCI dataset (or any dataset with the same
column names), just drop it into data/heart.csv and skip this script.

Columns:
    age       - age in years
    sex       - 1 = male, 0 = female
    cp        - chest pain type (0-3)
    trestbps  - resting blood pressure (mm Hg)
    chol      - serum cholesterol (mg/dl)
    fbs       - fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    restecg   - resting ECG results (0-2)
    thalach   - max heart rate achieved
    exang     - exercise induced angina (1 = yes, 0 = no)
    oldpeak   - ST depression induced by exercise
    slope     - slope of peak exercise ST segment (0-2)
    ca        - number of major vessels colored by fluoroscopy (0-4)
    thal      - 1 = normal, 2 = fixed defect, 3 = reversible defect
    target    - 1 = heart disease / heart attack risk, 0 = healthy
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 1000

def generate():
    target = np.random.binomial(1, 0.48, N)

    age = np.where(target == 1,
                    np.random.normal(56, 8, N),
                    np.random.normal(50, 9, N)).clip(29, 77).round().astype(int)

    sex = np.random.binomial(1, 0.68, N)

    cp = np.where(target == 1,
                  np.random.choice([0, 1, 2, 3], N, p=[0.55, 0.2, 0.15, 0.10]),
                  np.random.choice([0, 1, 2, 3], N, p=[0.25, 0.25, 0.25, 0.25]))

    trestbps = np.where(target == 1,
                         np.random.normal(134, 18, N),
                         np.random.normal(128, 16, N)).clip(94, 200).round().astype(int)

    chol = np.where(target == 1,
                     np.random.normal(250, 50, N),
                     np.random.normal(235, 45, N)).clip(126, 564).round().astype(int)

    fbs = np.random.binomial(1, 0.15, N)

    restecg = np.random.choice([0, 1, 2], N, p=[0.5, 0.48, 0.02])

    thalach = np.where(target == 1,
                        np.random.normal(139, 22, N),
                        np.random.normal(158, 19, N)).clip(71, 202).round().astype(int)

    exang = np.where(target == 1,
                      np.random.binomial(1, 0.45, N),
                      np.random.binomial(1, 0.14, N))

    oldpeak = np.where(target == 1,
                        np.random.exponential(1.4, N),
                        np.random.exponential(0.5, N)).clip(0, 6.2).round(1)

    slope = np.where(target == 1,
                      np.random.choice([0, 1, 2], N, p=[0.15, 0.55, 0.30]),
                      np.random.choice([0, 1, 2], N, p=[0.07, 0.30, 0.63]))

    ca = np.where(target == 1,
                  np.random.choice([0, 1, 2, 3, 4], N, p=[0.35, 0.30, 0.20, 0.10, 0.05]),
                  np.random.choice([0, 1, 2, 3, 4], N, p=[0.70, 0.18, 0.08, 0.03, 0.01]))

    thal = np.where(target == 1,
                     np.random.choice([1, 2, 3], N, p=[0.15, 0.20, 0.65]),
                     np.random.choice([1, 2, 3], N, p=[0.45, 0.45, 0.10]))

    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca,
        "thal": thal, "target": target
    })
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate()
    out_path = os.path.join("data", "heart.csv")
    df.to_csv(out_path, index=False)
    print(f"Synthetic dataset with {len(df)} rows written to {out_path}")
    print(df["target"].value_counts())
