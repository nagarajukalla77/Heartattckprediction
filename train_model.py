"""
train_model.py
---------------
Trains and compares multiple machine learning models for
heart attack prediction and saves the best model bundle.

Models:
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- SVM
- KNN
- MLP

Run:
    python train_model.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier


# ---------------------------------------------------
# Paths
# ---------------------------------------------------

DATA_PATH = os.path.join("data", "heart.csv")

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "heart_model.pkl"
)

# ---------------------------------------------------
# Dataset Columns
# ---------------------------------------------------

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

TARGET_COLUMN = "target"


# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

def load_data():

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing = set(required) - set(df.columns)

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    return df


# ---------------------------------------------------
# Candidate Models
# ---------------------------------------------------

def build_candidates():

    return {

        "LogisticRegression":

        LogisticRegression(
            max_iter=9000,
            class_weight="balanced",
            random_state=42
        ),

        "RandomForest":

        RandomForestClassifier(
            n_estimators=1000,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42
        ),

        "GradientBoosting":

        GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            random_state=42
        ),

        "XGBoost":

        XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        ),

        "SVM":

        SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
            random_state=42
        ),

        "KNN":

        KNeighborsClassifier(
            n_neighbors=7,
            weights="distance"
        ),

        "MLP":

        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=3000,
            random_state=42
        )
    }


# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------

def evaluate(model, X_test, y_test):

    preds = model.predict(X_test)

    proba = model.predict_proba(X_test)[:, 1]

    return {

        "accuracy":
        round(
            accuracy_score(
                y_test,
                preds
            ),
            4
        ),

        "precision":
        round(
            precision_score(
                y_test,
                preds
            ),
            4
        ),

        "recall":
        round(
            recall_score(
                y_test,
                preds
            ),
            4
        ),

        "f1":
        round(
            f1_score(
                y_test,
                preds
            ),
            4
        ),

        "roc_auc":
        round(
            roc_auc_score(
                y_test,
                proba
            ),
            4
        )
    }


# ---------------------------------------------------
# Main Training
# ---------------------------------------------------

def main():

    print("\nLoading dataset...")

    df = load_data()

    X = df[FEATURE_COLUMNS]

    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    candidates = build_candidates()

    results = {}

    trained_models = {}

    print("\nTraining Models...\n")

    for name, model in candidates.items():

        print(f"Training {name}...")

        model.fit(
            X_train_scaled,
            y_train
        )

        metrics = evaluate(
            model,
            X_test_scaled,
            y_test
        )

        cv_scores = cross_val_score(
            model,
            X_train_scaled,
            y_train,
            cv=5,
            scoring="roc_auc"
        )

        metrics["cv_roc_auc_mean"] = round(
            cv_scores.mean(),
            4
        )

        results[name] = metrics

        trained_models[name] = model

        print(
            f"{name}: {metrics}"
        )

    best_name = max(
        results,
        key=lambda x:
        results[x]["roc_auc"]
    )

    best_model = trained_models[
        best_name
    ]

    print("\n===================================")
    print(f"BEST MODEL: {best_name}")
    print(results[best_name])
    print("===================================\n")

    preds = best_model.predict(
        X_test_scaled
    )

    print(
        classification_report(
            y_test,
            preds
        )
    )

    print(
        "Confusion Matrix:\n"
    )

    print(
        confusion_matrix(
            y_test,
            preds
        )
    )

    feature_importance = {}

    if hasattr(
        best_model,
        "feature_importances_"
    ):

        feature_importance = dict(
            zip(
                FEATURE_COLUMNS,
                best_model.feature_importances_.round(4)
            )
        )

    elif hasattr(
        best_model,
        "coef_"
    ):

        feature_importance = dict(
            zip(
                FEATURE_COLUMNS,
                np.abs(
                    best_model.coef_[0]
                ).round(4)
            )
        )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    bundle = {

        "model":
        best_model,

        "scaler":
        scaler,

        "feature_columns":
        FEATURE_COLUMNS,

        "model_name":
        best_name,

        "metrics":
        results[best_name],

        "all_results":
        results,

        "feature_importance":
        feature_importance
    }

    with open(
        MODEL_PATH,
        "wb"
    ) as f:

        pickle.dump(
            bundle,
            f
        )

    with open(
        os.path.join(
            MODEL_DIR,
            "metrics.json"
        ),
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print(
        f"\nModel saved to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()