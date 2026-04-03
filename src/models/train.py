from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# 1. Data
from src.data.make_dataset import load_data

# 2. Features
from src.features.build_features import build_features
from src.features.preprocessing_pipeline import get_pipeline

# 3. Models
from src.models.evaluate import evaluate_model, plot_pr_curves, plot_roc_curves
from src.models.hyperparameter_tuning import tune_xgboost

# 4. Analytics
from src.analytics.shap_analysis import run_shap_analysis

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


BASE_DIR = Path(__file__).resolve().parents[2]
if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env")

DATA_PATH = BASE_DIR / os.getenv("DATA_PATH", "data/raw/telecom_churn.csv")
MODEL_PATH = BASE_DIR / os.getenv("MODEL_PATH", "models/best_model.pkl")
FIG_DIR = BASE_DIR / "reports" / "figures"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

for directory in (FIG_DIR, MODEL_DIR, REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def build_training_pipeline(preprocessor, model):
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("smote", SMOTE(sampling_strategy=0.5, random_state=42)),
            ("classifier", model),
        ]
    )


def load_training_data():
    print("Loading data...")
    df = load_data(DATA_PATH)

    if "Churn" not in df.columns:
        raise KeyError("Target column 'Churn' is missing from input data.")

    X = df.drop(columns=["Churn"], errors="ignore")
    y = df["Churn"].map({"Yes": 1, "No": 0})

    if y.isna().any():
        raise ValueError("Churn column must contain only 'Yes' and 'No'.")

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def run_threshold_analysis(best_model, X_test, y_test) -> None:
    print("Running threshold analysis...")
    probs = best_model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.10, 0.90, 0.05)
    precision_list = []
    recall_list = []
    f1_list = []

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        precision_list.append(precision_score(y_test, preds))
        recall_list.append(recall_score(y_test, preds))
        f1_list.append(f1_score(y_test, preds))

    best_idx = int(np.argmax(f1_list))
    best_threshold = float(thresholds[best_idx])

    print(f"Optimal threshold: {best_threshold:.2f}")
    print(f"Precision: {precision_list[best_idx]:.3f}")
    print(f"Recall: {recall_list[best_idx]:.3f}")
    print(f"F1 Score: {f1_list[best_idx]:.3f}")

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precision_list, label="Precision")
    plt.plot(thresholds, recall_list, label="Recall")
    plt.plot(thresholds, f1_list, label="F1 Score")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Optimization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "threshold_analysis.png", dpi=150)
    plt.close()


def main() -> None:
    X_train, X_test, y_train, y_test = load_training_data()

    # Drop raw date column before training to avoid expensive downstream handling.
    X_train = X_train.drop(columns=["signup_date"], errors="ignore")
    X_test = X_test.drop(columns=["signup_date"], errors="ignore")

    debug_sample_size = int(os.getenv("DEBUG_SAMPLE_SIZE", "5000"))
    if debug_sample_size > 0 and len(X_train) > debug_sample_size:
        X_train = X_train.sample(debug_sample_size, random_state=42)
        y_train = y_train.loc[X_train.index]
        print(f"Debug mode: training on {len(X_train)} sampled rows.")

    X_train = build_features(X_train)
    X_test = build_features(X_test)

    use_woe = os.getenv("USE_WOE", "0") == "1"
    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    preprocessor = get_pipeline(numeric_features, categorical_features, use_woe=use_woe)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    }

    results = {}
    trained_models = {}

    print("Training baseline models...")
    for name, estimator in models.items():
        print(f"Training {name}...")
        pipeline = build_training_pipeline(preprocessor, estimator)
        pipeline.fit(X_train, y_train)

        metrics, _ = evaluate_model(name, pipeline, X_test, y_test, FIG_DIR)
        results[name] = metrics
        trained_models[name] = pipeline

        model_filename = name.replace(" ", "_").lower() + ".pkl"
        joblib.dump(pipeline, MODEL_DIR / model_filename)

    print("Generating ROC and PR curves...")
    plot_roc_curves(
        {name: (trained_models[name], X_test) for name in trained_models},
        y_test,
        FIG_DIR / "06_roc_curves.png",
    )
    plot_pr_curves(
        {name: (trained_models[name], X_test) for name in trained_models},
        y_test,
        FIG_DIR / "07_pr_curves.png",
    )

    print("Tuning XGBoost...")
    xgb_pipe = build_training_pipeline(
        preprocessor,
        XGBClassifier(eval_metric="logloss", random_state=42),
    )
    search = tune_xgboost(xgb_pipe, X_train, y_train)
    best_model = search.best_estimator_

    print("Best Parameters:")
    print(search.best_params_)
    print(f"Best CV Score: {search.best_score_}")

    with (REPORT_DIR / "hypertuning_results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_params": search.best_params_,
                "best_score": float(search.best_score_),
            },
            f,
            indent=4,
        )

    metrics, _ = evaluate_model("XGBoost_Tuned", best_model, X_test, y_test, FIG_DIR)
    results["XGBoost_Tuned"] = metrics

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    run_threshold_analysis(best_model, X_test, y_test)

    print("Running SHAP analysis...")
    sample_size = min(500, len(X_test))
    run_shap_analysis(best_model, X_test.sample(sample_size, random_state=42), FIG_DIR)

    print("Creating model comparison table...")
    df_results = pd.DataFrame(results).T
    df_results.to_csv(REPORT_DIR / "model_comparison.csv")
    print(df_results)

    print("Training completed successfully.")


if __name__ == "__main__":
    main()
