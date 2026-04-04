"""
train.py
Production-ready training pipeline
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from src.models.evaluate import evaluate_model, plot_roc_curves, plot_pr_curves
from src.models.hyperparameter_tuning import tune_xgboost
from src.models.shap_evaluate import run_shap_analysis
from src.features.preprocessing_pipeline import get_pipeline

# Import config
from src.config import (
    TRAIN_PATH, TEST_PATH, MODEL_PATH, PREPROCESSOR_PATH,
    FIG_DIR, MODEL_DIR, REPORT_DIR,
    DEBUG, DEBUG_SAMPLE_SIZE
)


def load_data():
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


def preprocess_inputs(train, test):
    print("Preparing inputs...")

    X_train = train.drop("Churn", axis=1)
    y_train = train["Churn"].map({"Yes": 1, "No": 0})

    X_test = test.drop("Churn", axis=1)
    y_test = test["Churn"].map({"Yes": 1, "No": 0})

    # Drop problematic columns (IMPORTANT FIX)
    drop_cols = ["signup_date"]
    X_train = X_train.drop(columns=drop_cols, errors="ignore")
    X_test = X_test.drop(columns=drop_cols, errors="ignore")

    # Debug mode
    if DEBUG:
        print(f"Debug mode ON ({DEBUG_SAMPLE_SIZE} samples)")
        X_train = X_train.sample(DEBUG_SAMPLE_SIZE, random_state=42)
        y_train = y_train.loc[X_train.index]

    return X_train, X_test, y_train, y_test


def load_preprocessor(X_train):
    if PREPROCESSOR_PATH.exists():
        print("Loading preprocessing pipeline...")
        return joblib.load(PREPROCESSOR_PATH)

    print("Preprocessing pipeline not found. Building a new one...")
    use_woe = os.getenv("USE_WOE", "0") == "1"

    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    preprocessor = get_pipeline(numeric_features, categorical_features, use_woe=use_woe)

    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"Saved new preprocessing pipeline to: {PREPROCESSOR_PATH}")
    return preprocessor


def build_pipeline(preprocessor, model):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(sampling_strategy=0.5, random_state=42)),
        ("classifier", model)
    ])


def train_baseline_models(preprocessor, X_train, y_train, X_test, y_test):
    print("\nTraining baseline models...")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        pipe = build_pipeline(preprocessor, model)
        pipe.fit(X_train, y_train)

        metrics, _ = evaluate_model(name, pipe, X_test, y_test, FIG_DIR)
        results[name] = metrics
        trained_models[name] = pipe

        joblib.dump(pipe, MODEL_DIR / f"{name.replace(' ', '_').lower()}.pkl")

    return results, trained_models


def run_hypertuning(preprocessor, X_train, y_train, X_test, y_test, results):
    print("\nRunning XGBoost hyperparameter tuning...")

    xgb_pipe = build_pipeline(
        preprocessor,
        XGBClassifier(eval_metric="logloss", random_state=42)
    )

    search = tune_xgboost(xgb_pipe, X_train, y_train)
    best_model = search.best_estimator_

    print("Best Params:", search.best_params_)
    print("Best Score:", search.best_score_)

    # Save tuning results
    with open(REPORT_DIR / "hypertuning_results.json", "w") as f:
        json.dump({
            "best_params": search.best_params_,
            "best_score": float(search.best_score_)
        }, f, indent=4)

    # Evaluate tuned model
    metrics, _ = evaluate_model("XGBoost_Tuned", best_model, X_test, y_test, FIG_DIR)
    results["XGBoost_Tuned"] = metrics

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    return best_model, results


def run_threshold_analysis(best_model, X_test, y_test):
    print("\nRunning Threshold Analysis...")

    probs = best_model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.05)

    precision_list, recall_list, f1_list = [], [], []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        precision_list.append(precision_score(y_test, preds))
        recall_list.append(recall_score(y_test, preds))
        f1_list.append(f1_score(y_test, preds))

    best_idx = np.argmax(f1_list)
    best_threshold = thresholds[best_idx]

    print(f"Best Threshold: {best_threshold:.2f}")

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(thresholds, precision_list, label="Precision")
    plt.plot(thresholds, recall_list, label="Recall")
    plt.plot(thresholds, f1_list, label="F1")

    plt.legend()
    plt.title("Threshold Optimization")
    plt.savefig(FIG_DIR / "threshold_analysis.png")
    plt.close()


def save_results(results):
    print("\nSaving model comparison...")
    df = pd.DataFrame(results).T
    df.to_csv(REPORT_DIR / "model_comparison.csv")
    print(df)


def main():
    train, test = load_data()

    X_train, X_test, y_train, y_test = preprocess_inputs(train, test)

    preprocessor = load_preprocessor(X_train)

    results, trained_models = train_baseline_models(
        preprocessor, X_train, y_train, X_test, y_test
    )

    plot_roc_curves(
        {name: (trained_models[name], X_test) for name in trained_models},
        y_test,
        FIG_DIR / "06_roc_curves.png"
    )

    plot_pr_curves(
        {name: (trained_models[name], X_test) for name in trained_models},
        y_test,
        FIG_DIR / "07_pr_curves.png"
    )

    best_model, results = run_hypertuning(
        preprocessor, X_train, y_train, X_test, y_test, results
    )

    run_threshold_analysis(best_model, X_test, y_test)

    print("\nRunning SHAP analysis...")
    run_shap_analysis(best_model, X_test.sample(500), FIG_DIR)

    save_results(results)

    print("\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    main()


