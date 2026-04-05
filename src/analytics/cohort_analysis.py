"""
shap_advanced.py
Advanced SHAP analysis
"""

import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np

from src.config import TEST_PATH, MODEL_PATH, FIG_DIR


def load_data():
    df = pd.read_csv(TEST_PATH)
    return df


def load_model():
    return joblib.load(MODEL_PATH)


def preprocess(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})
    return X, y


def run_global_shap(model, X):
    explainer = shap.Explainer(model.named_steps["classifier"])
    shap_values = explainer(X)

    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_shap_global_full.png", dpi=150)
    plt.close()

    return shap_values


def dependence_plots(shap_values, X):
    shap.dependence_plot("tenure", shap_values.values, X, show=False)
    plt.savefig(FIG_DIR / "07_shap_dep_tenure.png", dpi=150)
    plt.close()

    shap.dependence_plot("MonthlyCharges", shap_values.values, X, show=False)
    plt.savefig(FIG_DIR / "07_shap_dep_charges.png", dpi=150)
    plt.close()

    shap.dependence_plot("engagement_score", shap_values.values, X, show=False)
    plt.savefig(FIG_DIR / "07_shap_dep_engagement.png", dpi=150)
    plt.close()


def interaction_plot(model, X):
    explainer = shap.Explainer(model.named_steps["classifier"])
    shap_values = explainer(X)

    shap.dependence_plot(
        ("tenure", "MonthlyCharges"),
        shap_values.values,
        X,
        show=False
    )

    plt.savefig(FIG_DIR / "08_shap_interaction.png", dpi=150)
    plt.close()


def segment_level_shap(df, shap_values):
    df["cluster"] = pd.qcut(df["tenure"], 4, labels=False)

    for cluster in df["cluster"].unique():
        subset = shap_values.values[df["cluster"] == cluster]
        mean_vals = np.abs(subset).mean(axis=0)

        top_idx = np.argsort(mean_vals)[-3:]

        print(f"\nCluster {cluster} Top Drivers:")
        for i in top_idx:
            print(f"- Feature {i}")


def main():
    df = load_data()
    model = load_model()

    X, y = preprocess(df)

    dependence_plots(shap_values, X)

    interaction_plot(model, X)

    segment_level_shap(df, shap_values)

    print("✅ Advanced SHAP completed!")


if __name__ == "__main__":
    main()