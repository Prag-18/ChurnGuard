"""
segmentation.py
Customer segmentation + CLV analysis
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.config import (
    TRAIN_PATH, MODEL_PATH,
    FIG_DIR, MODEL_DIR, REPORT_DIR
)


# -----------------------------
# Load Data
# -----------------------------
def load_data():
    print("📥 Loading data...")
    df = pd.read_csv(TRAIN_PATH)
    return df


# -----------------------------
# Feature Selection
# -----------------------------
def get_features(df):
    features = [
        "tenure", "MonthlyCharges", "TotalCharges",
        "avg_monthly_usage", "engagement_score",
        "complaints_count", "customer_support_calls"
    ]
    return df[features]


# -----------------------------
# Elbow Method
# -----------------------------
def elbow_method(X_scaled):
    inertia = []
    K = range(2, 11)

    for k in K:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    plt.figure(figsize=(10,6))
    plt.plot(K, inertia, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_elbow_kmeans.png", dpi=150)
    plt.close()


# -----------------------------
# Silhouette Analysis
# -----------------------------
def silhouette_analysis(X_scaled):
    scores = []
    K = range(2, 9)

    for k in K:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)
        print(f"K={k}, Silhouette Score={score:.3f}")

    plt.figure(figsize=(10,6))
    plt.plot(K, scores, marker='o')
    plt.title("Silhouette Scores")
    plt.xlabel("K")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_silhouette_scores.png", dpi=150)
    plt.close()


# -----------------------------
# Fit Final Model
# -----------------------------
def fit_kmeans(X_scaled):
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


# -----------------------------
# Assign Cluster Names
# -----------------------------
def assign_cluster_names(df):
    profile = df.groupby("cluster").mean(numeric_only=True)

    names = {}

    for i, row in profile.iterrows():
        if row["tenure"] < 12:
            names[i] = "New & Unsettled"
        elif row["complaints_count"] > 5:
            names[i] = "At-risk Premium"
        elif row["tenure"] > 48:
            names[i] = "Veteran Champion"
        else:
            names[i] = "Loyal Mid-tier"

    df["customer_segment"] = df["cluster"].map(names)
    return df, names


# -----------------------------
# PCA Visualization
# -----------------------------
def plot_pca(X_scaled, labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="viridis", alpha=0.6)
    plt.title("Customer Segments (PCA)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_cluster_pca.png", dpi=150)
    plt.close()


# -----------------------------
# CLV Calculation
# -----------------------------
def calculate_clv(df):
    print("💰 Calculating CLV...")

    model = joblib.load(MODEL_PATH)

    X = df.drop("Churn", axis=1)
    probs = model.predict_proba(X)[:,1]

    df["churn_prob"] = probs
    df["expected_remaining"] = np.maximum(0, 72 - df["tenure"])

    df["CLV"] = df["MonthlyCharges"] * df["expected_remaining"] * (1 - df["churn_prob"])

    return df


# -----------------------------
# Save Outputs
# -----------------------------
def save_outputs(df, km, scaler, labels):
    print("💾 Saving outputs...")

    joblib.dump(
        {"model": km, "scaler": scaler, "labels": labels},
        MODEL_DIR / "kmeans_segmentation.pkl"
    )

    df.to_csv("data/processed/customers_scored.csv", index=False)


# -----------------------------
# MAIN
# -----------------------------
def main():
    df = load_data()

    X = get_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    elbow_method(X_scaled)
    silhouette_analysis(X_scaled)

    km, labels = fit_kmeans(X_scaled)
    df["cluster"] = labels

    df, cluster_names = assign_cluster_names(df)

    plot_pca(X_scaled, labels)

    df = calculate_clv(df)

    save_outputs(df, km, scaler, cluster_names)

    print("\n✅ Segmentation + CLV completed!")


if __name__ == "__main__":
    main()