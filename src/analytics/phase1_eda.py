from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
INSIGHTS_PATH = PROJECT_ROOT / "reports" / "phase1_insights.md"
DATA_CANDIDATES = (
    PROJECT_ROOT / "data" / "raw" / "telecom_churn_20000.csv",
    PROJECT_ROOT / "data" / "raw" / "telecom_churn.csv",
)
REQUIRED_COLUMNS = {
    "signup_date",
    "Churn",
    "tenure",
    "number_of_logins",
    "complaints_count",
    "engagement_score",
    "Contract",
}


def pick_data_path(candidates: Iterable[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(f"Could not find dataset in:\n{checked}")


def validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required columns: {missing_text}")


def save_target_plot(df: pd.DataFrame) -> float:
    churn_rate = df["Churn"].eq("Yes").mean() * 100
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Churn", data=df, hue="Churn", legend=False, palette=["#4E79A7", "#E15759"])
    plt.title("Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase1_churn_distribution.png", dpi=150)
    plt.close()
    return churn_rate


def save_behavioral_plots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(x="Churn", y="number_of_logins", data=df, ax=axes[0], hue="Churn", legend=False, palette=["#4E79A7", "#E15759"])
    axes[0].set_title("Number of Logins vs Churn")
    sns.boxplot(x="Churn", y="complaints_count", data=df, ax=axes[1], hue="Churn", legend=False, palette=["#4E79A7", "#E15759"])
    axes[1].set_title("Complaints Count vs Churn")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase1_behavioral_analysis.png", dpi=150)
    plt.close(fig)


def save_engagement_plot(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Churn", y="engagement_score", data=df, hue="Churn", legend=False, palette=["#4E79A7", "#E15759"])
    plt.title("Engagement Score vs Churn")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase1_engagement_analysis.png", dpi=150)
    plt.close()


def save_kaplan_meier_plot(df: pd.DataFrame) -> None:
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df["tenure"], event_observed=df["ChurnBinary"])
    ax = kmf.plot(ci_show=True, color="#4E79A7", figsize=(7, 5))
    ax.set_title("Kaplan-Meier Survival Curve")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Survival Probability")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase1_kaplan_meier.png", dpi=150)
    plt.close()


def build_insight_text(df: pd.DataFrame, churn_rate: float) -> str:
    login_stats = df.groupby("Churn")["number_of_logins"].median()
    complaint_stats = df.groupby("Churn")["complaints_count"].median()
    engagement_stats = df.groupby("Churn")["engagement_score"].median()
    contract_churn = (
        df.groupby("Contract")["Churn"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )

    kmf = KaplanMeierFitter()
    kmf.fit(durations=df["tenure"], event_observed=df["ChurnBinary"])
    survival_12m = float(kmf.predict(12)) * 100

    top_contract = contract_churn.index[0]
    top_contract_rate = contract_churn.iloc[0]

    return (
        "# Phase 1 EDA Insights\n\n"
        f"- Dataset size: {len(df):,} rows\n"
        f"- Churn rate: {churn_rate:.2f}% (moderate class imbalance)\n"
        f"- Median logins: churned={login_stats['Yes']:.1f}, retained={login_stats['No']:.1f}\n"
        f"- Median complaints: churned={complaint_stats['Yes']:.1f}, retained={complaint_stats['No']:.1f}\n"
        f"- Median engagement: churned={engagement_stats['Yes']:.1f}, retained={engagement_stats['No']:.1f}\n"
        f"- Highest contract risk: {top_contract} ({top_contract_rate:.2f}% churn)\n"
        f"- Kaplan-Meier survival at 12 months: {survival_12m:.2f}%\n\n"
        "Story: Customers with low engagement, high complaints, and flexible contracts are most likely to churn, "
        "especially in early lifecycle stages.\n"
    )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data_path = pick_data_path(DATA_CANDIDATES)
    df = pd.read_csv(data_path)

    validate_columns(df)
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["ChurnBinary"] = df["Churn"].map({"Yes": 1, "No": 0})

    print("\nStep 1-2: Data Loaded + signup_date converted")
    print(df.head())
    print("\nStep 3: Basic Info")
    print(df.info())
    print(df.describe(include="all"))

    churn_rate = save_target_plot(df)
    save_behavioral_plots(df)
    save_engagement_plot(df)
    save_kaplan_meier_plot(df)

    insights = build_insight_text(df, churn_rate)
    INSIGHTS_PATH.write_text(insights, encoding="utf-8")

    print("\nStep 4-6 completed. Artifacts:")
    print(f"- {FIGURES_DIR / 'phase1_churn_distribution.png'}")
    print(f"- {FIGURES_DIR / 'phase1_behavioral_analysis.png'}")
    print(f"- {FIGURES_DIR / 'phase1_engagement_analysis.png'}")
    print(f"- {FIGURES_DIR / 'phase1_kaplan_meier.png'}")
    print(f"- {INSIGHTS_PATH}")


if __name__ == "__main__":
    main()
