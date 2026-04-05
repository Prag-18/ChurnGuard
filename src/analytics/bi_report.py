"""
bi_report.py
Generates business report
"""

import pandas as pd

from src.config import REPORT_DIR


def load_data():
    return pd.read_csv("data/processed/customers_scored.csv")


def generate_report(df):
    report = []

    report.append("# 📊 Phase 4 Business Report\n")

    # Executive summary
    report.append("## Executive Summary\n")
    report.append("- High churn risk concentrated in new customers\n")
    report.append("- High CLV customers at risk identified\n")

    # Segment summary
    report.append("\n## Segment Insights\n")

    segments = df.groupby("customer_segment").agg({
        "CLV": "mean",
        "churn_prob": "mean"
    })

    report.append(segments.to_markdown())

    # Revenue impact
    high_risk = df[df["churn_prob"] > 0.5]
    total_clv = high_risk["CLV"].sum()

    report.append("\n## Revenue Impact\n")
    report.append(f"Total CLV at risk: ${total_clv:,.2f}")

    return "\n".join(report)


def main():
    df = load_data()
    report = generate_report(df)

    # Create directory first
    report_dir = REPORT_DIR / "business_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Then save file
    with open(report_dir / "phase4_bi_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("✅ BI report generated!")


if __name__ == "__main__":
    main()