import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['figure.dpi'] = 150

def run_cohort_analysis():
    """Analyze customer churn behavior across different signup cohorts and tenure stages."""
    print("\n--- STEP 4: COHORT ANALYSIS ---")
    
    # Path
    scored_path = os.path.join('data', 'processed', 'customers_scored.csv')
    figures_dir = os.path.join('reports', 'figures')
    
    if not os.path.exists(scored_path):
        print("Error: customers_scored.csv not found.")
        return
        
    df = pd.read_csv(scored_path)
    
    # a) Create signup_quarter column
    print("Creating cohort columns from signup dates...")
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    # Format: "2021-Q3", "2022-Q1" etc.
    df['signup_quarter'] = df['signup_date'].dt.to_period('Q').astype(str)
    
    # Ensure tenure_group is categorical with correct order
    tenure_order = ['New', 'Regular', 'Loyal', 'Veteran']
    df['tenure_group'] = pd.Categorical(df['tenure_group'], categories=tenure_order, ordered=True)
    
    # b) Cohort churn rate heatmap
    print("Generating cohort churn rate heatmap (Quarter vs Tenure Stage)...")
    # Rows = signup_quarter, Columns = tenure_group, Values = churn rate (%)
    cohort_pivot = df.groupby(['signup_quarter', 'tenure_group'], observed=True)['Churn_Numeric'].mean().unstack() * 100
    
    plt.figure(figsize=(14, 9))
    sns.heatmap(
        cohort_pivot, 
        annot=True, 
        fmt=".1f", 
        cmap="YlOrRd", 
        cbar_kws={'label': 'Churn Rate (%)'},
        linewidths=.5
    )
    plt.title('Churn Rate Heatmap by Signup Cohort and Tenure Group', fontsize=14, fontweight='bold')
    plt.ylabel('Signup Quarter', fontsize=12)
    plt.xlabel('Tenure Group', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '10_cohort_heatmap.png'))
    plt.close()
    
    # c) Monthly churn trend line
    print("Calculating churn trend line over signup cohorts...")
    quarterly_churn = df.groupby('signup_quarter', observed=True)['Churn_Numeric'].mean().reset_index()
    quarterly_churn['Churn Rate (%)'] = quarterly_churn['Churn_Numeric'] * 100
    
    # Prepare for trend line (linear regression)
    quarterly_churn['time_idx'] = range(len(quarterly_churn))
    X = quarterly_churn[['time_idx']]
    y = quarterly_churn['Churn Rate (%)']
    model = LinearRegression()
    model.fit(X, y)
    quarterly_churn['trend'] = model.predict(X)
    
    plt.figure(figsize=(12, 7))
    sns.lineplot(
        x='signup_quarter', y='Churn Rate (%)', 
        data=quarterly_churn, marker='o', label='Cohort Churn Rate', 
        linewidth=2.5, markersize=10
    )
    sns.lineplot(
        x='signup_quarter', y='trend', 
        data=quarterly_churn, color='red', linestyle='--', 
        label=f'Trend (Slope: {model.coef_[0]:.2f}%/Q)'
    )
    
    plt.title('Churn Rate Trend by Signup Cohort', fontsize=14, fontweight='bold')
    plt.ylabel('Churn Rate (%)', fontsize=12)
    plt.xlabel('Signup Quarter', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '11_cohort_trend.png'))
    plt.close()
    
    # d) Cohort Insights
    print("\n" + "="*60)
    print("COHORT INSIGHTS SUMMARY")
    print("="*60)
    max_churn = quarterly_churn.loc[quarterly_churn['Churn Rate (%)'].idxmax()]
    min_churn = quarterly_churn.loc[quarterly_churn['Churn Rate (%)'].idxmin()]
    
    trend_slope = model.coef_[0]
    trend_dir = "deteriorating (increasing)" if trend_slope > 0 else "improving (decreasing)"
    
    print(f"- Highest Churn Quarter: {max_churn['signup_quarter']} at {max_churn['Churn Rate (%)']:.1f}%")
    print(f"- Lowest Churn Quarter:  {min_churn['signup_quarter']} at {min_churn['Churn Rate (%)']:.1f}%")
    print(f"- Overall Trend: Retention is {trend_dir} over time.")
    
    print("\nBUSINESS CONTEXT:")
    if trend_slope > 0:
        print("ALERT: Newer cohorts are churning at a higher rate. This suggests recent marketing or product changes might be attracting low-quality leads or the onboarding experience has degraded.")
    else:
        print("SUCCESS: Retention strategies are working! Newer cohorts are sticking around longer than older cohorts did at the same lifecycle stage.")

    print("\nCohort analysis complete. Outputs saved to reports/figures/.")

if __name__ == "__main__":
    run_cohort_analysis()