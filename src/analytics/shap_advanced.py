import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150

def run_shap_analysis():
    """Perform advanced SHAP analysis for global and segment-level insights."""
    print("\n--- STEP 3: ADVANCED SHAP ANALYSIS ---")
    
    # Paths
    model_path = os.path.join('models', 'best_model.pkl')
    test_path = os.path.join('data', 'processed', 'test.csv')
    scored_path = os.path.join('data', 'processed', 'customers_scored.csv')
    figures_dir = os.path.join('reports', 'figures')
    
    if not all(os.path.exists(p) for p in [model_path, test_path, scored_path]):
        print("Error: Required files for SHAP analysis are missing.")
        return
    
    # Load data and model
    model_pipeline = joblib.load(model_path)
    test_df = pd.read_csv(test_path)
    scored_df = pd.read_csv(scored_path)
    
    # Extract components from pipeline
    preprocessor = model_pipeline.named_steps['preprocessor']
    xgb_model = model_pipeline.named_steps['classifier']
    
    # Expected columns for preprocessing
    expected_cols = [
        'tenure', 'last_active_days', 'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 
        'avg_monthly_usage', 'number_of_logins', 'complaints_count', 
        'customer_support_calls', 'late_payments', 'charges_per_month', 
        'engagement_score', 'gender', 'Partner', 'Dependents', 
        'PhoneService', 'MultipleLines', 'InternetService', 
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
        'PaperlessBilling', 'PaymentMethod', 'tenure_group'
    ]
    
    # 1. Prepare Test Data
    X_test_raw = test_df[expected_cols]
    X_test_transformed = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.get_feature_names_out()
    X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names)
    
    # 2. Compute SHAP values for Test Set
    print("Computing SHAP values for full test set...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_transformed_df)
    
    # Handle list return for binary classification in some SHAP versions
    if isinstance(shap_values, list):
        shap_values = shap_values[1] # Use positive class
    
    # a) Global SHAP
    print("Saving global SHAP summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_transformed_df, show=False)
    plt.title('Global Feature Importance (SHAP Values)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '06_shap_global_full.png'))
    plt.close()
    
    # b) SHAP Dependence Plots
    print("Generating dependence plots...")
    
    # Function to get column name in transformed space
    def find_col(name):
        matches = [c for c in feature_names if name in c]
        return matches[0] if matches else None

    # tenure vs SHAP (color by Contract)
    contract_col = find_col('Contract_Month-to-month')
    plt.figure(figsize=(10, 6))
    shap.dependence_plot("num__tenure", shap_values, X_test_transformed_df, 
                         interaction_index=contract_col, show=False)
    plt.title('SHAP Dependence: Tenure colored by Contract Type', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '07_shap_dep_tenure.png'))
    plt.close()
    
    # MonthlyCharges vs SHAP (color by InternetService)
    internet_col = find_col('InternetService_Fiber optic')
    plt.figure(figsize=(10, 6))
    shap.dependence_plot("num__MonthlyCharges", shap_values, X_test_transformed_df, 
                         interaction_index=internet_col, show=False)
    plt.title('SHAP Dependence: Monthly Charges colored by Internet Service', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '07_shap_dep_charges.png'))
    plt.close()
    
    # engagement_score vs SHAP (color by complaints_count)
    plt.figure(figsize=(10, 6))
    shap.dependence_plot("num__engagement_score", shap_values, X_test_transformed_df, 
                         interaction_index="num__complaints_count", show=False)
    plt.title('SHAP Dependence: Engagement Score colored by Complaints', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '07_shap_dep_engagement.png'))
    plt.close()
    
    # c) SHAP Interaction Plot (Tenure vs MonthlyCharges)
    print("Generating interaction plot (Tenure vs Monthly Charges)...")
    plt.figure(figsize=(10, 6))
    shap.dependence_plot("num__tenure", shap_values, X_test_transformed_df, 
                         interaction_index="num__MonthlyCharges", show=False)
    plt.title('SHAP Interaction: Tenure & Monthly Charges', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '08_shap_interaction.png'))
    plt.close()
    
    # d) Segment-level SHAP
    print("\nAnalyzing top churn drivers per customer segment...")
    X_full_raw = scored_df[expected_cols]
    X_full_transformed = preprocessor.transform(X_full_raw)
    X_full_transformed_df = pd.DataFrame(X_full_transformed, columns=feature_names)
    
    # Compute SHAP for all customers (sample if very large, but 20k is fine for TreeExplainer)
    shap_values_full = explainer.shap_values(X_full_transformed_df)
    if isinstance(shap_values_full, list):
        shap_values_full = shap_values_full[1]
    
    segments = scored_df['customer_segment'].unique()
    segment_drivers = []
    
    for segment in segments:
        # Get indices for this segment
        segment_indices = scored_df[scored_df['customer_segment'] == segment].index
        seg_shap = shap_values_full[segment_indices]
        
        # Calculate mean absolute SHAP per feature
        mean_abs_shap = np.abs(seg_shap).mean(axis=0)
        top_3_idx = np.argsort(mean_abs_shap)[-3:][::-1]
        
        # Map back to readable names (strip num__/cat__ prefixes)
        top_drivers = [feature_names[i].split('__')[-1] for i in top_3_idx]
        
        segment_drivers.append({
            'Segment': segment,
            'Driver 1': top_drivers[0],
            'Driver 2': top_drivers[1],
            'Driver 3': top_drivers[2]
        })
    
    drivers_df = pd.DataFrame(segment_drivers)
    print("="*60)
    print("TOP 3 CHURN DRIVERS PER SEGMENT")
    print("="*60)
    print(drivers_df.to_string(index=False))
    
    # e) SHAP Force Plot for Top 5 Risk Customers
    print("\nGenerating HTML force plot for top 5 high-risk customers...")
    top_5_idx = scored_df.sort_values('churn_prob', ascending=False).head(5).index
    
    # Get expected value (base value)
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1:
        expected_value = expected_value[1] # Positive class
    elif isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[0]

    # Note: force_plot usually needs link='logit' for probability space
    force_plot = shap.force_plot(
        expected_value, 
        shap_values_full[top_5_idx], 
        X_full_transformed_df.iloc[top_5_idx],
        link='logit'
    )
    shap.save_html(os.path.join(figures_dir, '09_shap_force_plot.html'), force_plot)
    
    print("\nSHAP analysis complete. Plots saved to reports/figures/")

if __name__ == "__main__":
    run_shap_analysis()
