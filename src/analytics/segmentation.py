import pandas as pd
import numpy as np
import os
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150

def load_data(train_path, test_path):
    """Load and combine train/test data for analysis."""
    print(f"Loading data from {train_path} and {test_path}...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    df = pd.concat([train, test], ignore_index=True)
    
    # Ensure churn is numeric for calculations
    if 'Churn' in df.columns:
        df['Churn_Numeric'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    return df

def run_segmentation(df):
    """Perform K-Means clustering and save diagnostic plots."""
    print("\n--- STEP 1: CUSTOMER SEGMENTATION ---")
    features = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 
        'avg_monthly_usage', 'engagement_score', 
        'complaints_count', 'customer_support_calls'
    ]
    
    X = df[features].copy()
    
    # SHARPENED: Use PowerTransformer (Yeo-Johnson) for clustering
    # Standard scaling fails on skewed features like TotalCharges.
    # PowerTransformer makes data more Gaussian, which K-Means prefers.
    scaler = PowerTransformer(method='yeo-johnson')
    X_scaled = scaler.fit_transform(X)
    
    # Elbow Method
    print("Running Elbow Method (k=2 to 10)...")
    inertia = []
    k_range = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of clusters (k)', fontsize=12)
    plt.ylabel('Inertia (Sum of squared distances)', fontsize=12)
    plt.title('Elbow Method for Optimal Customer Segments', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.join('reports', 'figures'), exist_ok=True)
    plt.savefig(os.path.join('reports', 'figures', '01_elbow_kmeans.png'))
    plt.close()
    
    # Silhouette Score
    print("Running Silhouette Analysis (k=2 to 8)...")
    sil_scores = []
    k_range_sil = range(2, 9)
    for k in k_range_sil:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        sil_scores.append(score)
        print(f"  > Silhouette Score for k={k}: {score:.4f}")
        
    plt.figure(figsize=(10, 6))
    plt.plot(k_range_sil, sil_scores, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Number of clusters (k)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Silhouette Scores for Customer Segmentation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join('reports', 'figures', '02_silhouette_scores.png'))
    plt.close()
    
    # Final KMeans (k=4)
    print("Fitting final KMeans with k=4...")
    km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = km_final.fit_predict(X_scaled)
    
    # PCA Visualization
    print("Generating PCA visualization (2D projection)...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df['pca1'] = pca_result[:, 0]
    df['pca2'] = pca_result[:, 1]
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='pca1', y='pca2', hue='cluster', 
        palette='viridis', data=df, alpha=0.7, edgecolors='w'
    )
    plt.title('Customer Segments Visualized via PCA (2D Projection)', fontsize=14, fontweight='bold')
    plt.xlabel('PCA Component 1', fontsize=12)
    plt.ylabel('PCA Component 2', fontsize=12)
    plt.legend(title='Cluster ID')
    plt.tight_layout()
    plt.savefig(os.path.join('reports', 'figures', '03_cluster_pca.png'))
    plt.close()
    
    return km_final, scaler, df

def name_clusters(df):
    """Assign business names to clusters based on profiles."""
    print("Analyzing cluster profiles for business naming...")
    cluster_profiles = df.groupby('cluster').agg({
        'tenure': 'mean',
        'MonthlyCharges': 'mean',
        'complaints_count': 'mean',
        'Churn_Numeric': 'mean'
    }).reset_index()
    
    # Sorting clusters by tenure to identify lifecycle stages
    sorted_by_tenure = cluster_profiles.sort_values('tenure')
    
    names = {}
    
    # 1. Newest cluster -> "New & Unsettled"
    new_idx = sorted_by_tenure.iloc[0]['cluster']
    names[new_idx] = "New & Unsettled"
    
    # 2. Oldest cluster -> "Veteran Champion"
    vet_idx = sorted_by_tenure.iloc[3]['cluster']
    names[vet_idx] = "Veteran Champion"
    
    # 3. Middle clusters: Identify "At-risk Premium" vs "Loyal Mid-tier"
    rem_ids = [sorted_by_tenure.iloc[1]['cluster'], sorted_by_tenure.iloc[2]['cluster']]
    rem_profiles = cluster_profiles[cluster_profiles['cluster'].isin(rem_ids)]
    
    # The one with higher churn rate or higher complaints is "At-risk Premium"
    # Or higher MonthlyCharges
    at_risk_id = rem_profiles.sort_values('MonthlyCharges', ascending=False).iloc[0]['cluster']
    loyal_id = [i for i in rem_ids if i != at_risk_id][0]
    
    names[at_risk_id] = "At-risk Premium"
    names[loyal_id] = "Loyal Mid-tier"

    df['customer_segment'] = df['cluster'].map(names)
    return names, df

def calculate_clv(df):
    """Estimate Customer Lifetime Value (CLV)."""
    print("\n--- STEP 2: CUSTOMER LIFETIME VALUE (CLV) ---")
    model_path = os.path.join('models', 'best_model.pkl')
    pipeline_path = os.path.join('models', 'preprocessing_pipeline.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(pipeline_path):
        print("Error: Model or Pipeline file not found!")
        return df
        
    model = joblib.load(model_path)
    
    # Ensure we only pass features that the pipeline expects
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
    X_input = df[expected_cols].copy()
    
    # Predict churn probability directly using the model pipeline
    df['churn_prob'] = model.predict_proba(X_input)[:, 1]
    
    # CLV formula: CLV = MonthlyCharges * expected_remaining_tenure * (1 - churn_prob)
    # expected_remaining_tenure = max(0, 72 - tenure)
    df['expected_remaining_tenure'] = (72 - df['tenure']).clip(lower=0)
    df['CLV'] = df['MonthlyCharges'] * df['expected_remaining_tenure'] * (1 - df['churn_prob'])
    
    # Churn risk tier
    df['churn_risk_tier'] = pd.cut(df['churn_prob'], 
                                   bins=[-0.01, 0.25, 0.50, 0.75, 1.01], 
                                   labels=['Low', 'Medium', 'High', 'Critical'])
    
    return df

def save_clv_plots(df):
    """Save CLV distribution and scatter plots."""
    print("Saving CLV visualization plots...")
    # CLV by segment boxplot
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='customer_segment', y='CLV', data=df, palette='Set2')
    plt.title('CLV Distribution by Customer Segment', fontsize=14, fontweight='bold')
    plt.xlabel('Customer Segment', fontsize=12)
    plt.ylabel('Estimated CLV ($)', fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join('reports', 'figures', '04_clv_by_segment.png'))
    plt.close()
    
    # CLV vs Churn Prob scatter
    plt.figure(figsize=(12, 7))
    sns.scatterplot(
        x='churn_prob', y='CLV', 
        hue='customer_segment', size='MonthlyCharges',
        sizes=(20, 200), alpha=0.6, data=df, palette='viridis'
    )
    plt.title('CLV vs Churn Probability', fontsize=14, fontweight='bold')
    plt.xlabel('Churn Probability', fontsize=12)
    plt.ylabel('Estimated CLV ($)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, title='Segment')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join('reports', 'figures', '05_clv_vs_churnprob.png'))
    plt.close()

def print_summaries(df):
    """Print requested summaries and insights."""
    print("\n" + "="*70)
    print("CLUSTER PROFILE SUMMARY")
    print("="*70)
    profile = df.groupby('customer_segment').agg({
        'customer_segment': 'count',
        'tenure': 'mean',
        'MonthlyCharges': 'mean',
        'complaints_count': 'mean',
        'engagement_score': 'mean',
        'Churn_Numeric': 'mean'
    }).rename(columns={'customer_segment': 'Size', 'Churn_Numeric': 'Churn Rate%'})
    profile['Churn Rate%'] = profile['Churn Rate%'] * 100
    print(profile.round(2))
    
    print("\n" + "="*70)
    print("CLV SUMMARY PER SEGMENT")
    print("="*70)
    # Total CLV at risk calculation
    at_risk_df = df[df['churn_prob'] > 0.5]
    total_clv_at_risk = at_risk_df.groupby('customer_segment')['CLV'].sum()
    
    clv_summary = df.groupby('customer_segment').agg({
        'customer_segment': 'count',
        'CLV': 'mean',
        'churn_prob': 'mean'
    }).rename(columns={'customer_segment': 'Count', 'CLV': 'Avg CLV', 'churn_prob': 'Avg Churn Prob'})
    
    clv_summary['Total CLV at Risk'] = total_clv_at_risk
    print(clv_summary.round(2))
    
    print("\n" + "="*70)
    print("TOP 200 HIGHEST-CLV CHURNERS (VIP RETENTION TARGETS)")
    print("="*70)
    median_clv = df['CLV'].median()
    vips = df[(df['churn_prob'] > 0.5) & (df['CLV'] > median_clv)].sort_values('CLV', ascending=False).head(200)
    print(f"Total High-Value Churners found: {len(vips)}")
    print(vips[['customer_segment', 'tenure', 'MonthlyCharges', 'churn_prob', 'CLV']].head(10).to_string(index=False))
    
    # Revenue saved
    total_saved = vips['MonthlyCharges'].sum() * 12
    print(f"\nBUSINESS INSIGHT: If we retained these 200 VIPs, total revenue saved (12-month window): ${total_saved:,.2f}")

def main():
    # File paths
    train_path = os.path.join('data', 'processed', 'train.csv')
    test_path = os.path.join('data', 'processed', 'test.csv')
    
    # Load data
    df = load_data(train_path, test_path)
    
    # Step 1: Segmentation
    km, scaler, df = run_segmentation(df)
    names, df = name_clusters(df)
    
    # Step 2: CLV
    df = calculate_clv(df)
    save_clv_plots(df)
    print_summaries(df)
    
    # Save artifacts
    print("\n--- SAVING PHASE 4 OUTPUTS ---")
    models_dir = 'models'
    data_dir = os.path.join('data', 'processed')
    
    # Senior Check: Balance Interpretation
    counts = df['customer_segment'].value_counts()
    if counts.max() > counts.min() * 3:
        print(f"Note: Unequal segments detected ({counts.max()} vs {counts.min()}).")
        print("   This is expected: Most customers are 'Standard' while VIPs are a 'Niche'.")
    else:
        print("SUCCESS: Segments are well-balanced for operational targeting.")
        
    joblib.dump({
        'model': km, 
        'scaler': scaler, 
        'labels': names
    }, os.path.join(models_dir, 'kmeans_segmentation.pkl'))
    
    df.to_csv(os.path.join(data_dir, 'customers_scored.csv'), index=False)
    
    print(f"Successfully saved:")
    print(f"  - {os.path.join(models_dir, 'kmeans_segmentation.pkl')}")
    print(f"  - {os.path.join(data_dir, 'customers_scored.csv')}")
    
    print("\nSegmentation and CLV analysis complete.")

def what_if_simulator(customer_data, changes):
    """
    Simulate changes to a customer's profile and see the impact on churn probability.
    
    Inputs:
    - customer_data: dict or Series containing customer features
    - changes: dict of features to modify (e.g., {'Contract': 'One year'})
    
    Returns:
    - original_prob, new_prob, clv_change
    """
    model_path = os.path.join('models', 'best_model.pkl')
    model = joblib.load(model_path)
    
    # Expected columns
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
    
    # Original prediction
    df_orig = pd.DataFrame([customer_data])[expected_cols]
    orig_prob = model.predict_proba(df_orig)[0, 1]
    
    # Modified prediction
    df_new = df_orig.copy()
    for col, val in changes.items():
        if col in df_new.columns:
            df_new[col] = val
            
    new_prob = model.predict_proba(df_new)[0, 1]
    
    # CLV Change
    # Simplified CLV for simulation
    tenure = customer_data['tenure']
    monthly = customer_data['MonthlyCharges']
    rem_tenure = max(0, 72 - tenure)
    
    orig_clv = monthly * rem_tenure * (1 - orig_prob)
    new_clv = monthly * rem_tenure * (1 - new_prob)
    
    return {
        'original_prob': orig_prob,
        'new_prob': new_prob,
        'prob_change': new_prob - orig_prob,
        'clv_change': new_clv - orig_clv
    }

if __name__ == "__main__":
    main()