import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Applies consistent feature engineering to a dataframe.
    Must be identical across training, Streamlit, and API.
    """
    df_out = df.copy()
    
    # 0. Fill missing behavioral metrics with sensible defaults if missing
    # (To support basic UI forms that only ask for 19 core fields)
    behavioral_defaults = {
        'last_active_days': 15,
        'avg_monthly_usage': 50,
        'number_of_logins': 5,
        'complaints_count': 0,
        'customer_support_calls': 0,
        'late_payments': 0
    }
    for col, default_val in behavioral_defaults.items():
        if col not in df_out.columns:
            df_out[col] = default_val

    # Ensure numeric types
    df_out['tenure'] = pd.to_numeric(df_out['tenure'], errors='coerce').fillna(0)
    df_out['MonthlyCharges'] = pd.to_numeric(df_out['MonthlyCharges'], errors='coerce').fillna(0)
    df_out['TotalCharges'] = pd.to_numeric(df_out['TotalCharges'], errors='coerce')
    df_out['TotalCharges'] = df_out['TotalCharges'].fillna(df_out['MonthlyCharges'] * df_out['tenure'])

    # Convert SeniorCitizen string to int if needed
    if df_out['SeniorCitizen'].dtype == 'object':
        df_out['SeniorCitizen'] = df_out['SeniorCitizen'].map({'Yes': 1, 'No': 0, '1': 1, '0': 0}).fillna(0).astype(int)

    # 1. charges_per_month
    df_out['charges_per_month'] = df_out['TotalCharges'] / df_out['tenure'].clip(lower=1)
    
    # 2. TenureGroup
    bins = [-1, 12, 24, 48, 1000]
    labels = ['New', 'Regular', 'Loyal', 'Veteran']
    df_out['tenure_group'] = pd.cut(df_out['tenure'], bins=bins, labels=labels)
    df_out['tenure_group'] = df_out['tenure_group'].astype(str).replace('nan', 'New')

    # 3. engagement_score (derived composite score 0-100)
    if 'engagement_score' not in df_out.columns or df_out['engagement_score'].isnull().all():
        score = (df_out['avg_monthly_usage'] / 5) + (df_out['number_of_logins'] * 2) - df_out['last_active_days']
        df_out['engagement_score'] = score.clip(lower=0, upper=100)

    # 4. HasMultiServices
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    df_out['HasMultiServices'] = sum((df_out.get(svc, 'No') == 'Yes').astype(int) for svc in services)
    
    # 5. IsSeniorWithPartner
    df_out['IsSeniorWithPartner'] = ((df_out['SeniorCitizen'] == 1) & (df_out['Partner'] == 'Yes')).astype(int)

    return df_out
