import pandas as pd
import numpy as np

def build_features(df):
    df = df.copy()

    # Normalize signup_date into stable numeric signals.
    if "signup_date" in df.columns:
        signup_dt = pd.to_datetime(df["signup_date"], errors="coerce")
        df["signup_year"] = signup_dt.dt.year.fillna(signup_dt.dt.year.median())
        df["signup_month"] = signup_dt.dt.month.fillna(0).astype(int)
        df = df.drop(columns=["signup_date"])

    # 1. Charges per month
    df["charges_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["charges_per_month"] = df["charges_per_month"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 2. Tenure group
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["New", "Regular", "Loyal", "Veteran"],
        include_lowest=True
    )

    # 3. Has multiple services
    service_cols = [
        "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport"
    ]

    available_service_cols = [col for col in service_cols if col in df.columns]
    if available_service_cols:
        df["HasMultiServices"] = df[available_service_cols].apply(
            lambda x: sum([1 if val == "Yes" else 0 for val in x]),
            axis=1
        )
    else:
        df["HasMultiServices"] = 0

    # 4. Senior with partner
    df["IsSeniorWithPartner"] = np.where(
        (df["SeniorCitizen"] == 1) & (df["Partner"] == "Yes"),
        1, 0
    )

    # 5. Charge to income proxy (approximation)
    avg_charge = df["MonthlyCharges"].mean()
    df["ChargeToIncomeProxy"] = df["MonthlyCharges"] / avg_charge

    return df
