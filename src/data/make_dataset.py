from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CANDIDATES = (
    PROJECT_ROOT / "data" / "raw" / "telecom_churn.csv",
)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# -----------------------------
# 1. Load Raw Data
# -----------------------------
def load_data(filepath: str | Path) -> pd.DataFrame:
    print("Loading raw dataset...")
    return pd.read_csv(filepath)


# -----------------------------
# 2. Clean Data
# -----------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning data...")

    df = df.copy()

    # Convert TotalCharges to numeric (if exists)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Convert dates
    if "signup_date" in df.columns:
        df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
        df["days_since_signup"] = (pd.Timestamp.today() - df["signup_date"]).dt.days
    # Drop missing values
    missing_before = int(df.isnull().sum().sum())
    print(f"Missing values before cleaning: {missing_before}")

    df = df.dropna()

    missing_after = int(df.isnull().sum().sum())
    print(f"Missing values after cleaning: {missing_after}")

    # Remove duplicates
    duplicates = int(df.duplicated().sum())
    print(f"Duplicate rows: {duplicates}")
    df = df.drop_duplicates()

    return df


# -----------------------------
# 3. Validate Dataset
# -----------------------------
def validate_data(df: pd.DataFrame) -> None:
    print("Validating dataset...")

    assert "Churn" in df.columns, "Target column 'Churn' missing!"

    unique_vals = set(df["Churn"].dropna().astype(str).str.strip())
    print(f"Churn values: {sorted(unique_vals)}")

    if not unique_vals.issubset({"Yes", "No"}):
        raise ValueError("Churn column must contain only 'Yes' and 'No'")

    print("Validation passed!")


# -----------------------------
# 4. Train-Test Split
# -----------------------------
def split_data(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Splitting data...")

    train, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["Churn"],
        random_state=42,
    )

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    return train, test


# -----------------------------
# 5. Save Processed Data
# -----------------------------
def save_data(train: pd.DataFrame, test: pd.DataFrame) -> None:
    print("Saving processed data...")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print("Data saved successfully!")


def pick_raw_data_path() -> Path:
    for candidate in RAW_CANDIDATES:
        if candidate.exists():
            return candidate
    checked = "\n".join(f"- {path}" for path in RAW_CANDIDATES)
    raise FileNotFoundError(f"No raw dataset found. Checked:\n{checked}")


# -----------------------------
# 6. Main Pipeline
# -----------------------------
def main() -> None:
    filepath = pick_raw_data_path()

    df = load_data(filepath)
    df = clean_data(df)
    validate_data(df)

    train, test = split_data(df)
    save_data(train, test)

    print("Data pipeline completed successfully!")


# -----------------------------
# Run Script
# -----------------------------
if __name__ == "__main__":
    main()
