import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

    # Drop missing
    df = df.dropna()

    return df


def split_data(df, test_size=0.2):
    train, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["Churn"],   # IMPORTANT
        random_state=42
    )
    return train, test


def save_data(train, test):
    os.makedirs("data/processed", exist_ok=True)

    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)


if __name__ == "__main__":
    df = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = clean_data(df)
    train, test = split_data(df)
    save_data(train, test)

    print("✅ Data processing complete")