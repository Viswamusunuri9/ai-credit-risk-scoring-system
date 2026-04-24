import pandas as pd


def load_data(path="../data/german_credit_data - german_credit_data.csv"):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    # Drop unnecessary column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Handle missing values
    df = df.fillna("Unknown")

    return df