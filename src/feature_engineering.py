def create_features(df):
    # Credit burden feature (VERY IMPORTANT)
    df["credit_per_month"] = df["Credit amount"] / df["Duration"]

    return df