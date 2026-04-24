import pickle
import pandas as pd

import os

def load_model():
    base_dir = os.path.dirname(os.path.dirname(__file__))  # go up from src/
    model_path = os.path.join(base_dir, "models", "credit_model.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


def predict_risk(input_dict):
    model = load_model()

    df = pd.DataFrame([input_dict])

    # Feature engineering
    df["credit_per_month"] = df["Credit amount"] / df["Duration"]

    prob = model.predict_proba(df)[0][1]

    # Threshold logic (your key insight)
    if prob > 0.4:
        risk = "High Risk"
    else:
        risk = "Low Risk"

    return prob, risk