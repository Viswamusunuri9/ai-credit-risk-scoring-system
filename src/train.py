import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

from preprocessing import load_data, preprocess_data
from feature_engineering import create_features


def train_model():
    df = load_data()
    df = preprocess_data(df)
    df = create_features(df)

    # Target
    y = df["Risk"].map({"good": 0, "bad": 1})

    # Features
    X = df.drop(columns=["Risk"])

    # Identify columns
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # Model pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(eval_metric="logloss")), #use_label_encoder=False
        ]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    print(f"ROC-AUC: {auc}")

    # Save model
    with open("../models/credit_model.pkl", "wb") as f:
        pickle.dump(model, f)

    return model


if __name__ == "__main__":
    train_model()