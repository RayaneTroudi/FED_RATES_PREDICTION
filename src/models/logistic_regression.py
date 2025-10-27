from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.features.features_engineering import build_dataset
from src.features.features_config import get_feature_config

import pandas as pd

def logistic_regression_temporal(dataset, target_col, horizon=1):
    df = dataset.copy().sort_index()

    # --- construction de la cible ---
    df['target'] = (df[target_col].shift(-horizon) > df[target_col]).astype(int)
    df.dropna(inplace=True)

    # --- sélection stricte des features numériques ---
    feature_cols = [
        col for col in df.columns
        if col not in ['target', target_col] and not pd.api.types.is_datetime64_any_dtype(df[col])
    ]

    # --- séparation temporelle ---
    split = int(len(df) * 0.8)
    X_train, X_test = df[feature_cols].iloc[:split], df[feature_cols].iloc[split:]
    y_train, y_test = df['target'].iloc[:split], df['target'].iloc[split:]

    # --- modèle ---
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n=== LOGISTIC REGRESSION RESULTS ===")
    print("Accuracy:", round(acc, 4))
    print("\nCoefficients:")
    print(pd.Series(model.coef_[0], index=feature_cols))
    print("\nIntercept:", model.intercept_[0])
    print("\nFirst 10 Predictions:", y_pred[:10])

    return model, acc, df


dataset_for_training = build_dataset(config=get_feature_config())
logistic_regression_temporal(dataset_for_training, target_col='DFF', horizon=1)
