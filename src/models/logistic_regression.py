from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.features.features_engineering import build_dataset_meeting_transform
from src.features.features_config import get_feature_config


def logistic_regression_temporal(dataset, target_col, horizon=1, threshold=0.5):
    df = dataset.copy().sort_index()

    # --- construction de la cible ---
    df['target'] = (df[target_col].shift(-horizon) > df[target_col]).astype(int)
    df.dropna(inplace=True)

    # --- sélection stricte des features numériques ---
    feature_cols = [
        col for col in df.columns
        if col not in ['target', target_col]
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    # --- split temporel ---
    split = int(len(df) * 0.8)
    X_train, X_test = df[feature_cols].iloc[:split], df[feature_cols].iloc[split:]
    y_train, y_test = df['target'].iloc[:split], df['target'].iloc[split:]

    # --- normalisation sans fuite ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- modèle pondéré ---
    model = LogisticRegression(max_iter=5000, class_weight='balanced')
    model.fit(X_train, y_train)

    # --- prédictions avec seuil ajusté ---
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= threshold).astype(int)

    # --- évaluation ---
    acc = accuracy_score(y_test, y_pred)
    print("\n=== LOGISTIC REGRESSION RESULTS ===")
    print(f"Accuracy: {acc:.4f} | Threshold: {threshold}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("\nCoefficients:")
    print(pd.Series(model.coef_[0], index=feature_cols))
    print("\nIntercept:", model.intercept_[0])

    # --- histogramme des probabilités ---
    plt.figure(figsize=(7,4))
    plt.hist(y_pred_prob, bins=30, edgecolor='k')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold={threshold}')
    plt.legend()
    plt.title("Distribution des probabilités prédites")
    plt.xlabel("P(y=1)")
    plt.ylabel("Fréquence")
    plt.show()
    
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_test, y_pred_prob)

    plt.plot(thr, prec[:-1], label='Precision')
    plt.plot(thr, rec[:-1], label='Recall')
    plt.xlabel("Threshold")
    plt.legend()
    plt.show()


    return model, acc, df


# --- exécution principale ---
dataset_for_training = build_dataset_meeting_transform(config=get_feature_config())
logistic_regression_temporal(dataset_for_training, target_col='DFF', horizon=1, threshold=0.4)
