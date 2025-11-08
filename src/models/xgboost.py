import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# === Chargement et préparation ===
dataset = pd.read_csv("./data/processed/MEETING_WITH_TEXT_MACRO.csv")
dataset = dataset.sort_values("observation_date").reset_index(drop=True)

# --- Construction de la cible ---
horizon = 1
dataset["target_up"] = (dataset["DFF"].shift(-horizon) > dataset["DFF"]).astype(int)
dataset.dropna(inplace=True)

# --- Assignation manuelle des régimes ---
def assign_regime(date):
    if date < pd.Timestamp("2008-01-01"):
        return "TIGHTENING_PRE2008"
    elif date < pd.Timestamp("2017-01-01"):
        return "QE_ZERO_RATE"
    elif date < pd.Timestamp("2020-01-01"):
        return "NORMALIZATION_2017_2019"
    else:
        return "INFLATION_2021_2025"

dataset["regime"] = pd.to_datetime(dataset["observation_date"]).apply(assign_regime)

# --- Features à garder ---
features = [
    "DFF","CPIAUCSL_DIFF3","UNRATE_DIFF3","T10Y2Y_DIFF3","VIXCLS_MA30",
    "sentiment_score","text_comp_1","text_comp_2","text_comp_3",
    "sentiment_score*CPIAUCSL_DIFF3","sentiment_score*UNRATE_DIFF3"
]

# === Boucle par régime ===
results = []
for regime, subdf in dataset.groupby("regime"):
    X, y = subdf[features], subdf["target_up"]
    if len(subdf) < 20 or len(y.unique()) < 2:
        continue

    # Modèle par régime
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.3,
        scale_pos_weight=len(y[y==0]) / len(y[y==1]),
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:,1]

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)

    print(f"\n=== Régime : {regime} ===")
    print(f"Observations : {len(subdf)}")
    print(f"Accuracy : {acc:.3f}")
    print(f"AUC      : {auc:.3f}")

    importance = model.get_booster().get_score(importance_type="gain")
    imp_df = pd.DataFrame(list(importance.items()), columns=["Feature","Importance"])\
                .sort_values(by="Importance", ascending=False).head(8)
    print(imp_df)

    results.append({"regime":regime,"acc":acc,"auc":auc})

# === Synthèse globale ===
results_df = pd.DataFrame(results)
print("\n=== Résumé global ===")
print(results_df)
