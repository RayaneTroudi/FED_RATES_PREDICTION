import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# === Fonction : construction de la target ===
def build_target(df, target_col="DFF", horizon=1, invert_target=False):
    if invert_target:
        df["target"] = (df[target_col].shift(-horizon) < df[target_col]).astype(int)
    else:
        df["target"] = (df[target_col].shift(-horizon) > df[target_col]).astype(int)
    return df.dropna(subset=["target"]).copy()

# === Fonction : attribution du régime monétaire ===
def assign_regime(date):
    if date < pd.Timestamp("2008-09-01"):
        return "TIGHTENING_PRE2008"
    elif date < pd.Timestamp("2016-12-01"):
        return "QE_ZERO_RATE"
    elif date < pd.Timestamp("2020-01-01"):
        return "NORMALIZATION_2017_2019"
    else:
        return "INFLATION_2021_2025"

# === Fonction : entraînement par régime ===
def train_by_regime(df, regime_col="regime"):
    results = []
    for regime, sub in df.groupby(regime_col):
        if len(sub) < 20:
            continue
        print(f"\n=== RÉGIME : {regime} ===")
        sub = build_target(sub, target_col="DFF", horizon=1, invert_target=False)
        sub = sub.sort_values("observation_date")
        split_idx = int(len(sub) * 0.7)
        train_df = sub.iloc[:split_idx]
        test_df = sub.iloc[split_idx:]

        X_train = train_df.select_dtypes(include=[np.number]).drop(columns=["DFF", "target"], errors="ignore").fillna(0)
        y_train = train_df["target"].astype(int)
        X_test = test_df.select_dtypes(include=[np.number]).drop(columns=["DFF", "target"], errors="ignore").fillna(0)
        y_test = test_df["target"].astype(int)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegressionCV(
            Cs=10,
            cv=5,
            penalty="elasticnet",
            solver="saga",
            l1_ratios=[0.5],
            max_iter=10000,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        print(f"Accuracy : {acc:.3f}, AUC : {auc:.3f}")

        coef_df = pd.DataFrame({
            "Feature": X_train.columns,
            "Coefficient": model.coef_[0]
        }).sort_values("Coefficient", ascending=False)
        print(coef_df.head(10))

        results.append({
            "regime": regime,
            "accuracy": acc,
            "auc": auc,
            "n_obs": len(sub)
        })
    return pd.DataFrame(results)

# === Chargement du dataset ===
df = pd.read_csv("./data/processed/MEETING_WITH_TEXT_MACRO.csv")

# === Conversion des dates et création du régime ===
df["observation_date"] = pd.to_datetime(df["observation_date"])
df["regime"] = df["observation_date"].apply(assign_regime)

# === Entraînement par régime ===
summary = train_by_regime(df, regime_col="regime")

# === Synthèse finale ===
print("\n=== SYNTHÈSE GLOBALE ===")
print(summary)
