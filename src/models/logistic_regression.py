# =========================================================
#  LOGISTIC REGRESSION FED DECISION - VERSION FINALE AVEC GRAPHIQUE
# =========================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# === 1. Chargement du dataset ===
df = pd.read_csv("./data/processed/MEETING_WITH_TEXT_MACRO.csv", parse_dates=["observation_date"])
df = df.sort_values("observation_date")

# === 2. Ajout de variables de momentum (signaux économiques) ===
for col in ["CPIAUCSL", "UNRATE", "VIXCLS", "T10Y2Y"]:
    if col in df.columns:
        df[f"{col}_diff3"] = df[col].diff(3)
        df[f"{col}_diff6"] = df[col].diff(6)

# === 3. Cible à horizon plus long ===
df["target"] = (df["DFF"].shift(-3) > df["DFF"]).astype(int)
df = df.dropna(subset=["target"])

# === 4. Sélection des features pertinentes ===
keep = [col for col in df.columns if col.startswith(("CPI", "UNRATE", "VIX", "T10Y2Y", "sentiment"))]
X = df[keep].fillna(0)
y = df["target"]

# === 5. Split temporel ===
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_test = df["observation_date"].iloc[split_idx:]  # utile pour le graphique

# === 6. Standardisation ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 7. Régression logistique avec pondération asymétrique ===
class_weights = {0: 1, 1: 5}  # plus de poids sur la détection des hausses

model = LogisticRegressionCV(
    Cs=15,
    cv=5,
    penalty='elasticnet',
    solver='saga',
    l1_ratios=[0.3, 0.5],
    class_weight=class_weights,
    max_iter=20000,
    n_jobs=-1
)

print("Entraînement du modèle...")
model.fit(X_train_scaled, y_train)

# === 8. Évaluation ===
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n=== ÉVALUATION FINALE (RÉALISTE ET CIBLÉE) ===")
print(f"Accuracy : {acc:.3f}")
print(f"F1-score : {f1:.3f}")
print(f"AUC : {auc:.3f}")
print("\nRapport complet :")
print(classification_report(y_test, y_pred))

# === 9. Ajout des résultats dans un DataFrame pour visualisation ===
results_df = pd.DataFrame({
    "Date": dates_test.values,
    "True_Decision": y_test.values,
    "Predicted": y_pred,
    "Prob_Hike": y_proba
})

# === 10. Visualisation chronologique ===
plt.figure(figsize=(14,6))
plt.plot(results_df["Date"], results_df["Prob_Hike"], label="Probabilité prédite de HAUSSE", color="tab:blue")
plt.scatter(
    results_df["Date"], 
    results_df["True_Decision"],  # pour qu'on les voie au-dessus
    color="green", label="Décision réelle (1 = hausse)"
)
plt.scatter(
    results_df["Date"], 
    results_df["Predicted"], 
    color="red", marker="x", label="Prédiction du modèle (1 = hausse)"
)

plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
plt.title("Prédictions de hausses de taux de la Fed (Logistic Regression)")
plt.xlabel("Date de réunion FOMC")
plt.ylabel("Probabilité de hausse prédite")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# === 11. Exemple de dernière prédiction ===
last_prob = y_proba[-1]
last_pred = y_pred[-1]
last_date = results_df["Date"].iloc[-1]

decision = "HAUSSE probable" if last_pred == 1 else "TAUX STABLE probable"
print(f"\n Dernière prédiction ({last_date.date()}): {decision} ({last_prob*100:.1f}% de confiance)")
