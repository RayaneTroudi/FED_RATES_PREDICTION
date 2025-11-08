import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# 1. Construction du dataset avec états Markoviens
# ============================================================

def build_dataset_conditional_markov(df):
    """Construit les états (-1, +1) à partir du taux directeur et les variables explicatives."""
    df = df.sort_values("observation_date").reset_index(drop=True)
    df["state_t"] = np.sign(df["DFF"].diff())
    df["state_t"] = df["state_t"].replace(0, np.nan).ffill()
    df["state_t1"] = df["state_t"].shift(-1)
    df = df.dropna(subset=["state_t", "state_t1"]).reset_index(drop=True)
    return df


# ============================================================
# 2. Entraînement du modèle logistique conditionnel
# ============================================================

def train_conditional_markov(df, features, n_splits=5):
    """Régression logistique conditionnelle sur 5 folds temporels."""
    X = df[features + ["state_t"]].copy()
    y = (df["state_t1"] == 1).astype(int)  # target binaire : 1 = hausse, 0 = baisse

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    y_true_all, y_pred_all, y_prob_all = [], [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled), 1):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            l1_ratio=0.3,
            class_weight="balanced",
            C=1.0,
            max_iter=5000,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_prob_all.extend(y_prob)

        acc = accuracy_score(y_test, y_pred)
        print(f"\n=== Fold {fold} ===")
        print(f"Accuracy : {acc:.3f}")

    print("\n=== ÉVALUATION GLOBALE ===")
    print(confusion_matrix(y_true_all, y_pred_all))
    print(classification_report(y_true_all, y_pred_all, digits=3))

    try:
        auc = roc_auc_score(y_true_all, y_prob_all)
        print(f"AUC global : {auc:.3f}")
    except ValueError:
        print("AUC non défini (une seule classe prédite)")

    return model, scaler


# ============================================================
# 3. Matrice de transition conditionnelle moyenne
# ============================================================

def compute_conditional_transition_matrix(model, scaler, df, features):
    """Construit la matrice de transition conditionnelle moyenne à partir des probabilités logistiques."""
    X_scaled = scaler.transform(df[features + ["state_t"]])
    probs = model.predict_proba(X_scaled)[:, 1]
    df = df.copy()
    df["proba_hausse"] = probs
    df["proba_baisse"] = 1 - probs

    mat = pd.DataFrame(index=[-1.0, 1.0], columns=["Vers -1", "Vers +1"], dtype=float)
    for s in [-1.0, 1.0]:
        sub = df[df["state_t"] == s]
        if len(sub) > 0:
            mat.loc[s, "Vers -1"] = sub["proba_baisse"].mean()
            mat.loc[s, "Vers +1"] = sub["proba_hausse"].mean()
        else:
            mat.loc[s] = [0.0, 0.0]

    mat = mat.fillna(0.0)
    mat = mat.div(mat.sum(axis=1), axis=0)
    mat.index.name = "État courant"
    print("\n=== MATRICE DE TRANSITION CONDITIONNELLE (MOYENNE) ===")
    print(mat.round(3))
    return mat


# ============================================================
# 4. Exemple d’exécution sur données macro
# ============================================================

if __name__ == "__main__":
    # Exemple : on charge le dataset macro déjà construit (meetings ou daily)
    df = pd.read_csv("./data/processed/MEETING_WITH_TEXT_MACRO.csv")
    df["observation_date"] = pd.to_datetime(df["observation_date"])

    # Construction des états
    df = build_dataset_conditional_markov(df)

    # Sélection de quelques variables macro (simplifiées pour l’exemple)
    features = ["VIXCLS", "UNRATE", "CPIAUCSL", "T10Y2Y"]

    # Entraînement de la chaîne de Markov conditionnelle
    model, scaler = train_conditional_markov(df, features, n_splits=5)

    # Matrice de transition conditionnelle moyenne
    compute_conditional_transition_matrix(model, scaler, df, features)
