import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# ============================================================
# 1. Construction du dataset markovien binaire avec lissage temporel
# ============================================================

def build_dataset_with_state(path="./data/processed/DATASET_FINAL_MEETING_BASED.csv", ema_span=3):
    df = pd.read_csv(path)
    df = df.sort_values("observation_date").reset_index(drop=True)

    # États : -1 = baisse, +1 = hausse (suppression du "stable")
    df["state_t"] = np.sign(df["DFF"].diff()).fillna(0)
    df["state_t"] = df["state_t"].replace(0, np.nan).ffill().fillna(0)
    df["state_t1"] = df["state_t"].shift(-1)

    # Suppression du dernier meeting sans futur état
    df = df.dropna(subset=["state_t1"]).reset_index(drop=True)

    # Lissage EMA
    for col in df.columns:
        if col not in ["observation_date", "state_t", "state_t1"]:
            df[f"{col}_EMA{ema_span}"] = df[col].ewm(span=ema_span, adjust=False).mean()

    df = df.ffill().bfill()
    return df


# ============================================================
# 1bis. Ajout des variables retardées (Markov d'ordre k)
# ============================================================

def add_lagged_features(df, features, lags=3):
    """
    Ajoute efficacement les valeurs passées (lags) des variables macro et des états
    sans fragmentation mémoire (concaténation en bloc).
    """
    lagged_dfs = [df]  # on garde l’original en premier
    for lag in range(1, lags + 1):
        shifted = df[features + ["state_t"]].shift(lag)
        shifted.columns = [f"{c}_L{lag}" for c in shifted.columns]
        lagged_dfs.append(shifted)
    df_lagged = pd.concat(lagged_dfs, axis=1)
    df_lagged.dropna(inplace=True)
    return df_lagged



# ============================================================
# 2. Entraînement du modèle
# ============================================================

def train_conditional_markov_binary(df, features):
    print("\n=== Distribution des états (state_t1) ===")
    print(df["state_t1"].value_counts(normalize=True).rename("proportion"))

    X = df[features]
    y = df["state_t1"].apply(lambda x: -1 if x < 0 else 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=8)
    y_true_all, y_pred_all = [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

        print(f"\n--- Fold {fold+1} ---")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, digits=3))

    print("\n=== ÉVALUATION GLOBALE ===")
    print(confusion_matrix(y_true_all, y_pred_all))
    print(classification_report(y_true_all, y_pred_all, digits=3))

    return model, scaler


# ============================================================
# 3. Matrice de transition conditionnelle moyenne
# ============================================================

def build_conditional_transition_matrix_binary(model, scaler, df, features):
    X_scaled = scaler.transform(df[features])
    probs = model.predict_proba(X_scaled)
    classes = model.classes_

    df_probs = pd.DataFrame(probs, columns=[f"P_next_{int(c)}" for c in classes])
    df_concat = pd.concat([df[["state_t"]], df_probs], axis=1)

    states = sorted(df["state_t"].unique())
    mat = pd.DataFrame(index=states, columns=classes, dtype=float)

    for s in states:
        subset = df_concat[df_concat["state_t"] == s]
        mat.loc[s] = subset[[f"P_next_{int(c)}" for c in classes]].mean().values

    mat.index.name = "État courant"
    mat.columns = [f"Vers état {int(c)}" for c in classes]
    print("\n=== MATRICE DE TRANSITION CONDITIONNELLE (MOYENNE) ===")
    print(mat.round(3))
    return mat


# ============================================================
# 4. Prédiction manuelle : test de scénarios
# ============================================================

def predict_next_decision(model, scaler, current_state, latest_macro, feature_names):
    import pandas as pd
    import numpy as np

    data = {col: np.nan for col in feature_names}
    for k, v in latest_macro.items():
        if k in data:
            data[k] = v
    data["state_t"] = current_state

    X = pd.DataFrame([data]).fillna(0)
    X_scaled = scaler.transform(X[feature_names])
    probs = model.predict_proba(X_scaled)[0]
    pred_class = model.predict(X_scaled)[0]

    return {
        "proba_baisse": float(probs[0]),
        "proba_hausse": float(probs[1]),
        "decision": "Hausse" if pred_class == 1 else "Baisse"
    }


# ============================================================
# 5. Exécution principale avec mémoire
# ============================================================

if __name__ == "__main__":
    # 1. Construction du dataset meeting-based
    df = build_dataset_with_state(ema_span=3)

    # 2. Définition des features macro de base
    base_features = [c for c in df.columns if c not in ["observation_date", "state_t", "state_t1"]]

    # 3. Création d'un dataset avec mémoire sur 3 meetings
    df_lagged = add_lagged_features(df, base_features, lags=3)

    # 4. Définition des features finales
    features = [c for c in df_lagged.columns if c not in ["observation_date", "state_t", "state_t1"]]

    # 5. Entraînement
    model, scaler = train_conditional_markov_binary(df_lagged, features)

    # 6. Matrice de transition conditionnelle
    build_conditional_transition_matrix_binary(model, scaler, df_lagged, features)
