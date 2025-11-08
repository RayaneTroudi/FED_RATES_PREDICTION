import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# import of dataset by meeting
from src.features.features_engineering_meeting import build_dataset_meeting
from src.features.features_engineering_daily import build_dataset_daily
from src.features.features_config_meeting import get_feature_config


# ============================================================
# 1. Construction du dataset markovien binaire avec lissage temporel
# ============================================================

def build_dataset_with_state(ema_span=3):
    df = build_dataset_daily(config=get_feature_config())
    df = df.sort_values("observation_date").reset_index(drop=True)

    # États : -1 = baisse, +1 = hausse
    df["state_t"] = np.sign(df["DFF"].diff())
    df["state_t"] = df["state_t"].replace(0, np.nan).ffill()
    df["state_t1"] = df["state_t"].shift(-1)

    # Suppression du dernier meeting sans futur état
    df = df.dropna(subset=["state_t", "state_t1"]).reset_index(drop=True)

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
    """Ajoute efficacement les valeurs passées (lags) des variables macro et des états."""
    lagged_dfs = [df]
    for lag in range(1, lags + 1):
        shifted = df[features + ["state_t"]].shift(lag)
        shifted.columns = [f"{c}_L{lag}" for c in shifted.columns]
        lagged_dfs.append(shifted)
    df_lagged = pd.concat(lagged_dfs, axis=1)
    df_lagged.dropna(inplace=True)
    return df_lagged


# ============================================================
# 2. Entraînement du modèle (évaluation globale)
# ============================================================

def train_conditional_markov_binary(df, features):
    print("\n=== Distribution des états (state_t1) ===")
    print(df["state_t1"].value_counts(normalize=True).rename("proportion"))

    X = df[features]
    y = df["state_t1"].apply(lambda x: -1 if x < 0 else 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=8)
    y_true_all, y_pred_all, y_prob_all = [], [], []

    for train_idx, test_idx in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            penalty="l2",
            C=0.5,
            max_iter=2000,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_prob_all.extend(y_prob)

    # === Évaluation globale ===
    print("\n=== ÉVALUATION GLOBALE ===")
    cm = confusion_matrix(y_true_all, y_pred_all)
    report = classification_report(y_true_all, y_pred_all, digits=3)
    auc = roc_auc_score((np.array(y_true_all) == 1).astype(int), y_prob_all)

    print(cm)
    print(report)
    print(f"AUC global : {auc:.3f}")

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
        if len(subset) > 0:
            mat.loc[s] = subset[[f"P_next_{int(c)}" for c in classes]].mean().values
        else:
            mat.loc[s] = [np.nan, np.nan]

    mat.index.name = "État courant"
    mat.columns = [f"Vers état {int(c)}" for c in classes]
    print("\n=== MATRICE DE TRANSITION CONDITIONNELLE (MOYENNE) ===")
    print(mat.round(3))
    return mat


# ============================================================
# 4. Prédiction manuelle : test de scénarios
# ============================================================

def predict_next_decision(model, scaler, current_state, latest_macro, feature_names):
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
# 5. Segmentation par régimes monétaires
# ============================================================

def segment_and_evaluate(df_lagged, features, regime_splits):
    """Entraîne et évalue le modèle Markov conditionnel sur plusieurs sous-périodes."""
    results = {}
    for label, (start, end) in regime_splits.items():
        df_regime = df_lagged[
            (df_lagged["observation_date"] >= start) &
            (df_lagged["observation_date"] <= end)
        ].copy()
        if len(df_regime) < 40:
            continue  # trop court pour TimeSeriesSplit

        print(f"\n==============================")
        print(f"=== RÉGIME : {label.upper()} ===")
        print(f"==============================")
        model, scaler = train_conditional_markov_binary(df_regime, features)
        mat = build_conditional_transition_matrix_binary(model, scaler, df_regime, features)
        results[label] = {"model": model, "scaler": scaler, "matrix": mat}
    return results


# ============================================================
# 6. Visualisation Heatmaps
# ============================================================

def plot_transition_heatmaps(results):
    """Affiche les matrices de transition conditionnelle par régime."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))

    if n == 1:
        axes = [axes]

    for ax, (label, res) in zip(axes, results.items()):
        mat = res["matrix"]
        sns.heatmap(mat.astype(float), annot=True, fmt=".2f", cmap="RdBu_r", vmin=0, vmax=1, ax=ax)
        ax.set_title(label.replace("_", " "))
        ax.set_xlabel("Vers état")
        ax.set_ylabel("État courant")

    plt.suptitle("Matrices de transition conditionnelle par régime", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


# ============================================================
# 7. Exécution principale
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

    # 5. Segmentation par régimes monétaires
    regimes = {
        "Tightening_pre2008": ("1990-01-01", "2007-12-31"),
        "ZLB_QE": ("2008-01-01", "2015-12-31"),
        "Post_QE_COVID": ("2016-01-01", "2025-12-31"),
    }

    results = segment_and_evaluate(df_lagged, features, regimes)

    # 6. Visualisation
    plot_transition_heatmaps(results)
