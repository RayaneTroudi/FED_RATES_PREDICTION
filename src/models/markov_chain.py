import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score


# ============================================================
# 1. Dataset Markov d’ordre 1 (statu quo = baisse)
# ============================================================

def build_dataset_markov_order1(df):
    df = df.sort_values("observation_date").reset_index(drop=True)

    # États
    df["state_t"] = np.sign(df["DFF"].diff())
    df["state_t"] = df["state_t"].replace(0, np.nan).ffill()

    # État futur à prédire
    df["state_t1"] = np.sign(df["DFF"].shift(-1) - df["DFF"])
    df["state_t1"] = df["state_t1"].replace(0, -1)  # statu quo = baisse

    # Suppression des lignes incomplètes
    df = df.dropna(subset=["state_t", "state_t1"]).reset_index(drop=True)
    return df


# ============================================================
# 2. Régression logistique conditionnelle d’ordre 1
# ============================================================

def fit_conditional_logit_order1(df, features, n_splits=5):
    X = df[features + ["state_t"]].copy()
    y = (df["state_t1"] == 1).astype(int)  # 1 = hausse, 0 = baisse/statu quo

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_true, oof_pred, oof_prob = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled), 1):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = LogisticRegressionCV(
            Cs=5, cv=3, penalty="elasticnet", solver="saga",
            l1_ratios=[0.2], class_weight="balanced", max_iter=10000, random_state=42
        )
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        oof_true[test_idx] = y_test
        oof_pred[test_idx] = y_pred
        oof_prob[test_idx] = y_prob

        print(f"Fold {fold}: Accuracy = {accuracy_score(y_test, y_pred):.3f}")

    auc = roc_auc_score(oof_true, oof_prob)
    print("\n=== ÉVALUATION GLOBALE ===")
    print(confusion_matrix(oof_true, oof_pred))
    print(classification_report(oof_true, oof_pred, digits=3))
    print(f"AUC (globale) : {auc:.3f}")

    model_final = LogisticRegressionCV(
        Cs=5, cv=3, penalty="elasticnet", solver="saga",
        l1_ratios=[0.2], class_weight="balanced", max_iter=10000, random_state=42
    )
    model_final.fit(X_scaled, y)
    return model_final, scaler, oof_prob


# ============================================================
# 3. Matrice de transition conditionnelle d’ordre 1
# ============================================================

def build_transition_matrix_order1(df, model, scaler, features):
    X_scaled = scaler.transform(df[features + ["state_t"]])
    df["prob_up"] = model.predict_proba(X_scaled)[:, 1]

    states = [-1, 1]
    M = {}

    for s in states:
        subset = df[df["state_t"] == s]
        if subset.empty:
            continue
        p_up = np.clip(subset["prob_up"].mean(), 0.0, 1.0)
        M[s] = np.array([[1 - p_up, p_up],
                         [1 - p_up, p_up]])

    print("\n=== MATRICE DE TRANSITION CONDITIONNELLE (ordre 1) ===")
    for s in M:
        print(f"\nÉtat courant {s}:")
        print(pd.DataFrame(M[s],
              columns=["→ -1 (baisse/statu quo)", "→ +1 (hausse)"],
              index=["-1 (baisse/statu quo)", "+1 (hausse)"]))
    return M


# ============================================================
# 4. Simulation Markov d’ordre 1
# ============================================================

def simulate_markov_order1(M, init_state=1, n_steps=50, seed=42):
    np.random.seed(seed)
    states = [-1, 1]
    s = init_state
    seq = [s]
    for _ in range(n_steps):
        if s not in M:
            s = np.random.choice(states)
        p_up = M[s][0, 1]
        p_up = np.clip(p_up, 0.0, 1.0)
        s = np.random.choice(states, p=[1 - p_up, p_up])
        seq.append(s)
    return np.array(seq)


# ============================================================
# 5. Comparaison simulation vs réel + durée moyenne
# ============================================================

def mean_run_length(seq, state):
    seq = np.array(seq)
    runs = np.diff(np.where(np.concatenate(([seq[0] != state],
                                           seq[:-1] != seq[1:],
                                           [True])))[0])[::2]
    return runs.mean() if len(runs) else np.nan


def compare_simulated_vs_real(df, M, n_steps=50, init_state=1):
    real_seq = df["state_t1"].values[-n_steps:]
    sim_seq = simulate_markov_order1(M, init_state=init_state, n_steps=n_steps)

    plt.figure(figsize=(12,5))
    plt.step(range(len(sim_seq)), sim_seq, where="mid", lw=2, color="tab:orange", label="Simulé (Markov ordre 1)")
    plt.step(range(len(real_seq)), real_seq, where="mid", lw=2, color="black", label="Réalité (Fed)")
    plt.yticks([-1, 1], ["Baisse / statu quo", "Hausse"])
    plt.xlabel("Meetings")
    plt.title("Chaîne de Markov conditionnelle d’ordre 1 – Fed")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\n=== DURÉE MOYENNE DES CYCLES (en meetings) ===")
    for s in [-1, 1]:
        real_dur = mean_run_length(real_seq, s)
        sim_dur = mean_run_length(sim_seq, s)
        print(f"Régime {s:+d} : Réel = {real_dur:.2f} | Simulé = {sim_dur:.2f}")


# ============================================================
# 7. Étude de corrélation des variables explicatives
# ============================================================

def analyze_correlations(df, features):
    corr = df[features].corr(method='spearman')

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Corrélation de Pearson')
    plt.xticks(range(len(features)), features, rotation=90)
    plt.yticks(range(len(features)), features)
    plt.title("Matrice de corrélation des variables explicatives")
    plt.tight_layout()
    plt.show()

    # Affichage des corrélations les plus fortes (hors diagonale)
    corr_pairs = (
        corr.where(~np.eye(corr.shape[0], dtype=bool))
        .stack()
        .sort_values(ascending=False)
    )
    print("\n=== PAIRES DE VARIABLES LES PLUS CORRÉLÉES ===")
    print(corr_pairs.head(10))

# Appel après la définition des features :

# ============================================================
# 6. Pipeline complet
# ============================================================

if __name__ == "__main__":
    df = pd.read_csv("./data/processed/MEETING_WITH_TEXT_MACRO.csv")
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = build_dataset_markov_order1(df)

    macro_vars = ["VIXCLS", "UNRATE", "CPIAUCSL", "T10Y2Y"]
    features = [c for c in df.columns if any(v in c for v in macro_vars)
                and ("DIFF" in c or "MA" in c or "MOM" in c)]

    print(f"\n=== FEATURES UTILISÉES ({len(features)}) ===")
    print(features)

    model, scaler, y_prob = fit_conditional_logit_order1(df, features, n_splits=5)
    M = build_transition_matrix_order1(df, model, scaler, features)

    # Simulation vs réalité
    compare_simulated_vs_real(df, M, n_steps=100, init_state=1)
    analyze_correlations(df, features)



