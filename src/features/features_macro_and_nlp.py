# ======================================
# MERGE_TEXT_MACRO.PY
# ======================================
import pandas as pd

# === 1. Chargement des features NLP déjà calculées ===
nlp = pd.read_csv("./data/processed/FOMC_text_features_sentiment.csv", parse_dates=["meeting_date"])

# === 2. Chargement du dataset macro ===
macro = pd.read_csv("./data/processed/MEETING_BASED.csv", parse_dates=["observation_date"])

# === 3. Sélection des colonnes utiles NLP ===
cols_keep = ["meeting_date", "sentiment_score"] + [c for c in nlp.columns if c.startswith("text_comp_")]
nlp_reduced = nlp[cols_keep].copy()

# === 4. Jointure temporelle sur la date la plus proche ===
merged = pd.merge_asof(
    nlp_reduced.sort_values("meeting_date"),
    macro.sort_values("observation_date"),
    left_on="meeting_date",
    right_on="observation_date",
    direction="nearest"
)

# === 5. Nettoyage et sauvegarde ===
merged.drop(columns=["meeting_date"], inplace=True)
merged = merged.round(2)
merged.to_csv("./data/processed/MEETING_WITH_TEXT_MACRO.csv", index=False)
print("[INFO] Dataset combiné texte + macro → ./data/processed/MEETING_WITH_TEXT_MACRO.csv")
