import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# === 1. CHARGEMENT ===
merged = pd.read_csv("./data/processed/FOMC_statements.csv")

# === 2. PIPELINE TF-IDF + SVD ===
text_transformer = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1,3),
        max_features=1000,
        stop_words="english",
        token_pattern=r"\b[a-zA-Z]{3,}\b"
    )),
    ("log1p", FunctionTransformer(np.log1p)),
    ("svd", TruncatedSVD(n_components=10, random_state=0))
])

X_text_reduced = text_transformer.fit_transform(merged["text"])
cols = [f"text_comp_{i+1}" for i in range(X_text_reduced.shape[1])]
X_text_df = pd.DataFrame(X_text_reduced, columns=cols, index=merged.index)

# === 3. SCORE DE SENTIMENT (FinBERT) ===
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def finbert_score(text):
    inputs = tokenizer(str(text), return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = softmax(logits.numpy())[0]
    # classes: [negative, neutral, positive]
    return probs[2] - probs[0]

merged["sentiment_score"] = merged["text"].apply(finbert_score)

# === 4. CONCATENATION ===
final = pd.concat([merged, X_text_df], axis=1)
final.to_csv("./data/processed/FOMC_text_features_sentiment.csv", index=False)

# === 5. CHARGEMENT DES DONNÉES MACRO ===
nlp = pd.read_csv("./data/processed/FOMC_text_features_sentiment.csv", parse_dates=["meeting_date"])
macro = pd.read_csv("./data/processed/DAILY_BASED.csv", parse_dates=["observation_date"])

# === 6. SÉLECTION DES COLONNES ===
cols_keep = ["meeting_date", "sentiment_score"] + [c for c in nlp.columns if c.startswith("text_comp_")]
nlp_reduced = nlp[cols_keep].copy()

# === 7. JOINTURE SUR LA DATE LA PLUS PROCHE ===
merged = pd.merge_asof(
    nlp_reduced.sort_values("meeting_date"),
    macro.sort_values("observation_date"),
    left_on="meeting_date",
    right_on="observation_date",
    direction="nearest"
)

# === 8. NETTOYAGE ET SAUVEGARDE ===
merged.drop(columns=["meeting_date"], inplace=True)
merged = merged.round(2)
merged.to_csv("./data/processed/MEETING_WITH_TEXT_MACRO.csv", index=False)
