# ======================================
# BUILD_TEXT_FEATURES.PY
# ======================================
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# === 1. Chargement des textes ===
statements = pd.read_csv("./data/processed/FOMC_statements.csv")

# === 2. TF-IDF + SVD ===
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

X_text_reduced = text_transformer.fit_transform(statements["text"])
cols = [f"text_comp_{i+1}" for i in range(X_text_reduced.shape[1])]
X_text_df = pd.DataFrame(X_text_reduced, columns=cols, index=statements.index)

# === 3. FinBERT sentiment ===
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def finbert_score(text):
    inputs = tokenizer(str(text), return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = softmax(logits.numpy())[0]
    return probs[2] - probs[0]  # positive - negative

statements["sentiment_score"] = statements["text"].apply(finbert_score)

# === 4. Sauvegarde finale ===
final = pd.concat([statements, X_text_df], axis=1)
final.to_csv("./data/processed/FOMC_text_features_sentiment.csv", index=False)
print("[INFO] Features textuelles enregistrées → ./data/processed/FOMC_text_features_sentiment.csv")
