import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from pathlib import Path

# ==============================
# Path setup (IMPORTANT)
# ==============================
BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "support_logs.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# Create folders if they don't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# Step 1: Load and clean data
# ==============================
print("🔹 Step 1: Loading dataset...")

data = pd.read_csv(RAW_DATA_PATH)

# 🔴 IMPORTANT: column name check
# Change 'text' below if your CSV column is named differently
TEXT_COLUMN = "text"

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\S+@\S+', '', text)          # remove emails
    text = re.sub(r'\b\d{10}\b', '', text)       # remove phone numbers
    text = re.sub(r'[^a-z\s]', '', text)         # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data["cleaned_text"] = data[TEXT_COLUMN].apply(preprocess_text)

cleaned_path = PROCESSED_DIR / "cleaned_logs.csv"
data.to_csv(cleaned_path, index=False)

print(f"✅ Step 1 complete: Cleaned data saved to {cleaned_path}")

# ==============================
# Step 2: TF-IDF + PCA + KMeans
# ==============================
print("🔹 Step 2: TF-IDF + PCA + KMeans...")

vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X = vectorizer.fit_transform(data["cleaned_text"]).toarray()

# Reduce dimensions
pca = PCA(n_components=50, random_state=42)
X_reduced = pca.fit_transform(X)

# Clustering
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_reduced)

data["cluster"] = clusters

clustered_path = PROCESSED_DIR / "clustered_logs.csv"
data.to_csv(clustered_path, index=False)

print(f"✅ Step 2 complete: Clustered data saved to {clustered_path}")

# ==============================
# Step 3: Semi-supervised labeling
# ==============================
print("🔹 Step 3: Semi-supervised labeling...")

data["label"] = data["cluster"]

labeled_path = PROCESSED_DIR / "labeled_logs.csv"
data.to_csv(labeled_path, index=False)

print(f"✅ Step 3 complete: Labeled data saved to {labeled_path}")

# ==============================
# Step 4: Random Forest baseline
# ==============================
print("🔹 Step 4: Training Random Forest...")

X_train, X_test, y_train, y_test = train_test_split(
    X, data["label"], test_size=0.2, random_state=42
)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("✅ Step 4 complete: Random Forest trained\n")
print(classification_report(y_test, y_pred))

# ==============================
# Save models
# ==============================
joblib.dump(rf, MODELS_DIR / "rf_baseline.pkl")
joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
joblib.dump(pca, MODELS_DIR / "pca_transformer.pkl")
joblib.dump(kmeans, MODELS_DIR / "kmeans_model.pkl")

print("✅ All models saved successfully (Phase 1 complete)")
