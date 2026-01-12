import pandas as pd
import numpy as np
import torch
import faiss
import joblib
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer

# ------------------------------
# Step 0: Load models and data
# ------------------------------
data = pd.read_csv('data/processed/cleaned_logs.csv')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
pca = joblib.load('models/pca_transformer.pkl')
mlp_model = torch.load('models/mlp_model_advanced.pth', map_location='cpu')

# Load label encoder
from sklearn.preprocessing import LabelEncoder
labels = pd.read_csv('data/processed/labeled_logs.csv')['label']
le = LabelEncoder()
le.fit(labels)

# ------------------------------
# Step 1: Create FAISS vector store (cosine similarity)
# ------------------------------
X = vectorizer.transform(data['cleaned_text']).toarray().astype('float32')

# Normalize vectors for cosine similarity
X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)

index = faiss.IndexFlatIP(X_norm.shape[1])
index.add(X_norm)
print(f"Step 1 complete: FAISS vector store created with {index.ntotal} entries.")

# ------------------------------
# Step 2: Multi-step ReAct Agent
# ------------------------------
def retrieve_logs(query, top_k=5):
    q_vec = vectorizer.transform([query]).toarray().astype('float32')
    q_vec_norm = q_vec / np.linalg.norm(q_vec, axis=1, keepdims=True)
    distances, indices = index.search(q_vec_norm, top_k)
    return data.iloc[indices[0]]['cleaned_text'].tolist()

def classify_intent(texts):
    mlp_model.eval()
    X_input = vectorizer.transform(texts).toarray()
    X_input_pca = pca.transform(X_input).astype(np.float32)
    X_tensor = torch.tensor(X_input_pca, dtype=torch.float32)
    with torch.no_grad():
        y_pred = mlp_model(X_tensor)
    labels_pred = torch.argmax(y_pred, axis=1).numpy()
    return [le.inverse_transform([label])[0] for label in labels_pred]

# Example query
query = "Customer unable to login"
retrieved = retrieve_logs(query)
predicted = classify_intent(retrieved)
print("Step 2 complete: ReAct agent prediction:", predicted)

# ------------------------------
# Step 3: Explainability (SHAP + LIME)
# ------------------------------
# SHAP
explainer_shap = shap.Explainer(mlp_model, torch.tensor(pca.transform(X[:100]), dtype=torch.float32))
shap_values = explainer_shap(torch.tensor(pca.transform(X[:5]), dtype=torch.float32))
print("Step 3a complete: SHAP values computed.")

# LIME
explainer_lime = LimeTabularExplainer(
    training_data=pca.transform(X[:100]),
    feature_names=[f"f{i}" for i in range(pca.transform(X[:100]).shape[1])],
    class_names=le.classes_,
    mode='classification'
)
i = 0
exp = explainer_lime.explain_instance(pca.transform(X[i])[0], mlp_model.forward)
print("Step 3b complete: LIME explanation generated.")

# ------------------------------
# Step 4: Model Card & Impact Assessment
# ------------------------------
model_card = """
Model Card: Advanced Customer Support AI

Purpose:
- End-to-end AI system for customer support log classification
- Multi-step ReAct agent + neural network + clustering

Data:
- 1,000 anonymized support logs

Bias Mitigation:
- Class-weighted MLP training
- Q-Learning optimized decisions

Explainability:
- SHAP and LIME used for model interpretation

Limitations:
- Small dataset, may not generalize to all domains
- Assumes clean textual input

Usage:
- Internal support analytics
- Not for external deployment without additional validation

Impact Assessment:
- Fairness: Bias mitigation reduces class imbalance
- Safety: PII removed from all logs
- Transparency: Model explanations available via SHAP & LIME
"""

with open('Model_Card_Advanced.txt', 'w') as f:
    f.write(model_card)

print("Step 4 complete: Model Card & Impact Assessment saved.")
print("Phase 3 complete: Agents, Explainability, and Compliance ready.")
