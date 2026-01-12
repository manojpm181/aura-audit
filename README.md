# Aura-Audit Internship Evaluation  
## Advanced Responsible AI System

---

## 📌 Overview

This repository contains my submission for the **Aura-Audit Intern Evaluation** at **QAF Lab India**.

The goal of this project is to design and implement a **responsible, end-to-end AI system for customer support**, covering:

- Data cleaning and anonymization  
- Machine learning and neural modeling  
- Reinforcement learning optimization  
- Agentic retrieval (RAG)  
- Bias mitigation and explainability  
- Governance, compliance, and impact assessment  

The implementation strictly follows the **11-step evaluation flow** provided by QAF Lab and is built with **modular, production-ready Python code**.

---

## 📁 Project Structure

![Project Structure](https://github.com/user-attachments/assets/92374a4d-628a-445c-802e-3ecea9d3d624)


---

## 🧠 Evaluation Phases

### **Phase 1 – Foundation (60 min)**

**Implemented in:**  
`src/phase1_foundation_advanced.py`

**Steps:**
1. **Data Normalization & PII Removal**
   - Emails, phone numbers, names, punctuation removed
2. **Feature Engineering**
   - TF-IDF (unigrams + bigrams)
   - PCA for dimensionality reduction
3. **Unsupervised Discovery**
   - K-Means clustering
4. **Semi-Supervised Labeling**
   - Cluster-based label propagation
5. **Supervised Baseline**
   - Random Forest classifier

**Artifacts Generated:**
- Cleaned dataset  
- Clustered dataset  
- Baseline ML model  

---

### **Phase 2 – Neural & Reinforcement Learning (60 min)**

**Implemented in:**  
`src/phase2_neural_rl_advanced.py`

**Steps:**
6. **Neural Network Classifier**
   - MLP with Batch Normalization & Dropout
7. **Bias Mitigation**
   - Class-weighted loss
8. **Reinforcement Learning**
   - Q-Learning with reward shaping for decision optimization

**Artifacts Generated:**
- Trained neural network model  
- Q-learning decision table  

---

### **Phase 3 – Agents & Governance (60 min)**

**Implemented in:**  
`src/phase3_agents_governance_advanced.py`

**Steps:**
9. **RAG Pipeline**
   - FAISS vector store with cosine similarity
10. **Agentic Loop**
    - Multi-step ReAct agent combining retrieval + classification
11. **Auditing & Compliance**
    - SHAP explainability
    - LIME explanations
    - Model Card & Impact Assessment

**Artifacts Generated:**
- FAISS index  
- Explainability outputs  
- `Model_Card_Advanced.txt`  

---

## ⚙️ Installation

### 1️⃣ Clone the repository

    git clone <your-repo-link>
    cd aura_audit
### 2️⃣ Install dependencies

    pip install -r requirements.txt
    pip install pandas numpy scikit-learn torch faiss-cpu shap lime joblib
## ▶️ How to Run (IMPORTANT)

Run the scripts **in this exact order**:

### Phase 1

    python src/phase1_foundation_advanced.py
### Phase 2

    python src/phase2_neural_rl_advanced.py
### Phase 3

    python src/phase3_agents_governance_advanced.py

### 🔁 Reproducibility

Global random seed: 42

Deterministic pipelines for clustering, ML, and neural training

Modular, auditable code structure

### ⚖️ Ethical AI & Governance

PII removal ensures data privacy

Bias mitigation via re-weighting

Explainability via SHAP & LIME

Transparent Model Card and Impact Assessment included

### 👤 Author

Manoj P M
Aura-Audit Internship Candidate
QAF Lab India
