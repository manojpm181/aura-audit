import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import joblib

# ------------------------------
# Step 0: Load Phase 1 models and data
# ------------------------------
rf = joblib.load('models/rf_baseline.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
pca = joblib.load('models/pca_transformer.pkl')
kmeans = joblib.load('models/kmeans_model.pkl')

data = pd.read_csv('data/processed/labeled_logs.csv')

# TF-IDF features
X = vectorizer.transform(data['cleaned_text']).toarray()
X_reduced = pca.transform(X)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(data['label'])

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# ------------------------------
# Step 1: Advanced MLP Model
# ------------------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.bn2 = nn.BatchNorm1d(hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, output_dim)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

model = MLPClassifier(input_dim=X_train.shape[1], hidden_dim=128, output_dim=len(le.classes_))

# Compute class weights for bias mitigation
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train.numpy()), y=y_train.numpy())
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------------
# Step 2: Train MLP
# ------------------------------
epochs = 30
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    y_pred = torch.argmax(model(X_test), axis=1)
from sklearn.metrics import classification_report
print("\nMLP Performance:\n", classification_report(y_test.numpy(), y_pred.numpy()))

# Save model
torch.save(model.state_dict(), 'models/mlp_model_advanced.pth')
print("Step 2 complete: Advanced MLP trained and saved.")

# ------------------------------
# Step 3: Q-Learning for decision optimization
# ------------------------------
n_states = len(le.classes_)
n_actions = len(le.classes_)
q_table = np.zeros((n_states, n_actions))
alpha, gamma, epsilon = 0.1, 0.9, 0.2
episodes = 1000

for episode in range(episodes):
    state = np.random.randint(0, n_states)
    action = np.random.randint(0, n_actions) if np.random.rand() < epsilon else np.argmax(q_table[state])
    reward = 1 if action == state else 0
    q_table[state, action] += alpha * (reward + gamma * np.max(q_table[state]) - q_table[state, action])

np.save('models/q_table.npy', q_table)
print("Step 3 complete: Q-Learning table generated and saved.")

# ------------------------------
# Phase 2 complete
# ------------------------------
print("Phase 2 complete: Neural network, Q-Learning, and bias mitigation ready for Phase 3.")
