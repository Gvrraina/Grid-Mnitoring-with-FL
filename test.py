import torch
import pandas as pd
import numpy as np
from variational_autoencoder import VariationalAutoencoder  # Use this if model was trained with VAE

# --- Load model ---
INPUT_DIM = 43
MODEL_PATH = "global_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VariationalAutoencoder(INPUT_DIM).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- Load and prepare data ---
benign_df = pd.read_csv("benign_cleaned.csv")
attack_df = pd.read_csv("attack_cleaned.csv")

benign_df["target"] = 0
attack_df["target"] = 1
data = pd.concat([benign_df, attack_df], axis=0).reset_index(drop=True)

y = data["target"].values
X = data.drop(columns=["target"])

X = X.apply(pd.to_numeric, errors="coerce")
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

# --- Get embeddings ---
with torch.no_grad():
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    mu, logvar = model.encode(X_tensor)
    z = mu.cpu().numpy()  # use the latent mean as representation

# --- Train classifier ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

z_train, z_test, y_train, y_test = train_test_split(z, y, test_size=0.2, random_state=42, stratify=y)

clf = LogisticRegression(max_iter=200, class_weight='balanced')
clf.fit(z_train, y_train)

y_pred = clf.predict(z_test)
y_prob = clf.predict_proba(z_test)[:, 1]

print(classification_report(y_test, y_pred, digits=3))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(np.isnan(mse_benign).sum(), np.isnan(mse_attack).sum())
