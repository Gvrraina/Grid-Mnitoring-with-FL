import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from variational_autoencoder import VariationalAutoencoder, vae_loss
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score, precision_score, f1_score

torch.manual_seed(42)
np.random.seed(42)
 
# --- CONFIG ---
INPUT_DIM = 43  # must match training
MODEL_PATH = "global_model.pth"
BENIGN_PATH = "benign_cleaned.csv"
ATTACK_PATH = "attack_cleaned.csv"

# --- Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VariationalAutoencoder(INPUT_DIM).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("✅ Loaded VAE model successfully.")

# --- Load and Clean Data ---
def clean_data(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

benign_df = clean_data(pd.read_csv(BENIGN_PATH))
attack_df = clean_data(pd.read_csv(ATTACK_PATH))

print(f"✅ Benign shape: {benign_df.shape}")
print(f"✅ Attack shape: {attack_df.shape}")

# --- Apply SMOTE to balance benign and attack data ---
print("\n⚖️ Applying SMOTE to balance benign and attack data...")

# Combine benign and attack datasets
X_combined = np.vstack([benign_df.values, attack_df.values])
y_combined = np.hstack([np.zeros(len(benign_df)), np.ones(len(attack_df))])

# Apply SMOTE
smote = SMOTE(random_state=42, sampling_strategy='auto', k_neighbors=5)
X_balanced, y_balanced = smote.fit_resample(X_combined, y_combined)

# Split back into balanced benign/attack sets
X_benign_bal = X_balanced[y_balanced == 0]
X_attack_bal = X_balanced[y_balanced == 1]

print(f"✅ After SMOTE → Benign: {len(X_benign_bal)}, Attack: {len(X_attack_bal)}")

# Convert to tensors
X_benign = torch.tensor(X_benign_bal, dtype=torch.float32).to(device)
X_attack = torch.tensor(X_attack_bal, dtype=torch.float32).to(device)


# --- Compute VAE Reconstruction Error ---
def compute_vae_mse(model, data, batch_size=512):
    model.eval()
    mse_list = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            recon, mu, logvar = model(batch)
            mse_batch = torch.mean((recon - batch) ** 2, dim=1)  # per-sample MSE
            mse_list.extend(mse_batch.cpu().numpy())
    return np.array(mse_list)

print("\n🔍 Computing reconstruction errors...")
mse_benign = compute_vae_mse(model, X_benign)
mse_attack = compute_vae_mse(model, X_attack)

# --- Adaptive Threshold Search ---
percentiles = np.arange(80, 100, 1)
best_recall, best_precision, best_f1, best_p, best_threshold = 0, 0, 0, 0, 0

for p in percentiles:
    threshold = np.percentile(mse_benign, p)
    benign_preds = mse_benign > threshold
    attack_preds = mse_attack > threshold

    benign_labels = np.zeros_like(benign_preds)
    attack_labels = np.ones_like(attack_preds)
    preds = np.concatenate([benign_preds, attack_preds]).astype(int)
    labels = np.concatenate([benign_labels, attack_labels])

    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels, preds)

    if recall > best_recall:
        best_recall, best_precision, best_f1, best_p, best_threshold = recall, precision, f1, p, threshold

print(f"\n🔥 Best threshold (for recall): {best_threshold:.6f} at {best_p}th percentile")
print(f"Recall: {best_recall*100:.2f}%, Precision: {best_precision*100:.2f}%, F1: {best_f1*100:.2f}%")

# --- Final classification using best threshold ---
benign_preds = mse_benign > best_threshold
attack_preds = mse_attack > best_threshold

benign_correct = np.sum(~benign_preds)
attack_correct = np.sum(attack_preds)
accuracy = (benign_correct + attack_correct) / (len(benign_preds) + len(attack_preds))

print(f"\n📊 Final Evaluation with Optimized Threshold:")
print(f"Benign correctly classified: {benign_correct}/{len(benign_preds)}")
print(f"Attack correctly classified: {attack_correct}/{len(attack_preds)}")
print(f"Overall Accuracy: {accuracy*100:.2f}%")

# --- Save detailed results ---
results_df = pd.DataFrame({
    "Type": ["Benign"] * len(mse_benign) + ["Attack"] * len(mse_attack),
    "MSE": np.concatenate([mse_benign, mse_attack]),
    "Prediction": np.concatenate([
        np.where(benign_preds, "Attack", "Benign"),
        np.where(attack_preds, "Attack", "Benign")
    ])
})
results_df.to_csv("vae_evaluation_results.csv", index=False, mode="w", encoding="utf-8")
print("\n💾 Saved detailed results → vae_evaluation_results.csv")
print(np.isnan(mse_benign).sum(), np.isnan(mse_attack).sum())
