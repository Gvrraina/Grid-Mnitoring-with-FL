import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from masked_denoising_autoencoder import MaskedDenoisingAutoencoder

# --- CONFIG ---
INPUT_DIM = 43  # must match training
MODEL_PATH = "global_model.pth"
BENIGN_PATH = "benign_cleaned.csv"
ATTACK_PATH = "attack_cleaned.csv"
THRESHOLD_PERCENTILE = 95  # you can tune this

# --- Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MaskedDenoisingAutoencoder(INPUT_DIM).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("✅ Loaded model successfully.")

# --- Load Data ---
benign_df = pd.read_csv(BENIGN_PATH)
attack_df = pd.read_csv(ATTACK_PATH)

# --- Clean (safety step) ---
for df_name, df in [("Benign", benign_df), ("Attack", attack_df)]:
    before = df.shape
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    after = df.shape
    print(f"✅ {df_name} shape before: {before} → after: {after}")

X_benign = torch.tensor(benign_df.values, dtype=torch.float32).to(device)
X_attack = torch.tensor(attack_df.values, dtype=torch.float32).to(device)
mask = torch.ones_like(X_benign)

# --- Compute MSE reconstruction error ---
def compute_mse(model, data):
    model.eval()
    losses = []
    with torch.no_grad():
        for x in data:
            x = x.unsqueeze(0)  # batch dimension
            output = model(x, torch.ones_like(x))
            mse = torch.mean((output - x) ** 2).item()
            losses.append(mse)
    return np.array(losses)

print("\n🔍 Computing reconstruction errors...")
mse_benign = compute_mse(model, X_benign)
mse_attack = compute_mse(model, X_attack)

# --- Compute threshold ---
threshold = np.percentile(mse_benign, THRESHOLD_PERCENTILE)
print(f"\nMSE Threshold ({THRESHOLD_PERCENTILE}th percentile): {threshold:.6f}")

# --- Classify ---
benign_preds = mse_benign > threshold
attack_preds = mse_attack > threshold

# --- Metrics ---
benign_correct = np.sum(~benign_preds)
attack_correct = np.sum(attack_preds)
total = len(benign_preds) + len(attack_preds)
accuracy = (benign_correct + attack_correct) / total

print(f"\n📊 Evaluation Results:")
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

results_df.to_csv("evaluation_results.csv", index=False)
print("\n💾 Saved detailed results → evaluation_results.csv")
