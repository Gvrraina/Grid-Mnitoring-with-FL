import streamlit as st
import pandas as pd
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from variational_autoencoder import VariationalAutoencoder

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "global_model.pth"
INPUT_DIM = 43
THRESHOLD = 0.255  # from your evaluate.py best_threshold
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    model = VariationalAutoencoder(INPUT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()
st.sidebar.success("✅ VAE model loaded successfully")

# -------------------------------
# LOAD DATA (simulate real-time feed)
# -------------------------------
@st.cache_data
def load_data():
    benign = pd.read_csv("benign_cleaned.csv")
    attack = pd.read_csv("attack_cleaned.csv")
    benign["target"] = 0
    attack["target"] = 1
    data = pd.concat([benign, attack], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    return data

data = load_data()

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def compute_mse(model, batch):
    batch_tensor = torch.tensor(batch.values, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        recon, _, _ = model(batch_tensor)
        mse = torch.mean((recon - batch_tensor) ** 2, dim=1)
    mse = mse.cpu().numpy()
    # Handle NaN or inf values
    mse = np.nan_to_num(mse, nan=0.0, posinf=1.0, neginf=0.0)
    return mse

def anomaly_score(mse):
    # Normalize safely to 0–1 range
    mse = np.nan_to_num(mse, nan=0.0, posinf=1.0, neginf=0.0)
    diff = mse.max() - mse.min()
    if diff == 0:
        return np.zeros_like(mse)
    return np.clip((mse - mse.min()) / diff, 0, 1)


# -------------------------------
# STREAMLIT DASHBOARD
# -------------------------------
st.title("🔍 Real-Time Network Packet Anomaly Monitor (VAE + Federated Learning)")
st.caption("Live simulation using Variational Autoencoder for network traffic anomaly detection.")

placeholder = st.empty()
batch_size = st.sidebar.slider("Batch Size", 64, 512, BATCH_SIZE, step=64)
speed = st.sidebar.slider("Update Interval (sec)", 0.5, 5.0, 1.0, step=0.5)
threshold = st.sidebar.number_input("Reconstruction Threshold", value=THRESHOLD)

# Tracking history
history = []

# Simulation loop
for start in range(0, len(data), batch_size):
    batch = data.iloc[start:start+batch_size].drop(columns=["target"])
    y_true = data.iloc[start:start+batch_size]["target"]

    mse = compute_mse(model, batch)
    scores = anomaly_score(mse)
    preds = (mse > threshold).astype(int)

    df_batch = pd.DataFrame({
        "Index": batch.index,
        "Reconstruction Error": mse,
        "Anomaly Score": scores,
        "Predicted": np.where(preds == 1, "Attack", "Benign"),
        "True": np.where(y_true == 1, "Attack", "Benign")
    })

    history.extend(scores.tolist())

    with placeholder.container():
        st.subheader("📦 Latest Packet Batch")
        st.dataframe(df_batch.head(10), use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Confidence & Error Distribution")
            fig, ax = plt.subplots()
            ax.hist(scores, bins=30)
            ax.axvline(np.mean(scores), color='red', linestyle='--', label='Mean')
            ax.set_title("Anomaly Score Distribution")
            ax.set_xlabel("Score")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        with col2:
            st.subheader("⚡ Latest Batch Results")
            benign_count = (preds == 0).sum()
            attack_count = (preds == 1).sum()
            st.metric("Benign Packets", benign_count)
            st.metric("Detected Attacks", attack_count)
            accuracy = np.mean(preds == y_true.values)
            st.metric("Batch Accuracy", f"{accuracy*100:.2f}%")

        valid_scores = scores[np.isfinite(scores)]
        if len(valid_scores) == 0:
            st.warning("⚠️ No valid scores to display for this batch (all NaN).")
            continue  # skip plotting this batch
       
        

        # Historical trend of reconstruction error
        st.subheader("📊 Live Reconstruction Error Trend")
        fig2, ax2 = plt.subplots()
        ax2.plot(history[-500:], color='blue')
        ax2.set_title("Reconstruction Error Over Time")
        ax2.set_xlabel("Packet Index")
        ax2.set_ylabel("Error")
        st.pyplot(fig2)

    time.sleep(speed)

st.success("✅ Simulation completed.")
