import os
import sys
import flwr as fl
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from variational_autoencoder import VariationalAutoencoder, vae_loss

# ------------------------------
# CONFIG
# ------------------------------
NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
CLIENTS_DIR = "clients_data"

# ------------------------------
# LOAD CLIENT DATA
# ------------------------------
if len(sys.argv) != 2:
    print("Usage: python federated_client.py <client_id>")
    sys.exit(1)

client_id = int(sys.argv[1])
client_path = os.path.join(CLIENTS_DIR, f"client_{client_id}_benign.csv")

if not os.path.exists(client_path):
    raise FileNotFoundError(f"❌ Client data not found: {client_path}")

print(f"📂 Loading data for Client {client_id} → {client_path}")
df = pd.read_csv(client_path)

# Convert to tensor
X_client = torch.tensor(df.values, dtype=torch.float32)
print(f"✅ Client {client_id} data shape: {X_client.shape}")

# DataLoader
train_loader = DataLoader(TensorDataset(X_client), batch_size=BATCH_SIZE, shuffle=True)

# ------------------------------
# MODEL SETUP
# ------------------------------
input_dim = X_client.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VariationalAutoencoder(input_dim=input_dim, hidden_dim=64, latent_dim=16).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ------------------------------
# FLOWER CLIENT DEFINITION
# ------------------------------
class AutoencoderClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, key in enumerate(state_dict.keys()):
            state_dict[key] = torch.tensor(parameters[k])
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        for epoch in range(NUM_EPOCHS):
            total_loss = 0.0
            for x_batch, in self.train_loader:
                x_batch = x_batch.to(self.device)

                self.optimizer.zero_grad()
                recon, mu, logvar = self.model(x_batch)
                loss = vae_loss(recon, x_batch, mu, logvar)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"Client {client_id} | Epoch [{epoch+1}/{NUM_EPOCHS}] | Loss: {avg_loss:.6f}")

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        losses = []

        with torch.no_grad():
            for x_batch, in self.train_loader:
                x_batch = x_batch.to(self.device)
                recon, mu, logvar = self.model(x_batch)
                loss = vae_loss(recon, x_batch, mu, logvar)
                losses.append(loss.item())

        mean_loss = float(np.mean(losses))
        print(f"Client {client_id} | Evaluation Loss: {mean_loss:.6f}")
        return mean_loss, len(self.train_loader.dataset), {}

# ------------------------------
# RUN CLIENT
# ------------------------------
client = AutoencoderClient(model, train_loader, optimizer, device)
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
