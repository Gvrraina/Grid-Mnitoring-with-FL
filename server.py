import flwr as fl
import torch
from flwr.common import parameters_to_ndarrays
from variational_autoencoder import VariationalAutoencoder

# -----------------------------
# CONFIGURATION
# -----------------------------
INPUT_DIM = 43            # number of features in your dataset (excluding target)
NUM_ROUNDS = 5
SAVE_PATH = "global_model.pth"

# -----------------------------
# CUSTOM STRATEGY WITH MODEL SAVE
# -----------------------------
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, save_path="global_model.pth", total_rounds=5, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.save_path = save_path
        self.total_rounds = total_rounds

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(rnd, results, failures)

        if aggregated_parameters is not None:
            print(f"✅ Aggregated parameters for round {rnd}")

            # Save model on the final round
            if rnd == self.total_rounds:
                print("💾 Saving final global model weights...")
                weights = parameters_to_ndarrays(aggregated_parameters)
                params_dict = zip(self.model.state_dict().keys(), weights)
                state_dict = {k: torch.tensor(v) for k, v in params_dict}
                self.model.load_state_dict(state_dict, strict=True)
                torch.save(self.model.state_dict(), self.save_path)
                print(f"✅ Global model saved → {self.save_path}")

        return aggregated_parameters, metrics


# -----------------------------
# SERVER STARTUP
# -----------------------------
def start_server():
    # Initialize model (only for saving aggregated weights)
    model = VariationalAutoencoder(input_dim=INPUT_DIM, hidden_dim=64, latent_dim=16)

    strategy = SaveModelStrategy(
        model=model,
        save_path=SAVE_PATH,
        total_rounds=NUM_ROUNDS,
    )

    print("🚀 Starting Flower Server...")
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
    print("🏁 Federated Training Completed.")


if __name__ == "__main__":
    start_server()
