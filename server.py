import flwr as fl
import torch
from flwr.common import parameters_to_ndarrays
from masked_denoising_autoencoder import MaskedDenoisingAutoencoder

# --- Configuration ---
INPUT_DIM = 43            # set this to number of features in your benign_cleaned.csv
NUM_ROUNDS = 5
SAVE_PATH = "global_model.pth"

# --- Custom FedAvg Strategy with Model Saving ---
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, save_path="global_model.pth", total_rounds=5, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.save_path = save_path
        self.total_rounds = total_rounds  # keep track of total rounds manually

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(rnd, results, failures)

        if aggregated_parameters is not None:
            print(f"✅ Aggregated parameters for round {rnd}")

            # Save model only on the last round
            if rnd == self.total_rounds:
                print("💾 Saving final global model...")
                weights = parameters_to_ndarrays(aggregated_parameters)
                params_dict = zip(self.model.state_dict().keys(), weights)
                state_dict = {k: torch.tensor(v) for k, v in params_dict}
                self.model.load_state_dict(state_dict, strict=True)
                torch.save(self.model.state_dict(), self.save_path)
                print(f"✅ Global model saved at {self.save_path}")

        return aggregated_parameters, metrics


def start_server():
    # Initialize model (needed for saving)
    model = MaskedDenoisingAutoencoder(INPUT_DIM)
    strategy = SaveModelStrategy(model=model, save_path=SAVE_PATH, total_rounds=NUM_ROUNDS)

    print("🚀 Starting Flower server...")
    history = fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    print("🏁 Training complete.")
    if hasattr(history, "losses_distributed"):
        print("Loss history:", history.losses_distributed)


if __name__ == "__main__":
    start_server()
