import torch
import torch.nn as nn

class MaskedDenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, mask):
        noise = torch.randn_like(x) * 0.05
        x_noisy = x + noise * mask
        encoded = self.encoder(x_noisy)
        decoded = self.decoder(encoded)
        return decoded


def masked_mse_loss(input, target, mask):
    diff = (input - target) * mask
    loss = torch.sum(diff ** 2) / torch.sum(mask)
    return loss
