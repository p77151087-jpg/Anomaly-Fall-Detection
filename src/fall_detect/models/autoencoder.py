import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, value):
        return self.decoder(self.encoder(value))


@torch.inference_mode()
def reconstruction_scores(model, normalized, device, batch_size=256, residual_weights=None):
    import numpy as np

    from ..io_utils import ContractError

    model.eval()
    weights = None
    if residual_weights is not None:
        weights = torch.as_tensor(residual_weights, dtype=torch.float32, device=device)
        if (
            weights.shape != (model.input_dim,)
            or not torch.isfinite(weights).all()
            or not (weights > 0).all()
        ):
            raise ContractError(
                "Residual weights must be finite positive values matching input_dim"
            )
    result = []
    for start in range(0, len(normalized), batch_size):
        batch = torch.as_tensor(
            normalized[start : start + batch_size], dtype=torch.float32, device=device
        )
        error = (batch - model(batch)) ** 2
        if weights is not None:
            error = error * weights
        result.extend(error.mean(dim=1).cpu().tolist())
    return np.asarray(result, dtype=np.float64)
