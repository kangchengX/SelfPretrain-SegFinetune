import torch.nn as nn
import torch

class Decoder(nn.Module):
    """MLP decoder for segmentation task."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        """
        Initialize the class.
        
        Args:
            latent_dim (int): embedding vector dimension in ViT.
            hidden_dim (int): hidden non-linear dimension.
            output_dim (int): task output dimension.
        """
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor):
        return self.decoder(x)


class AutoEncoder(nn.Module):
    """Autoencoder for segmentation task."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        """
        Initialize the class.

        Args:
            encoder (nn.Moudule): embedding vector encoder.
            decoder (nn.Moudule): task output decoder.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Forward process.

        Args:
            x (Tensor): the inputs.
            mask (Tensor): the random mask applied to inputs.
        """
        x_masked = mask * x
        encode_patch = self.encoder(x_masked)
        reconstruct = self.decoder(encode_patch)

        return reconstruct
