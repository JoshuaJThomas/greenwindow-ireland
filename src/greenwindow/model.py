"""PatchTST (Nie et al., ICLR 2023): channel-independent patched
transformer with RevIN instance normalisation.

Faithful to the paper: every channel is patched and encoded by the same
shared transformer (no cross-channel mixing), and RevIN normalises each
channel by its own lookback statistics - the property that matters most
under this dataset's train->test distribution shift. The forecast head
emits all channels; train.py takes channel 0 (the target) for the loss,
so covariate channels act as auxiliary context only through the shared
weights, exactly as in the multivariate setting of the paper.
"""
import torch
from torch import nn


class RevIN(nn.Module):
    """Reversible instance normalisation over the lookback dimension."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, L, C)
        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        return (x - mean) / std, mean, std

    @staticmethod
    def denormalize(y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        # y: (B, H, C); mean/std: (B, 1, C)
        return y * std + mean


class PatchTST(nn.Module):
    def __init__(self, lookback: int, horizon: int, n_channels: int,
                 patch_len: int = 16, stride: int = 8, d_model: int = 128,
                 n_heads: int = 8, n_layers: int = 3, d_ff: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        if lookback < patch_len:
            raise ValueError("lookback must be >= patch_len")
        self.horizon = horizon
        self.n_channels = n_channels
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (lookback - patch_len) // stride + 1
        self.revin = RevIN()
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(self.n_patches * d_model, horizon)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) -> (B, H, C)
        b, length, c = x.shape
        x, mean, std = self.revin.normalize(x)
        x = x.permute(0, 2, 1).reshape(b * c, length)  # channel-independent
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        tokens = self.dropout(self.embed(patches) + self.pos)
        encoded = self.encoder(tokens)
        out = self.head(encoded.reshape(b * c, -1))  # (B*C, H)
        out = out.reshape(b, c, self.horizon).permute(0, 2, 1)
        return self.revin.denormalize(out, mean, std)
