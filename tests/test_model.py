import pytest
import torch

from greenwindow.model import PatchTST, RevIN


def small(n_channels=1) -> PatchTST:
    torch.manual_seed(0)
    return PatchTST(lookback=64, horizon=24, n_channels=n_channels,
                    d_model=32, n_heads=4, n_layers=2, d_ff=64).eval()


def test_output_shape():
    out = small(n_channels=5)(torch.randn(3, 64, 5))
    assert out.shape == (3, 24, 5)


def test_revin_round_trip():
    x = torch.randn(2, 64, 3) * 50 + 300
    z, mean, std = RevIN().normalize(x)
    assert torch.allclose(RevIN.denormalize(z, mean, std), x, atol=1e-3)


def test_forecast_follows_shifts_and_rescaling_of_its_input():
    # RevIN makes the model level- and scale-free: a grid running 100
    # g/kWh dirtier, or swinging twice as hard, gets a forecast shifted
    # or stretched the same way. This is what lets a model trained on
    # 2020-25 data cope with a greener grid.
    model, x = small(), torch.randn(1, 64, 1) * 40 + 250
    with torch.no_grad():
        base = model(x)
        assert torch.allclose(model(x + 100), base + 100, atol=1e-2)
        assert torch.allclose(model(2 * x), 2 * base, atol=1e-2)


def test_lookback_shorter_than_a_patch_is_rejected():
    with pytest.raises(ValueError):
        PatchTST(lookback=8, horizon=4, n_channels=1, patch_len=16)
