"""B0 diagnostic residual CNN; no communication-core imports or normalization."""

import os

import numpy as np
import torch
from torch import nn


PARAMETER_COUNT = 23144
CONV_MAC_PER_FRAME = 23592960


class ResidualCNN(nn.Module):
    """Return Delta_H only, in frozen [B,8,K,T] real/imag channel order."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(8, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 8, 3, stride=1, padding=1),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, h_base):
        if h_base.ndim != 4 or tuple(h_base.shape[1:]) != (8, 64, 16):
            raise ValueError("B0 input must have shape [B,8,64,16]")
        if h_base.dtype != torch.float32 or not bool(torch.isfinite(h_base).all()):
            raise ValueError("B0 requires finite float32 input without normalization")
        return self.layers(h_base)


def spatial_data_mask(data_mask, device=None):
    """Take the actual common data RE mask; this is external to CNN.forward."""
    mask = torch.as_tensor(data_mask, device=device)
    if tuple(mask.shape) == (2, 64, 16):
        if not torch.equal(mask[0], mask[1]):
            raise ValueError("B0 requires the actual common two-stream data mask")
        mask = mask[0]
    if tuple(mask.shape) != (64, 16) or mask.dtype != torch.bool or not bool(mask.any()):
        raise ValueError("nonempty bool data mask [64,16] or [2,64,16] required")
    return mask


def predict_channel(model, h_base, data_mask):
    """Apply the learned residual only on data REs; preserve all pilot-union REs."""
    residual = model(h_base)
    mask = spatial_data_mask(data_mask, h_base.device)[None, None]
    return torch.where(mask, h_base + residual, h_base)


def data_complex_mse(prediction, target, data_mask):
    """Mean (e_Re^2+e_Im^2) over B*Nr*Nt*actual_data_REs, never NMSE."""
    if (prediction.shape != target.shape or prediction.ndim != 4
            or tuple(prediction.shape[1:]) != (8, 64, 16)
            or prediction.dtype != torch.float32 or target.dtype != torch.float32):
        raise ValueError("matching float32 [B,8,64,16] prediction/target required")
    mask = spatial_data_mask(data_mask, prediction.device)
    error = (prediction - target).reshape(-1, 4, 2, 64, 16)
    complex_square_error = error.square().sum(dim=2)
    return complex_square_error[..., mask].mean()


def training_step(model, optimizer, h_base, target, data_mask):
    optimizer.zero_grad(set_to_none=True)
    loss = data_complex_mse(predict_channel(model, h_base, data_mask), target, data_mask)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite B0 training loss")
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def configure_reproducibility(training_seed, cpu_threads=4):
    if (isinstance(training_seed, bool) or not isinstance(training_seed, int)
            or not 0 <= training_seed < 2**32):
        raise ValueError("training_seed must be an integer in [0,2^32)")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(training_seed)
    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return {"numpy_seed": training_seed, "torch_seed": training_seed,
            "cpu_threads": torch.get_num_threads(), "deterministic_algorithms": True,
            "cudnn_benchmark": False, "cudnn_deterministic": True,
            "TF32_enabled": False, "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "claim_scope": "same environment and seed only"}
