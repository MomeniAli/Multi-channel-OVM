"""
Phase mask parameter utilities and normalization helpers.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default processing configuration mirrors the legacy script.
_PHASE_UNIT_CFG = {
    "normalize_before": False,
    "norm_eps": 1e-6,
    "squash_mode": "tanh",
    "temperature": 1.0,
    "smooth_kernel": 0,
}


def configure_phase_unit(
    *,
    normalize_before: bool,
    norm_eps: float,
    squash_mode: str,
    temperature: float,
    smooth_kernel: int = 0,
) -> None:
    _PHASE_UNIT_CFG.update(
        {
            "normalize_before": normalize_before,
            "norm_eps": norm_eps,
            "squash_mode": squash_mode.lower(),
            "temperature": temperature,
            "smooth_kernel": int(max(0, smooth_kernel)),
        }
    )


def _smooth_phase(tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size < 3:
        return tensor
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    pad = kernel_size // 2
    orig_dtype = tensor.dtype
    if tensor.dim() == 3:
        work = tensor.unsqueeze(0)
    elif tensor.dim() == 4:
        work = tensor
    else:
        raise ValueError("_smooth_phase expects a 3D (C,H,W) or 4D (B,C,H,W) tensor.")
    if work.dtype != torch.float32:
        work = work.float()
    channels = work.size(1)
    weight = torch.ones(channels, 1, kernel_size, kernel_size, device=work.device, dtype=work.dtype)
    weight /= kernel_size * kernel_size
    smoothed = F.conv2d(work, weight, padding=pad, groups=channels)
    if smoothed.dtype != orig_dtype:
        smoothed = smoothed.to(orig_dtype)
    return smoothed if tensor.dim() == 4 else smoothed.squeeze(0)



def phase_to_unit(phase: torch.Tensor) -> torch.Tensor:
    """
    Map raw phase parameters to [0,1] with optional normalization and temperature scaling.
    """
    cfg = _PHASE_UNIT_CFG
    if phase.dim() < 3:
        raise ValueError("phase_to_unit expects a tensor with shape (C,H,W) or (B,C,H,W).")

    phase_proc = phase

    # Normalize over spatial dims only => no cross-channel / cross-batch coupling
    if cfg.get("normalize_before", False):
        eps = float(cfg.get("norm_eps", 1e-6))
        spatial_dims = (-2, -1)  # H, W
        mean_val = phase_proc.mean(dim=spatial_dims, keepdim=True)
        centered = phase_proc - mean_val
        std_val = centered.std(dim=spatial_dims, keepdim=True, unbiased=False)
        phase_proc = centered / (std_val + eps)

    smooth_kernel = int(cfg.get("smooth_kernel", 0))
    if smooth_kernel >= 3:
        phase_proc = _smooth_phase(phase_proc, smooth_kernel)

    temp = float(cfg.get("temperature", 1.0))
    if not math.isfinite(temp) or temp <= 0:
        temp = 1.0

    squash_mode = str(cfg.get("squash_mode", "tanh")).lower()
    phase_scaled = phase_proc / temp

    if squash_mode == "sigmoid":
        return torch.sigmoid(phase_scaled)

    return 0.5 * (torch.tanh(phase_scaled) + 1.0)



class PhaseSystem(nn.Module):
    """
    Container for trainable phase parameters controlling each optical layer.
    """

    def __init__(
        self,
        n_layers: int,
        phase_shape: Tuple[int, ...],
        init_method: str,
        init_sigma: float,
        device: torch.device,
        structural_nonlinearity: bool = False,
        alpha_init: float = 1.0,
        alpha_max: float = 1.0,
        alpha_map_hw: Tuple[int, int] | None = None,
        scff_proto_hw: Tuple[int, int] | None = None,
        scff_proto_count: int | None = None,
    ) -> None:
        super().__init__()
        if len(phase_shape) < 3:
            raise ValueError("phase_shape must be (C, H, W)")
        self.num_channels = int(phase_shape[0])
        params: List[nn.Parameter] = []
        for _ in range(n_layers):
            tensor = torch.zeros(phase_shape, device=device)
            if init_method == "randn_sigma":
                tensor.uniform_(-0.5, 0.5)
            elif init_method == "zeros":
                tensor.zero_()
            elif init_method == "zero_phase":
                target_unit = 1e-4  # how close to zero you want the unit phase
                temp = float(_PHASE_UNIT_CFG.get("temperature", 1.0))
                raw_val = temp * math.atanh(2 * target_unit - 1)
                tensor.fill_(raw_val)
            else:
                tensor.zero_()
            params.append(nn.Parameter(tensor))
        self.structural_nonlinearity = bool(structural_nonlinearity)
        self.alpha_init = float(alpha_init)
        self.alpha_max = max(float(alpha_max), 1e-6)
        if alpha_map_hw is not None:
            if len(alpha_map_hw) != 2:
                raise ValueError("alpha_map_hw must be a length-2 iterable when provided.")
            h_val = max(1, int(alpha_map_hw[0]))
            w_val = max(1, int(alpha_map_hw[1]))
            self.alpha_map_hw = (h_val, w_val)
        else:
            self.alpha_map_hw = None
        if self.structural_nonlinearity:
            frac = self.alpha_init / self.alpha_max if self.alpha_max > 0 else 0.5
            frac = min(max(frac, 1e-6), 1.0 - 1e-6)
            raw_alpha_val = math.log(frac / (1.0 - frac))
            alpha_shape = (
                self.num_channels,
                (self.alpha_map_hw[0] if self.alpha_map_hw is not None else 1),
                (self.alpha_map_hw[1] if self.alpha_map_hw is not None else 1),
            )
            alpha_list = [
                nn.Parameter(torch.full(alpha_shape, raw_alpha_val, device=device))
                for _ in range(n_layers)
            ]
            self.raw_alpha_params = nn.ParameterList(alpha_list)
        else:
            self.raw_alpha_params = None
        self.params = nn.ParameterList(params)
        self.init_method = init_method
        self.init_sigma = init_sigma
        self.scff_proto_hw = tuple(scff_proto_hw) if scff_proto_hw is not None else None
        self.scff_r_bank: torch.Tensor | None
        self.scff_r_inited: torch.Tensor | None
        if self.scff_proto_hw is None:
            self.scff_r_bank = None
            self.scff_r_inited = None
        else:
            h_proto, w_proto = self.scff_proto_hw
            if h_proto <= 0 or w_proto <= 0:
                raise ValueError("scff_proto_hw must contain positive integers.")
            proto_count = n_layers if scff_proto_count is None else int(scff_proto_count)
            if proto_count <= 0:
                raise ValueError("scff_proto_count must be positive when provided.")
            proto_dim = int(h_proto) * int(w_proto)
            proto_shape = (proto_count, self.num_channels, proto_dim)
            proto_init = torch.zeros(proto_shape, device=device)
            self.register_buffer("scff_r_bank", proto_init)
            self.register_buffer(
                "scff_r_inited",
                torch.zeros(int(proto_count), dtype=torch.bool, device=device),
            )

    def phase_parameters(self) -> List[torch.nn.Parameter]:
        return list(self.params)

    def to_unit(self, phase: torch.Tensor) -> torch.Tensor:
        return phase_to_unit(phase)

    def expand_to_batch(
        self,
        phase_param: torch.Tensor,
        batch: int,
        x_input: torch.Tensor | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        """
        Broadcast a (C,H,W) phase tensor across the batch so each item receives one
        of the C phases, repeated as needed to match the batch size.
        """
        C = phase_param.size(0)
        if batch % C != 0:
            raise ValueError(f"Batch size {batch} must be divisible by phase channels {C}")
        reps = batch // C
        if self.structural_nonlinearity:
            if x_input is None:
                raise ValueError("x_input must be provided when structural_nonlinearity is enabled.")
            phase_unit = self.to_unit(phase_param)
            phase_expanded = (
                phase_unit.unsqueeze(0)
                .repeat(reps, 1, 1, 1)
                .reshape(batch, 1, *phase_param.shape[-2:])
            )
            target_hw = phase_param.shape[-2:]
            x_work = x_input
            if x_work.dim() == 3:
                x_work = x_work.unsqueeze(0)
            if x_work.dim() != 4:
                raise ValueError(
                    f"x_input must be 4D (B,1,H,W) when structural_nonlinearity is enabled; got {tuple(x_input.shape)}"
                )
            if x_work.shape[0] != batch:
                raise ValueError(
                    f"x_input batch {x_work.shape[0]} must match expanded phase batch {batch} "
                    f"(channels {C}, reps {reps})."
                )
            if x_work.shape[-2:] != target_hw:
                x_work = F.interpolate(
                    x_work,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            if x_work.shape[1] != 1:
                raise ValueError(
                    f"x_input must have channel dimension 1 for structural_nonlinearity; got {x_work.shape[1]}"
                )
            x_aligned = x_work.to(dtype=phase_param.dtype, device=phase_param.device)
            raw_alpha = None
            if self.raw_alpha_params is not None and layer_idx is not None and 0 <= layer_idx < len(self.raw_alpha_params):
                raw_alpha = self.raw_alpha_params[layer_idx]
            if raw_alpha is None:
                frac = self.alpha_init / self.alpha_max if self.alpha_max > 0 else 0.5
                frac = min(max(frac, 1e-6), 1.0 - 1e-6)
                raw_val = math.log(frac / (1.0 - frac))
                alpha_shape = (
                    self.num_channels,
                    (self.alpha_map_hw[0] if self.alpha_map_hw is not None else 1),
                    (self.alpha_map_hw[1] if self.alpha_map_hw is not None else 1),
                )
                raw_alpha = phase_param.new_full(alpha_shape, raw_val)
            raw_alpha = raw_alpha.to(dtype=phase_param.dtype, device=phase_param.device)
            alpha_layer = self.alpha_max * torch.sigmoid(raw_alpha)
            if alpha_layer.dim() == 3:
                if alpha_layer.shape[-2:] != target_hw:
                    alpha_layer = F.interpolate(
                        alpha_layer.unsqueeze(0),
                        size=target_hw,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
            else:
                alpha_layer = alpha_layer.view(self.num_channels, 1, 1).expand(self.num_channels, *target_hw)
            alpha_expanded = alpha_layer.unsqueeze(0).repeat(reps, 1, 1, 1).reshape(batch, 1, *target_hw)
            return (phase_expanded + (alpha_expanded * x_aligned - alpha_expanded / 2)).clamp(0.0, 1.0)

        phase_unit = self.to_unit(phase_param)  # (C, H, W)
        # (reps, C, H, W) -> (batch, 1, H, W)
        return phase_unit.unsqueeze(0).repeat(reps, 1, 1, 1).reshape(batch, 1, *phase_unit.shape[-2:])
