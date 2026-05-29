"""
Training script for patchified-input ONN with per-layer digital mixing.

Differences vs. optical_onn_training.py:
- Input can be either:
  - patchified channels: (1, 16, H, W)
  - native single-channel image: (1, 1, h, w), projected by trainable Conv2d(1->16)
    and then bilinearly upsampled to configured (H, W)
- Channels are treated as the physical batch: (16, 1, H, W).
- Digital mixers (Conv2d) mix channels between optical layers (n_layers - 1 mixers).
- Readout performs per-channel regression and fuses channel predictions.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from pickle import UnpicklingError
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from model_training.onn_online_training.models import Surrogate_OpticalNet_Unet
from model_training.onn_online_training.dataloader_patch import build_patch_loaders
from model_training.onn_online_training.utils import encoding_x_phase_physical, percentile_scale_spatial
from model_training.onn_online_training.surrogate_model_training import make_optical_sys, reset_live_plot
from model_training.onn_online_training.config_loader import (
    load_training_config,
    resolve_onn_seed,
    write_config_snapshot,
)
from model_training.onn_online_training.phase_system import (
    PhaseSystem,
    configure_phase_unit,
    phase_to_unit,
)
from model_training.onn_online_training.optical_bridge import OpticalBridge
from model_training.onn_online_training.training_viz import (
    USE_NOTEBOOK_PROGRESS,
    NotebookProgressBar,
    ensure_dark_tqdm_theme,
    create_training_viz_layout,
    init_metrics_plot_state,
    update_metrics_plot,
    init_phase_viz_state,
    update_phase_viz,
)
from model_training.onn_online_training.fine_tunning import (
    accumulate_fine_tune_samples,
    init_fine_tune_state,
    maybe_fine_tune_surrogate,
    reset_fine_tune_viz,
)

try:
    from IPython.display import display
except Exception:  # pragma: no cover - optional in non-notebook environments
    display = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _prepare_data_loaders(data_cfg: Dict[str, object]):
    dataset_name = str(data_cfg.get("dataset", "fashion_mnist"))
    batch_size = int(data_cfg.get("batch_size", 1))
    num_channels = int(data_cfg.get("num_channels", 16))
    hw_cfg = data_cfg.get("hw", (112, 112))
    hw = (int(hw_cfg[0]), int(hw_cfg[1]))
    use_rot4_tiling = bool(data_cfg.get("use_rot4_tiling", True))
    method = str(data_cfg.get("method", "patchify")).strip().lower()
    overlap_percent = float(data_cfg.get("overlap_percent", 0.0))
    augment = bool(data_cfg.get("augment", False))
    data_root = str(data_cfg.get("data_root", "./data"))
    return build_patch_loaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_channels=num_channels,
        hw=hw,
        use_rot4_tiling=use_rot4_tiling,
        method=method,
        overlap_percent=overlap_percent,
        augment=augment,
        data_root=data_root,
    )


def _load_surrogate(sur_cfg: Dict[str, object], device: torch.device, channel_num: int):
    enc_canvas_hw = tuple(sur_cfg.get("enc_canvas_hw", [168, 168]))
    surrogate_dir_default = Path(
        "/home/adminlwe/Documents/lwe-opu/experiment/model_training/pre_trained_model_save/Surrogate_OpticalNet_Unet/exp2"
    )
    surrogate_dir = Path(sur_cfg.get("save_dir", str(surrogate_dir_default)))
    surrogate_dir.mkdir(parents=True, exist_ok=True)
    default_surrogate_ckpt = surrogate_dir / "ckpt_best.pt"
    surrogate_ckpt_raw = sur_cfg.get("ckpt_path", str(default_surrogate_ckpt))
    surrogate_ckpt_path = Path(surrogate_ckpt_raw)
    if not surrogate_ckpt_path.is_absolute():
        surrogate_ckpt_path = surrogate_dir / surrogate_ckpt_path

    surrogate = Surrogate_OpticalNet_Unet(enc_canvas_hw=enc_canvas_hw, num_channels=channel_num).to(device)
    if surrogate_ckpt_path.exists():
        try:
            ckpt_obj = torch.load(surrogate_ckpt_path, map_location=device, weights_only=True)
        except (TypeError, RuntimeError, UnpicklingError) as err:
            print(
                "[warn] `torch.load(..., weights_only=True)` failed for surrogate checkpoint "
                f"({err}). Retrying with weights_only=False; ensure this checkpoint is trusted."
            )
            ckpt_obj = torch.load(surrogate_ckpt_path, map_location=device)
        if isinstance(ckpt_obj, dict) and "model_state" in ckpt_obj:
            state_dict = ckpt_obj["model_state"]
        else:
            state_dict = ckpt_obj
        surrogate.load_state_dict(state_dict)
    else:
        print(
            f"[warn] Surrogate checkpoint not found at {surrogate_ckpt_path}; "
            "starting from current model weights."
        )
    surrogate.eval()
    for param in surrogate.parameters():
        param.requires_grad_(False)
    return surrogate, surrogate_ckpt_path, enc_canvas_hw


def _resolve_output_hw(
    tiles_cfg: Dict[str, object], enc_canvas_hw: Tuple[int, int], optical_layer: nn.Module
) -> Tuple[int, int]:
    optical_hw: Optional[Tuple[int, int]] = None
    if hasattr(optical_layer, "Npix"):
        raw_hw = getattr(optical_layer, "Npix")
        if isinstance(raw_hw, (tuple, list)) and len(raw_hw) >= 2:
            optical_hw = (int(raw_hw[0]), int(raw_hw[1]))
    cfg_out_hw = tiles_cfg.get("out_hw")
    if cfg_out_hw is not None:
        out_hw = tuple(cfg_out_hw)
    elif optical_hw is not None:
        out_hw = optical_hw
    else:
        out_hw = tuple(enc_canvas_hw)
    if optical_hw is not None and out_hw != optical_hw:
        out_hw = optical_hw
    return out_hw


def _build_optimizer(module: nn.Module, opt_name: str, lr: float, weight_decay: float):
    opt_name = opt_name.lower()
    if opt_name == "adamw":
        return torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_name == "adam":
        return torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {opt_name}")


def _ensure_initial_lrs(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        if "initial_lr" not in group:
            group["initial_lr"] = group.get("lr", 0.0)


def _build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
    last_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine annealing with linear warmup and floor at min_lr."""
    warmup_steps = max(int(warmup_steps), 0)
    total_steps = max(int(total_steps), warmup_steps + 1)
    base_lr = float(base_lr)
    min_lr = float(min_lr)
    min_lr = min(min_lr, base_lr)
    min_ratio = (min_lr / base_lr) if base_lr > 0 else 0.0

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = min(
            1.0,
            (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps)),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
        last_epoch=last_epoch,
    )


def _parse_ckpt_every(value):
    ckpt_auto_mode = False
    if isinstance(value, str):
        ckpt_every_str = value.strip().lower()
        if ckpt_every_str == "auto":
            ckpt_auto_mode = True
            ckpt_every = None
        else:
            ckpt_every = int(float(ckpt_every_str))
    else:
        ckpt_every = int(value)
    if not ckpt_auto_mode and ckpt_every is not None and ckpt_every < 0:
        raise ValueError("onn.ckpt_every must be non-negative when specified as an integer")
    return ckpt_auto_mode, ckpt_every


def _load_checkpoint(
    path: Path,
    device: torch.device,
    phase_system: PhaseSystem,
    history: Dict[str, List[float]],
    mixers: nn.ModuleList,
    channel_readouts: nn.ModuleList | None = None,
    input_adapter: nn.Module | None = None,
    extra_tensors: Dict[str, torch.nn.Parameter] | None = None,
    fusion_weights_raw: torch.nn.Parameter | None = None,
) -> Tuple[int, int, float]:
    resume_epoch = 0
    it_counter = 0
    best_eval = float("inf")
    if not path.exists():
        return resume_epoch, it_counter, best_eval
    try:
        ckpt_loaded = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt_loaded = torch.load(path, map_location=device)
    except Exception as exc:
        print(f"[warn] Failed to load checkpoint {path}: {exc}")
        return resume_epoch, it_counter, best_eval
    if not isinstance(ckpt_loaded, dict):
        return resume_epoch, it_counter, best_eval

    def _copy_params(saved_list):
        assigned = False
        phase_params = phase_system.phase_parameters()
        if isinstance(saved_list, (list, tuple)) and len(saved_list) == len(phase_params):
            for param, saved in zip(phase_params, saved_list):
                if isinstance(saved, torch.Tensor):
                    param.data.copy_(saved.to(device))
                    assigned = True
        return assigned

    state_dict = ckpt_loaded.get("phase_system_state")
    copied = False
    if isinstance(state_dict, dict):
        try:
            phase_system.load_state_dict(state_dict, strict=False)
            copied = True
        except RuntimeError as exc:
            print(f"[warn] Failed to load full phase_system state_dict: {exc}")
    if not copied:
        loaded_raw = ckpt_loaded.get("phase_params_raw")
        copied = _copy_params(loaded_raw)
        if not copied:
            saved_params = ckpt_loaded.get("phase_params")
            _copy_params(saved_params)

    mixer_state = ckpt_loaded.get("mixer_state")
    if isinstance(mixer_state, dict):
        try:
            mixers.load_state_dict(mixer_state, strict=False)
        except RuntimeError as exc:
            print(f"[warn] Failed to load mixer_state: {exc}")
    if channel_readouts is not None:
        readout_state = ckpt_loaded.get("readout_heads_state")
        if isinstance(readout_state, dict):
            try:
                channel_readouts.load_state_dict(readout_state, strict=False)
            except RuntimeError as exc:
                print(f"[warn] Failed to load readout_heads_state: {exc}")

    if input_adapter is not None:
        input_adapter_state = ckpt_loaded.get("input_adapter_state")
        if isinstance(input_adapter_state, dict):
            try:
                input_adapter.load_state_dict(input_adapter_state, strict=False)
            except RuntimeError as exc:
                print(f"[warn] Failed to load input_adapter_state: {exc}")

    if fusion_weights_raw is not None:
        saved_fusion = ckpt_loaded.get("fusion_weights_raw")
        if isinstance(saved_fusion, torch.Tensor):
            fusion_weights_raw.data.copy_(saved_fusion.to(device))

    saved_history = ckpt_loaded.get("history")
    if isinstance(saved_history, dict):
        for key in history.keys():
            value = saved_history.get(key)
            if isinstance(value, list):
                history[key] = value
    if extra_tensors:
        for key, param in extra_tensors.items():
            saved_val = ckpt_loaded.get(key)
            if isinstance(saved_val, torch.Tensor):
                try:
                    param.data.copy_(saved_val.to(device))
                except Exception as exc:
                    print(f"[warn] Failed to load saved tensor for {key}: {exc}")
    resume_epoch = int(ckpt_loaded.get("epoch", 0))
    it_counter = int(ckpt_loaded.get("iter", 0))
    eval_history = history.get("eval_loss", [])
    if eval_history:
        finite_vals = [
            float(v)
            for v in eval_history
            if isinstance(v, (int, float)) and math.isfinite(float(v))
        ]
        if finite_vals:
            best_eval = min(finite_vals)
    return resume_epoch, it_counter, best_eval


class _MetricBuffer:
    """Accumulates scalar values and returns their average once the window is filled."""

    def __init__(self, window_size: int):
        self.window = max(int(window_size), 1)
        self.values: List[float] = []
        self.last_iter: Optional[int] = None

    def add(self, value: float, iteration: int) -> None:
        self.values.append(float(value))
        self.last_iter = iteration

    def ready(self) -> bool:
        return len(self.values) >= self.window

    def pop_average(self, force: bool = False) -> Tuple[Optional[int], Optional[float]]:
        if not self.values:
            return None, None
        if not force and len(self.values) < self.window:
            return None, None
        avg_val = float(sum(self.values) / len(self.values))
        last_iter = self.last_iter
        self.values.clear()
        self.last_iter = None
        return last_iter, avg_val


def _save_checkpoint(
    path: Path,
    epoch_idx: int,
    iter_idx: int,
    phase_system: PhaseSystem,
    mixers: nn.ModuleList,
    channel_readouts: nn.ModuleList | None,
    input_adapter: nn.Module | None,
    history: Dict[str, List[float]],
    cfg: Dict[str, object],
    extra_state: Dict[str, torch.Tensor] | None = None,
    fusion_weights_raw: torch.Tensor | None = None,
) -> None:
    phase_params = phase_system.phase_parameters()
    ckpt_obj = {
        "phase_params_raw": [p.detach().cpu() for p in phase_params],
        "phase_params": [phase_to_unit(p).detach().cpu() for p in phase_params],
        "phase_system_state": phase_system.state_dict(),
        "mixer_state": mixers.state_dict(),
        "history": history,
        "cfg": cfg,
        "iter": iter_idx,
        "epoch": epoch_idx,
    }
    if input_adapter is not None:
        ckpt_obj["input_adapter_state"] = input_adapter.state_dict()
    if channel_readouts is not None:
        ckpt_obj["readout_heads_state"] = channel_readouts.state_dict()
    if fusion_weights_raw is not None:
        ckpt_obj["fusion_weights_raw"] = fusion_weights_raw.detach().cpu()
    if extra_state:
        for key, tensor in extra_state.items():
            if isinstance(tensor, torch.Tensor):
                ckpt_obj[key] = tensor.detach().cpu()
    torch.save(ckpt_obj, path)


def _build_mixers(n_layers: int, channel_num: int, mixer_cfg: Dict[str, object]) -> nn.ModuleList:
    k = int(mixer_cfg.get("kernel_size", 1))
    if k <= 0:
        raise ValueError("mixer.kernel_size must be positive")
    bias = bool(mixer_cfg.get("bias", True))
    padding = k // 2
    n_mixers = max(int(n_layers) - 1, 0)
    mixers = nn.ModuleList(
        [
            nn.Conv2d(channel_num, channel_num, kernel_size=k, padding=padding, bias=bias)
            for _ in range(n_mixers)
        ]
    )
    return mixers


def _build_input_adapter(
    data_method: str,
    channel_num: int,
    input_conv_cfg: Dict[str, object],
) -> nn.Module | None:
    if data_method != "upsampler":
        return None
    k = int(input_conv_cfg.get("kernel_size", 3))
    if k <= 0:
        raise ValueError("input_conv.kernel_size must be positive")
    bias = bool(input_conv_cfg.get("bias", True))
    padding = int(input_conv_cfg.get("padding", k // 2))
    if padding < 0:
        raise ValueError("input_conv.padding must be non-negative")
    return nn.Conv2d(1, channel_num, kernel_size=k, padding=padding, bias=bias)


def _prepare_input_channels(
    x_batch: torch.Tensor,
    *,
    channel_num: int,
    data_method: str,
    input_adapter: nn.Module | None,
    upsample_hw: Tuple[int, int] | None = None,
) -> torch.Tensor:
    if x_batch.dim() != 4:
        raise ValueError(f"Expected x_batch (1,C,H,W), got {tuple(x_batch.shape)}")
    if x_batch.size(0) != 1:
        raise ValueError(f"x_batch must have batch size 1, got {tuple(x_batch.shape)}")
    if data_method == "patchify":
        if x_batch.size(1) != channel_num:
            raise ValueError(f"Patchify mode expects x_batch (1,{channel_num},H,W), got {tuple(x_batch.shape)}")
        return x_batch
    if data_method == "upsampler":
        if x_batch.size(1) != 1:
            raise ValueError(f"Upsampler mode expects x_batch (1,1,H,W), got {tuple(x_batch.shape)}")
        if input_adapter is None:
            raise ValueError("input_adapter is required when data.method='upsampler'")
        x_channels = input_adapter(x_batch)
        if upsample_hw is not None:
            target_hw = (int(upsample_hw[0]), int(upsample_hw[1]))
            if x_channels.shape[-2:] != target_hw:
                x_channels = F.interpolate(
                    x_channels,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
        if x_channels.size(1) != channel_num:
            raise ValueError(
                f"input_adapter output must be (1,{channel_num},H,W), got {tuple(x_channels.shape)}"
            )
        return x_channels
    raise ValueError(f"Unsupported data.method '{data_method}' (expected patchify/upsampler)")


def _apply_mix_norm(x: torch.Tensor, mixer_cfg: Dict[str, object]) -> torch.Tensor:
    norm_mode = str(mixer_cfg.get("norm", "clamp")).lower()
    if norm_mode in {"clamp", "relu"}:
        return F.relu(x)
    if norm_mode == "percentile":
        peak = float(mixer_cfg.get("percentile_peak", 0.99))
        q = float(mixer_cfg.get("percentile_q", 0.99))
        eps = float(mixer_cfg.get("percentile_eps", 1e-6))
        return percentile_scale_spatial(x, peak=peak, q=q, eps=eps)
    if norm_mode in {"none", "off", "null"}:
        return x
    raise ValueError(f"Unknown mixer.norm '{norm_mode}' (expected relu/clamp/percentile/none).")


def _fuse_channel_predictions(
    preds_ch: torch.Tensor,
    fuse_mode: str,
    fusion_weights_raw: Optional[torch.Tensor],
) -> torch.Tensor:
    mode = str(fuse_mode).lower()
    if mode == "mean":
        return preds_ch.mean(dim=0)
    if mode == "weighted":
        if fusion_weights_raw is None:
            raise ValueError("fusion_weights_raw is required for weighted fuse_mode")
        w = torch.softmax(fusion_weights_raw, dim=0)
        return (w[:, None, None] * preds_ch).sum(dim=0)
    raise ValueError(f"Unsupported fuse_mode '{fuse_mode}'")


def _reconstruct_full_input(
    x_batch: torch.Tensor,
    grid: int = 4,
    *,
    use_rot4_tiling: bool = True,
) -> torch.Tensor:
    if x_batch.dim() != 4:
        raise ValueError(f"Expected x_batch (1,C,H,W) for reconstruction, got {tuple(x_batch.shape)}")
    B, C, H, W = x_batch.shape
    if B != 1:
        raise ValueError(f"Expected batch size 1 for reconstruction, got {B}")
    if grid * grid != C:
        raise ValueError(f"grid {grid}x{grid} does not match channel count {C}")
    patch_h, patch_w = H // grid, W // grid
    x_ch = x_batch.view(C, 1, H, W)

    if use_rot4_tiling:
        # Viz-only inverse for deterministic 2x2 rotation tiling:
        # each channel contains [0,90;180,270] rotations before full-canvas resize.
        # Recover an original-like patch by taking the top-left quadrant (0 deg).
        tile_h, tile_w = patch_h * 2, patch_w * 2
        if tile_h > H or tile_w > W:
            raise ValueError(
                f"Expected channel image at least {(tile_h, tile_w)} for tiled reconstruction, got {(H, W)}"
            )
        tiled = F.interpolate(x_ch, size=(tile_h, tile_w), mode="bilinear", align_corners=False)
        patches = tiled[:, :, :patch_h, :patch_w]
    else:
        # Without rotation tiling, each channel is directly interpolated from a patch.
        # Downsample back to patch resolution for visualization.
        patches = F.interpolate(x_ch, size=(patch_h, patch_w), mode="bilinear", align_corners=False)

    canvas = x_batch.new_zeros(1, 1, H, W)
    idx = 0
    for r in range(grid):
        for c in range(grid):
            r0 = r * patch_h
            c0 = c * patch_w
            canvas[0, 0, r0:r0 + patch_h, c0:c0 + patch_w] = patches[idx, 0]
            idx += 1
    return canvas


def _build_center_mask(
    output_hw: Tuple[int, int],
    mask_hw: Tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    out_h, out_w = int(output_hw[0]), int(output_hw[1])
    mask_h, mask_w = int(mask_hw[0]), int(mask_hw[1])
    if mask_h <= 0 or mask_w <= 0:
        raise ValueError("readout.mask_h/readout.mask_w must be positive.")
    if mask_h > out_h or mask_w > out_w:
        raise ValueError(
            f"Center mask {(mask_h, mask_w)} is larger than output map {(out_h, out_w)}."
        )
    r0 = (out_h - mask_h) // 2
    c0 = (out_w - mask_w) // 2
    r1 = r0 + mask_h
    c1 = c0 + mask_w
    mask = torch.zeros(1, 1, out_h, out_w, device=device, dtype=dtype)
    mask[:, :, r0:r1, c0:c1] = 1.0
    return mask, (r0, r1, c0, c1)


def _readout_regression(
    y_final: torch.Tensor,
    *,
    mask_hw: Tuple[int, int],
    pool_hw: Tuple[int, int],
    channel_readouts: nn.ModuleList,
    linear_in_features: int,
    fuse_mode: str,
    fusion_weights_raw: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if y_final.dim() != 4 or y_final.size(1) != 1:
        raise ValueError(f"Expected y_final (C,1,H,W), got {tuple(y_final.shape)}")
    channel_num = int(y_final.size(0))
    if len(channel_readouts) != channel_num:
        raise ValueError(
            f"Expected {channel_num} readout heads, got {len(channel_readouts)}."
        )

    center_mask, (r0, r1, c0, c1) = _build_center_mask(
        tuple(y_final.shape[-2:]),
        mask_hw,
        device=y_final.device,
        dtype=y_final.dtype,
    )
    masked_region = y_final[:, :, r0:r1, c0:c1] * center_mask[:, :, r0:r1, c0:c1]
    pooled = F.avg_pool2d(masked_region, kernel_size=pool_hw, stride=pool_hw)
    features = pooled.flatten(start_dim=1)
    if features.size(1) != int(linear_in_features):
        raise ValueError(
            f"readout.linear_in_features={linear_in_features}, "
            f"but pooled features are {features.size(1)}."
        )

    preds_per_channel: List[torch.Tensor] = []
    for ch_idx, head in enumerate(channel_readouts):
        preds_per_channel.append(F.relu(head(features[ch_idx:ch_idx + 1])))
    preds_ch = torch.stack(preds_per_channel, dim=0)  # (C, 1, 30)
    pred = _fuse_channel_predictions(preds_ch, fuse_mode, fusion_weights_raw)
    outside = y_final * (1.0 - center_mask)
    penalty_loss = outside.pow(2).mean()
    return pred, preds_ch, penalty_loss, center_mask


def _init_keypoint_viz_state(output_widget: Optional[object] = None) -> Dict[str, object]:
    return {
        "handle": None,
        "fig": None,
        "axes": None,
        "num_examples": 0,
        "examples_per_row": 2,
        "output": output_widget,
    }


def _points_to_pixels(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = points.reshape(-1, 2)
    # Facial keypoints are normalized in [0,1]; map back to 96x96 pixels.
    xs = flat[:, 0] * 96.0
    ys = flat[:, 1] * 96.0
    return xs, ys


def _update_keypoint_viz(
    state: Dict[str, object],
    images: torch.Tensor,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    channel_maps: torch.Tensor,
    *,
    channel_label: str = "Channel 6 output",
    sample_sources: Optional[List[str]] = None,
    mask_h: int = 60,
    mask_w: int = 60,
    pool_h: int = 10,
    pool_w: int = 10,
) -> None:
    if images.dim() == 3:
        images = images.unsqueeze(1)
    if images.dim() != 4:
        raise ValueError(f"Expected images (N,1,H,W), got {tuple(images.shape)}")
    n_samples = int(images.size(0))
    if n_samples <= 0:
        return
    # Improve readability in notebooks: Times-style text and clearer math rendering.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
        }
    )

    images_np = images.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    if channel_maps.dim() == 4 and channel_maps.size(1) == 1:
        channel_maps = channel_maps.squeeze(1)
    if channel_maps.dim() != 3:
        raise ValueError(f"Expected channel_maps (N,H,W), got {tuple(channel_maps.shape)}")
    if int(channel_maps.size(0)) != n_samples:
        raise ValueError(
            f"channel_maps batch must match images batch: {channel_maps.size(0)} vs {n_samples}"
        )
    channel_maps_np = channel_maps.detach().cpu().numpy()

    if (
        state["fig"] is None
        or state["num_examples"] != n_samples
        or int(state.get("examples_per_row", 1)) != 2
    ):
        examples_per_row = 2
        rows = int(math.ceil(n_samples / float(examples_per_row)))
        cols = examples_per_row * 2
        fig, axes = plt.subplots(rows, cols, figsize=(8.8 * examples_per_row, 4.2 * rows))
        axes = np.atleast_2d(axes)
        handle = state.get("handle")
        if handle is None:
            output_widget = state.get("output")
            if output_widget is not None:
                with output_widget:
                    if display is not None:
                        handle = display(fig, display_id=True)
            elif display is not None:
                handle = display(fig, display_id=True)
        state.update(
            {
                "fig": fig,
                "axes": axes,
                "handle": handle,
                "num_examples": n_samples,
                "examples_per_row": 2,
            }
        )

    fig = state["fig"]
    axes = np.atleast_2d(state["axes"])
    fig.patch.set_facecolor("white")
    for ax in axes.ravel():
        ax.clear()
        ax.axis("off")

    for idx in range(n_samples):
        row_idx = idx // 2
        pair_idx = idx % 2
        ax_out = axes[row_idx, pair_idx * 2]
        ax_face = axes[row_idx, pair_idx * 2 + 1]

        face = images_np[idx, 0]
        height, width = int(face.shape[0]), int(face.shape[1])
        true_x, true_y = _points_to_pixels(y_true_np[idx])
        pred_x, pred_y = _points_to_pixels(y_pred_np[idx])

        ax_out.clear()
        ax_out.axis("off")
        ch_map = channel_maps_np[idx]
        h_out, w_out = int(ch_map.shape[0]), int(ch_map.shape[1])
        ax_out.imshow(ch_map, cmap="magma")
        ax_out.set_title(f"{channel_label} #{idx + 1}")
        ax_face.imshow(face, cmap="gray", vmin=0.0, vmax=1.0)
        ax_face.scatter(
            true_x,
            true_y,
            c="#00c853",
            s=56,
            marker="o",
            edgecolors="black",
            linewidths=0.45,
            label="True",
            alpha=0.95,
        )
        ax_face.scatter(
            pred_x,
            pred_y,
            c="#ff3d00",
            s=56,
            marker="x",
            linewidths=1.7,
            label="Pred",
            alpha=0.95,
        )
        ax_face.set_xlim(0, width - 1)
        ax_face.set_ylim(height - 1, 0)
        source_label = "Eval"
        if sample_sources is not None and idx < len(sample_sources):
            source_label = str(sample_sources[idx])
        ax_face.set_title(f"{source_label} image #{idx + 1}")
        ax_face.axis("off")
        if idx == 0:
            ax_face.legend(loc="upper right", fontsize=10, frameon=True)

        if mask_h <= h_out and mask_w <= w_out:
            r0 = (h_out - int(mask_h)) // 2
            c0 = (w_out - int(mask_w)) // 2
            r1 = r0 + int(mask_h)
            c1 = c0 + int(mask_w)
            rect = plt.Rectangle(
                (c0 - 0.5, r0 - 0.5),
                c1 - c0,
                r1 - r0,
                fill=False,
                edgecolor="#00e5ff",
                linewidth=1.6,
            )
            ax_out.add_patch(rect)
            if pool_h > 0 and pool_w > 0:
                n_rows = int(mask_h) // int(pool_h)
                n_cols = int(mask_w) // int(pool_w)
                for r in range(1, n_rows):
                    y_line = r0 + r * int(pool_h) - 0.5
                    ax_out.plot([c0 - 0.5, c1 - 0.5], [y_line, y_line], color="#00e5ff", linewidth=0.8, alpha=0.9)
                for c in range(1, n_cols):
                    x_line = c0 + c * int(pool_w) - 0.5
                    ax_out.plot([x_line, x_line], [r0 - 0.5, r1 - 0.5], color="#00e5ff", linewidth=0.8, alpha=0.9)

    fig.tight_layout(pad=1.0)
    fig.canvas.draw_idle()
    handle = state.get("handle")
    if handle is not None and hasattr(handle, "update"):
        handle.update(fig)


def _forward_with_mix(
    x_phys: torch.Tensor,
    phase_system: PhaseSystem,
    bridge: OpticalBridge,
    mixers: nn.ModuleList,
    mixer_cfg: Dict[str, object],
    *,
    track_grads: bool,
) -> Tuple[
    torch.Tensor,
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
]:
    x_current = x_phys
    layer_inputs: List[torch.Tensor] = []
    layer_phases: List[torch.Tensor] = []
    layer_outputs: List[torch.Tensor] = []
    layer_inputs_pre: List[torch.Tensor] = []

    base_canvas_for_phase: Optional[torch.Tensor] = None
    if getattr(phase_system, "structural_nonlinearity", False):
        B0 = x_phys.size(0)
        dummy_phase = torch.zeros(B0, 1, *bridge.enc_canvas_hw, device=x_phys.device, dtype=x_phys.dtype)
        base_canvas_for_phase, _ = encoding_x_phase_physical(
            x_phys,
            dummy_phase,
            canvas_hw=bridge.enc_canvas_hw,
            x_mode=bridge.modes[0],
            x_norm_mod=bridge.x_norm_modes[0],
        )

    for idx, phase_param in enumerate(phase_system.phase_parameters()):
        B = x_current.size(0)
        layer_inputs_pre.append(x_current.detach())
        layer_mode = bridge.modes[idx]
        layer_norm_mode = bridge.x_norm_modes[idx]
        x_canvas_for_phase: Optional[torch.Tensor] = None
        if getattr(phase_system, "structural_nonlinearity", False):
            x_canvas_for_phase = base_canvas_for_phase
        phase_batch = phase_system.expand_to_batch(
            phase_param,
            B,
            x_input=x_canvas_for_phase,
            layer_idx=idx,
        )
        y_layer = bridge._apply_layer(
            idx,
            x_current,
            phase_batch,
            layer_mode,
            layer_norm_mode,
            track_grads=track_grads,
        )
        y_layer = bridge._apply_gain(y_layer, idx)
        x_encoded = bridge.encoded_cache[idx] if bridge.encoded_cache[idx] is not None else y_layer
        phase_encoded = bridge.phase_cache[idx] if bridge.phase_cache[idx] is not None else phase_batch
        layer_inputs.append(x_encoded.detach())
        layer_phases.append(phase_encoded.detach())
        layer_outputs.append(y_layer.detach())

        if idx < len(mixers):
            y_ch = y_layer.squeeze(1).unsqueeze(0)  # (1, 16, H, W)
            y_mix = mixers[idx](y_ch)
            y_mix = _apply_mix_norm(y_mix, mixer_cfg)
            x_current = y_mix.squeeze(0).unsqueeze(1)  # (16, 1, H, W)
        else:
            x_current = y_layer

    return x_current, layer_inputs, layer_phases, layer_outputs, layer_inputs_pre


def _evaluate(
    val_loader,
    device: torch.device,
    phase_system: PhaseSystem,
    bridge: OpticalBridge,
    mixers: nn.ModuleList,
    channel_readouts: nn.ModuleList,
    input_adapter: nn.Module | None,
    mixer_cfg: Dict[str, object],
    readout_cfg: Dict[str, object],
    channel_num: int,
    data_method: str,
    input_hw: Tuple[int, int] | None,
    *,
    fusion_weights_raw: torch.Tensor | None = None,
    fuse_mode: str = "mean",
    outside_mask_penalty_weight: float = 5.0e-1,
) -> Tuple[float, float]:
    total_loss = 0.0
    total_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            labels_val = y_val.to(device=device, dtype=torch.float32)
            if labels_val.dim() == 1:
                labels_val = labels_val.unsqueeze(0)
            x_channels = _prepare_input_channels(
                x_val,
                channel_num=channel_num,
                data_method=data_method,
                input_adapter=input_adapter,
                upsample_hw=input_hw,
            )
            x_phys = x_channels.squeeze(0).unsqueeze(1)
            y_final, _, _, _, _ = _forward_with_mix(
                x_phys,
                phase_system,
                bridge,
                mixers,
                mixer_cfg,
                track_grads=False,
            )
            pred_val, _, penalty_loss, _ = _readout_regression(
                y_final,
                mask_hw=(readout_cfg["mask_h"], readout_cfg["mask_w"]),
                pool_hw=(readout_cfg["pool_h"], readout_cfg["pool_w"]),
                channel_readouts=channel_readouts,
                linear_in_features=readout_cfg["linear_in_features"],
                fuse_mode=fuse_mode,
                fusion_weights_raw=fusion_weights_raw,
            )
            mse_loss = F.mse_loss(pred_val, labels_val)
            rmse_loss = torch.sqrt(mse_loss + 1.0e-12)
            loss = rmse_loss + float(outside_mask_penalty_weight) * penalty_loss
            total_loss += float(loss.item())
            total_mse += float(mse_loss.item())
            total_samples += int(labels_val.size(0))

    if total_samples <= 0:
        return float("nan"), float("nan")
    eval_loss = total_loss / float(total_samples)
    eval_mse = total_mse / float(total_samples)
    return eval_loss, eval_mse


def train(
    layer: Optional[nn.Module] = None,
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, object]] = None,
) -> None:
    cfg, cfg_path = load_training_config(config_path, overrides)
    seed = resolve_onn_seed(cfg, default=1337)
    print(f"Using seed: {seed}")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    channel_num = int(cfg.get("channel_num", 16))
    data_cfg = cfg.get("data", {})
    hw_cfg = data_cfg.get("hw", (112, 112))
    input_hw = (int(hw_cfg[0]), int(hw_cfg[1]))
    data_method = str(data_cfg.get("method", "patchify")).strip().lower()
    if data_method not in {"patchify", "upsampler"}:
        raise ValueError("data.method must be 'patchify' or 'upsampler'")
    use_rot4_tiling = bool(data_cfg.get("use_rot4_tiling", True))
    batch_size = int(data_cfg.get("batch_size", 1))
    if batch_size != 1:
        raise ValueError("data.batch_size must be 1 for this training script.")
    if int(data_cfg.get("num_channels", channel_num)) != channel_num:
        data_cfg["num_channels"] = channel_num
    train_loader, val_loader = _prepare_data_loaders(data_cfg)

    sur_cfg = cfg.get("surrogate", {})
    surrogate, surrogate_ckpt_path, enc_canvas_hw = _load_surrogate(sur_cfg, device, channel_num)

    onn_cfg = cfg.get("onn", {})
    in_silico_mode = bool(onn_cfg.get("in_silico_mode", False))
    optical_layer = layer
    if optical_layer is None and (not in_silico_mode):
        raise ValueError("train() requires a physical optical layer instance unless in_silico_mode=True.")
    if in_silico_mode:
        optical_sys = None
    else:
        optical_sys = make_optical_sys(optical_layer)

    n_layers = int(onn_cfg.get("n_layers", 3))
    phase_mask_shape_list = list(
        onn_cfg.get(
            "phase_mask_shape",
            [channel_num, enc_canvas_hw[0], enc_canvas_hw[1]],
        )
    )
    phase_mask_shape_list[0] = channel_num
    phase_mask_shape = tuple(phase_mask_shape_list)
    phase_elements = 1
    for dim in phase_mask_shape:
        phase_elements *= int(dim)
    n_optical_params = int(n_layers) * phase_elements
    init_method = onn_cfg.get("phase_init", "zeros")
    init_sigma = float(onn_cfg.get("phase_init_sigma", 0.05))
    use_amp = bool(onn_cfg.get("use_amp", True))
    opt_name = onn_cfg.get("opt", "adamw")
    lr = float(onn_cfg.get("lr", 1e-2))
    lr_min = float(onn_cfg.get("lr_min", 0.0))
    warmup_iters = int(onn_cfg.get("warmup_iters", 0))
    lr_scheduler_name = str(onn_cfg.get("lr_scheduler", "cosine")).lower()
    weight_decay = float(onn_cfg.get("weight_decay", 0.0))
    accum_steps = max(1, int(onn_cfg.get("accum_steps", 1)))
    grad_clip_raw = onn_cfg.get("grad_clip_norm", None)
    grad_clip_norm: Optional[float]
    try:
        grad_clip_norm = float(grad_clip_raw) if grad_clip_raw is not None else None
    except Exception:
        grad_clip_norm = None
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        grad_clip_norm = None
    structural_nonlinearity = bool(onn_cfg.get("structural_nonlinearity", False))
    alpha_init = float(onn_cfg.get("alpha_init", 1.0))
    alpha_max = float(onn_cfg.get("alpha_max", 1.0))
    alpha_map_hw_raw = onn_cfg.get("alpha_map_hw", None)
    alpha_map_hw: Tuple[int, int] | None = None
    if alpha_map_hw_raw is not None:
        if isinstance(alpha_map_hw_raw, (list, tuple)) and len(alpha_map_hw_raw) >= 2:
            alpha_map_hw = (int(alpha_map_hw_raw[0]), int(alpha_map_hw_raw[1]))
        else:
            raise ValueError("onn.alpha_map_hw must be a length-2 list/tuple like [H, W] when provided.")

    phase_proc_cfg = cfg.get("phase_processing", {})
    configure_phase_unit(
        normalize_before=bool(phase_proc_cfg.get("normalize_before", False)),
        norm_eps=float(phase_proc_cfg.get("norm_eps", 1e-6)),
        squash_mode=str(phase_proc_cfg.get("squash", "tanh")),
        temperature=float(phase_proc_cfg.get("temperature", 1.0)),
        smooth_kernel=int(phase_proc_cfg.get("smooth_kernel", 0)),
    )

    readout_raw = cfg.get("readout", {})
    readout_cfg = {
        "mask_h": int(readout_raw.get("mask_h", 60)),
        "mask_w": int(readout_raw.get("mask_w", 60)),
        "pool_h": int(readout_raw.get("pool_h", 10)),
        "pool_w": int(readout_raw.get("pool_w", 10)),
        "num_channels": int(readout_raw.get("num_channels", channel_num)),
        "linear_in_features": int(readout_raw.get("linear_in_features", 36)),
        "linear_out_features": int(readout_raw.get("linear_out_features", 30)),
    }
    if readout_cfg["num_channels"] != channel_num:
        raise ValueError(
            f"readout.num_channels ({readout_cfg['num_channels']}) must match channel_num ({channel_num})."
        )
    if readout_cfg["mask_h"] <= 0 or readout_cfg["mask_w"] <= 0:
        raise ValueError("readout.mask_h/readout.mask_w must be positive")
    if readout_cfg["pool_h"] <= 0 or readout_cfg["pool_w"] <= 0:
        raise ValueError("readout.pool_h/readout.pool_w must be positive")
    if readout_cfg["mask_h"] % readout_cfg["pool_h"] != 0 or readout_cfg["mask_w"] % readout_cfg["pool_w"] != 0:
        raise ValueError("readout.mask_h/mask_w must be divisible by readout.pool_h/pool_w")
    expected_in_features = (
        (readout_cfg["mask_h"] // readout_cfg["pool_h"]) * (readout_cfg["mask_w"] // readout_cfg["pool_w"])
    )
    if readout_cfg["linear_in_features"] != expected_in_features:
        raise ValueError(
            f"readout.linear_in_features={readout_cfg['linear_in_features']} does not match pooled "
            f"feature count {expected_in_features}."
        )
    if readout_cfg["linear_out_features"] != 30:
        raise ValueError(
            "Facial keypoint regression expects readout.linear_out_features=30."
        )

    outside_mask_penalty_weight = float(readout_raw.get("outside_mask_penalty_weight", 1.0e-2))
    learnable_channel_average = bool(readout_raw.get("learnable_channel_average", True))
    fuse_mode = "weighted" if learnable_channel_average else "mean"
    fusion_weights_raw: Optional[nn.Parameter] = None
    if fuse_mode == "weighted":
        init_tensor = torch.zeros(channel_num, device=device)
        fusion_weights_raw = nn.Parameter(init_tensor)
    channel_readouts = nn.ModuleList(
        [
            nn.Linear(readout_cfg["linear_in_features"], readout_cfg["linear_out_features"])
            for _ in range(channel_num)
        ]
    ).to(device)

    max_epochs = int(onn_cfg.get("max_epochs", 10))
    eval_every = int(onn_cfg.get("eval_every", 200))
    ckpt_auto_mode, ckpt_every = _parse_ckpt_every(onn_cfg.get("ckpt_every", 1000))
    viz_every = int(onn_cfg.get("viz_every", 200))
    phase_viz_every = max(0, int(onn_cfg.get("phase_viz_every", viz_every)))
    # Average and plot metrics over the same cadence as visualizations.
    metrics_avg_every = max(1, viz_every if viz_every > 0 else 1)
    metrics_update_every = max(1, int(onn_cfg.get("metrics_viz_every", metrics_avg_every)))

    optical_default_dir = Path(
        "/home/adminlwe/Documents/lwe-opu/experiment/model_training/pre_trained_model_save/Optical_neural_net"
    )
    run_dir = Path(onn_cfg.get("save_dir", str(optical_default_dir)))
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_layer_path = checkpoint_dir / f"_mul_mix_{n_layers}_Facial.pt"

    write_config_snapshot(cfg, run_dir)

    phase_system = PhaseSystem(
        n_layers=n_layers,
        phase_shape=phase_mask_shape,
        init_method=init_method,
        init_sigma=init_sigma,
        device=device,
        structural_nonlinearity=structural_nonlinearity,
        alpha_init=alpha_init,
        alpha_max=alpha_max,
        alpha_map_hw=alpha_map_hw,
    ).to(device)
    if phase_system.num_channels != channel_num:
        raise ValueError(
            f"Configured channel_num={channel_num} but PhaseSystem uses {phase_system.num_channels}; "
            "ensure phase_mask_shape[0] matches channel_num."
        )

    mixer_cfg = cfg.get("mixer", {})
    input_conv_cfg = cfg.get("input_conv", {})
    input_adapter = _build_input_adapter(data_method, channel_num, input_conv_cfg)
    if input_adapter is not None:
        input_adapter = input_adapter.to(device)
    mixers = _build_mixers(n_layers, channel_num, mixer_cfg).to(device)

    # No tile mode for first layer in this variant.
    modes = ["interp"] * n_layers
    x_norm_modes = ["percentile"] * n_layers
    bridge = OpticalBridge(
        optical_sys,
        surrogate,
        enc_canvas_hw,
        modes,
        x_norm_modes,
        in_silico_mode=in_silico_mode,
        channel_num=channel_num,
        learnable_gain=False,
        gain_init=8.0,
    )

    optim_module = nn.Module()
    optim_module.phase_system = phase_system
    optim_module.mixers = mixers
    optim_module.channel_readouts = channel_readouts
    if input_adapter is not None:
        optim_module.input_adapter = input_adapter
    if fusion_weights_raw is not None:
        optim_module.fusion_weights_raw = fusion_weights_raw
    if bridge.gain_raw is not None:
        optim_module.bridge_gain_raw = bridge.gain_raw

    optimizer = _build_optimizer(optim_module, opt_name, lr, weight_decay)
    _ensure_initial_lrs(optimizer)
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    fine_cfg = cfg.get("fine_tune_surrogate", {})
    requested_fine_tune = bool(fine_cfg.get("enable", False))
    fine_enable = requested_fine_tune and (not in_silico_mode)
    if requested_fine_tune and in_silico_mode:
        print("[info] In-silico mode disables surrogate fine-tuning.")

    reset_live_plot()
    reset_fine_tune_viz()
    viz_outputs = create_training_viz_layout(
        n_layers=n_layers,
        in_silico_mode=in_silico_mode,
        fine_tuning_enabled=fine_enable,
        n_optical_params=n_optical_params,
    )
    metrics_plot_state = init_metrics_plot_state(
        log_scale=False,
        output_widget=viz_outputs.get("metrics"),
        channel_num=channel_num,
        show_accuracy=False,
    )
    sample_viz_state = _init_keypoint_viz_state(output_widget=viz_outputs.get("samples"))
    phase_viz_state = init_phase_viz_state(output_widget=viz_outputs.get("phase"))

    history = {
        "train_loss": [],
        "eval_loss": [],
        "iters": [],
        "train_mse": [],
        "eval_mse": [],
        "eval_iters": [],
    }

    extra_ckpt_tensors: Dict[str, torch.nn.Parameter] = {}
    if bridge.gain_raw is not None:
        extra_ckpt_tensors["bridge_gain_raw"] = bridge.gain_raw
    checkpoint_extra_state = {k: v for k, v in extra_ckpt_tensors.items() if v is not None}

    resume_epoch, it_counter, best_eval = _load_checkpoint(
        checkpoint_layer_path,
        device,
        phase_system,
        history,
        mixers,
        channel_readouts=channel_readouts,
        input_adapter=input_adapter,
        extra_tensors=extra_ckpt_tensors,
        fusion_weights_raw=fusion_weights_raw,
    )
    if not checkpoint_layer_path.exists():
        _save_checkpoint(
            checkpoint_layer_path,
            resume_epoch,
            it_counter,
            phase_system,
            mixers,
            channel_readouts,
            input_adapter,
            history,
            cfg,
            extra_state=checkpoint_extra_state,
            fusion_weights_raw=fusion_weights_raw,
        )

    buffer_max = int(fine_cfg.get("buffer_max", 2048))
    buffer_freq = int(fine_cfg.get("freq_iters", 100))
    fine_steps = int(fine_cfg.get("steps", 50))
    fine_lr = float(fine_cfg.get("lr", 1e-4))
    fine_weight_decay = float(fine_cfg.get("weight_decay", 0.0))
    fine_viz_every = int(fine_cfg.get("viz_every", 0))
    fine_sample_every = int(fine_cfg.get("sample_every", 0))
    fine_show_sample = bool(fine_cfg.get("show_sample", True))
    fine_state = init_fine_tune_state(n_layers, buffer_max) if fine_enable else None
    raw_train_dataset = getattr(train_loader.dataset, "dataset", None)
    val_viz_iter = iter(val_loader)
    raw_val_dataset = getattr(val_loader.dataset, "dataset", None)
    val_viz_raw_iter = iter(raw_val_dataset) if raw_val_dataset is not None else None

    train_loss_buffer = _MetricBuffer(metrics_avg_every)
    eval_loss_buffer = _MetricBuffer(metrics_update_every)

    def _emit_train_metrics(force: bool = False) -> None:
        it_loss, avg_loss = train_loss_buffer.pop_average(force=force)
        if avg_loss is None:
            return
        iter_for_plot = it_loss or it_counter
        update_metrics_plot(
            metrics_plot_state,
            iter_for_plot,
            metrics_avg_every,
            train_loss=avg_loss,
            force=force,
        )

    def _emit_eval_metrics(force: bool = False) -> None:
        it_loss, avg_loss = eval_loss_buffer.pop_average(force=force)
        if avg_loss is None:
            return
        iter_for_plot = it_loss or it_counter
        update_metrics_plot(
            metrics_plot_state,
            iter_for_plot,
            metrics_avg_every,
            eval_loss=avg_loss,
            force=True,
        )

    try:
        steps_per_epoch = len(train_loader)
    except TypeError:
        steps_per_epoch = None
    total_steps_full = steps_per_epoch * max_epochs if steps_per_epoch is not None else None
    if lr_scheduler_name == "cosine":
        if total_steps_full is None:
            print("[warn] Skipping LR scheduler because total steps could not be inferred.")
        else:
            scheduler = _build_cosine_warmup_scheduler(
                optimizer,
                base_lr=lr,
                min_lr=lr_min,
                warmup_steps=warmup_iters,
                total_steps=total_steps_full,
                last_epoch=it_counter,
            )
    elif lr_scheduler_name not in {"none", "off", "null"}:
        print(f"[warn] Unknown lr_scheduler '{lr_scheduler_name}', skipping scheduler.")
    epochs_remaining = max(max_epochs - resume_epoch, 0)
    total_batches_remaining = (
        steps_per_epoch * epochs_remaining if (steps_per_epoch is not None and epochs_remaining > 0) else None
    )
    ensure_dark_tqdm_theme()
    progress_output_widget = viz_outputs.get("progress")
    if epochs_remaining > 0:
        if USE_NOTEBOOK_PROGRESS:
            progress_bar = NotebookProgressBar(
                total=total_batches_remaining,
                desc=f"Epoch {resume_epoch + 1}/{max_epochs}",
                output=progress_output_widget,
            )
        else:
            progress_bar = tqdm(
                total=total_batches_remaining,
                desc=f"Epoch {resume_epoch + 1}/{max_epochs}",
                dynamic_ncols=True,
                unit="iter",
                leave=True,
                colour="black",
            )
    else:
        progress_bar = None

    last_train_loss: Optional[float] = None
    last_train_mse: Optional[float] = None
    latest_eval_loss: Optional[float] = None
    latest_eval_mse: Optional[float] = None
    last_batch_idx: Optional[int] = None
    optimizer.zero_grad(set_to_none=True)

    def _refresh_progress(batch_idx: Optional[int] = None, *, refresh: bool = False) -> None:
        nonlocal last_batch_idx
        if progress_bar is None:
            return
        idx = batch_idx if batch_idx is not None else last_batch_idx
        if idx is not None:
            last_batch_idx = idx
        postfix = {"iter": it_counter}
        if idx is not None:
            if steps_per_epoch is not None and steps_per_epoch > 0:
                postfix["batch"] = f"{idx}/{steps_per_epoch}"
            else:
                postfix["batch"] = idx
        if last_train_loss is not None:
            postfix["loss"] = f"{last_train_loss:.4f}"
        if last_train_mse is not None:
            postfix["mse"] = f"{last_train_mse:.4f}"
        if scheduler is not None:
            lr_current = scheduler.get_last_lr()
            if lr_current:
                postfix["lr"] = f"{lr_current[0]:.2e}"
        if latest_eval_loss is not None:
            postfix["eval_loss"] = f"{latest_eval_loss:.4f}"
        if latest_eval_mse is not None:
            postfix["eval_mse"] = f"{latest_eval_mse:.4f}"
        progress_bar.set_postfix(postfix, refresh=refresh)
        if refresh and hasattr(progress_bar, "refresh"):
            progress_bar.refresh()

    try:
        for epoch in range(resume_epoch, max_epochs):
            if progress_bar is not None:
                progress_bar.set_description(f"Epoch {epoch + 1}/{max_epochs}")
            for batch_idx, (x_batch, labels) in enumerate(train_loader, start=1):
                it_counter += 1
                phase_system.train()
                surrogate.eval()
                x_batch = x_batch.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.float32)
                if labels.dim() == 1:
                    labels = labels.unsqueeze(0)
                if labels.size(-1) != readout_cfg["linear_out_features"]:
                    raise ValueError(
                        f"Expected labels shape (N,{readout_cfg['linear_out_features']}), got {tuple(labels.shape)}"
                    )
                x_channels = _prepare_input_channels(
                    x_batch,
                    channel_num=channel_num,
                    data_method=data_method,
                    input_adapter=input_adapter,
                    upsample_hw=input_hw,
                )
                x_phys = x_channels.squeeze(0).unsqueeze(1)
                if x_phys.shape != (channel_num, 1, x_channels.size(2), x_channels.size(3)):
                    raise ValueError(f"x_phys must be ({channel_num},1,H,W), got {tuple(x_phys.shape)}")

                with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                    (
                        y_final,
                        layer_inputs,
                        layer_phases,
                        layer_outputs,
                        layer_inputs_pre,
                    ) = _forward_with_mix(
                        x_phys,
                        phase_system,
                        bridge,
                        mixers,
                        mixer_cfg,
                        track_grads=True,
                    )
                    pred_fused, _, penalty_loss, _ = _readout_regression(
                        y_final,
                        mask_hw=(readout_cfg["mask_h"], readout_cfg["mask_w"]),
                        pool_hw=(readout_cfg["pool_h"], readout_cfg["pool_w"]),
                        channel_readouts=channel_readouts,
                        linear_in_features=readout_cfg["linear_in_features"],
                        fuse_mode=fuse_mode,
                        fusion_weights_raw=fusion_weights_raw,
                    )
                    mse_loss = F.mse_loss(pred_fused, labels)
                    rmse_loss = torch.sqrt(mse_loss + 1.0e-12)
                    total_loss = rmse_loss + outside_mask_penalty_weight * penalty_loss

                loss = total_loss / float(accum_steps)
                scaler.scale(loss).backward()

                is_accum_step = (it_counter % accum_steps == 0)
                is_last_in_epoch = steps_per_epoch is not None and batch_idx == steps_per_epoch
                if is_accum_step or is_last_in_epoch:
                    if grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(optim_module.parameters(), grad_clip_norm)
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = scaler.get_scale()
                    if scheduler is not None:
                        if scale_after >= scale_before:
                            scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                loss_val = float(total_loss.detach().cpu())
                mse_val = float(mse_loss.detach().cpu())

                history["train_loss"].append(loss_val)
                history["train_mse"].append(mse_val)
                history["iters"].append(it_counter)
                train_loss_buffer.add(loss_val, it_counter)
                if train_loss_buffer.ready():
                    _emit_train_metrics()
                last_train_loss = loss_val
                last_train_mse = mse_val
                _refresh_progress(batch_idx)
                if progress_bar is not None:
                    progress_bar.update(1)

                if fine_state is not None:
                    accumulate_fine_tune_samples(
                        fine_state,
                        layer_inputs=layer_inputs,
                        layer_phases=layer_phases,
                        layer_outputs=layer_outputs,
                        layer_raw_inputs=layer_inputs_pre,
                    )
                    maybe_fine_tune_surrogate(
                        fine_state,
                        iteration=it_counter,
                        buffer_freq=buffer_freq,
                        surrogate=surrogate,
                        device=device,
                        enc_canvas_hw=enc_canvas_hw,
                        fine_steps=fine_steps,
                        fine_lr=fine_lr,
                        fine_weight_decay=fine_weight_decay,
                        use_amp=bool(use_amp and device.type == "cuda"),
                        viz_every=fine_viz_every,
                        sample_every=fine_sample_every,
                        show_sample=fine_show_sample,
                        surrogate_ckpt_path=surrogate_ckpt_path,
                    )

                if it_counter % eval_every == 0:
                    eval_loss, eval_mse = _evaluate(
                        val_loader,
                        device,
                        phase_system,
                        bridge,
                        mixers,
                        channel_readouts,
                        input_adapter,
                        mixer_cfg,
                        readout_cfg,
                        channel_num,
                        data_method,
                        input_hw,
                        fusion_weights_raw=fusion_weights_raw.detach() if fusion_weights_raw is not None else None,
                        fuse_mode=fuse_mode,
                        outside_mask_penalty_weight=outside_mask_penalty_weight,
                    )
                    history["eval_loss"].append(eval_loss)
                    history["eval_mse"].append(eval_mse)
                    history["eval_iters"].append(it_counter)
                    eval_loss_buffer.add(eval_loss, it_counter)
                    _emit_eval_metrics(force=True)
                    latest_eval_loss = eval_loss
                    latest_eval_mse = eval_mse
                    if progress_bar is not None:
                        _refresh_progress(batch_idx, refresh=True)
                    if eval_loss < best_eval:
                        best_eval = eval_loss
                        if ckpt_auto_mode:
                            _save_checkpoint(
                                checkpoint_layer_path,
                                epoch,
                                it_counter,
                                phase_system,
                                mixers,
                                channel_readouts,
                                input_adapter,
                                history,
                                cfg,
                                extra_state=checkpoint_extra_state,
                                fusion_weights_raw=fusion_weights_raw,
                            )

                if (not ckpt_auto_mode) and ckpt_every and ckpt_every > 0 and it_counter % ckpt_every == 0:
                    _save_checkpoint(
                        checkpoint_layer_path,
                        epoch,
                        it_counter,
                        phase_system,
                        mixers,
                        channel_readouts,
                        input_adapter,
                        history,
                        cfg,
                        extra_state=checkpoint_extra_state,
                        fusion_weights_raw=fusion_weights_raw,
                    )

                if viz_every > 0 and it_counter % viz_every == 0:
                    phase_system.eval()
                    try:
                        face_vis_list: List[torch.Tensor] = []
                        y_eval_list: List[torch.Tensor] = []
                        pred_eval_list: List[torch.Tensor] = []
                        channel6_vis_list: List[torch.Tensor] = []
                        sample_sources: List[str] = []
                        vis_channel_idx = min(max(channel_num - 1, 0), 5)
                        with torch.no_grad():
                            def _append_viz_example(
                                x_viz: torch.Tensor,
                                y_viz: torch.Tensor,
                                source_label: str,
                                x_raw_viz: Optional[torch.Tensor] = None,
                            ) -> None:
                                if x_viz.dim() == 3:
                                    x_viz = x_viz.unsqueeze(0)
                                if x_viz.size(0) > 1:
                                    x_viz = x_viz[:1]
                                if y_viz.dim() == 1:
                                    y_viz = y_viz.unsqueeze(0)
                                if y_viz.size(0) > 1:
                                    y_viz = y_viz[:1]

                                x_viz = x_viz.to(device=device, dtype=torch.float32)
                                y_viz = y_viz.to(device=device, dtype=torch.float32)
                                x_eval_channels = _prepare_input_channels(
                                    x_viz,
                                    channel_num=channel_num,
                                    data_method=data_method,
                                    input_adapter=input_adapter,
                                    upsample_hw=input_hw,
                                )
                                x_eval_phys = x_eval_channels.squeeze(0).unsqueeze(1)
                                y_eval_final, _, _, _, _ = _forward_with_mix(
                                    x_eval_phys,
                                    phase_system,
                                    bridge,
                                    mixers,
                                    mixer_cfg,
                                    track_grads=False,
                                )
                                pred_eval, _, _, _ = _readout_regression(
                                    y_eval_final,
                                    mask_hw=(readout_cfg["mask_h"], readout_cfg["mask_w"]),
                                    pool_hw=(readout_cfg["pool_h"], readout_cfg["pool_w"]),
                                    channel_readouts=channel_readouts,
                                    linear_in_features=readout_cfg["linear_in_features"],
                                    fuse_mode=fuse_mode,
                                    fusion_weights_raw=fusion_weights_raw,
                                )
                                if isinstance(x_raw_viz, torch.Tensor):
                                    if x_raw_viz.dim() == 2:
                                        x_raw_viz = x_raw_viz.unsqueeze(0)
                                    elif x_raw_viz.dim() == 3 and x_raw_viz.size(0) != 1:
                                        x_raw_viz = x_raw_viz.mean(dim=0, keepdim=True)
                                    face_vis = x_raw_viz.unsqueeze(0).to(
                                        device=device,
                                        dtype=torch.float32,
                                    ).clamp(0.0, 1.0).detach()
                                else:
                                    face_vis = (
                                        x_viz.mean(dim=1, keepdim=True)
                                        if x_viz.size(1) > 1
                                        else x_viz[:, :1]
                                    ).detach().clamp(0.0, 1.0)
                                face_vis_list.append(face_vis)
                                y_eval_list.append(y_viz[:1].detach())
                                pred_eval_list.append(pred_eval[:1].detach())
                                channel6_vis_list.append(y_eval_final[vis_channel_idx:vis_channel_idx + 1, 0].detach())
                                sample_sources.append(source_label)

                            for _ in range(2):
                                x_train_raw_viz = None
                                train_viz_ds = train_loader.dataset
                                if len(train_viz_ds) <= 0:
                                    continue
                                sample_idx = random.randrange(len(train_viz_ds))
                                x_train_viz, y_train_viz = train_viz_ds[sample_idx]
                                if raw_train_dataset is not None:
                                    try:
                                        x_train_raw_viz, _ = raw_train_dataset[sample_idx]
                                    except Exception:
                                        x_train_raw_viz = None
                                if not isinstance(y_train_viz, torch.Tensor):
                                    y_train_viz = torch.as_tensor(y_train_viz)
                                _append_viz_example(
                                    x_train_viz,
                                    y_train_viz,
                                    source_label="Train",
                                    x_raw_viz=x_train_raw_viz,
                                )

                            for _ in range(2):
                                try:
                                    x_eval_viz, y_eval_viz = next(val_viz_iter)
                                except StopIteration:
                                    val_viz_iter = iter(val_loader)
                                    x_eval_viz, y_eval_viz = next(val_viz_iter)
                                x_raw_viz = None
                                if val_viz_raw_iter is not None:
                                    try:
                                        x_raw_viz, _ = next(val_viz_raw_iter)
                                    except StopIteration:
                                        val_viz_raw_iter = iter(raw_val_dataset)
                                        x_raw_viz, _ = next(val_viz_raw_iter)
                                _append_viz_example(
                                    x_eval_viz,
                                    y_eval_viz,
                                    source_label="Eval",
                                    x_raw_viz=x_raw_viz,
                                )

                        _update_keypoint_viz(
                            sample_viz_state,
                            torch.cat(face_vis_list, dim=0),
                            torch.cat(y_eval_list, dim=0),
                            torch.cat(pred_eval_list, dim=0),
                            torch.cat(channel6_vis_list, dim=0),
                            channel_label=f"Channel {vis_channel_idx + 1} output",
                            sample_sources=sample_sources,
                            mask_h=readout_cfg["mask_h"],
                            mask_w=readout_cfg["mask_w"],
                            pool_h=readout_cfg["pool_h"],
                            pool_w=readout_cfg["pool_w"],
                        )
                    finally:
                        phase_system.train()

                if phase_viz_every > 0 and it_counter % phase_viz_every == 0:
                    phase_viz_inputs: List[torch.Tensor] = []
                    for layer_idx, phase_param in enumerate(phase_system.phase_parameters()):
                        phase_to_plot = phase_to_unit(phase_param)
                        cached_phase = bridge.phase_cache[layer_idx] if layer_idx < len(bridge.phase_cache) else None
                        if isinstance(cached_phase, torch.Tensor) and cached_phase.numel() > 0:
                            try:
                                B, _, Hc, Wc = cached_phase.shape
                                C = phase_system.num_channels
                                reps = B // C
                                reshaped = cached_phase.view(reps, C, 1, Hc, Wc)
                                phase_to_plot = reshaped[0, :, 0]
                            except Exception:
                                pass
                        phase_viz_inputs.append(phase_to_plot.detach())
                    update_phase_viz(
                        phase_viz_state,
                        phase_viz_inputs,
                        enc_canvas_hw,
                        it_counter,
                    )
            if progress_bar is not None:
                progress_bar.refresh()
    finally:
        _emit_train_metrics(force=True)
        _emit_eval_metrics(force=True)
        if progress_bar is not None:
            progress_bar.close()

    if not ckpt_auto_mode:
        _save_checkpoint(
            checkpoint_layer_path,
            max_epochs,
            it_counter,
            phase_system,
            mixers,
            channel_readouts,
            input_adapter,
            history,
            cfg,
            extra_state=checkpoint_extra_state,
            fusion_weights_raw=fusion_weights_raw,
        )
    print(f"Training completed. Results saved to {run_dir}")


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an optical neural network (ONN) with channel mixing.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: config.yaml in this directory)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config entries, e.g. --override onn.lr=0.005 --override onn.n_layers=4",
    )
    return parser.parse_args()


def main():
    args = parse_cli()
    overrides = {}
    for item in args.override:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        if value.isdigit():
            value_cast = int(value)
        else:
            try:
                value_cast = float(value)
            except ValueError:
                if value.lower() in ("true", "false"):
                    value_cast = value.lower() == "true"
                else:
                    value_cast = value
        overrides[key] = value_cast
    train(config_path=args.config, overrides=overrides)


if __name__ == "__main__":
    main()
