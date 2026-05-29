"""
End-to-end VLM caption training for ONN encoder + Transformer decoder.

Pipeline:
image -> optional input adapter -> ONN layers with channel mixers -> encoder readout
-> decoder cross-attention memory -> caption logits / autoregressive generation.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import string
import textwrap
import unicodedata
from collections import Counter
from pathlib import Path
from pickle import UnpicklingError
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional visualization dependency
    plt = None

try:
    from .models import Surrogate_OpticalNet_Unet
    from .dataloader_patch import build_vlm_caption_loaders_from_cfg
    from .utils import encoding_x_phase_physical, percentile_scale_spatial
    from .surrogate_model_training import make_optical_sys, reset_live_plot
    from .config_loader import load_training_config, resolve_onn_seed, write_config_snapshot
    from .phase_system import PhaseSystem, configure_phase_unit, phase_to_unit
    from .optical_bridge import OpticalBridge
    from .training_viz import (
        USE_NOTEBOOK_PROGRESS,
        NotebookProgressBar,
        ensure_dark_tqdm_theme,
        create_training_viz_layout,
        init_phase_viz_state,
    )
    from .fine_tuning import (
        accumulate_fine_tune_samples,
        init_fine_tune_state,
        maybe_fine_tune_surrogate,
        reset_fine_tune_viz,
    )
except ImportError:
    from models import Surrogate_OpticalNet_Unet
    from dataloader_patch import build_vlm_caption_loaders_from_cfg
    from utils import encoding_x_phase_physical, percentile_scale_spatial
    from surrogate_model_training import make_optical_sys, reset_live_plot
    from config_loader import load_training_config, resolve_onn_seed, write_config_snapshot
    from phase_system import PhaseSystem, configure_phase_unit, phase_to_unit
    from optical_bridge import OpticalBridge
    from training_viz import (
        USE_NOTEBOOK_PROGRESS,
        NotebookProgressBar,
        ensure_dark_tqdm_theme,
        create_training_viz_layout,
        init_phase_viz_state,
    )
    from fine_tuning import (
        accumulate_fine_tune_samples,
        init_fine_tune_state,
        maybe_fine_tune_surrogate,
        reset_fine_tune_viz,
    )


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VLM_CONFIG_PATH = PROJECT_ROOT / "config_mix_VLM.yaml"

try:
    from IPython.display import display
except Exception:  # pragma: no cover - optional notebook dependency
    display = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _prepare_vlm_data_loaders(
    data_cfg: Dict[str, object],
    tokenizer_cfg: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    loader_cfg = dict(data_cfg)
    if tokenizer_cfg:
        for key, value in tokenizer_cfg.items():
            loader_cfg.setdefault(key, value)
    if "simple_tokenizer_lowercase" not in loader_cfg and "lowercase" in loader_cfg:
        loader_cfg["simple_tokenizer_lowercase"] = bool(loader_cfg["lowercase"])
    # In VLM patchify mode, patch projection is handled inside this training file.
    # Keep loader output as RGB images (upsampler path) rather than pre-patchified tensors.
    method = str(data_cfg.get("method", "upsampler")).strip().lower()
    if method == "patchify":
        loader_cfg["method"] = "upsampler"
    return build_vlm_caption_loaders_from_cfg(loader_cfg)


def _load_surrogate(sur_cfg: Dict[str, object], device: torch.device, channel_num: int):
    enc_canvas_hw = tuple(sur_cfg.get("enc_canvas_hw", [168, 168]))
    surrogate_dir_default = PROJECT_ROOT / "pre_trained_model_save" / "Surrogate_OpticalNet_Unet" / "exp2"
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
    encoder_readout: nn.Module | None = None,
    caption_decoder: nn.Module | None = None,
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
    if encoder_readout is not None:
        encoder_readout_state = ckpt_loaded.get("encoder_readout_state")
        if isinstance(encoder_readout_state, dict):
            try:
                encoder_readout.load_state_dict(encoder_readout_state, strict=False)
            except RuntimeError as exc:
                print(f"[warn] Failed to load encoder_readout_state: {exc}")
    if caption_decoder is not None:
        caption_decoder_state = ckpt_loaded.get("caption_decoder_state")
        if isinstance(caption_decoder_state, dict):
            try:
                caption_decoder.load_state_dict(caption_decoder_state, strict=False)
            except RuntimeError as exc:
                print(f"[warn] Failed to load caption_decoder_state: {exc}")

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
    eval_history = history.get("eval_caption_loss", history.get("eval_caption_loss_avg", history.get("eval_loss", [])))
    if eval_history:
        finite_vals = [
            float(v)
            for v in eval_history
            if isinstance(v, (int, float)) and math.isfinite(float(v))
        ]
        if finite_vals:
            best_eval = min(finite_vals)
    return resume_epoch, it_counter, best_eval


def _save_checkpoint(
    path: Path,
    epoch_idx: int,
    iter_idx: int,
    phase_system: PhaseSystem,
    mixers: nn.ModuleList,
    channel_readouts: nn.ModuleList | None,
    input_adapter: nn.Module | None,
    encoder_readout: nn.Module | None,
    caption_decoder: nn.Module | None,
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
    if encoder_readout is not None:
        ckpt_obj["encoder_readout_state"] = encoder_readout.state_dict()
    if caption_decoder is not None:
        ckpt_obj["caption_decoder_state"] = caption_decoder.state_dict()
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
    patchify_target_hw: Optional[Tuple[int, int]] = None,
    patchify_overlap_percent: float = 0.0,
) -> nn.Module | None:
    class PatchifyProjector(nn.Module):
        def __init__(
            self,
            channel_num_local: int,
            target_hw_local: Tuple[int, int],
            overlap_percent_local: float,
        ) -> None:
            super().__init__()
            grid = int(round(math.sqrt(channel_num_local)))
            if grid * grid != channel_num_local:
                raise ValueError("Patchify projector requires channel_num to be a perfect square.")
            self.channel_num = int(channel_num_local)
            self.grid = int(grid)
            self.target_hw = (int(target_hw_local[0]), int(target_hw_local[1]))
            self.overlap_percent = float(overlap_percent_local)
            if self.overlap_percent < 0.0:
                raise ValueError("patchify overlap_percent must be non-negative.")
            if self.overlap_percent >= 100.0:
                raise ValueError("patchify overlap_percent must be < 100.")
            # Independent projector per patch/channel (no shared weights).
            self.patch_projectors = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(3, 16, kernel_size=3, padding=1),
                        nn.GELU(),
                        # Upsample avoids transposed-conv checkerboard artifacts.
                        nn.Upsample(size=self.target_hw, mode="bilinear", align_corners=False),
                        nn.Conv2d(16, 1, kernel_size=3, padding=1),
                    )
                    for _ in range(self.channel_num)
                ]
            )

        def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
            if x_rgb.dim() != 4:
                raise ValueError(f"Patchify projector expects (B,3,H,W), got {tuple(x_rgb.shape)}")
            if x_rgb.size(1) != 3:
                raise ValueError(f"Patchify projector expects 3-channel RGB, got C={int(x_rgb.size(1))}.")
            bsz, _, height, width = x_rgb.shape
            if height % self.grid != 0 or width % self.grid != 0:
                raise ValueError(
                    f"Patchify projector expects H/W divisible by {self.grid}, got {(height, width)}."
                )
            base_stride_h = height // self.grid
            base_stride_w = width // self.grid
            overlap_ratio = self.overlap_percent / 100.0
            patch_h = int(round(base_stride_h / (1.0 - overlap_ratio)))
            patch_w = int(round(base_stride_w / (1.0 - overlap_ratio)))
            patch_h = min(max(1, patch_h), height)
            patch_w = min(max(1, patch_w), width)

            center_shift_h = (patch_h - base_stride_h) // 2
            center_shift_w = (patch_w - base_stride_w) // 2
            y_positions = [i * base_stride_h - center_shift_h for i in range(self.grid)]
            x_positions = [i * base_stride_w - center_shift_w for i in range(self.grid)]

            patch_list: List[torch.Tensor] = []
            for y_start in y_positions:
                y_start = int(min(max(y_start, 0), max(height - 1, 0)))
                y_end = int(min(y_start + patch_h, height))
                for x_start in x_positions:
                    x_start = int(min(max(x_start, 0), max(width - 1, 0)))
                    x_end = int(min(x_start + patch_w, width))
                    patch = x_rgb[:, :, y_start:y_end, x_start:x_end]  # (B,3,h,w)
                    if patch.shape[-2:] != (patch_h, patch_w):
                        patch = F.interpolate(
                            patch,
                            size=(patch_h, patch_w),
                            mode="bilinear",
                            align_corners=False,
                        )
                    patch_list.append(patch)
            if len(patch_list) != self.channel_num:
                raise ValueError(f"Patchify projector expected {self.channel_num} patches, got {len(patch_list)}")
            # (B,16,3,patch_h,patch_w)
            x_patch = torch.stack(patch_list, dim=1)
            # Apply channel-specific projection per patch channel.
            projected_channels: List[torch.Tensor] = []
            for idx, projector in enumerate(self.patch_projectors):
                patch_i = x_patch[:, idx]  # (B,3,patch_h,patch_w)
                out_i = projector(patch_i)  # (B,1,h,w)
                if out_i.dim() != 4 or out_i.size(1) != 1:
                    raise ValueError(f"Patchify projector[{idx}] must produce (B,1,H,W), got {tuple(out_i.shape)}")
                if out_i.shape[-2:] != self.target_hw:
                    out_i = F.interpolate(out_i, size=self.target_hw, mode="bilinear", align_corners=False)
                projected_channels.append(out_i)
            # Stack to ONN channel map: (B,16,112,112)
            return torch.cat(projected_channels, dim=1)

    if data_method == "patchify":
        cfg_target = input_conv_cfg.get("patchify_target_hw", None)
        if cfg_target is not None:
            if not (isinstance(cfg_target, (list, tuple)) and len(cfg_target) >= 2):
                raise ValueError("input_conv.patchify_target_hw must be [H, W] when provided.")
            target_hw = (int(cfg_target[0]), int(cfg_target[1]))
        elif patchify_target_hw is not None:
            target_hw = (int(patchify_target_hw[0]), int(patchify_target_hw[1]))
        else:
            target_hw = (112, 112)
        return PatchifyProjector(
            channel_num_local=channel_num,
            target_hw_local=target_hw,
            overlap_percent_local=float(patchify_overlap_percent),
        )
    if data_method != "upsampler":
        return None
    k = int(input_conv_cfg.get("kernel_size", 3))
    if k <= 0:
        raise ValueError("input_conv.kernel_size must be positive")
    in_channels = int(input_conv_cfg.get("in_channels", 3))
    if in_channels <= 0:
        raise ValueError("input_conv.in_channels must be positive")
    bias = bool(input_conv_cfg.get("bias", True))
    padding = int(input_conv_cfg.get("padding", k // 2))
    if padding < 0:
        raise ValueError("input_conv.padding must be non-negative")
    return nn.Conv2d(in_channels, channel_num, kernel_size=k, padding=padding, bias=bias)


def _prepare_input_channels(
    x_batch: torch.Tensor,
    *,
    channel_num: int,
    data_method: str,
    input_adapter: nn.Module | None,
    upsample_hw: Tuple[int, int] | None = None,
    input_preproc_cfg: Optional[Dict[str, object]] = None,
) -> torch.Tensor:
    def _coerce_channel_values(raw_value: object, channels: int, key: str) -> List[float]:
        if isinstance(raw_value, (int, float)):
            return [float(raw_value)] * channels
        if isinstance(raw_value, (list, tuple)):
            vals = [float(v) for v in raw_value]
            if len(vals) == 1 and channels > 1:
                return vals * channels
            if len(vals) != channels:
                raise ValueError(f"input_preprocess.{key} must have length {channels}, got {len(vals)}.")
            return vals
        raise ValueError(f"input_preprocess.{key} must be a number or a list/tuple.")

    def _normalize_before_patchify(x_rgb: torch.Tensor) -> torch.Tensor:
        if not isinstance(input_preproc_cfg, dict):
            return x_rgb
        if not bool(input_preproc_cfg.get("normalize_before_patchify", False)):
            return x_rgb
        ch = int(x_rgb.size(1))
        mean_vals = _coerce_channel_values(input_preproc_cfg.get("mean", [0.5, 0.5, 0.5]), ch, "mean")
        std_vals = _coerce_channel_values(input_preproc_cfg.get("std", [0.5, 0.5, 0.5]), ch, "std")
        std_eps = max(1e-12, float(input_preproc_cfg.get("std_eps", 1e-6)))
        mean_t = torch.tensor(mean_vals, device=x_rgb.device, dtype=x_rgb.dtype).view(1, ch, 1, 1)
        std_t = torch.tensor(std_vals, device=x_rgb.device, dtype=x_rgb.dtype).view(1, ch, 1, 1)
        std_t = std_t.clamp_min(std_eps)
        return (x_rgb - mean_t) / std_t

    if x_batch.dim() != 4:
        raise ValueError(f"Expected x_batch (1,C,H,W), got {tuple(x_batch.shape)}")
    if x_batch.size(0) != 1:
        raise ValueError(f"x_batch must have batch size 1, got {tuple(x_batch.shape)}")
    if data_method == "patchify":
        # Preferred patchify path: start from RGB and project per patch into optical channels.
        if x_batch.size(1) == 3:
            if input_adapter is None:
                raise ValueError("patchify mode requires patch projector input_adapter.")
            x_pre = _normalize_before_patchify(x_batch)
            x_channels = input_adapter(x_pre)
            if x_channels.size(1) != channel_num:
                raise ValueError(
                    f"Patchify projection output must be (1,{channel_num},H,W), got {tuple(x_channels.shape)}"
                )
            return x_channels
        # Backward-compatible fallback for pre-patchified tensors.
        if x_batch.size(1) == channel_num:
            return x_batch
        raise ValueError(
            f"Patchify mode expects x_batch channels in {{3,{channel_num}}}, got {tuple(x_batch.shape)}"
        )
    if data_method == "upsampler":
        if input_adapter is None:
            raise ValueError("input_adapter is required when data.method='upsampler'")
        expected_in_channels = int(getattr(input_adapter, "in_channels", 1))
        if x_batch.size(1) != expected_in_channels:
            raise ValueError(
                f"Upsampler mode expects x_batch (1,{expected_in_channels},H,W), got {tuple(x_batch.shape)}"
            )
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


def _apply_input_positional_encoding(
    x_phys: torch.Tensor,
    pe_cfg: Optional[Dict[str, object]],
) -> torch.Tensor:
    """
    Inject spatial PE immediately after x_channels -> x_phys reshape.
    x_phys is expected as (C,1,H,W) where C is the optical-channel batch.
    """
    if not isinstance(pe_cfg, dict) or not bool(pe_cfg.get("enabled", False)):
        return x_phys
    if x_phys.dim() != 4 or x_phys.size(1) != 1:
        raise ValueError(f"Input PE expects x_phys shaped (C,1,H,W), got {tuple(x_phys.shape)}")

    mode = str(pe_cfg.get("mode", "sin2d")).strip().lower()
    if mode not in {"sin2d"}:
        raise ValueError(f"Unsupported input_positional_encoding.mode '{mode}'. Expected 'sin2d'.")

    scale = float(pe_cfg.get("scale", 0.05))
    if not math.isfinite(scale) or scale == 0.0:
        return x_phys
    num_bands = max(1, int(pe_cfg.get("num_bands", 4)))
    injection = str(pe_cfg.get("injection", "add")).strip().lower()
    zero_mean = bool(pe_cfg.get("zero_mean", True))
    clamp_01 = bool(pe_cfg.get("clamp_01", False))

    _, _, h, w = x_phys.shape
    dtype = x_phys.dtype
    device = x_phys.device
    yy = torch.linspace(0.0, 1.0, steps=h, device=device, dtype=dtype).view(1, 1, h, 1)
    xx = torch.linspace(0.0, 1.0, steps=w, device=device, dtype=dtype).view(1, 1, 1, w)

    pe = torch.zeros((1, 1, h, w), device=device, dtype=dtype)
    two_pi = 2.0 * math.pi
    for band_idx in range(num_bands):
        freq = float(2 ** band_idx)
        pe = pe + torch.sin(two_pi * freq * xx) + torch.cos(two_pi * freq * xx)
        pe = pe + torch.sin(two_pi * freq * yy) + torch.cos(two_pi * freq * yy)
    pe = pe / pe.abs().amax().clamp_min(1e-6)
    if zero_mean:
        pe = pe - pe.mean()

    if injection == "add":
        out = x_phys + (scale * pe)
    elif injection in {"mul", "mul_add", "film"}:
        out = x_phys * (1.0 + (scale * pe))
    else:
        raise ValueError(
            f"Unsupported input_positional_encoding.injection '{injection}'. "
            "Expected 'add' or 'mul'."
        )
    if clamp_01:
        out = out.clamp(0.0, 1.0)
    return out


def _build_encoder_readout_input(
    y_map: torch.Tensor,
    x_channels: torch.Tensor,
    encoder_input_skip_cfg: Optional[Dict[str, object]],
) -> torch.Tensor:
    """
    Optionally fuse pre-optical input channels into encoder readout input.
    Expected:
      y_map      -> (B, C_onn, H, W)
      x_channels -> (B, C_in, H0, W0)
    """
    if not isinstance(encoder_input_skip_cfg, dict) or not bool(encoder_input_skip_cfg.get("enabled", False)):
        return y_map
    mode = str(encoder_input_skip_cfg.get("mode", "concat")).strip().lower()
    if mode != "concat":
        raise ValueError(
            f"Unsupported encoder_readout.input_skip.mode '{mode}'. Expected 'concat'."
        )
    skip_map = x_channels
    if skip_map.shape[-2:] != y_map.shape[-2:]:
        skip_map = F.interpolate(skip_map, size=y_map.shape[-2:], mode="bilinear", align_corners=False)
    if skip_map.size(0) != y_map.size(0):
        raise ValueError(
            "encoder_readout.input_skip requires matching batch size between y_map and x_channels, got "
            f"{int(y_map.size(0))} vs {int(skip_map.size(0))}."
        )
    return torch.cat([y_map, skip_map], dim=1)


class EncoderReadout(nn.Module):
    """Convert final ONN feature map (B,C,H,W) into decoder memory tokens."""

    def __init__(
        self,
        *,
        channel_num: int,
        emb_size: int,
        readout_type: str = "avg_pool_flatten_proj",
        pool_h: int = 5,
        pool_w: int = 5,
        flatten_mode: str = "channel_first",
        conv_in_channels: Optional[int] = None,
        conv_out_channels: int = 64,
        conv_kernel_size: int = 7,
        conv_stride: int = 7,
        conv_padding: int = 0,
        conv_output_h: Optional[int] = None,
        conv_output_w: Optional[int] = None,
        use_channel_embedding: bool = True,
        use_token_positional_embedding: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.channel_num = int(channel_num)
        self.emb_size = int(emb_size)
        self.readout_type = str(readout_type).strip().lower()
        self.pool_h = int(pool_h)
        self.pool_w = int(pool_w)
        self.flatten_mode = str(flatten_mode).strip().lower()
        self.conv_in_channels = int(conv_in_channels) if conv_in_channels is not None else int(channel_num)
        self.conv_out_channels = int(conv_out_channels)
        self.conv_kernel_size = int(conv_kernel_size)
        self.conv_stride = int(conv_stride)
        self.conv_padding = int(conv_padding)
        self.conv_output_h = int(conv_output_h) if conv_output_h is not None else None
        self.conv_output_w = int(conv_output_w) if conv_output_w is not None else None
        self.use_token_positional_embedding = bool(use_token_positional_embedding)

        self.proj: Optional[nn.Linear] = None
        self.patch_conv: Optional[nn.Conv2d] = None
        self.channel_embedding: Optional[nn.Embedding] = None
        self.token_pos_embedding: Optional[nn.Embedding] = None

        if self.readout_type in {"avg_pool_flatten_proj", "avg_pool"}:
            if self.pool_h <= 0 or self.pool_w <= 0:
                raise ValueError("encoder_readout.pool_h/pool_w must be positive.")
            if self.flatten_mode not in {"channel_first"}:
                raise ValueError("encoder_readout.flatten_mode currently supports only 'channel_first'.")
            feat_dim = self.pool_h * self.pool_w
            self.proj = nn.Linear(feat_dim, self.emb_size)
            self.channel_embedding = (
                nn.Embedding(self.channel_num, self.emb_size) if bool(use_channel_embedding) else None
            )
            if self.use_token_positional_embedding:
                self.token_pos_embedding = nn.Embedding(self.channel_num, self.emb_size)
        elif self.readout_type == "conv_patch_tokens":
            if self.conv_in_channels <= 0 or self.conv_out_channels <= 0:
                raise ValueError("encoder_readout conv in_channels/out_channels must be positive.")
            if self.conv_kernel_size <= 0 or self.conv_stride <= 0 or self.conv_padding < 0:
                raise ValueError("encoder_readout conv kernel_size/stride must be positive and padding >= 0.")
            if (self.conv_output_h is None) != (self.conv_output_w is None):
                raise ValueError("encoder_readout conv_output_h and conv_output_w must be both set or both omitted.")
            if self.conv_output_h is None and self.conv_output_w is None:
                # Auto target from emb_size when emb_size is a perfect square (e.g., 256 -> 16x16).
                side = int(math.isqrt(self.emb_size))
                if side * side == self.emb_size:
                    self.conv_output_h = side
                    self.conv_output_w = side
            if self.conv_output_h is not None and self.conv_output_w is not None:
                if self.conv_output_h <= 0 or self.conv_output_w <= 0:
                    raise ValueError("encoder_readout conv_output_h/conv_output_w must be positive.")
                if (self.conv_output_h * self.conv_output_w) != self.emb_size:
                    raise ValueError(
                        "encoder_readout conv_output_h * conv_output_w must equal emb_size for conv_patch_tokens."
                    )
            self.patch_conv = nn.Conv2d(
                in_channels=self.conv_in_channels,
                out_channels=self.conv_out_channels,
                kernel_size=self.conv_kernel_size,
                stride=self.conv_stride,
                padding=self.conv_padding,
            )
            if self.use_token_positional_embedding:
                self.token_pos_embedding = nn.Embedding(self.conv_out_channels, self.emb_size)
        else:
            raise ValueError(
                f"Unsupported encoder_readout.type '{self.readout_type}'. "
                "Expected 'avg_pool_flatten_proj' or 'conv_patch_tokens'."
            )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"EncoderReadout expects (B,C,H,W), got {tuple(x.shape)}")
        if self.readout_type in {"avg_pool_flatten_proj", "avg_pool"}:
            if x.size(1) != self.channel_num:
                raise ValueError(
                    f"EncoderReadout expected C={self.channel_num}, got C={int(x.size(1))}."
                )
            if self.proj is None:
                raise RuntimeError("EncoderReadout avg_pool projection is not initialized.")
            # Adaptive average pooling keeps readout stable for varying optical H/W.
            pooled = F.adaptive_avg_pool2d(x, (self.pool_h, self.pool_w))  # (B, C, Ph, Pw)
            flat = pooled.flatten(start_dim=2)  # (B, C, Ph*Pw)
            memory = self.proj(flat)  # (B, C, E)
            if self.channel_embedding is not None:
                idx = torch.arange(self.channel_num, device=x.device).unsqueeze(0)
                memory = memory + self.channel_embedding(idx)
            if self.token_pos_embedding is not None:
                tok_n = int(memory.size(1))
                if tok_n > int(self.token_pos_embedding.num_embeddings):
                    raise ValueError(
                        f"Token count {tok_n} exceeds token positional embedding size "
                        f"{int(self.token_pos_embedding.num_embeddings)}."
                    )
                tok_idx = torch.arange(tok_n, device=x.device).unsqueeze(0)
                memory = memory + self.token_pos_embedding(tok_idx)
            return self.dropout(memory)

        if self.readout_type == "conv_patch_tokens":
            if self.patch_conv is None:
                raise RuntimeError("EncoderReadout conv patch tokenizer is not initialized.")
            if x.size(1) != self.conv_in_channels:
                raise ValueError(
                    f"Conv patch readout expected C={self.conv_in_channels}, got C={int(x.size(1))}."
                )
            # (B, Cin, H, W) -> (B, Ct, Ht, Wt)
            conv_tokens = self.patch_conv(x)
            if self.conv_output_h is not None and self.conv_output_w is not None:
                conv_tokens = F.adaptive_avg_pool2d(conv_tokens, (self.conv_output_h, self.conv_output_w))
            # Flatten spatial dimensions per token channel:
            # (B, Ct, Ht, Wt) -> (B, Ct, Ht*Wt)
            memory = conv_tokens.flatten(start_dim=2)
            if memory.size(-1) != self.emb_size:
                raise ValueError(
                    "Conv patch readout produced token dim "
                    f"{int(memory.size(-1))}, expected emb_size={self.emb_size}. "
                    "Adjust conv kernel/stride/padding, conv_output_h/conv_output_w, or emb_size."
                )
            if self.token_pos_embedding is not None:
                tok_n = int(memory.size(1))
                if tok_n > int(self.token_pos_embedding.num_embeddings):
                    raise ValueError(
                        f"Token count {tok_n} exceeds token positional embedding size "
                        f"{int(self.token_pos_embedding.num_embeddings)}."
                    )
                tok_idx = torch.arange(tok_n, device=x.device).unsqueeze(0)
                memory = memory + self.token_pos_embedding(tok_idx)
            return self.dropout(memory)

        raise RuntimeError(f"Unexpected encoder_readout.type '{self.readout_type}'.")


def _build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


def _resolve_activation(name: str) -> nn.Module:
    mode = str(name).strip().lower()
    if mode == "relu":
        return nn.ReLU()
    if mode == "gelu":
        return nn.GELU()
    if mode in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError(f"Unsupported decoder activation '{name}' (expected relu/gelu/silu).")


def _sinusoidal_position_encoding(max_len: int, emb_size: int) -> torch.Tensor:
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, emb_size, 2, dtype=torch.float32) * (-math.log(10000.0) / float(emb_size))
    )
    pe = torch.zeros(max_len, emb_size, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class CaptionEmbedding(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        emb_size: int,
        max_seq_len: int,
        dropout: float,
        use_learned_positional_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.emb_size = int(emb_size)
        self.max_seq_len = int(max_seq_len)
        if self.vocab_size <= 0:
            raise ValueError("decoder vocab_size must be positive.")
        if self.max_seq_len <= 0:
            raise ValueError("decoder max_seq_len must be positive.")
        self.token_embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.use_learned_positional_embedding = bool(use_learned_positional_embedding)
        if self.use_learned_positional_embedding:
            self.pos_embedding = nn.Embedding(self.max_seq_len, self.emb_size)
            self.register_buffer("sinusoidal_pos", None, persistent=False)
        else:
            self.pos_embedding = None
            pe = _sinusoidal_position_encoding(self.max_seq_len, self.emb_size)
            self.register_buffer("sinusoidal_pos", pe, persistent=False)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError(f"CaptionEmbedding expects input_ids (B,T), got {tuple(input_ids.shape)}")
        bsz, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds decoder max_seq_len={self.max_seq_len}."
            )
        tok = self.token_embedding(input_ids)  # (B, T, E)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        if self.use_learned_positional_embedding:
            pos = self.pos_embedding(pos_ids)
        else:
            pos = self.sinusoidal_pos[:seq_len].unsqueeze(0).to(input_ids.device, dtype=tok.dtype)
            pos = pos.expand(bsz, -1, -1)
        return self.dropout(tok + pos)


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        *,
        emb_size: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        attn_dropout: float,
        layer_norm_eps: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            emb_size,
            num_heads,
            dropout=float(attn_dropout),
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            emb_size,
            num_heads,
            dropout=float(attn_dropout),
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(emb_size, eps=float(layer_norm_eps))
        self.norm2 = nn.LayerNorm(emb_size, eps=float(layer_norm_eps))
        self.norm3 = nn.LayerNorm(emb_size, eps=float(layer_norm_eps))
        self.dropout = nn.Dropout(float(dropout))
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, int(ff_dim)),
            _resolve_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(int(ff_dim), emb_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        *,
        causal_mask: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor],
        memory_key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        self_attn_out, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=causal_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout(self_attn_out))

        cross_attn_out, _ = self.cross_attn(
            x,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        x = self.norm2(x + self.dropout(cross_attn_out))

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


class CaptionTransformerDecoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        emb_size: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        attn_dropout: float,
        max_seq_len: int,
        tie_weights: bool,
        use_learned_positional_embedding: bool,
        layer_norm_eps: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.embedding = CaptionEmbedding(
            vocab_size=int(vocab_size),
            emb_size=int(emb_size),
            max_seq_len=int(max_seq_len),
            dropout=float(dropout),
            use_learned_positional_embedding=bool(use_learned_positional_embedding),
        )
        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    emb_size=int(emb_size),
                    num_heads=int(num_heads),
                    ff_dim=int(ff_dim),
                    dropout=float(dropout),
                    attn_dropout=float(attn_dropout),
                    layer_norm_eps=float(layer_norm_eps),
                    activation=activation,
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )
        self.norm = nn.LayerNorm(int(emb_size), eps=float(layer_norm_eps))
        self.output_proj = nn.Linear(int(emb_size), int(vocab_size), bias=False)
        if bool(tie_weights):
            if self.output_proj.weight.shape == self.embedding.token_embedding.weight.shape:
                self.output_proj.weight = self.embedding.token_embedding.weight
            else:
                raise ValueError("Cannot tie decoder weights because embedding/output shapes do not match.")

    def forward(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor,
        *,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError(f"Decoder input_ids must be (B,T), got {tuple(input_ids.shape)}")
        if memory.dim() != 3:
            raise ValueError(f"Decoder memory must be (B,S,E), got {tuple(memory.shape)}")
        if input_ids.size(0) != memory.size(0):
            raise ValueError(
                f"Batch size mismatch between input_ids ({input_ids.size(0)}) and memory ({memory.size(0)})."
            )
        x = self.embedding(input_ids)  # (B,T,E)
        causal_mask = _build_causal_mask(int(input_ids.size(1)), input_ids.device)
        for block in self.layers:
            x = block(
                x,
                memory,
                causal_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        x = self.norm(x)
        logits = self.output_proj(x)  # (B,T,V)
        return logits


def _compute_caption_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    caption_loss_type: str,
    ignore_index: int,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    mode = str(caption_loss_type).strip().lower()
    if mode not in {"cross_entropy", "ce"}:
        raise ValueError(f"Unsupported caption_loss_type '{caption_loss_type}'. Use 'cross_entropy'.")
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=int(ignore_index),
        label_smoothing=float(label_smoothing),
        reduction=reduction,
    )


def _validate_vlm_batch_tensors(
    images: torch.Tensor,
    caption_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    if images.dim() != 4:
        raise ValueError(f"Batch['image'] must be (B,C,H,W), got {tuple(images.shape)}")
    if caption_ids.dim() != 2:
        raise ValueError(f"Batch['caption_ids'] must be (B,T), got {tuple(caption_ids.shape)}")
    if attention_mask.dim() != 2:
        raise ValueError(f"Batch['attention_mask'] must be (B,T), got {tuple(attention_mask.shape)}")
    if caption_ids.shape != attention_mask.shape:
        raise ValueError(
            "Batch['caption_ids'] and Batch['attention_mask'] must have identical shape, got "
            f"{tuple(caption_ids.shape)} vs {tuple(attention_mask.shape)}"
        )
    if images.size(0) != caption_ids.size(0):
        raise ValueError(
            f"Batch size mismatch between images and captions: {images.size(0)} vs {caption_ids.size(0)}"
        )


def _encode_optical_memory_batch(
    images: torch.Tensor,
    *,
    channel_num: int,
    data_method: str,
    input_adapter: nn.Module | None,
    input_hw: Tuple[int, int] | None,
    phase_system: PhaseSystem,
    bridge: OpticalBridge,
    mixers: nn.ModuleList,
    mixer_cfg: Dict[str, object],
    encoder_readout: EncoderReadout,
    track_grads: bool,
    fine_tune_state: Optional[object] = None,
    input_pe_cfg: Optional[Dict[str, object]] = None,
    input_preproc_cfg: Optional[Dict[str, object]] = None,
    encoder_input_skip_cfg: Optional[Dict[str, object]] = None,
) -> torch.Tensor:
    if images.dim() != 4:
        raise ValueError(f"Expected image batch (B,C,H,W), got {tuple(images.shape)}")
    memories: List[torch.Tensor] = []
    batch_size = int(images.size(0))
    for b in range(batch_size):
        x_single = images[b:b + 1]
        x_channels = _prepare_input_channels(
            x_single,
            channel_num=channel_num,
            data_method=data_method,
            input_adapter=input_adapter,
            upsample_hw=input_hw,
            input_preproc_cfg=input_preproc_cfg,
        )  # (1,C,H,W)
        x_phys = x_channels.squeeze(0).unsqueeze(1)  # (C,1,H,W)
        x_phys = _apply_input_positional_encoding(x_phys, input_pe_cfg)
        y_final, layer_inputs, layer_phases, layer_outputs, layer_inputs_pre = _forward_with_mix(
            x_phys,
            phase_system,
            bridge,
            mixers,
            mixer_cfg,
            track_grads=track_grads,
        )
        if fine_tune_state is not None and track_grads:
            accumulate_fine_tune_samples(
                fine_tune_state,
                layer_inputs=layer_inputs,
                layer_phases=layer_phases,
                layer_outputs=layer_outputs,
                layer_raw_inputs=layer_inputs_pre,
            )
        # Convert physical-batch output (C,1,H,W) to standard batch (1,C,H,W).
        y_map = y_final.squeeze(1).unsqueeze(0)
        readout_input = _build_encoder_readout_input(y_map, x_channels, encoder_input_skip_cfg)
        memory = encoder_readout(readout_input)  # (1,C,E)
        memories.append(memory)
    return torch.cat(memories, dim=0)  # (B,C,E)


def _extract_phase_channel_for_viz(
    phase_system: PhaseSystem,
    bridge: OpticalBridge,
    *,
    layer_idx: int,
    channel_idx: int,
) -> torch.Tensor:
    phase_params = phase_system.phase_parameters()
    if not phase_params:
        raise ValueError("No phase parameters available for visualization.")
    layer_idx = max(0, min(int(layer_idx), len(phase_params) - 1))
    phase_to_plot = phase_to_unit(phase_params[layer_idx])
    cached_phase = bridge.phase_cache[layer_idx] if layer_idx < len(bridge.phase_cache) else None
    if isinstance(cached_phase, torch.Tensor) and cached_phase.numel() > 0:
        try:
            bsz, _, h, w = cached_phase.shape
            channels = int(phase_system.num_channels)
            reps = max(1, bsz // max(1, channels))
            reshaped = cached_phase.view(reps, channels, 1, h, w)
            phase_to_plot = reshaped[0, :, 0]
        except Exception:
            pass
    if phase_to_plot.dim() == 4:
        phase_to_plot = phase_to_plot[0]
    if phase_to_plot.dim() == 2:
        phase_to_plot = phase_to_plot.unsqueeze(0)
    if phase_to_plot.dim() != 3:
        raise ValueError(f"Unexpected phase tensor shape for viz: {tuple(phase_to_plot.shape)}")
    ch = max(0, min(int(channel_idx), int(phase_to_plot.size(0)) - 1))
    return phase_to_plot[ch].detach()


@torch.no_grad()
def _collect_phase_trace_rows(
    images: torch.Tensor,
    *,
    channel_indices_one_based: List[int],
    channel_num: int,
    data_method: str,
    input_adapter: nn.Module | None,
    input_hw: Tuple[int, int] | None,
    phase_system: PhaseSystem,
    bridge: OpticalBridge,
    mixers: nn.ModuleList,
    mixer_cfg: Dict[str, object],
    encoder_readout: EncoderReadout,
    input_pe_cfg: Optional[Dict[str, object]] = None,
    input_preproc_cfg: Optional[Dict[str, object]] = None,
    encoder_input_skip_cfg: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    if images.dim() != 4 or images.size(0) <= 0:
        return []
    x_single = images[0:1]
    x_channels = _prepare_input_channels(
        x_single,
        channel_num=channel_num,
        data_method=data_method,
        input_adapter=input_adapter,
        upsample_hw=input_hw,
        input_preproc_cfg=input_preproc_cfg,
    )  # (1,C,H,W)
    x_phys = x_channels.squeeze(0).unsqueeze(1)  # (C,1,H,W)
    x_phys = _apply_input_positional_encoding(x_phys, input_pe_cfg)
    y_final, layer_inputs, _, _, _ = _forward_with_mix(
        x_phys,
        phase_system,
        bridge,
        mixers,
        mixer_cfg,
        track_grads=False,
    )
    y_map = y_final.squeeze(1).unsqueeze(0)  # (1,C,H,W)
    encoded_l1 = layer_inputs[0] if layer_inputs else None

    was_training = encoder_readout.training
    encoder_readout.eval()
    try:
        readout_input = _build_encoder_readout_input(y_map, x_channels, encoder_input_skip_cfg)
        memory = encoder_readout(readout_input)  # (1,S,E)
    finally:
        if was_training:
            encoder_readout.train()

    img_rgb = x_single[0].detach().cpu()
    if img_rgb.dim() == 3 and img_rgb.size(0) == 1:
        img_rgb = img_rgb.repeat(3, 1, 1)
    rows: List[Dict[str, object]] = []
    last_layer_idx = max(0, len(phase_system.phase_parameters()) - 1)
    for ch_one_based in channel_indices_one_based:
        k = int(ch_one_based)
        ch_idx = max(0, k - 1)
        input_ch = min(ch_idx, int(x_channels.size(1)) - 1)
        out_ch = min(ch_idx, int(y_map.size(1)) - 1)
        tok_idx = min(ch_idx, int(memory.size(1)) - 1)
        if isinstance(encoded_l1, torch.Tensor) and encoded_l1.dim() == 4 and encoded_l1.size(1) >= 1:
            encoded_input_map = encoded_l1[input_ch, 0].detach().cpu()
        else:
            encoded_input_map = x_channels[0, input_ch].detach().cpu()
        token_vec = memory[0, tok_idx].detach().cpu()
        side = int(math.isqrt(int(token_vec.numel())))
        if side * side == int(token_vec.numel()):
            token_map = token_vec.view(side, side)
        else:
            token_map = token_vec.view(1, -1)
        rows.append(
            {
                "channel_one_based": k,
                "input_rgb": img_rgb,
                "encoded_input_map": encoded_input_map,
                "phase_first": _extract_phase_channel_for_viz(
                    phase_system,
                    bridge,
                    layer_idx=0,
                    channel_idx=ch_idx,
                ).cpu(),
                "phase_last": _extract_phase_channel_for_viz(
                    phase_system,
                    bridge,
                    layer_idx=last_layer_idx,
                    channel_idx=ch_idx,
                ).cpu(),
                "onn_last_output": y_map[0, out_ch].detach().cpu(),
                "decoder_token_map": token_map.cpu(),
            }
        )
    return rows


def _plot_vlm_phase_trace_panel(
    state: Dict[str, object],
    rows: List[Dict[str, object]],
    *,
    iteration: int,
) -> None:
    if plt is None or not rows:
        return
    output_widget = state.get("output")
    prev_fig = state.get("fig")
    if prev_fig is not None:
        try:
            plt.close(prev_fig)
        except Exception:
            pass

    n_rows = len(rows)
    n_cols = 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.8 * n_rows), squeeze=False)
    fig.patch.set_facecolor("#1e1e1e")
    fig.suptitle(f"Phase/Channel Trace (iter {iteration})", color="#e7ebff", fontsize=14, fontweight="bold")

    def _draw_map(ax, arr: np.ndarray, title: str, *, cmap: str = "magma", fixed_01: bool = False) -> None:
        ax.set_facecolor("#1e1e1e")
        if fixed_01:
            vmin, vmax = 0.0, 1.0
        else:
            finite = np.isfinite(arr)
            if not finite.any():
                vmin, vmax = 0.0, 1.0
                arr = np.zeros_like(arr, dtype=np.float32)
            else:
                vmin = float(np.nanpercentile(arr, 1.0))
                vmax = float(np.nanpercentile(arr, 99.0))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                    vmin = float(np.nanmin(arr))
                    vmax = float(np.nanmax(arr))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                    vmin, vmax = 0.0, 1.0
        im = ax.imshow(arr, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(title, color="#e7ebff", fontsize=10, fontweight="bold")
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=7, colors="#e7ebff")
        cbar.outline.set_edgecolor("#e7ebff")

    def _prepare_input_for_imshow(raw: object) -> Tuple[np.ndarray, str | None]:
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 2:
            return np.clip(arr, 0.0, 1.0), "gray"
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                return np.clip(arr[0], 0.0, 1.0), "gray"
            if arr.shape[0] in {3, 4}:
                rgb = np.transpose(arr[:3], (1, 2, 0))
                return np.clip(rgb, 0.0, 1.0), None
            if arr.shape[-1] in {3, 4}:
                rgb = arr[..., :3]
                return np.clip(rgb, 0.0, 1.0), None
            # For non-RGB multi-channel tensors (e.g., spectral stacks), visualize
            # a channel-mean 2D projection so matplotlib receives valid image data.
            return np.clip(np.nanmean(arr, axis=0), 0.0, 1.0), "gray"
        squeezed = np.squeeze(arr)
        if squeezed.ndim == 2:
            return np.clip(squeezed, 0.0, 1.0), "gray"
        return np.zeros((1, 1), dtype=np.float32), "gray"

    for r_idx, row in enumerate(rows):
        k = int(row["channel_one_based"])
        input_rgb = row["input_rgb"]
        if isinstance(input_rgb, torch.Tensor):
            img = input_rgb.detach().cpu().numpy()
        else:
            img = np.asarray(input_rgb)
        img, img_cmap = _prepare_input_for_imshow(img)
        ax0 = axes[r_idx, 0]
        ax0.set_facecolor("#1e1e1e")
        if img_cmap is None:
            ax0.imshow(img, interpolation="nearest")
        else:
            ax0.imshow(img, interpolation="nearest", cmap=img_cmap, vmin=0.0, vmax=1.0)
        ax0.set_title(f"Input image (k={k})", color="#e7ebff", fontsize=10, fontweight="bold")
        ax0.axis("off")

        encoded_input_map = np.asarray(row["encoded_input_map"], dtype=np.float32)
        phase_first = np.asarray(row["phase_first"], dtype=np.float32)
        phase_last = np.asarray(row["phase_last"], dtype=np.float32)
        onn_last = np.asarray(row["onn_last_output"], dtype=np.float32)
        dec_map = np.asarray(row["decoder_token_map"], dtype=np.float32)

        _draw_map(axes[r_idx, 1], encoded_input_map, f"Step5 Encoded input ch {k}")
        _draw_map(axes[r_idx, 2], phase_first, f"Phase L1 ch {k}", fixed_01=True)
        _draw_map(axes[r_idx, 3], phase_last, f"Phase LN ch {k}", fixed_01=True)
        _draw_map(axes[r_idx, 4], onn_last, f"ONN last output ch {k}")
        _draw_map(axes[r_idx, 5], dec_map, f"Decoder memory token {k}")

    fig.subplots_adjust(left=0.02, right=0.995, top=0.90, bottom=0.04, wspace=0.28, hspace=0.35)
    handle = state.get("handle")
    if handle is None:
        if output_widget is not None:
            with output_widget:
                handle = display(fig, display_id=True)
        else:
            handle = display(fig, display_id=True)
    elif hasattr(handle, "update"):
        handle.update(fig)
    state.update(
        {
            "handle": handle,
            "fig": fig,
            "axes": axes,
            "rows": n_rows,
            "cols": n_cols,
            "output": output_widget,
            "mode": "vlm_phase_trace",
        }
    )


def _decode_token_ids(
    tokenizer: object,
    token_ids: List[int],
    *,
    skip_special_tokens: bool = True,
) -> str:
    ids = [int(x) for x in token_ids]
    if hasattr(tokenizer, "decode"):
        try:
            return str(tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)).strip()
        except TypeError:
            return str(tokenizer.decode(ids)).strip()
    return " ".join(str(x) for x in ids)


def _bleu_tokenize_caption(text: str) -> List[str]:
    content = "" if text is None else str(text)
    content = unicodedata.normalize("NFKC", content).lower()
    content = content.translate(str.maketrans({ch: " " for ch in string.punctuation}))
    content = re.sub(r"\s+", " ", content).strip()
    if not content:
        return []
    return content.split(" ")


def _extract_ngrams(tokens: List[str], n: int) -> Counter:
    if n <= 0:
        raise ValueError("n must be positive for n-gram extraction.")
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _closest_reference_length(candidate_len: int, reference_lens: List[int]) -> int:
    if not reference_lens:
        return 0
    return min(reference_lens, key=lambda r: (abs(r - candidate_len), r))


def _compute_bleu_1_to_4(
    predictions: List[str],
    references: List[List[str]],
) -> Dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must have the same length, got {len(predictions)} vs {len(references)}."
        )
    if not predictions:
        return {"bleu1": float("nan"), "bleu2": float("nan"), "bleu3": float("nan"), "bleu4": float("nan")}

    match_counts = [0.0, 0.0, 0.0, 0.0]
    total_counts = [0.0, 0.0, 0.0, 0.0]
    cand_total_len = 0
    ref_total_len = 0

    for pred_caption, ref_caption_list in zip(predictions, references):
        pred_tokens = _bleu_tokenize_caption(pred_caption)
        ref_tokens_list = [_bleu_tokenize_caption(ref) for ref in ref_caption_list if str(ref).strip()]
        if not ref_tokens_list:
            continue

        cand_len = len(pred_tokens)
        ref_lens = [len(tokens) for tokens in ref_tokens_list]
        cand_total_len += cand_len
        ref_total_len += _closest_reference_length(cand_len, ref_lens)

        for n in range(1, 5):
            pred_ngrams = _extract_ngrams(pred_tokens, n)
            pred_total = float(sum(pred_ngrams.values()))
            total_counts[n - 1] += pred_total
            if pred_total <= 0:
                continue
            max_ref_counts: Counter = Counter()
            for ref_tokens in ref_tokens_list:
                ref_counts = _extract_ngrams(ref_tokens, n)
                for gram, count in ref_counts.items():
                    if count > max_ref_counts.get(gram, 0):
                        max_ref_counts[gram] = count
            overlap = 0
            for gram, count in pred_ngrams.items():
                overlap += min(count, max_ref_counts.get(gram, 0))
            match_counts[n - 1] += float(overlap)

    if cand_total_len <= 0:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

    precisions: List[float] = []
    for idx in range(4):
        denom = total_counts[idx]
        if denom <= 0:
            precisions.append(0.0)
        else:
            precisions.append(match_counts[idx] / denom)

    if cand_total_len > ref_total_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1.0 - (float(ref_total_len) / float(cand_total_len)))

    scores: Dict[str, float] = {}
    for n in range(1, 5):
        p_slice = precisions[:n]
        if any(p <= 0.0 for p in p_slice):
            bleu_n = 0.0
        else:
            avg_log_precision = sum(math.log(p) for p in p_slice) / float(n)
            bleu_n = brevity_penalty * math.exp(avg_log_precision)
        scores[f"bleu{n}"] = float(bleu_n)
    return scores


def _nan_bleu_scores() -> Dict[str, float]:
    return {"bleu1": float("nan"), "bleu2": float("nan"), "bleu3": float("nan"), "bleu4": float("nan")}


def _rolling_average(values: List[float], window: int) -> float:
    if not values:
        return float("nan")
    w = max(1, int(window))
    tail = values[-w:]
    finite = [float(v) for v in tail if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not finite:
        return float("nan")
    return float(sum(finite) / float(len(finite)))


def _rolling_bleu_from_history(
    history: Dict[str, List[float]],
    *,
    prefix: str,
    window: int,
) -> Dict[str, float]:
    return {
        "bleu1": _rolling_average(history.get(f"{prefix}_bleu1", []), window),
        "bleu2": _rolling_average(history.get(f"{prefix}_bleu2", []), window),
        "bleu3": _rolling_average(history.get(f"{prefix}_bleu3", []), window),
        "bleu4": _rolling_average(history.get(f"{prefix}_bleu4", []), window),
    }


def _build_references_by_image(dataset: object) -> Dict[str, List[str]]:
    refs_by_image: Dict[str, List[str]] = {}
    samples = getattr(dataset, "samples", None)
    if not isinstance(samples, list):
        return refs_by_image
    for sample in samples:
        image_name = getattr(sample, "image_filename", None)
        caption = getattr(sample, "caption", None)
        if image_name is None or caption is None:
            continue
        key = str(image_name)
        refs_by_image.setdefault(key, []).append(str(caption))
    return refs_by_image


@torch.no_grad()
def _compute_batch_bleu_from_memory(
    memory: torch.Tensor,
    image_filenames: object,
    *,
    tokenizer: object,
    caption_decoder: CaptionTransformerDecoder,
    references_by_image: Dict[str, List[str]],
    bos_idx: int,
    eos_idx: Optional[int],
    pad_idx: int,
    max_gen_len: int,
    generation_method: str,
    stop_on_eos: bool,
    beam_size: int,
) -> Dict[str, float]:
    if not isinstance(image_filenames, list) or not image_filenames:
        return _nan_bleu_scores()
    if not isinstance(references_by_image, dict) or not references_by_image:
        return _nan_bleu_scores()

    select_indices: List[int] = []
    references: List[List[str]] = []
    seen: set[str] = set()
    for idx, image_name in enumerate(image_filenames):
        key = str(image_name)
        if key in seen:
            continue
        refs = references_by_image.get(key, [])
        if not refs:
            continue
        seen.add(key)
        select_indices.append(int(idx))
        references.append([str(ref) for ref in refs])
    if not select_indices:
        return _nan_bleu_scores()

    idx_tensor = torch.as_tensor(select_indices, device=memory.device, dtype=torch.long)
    memory_sel = memory.index_select(0, idx_tensor)
    decoder_was_training = bool(caption_decoder.training)
    caption_decoder.eval()
    token_ids = _generate_captions_from_memory(
        memory_sel,
        caption_decoder,
        bos_idx=int(bos_idx),
        eos_idx=eos_idx,
        pad_idx=int(pad_idx),
        max_gen_len=int(max_gen_len),
        generation_method=str(generation_method),
        stop_on_eos=bool(stop_on_eos),
        beam_size=int(beam_size),
    )
    predictions = [
        _decode_token_ids(tokenizer, token_ids[i].tolist(), skip_special_tokens=True)
        for i in range(int(token_ids.size(0)))
    ]
    if decoder_was_training:
        caption_decoder.train()
    return _compute_bleu_1_to_4(predictions, references)


def _init_vlm_metrics_plot_state(
    *,
    log_scale: bool = False,
    output_widget: Optional[object] = None,
) -> Dict[str, object]:
    return {
        "handle": None,
        "fig": None,
        "ax_loss": None,
        "ax_bleu": None,
        "loss_train_line": None,
        "loss_eval_line": None,
        "loss_train_x": [],
        "loss_train_y": [],
        "loss_eval_x": [],
        "loss_eval_y": [],
        "last_train_bleu": _nan_bleu_scores(),
        "last_eval_bleu": _nan_bleu_scores(),
        "last_update": -1,
        "log_scale": bool(log_scale),
        "output": output_widget,
    }


def _format_bleu_value(val: float) -> str:
    if isinstance(val, (int, float)) and math.isfinite(float(val)):
        return f"{float(val):.4f}"
    return "--"


def _draw_bleu_table(
    ax_bleu,
    train_bleu_scores: Dict[str, float],
    eval_bleu_scores: Dict[str, float],
) -> None:
    ax_bleu.clear()
    ax_bleu.set_facecolor("#1e1e1e")
    ax_bleu.axis("off")
    ax_bleu.set_title("Caption BLEU", color="#f5f7ff", fontsize=13, fontweight="bold", pad=8.0)

    rows = [("BLEU-1", "bleu1"), ("BLEU-2", "bleu2"), ("BLEU-3", "bleu3"), ("BLEU-4", "bleu4")]
    cell_text = [
        [
            label,
            _format_bleu_value(float(train_bleu_scores.get(key, float("nan")))),
            _format_bleu_value(float(eval_bleu_scores.get(key, float("nan")))),
        ]
        for label, key in rows
    ]
    table = ax_bleu.table(
        cellText=cell_text,
        colLabels=["Metric", "Score (Train)", "Score (Eval)"],
        colLoc="center",
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.1, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#2e394f")
        if row == 0:
            cell.set_facecolor("#2a3144")
            cell.set_text_props(color="#f5f7ff", weight="bold")
        else:
            cell.set_facecolor("#1e1e1e")
            cell.set_text_props(color="#e7ebff")


def _update_vlm_metrics_plot(
    state: Dict[str, object],
    iteration: int,
    update_every: int,
    *,
    train_loss: Optional[float] = None,
    eval_loss: Optional[float] = None,
    train_bleu_scores: Optional[Dict[str, float]] = None,
    eval_bleu_scores: Optional[Dict[str, float]] = None,
    force: bool = False,
) -> None:
    output_widget = state.get("output")
    if train_loss is not None:
        state["loss_train_x"].append(iteration)
        state["loss_train_y"].append(train_loss)
    if eval_loss is not None:
        state["loss_eval_x"].append(iteration)
        state["loss_eval_y"].append(eval_loss)
    if train_bleu_scores is not None:
        state["last_train_bleu"] = {
            "bleu1": float(train_bleu_scores.get("bleu1", float("nan"))),
            "bleu2": float(train_bleu_scores.get("bleu2", float("nan"))),
            "bleu3": float(train_bleu_scores.get("bleu3", float("nan"))),
            "bleu4": float(train_bleu_scores.get("bleu4", float("nan"))),
        }
    if eval_bleu_scores is not None:
        state["last_eval_bleu"] = {
            "bleu1": float(eval_bleu_scores.get("bleu1", float("nan"))),
            "bleu2": float(eval_bleu_scores.get("bleu2", float("nan"))),
            "bleu3": float(eval_bleu_scores.get("bleu3", float("nan"))),
            "bleu4": float(eval_bleu_scores.get("bleu4", float("nan"))),
        }

    if state.get("handle") is None:
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2), gridspec_kw={"width_ratios": [2.8, 1.2]})
        ax_loss, ax_bleu = axes
        fig.patch.set_facecolor("#1e1e1e")
        for ax in [ax_loss, ax_bleu]:
            ax.set_facecolor("#1e1e1e")
        ax_loss.tick_params(colors="#f5f7ff", labelsize=11)
        for label in ax_loss.get_xticklabels() + ax_loss.get_yticklabels():
            label.set_fontweight("bold")
        ax_loss.grid(True, alpha=0.35, color="#2e394f")
        ax_loss.set_title("ONN Training & Eval Loss", color="#f5f7ff", fontsize=14, fontweight="bold")
        ax_loss.set_xlabel("Iteration", color="#f5f7ff", fontsize=12, fontweight="bold")
        ax_loss.set_ylabel("Loss", color="#f5f7ff", fontsize=12, fontweight="bold")
        if bool(state.get("log_scale", False)):
            ax_loss.set_yscale("log")
        (loss_train_line,) = ax_loss.plot(
            [], [], label="train", color="#4dd0e1", linewidth=1.8, marker="o", markersize=3
        )
        (loss_eval_line,) = ax_loss.plot(
            [], [], label="eval", color="#ffd54f", linewidth=1.8, marker="s", markersize=3
        )
        leg = ax_loss.legend(loc="upper right")
        if leg:
            frame = leg.get_frame()
            frame.set_facecolor("#1e1e1e")
            frame.set_edgecolor("#2e394f")
            for txt in leg.get_texts():
                txt.set_color("#f5f7ff")
                txt.set_fontweight("bold")
                txt.set_fontsize(11)

        _draw_bleu_table(
            ax_bleu,
            state.get("last_train_bleu", _nan_bleu_scores()),
            state.get("last_eval_bleu", _nan_bleu_scores()),
        )
        fig.tight_layout()
        if output_widget is not None:
            with output_widget:
                handle = display(fig, display_id=True)
        else:
            handle = display(fig, display_id=True)
        state.update(
            {
                "handle": handle,
                "fig": fig,
                "ax_loss": ax_loss,
                "ax_bleu": ax_bleu,
                "loss_train_line": loss_train_line,
                "loss_eval_line": loss_eval_line,
                "output": output_widget,
            }
        )
        state["last_update"] = -1

    last = int(state.get("last_update", -1))
    if last == -1:
        force = True
    if not force and (iteration - last) < int(update_every):
        return

    ax_loss = state["ax_loss"]
    state["loss_train_line"].set_data(state["loss_train_x"], state["loss_train_y"])
    state["loss_eval_line"].set_data(state["loss_eval_x"], state["loss_eval_y"])

    loss_x = state["loss_train_x"] + state["loss_eval_x"]
    loss_y = state["loss_train_y"] + state["loss_eval_y"]
    if loss_x:
        xmin, xmax = min(loss_x), max(loss_x)
        if xmin == xmax:
            xmin -= 1
            xmax += 1
        ax_loss.set_xlim(xmin, xmax + 1)
    if loss_y:
        ax_loss.relim()
        ax_loss.autoscale_view(scalex=False, scaley=True)

    _draw_bleu_table(
        state["ax_bleu"],
        state.get("last_train_bleu", _nan_bleu_scores()),
        state.get("last_eval_bleu", _nan_bleu_scores()),
    )

    state["fig"].canvas.draw_idle()
    handle = state.get("handle")
    if handle is not None and hasattr(handle, "update"):
        handle.update(state["fig"])
    state["last_update"] = iteration


def _init_caption_examples_viz_state(output_widget: Optional[object] = None) -> Dict[str, object]:
    return {
        "fig": None,
        "axes": None,
        "handle": None,
        "output": output_widget,
    }


def _image_to_display_array(image: torch.Tensor) -> np.ndarray:
    if image.dim() != 3:
        raise ValueError(f"Expected image (C,H,W), got {tuple(image.shape)}")
    img = image.detach().cpu().float().clamp(0.0, 1.0)
    if img.size(0) == 1:
        return img[0].numpy()
    if img.size(0) in {3, 4}:
        rgb = img[:3]
        # Keep grayscale previews grayscale even when loaded as RGB triplets.
        if torch.max(torch.abs(rgb[0] - rgb[1])) < 1e-6 and torch.max(torch.abs(rgb[1] - rgb[2])) < 1e-6:
            return rgb[0].numpy()
        return rgb.permute(1, 2, 0).contiguous().numpy()
    # Non-image channel stacks (e.g., patchified C=16) are shown as grayscale summary.
    return img.mean(dim=0).numpy()


def _extract_preview_image_from_dataset(
    dataset: object,
    idx: int,
    *,
    fallback: torch.Tensor,
) -> torch.Tensor:
    image = fallback
    try:
        image_pipeline = getattr(dataset, "image_pipeline", None)
        base_dataset = getattr(image_pipeline, "dataset", None)
        if base_dataset is not None:
            base_item = base_dataset[int(idx)]
            if isinstance(base_item, (list, tuple)) and len(base_item) >= 1 and isinstance(base_item[0], torch.Tensor):
                image = base_item[0]
    except Exception:
        image = fallback

    if not isinstance(image, torch.Tensor):
        return fallback.detach().cpu().float().clamp(0.0, 1.0)
    img = image.detach().cpu().float()
    if img.dim() == 2:
        img = img.unsqueeze(0)
    if img.dim() != 3:
        return fallback.detach().cpu().float().clamp(0.0, 1.0)
    # Keep preview grayscale by default.
    if img.size(0) > 1:
        if img.size(0) in {3, 4}:
            rgb = img[:3]
            img = (0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]).unsqueeze(0)
        else:
            img = img.mean(dim=0, keepdim=True)
    return img.clamp(0.0, 1.0)


def _update_caption_examples_plot(
    state: Dict[str, object],
    images: torch.Tensor,
    gt_texts: List[str],
    pred_texts: List[str],
    file_names: List[str],
    source_labels: Optional[List[str]] = None,
) -> None:
    if plt is None:
        return
    output_widget = state.get("output")
    n_examples = min(2, int(images.size(0)), len(gt_texts), len(pred_texts))
    if n_examples <= 0:
        return

    if state.get("fig") is None or state.get("axes") is None:
        fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), width_ratios=[0.8, 2.2])
        fig.patch.set_facecolor("#1e1e1e")
        handle = None
        if output_widget is not None:
            with output_widget:
                handle = display(fig, display_id=True) if display is not None else None
        elif display is not None:
            handle = display(fig, display_id=True)
        state["fig"] = fig
        state["axes"] = np.atleast_2d(axes)
        state["handle"] = handle

    fig = state["fig"]
    axes = np.atleast_2d(state["axes"])
    fig.patch.set_facecolor("#1e1e1e")
    for ax in axes.ravel():
        ax.clear()
        ax.set_facecolor("#1e1e1e")
        ax.axis("off")

    for i in range(n_examples):
        image_ax = axes[i, 0]
        text_ax = axes[i, 1]
        source_name = (
            str(source_labels[i]) if source_labels is not None and i < len(source_labels) else f"Example {i + 1}"
        )
        img_np = _image_to_display_array(images[i])
        if img_np.ndim == 2:
            image_ax.imshow(img_np, cmap="gray", vmin=0.0, vmax=1.0)
        else:
            image_ax.imshow(img_np)
        image_ax.set_title(
            f"{source_name}: {file_names[i]}",
            fontsize=10,
            fontweight="bold",
            color="#e7ebff",
            pad=6.0,
        )
        image_ax.axis("off")

        gt_wrapped = textwrap.fill(gt_texts[i], width=78)
        pred_wrapped = textwrap.fill(pred_texts[i], width=78)
        text_ax.text(
            0.06,
            0.98,
            f"$\\bf{{True:}}$\n{gt_wrapped}\n\n$\\bf{{Pred:}}$\n{pred_wrapped}",
            va="top",
            ha="left",
            fontsize=12,
            family="monospace",
            color="#e7ebff",
            wrap=True,
            clip_on=True,
        )
        text_ax.set_title(
            f"{source_name} Caption",
            loc="left",
            fontsize=14,
            fontweight="bold",
            color="#e7ebff",
            pad=6.0,
        )
        text_ax.axis("off")

    fig.subplots_adjust(left=0.03, right=0.99, top=0.97, bottom=0.05, wspace=0.18, hspace=0.22)
    handle = state.get("handle")
    if handle is not None and hasattr(handle, "update"):
        handle.update(fig)
    else:
        fig.canvas.draw_idle()
        try:
            plt.pause(0.001)
        except Exception:
            pass


def _generate_captions_from_memory(
    memory: torch.Tensor,
    decoder: CaptionTransformerDecoder,
    *,
    bos_idx: int,
    eos_idx: Optional[int],
    pad_idx: int,
    max_gen_len: int,
    generation_method: str = "greedy",
    stop_on_eos: bool = True,
    beam_size: int = 3,
) -> torch.Tensor:
    try:
        dec_param = next(decoder.parameters())
        dec_device = dec_param.device
        dec_dtype = dec_param.dtype
    except StopIteration:
        dec_device = memory.device
        dec_dtype = memory.dtype
    if memory.device != dec_device or memory.dtype != dec_dtype:
        memory = memory.to(device=dec_device, dtype=dec_dtype)

    mode = str(generation_method).strip().lower()
    parsed_beam_size: Optional[int] = None
    beam_match = re.fullmatch(r"beam[_-]?(\d+)", mode)
    if beam_match is not None:
        parsed_beam_size = int(beam_match.group(1))
        mode = "beam"
    if mode not in {"greedy", "beam", "beam_search"}:
        raise ValueError(
            f"Unsupported generation_method '{generation_method}'. "
            "Use one of {'greedy', 'beam', 'beam_search', 'beam3', 'beam5'}."
        )
    beam_k = int(parsed_beam_size if parsed_beam_size is not None else beam_size)
    if beam_k <= 0:
        raise ValueError(f"beam_size must be >= 1, got {beam_k}.")
    batch_size = int(memory.size(0))

    if mode == "greedy":
        generated = torch.full(
            (batch_size, 1),
            int(bos_idx),
            device=memory.device,
            dtype=torch.long,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=memory.device)
        max_steps = max(1, int(max_gen_len) - 1)
        for _ in range(max_steps):
            tgt_pad_mask = generated.eq(int(pad_idx))
            logits = decoder(generated, memory, tgt_key_padding_mask=tgt_pad_mask)
            next_token = logits[:, -1, :].argmax(dim=-1)
            if eos_idx is not None and bool(stop_on_eos):
                next_token = torch.where(finished, torch.full_like(next_token, int(pad_idx)), next_token)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            if eos_idx is not None and bool(stop_on_eos):
                finished = finished | next_token.eq(int(eos_idx))
                if bool(finished.all()):
                    break
        return generated

    # Beam search path (per-sample decoding for clarity and stability).
    def _rank_score(logprob_sum: float, seq_len: int) -> float:
        # Length-normalized ranking reduces short-caption bias.
        norm = float(max(1, seq_len - 1))
        return float(logprob_sum) / norm

    max_steps = max(1, int(max_gen_len) - 1)
    beam_outputs: List[torch.Tensor] = []
    for sample_idx in range(batch_size):
        mem_i = memory[sample_idx:sample_idx + 1]
        beams: List[Tuple[List[int], float, bool]] = [([int(bos_idx)], 0.0, False)]
        for _ in range(max_steps):
            candidates: List[Tuple[List[int], float, bool]] = []
            all_finished = True
            for seq, score_sum, finished_flag in beams:
                if finished_flag and bool(stop_on_eos):
                    candidates.append((seq, score_sum, True))
                    continue
                all_finished = False
                seq_tensor = torch.tensor(seq, device=mem_i.device, dtype=torch.long).unsqueeze(0)
                tgt_pad_mask = seq_tensor.eq(int(pad_idx))
                logits = decoder(seq_tensor, mem_i, tgt_key_padding_mask=tgt_pad_mask)
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
                topk = min(int(beam_k), int(log_probs.numel()))
                top_vals, top_idx = torch.topk(log_probs, k=topk, dim=-1)
                for lp, token_idx in zip(top_vals.tolist(), top_idx.tolist()):
                    tok = int(token_idx)
                    new_seq = seq + [tok]
                    new_score_sum = float(score_sum) + float(lp)
                    new_finished = bool(
                        eos_idx is not None and bool(stop_on_eos) and tok == int(eos_idx)
                    )
                    candidates.append((new_seq, new_score_sum, new_finished))
            if not candidates:
                break
            candidates.sort(key=lambda x: _rank_score(x[1], len(x[0])), reverse=True)
            beams = candidates[: int(beam_k)]
            if all_finished:
                break

        finished_beams = [b for b in beams if b[2]]
        select_pool = finished_beams if (finished_beams and bool(stop_on_eos)) else beams
        best_seq, _, _ = max(select_pool, key=lambda x: _rank_score(x[1], len(x[0])))
        beam_outputs.append(torch.tensor(best_seq, device=memory.device, dtype=torch.long))

    out_len = max(int(t.size(0)) for t in beam_outputs) if beam_outputs else 1
    generated = torch.full(
        (batch_size, out_len),
        device=memory.device,
        fill_value=int(pad_idx),
        dtype=torch.long,
    )
    for idx, seq_tensor in enumerate(beam_outputs):
        seq_len = int(seq_tensor.size(0))
        generated[idx, :seq_len] = seq_tensor
    return generated


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
    saved_interlayer_states: List[torch.Tensor] = []

    phase_params = list(phase_system.phase_parameters())
    n_layers = len(phase_params)

    reverse_skip_cfg_raw = mixer_cfg.get("reverse_skip", {})
    reverse_skip_cfg = reverse_skip_cfg_raw if isinstance(reverse_skip_cfg_raw, dict) else {}
    reverse_skip_enabled = bool(reverse_skip_cfg.get("enabled", True))
    reverse_skip_alpha = float(reverse_skip_cfg.get("alpha", 1.0))
    reverse_skip_x_mode = str(reverse_skip_cfg.get("x_mode", "interp")).strip().lower()
    reverse_skip_x_norm_mod = reverse_skip_cfg.get("x_norm_mod", "percentile")
    reverse_skip_fusion_cfg = dict(reverse_skip_cfg)
    reverse_skip_fusion_cfg["norm"] = str(reverse_skip_cfg.get("fusion_norm", "percentile")).strip().lower()
    reverse_skip_fusion_cfg.setdefault("percentile_peak", mixer_cfg.get("percentile_peak", 0.99))
    reverse_skip_fusion_cfg.setdefault("percentile_q", mixer_cfg.get("percentile_q", 0.99))
    reverse_skip_fusion_cfg.setdefault("percentile_eps", mixer_cfg.get("percentile_eps", 1e-6))

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

    for idx, phase_param in enumerate(phase_params):
        if reverse_skip_enabled:
            src_idx = n_layers - 1 - idx
            # Mirror layer idx to source layer (N-1-idx) and only use already-computed states.
            if src_idx < idx and src_idx < len(saved_interlayer_states):
                x_skip_src = saved_interlayer_states[src_idx]
                skip_phase = torch.zeros(
                    x_skip_src.size(0),
                    1,
                    *x_current.shape[-2:],
                    device=x_skip_src.device,
                    dtype=x_skip_src.dtype,
                )
                x_skip, _ = encoding_x_phase_physical(
                    x_skip_src,
                    skip_phase,
                    canvas_hw=x_current.shape[-2:],
                    x_mode=reverse_skip_x_mode,
                    x_norm_mod=reverse_skip_x_norm_mod,
                )
                x_current = x_current + (reverse_skip_alpha * x_skip)
                x_current = _apply_mix_norm(
                    x_current.squeeze(1).unsqueeze(0),
                    reverse_skip_fusion_cfg,
                ).squeeze(0).unsqueeze(1)

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
        # Save the actual normalized inter-layer tensor that is fed to the next optical layer.
        saved_interlayer_states.append(x_current)

    return x_current, layer_inputs, layer_phases, layer_outputs, layer_inputs_pre


@torch.no_grad()
def _evaluate_vlm(
    val_loader,
    *,
    device: torch.device,
    phase_system: PhaseSystem,
    bridge: OpticalBridge,
    mixers: nn.ModuleList,
    encoder_readout: EncoderReadout,
    caption_decoder: CaptionTransformerDecoder,
    input_adapter: nn.Module | None,
    mixer_cfg: Dict[str, object],
    channel_num: int,
    data_method: str,
    input_hw: Tuple[int, int] | None,
    caption_loss_type: str,
    ignore_index: int,
    label_smoothing: float,
    tokenizer: object,
    bos_idx: int,
    eos_idx: Optional[int],
    pad_idx: int,
    max_gen_len: int,
    generation_method: str,
    stop_on_eos: bool,
    beam_size: int,
    references_by_image: Optional[Dict[str, List[str]]] = None,
    input_pe_cfg: Optional[Dict[str, object]] = None,
    input_preproc_cfg: Optional[Dict[str, object]] = None,
    encoder_input_skip_cfg: Optional[Dict[str, object]] = None,
) -> Tuple[float, Dict[str, float]]:
    total_loss_sum = 0.0
    total_tokens = 0
    bleu_predictions: List[str] = []
    bleu_references: List[List[str]] = []
    seen_eval_images: set[str] = set()

    phase_system.eval()
    mixers.eval()
    encoder_readout.eval()
    caption_decoder.eval()
    if input_adapter is not None:
        input_adapter.eval()

    for batch in val_loader:
        if not isinstance(batch, dict):
            raise ValueError("VLM loader must return dict batches.")
        images = batch["image"].to(device=device, dtype=torch.float32)
        caption_ids = batch["caption_ids"].to(device=device, dtype=torch.long)
        attention_mask = batch["attention_mask"].to(device=device, dtype=torch.long)
        _validate_vlm_batch_tensors(images, caption_ids, attention_mask)
        if caption_ids.size(1) < 2:
            continue

        memory = _encode_optical_memory_batch(
            images,
            channel_num=channel_num,
            data_method=data_method,
            input_adapter=input_adapter,
            input_hw=input_hw,
            phase_system=phase_system,
            bridge=bridge,
            mixers=mixers,
            mixer_cfg=mixer_cfg,
            encoder_readout=encoder_readout,
            track_grads=False,
            fine_tune_state=None,
            input_pe_cfg=input_pe_cfg,
            input_preproc_cfg=input_preproc_cfg,
            encoder_input_skip_cfg=encoder_input_skip_cfg,
        )  # (B, C, E)
        if isinstance(references_by_image, dict) and references_by_image:
            image_filenames = batch.get("image_filename", None)
            if isinstance(image_filenames, list):
                unseen_indices: List[int] = []
                unseen_names: List[str] = []
                for sample_idx, image_name in enumerate(image_filenames):
                    key = str(image_name)
                    if key in seen_eval_images:
                        continue
                    refs = references_by_image.get(key, None)
                    if not refs:
                        continue
                    seen_eval_images.add(key)
                    unseen_indices.append(int(sample_idx))
                    unseen_names.append(key)
                if unseen_indices:
                    idx_tensor = torch.as_tensor(unseen_indices, device=memory.device, dtype=torch.long)
                    memory_bleu = memory.index_select(0, idx_tensor)
                    token_ids = _generate_captions_from_memory(
                        memory_bleu,
                        caption_decoder,
                        bos_idx=int(bos_idx),
                        eos_idx=eos_idx,
                        pad_idx=int(pad_idx),
                        max_gen_len=int(max_gen_len),
                        generation_method=str(generation_method),
                        stop_on_eos=bool(stop_on_eos),
                        beam_size=int(beam_size),
                    )
                    pred_texts = [
                        _decode_token_ids(tokenizer, token_ids[i].tolist(), skip_special_tokens=True)
                        for i in range(int(token_ids.size(0)))
                    ]
                    for key, pred_text in zip(unseen_names, pred_texts):
                        refs = references_by_image.get(key, [])
                        if refs:
                            bleu_predictions.append(str(pred_text))
                            bleu_references.append([str(ref) for ref in refs])
        decoder_input = caption_ids[:, :-1]
        targets = caption_ids[:, 1:]
        tgt_padding_mask = attention_mask[:, :-1].eq(0)
        logits = caption_decoder(
            decoder_input,
            memory,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=None,
        )
        loss_sum = _compute_caption_loss(
            logits,
            targets,
            caption_loss_type=caption_loss_type,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction="sum",
        )
        valid = targets.ne(int(ignore_index))
        valid_tokens = int(valid.sum().item())
        if valid_tokens <= 0:
            continue
        total_loss_sum += float(loss_sum.item())
        total_tokens += valid_tokens

    bleu_scores = _compute_bleu_1_to_4(bleu_predictions, bleu_references) if bleu_predictions else {
        "bleu1": float("nan"),
        "bleu2": float("nan"),
        "bleu3": float("nan"),
        "bleu4": float("nan"),
    }
    if total_tokens <= 0:
        return float("nan"), bleu_scores
    avg_loss = total_loss_sum / float(total_tokens)
    return avg_loss, bleu_scores


def _to_single_caption_sample(sample: object) -> Optional[Dict[str, object]]:
    if not isinstance(sample, dict):
        return None
    image = sample.get("image")
    caption = sample.get("caption")
    image_filename = sample.get("image_filename", "unknown")
    if not isinstance(image, torch.Tensor) or image.dim() != 3:
        return None
    if not isinstance(caption, str):
        caption_ids = sample.get("caption_ids", None)
        if isinstance(caption_ids, torch.Tensor):
            caption = " ".join(str(int(x)) for x in caption_ids.tolist())
        else:
            caption = str(caption)
    return {"image": image, "caption": caption, "image_filename": str(image_filename)}


@torch.no_grad()
def _preview_caption_examples(
    train_dataset,
    eval_dataset,
    *,
    device: torch.device,
    tokenizer: object,
    phase_system: PhaseSystem,
    bridge: OpticalBridge,
    mixers: nn.ModuleList,
    encoder_readout: EncoderReadout,
    caption_decoder: CaptionTransformerDecoder,
    input_adapter: nn.Module | None,
    mixer_cfg: Dict[str, object],
    channel_num: int,
    data_method: str,
    input_hw: Tuple[int, int] | None,
    bos_idx: int,
    eos_idx: Optional[int],
    pad_idx: int,
    max_gen_len: int,
    generation_method: str,
    stop_on_eos: bool,
    beam_size: int,
    sample_viz_state: Optional[Dict[str, object]] = None,
    plot_examples: bool = True,
    input_pe_cfg: Optional[Dict[str, object]] = None,
    input_preproc_cfg: Optional[Dict[str, object]] = None,
    encoder_input_skip_cfg: Optional[Dict[str, object]] = None,
) -> None:
    if (not bool(plot_examples)) or sample_viz_state is None:
        return
    if train_dataset is None or eval_dataset is None:
        return
    try:
        n_train = len(train_dataset)
        n_eval = len(eval_dataset)
    except Exception:
        return
    if n_train <= 0 or n_eval <= 0:
        return

    train_idx = random.randrange(n_train)
    eval_idx = random.randrange(n_eval)
    train_item = _to_single_caption_sample(train_dataset[train_idx])
    eval_item = _to_single_caption_sample(eval_dataset[eval_idx])
    if train_item is None or eval_item is None:
        return

    model_images_cpu = torch.stack([train_item["image"], eval_item["image"]], dim=0).detach().cpu()
    preview_images_cpu = torch.stack(
        [
            _extract_preview_image_from_dataset(
                train_dataset,
                train_idx,
                fallback=train_item["image"],
            ),
            _extract_preview_image_from_dataset(
                eval_dataset,
                eval_idx,
                fallback=eval_item["image"],
            ),
        ],
        dim=0,
    )
    images = model_images_cpu.to(device=device, dtype=torch.float32)
    pred_texts = generate_captions(
        images,
        device=device,
        tokenizer=tokenizer,
        channel_num=channel_num,
        data_method=data_method,
        input_adapter=input_adapter,
        input_hw=input_hw,
        phase_system=phase_system,
        bridge=bridge,
        mixers=mixers,
        mixer_cfg=mixer_cfg,
        encoder_readout=encoder_readout,
        caption_decoder=caption_decoder,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        pad_idx=pad_idx,
        max_gen_len=max_gen_len,
        generation_method=generation_method,
        stop_on_eos=stop_on_eos,
        beam_size=beam_size,
        input_pe_cfg=input_pe_cfg,
        input_preproc_cfg=input_preproc_cfg,
        encoder_input_skip_cfg=encoder_input_skip_cfg,
    )
    _update_caption_examples_plot(
        sample_viz_state,
        preview_images_cpu,
        gt_texts=[str(train_item["caption"]), str(eval_item["caption"])],
        pred_texts=pred_texts,
        file_names=[str(train_item["image_filename"]), str(eval_item["image_filename"])],
        source_labels=["Train", "Eval"],
    )


@torch.no_grad()
def generate_captions(
    images: torch.Tensor,
    *,
    device: torch.device,
    tokenizer: object,
    phase_system: PhaseSystem,
    bridge: OpticalBridge,
    mixers: nn.ModuleList,
    encoder_readout: EncoderReadout,
    caption_decoder: CaptionTransformerDecoder,
    input_adapter: nn.Module | None,
    mixer_cfg: Dict[str, object],
    channel_num: int,
    data_method: str,
    input_hw: Tuple[int, int] | None,
    bos_idx: int,
    eos_idx: Optional[int],
    pad_idx: int,
    max_gen_len: int,
    generation_method: str = "greedy",
    stop_on_eos: bool = True,
    beam_size: int = 3,
    input_pe_cfg: Optional[Dict[str, object]] = None,
    input_preproc_cfg: Optional[Dict[str, object]] = None,
    encoder_input_skip_cfg: Optional[Dict[str, object]] = None,
) -> List[str]:
    """
    Inference path (no teacher forcing):
    image -> ONN -> encoder_readout memory -> autoregressive decoder generation.
    """
    images = images.to(device=device, dtype=torch.float32)
    phase_system.eval()
    mixers.eval()
    encoder_readout.eval()
    caption_decoder.eval()
    if input_adapter is not None:
        input_adapter.eval()
    memory = _encode_optical_memory_batch(
        images,
        channel_num=channel_num,
        data_method=data_method,
        input_adapter=input_adapter,
        input_hw=input_hw,
        phase_system=phase_system,
        bridge=bridge,
        mixers=mixers,
        mixer_cfg=mixer_cfg,
        encoder_readout=encoder_readout,
        track_grads=False,
        fine_tune_state=None,
        input_pe_cfg=input_pe_cfg,
        input_preproc_cfg=input_preproc_cfg,
        encoder_input_skip_cfg=encoder_input_skip_cfg,
    )
    token_ids = _generate_captions_from_memory(
        memory,
        caption_decoder,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        pad_idx=pad_idx,
        max_gen_len=max_gen_len,
        generation_method=generation_method,
        stop_on_eos=stop_on_eos,
        beam_size=int(beam_size),
    )
    decoded = [
        _decode_token_ids(tokenizer, token_ids[i].tolist(), skip_special_tokens=True)
        for i in range(int(token_ids.size(0)))
    ]
    return decoded


def train(
    layer: Optional[nn.Module] = None,
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, object]] = None,
) -> None:
    config_path = str(DEFAULT_VLM_CONFIG_PATH if config_path is None else config_path)
    cfg, _ = load_training_config(config_path, overrides)
    seed = resolve_onn_seed(cfg, default=1337)
    print(f"Using seed: {seed}")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    channel_num = int(cfg.get("channel_num", 16))
    data_cfg = cfg.get("data", {})
    tokenizer_cfg = cfg.get("tokenizer", {})
    hw_cfg = data_cfg.get("hw", data_cfg.get("image_size", (112, 112)))
    input_hw = (int(hw_cfg[0]), int(hw_cfg[1]))
    data_method = str(data_cfg.get("method", "upsampler")).strip().lower()
    if data_method not in {"patchify", "upsampler"}:
        raise ValueError("data.method must be 'patchify' or 'upsampler'")
    if data_method != "upsampler":
        print("[warn] VLM path is primarily designed for data.method='upsampler'.")
    if int(data_cfg.get("num_channels", channel_num)) != channel_num:
        data_cfg["num_channels"] = channel_num
    input_preproc_raw = cfg.get("input_preprocess", {})
    input_preproc_cfg = {
        "normalize_before_patchify": bool(input_preproc_raw.get("normalize_before_patchify", False)),
        "mean": input_preproc_raw.get("mean", [0.5, 0.5, 0.5]),
        "std": input_preproc_raw.get("std", [0.5, 0.5, 0.5]),
        "std_eps": float(input_preproc_raw.get("std_eps", 1e-6)),
    }
    if input_preproc_cfg["normalize_before_patchify"]:
        print(
            "[info] input_preprocess.normalize_before_patchify enabled: "
            f"mean={input_preproc_cfg['mean']} std={input_preproc_cfg['std']}"
        )
    input_pe_raw = cfg.get("input_positional_encoding", {})
    input_pe_cfg = {
        "enabled": bool(input_pe_raw.get("enabled", False)),
        "mode": str(input_pe_raw.get("mode", "sin2d")),
        "num_bands": int(input_pe_raw.get("num_bands", 4)),
        "scale": float(input_pe_raw.get("scale", 0.05)),
        "injection": str(input_pe_raw.get("injection", "add")),
        "zero_mean": bool(input_pe_raw.get("zero_mean", True)),
        "clamp_01": bool(input_pe_raw.get("clamp_01", False)),
    }
    if input_pe_cfg["enabled"]:
        print(
            "[info] input_positional_encoding enabled: "
            f"mode={input_pe_cfg['mode']} bands={input_pe_cfg['num_bands']} "
            f"scale={input_pe_cfg['scale']:.4f} injection={input_pe_cfg['injection']}"
        )

    vlm_loaders = _prepare_vlm_data_loaders(data_cfg, tokenizer_cfg)
    train_loader = vlm_loaders["train_loader"]
    val_loader = vlm_loaders["val_loader"]
    tokenizer = vlm_loaders["tokenizer"]
    tokenizer_info = vlm_loaders.get("tokenizer_info", {})
    split_info = vlm_loaders.get("split_info", {})
    loaded_dataset_name = str(vlm_loaders.get("dataset_name", data_cfg.get("dataset_name", "unknown")))
    loaded_paths = vlm_loaders.get("paths", {})
    train_references_by_image = _build_references_by_image(getattr(train_loader, "dataset", None))
    val_references_by_image = _build_references_by_image(getattr(val_loader, "dataset", None))
    print(
        "[vlm] dataset loaded: "
        f"name={loaded_dataset_name} "
        f"images_dir={loaded_paths.get('images_dir', 'unknown')} "
        f"captions_file={loaded_paths.get('captions_file', 'unknown')}"
    )
    print(f"[vlm] split_info: {split_info}")
    try:
        train_preview_item = _to_single_caption_sample(train_loader.dataset[0])
    except Exception:
        train_preview_item = None
    if train_preview_item is not None:
        preview_caption = str(train_preview_item.get("caption", "")).strip()
        print(f"[vlm] train sample preview caption: {preview_caption}")

    sur_cfg = cfg.get("surrogate", {})
    surrogate, surrogate_ckpt_path, enc_canvas_hw = _load_surrogate(sur_cfg, device, channel_num)

    onn_cfg = cfg.get("onn", {})
    in_silico_mode = bool(onn_cfg.get("in_silico_mode", False))
    fine_cfg = cfg.get("fine_tune_surrogate", {})
    requested_fine_tune = bool(fine_cfg.get("enable", False))
    fine_enable = requested_fine_tune and (not in_silico_mode)
    if requested_fine_tune and in_silico_mode:
        print("[info] In-silico mode disables surrogate fine-tuning.")
    optical_layer = layer
    if optical_layer is None and (not in_silico_mode):
        raise ValueError("train() requires a physical optical layer instance unless in_silico_mode=True.")
    optical_sys = None if in_silico_mode else make_optical_sys(optical_layer)

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

    encoder_readout_raw = cfg.get("encoder_readout", {})
    if not bool(encoder_readout_raw.get("enabled", True)):
        raise ValueError("encoder_readout.enabled must be true for VLM mode.")
    encoder_readout_type = str(encoder_readout_raw.get("type", "avg_pool_flatten_proj")).strip().lower()
    if encoder_readout_type not in {"avg_pool_flatten_proj", "avg_pool", "conv_patch_tokens"}:
        raise ValueError(
            "encoder_readout.type must be one of "
            "'avg_pool_flatten_proj' or 'conv_patch_tokens'."
        )
    encoder_readout_cfg = {
        "type": encoder_readout_type,
        "emb_size": int(encoder_readout_raw.get("emb_size", 256)),
        "use_token_positional_embedding": bool(
            encoder_readout_raw.get("use_token_positional_embedding", False)
        ),
        "dropout": float(encoder_readout_raw.get("dropout", 0.0)),
    }
    encoder_input_skip_raw = encoder_readout_raw.get("input_skip", {})
    if isinstance(encoder_input_skip_raw, dict):
        encoder_input_skip_cfg = {
            "enabled": bool(encoder_input_skip_raw.get("enabled", False)),
            "mode": str(encoder_input_skip_raw.get("mode", "concat")),
        }
    else:
        encoder_input_skip_cfg = {
            "enabled": bool(encoder_input_skip_raw),
            "mode": "concat",
        }
    if encoder_input_skip_cfg["enabled"] and encoder_readout_type != "conv_patch_tokens":
        raise ValueError("encoder_readout.input_skip is supported only when type='conv_patch_tokens'.")
    if encoder_readout_type in {"avg_pool_flatten_proj", "avg_pool"}:
        encoder_readout_cfg.update(
            {
                "pool_h": int(encoder_readout_raw.get("pool_h", 5)),
                "pool_w": int(encoder_readout_raw.get("pool_w", 5)),
                "flatten_mode": str(encoder_readout_raw.get("flatten_mode", "channel_first")),
                "use_channel_embedding": bool(encoder_readout_raw.get("use_channel_embedding", True)),
            }
        )
    elif encoder_readout_type == "conv_patch_tokens":
        base_in_channels = int(encoder_readout_raw.get("in_channels", channel_num))
        effective_in_channels = (
            base_in_channels + channel_num if encoder_input_skip_cfg["enabled"] else base_in_channels
        )
        encoder_readout_cfg.update(
            {
                "in_channels": int(effective_in_channels),
                "out_channels": int(encoder_readout_raw.get("out_channels", 64)),
                "kernel_size": int(encoder_readout_raw.get("kernel_size", 7)),
                "stride": int(encoder_readout_raw.get("stride", 7)),
                "padding": int(encoder_readout_raw.get("padding", 0)),
                "output_h": (
                    None
                    if encoder_readout_raw.get("output_h", None) is None
                    else int(encoder_readout_raw.get("output_h"))
                ),
                "output_w": (
                    None
                    if encoder_readout_raw.get("output_w", None) is None
                    else int(encoder_readout_raw.get("output_w"))
                ),
                "use_channel_embedding": False,
            }
        )
        if encoder_input_skip_cfg["enabled"]:
            print(
                "[info] encoder_readout.input_skip enabled: "
                f"conv tokenizer in_channels={base_in_channels}+{channel_num}={effective_in_channels}."
            )
    encoder_readout = EncoderReadout(
        channel_num=channel_num,
        emb_size=encoder_readout_cfg["emb_size"],
        readout_type=encoder_readout_cfg["type"],
        pool_h=int(encoder_readout_cfg.get("pool_h", 5)),
        pool_w=int(encoder_readout_cfg.get("pool_w", 5)),
        flatten_mode=str(encoder_readout_cfg.get("flatten_mode", "channel_first")),
        conv_in_channels=int(encoder_readout_cfg.get("in_channels", channel_num)),
        conv_out_channels=int(encoder_readout_cfg.get("out_channels", 64)),
        conv_kernel_size=int(encoder_readout_cfg.get("kernel_size", 7)),
        conv_stride=int(encoder_readout_cfg.get("stride", 7)),
        conv_padding=int(encoder_readout_cfg.get("padding", 0)),
        conv_output_h=encoder_readout_cfg.get("output_h", None),
        conv_output_w=encoder_readout_cfg.get("output_w", None),
        use_channel_embedding=encoder_readout_cfg["use_channel_embedding"],
        use_token_positional_embedding=encoder_readout_cfg["use_token_positional_embedding"],
        dropout=encoder_readout_cfg["dropout"],
    ).to(device)

    decoder_raw = cfg.get("Decoder", cfg.get("decoder", {}))
    decoder_emb_size = int(decoder_raw.get("emb_size", encoder_readout_cfg["emb_size"]))
    if decoder_emb_size != encoder_readout_cfg["emb_size"]:
        raise ValueError(
            "Decoder emb_size must match encoder_readout.emb_size for direct cross-attention."
        )
    tokenizer_vocab_size = int(tokenizer_info.get("vocab_size", len(tokenizer)))
    decoder_vocab_cfg = decoder_raw.get("vocab_size", "tokenizer")
    if isinstance(decoder_vocab_cfg, str) and decoder_vocab_cfg.strip().lower() in {"auto", "tokenizer"}:
        decoder_vocab_size = tokenizer_vocab_size
    else:
        decoder_vocab_size = int(decoder_vocab_cfg)
        if decoder_vocab_size < tokenizer_vocab_size:
            raise ValueError(
                f"Decoder vocab_size ({decoder_vocab_size}) must be >= tokenizer vocab ({tokenizer_vocab_size})."
            )
    decoder_cfg = {
        "vocab_size": decoder_vocab_size,
        "emb_size": decoder_emb_size,
        "num_layers": int(decoder_raw.get("num_layers", 4)),
        "num_heads": int(decoder_raw.get("num_heads", 8)),
        "ff_dim": int(decoder_raw.get("ff_dim", 1024)),
        "dropout": float(decoder_raw.get("dropout", 0.1)),
        "attn_dropout": float(decoder_raw.get("attn_dropout", 0.1)),
        "max_seq_len": int(decoder_raw.get("max_seq_len", data_cfg.get("max_caption_length", 32))),
        "tie_weights": bool(decoder_raw.get("tie_weights", False)),
        "use_learned_positional_embedding": bool(
            decoder_raw.get("use_learned_positional_embedding", True)
        ),
        "layer_norm_eps": float(decoder_raw.get("layer_norm_eps", 1e-5)),
        "activation": str(decoder_raw.get("activation", "gelu")),
    }
    decoder_max_seq_len = int(decoder_cfg["max_seq_len"])
    if decoder_max_seq_len < 2:
        raise ValueError("Decoder.max_seq_len must be >= 2.")
    data_max_caption_len = int(data_cfg.get("max_caption_length", decoder_max_seq_len))
    tokenizer_max_caption_len = int(tokenizer_cfg.get("max_caption_length", data_max_caption_len))
    if data_max_caption_len < 2:
        raise ValueError("data.max_caption_length must be >= 2.")
    if tokenizer_max_caption_len < 2:
        raise ValueError("tokenizer.max_caption_length must be >= 2.")
    configured_caption_len = max(data_max_caption_len, tokenizer_max_caption_len)
    if configured_caption_len > decoder_max_seq_len:
        raise ValueError(
            f"Configured caption length ({configured_caption_len}) exceeds Decoder.max_seq_len ({decoder_max_seq_len})."
        )
    caption_decoder = CaptionTransformerDecoder(**decoder_cfg).to(device)

    cap_cfg = cfg.get("captioning", {})
    caption_loss_type = str(cap_cfg.get("caption_loss_type", "cross_entropy"))
    label_smoothing = float(cap_cfg.get("label_smoothing", 0.0))
    teacher_forcing = bool(cap_cfg.get("teacher_forcing", True))
    if not teacher_forcing:
        raise ValueError("captioning.teacher_forcing must be true for this training path.")
    generation_method = str(cap_cfg.get("generation_method", "greedy"))
    beam_size = int(cap_cfg.get("beam_size", 3))
    max_gen_len = int(cap_cfg.get("max_gen_len", decoder_max_seq_len))
    stop_on_eos = bool(cap_cfg.get("eos_handling", True))
    use_bos_handling = bool(cap_cfg.get("bos_handling", True))
    if beam_size < 1:
        raise ValueError("captioning.beam_size must be >= 1.")
    if max_gen_len < 2:
        raise ValueError("captioning.max_gen_len must be >= 2.")
    if max_gen_len > decoder_max_seq_len:
        raise ValueError(
            f"captioning.max_gen_len ({max_gen_len}) exceeds Decoder.max_seq_len ({decoder_max_seq_len})."
        )

    bos_idx_raw = cap_cfg.get("bos_idx", "auto")
    eos_idx_raw = cap_cfg.get("eos_idx", "auto")
    pad_idx_raw = cap_cfg.get("pad_idx", "auto")
    bos_idx = tokenizer_info.get("bos_token_id", None) if str(bos_idx_raw).lower() == "auto" else bos_idx_raw
    eos_idx = tokenizer_info.get("eos_token_id", None) if str(eos_idx_raw).lower() == "auto" else eos_idx_raw
    pad_idx = tokenizer_info.get("pad_token_id", 0) if str(pad_idx_raw).lower() == "auto" else pad_idx_raw
    if bos_idx is None:
        raise ValueError("BOS token id is required (tokenizer_info.bos_token_id or captioning.bos_idx).")
    if not use_bos_handling:
        raise ValueError("captioning.bos_handling=false is not supported in this decoder path.")
    if pad_idx is None:
        pad_idx = 0
    bos_idx = int(bos_idx)
    eos_idx = None if eos_idx is None else int(eos_idx)
    pad_idx = int(pad_idx)
    if bos_idx < 0 or bos_idx >= decoder_vocab_size:
        raise ValueError(f"captioning.bos_idx/tokenizer bos token id ({bos_idx}) must be in [0, {decoder_vocab_size}).")
    if pad_idx < 0 or pad_idx >= decoder_vocab_size:
        raise ValueError(f"captioning.pad_idx/tokenizer pad token id ({pad_idx}) must be in [0, {decoder_vocab_size}).")
    if eos_idx is not None and (eos_idx < 0 or eos_idx >= decoder_vocab_size):
        raise ValueError(
            f"captioning.eos_idx/tokenizer eos token id ({eos_idx}) must be in [0, {decoder_vocab_size})."
        )
    ignore_index_cfg = cap_cfg.get("ignore_index", "pad_idx")
    if isinstance(ignore_index_cfg, str) and ignore_index_cfg.strip().lower() in {"pad", "pad_idx", "auto"}:
        ignore_index = int(pad_idx)
    else:
        ignore_index = int(ignore_index_cfg)
    if ignore_index >= decoder_vocab_size and ignore_index >= 0:
        raise ValueError(
            f"captioning.ignore_index ({ignore_index}) must be < decoder vocab size ({decoder_vocab_size}) "
            "or a negative sentinel."
        )

    val_gen_cfg = cap_cfg.get("validation_generation", {})
    val_gen_every = int(val_gen_cfg.get("every", int(onn_cfg.get("eval_every", 500))))
    eval_beam_size = int(val_gen_cfg.get("beam_size", cap_cfg.get("eval_beam_size", 5)))
    if eval_beam_size < 1:
        raise ValueError("captioning.validation_generation.beam_size must be >= 1.")
    viz_cfg = cap_cfg.get("visualization", {})
    metrics_plot_every = int(
        viz_cfg.get("metrics_every", onn_cfg.get("metrics_viz_every", onn_cfg.get("viz_every", 0)))
    )
    if metrics_plot_every < 0:
        metrics_plot_every = 0
    preview_plot_examples = bool(viz_cfg.get("plot_examples", True))
    sample_viz_every = max(0, int(viz_cfg.get("samples_every", onn_cfg.get("viz_every", val_gen_every))))
    phase_viz_every = max(0, int(onn_cfg.get("phase_viz_every", onn_cfg.get("viz_every", 0))))
    phase_viz_channels_cfg = onn_cfg.get("phase_viz_channels", [3, 5])
    if isinstance(phase_viz_channels_cfg, (list, tuple)):
        phase_viz_channels = [int(x) for x in phase_viz_channels_cfg if int(x) > 0]
    else:
        phase_viz_channels = [int(phase_viz_channels_cfg)]
    if not phase_viz_channels:
        phase_viz_channels = [3, 5]

    max_epochs = int(onn_cfg.get("max_epochs", 10))
    eval_every = int(onn_cfg.get("eval_every", 200))
    ckpt_auto_mode, ckpt_every = _parse_ckpt_every(onn_cfg.get("ckpt_every", 1000))

    optical_default_dir = PROJECT_ROOT / "pre_trained_model_save" / "Optical_neural_net"
    run_dir = Path(onn_cfg.get("save_dir", str(optical_default_dir)))
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_layer_path = checkpoint_dir / f"_mul_mix_{n_layers}_VLM_caption.pt"
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
    input_adapter = _build_input_adapter(
        data_method,
        channel_num,
        input_conv_cfg,
        patchify_target_hw=(int(phase_mask_shape[1]), int(phase_mask_shape[2])),
        patchify_overlap_percent=float(data_cfg.get("overlap_percent", 0.0)),
    )
    if input_adapter is not None:
        input_adapter = input_adapter.to(device)
    if data_method == "patchify":
        print(
            "[info] data.method='patchify': input_conv is disabled; "
            f"using patch projector adapter (overlap_percent={float(data_cfg.get('overlap_percent', 0.0)):.2f})."
        )
    mixers = _build_mixers(n_layers, channel_num, mixer_cfg).to(device)

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
    optim_module.encoder_readout = encoder_readout
    optim_module.caption_decoder = caption_decoder
    if input_adapter is not None:
        optim_module.input_adapter = input_adapter
    if bridge.gain_raw is not None:
        optim_module.bridge_gain_raw = bridge.gain_raw

    optimizer = _build_optimizer(optim_module, opt_name, lr, weight_decay)
    _ensure_initial_lrs(optimizer)
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    reset_live_plot()
    reset_fine_tune_viz()
    mode_caption = "ONN encoder and transformer decoder are optimized jointly with teacher forcing."
    if requested_fine_tune and in_silico_mode:
        mode_caption = (
            "ONN encoder and transformer decoder are optimized jointly with teacher forcing. "
            "Surrogate fine-tuning is enabled in YAML but inactive in in-silico mode."
        )
    viz_outputs = create_training_viz_layout(
        n_layers=n_layers,
        in_silico_mode=in_silico_mode,
        fine_tuning_enabled=requested_fine_tune,
        n_optical_params=n_optical_params,
        training_mode_title="End-to-end VLM captioning",
        training_mode_caption=mode_caption,
    )
    metrics_plot_state = _init_vlm_metrics_plot_state(
        log_scale=False,
        output_widget=viz_outputs.get("metrics"),
    )
    print("[viz] VLM metrics panel enabled: loss plot (left) + BLEU-1..4 table (right).")
    sample_viz_state = _init_caption_examples_viz_state(output_widget=viz_outputs.get("samples"))
    phase_viz_state = init_phase_viz_state(output_widget=viz_outputs.get("phase"))

    history = {
        "train_caption_loss": [],
        "eval_caption_loss": [],
        "train_bleu1": [],
        "train_bleu2": [],
        "train_bleu3": [],
        "train_bleu4": [],
        "train_bleu_iters": [],
        "eval_bleu1": [],
        "eval_bleu2": [],
        "eval_bleu3": [],
        "eval_bleu4": [],
        "eval_bleu_iters": [],
        "iters": [],
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
        channel_readouts=None,
        input_adapter=input_adapter,
        encoder_readout=encoder_readout,
        caption_decoder=caption_decoder,
        extra_tensors=extra_ckpt_tensors,
        fusion_weights_raw=None,
    )
    if not checkpoint_layer_path.exists():
        _save_checkpoint(
            checkpoint_layer_path,
            resume_epoch,
            it_counter,
            phase_system,
            mixers,
            None,
            input_adapter,
            encoder_readout,
            caption_decoder,
            history,
            cfg,
            extra_state=checkpoint_extra_state,
            fusion_weights_raw=None,
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
    if plt is None:
        print("[warn] matplotlib not available; skipping training/preview/phase plots.")
    elif metrics_plot_every > 0:
        _update_vlm_metrics_plot(
            metrics_plot_state,
            iteration=it_counter,
            update_every=max(1, metrics_plot_every),
            force=True,
        )

    optimizer.zero_grad(set_to_none=True)
    latest_eval_loss: Optional[float] = None
    latest_train_bleu: Dict[str, float] = _nan_bleu_scores()

    try:
        for epoch in range(resume_epoch, max_epochs):
            if progress_bar is not None:
                progress_bar.set_description(f"Epoch {epoch + 1}/{max_epochs}")
            for batch_idx, batch in enumerate(train_loader, start=1):
                if not isinstance(batch, dict):
                    raise ValueError("VLM loader must return dict batches.")
                it_counter += 1
                phase_system.train()
                mixers.train()
                encoder_readout.train()
                caption_decoder.train()
                if input_adapter is not None:
                    input_adapter.train()
                surrogate.eval()

                images = batch["image"].to(device=device, dtype=torch.float32)  # (B,C_in,H,W)
                caption_ids = batch["caption_ids"].to(device=device, dtype=torch.long)  # (B,T)
                attention_mask = batch["attention_mask"].to(device=device, dtype=torch.long)  # (B,T)
                _validate_vlm_batch_tensors(images, caption_ids, attention_mask)
                if caption_ids.size(1) < 2:
                    continue
                decoder_input = caption_ids[:, :-1]
                targets = caption_ids[:, 1:]
                tgt_padding_mask = attention_mask[:, :-1].eq(0)

                with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                    memory = _encode_optical_memory_batch(
                        images,
                        channel_num=channel_num,
                        data_method=data_method,
                        input_adapter=input_adapter,
                        input_hw=input_hw,
                        phase_system=phase_system,
                        bridge=bridge,
                        mixers=mixers,
                        mixer_cfg=mixer_cfg,
                        encoder_readout=encoder_readout,
                        track_grads=True,
                        fine_tune_state=fine_state,
                        input_pe_cfg=input_pe_cfg,
                        input_preproc_cfg=input_preproc_cfg,
                        encoder_input_skip_cfg=encoder_input_skip_cfg,
                    )  # (B,16,E)
                    logits = caption_decoder(
                        decoder_input,
                        memory,
                        tgt_key_padding_mask=tgt_padding_mask,
                        memory_key_padding_mask=None,
                    )  # (B,T-1,V)
                    loss_ce = _compute_caption_loss(
                        logits,
                        targets,
                        caption_loss_type=caption_loss_type,
                        ignore_index=ignore_index,
                        label_smoothing=label_smoothing,
                        reduction="mean",
                    )

                loss = loss_ce / float(accum_steps)
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
                    if scheduler is not None and scale_after >= scale_before:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                loss_val = float(loss_ce.detach().cpu())
                history["train_caption_loss"].append(loss_val)
                history["iters"].append(it_counter)
                if plt is not None and metrics_plot_every > 0:
                    train_bleu_for_update: Optional[Dict[str, float]] = None
                    last_update = int(metrics_plot_state.get("last_update", -1))
                    should_refresh_bleu = (last_update == -1) or (
                        (it_counter - last_update) >= max(1, metrics_plot_every)
                    )
                    if should_refresh_bleu:
                        train_bleu_raw = _compute_batch_bleu_from_memory(
                            memory.detach(),
                            batch.get("image_filename", None),
                            tokenizer=tokenizer,
                            caption_decoder=caption_decoder,
                            references_by_image=train_references_by_image,
                            bos_idx=int(bos_idx),
                            eos_idx=(None if eos_idx is None else int(eos_idx)),
                            pad_idx=int(pad_idx),
                            max_gen_len=int(max_gen_len),
                            generation_method=str(generation_method),
                            stop_on_eos=bool(stop_on_eos),
                            beam_size=int(beam_size),
                        )
                        history["train_bleu1"].append(float(train_bleu_raw.get("bleu1", float("nan"))))
                        history["train_bleu2"].append(float(train_bleu_raw.get("bleu2", float("nan"))))
                        history["train_bleu3"].append(float(train_bleu_raw.get("bleu3", float("nan"))))
                        history["train_bleu4"].append(float(train_bleu_raw.get("bleu4", float("nan"))))
                        history["train_bleu_iters"].append(it_counter)
                        train_bleu_for_update = {
                            "bleu1": float(train_bleu_raw.get("bleu1", float("nan"))),
                            "bleu2": float(train_bleu_raw.get("bleu2", float("nan"))),
                            "bleu3": float(train_bleu_raw.get("bleu3", float("nan"))),
                            "bleu4": float(train_bleu_raw.get("bleu4", float("nan"))),
                        }
                        latest_train_bleu = dict(train_bleu_for_update)
                    _update_vlm_metrics_plot(
                        metrics_plot_state,
                        it_counter,
                        max(1, metrics_plot_every),
                        train_loss=loss_val,
                        train_bleu_scores=train_bleu_for_update,
                    )

                if progress_bar is not None:
                    postfix = {"iter": it_counter, "train_loss": f"{loss_val:.4f}"}
                    if scheduler is not None:
                        lr_current = scheduler.get_last_lr()
                        if lr_current:
                            postfix["lr"] = f"{lr_current[0]:.2e}"
                    if latest_eval_loss is not None:
                        postfix["eval_loss"] = f"{latest_eval_loss:.4f}"
                    progress_bar.set_postfix(postfix)
                    progress_bar.update(1)

                if fine_state is not None:
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
                    eval_loss, eval_bleu = _evaluate_vlm(
                        val_loader,
                        device=device,
                        phase_system=phase_system,
                        bridge=bridge,
                        mixers=mixers,
                        encoder_readout=encoder_readout,
                        caption_decoder=caption_decoder,
                        input_adapter=input_adapter,
                        mixer_cfg=mixer_cfg,
                        channel_num=channel_num,
                        data_method=data_method,
                        input_hw=input_hw,
                        caption_loss_type=caption_loss_type,
                        ignore_index=ignore_index,
                        label_smoothing=label_smoothing,
                        tokenizer=tokenizer,
                        bos_idx=int(bos_idx),
                        eos_idx=(None if eos_idx is None else int(eos_idx)),
                        pad_idx=int(pad_idx),
                        max_gen_len=int(max_gen_len),
                        generation_method=str(generation_method),
                        stop_on_eos=bool(stop_on_eos),
                        beam_size=int(eval_beam_size),
                        references_by_image=val_references_by_image,
                        input_pe_cfg=input_pe_cfg,
                        input_preproc_cfg=input_preproc_cfg,
                        encoder_input_skip_cfg=encoder_input_skip_cfg,
                    )
                    history["eval_caption_loss"].append(eval_loss)
                    history["eval_bleu1"].append(float(eval_bleu.get("bleu1", float("nan"))))
                    history["eval_bleu2"].append(float(eval_bleu.get("bleu2", float("nan"))))
                    history["eval_bleu3"].append(float(eval_bleu.get("bleu3", float("nan"))))
                    history["eval_bleu4"].append(float(eval_bleu.get("bleu4", float("nan"))))
                    history["eval_bleu_iters"].append(it_counter)
                    history["eval_iters"].append(it_counter)
                    latest_eval_loss = eval_loss
                    if plt is not None and metrics_plot_every > 0:
                        _update_vlm_metrics_plot(
                            metrics_plot_state,
                            it_counter,
                            max(1, metrics_plot_every),
                            eval_loss=eval_loss,
                            train_bleu_scores=latest_train_bleu,
                            eval_bleu_scores=eval_bleu,
                            force=True,
                        )
                    print(
                        f"[eval] iter={it_counter} loss={eval_loss:.4f} "
                        f"BLEU-1/2/3/4={float(eval_bleu.get('bleu1', float('nan'))):.4f}/"
                        f"{float(eval_bleu.get('bleu2', float('nan'))):.4f}/"
                        f"{float(eval_bleu.get('bleu3', float('nan'))):.4f}/"
                        f"{float(eval_bleu.get('bleu4', float('nan'))):.4f}"
                    )
                    if eval_loss < best_eval:
                        best_eval = eval_loss
                        if ckpt_auto_mode:
                            _save_checkpoint(
                                checkpoint_layer_path,
                                epoch,
                                it_counter,
                                phase_system,
                                mixers,
                                None,
                                input_adapter,
                                encoder_readout,
                                caption_decoder,
                                history,
                                cfg,
                                extra_state=checkpoint_extra_state,
                                fusion_weights_raw=None,
                            )

                if (
                    preview_plot_examples
                    and sample_viz_every > 0
                    and (it_counter % sample_viz_every == 0)
                ):
                    _preview_caption_examples(
                        train_loader.dataset,
                        val_loader.dataset,
                        device=device,
                        tokenizer=tokenizer,
                        phase_system=phase_system,
                        bridge=bridge,
                        mixers=mixers,
                        encoder_readout=encoder_readout,
                        caption_decoder=caption_decoder,
                        input_adapter=input_adapter,
                        mixer_cfg=mixer_cfg,
                        channel_num=channel_num,
                        data_method=data_method,
                        input_hw=input_hw,
                        bos_idx=int(bos_idx),
                        eos_idx=(None if eos_idx is None else int(eos_idx)),
                        pad_idx=int(pad_idx),
                        max_gen_len=max_gen_len,
                        generation_method=generation_method,
                        stop_on_eos=stop_on_eos,
                        beam_size=int(eval_beam_size),
                        sample_viz_state=sample_viz_state,
                        plot_examples=True,
                        input_pe_cfg=input_pe_cfg,
                        input_preproc_cfg=input_preproc_cfg,
                        encoder_input_skip_cfg=encoder_input_skip_cfg,
                    )

                if (
                    plt is not None
                    and phase_viz_every > 0
                    and (it_counter % phase_viz_every == 0)
                ):
                    phase_rows = _collect_phase_trace_rows(
                        images.detach(),
                        channel_indices_one_based=phase_viz_channels,
                        channel_num=channel_num,
                        data_method=data_method,
                        input_adapter=input_adapter,
                        input_hw=input_hw,
                        phase_system=phase_system,
                        bridge=bridge,
                        mixers=mixers,
                        mixer_cfg=mixer_cfg,
                        encoder_readout=encoder_readout,
                        input_pe_cfg=input_pe_cfg,
                        input_preproc_cfg=input_preproc_cfg,
                        encoder_input_skip_cfg=encoder_input_skip_cfg,
                    )
                    _plot_vlm_phase_trace_panel(
                        phase_viz_state,
                        phase_rows,
                        iteration=it_counter,
                    )

                if (not ckpt_auto_mode) and ckpt_every and ckpt_every > 0 and it_counter % ckpt_every == 0:
                    _save_checkpoint(
                        checkpoint_layer_path,
                        epoch,
                        it_counter,
                        phase_system,
                        mixers,
                        None,
                        input_adapter,
                        encoder_readout,
                        caption_decoder,
                        history,
                        cfg,
                        extra_state=checkpoint_extra_state,
                        fusion_weights_raw=None,
                    )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    if not ckpt_auto_mode:
        _save_checkpoint(
            checkpoint_layer_path,
            max_epochs,
            it_counter,
            phase_system,
            mixers,
            None,
            input_adapter,
            encoder_readout,
            caption_decoder,
            history,
            cfg,
            extra_state=checkpoint_extra_state,
            fusion_weights_raw=None,
        )
    print(f"Training completed. Results saved to {run_dir}")


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ONN encoder + Transformer decoder for image captioning.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_VLM_CONFIG_PATH),
        help="Path to YAML config file (default: config_mix_VLM.yaml in this directory)",
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
