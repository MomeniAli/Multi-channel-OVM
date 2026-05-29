"""
Binary-decision ONN training script.

Each logical image is expanded into the optical batch dimension:
- channels 0..9 receive the same input image and act as class-specific detectors
- channels 10..15 are dummy channels kept only for phase/surrogate compatibility

There is no digital mixing between optical layers. The final active channels make
YES/NO decisions from two large output-plane regions. Class logits are
yes_score - no_score for class channels 0..9.

Optionally, the single logical input image can be encoded as a 2x2 tile of
0/90/180/270 degree rotations before it is repeated across class channels.
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
from tqdm.auto import tqdm

try:
    from .models import Surrogate_OpticalNet_Unet
    from .dataloader_patch import build_patch_loaders
    from .utils import encoding_x_phase_physical
    from .surrogate_model_training import make_optical_sys, reset_live_plot
    from .config_loader import load_training_config, resolve_onn_seed, write_config_snapshot
    from .phase_system import PhaseSystem, configure_phase_unit, phase_to_unit
    from .optical_bridge import OpticalBridge
    from .training_viz import (
        USE_NOTEBOOK_PROGRESS,
        NotebookProgressBar,
        ensure_dark_tqdm_theme,
        create_training_viz_layout,
        init_metrics_plot_state,
        update_metrics_plot,
        init_sample_viz_state,
        update_sample_viz,
        init_phase_viz_state,
        update_phase_viz,
        compute_tile_boxes,
    )
    from .fine_tuning import (
        accumulate_fine_tune_samples,
        init_fine_tune_state,
        maybe_fine_tune_surrogate,
        reset_fine_tune_viz,
    )
except ImportError:
    from models import Surrogate_OpticalNet_Unet
    from dataloader_patch import build_patch_loaders
    from utils import encoding_x_phase_physical
    from surrogate_model_training import make_optical_sys, reset_live_plot
    from config_loader import load_training_config, resolve_onn_seed, write_config_snapshot
    from phase_system import PhaseSystem, configure_phase_unit, phase_to_unit
    from optical_bridge import OpticalBridge
    from training_viz import (
        USE_NOTEBOOK_PROGRESS,
        NotebookProgressBar,
        ensure_dark_tqdm_theme,
        create_training_viz_layout,
        init_metrics_plot_state,
        update_metrics_plot,
        init_sample_viz_state,
        update_sample_viz,
        init_phase_viz_state,
        update_phase_viz,
        compute_tile_boxes,
    )
    from fine_tuning import (
        accumulate_fine_tune_samples,
        init_fine_tune_state,
        maybe_fine_tune_surrogate,
        reset_fine_tune_viz,
    )


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BINARY_CONFIG_PATH = PROJECT_ROOT / "config_binary_decision.yaml"
ECOC_CODEBOOK_TYPE = "fixed_10x16_v1"
FIXED_ECOC_CODEBOOK_v0 = torch.tensor(
    [
        [-1, -1, +1, +1, +1, +1, -1, -1, +1, +1, +1, -1, +1, -1, -1, -1],
        [+1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, +1, +1],
        [-1, -1, +1, +1, -1, -1, -1, +1, +1, -1, +1, +1, -1, +1, +1, -1],
        [+1, +1, -1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, -1, +1],
        [-1, +1, +1, -1, -1, +1, +1, +1, +1, -1, -1, +1, +1, -1, -1, -1],
        [+1, -1, -1, +1, +1, -1, -1, -1, -1, +1, +1, -1, -1, +1, +1, +1],
        [-1, +1, -1, +1, +1, -1, -1, -1, +1, -1, -1, +1, +1, +1, -1, +1],
        [+1, -1, +1, -1, -1, +1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1],
        [-1, +1, -1, -1, -1, +1, -1, -1, +1, +1, +1, +1, -1, -1, +1, +1],
        [+1, -1, +1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, +1, -1, -1],
    ],
    dtype=torch.float32,
)
FIXED_ECOC_CODEBOOK_v1 = torch.tensor(
    [
        [+1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, -1, +1, -1, +1, -1],
        [+1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, -1, -1, +1, +1, +1],
        [-1, -1, +1, -1, +1, -1, -1, +1, +1, +1, -1, +1, -1, +1, -1, +1],
        [-1, +1, +1, +1, +1, -1, -1, +1, -1, -1, -1, +1, +1, -1, +1, -1],
        [+1, +1, +1, -1, -1, -1, +1, -1, +1, -1, -1, +1, +1, -1, -1, +1],
        [+1, -1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, -1],
        [+1, +1, +1, -1, +1, +1, +1, -1, -1, +1, -1, -1, -1, -1, +1, -1],
        [-1, +1, -1, +1, -1, +1, -1, -1, -1, +1, +1, +1, -1, +1, -1, +1],
        [-1, -1, -1, -1, -1, +1, +1, +1, -1, -1, +1, +1, +1, -1, +1, +1],
        [-1, -1, -1, +1, +1, +1, +1, -1, +1, +1, -1, -1, +1, +1, -1, -1],
    ],
    dtype=torch.float32,
)
FIXED_ECOC_CODEBOOK = FIXED_ECOC_CODEBOOK_v1


def get_fixed_ecoc_codebook(
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    codebook = FIXED_ECOC_CODEBOOK.clone().detach().to(device=device, dtype=dtype)
    if codebook.shape != (10, 16):
        raise RuntimeError(f"Fixed ECOC codebook must have shape (10,16), got {tuple(codebook.shape)}")
    if not torch.all((codebook == 1) | (codebook == -1)):
        raise RuntimeError("Fixed ECOC codebook entries must all be +/-1")
    if not torch.all(codebook.sum(dim=1) == 0):
        raise RuntimeError("Fixed ECOC codebook rows must each contain eight +1 and eight -1 entries")
    if not torch.all(codebook.sum(dim=0) == 0):
        raise RuntimeError("Fixed ECOC codebook columns must each contain five +1 and five -1 entries")
    return codebook


def _assert_checkpoint_ecoc_codebook_matches(saved_codebook: torch.Tensor, device: torch.device) -> None:
    expected = get_fixed_ecoc_codebook(device=device, dtype=torch.float32)
    saved = saved_codebook.detach().to(device=device, dtype=torch.float32)
    if saved.shape != expected.shape or not torch.equal(saved, expected):
        raise RuntimeError(
            "Checkpoint ECOC codebook does not match hard-coded "
            f"{ECOC_CODEBOOK_TYPE}. Refusing to resume with a different codebook."
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _softplus_inverse(x: float, eps: float = 1e-6) -> float:
    x = max(float(x), eps)
    return float(math.log(math.expm1(x)))


def _positive_param(raw: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.softplus(raw) + eps


def _prepare_data_loaders(data_cfg: Dict[str, object]):
    dataset_name = str(data_cfg.get("dataset", "fashion_mnist"))
    batch_size = int(data_cfg.get("batch_size", 1))
    num_channels = int(data_cfg.get("num_channels", 16))
    hw_cfg = data_cfg.get("hw", (112, 112))
    hw = (int(hw_cfg[0]), int(hw_cfg[1]))
    use_rot4_tiling = bool(data_cfg.get("use_rot4_tiling", True))
    method = str(data_cfg.get("method", "upsampler")).strip().lower()
    overlap_percent = float(data_cfg.get("overlap_percent", 0.0))
    augment = bool(data_cfg.get("augment", False))
    data_root = str(data_cfg.get("data_root", "./data"))
    val_split = float(data_cfg.get("val_split", 0.03))
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
        val_split=val_split,
    )


def _load_surrogate(sur_cfg: Dict[str, object], device: torch.device, channel_num: int):
    enc_canvas_hw = tuple(sur_cfg.get("enc_canvas_hw", [112, 112]))
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
        state_dict = ckpt_obj["model_state"] if isinstance(ckpt_obj, dict) and "model_state" in ckpt_obj else ckpt_obj
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
    warmup_steps = max(int(warmup_steps), 0)
    total_steps = max(int(total_steps), warmup_steps + 1)
    base_lr = float(base_lr)
    min_lr = min(float(min_lr), base_lr)
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


def _parse_optional_hw(value, *, key: str) -> Tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"", "none", "null", "full", "full_half", "auto"}:
            return None
        raise ValueError(f"{key} must be null or a length-2 list like [H, W], got {value!r}")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{key} must be null or a length-2 list like [H, W]")
    h_val = int(value[0])
    w_val = int(value[1])
    if h_val <= 0 or w_val <= 0:
        raise ValueError(f"{key} entries must be positive, got {value!r}")
    return h_val, w_val


def _parse_nonnegative_hw(value, *, key: str, default: Tuple[int, int] = (0, 0)) -> Tuple[int, int]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{key} must be a length-2 list like [H, W]")
    h_val = int(value[0])
    w_val = int(value[1])
    if h_val < 0 or w_val < 0:
        raise ValueError(f"{key} entries must be non-negative, got {value!r}")
    return h_val, w_val


def _load_checkpoint(
    path: Path,
    device: torch.device,
    phase_system: PhaseSystem,
    history: Dict[str, List[float]],
    extra_tensors: Dict[str, torch.nn.Parameter] | None = None,
    expected_ecoc_codebook: torch.Tensor | None = None,
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
    if expected_ecoc_codebook is not None:
        saved_codebook = ckpt_loaded.get("ecoc_codebook")
        if isinstance(saved_codebook, torch.Tensor):
            _assert_checkpoint_ecoc_codebook_matches(saved_codebook, device)

    state_dict = ckpt_loaded.get("phase_system_state")
    copied = False
    if isinstance(state_dict, dict):
        try:
            phase_system.load_state_dict(state_dict, strict=False)
            copied = True
        except RuntimeError as exc:
            print(f"[warn] Failed to load full phase_system state_dict: {exc}")
    if not copied:
        saved_list = ckpt_loaded.get("phase_params_raw", ckpt_loaded.get("phase_params"))
        phase_params = phase_system.phase_parameters()
        if isinstance(saved_list, (list, tuple)) and len(saved_list) == len(phase_params):
            for param, saved in zip(phase_params, saved_list):
                if isinstance(saved, torch.Tensor):
                    param.data.copy_(saved.to(device))

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
    history: Dict[str, List[float]],
    cfg: Dict[str, object],
    extra_state: Dict[str, object] | None = None,
) -> None:
    phase_params = phase_system.phase_parameters()
    ckpt_obj = {
        "phase_params_raw": [p.detach().cpu() for p in phase_params],
        "phase_params": [phase_to_unit(p).detach().cpu() for p in phase_params],
        "phase_system_state": phase_system.state_dict(),
        "history": history,
        "cfg": cfg,
        "iter": iter_idx,
        "epoch": epoch_idx,
    }
    binary_cfg = cfg.get("binary_decision", {}) if isinstance(cfg, dict) else {}
    if isinstance(binary_cfg, dict):
        mode = str(binary_cfg.get("mode", "one_vs_rest")).strip().lower()
        ckpt_obj["binary_decision_mode"] = mode
        if mode == "ecoc":
            ckpt_obj["ecoc_codebook_type"] = ECOC_CODEBOOK_TYPE
    if extra_state:
        for key, value in extra_state.items():
            if isinstance(value, torch.Tensor):
                ckpt_obj[key] = value.detach().cpu()
            elif isinstance(value, (str, int, float, bool)) or value is None:
                ckpt_obj[key] = value
    torch.save(ckpt_obj, path)


class _MetricBuffer:
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


def _resolve_binary_cfg(cfg: Dict[str, object], channel_num: int) -> Dict[str, object]:
    binary_cfg = dict(cfg.get("binary_decision", {}))
    if not bool(binary_cfg.get("enable", True)):
        raise ValueError("optical_onn_training_binary_decision.py requires binary_decision.enable=true")
    if channel_num != 16:
        raise ValueError(f"binary_decision requires channel_num=16, got {channel_num}")
    num_classes = int(binary_cfg.get("num_classes", cfg.get("tiles", {}).get("num_classes", 10)))
    if num_classes != 10:
        raise ValueError(f"binary_decision requires num_classes=10, got {num_classes}")
    mode = str(binary_cfg.get("mode", "one_vs_rest")).strip().lower()
    if mode not in {"one_vs_rest", "ecoc"}:
        raise ValueError("binary_decision.mode must be one of {'one_vs_rest', 'ecoc'}")
    if mode == "one_vs_rest":
        active_class_channels = int(binary_cfg.get("active_class_channels", 10))
        dummy_channels = int(binary_cfg.get("dummy_channels", channel_num - active_class_channels))
        if active_class_channels != 10:
            raise ValueError("binary_decision.active_class_channels must be 10 for one_vs_rest classes 0..9")
        if dummy_channels != 6:
            raise ValueError("binary_decision.dummy_channels must be 6 for one_vs_rest with channel_num=16")
    else:
        active_class_channels = channel_num
        dummy_channels = 0
    dummy_mode = str(binary_cfg.get("dummy_mode", "zeros")).strip().lower()
    if dummy_mode not in {"zeros", "fixed_random"}:
        raise ValueError("binary_decision.dummy_mode must be 'zeros' or 'fixed_random'")
    input_encoding = str(binary_cfg.get("input_encoding", "single")).strip().lower()
    if input_encoding in {"rot4", "rot4_tiling"}:
        input_encoding = "rot4_tile"
    if input_encoding not in {"single", "rot4_tile"}:
        raise ValueError("binary_decision.input_encoding must be 'single' or 'rot4_tile'")
    yes_region = str(binary_cfg.get("yes_region", "top")).strip().lower()
    no_region = str(binary_cfg.get("no_region", "bottom")).strip().lower()
    if yes_region not in {"top", "bottom"} or no_region not in {"top", "bottom"}:
        raise ValueError("binary_decision yes_region/no_region currently support only 'top' and 'bottom'")
    if yes_region == no_region:
        raise ValueError("binary_decision yes_region and no_region must be different")
    yes_no_tile_hw = _parse_optional_hw(
        binary_cfg.get("yes_no_tile_hw", None),
        key="binary_decision.yes_no_tile_hw",
    )
    yes_no_tile_margin = _parse_nonnegative_hw(
        binary_cfg.get("yes_no_tile_margin", (0, 0)),
        key="binary_decision.yes_no_tile_margin",
    )
    yes_no_tile_gap = int(binary_cfg.get("yes_no_tile_gap", 0))
    if yes_no_tile_gap < 0:
        raise ValueError("binary_decision.yes_no_tile_gap must be non-negative")
    ecoc_raw_cfg = dict(binary_cfg.get("ecoc", {}))
    onn_cfg = cfg.get("onn", {}) if isinstance(cfg.get("onn", {}), dict) else {}
    margin_loss_weight = float(binary_cfg.get("margin_loss_weight", onn_cfg.get("margin_loss_weight", 0.0)))
    margin_loss_gap = float(binary_cfg.get("margin_loss_gap", onn_cfg.get("margin_loss_gap", 0.0)))
    lambda_spill = float(binary_cfg.get("lambda_spill", 0.0))
    if margin_loss_weight < 0:
        raise ValueError("binary_decision.margin_loss_weight must be non-negative")
    if margin_loss_gap < 0:
        raise ValueError("binary_decision.margin_loss_gap must be non-negative")
    if lambda_spill < 0:
        raise ValueError("binary_decision.lambda_spill must be non-negative")
    ecoc_cfg = {
        "num_bits": int(ecoc_raw_cfg.get("num_bits", 16)),
        "codebook_type": str(ecoc_raw_cfg.get("codebook_type", ECOC_CODEBOOK_TYPE)),
        "normalize_decoder": bool(ecoc_raw_cfg.get("normalize_decoder", True)),
        "use_aux_bit_loss": bool(ecoc_raw_cfg.get("use_aux_bit_loss", False)),
        "lambda_bit": float(ecoc_raw_cfg.get("lambda_bit", 0.0)),
    }
    if ecoc_cfg["num_bits"] != 16:
        raise ValueError("binary_decision.ecoc.num_bits must be 16")
    if ecoc_cfg["codebook_type"] != ECOC_CODEBOOK_TYPE:
        raise ValueError(f"binary_decision.ecoc.codebook_type must be {ECOC_CODEBOOK_TYPE!r}")

    binary_cfg.update(
        {
            "enable": True,
            "mode": mode,
            "num_classes": num_classes,
            "active_class_channels": active_class_channels,
            "dummy_channels": dummy_channels,
            "dummy_mode": dummy_mode,
            "dummy_seed": int(binary_cfg.get("dummy_seed", 1234)),
            "input_encoding": input_encoding,
            "yes_region": yes_region,
            "no_region": no_region,
            "use_log_ratio_margin": bool(binary_cfg.get("use_log_ratio_margin", True)),
            "use_aux_binary_loss": bool(binary_cfg.get("use_aux_binary_loss", False)),
            "lambda_bin": float(binary_cfg.get("lambda_bin", 0.0)),
            "binary_pos_weight": float(binary_cfg.get("binary_pos_weight", 9.0)),
            "normalize_yes_no": bool(binary_cfg.get("normalize_yes_no", True)),
            "eps": float(binary_cfg.get("eps", 1e-6)),
            "margin_loss_weight": margin_loss_weight,
            "margin_loss_gap": margin_loss_gap,
            "lambda_spill": lambda_spill,
            "debug_first_iters": int(binary_cfg.get("debug_first_iters", 0)),
            "yes_no_tile_hw": yes_no_tile_hw,
            "yes_no_tile_margin": yes_no_tile_margin,
            "yes_no_tile_gap": yes_no_tile_gap,
            "ecoc": ecoc_cfg,
        }
    )
    return binary_cfg


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
        tile_h, tile_w = patch_h * 2, patch_w * 2
        if tile_h > H or tile_w > W:
            raise ValueError(
                f"Expected channel image at least {(tile_h, tile_w)} for tiled reconstruction, got {(H, W)}"
            )
        tiled = F.interpolate(x_ch, size=(tile_h, tile_w), mode="bilinear", align_corners=False)
        patches = tiled[:, :, :patch_h, :patch_w]
    else:
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


def _rot4_tile_input_image(x_img: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """Encode one image as a 2x2 canvas: 0, 90, 180, and 270 degree rotations."""
    if x_img.dim() != 4 or x_img.size(0) != 1 or x_img.size(1) != 1:
        raise ValueError(f"Expected x_img (1,1,H,W) for rot4 tiling, got {tuple(x_img.shape)}")

    H, W = int(target_hw[0]), int(target_hw[1])
    if H < 2 or W < 2:
        raise ValueError(f"target_hw must be at least 2x2 for rot4 tiling, got {target_hw}")

    h_top = H // 2
    h_bottom = H - h_top
    w_left = W // 2
    w_right = W - w_left
    rotations = [
        torch.rot90(x_img, k=0, dims=(-2, -1)),
        torch.rot90(x_img, k=-1, dims=(-2, -1)),  # 90 deg clockwise
        torch.rot90(x_img, k=2, dims=(-2, -1)),
        torch.rot90(x_img, k=1, dims=(-2, -1)),   # 270 deg clockwise
    ]
    tile_sizes = [
        (h_top, w_left),
        (h_top, w_right),
        (h_bottom, w_left),
        (h_bottom, w_right),
    ]
    tiles = [
        F.interpolate(rot, size=size, mode="bilinear", align_corners=False)
        for rot, size in zip(rotations, tile_sizes)
    ]
    top = torch.cat([tiles[0], tiles[1]], dim=-1)
    bottom = torch.cat([tiles[2], tiles[3]], dim=-1)
    return torch.cat([top, bottom], dim=-2).clamp(0.0, 1.0)


def _prepare_binary_decision_input(
    x_batch: torch.Tensor,
    *,
    channel_num: int,
    binary_cfg: Dict[str, object],
    data_method: str,
    target_hw: Tuple[int, int],
    use_rot4_tiling: bool,
    device: torch.device,
) -> torch.Tensor:
    if x_batch.dim() != 4:
        raise ValueError(f"Expected x_batch (1,C,H,W), got {tuple(x_batch.shape)}")
    if x_batch.size(0) != 1:
        raise ValueError(f"Binary decision mode requires data.batch_size=1, got {tuple(x_batch.shape)}")

    active_class_channels = int(binary_cfg["active_class_channels"])
    mode = str(binary_cfg.get("mode", "one_vs_rest"))
    if mode == "one_vs_rest":
        assert active_class_channels == 10
        assert int(binary_cfg["dummy_channels"]) == 6
    elif mode == "ecoc":
        assert active_class_channels == channel_num == 16
        assert int(binary_cfg["dummy_channels"]) == 0
    else:
        raise ValueError(f"Unsupported binary_decision.mode {mode!r}")
    assert channel_num >= active_class_channels
    data_method = str(data_method).strip().lower()

    if data_method == "upsampler":
        if x_batch.size(1) != 1:
            raise ValueError(f"Upsampler mode expects x_batch (1,1,H,W), got {tuple(x_batch.shape)}")
        x_img = x_batch
    elif data_method == "patchify":
        if x_batch.size(1) == 1:
            x_img = x_batch
        elif x_batch.size(1) == channel_num:
            grid = int(round(math.sqrt(channel_num)))
            x_img = _reconstruct_full_input(x_batch, grid=grid, use_rot4_tiling=use_rot4_tiling)
        else:
            raise ValueError(f"Patchify mode expected 1 or {channel_num} channels, got {tuple(x_batch.shape)}")
    else:
        raise ValueError(f"Unsupported data.method '{data_method}'")

    x_img = x_img.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    input_encoding = str(binary_cfg.get("input_encoding", "single")).strip().lower()
    if input_encoding == "rot4_tile":
        x_img = _rot4_tile_input_image(x_img, target_hw)
    elif tuple(x_img.shape[-2:]) != tuple(target_hw):
        x_img = F.interpolate(x_img, size=target_hw, mode="bilinear", align_corners=False)
    active = x_img.repeat(active_class_channels, 1, 1, 1)

    dummy_channels = int(binary_cfg["dummy_channels"])
    if dummy_channels > 0:
        dummy_mode = str(binary_cfg["dummy_mode"])
        if dummy_mode == "zeros":
            dummy = torch.zeros(
                dummy_channels,
                1,
                int(target_hw[0]),
                int(target_hw[1]),
                device=device,
                dtype=x_img.dtype,
            )
        elif dummy_mode == "fixed_random":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(binary_cfg["dummy_seed"]))
            dummy = torch.rand(
                dummy_channels,
                1,
                int(target_hw[0]),
                int(target_hw[1]),
                generator=generator,
                dtype=x_img.dtype,
            ).to(device=device)
        else:
            raise ValueError(f"Unsupported dummy_mode {dummy_mode!r}")
        x_phys = torch.cat([active, dummy], dim=0)
    else:
        x_phys = active

    assert x_phys.ndim == 4
    assert x_phys.shape[0] == channel_num
    assert x_phys.shape[1] == 1
    assert x_phys.shape[-2:] == tuple(target_hw)
    return x_phys


def build_binary_yes_no_masks(
    out_hw: Tuple[int, int],
    *,
    yes_region: str = "top",
    no_region: str = "bottom",
    tile_hw: Tuple[int, int] | None = None,
    tile_margin: Tuple[int, int] = (0, 0),
    tile_gap: int = 0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return masks with convention binary_scores[:, 0]=NO and binary_scores[:, 1]=YES."""
    H, W = int(out_hw[0]), int(out_hw[1])
    if H < 2 or W < 1:
        raise ValueError(f"Invalid output size for binary masks: {out_hw}")
    yes_region = str(yes_region).strip().lower()
    no_region = str(no_region).strip().lower()
    if yes_region not in {"top", "bottom"} or no_region not in {"top", "bottom"}:
        raise ValueError("yes_region/no_region must be 'top' or 'bottom'")
    if yes_region == no_region:
        raise ValueError("yes_region and no_region must be different")

    masks = torch.zeros((2, H, W), dtype=torch.float32, device=device)

    margin_y, margin_x = int(tile_margin[0]), int(tile_margin[1])
    gap = int(tile_gap)
    if margin_y < 0 or margin_x < 0 or gap < 0:
        raise ValueError("binary_decision tile margins/gap must be non-negative")
    avail_h = H - 2 * margin_y - gap
    avail_w = W - 2 * margin_x
    if avail_h <= 0 or avail_w <= 0:
        raise ValueError(
            f"binary_decision tile margins/gap leave no usable area: out_hw={out_hw}, "
            f"margin={tile_margin}, gap={gap}"
        )
    if tile_hw is None:
        tile_h = avail_h // 2
        tile_w = avail_w
    else:
        tile_h, tile_w = int(tile_hw[0]), int(tile_hw[1])
    if tile_h <= 0 or tile_w <= 0:
        raise ValueError(f"binary_decision.yes_no_tile_hw must be positive, got {tile_hw!r}")
    content_h = 2 * tile_h + gap
    if content_h > H - 2 * margin_y:
        raise ValueError(
            f"YES/NO tiles do not fit vertically: 2*{tile_h}+gap {gap} exceeds "
            f"height {H} minus margins {2 * margin_y}"
        )
    if tile_w > W - 2 * margin_x:
        raise ValueError(
            f"YES/NO tile width {tile_w} exceeds width {W} minus margins {2 * margin_x}"
        )

    stack_r0 = margin_y + ((H - 2 * margin_y - content_h) // 2)
    c0 = margin_x + ((W - 2 * margin_x - tile_w) // 2)
    top_box = (stack_r0, c0, stack_r0 + tile_h, c0 + tile_w)
    bottom_r0 = stack_r0 + tile_h + gap
    bottom_box = (bottom_r0, c0, bottom_r0 + tile_h, c0 + tile_w)
    region_boxes = {
        "top": top_box,
        "bottom": bottom_box,
    }

    no_r0, no_c0, no_r1, no_c1 = region_boxes[no_region]
    yes_r0, yes_c0, yes_r1, yes_c1 = region_boxes[yes_region]
    masks[0, no_r0:no_r1, no_c0:no_c1] = 1.0
    masks[1, yes_r0:yes_r1, yes_c0:yes_c1] = 1.0
    area = masks.sum(dim=(1, 2))
    if not torch.allclose(area[0], area[1]):
        raise RuntimeError(f"YES/NO masks must have equal area, got {area.tolist()}")
    return masks


def compute_binary_yes_no_scores(
    y_active: torch.Tensor,
    masks: torch.Tensor,
    *,
    eps: float = 1e-6,
    normalize: bool = True,
) -> torch.Tensor:
    if y_active.dim() != 4 or y_active.size(1) != 1:
        raise ValueError(f"Expected y_active (N,1,H,W), got {tuple(y_active.shape)}")
    if masks.dim() != 3 or masks.size(0) != 2:
        raise ValueError(f"Expected masks (2,H,W), got {tuple(masks.shape)}")
    if tuple(y_active.shape[-2:]) != tuple(masks.shape[-2:]):
        raise ValueError(
            f"Output/mask shape mismatch: y_active={tuple(y_active.shape[-2:])}, masks={tuple(masks.shape[-2:])}"
        )

    work = y_active
    if normalize:
        total_power = work.sum(dim=(1, 2, 3), keepdim=True).clamp_min(float(eps))
        work = work / total_power.detach()
    spatial = work.squeeze(1)
    masks = masks.to(device=y_active.device, dtype=spatial.dtype)
    mask_area = masks.sum(dim=(1, 2)).clamp_min(float(eps)).view(1, 2)
    scores = (spatial.unsqueeze(1) * masks.unsqueeze(0)).flatten(-2).sum(dim=-1) / mask_area
    return scores


def _apply_binary_score_scaling(
    binary_scores: torch.Tensor,
    readout_cfg: Dict[str, object],
    temperature_param: torch.Tensor | None,
) -> torch.Tensor:
    scaled = binary_scores
    if bool(readout_cfg["use_log_scaling"]):
        scaled = torch.log(scaled.clamp_min(float(readout_cfg["log_eps"])))

    if temperature_param is not None:
        temp = temperature_param
        if temp.dim() == 0:
            temp = temp.view(1)
        if temp.numel() == 1:
            scaled = scaled * temp.view(1, 1)
        elif temp.numel() == scaled.size(0):
            scaled = scaled * temp.view(-1, 1)
        else:
            raise ValueError(
                f"readout temperature shape {tuple(temp.shape)} must be scalar or active-channel length {scaled.size(0)}"
            )
    elif float(readout_cfg["temperature"]) != 1.0:
        scaled = scaled * float(readout_cfg["temperature"])
    return scaled


def _apply_margin_temperature(
    margins: torch.Tensor,
    readout_cfg: Dict[str, object],
    temperature_param: torch.Tensor | None,
) -> torch.Tensor:
    scaled = margins
    if temperature_param is not None:
        temp = temperature_param
        if temp.dim() == 0:
            temp = temp.view(1)
        if temp.numel() == 1:
            scaled = scaled * temp.view(1)
        elif temp.numel() == scaled.numel():
            scaled = scaled * temp.view_as(scaled)
        else:
            raise ValueError(
                f"readout temperature shape {tuple(temp.shape)} must be scalar or margin length {scaled.numel()}"
            )
    elif float(readout_cfg["temperature"]) != 1.0:
        scaled = scaled * float(readout_cfg["temperature"])
    return scaled


def binary_scores_to_class_logits(binary_scores: torch.Tensor) -> torch.Tensor:
    if binary_scores.dim() != 2 or binary_scores.size(1) != 2:
        raise ValueError(f"Expected binary_scores (10,2), got {tuple(binary_scores.shape)}")
    no_scores = binary_scores[:, 0]
    yes_scores = binary_scores[:, 1]
    return (yes_scores - no_scores).unsqueeze(0)


def _compute_yes_no_margins(binary_scores_raw: torch.Tensor, binary_cfg: Dict[str, object]) -> torch.Tensor:
    if binary_scores_raw.dim() != 2 or binary_scores_raw.size(1) != 2:
        raise ValueError(f"Expected binary_scores_raw (N,2), got {tuple(binary_scores_raw.shape)}")
    no_scores = binary_scores_raw[:, 0]
    yes_scores = binary_scores_raw[:, 1]
    eps = float(binary_cfg.get("eps", 1e-6))
    if bool(binary_cfg.get("use_log_ratio_margin", True)):
        return torch.log(yes_scores.clamp_min(eps)) - torch.log(no_scores.clamp_min(eps))
    return yes_scores - no_scores


def _decode_ecoc_logits(
    margins_2d: torch.Tensor,
    codebook: torch.Tensor,
    binary_cfg: Dict[str, object],
) -> torch.Tensor:
    if margins_2d.shape != (1, 16):
        raise ValueError(f"ECOC margins_2d must be (1,16), got {tuple(margins_2d.shape)}")
    if codebook.shape != (10, 16):
        raise ValueError(f"ECOC codebook must be (10,16), got {tuple(codebook.shape)}")
    class_logits = margins_2d @ codebook.to(device=margins_2d.device, dtype=margins_2d.dtype).T
    if bool(binary_cfg.get("ecoc", {}).get("normalize_decoder", True)):
        class_logits = class_logits / math.sqrt(16)
    if class_logits.shape != (1, 10):
        raise ValueError(f"ECOC class_logits must be (1,10), got {tuple(class_logits.shape)}")
    return class_logits


def _compute_binary_readout_details(
    y_final: torch.Tensor,
    binary_cfg: Dict[str, object],
    readout_cfg: Dict[str, object],
    temperature_param: torch.Tensor | None,
    ecoc_codebook: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    active_class_channels = int(binary_cfg["active_class_channels"])
    if y_final.dim() != 4 or y_final.size(0) < active_class_channels or y_final.size(1) != 1:
        raise ValueError(f"Expected y_final ({active_class_channels}+,1,H,W), got {tuple(y_final.shape)}")
    y_active = y_final[:active_class_channels]
    masks = build_binary_yes_no_masks(
        tuple(y_active.shape[-2:]),
        yes_region=str(binary_cfg["yes_region"]),
        no_region=str(binary_cfg["no_region"]),
        tile_hw=binary_cfg.get("yes_no_tile_hw"),
        tile_margin=binary_cfg.get("yes_no_tile_margin", (0, 0)),
        tile_gap=int(binary_cfg.get("yes_no_tile_gap", 0)),
        device=y_active.device,
    )
    binary_scores_raw = compute_binary_yes_no_scores(
        y_active,
        masks,
        eps=float(binary_cfg["eps"]),
        normalize=bool(binary_cfg["normalize_yes_no"]),
    )
    mode = str(binary_cfg.get("mode", "one_vs_rest"))
    if mode == "one_vs_rest":
        binary_scores = _apply_binary_score_scaling(binary_scores_raw, readout_cfg, temperature_param)
        class_logits = binary_scores_to_class_logits(binary_scores)
        margins_2d = class_logits
        if class_logits.shape != (1, 10):
            raise ValueError(f"one_vs_rest class_logits must be (1,10), got {tuple(class_logits.shape)}")
    elif mode == "ecoc":
        binary_scores = binary_scores_raw
        margins = _compute_yes_no_margins(binary_scores_raw, binary_cfg)
        margins = _apply_margin_temperature(margins, readout_cfg, temperature_param)
        margins_2d = margins.unsqueeze(0)
        if margins_2d.shape != (1, 16):
            raise ValueError(f"ECOC margins_2d must be (1,16), got {tuple(margins_2d.shape)}")
        codebook = (
            get_fixed_ecoc_codebook(device=y_final.device, dtype=margins_2d.dtype)
            if ecoc_codebook is None
            else ecoc_codebook.to(device=y_final.device, dtype=margins_2d.dtype)
        )
        class_logits = _decode_ecoc_logits(margins_2d, codebook, binary_cfg)
    else:
        raise ValueError(f"Unsupported binary_decision.mode {mode!r}")
    if binary_scores.shape != (active_class_channels, 2):
        raise ValueError(f"binary_scores must be ({active_class_channels},2), got {tuple(binary_scores.shape)}")
    return y_active, masks, binary_scores_raw, binary_scores, class_logits, margins_2d


def _compute_binary_readout(
    y_final: torch.Tensor,
    binary_cfg: Dict[str, object],
    readout_cfg: Dict[str, object],
    temperature_param: torch.Tensor | None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    y_active, masks, binary_scores_raw, binary_scores, class_logits, _ = _compute_binary_readout_details(
        y_final,
        binary_cfg,
        readout_cfg,
        temperature_param,
    )
    return y_active, masks, binary_scores_raw, binary_scores, class_logits


def _compute_decoded_margin_loss(
    class_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin_gap: float,
) -> torch.Tensor:
    if class_logits.dim() != 2:
        raise ValueError(f"Expected class_logits (B,C), got {tuple(class_logits.shape)}")
    if class_logits.size(1) < 2:
        raise ValueError("Decoded margin loss requires at least two classes")
    labels = labels.view(-1).long()
    if labels.numel() != class_logits.size(0):
        raise ValueError(
            f"labels must have batch length {class_logits.size(0)}, got shape {tuple(labels.shape)}"
        )
    target_logits = class_logits.gather(1, labels.view(-1, 1)).squeeze(1)
    target_mask = F.one_hot(labels, num_classes=class_logits.size(1)).to(
        dtype=torch.bool,
        device=class_logits.device,
    )
    wrong_logits = class_logits.masked_fill(target_mask, -torch.inf).max(dim=1).values
    return F.relu(float(margin_gap) - (target_logits - wrong_logits)).mean()


def _compute_spillover_loss(
    y_active: torch.Tensor,
    masks: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    if y_active.dim() != 4 or y_active.size(1) != 1:
        raise ValueError(f"Expected y_active (N,1,H,W), got {tuple(y_active.shape)}")
    if masks.dim() != 3 or masks.size(0) != 2:
        raise ValueError(f"Expected masks (2,H,W), got {tuple(masks.shape)}")
    if tuple(y_active.shape[-2:]) != tuple(masks.shape[-2:]):
        raise ValueError(
            f"Output/mask shape mismatch: y_active={tuple(y_active.shape[-2:])}, masks={tuple(masks.shape[-2:])}"
        )

    spatial = y_active.squeeze(1)
    masks = masks.to(device=y_active.device, dtype=spatial.dtype)
    inside_mask = masks.sum(dim=0).clamp_max(1.0)
    inside_power = (spatial * inside_mask.unsqueeze(0)).flatten(1).sum(dim=1)
    total_power = spatial.flatten(1).sum(dim=1).clamp_min(float(eps))
    inside_fraction = (inside_power / total_power).clamp(0.0, 1.0)
    return (1.0 - inside_fraction).mean()


def _compute_binary_losses(
    class_logits: torch.Tensor,
    labels: torch.Tensor,
    binary_cfg: Dict[str, object],
    margins_2d: torch.Tensor | None = None,
    ecoc_codebook: torch.Tensor | None = None,
    spillover_loss: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    loss_cls = F.cross_entropy(class_logits, labels)
    loss_bin = class_logits.new_zeros(())
    loss_margin = class_logits.new_zeros(())
    loss_spill = class_logits.new_zeros(()) if spillover_loss is None else spillover_loss.to(
        device=class_logits.device,
        dtype=class_logits.dtype,
    )
    margin_loss_weight = float(binary_cfg.get("margin_loss_weight", 0.0))
    lambda_spill = float(binary_cfg.get("lambda_spill", 0.0))
    if margin_loss_weight > 0:
        loss_margin = _compute_decoded_margin_loss(
            class_logits,
            labels,
            margin_gap=float(binary_cfg.get("margin_loss_gap", 0.0)),
        )
    mode = str(binary_cfg.get("mode", "one_vs_rest"))
    if mode == "ecoc":
        ecoc_cfg = binary_cfg.get("ecoc", {})
        if bool(ecoc_cfg.get("use_aux_bit_loss", False)):
            if margins_2d is None:
                raise ValueError("ECOC auxiliary bit loss requires margins_2d")
            if margins_2d.shape != (1, 16):
                raise ValueError(f"ECOC margins_2d must be (1,16), got {tuple(margins_2d.shape)}")
            codebook = (
                get_fixed_ecoc_codebook(device=class_logits.device, dtype=class_logits.dtype)
                if ecoc_codebook is None
                else ecoc_codebook.to(device=class_logits.device, dtype=class_logits.dtype)
            )
            target_bits_pm1 = codebook[labels]
            target_bits_01 = (target_bits_pm1 > 0).to(dtype=class_logits.dtype)
            loss_bin = F.binary_cross_entropy_with_logits(margins_2d, target_bits_01)
        total_loss = (
            loss_cls
            + float(ecoc_cfg.get("lambda_bit", 0.0)) * loss_bin
            + margin_loss_weight * loss_margin
            + lambda_spill * loss_spill
        )
        return total_loss, loss_cls, loss_bin, loss_margin, loss_spill
    if class_logits.shape != (1, 10):
        raise ValueError(f"one_vs_rest class_logits must be (1,10), got {tuple(class_logits.shape)}")
    if bool(binary_cfg["use_aux_binary_loss"]):
        label_scalar = int(labels.item())
        target_bin = torch.zeros(class_logits.size(1), device=class_logits.device, dtype=class_logits.dtype)
        target_bin[label_scalar] = 1.0
        pos_weight = torch.as_tensor(
            float(binary_cfg["binary_pos_weight"]),
            device=class_logits.device,
            dtype=class_logits.dtype,
        )
        loss_bin = F.binary_cross_entropy_with_logits(
            class_logits.squeeze(0),
            target_bin,
            pos_weight=pos_weight,
        )
    total_loss = (
        loss_cls
        + float(binary_cfg["lambda_bin"]) * loss_bin
        + margin_loss_weight * loss_margin
        + lambda_spill * loss_spill
    )
    return total_loss, loss_cls, loss_bin, loss_margin, loss_spill


def _binary_margin_stats(
    class_logits: torch.Tensor,
    binary_scores: torch.Tensor,
    labels: torch.Tensor,
    binary_cfg: Dict[str, object] | None = None,
    margins_2d: torch.Tensor | None = None,
    ecoc_codebook: torch.Tensor | None = None,
) -> Dict[str, float]:
    label_scalar = int(labels.item())
    class_scores = class_logits.detach().squeeze(0)
    pred = int(torch.argmax(class_scores).item())
    neg_mask = torch.ones_like(class_scores, dtype=torch.bool)
    neg_mask[label_scalar] = False
    max_negative_margin = class_scores[neg_mask].max()
    true_margin = class_scores[label_scalar]
    stats = {
        "pred": float(pred),
        "true_yes_score": float("nan"),
        "true_no_score": float("nan"),
        "true_margin": float(true_margin.cpu()),
        "max_negative_margin": float(max_negative_margin.cpu()),
        "margin_gap": float((true_margin - max_negative_margin).cpu()),
        "bit_acc": float("nan"),
    }
    mode = str((binary_cfg or {}).get("mode", "one_vs_rest"))
    if mode == "one_vs_rest":
        stats["true_yes_score"] = float(binary_scores.detach()[label_scalar, 1].cpu())
        stats["true_no_score"] = float(binary_scores.detach()[label_scalar, 0].cpu())
        return stats
    if margins_2d is not None:
        codebook = (
            get_fixed_ecoc_codebook(device=margins_2d.device, dtype=margins_2d.dtype)
            if ecoc_codebook is None
            else ecoc_codebook.to(device=margins_2d.device, dtype=margins_2d.dtype)
        )
        target_bits = codebook[label_scalar]
        measured_sign = torch.where(
            margins_2d.detach().squeeze(0) >= 0,
            torch.ones_like(target_bits),
            -torch.ones_like(target_bits),
        )
        stats["bit_acc"] = float((measured_sign == target_bits).float().mean().cpu())
    return stats


def _debug_log_binary_iteration(
    *,
    iteration: int,
    x_phys: torch.Tensor,
    y_final: torch.Tensor,
    y_active: torch.Tensor,
    binary_scores: torch.Tensor,
    margins_2d: torch.Tensor,
    class_logits: torch.Tensor,
    labels: torch.Tensor,
    stats: Dict[str, float],
    binary_cfg: Dict[str, object],
    channel_num: int,
    loss_cls: torch.Tensor,
    acc: float,
    ecoc_codebook: torch.Tensor | None = None,
) -> None:
    def _summary(name: str, tensor: torch.Tensor) -> str:
        t = tensor.detach().float()
        return (
            f"{name} shape={tuple(tensor.shape)} "
            f"min={float(t.min().cpu()):.6g} max={float(t.max().cpu()):.6g} "
            f"mean={float(t.mean().cpu()):.6g}"
        )

    label_scalar = int(labels.item())
    pred = int(stats["pred"])
    mode = str(binary_cfg.get("mode", "one_vs_rest"))
    print(
        f"[binary-debug iter={iteration}] "
        f"mode={mode} | "
        f"{_summary('x_phys', x_phys)} | "
        f"{_summary('y_final', y_final)} | "
        f"y_active shape={tuple(y_active.shape)} | "
        f"binary_scores shape={tuple(binary_scores.shape)} | "
        f"margins shape={tuple(margins_2d.shape)} | "
        f"class_logits shape={tuple(class_logits.shape)} | "
        f"label={label_scalar} pred={pred} | loss_cls={float(loss_cls.detach().cpu()):.6g} acc={acc:.6g} | "
        f"true_score={stats['true_margin']:.6g} max_wrong={stats['max_negative_margin']:.6g} "
        f"decoded_margin_gap={stats['margin_gap']:.6g}"
    )
    if mode == "one_vs_rest":
        active_class_channels = int(binary_cfg["active_class_channels"])
        print(
            f"[binary-debug iter={iteration}] one_vs_rest true_yes={stats['true_yes_score']:.6g} "
            f"true_no={stats['true_no_score']:.6g}; dummy channels "
            f"{active_class_channels}..{channel_num - 1} excluded"
        )
    elif mode == "ecoc":
        codebook = (
            get_fixed_ecoc_codebook(device=margins_2d.device, dtype=margins_2d.dtype)
            if ecoc_codebook is None
            else ecoc_codebook.to(device=margins_2d.device, dtype=margins_2d.dtype)
        )
        print(
            f"[binary-debug iter={iteration}] ecoc codebook_type={ECOC_CODEBOOK_TYPE} "
            f"codebook_shape={tuple(codebook.shape)} bit_acc={stats['bit_acc']:.6g}"
        )


def _forward_no_mix(
    x_phys: torch.Tensor,
    phase_system: PhaseSystem,
    bridge: OpticalBridge,
    *,
    channel_num: int,
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
        dummy_phase = torch.zeros(
            x_phys.size(0),
            1,
            *bridge.enc_canvas_hw,
            device=x_phys.device,
            dtype=x_phys.dtype,
        )
        base_canvas_for_phase, _ = encoding_x_phase_physical(
            x_phys,
            dummy_phase,
            canvas_hw=bridge.enc_canvas_hw,
            x_mode=bridge.modes[0],
            x_norm_mod=bridge.x_norm_modes[0],
        )

    for idx, phase_param in enumerate(phase_system.phase_parameters()):
        assert x_current.shape[0] == channel_num
        assert x_current.shape[1] == 1
        layer_inputs_pre.append(x_current.detach())
        x_canvas_for_phase: Optional[torch.Tensor] = None
        if getattr(phase_system, "structural_nonlinearity", False):
            x_canvas_for_phase = base_canvas_for_phase
        phase_batch = phase_system.expand_to_batch(
            phase_param,
            x_current.size(0),
            x_input=x_canvas_for_phase,
            layer_idx=idx,
        )
        y_layer = bridge._apply_layer(
            idx,
            x_current,
            phase_batch,
            bridge.modes[idx],
            bridge.x_norm_modes[idx],
            track_grads=track_grads,
        )
        assert y_layer.ndim == 4
        y_layer = bridge._apply_gain(y_layer, idx)
        x_encoded = bridge.encoded_cache[idx] if bridge.encoded_cache[idx] is not None else y_layer
        phase_encoded = bridge.phase_cache[idx] if bridge.phase_cache[idx] is not None else phase_batch
        layer_inputs.append(x_encoded.detach())
        layer_phases.append(phase_encoded.detach())
        layer_outputs.append(y_layer.detach())
        x_current = y_layer

    return x_current, layer_inputs, layer_phases, layer_outputs, layer_inputs_pre


def _evaluate(
    val_loader,
    device: torch.device,
    phase_system: PhaseSystem,
    bridge: OpticalBridge,
    readout_cfg: Dict[str, object],
    binary_cfg: Dict[str, object],
    channel_num: int,
    data_method: str,
    input_hw: Tuple[int, int],
    use_rot4_tiling: bool,
    *,
    temperature_param: torch.Tensor | None = None,
    ecoc_codebook: torch.Tensor | None = None,
) -> Tuple[float, float]:
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            labels_val = y_val.to(device)
            x_phys = _prepare_binary_decision_input(
                x_val,
                channel_num=channel_num,
                binary_cfg=binary_cfg,
                data_method=data_method,
                target_hw=input_hw,
                use_rot4_tiling=use_rot4_tiling,
                device=device,
            )
            y_final, _, _, _, _ = _forward_no_mix(
                x_phys,
                phase_system,
                bridge,
                channel_num=channel_num,
                track_grads=False,
            )
            y_active, yes_no_masks, _, _, class_logits, margins_2d = _compute_binary_readout_details(
                y_final,
                binary_cfg,
                readout_cfg,
                temperature_param,
                ecoc_codebook,
            )
            spillover_loss = _compute_spillover_loss(
                y_active,
                yes_no_masks,
                eps=float(binary_cfg["eps"]),
            )
            loss, _, _, _, _ = _compute_binary_losses(
                class_logits,
                labels_val,
                binary_cfg,
                margins_2d=margins_2d,
                ecoc_codebook=ecoc_codebook,
                spillover_loss=spillover_loss,
            )
            pred = class_logits.argmax(dim=1)
            total_loss += float(loss.item())
            total_acc += float((pred == labels_val).float().sum().item())
            total_samples += int(labels_val.size(0))

    if total_samples <= 0:
        return float("nan"), float("nan")
    return total_loss / float(total_samples), total_acc / float(total_samples)


def train(
    layer: Optional[nn.Module] = None,
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, object]] = None,
) -> None:
    config_path = str(DEFAULT_BINARY_CONFIG_PATH if config_path is None else config_path)
    cfg, cfg_path = load_training_config(config_path, overrides)
    print(f"Using config: {cfg_path}")
    seed = resolve_onn_seed(cfg, default=1337)
    print(f"Using seed: {seed}")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    channel_num = int(cfg.get("channel_num", 16))
    binary_cfg = _resolve_binary_cfg(cfg, channel_num)
    binary_mode = str(binary_cfg["mode"])
    active_class_channels = int(binary_cfg["active_class_channels"])
    ecoc_codebook = (
        get_fixed_ecoc_codebook(device=device, dtype=torch.float32)
        if binary_mode == "ecoc"
        else None
    )
    if binary_mode == "one_vs_rest":
        print(
            f"Binary decision mode=one_vs_rest: active class channels 0..{active_class_channels - 1}; "
            f"dummy channels {active_class_channels}..{channel_num - 1} excluded from readout/loss/accuracy."
        )
    else:
        print(
            "Binary decision mode=ecoc: active code-bit channels 0..15; no dummy channels; "
            f"codebook={ECOC_CODEBOOK_TYPE} shape={tuple(ecoc_codebook.shape)}."
        )

    data_cfg = cfg.get("data", {})
    hw_cfg = data_cfg.get("hw", (112, 112))
    input_hw = (int(hw_cfg[0]), int(hw_cfg[1]))
    data_method = str(data_cfg.get("method", "upsampler")).strip().lower()
    if data_method not in {"patchify", "upsampler"}:
        raise ValueError("data.method must be 'patchify' or 'upsampler'")
    use_rot4_tiling = bool(data_cfg.get("use_rot4_tiling", False))
    batch_size = int(data_cfg.get("batch_size", 1))
    if batch_size != 1:
        raise ValueError("data.batch_size must be 1 for binary decision mode.")
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

    readout_raw_cfg = cfg.get("readout", {})
    readout_cfg = {
        "use_log_scaling": bool(readout_raw_cfg.get("use_log_scaling", True)),
        "temperature": float(readout_raw_cfg.get("temperature", 1.0)),
        "log_eps": float(readout_raw_cfg.get("log_eps", 1e-6)),
        "temperature_learnable": bool(readout_raw_cfg.get("temperature_learnable", False)),
        "temperature_per_channel": bool(readout_raw_cfg.get("temperature_per_channel", False)),
    }
    if readout_cfg["temperature"] <= 0:
        raise ValueError("readout.temperature must be positive")

    temperature_param: torch.Tensor | None = None
    if readout_cfg["temperature_learnable"]:
        temp_shape = (active_class_channels,) if readout_cfg["temperature_per_channel"] else (1,)
        temp_raw_init = _softplus_inverse(readout_cfg["temperature"])
        temperature_param = nn.Parameter(torch.full(temp_shape, temp_raw_init, device=device))

    max_epochs = int(onn_cfg.get("max_epochs", 10))
    eval_every = int(onn_cfg.get("eval_every", 200))
    ckpt_auto_mode, ckpt_every = _parse_ckpt_every(onn_cfg.get("ckpt_every", 1000))
    viz_every = int(onn_cfg.get("viz_every", 200))
    phase_viz_every = max(0, int(onn_cfg.get("phase_viz_every", viz_every)))
    metrics_avg_every = max(1, viz_every if viz_every > 0 else 1)
    metrics_update_every = max(1, int(onn_cfg.get("metrics_viz_every", metrics_avg_every)))

    optical_default_dir = PROJECT_ROOT / "pre_trained_model_save" / "Optical_neural_net"
    run_dir = Path(onn_cfg.get("save_dir", str(optical_default_dir)))
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_layer_path = checkpoint_dir / f"_binary_decision_{n_layers}.pt"

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
    if temperature_param is not None:
        optim_module.readout_temperature_raw = temperature_param
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
    )
    sample_viz_state = init_sample_viz_state(output_widget=viz_outputs.get("samples"))
    phase_viz_state = init_phase_viz_state(output_widget=viz_outputs.get("phase"))

    history = {
        "train_loss": [],
        "train_loss_cls": [],
        "train_loss_bin": [],
        "train_loss_margin": [],
        "train_loss_spill": [],
        "train_true_yes_score": [],
        "train_true_no_score": [],
        "train_true_margin": [],
        "train_max_negative_margin": [],
        "train_margin_gap": [],
        "train_bit_acc": [],
        "train_acc": [],
        "eval_loss": [],
        "eval_acc": [],
        "iters": [],
        "eval_iters": [],
    }

    extra_ckpt_tensors: Dict[str, torch.nn.Parameter] = {}
    if temperature_param is not None:
        extra_ckpt_tensors["readout_temperature_raw"] = temperature_param
    if bridge.gain_raw is not None:
        extra_ckpt_tensors["bridge_gain_raw"] = bridge.gain_raw
    checkpoint_extra_state: Dict[str, object] = {k: v for k, v in extra_ckpt_tensors.items() if v is not None}
    checkpoint_extra_state["binary_decision_mode"] = binary_mode
    if ecoc_codebook is not None:
        checkpoint_extra_state["ecoc_codebook_type"] = ECOC_CODEBOOK_TYPE
        checkpoint_extra_state["ecoc_codebook"] = ecoc_codebook

    resume_epoch, it_counter, best_eval = _load_checkpoint(
        checkpoint_layer_path,
        device,
        phase_system,
        history,
        extra_tensors=extra_ckpt_tensors,
        expected_ecoc_codebook=ecoc_codebook,
    )
    if len(history["train_loss_spill"]) < len(history["train_loss"]):
        missing = len(history["train_loss"]) - len(history["train_loss_spill"])
        history["train_loss_spill"] = [float("nan")] * missing + history["train_loss_spill"]
    if not checkpoint_layer_path.exists():
        _save_checkpoint(
            checkpoint_layer_path,
            resume_epoch,
            it_counter,
            phase_system,
            history,
            cfg,
            extra_state=checkpoint_extra_state,
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

    train_loss_buffer = _MetricBuffer(metrics_avg_every)
    train_acc_buffer = _MetricBuffer(metrics_avg_every)
    eval_loss_buffer = _MetricBuffer(metrics_update_every)
    eval_acc_buffer = _MetricBuffer(metrics_update_every)

    def _emit_train_metrics(force: bool = False) -> None:
        it_loss, avg_loss = train_loss_buffer.pop_average(force=force)
        it_acc, avg_acc = train_acc_buffer.pop_average(force=force)
        if avg_loss is None and avg_acc is None:
            return
        iter_for_plot = it_loss or it_acc or it_counter
        update_metrics_plot(
            metrics_plot_state,
            iter_for_plot,
            metrics_avg_every,
            train_loss=avg_loss,
            train_acc=avg_acc,
            force=force,
        )

    def _emit_eval_metrics(force: bool = False) -> None:
        it_loss, avg_loss = eval_loss_buffer.pop_average(force=force)
        it_acc, avg_acc = eval_acc_buffer.pop_average(force=force)
        if avg_loss is None and avg_acc is None:
            return
        iter_for_plot = it_loss or it_acc or it_counter
        update_metrics_plot(
            metrics_plot_state,
            iter_for_plot,
            metrics_avg_every,
            eval_loss=avg_loss,
            eval_acc=avg_acc,
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
    last_train_acc: Optional[float] = None
    latest_eval_loss: Optional[float] = None
    latest_eval_acc: Optional[float] = None
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
            postfix["batch"] = f"{idx}/{steps_per_epoch}" if steps_per_epoch else idx
        if last_train_loss is not None:
            postfix["loss"] = f"{last_train_loss:.4f}"
        if last_train_acc is not None:
            postfix["acc"] = f"{last_train_acc:.4f}"
        if scheduler is not None:
            lr_current = scheduler.get_last_lr()
            if lr_current:
                postfix["lr"] = f"{lr_current[0]:.2e}"
        if latest_eval_loss is not None:
            postfix["eval_loss"] = f"{latest_eval_loss:.4f}"
        if latest_eval_acc is not None:
            postfix["eval_acc"] = f"{latest_eval_acc:.4f}"
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
                x_batch = x_batch.to(device)
                labels = labels.to(device)
                x_phys = _prepare_binary_decision_input(
                    x_batch,
                    channel_num=channel_num,
                    binary_cfg=binary_cfg,
                    data_method=data_method,
                    target_hw=input_hw,
                    use_rot4_tiling=use_rot4_tiling,
                    device=device,
                )

                with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                    (
                        y_final,
                        layer_inputs,
                        layer_phases,
                        layer_outputs,
                        layer_inputs_pre,
                    ) = _forward_no_mix(
                        x_phys,
                        phase_system,
                        bridge,
                        channel_num=channel_num,
                        track_grads=True,
                    )
                    learned_temp = _positive_param(temperature_param) if temperature_param is not None else None
                    (
                        y_active,
                        yes_no_masks,
                        binary_scores_raw,
                        binary_scores,
                        class_logits,
                        margins_2d,
                    ) = _compute_binary_readout_details(
                        y_final,
                        binary_cfg,
                        readout_cfg,
                        learned_temp,
                        ecoc_codebook,
                    )
                    spillover_loss = _compute_spillover_loss(
                        y_active,
                        yes_no_masks,
                        eps=float(binary_cfg["eps"]),
                    )
                    total_loss, loss_cls, loss_bin, loss_margin, loss_spill = _compute_binary_losses(
                        class_logits,
                        labels,
                        binary_cfg,
                        margins_2d=margins_2d,
                        ecoc_codebook=ecoc_codebook,
                        spillover_loss=spillover_loss,
                    )

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
                    if scheduler is not None and scale_after >= scale_before:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                loss_val = float(total_loss.detach().cpu())
                loss_cls_val = float(loss_cls.detach().cpu())
                loss_bin_val = float(loss_bin.detach().cpu())
                loss_margin_val = float(loss_margin.detach().cpu())
                loss_spill_val = float(loss_spill.detach().cpu())
                pred = torch.argmax(class_logits.detach(), dim=1)
                acc = float((pred == labels).float().mean().item())
                stats = _binary_margin_stats(
                    class_logits,
                    binary_scores,
                    labels,
                    binary_cfg=binary_cfg,
                    margins_2d=margins_2d,
                    ecoc_codebook=ecoc_codebook,
                )

                history["train_loss"].append(loss_val)
                history["train_loss_cls"].append(loss_cls_val)
                history["train_loss_bin"].append(loss_bin_val)
                history["train_loss_margin"].append(loss_margin_val)
                history["train_loss_spill"].append(loss_spill_val)
                history["train_true_yes_score"].append(stats["true_yes_score"])
                history["train_true_no_score"].append(stats["true_no_score"])
                history["train_true_margin"].append(stats["true_margin"])
                history["train_max_negative_margin"].append(stats["max_negative_margin"])
                history["train_margin_gap"].append(stats["margin_gap"])
                history["train_bit_acc"].append(stats["bit_acc"])
                history["train_acc"].append(acc)
                history["iters"].append(it_counter)

                train_loss_buffer.add(loss_val, it_counter)
                train_acc_buffer.add(acc, it_counter)
                if train_loss_buffer.ready() and train_acc_buffer.ready():
                    _emit_train_metrics()

                if it_counter <= int(binary_cfg["debug_first_iters"]):
                    _debug_log_binary_iteration(
                        iteration=it_counter,
                        x_phys=x_phys,
                        y_final=y_final,
                        y_active=y_active,
                        binary_scores=binary_scores,
                        margins_2d=margins_2d,
                        class_logits=class_logits,
                        labels=labels,
                        stats=stats,
                        binary_cfg=binary_cfg,
                        channel_num=channel_num,
                        loss_cls=loss_cls,
                        acc=acc,
                        ecoc_codebook=ecoc_codebook,
                    )

                last_train_loss = loss_val
                last_train_acc = acc
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
                    eval_temp = _positive_param(temperature_param).detach() if temperature_param is not None else None
                    eval_loss, eval_acc = _evaluate(
                        val_loader,
                        device,
                        phase_system,
                        bridge,
                        readout_cfg,
                        binary_cfg,
                        channel_num,
                        data_method,
                        input_hw,
                        use_rot4_tiling,
                        temperature_param=eval_temp,
                        ecoc_codebook=ecoc_codebook,
                    )
                    history["eval_loss"].append(eval_loss)
                    history["eval_acc"].append(eval_acc)
                    history["eval_iters"].append(it_counter)
                    eval_loss_buffer.add(eval_loss, it_counter)
                    eval_acc_buffer.add(eval_acc, it_counter)
                    _emit_eval_metrics(force=True)
                    latest_eval_loss = eval_loss
                    latest_eval_acc = eval_acc
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
                                history,
                                cfg,
                                extra_state=checkpoint_extra_state,
                            )

                if (not ckpt_auto_mode) and ckpt_every and ckpt_every > 0 and it_counter % ckpt_every == 0:
                    _save_checkpoint(
                        checkpoint_layer_path,
                        epoch,
                        it_counter,
                        phase_system,
                        history,
                        cfg,
                        extra_state=checkpoint_extra_state,
                    )

                if viz_every > 0 and it_counter % viz_every == 0:
                    labels_cpu = labels[:1].detach().cpu()
                    out_channel_ids = list(range(min(3, channel_num)))
                    y_vis = y_final[out_channel_ids].detach().squeeze(1).unsqueeze(0)
                    if data_method == "patchify":
                        grid = int(round(math.sqrt(channel_num)))
                        encoded_vis = _reconstruct_full_input(
                            x_batch[:1],
                            grid=grid,
                            use_rot4_tiling=use_rot4_tiling,
                        ).detach()
                        input_grid_shape = (grid, grid)
                    else:
                        x_vis_img = x_batch[:1].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
                        if str(binary_cfg.get("input_encoding", "single")).strip().lower() == "rot4_tile":
                            encoded_vis = _rot4_tile_input_image(x_vis_img, input_hw).detach()
                        else:
                            encoded_vis = F.interpolate(
                                x_vis_img,
                                size=input_hw,
                                mode="bilinear",
                                align_corners=False,
                            ).detach()
                        input_grid_shape = (1, 1)
                    tile_masks_vis = build_binary_yes_no_masks(
                        tuple(y_final.shape[-2:]),
                        yes_region=str(binary_cfg["yes_region"]),
                        no_region=str(binary_cfg["no_region"]),
                        tile_hw=binary_cfg.get("yes_no_tile_hw"),
                        tile_margin=binary_cfg.get("yes_no_tile_margin", (0, 0)),
                        tile_gap=int(binary_cfg.get("yes_no_tile_gap", 0)),
                        device=y_vis.device,
                    )
                    if tile_masks_vis.device != y_vis.device:
                        tile_masks_vis = tile_masks_vis.to(y_vis.device)
                    preds_cpu = torch.argmax(class_logits, dim=1).detach().cpu()
                    tile_boxes = [
                        (-1, r0, c0, r1, c1)
                        for _, r0, c0, r1, c1 in compute_tile_boxes(tile_masks_vis)
                    ]
                    update_sample_viz(
                        sample_viz_state,
                        encoded_vis,
                        y_vis,
                        labels_cpu,
                        preds_cpu,
                        tile_boxes,
                        input_grid_shape=input_grid_shape,
                        output_channel_ids=out_channel_ids,
                    )

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
            history,
            cfg,
            extra_state=checkpoint_extra_state,
        )
    print(f"Training completed. Results saved to {run_dir}")


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a binary-decision optical neural network (ONN).")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_BINARY_CONFIG_PATH),
        help="Path to YAML config file (default: config_binary_decision.yaml)",
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
