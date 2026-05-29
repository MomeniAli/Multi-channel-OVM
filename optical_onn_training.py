"""
Training script for an Optical Neural Network (ONN) with a physical forward pass and
surrogate-based backward pass. This refactor keeps behaviour identical while organising
the code into reusable components that can be imported elsewhere.
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
    from .utils import build_mnist_loaders, encoding_x_phase_physical
    from .surrogate_model_training import make_optical_sys, reset_live_plot
    from .config_loader import load_training_config, resolve_onn_seed, write_config_snapshot
    from .phase_system import PhaseSystem, configure_phase_unit, phase_to_unit
    from .optical_bridge import OpticalBridge
    from .readout import TileMaskCache, apply_readout_scaling, compute_tile_scores, energy_margin_ratio_loss
    from .training_viz import (
        USE_NOTEBOOK_PROGRESS,
        NotebookProgressBar,
        ensure_dark_tqdm_theme,
        create_training_viz_layout,
        show_onn_training_summary_card,
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
    from utils import build_mnist_loaders, encoding_x_phase_physical
    from surrogate_model_training import make_optical_sys, reset_live_plot
    from config_loader import load_training_config, resolve_onn_seed, write_config_snapshot
    from phase_system import PhaseSystem, configure_phase_unit, phase_to_unit
    from optical_bridge import OpticalBridge
    from readout import TileMaskCache, apply_readout_scaling, compute_tile_scores, energy_margin_ratio_loss
    from training_viz import (
        USE_NOTEBOOK_PROGRESS,
        NotebookProgressBar,
        ensure_dark_tqdm_theme,
        create_training_viz_layout,
        show_onn_training_summary_card,
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _softplus_inverse(x: float, eps: float = 1e-6) -> float:
    """Inverse of softplus for positive initialisation."""
    x = max(float(x), eps)
    return float(math.log(math.expm1(x)))


def _positive_param(raw: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Maps an unconstrained parameter to positive domain."""
    return F.softplus(raw) + eps


def _prepare_data_loaders(data_cfg: Dict[str, object]):
    batch_size = int(data_cfg.get("batch_size", 64))
    augment = bool(data_cfg.get("augment", False))
    return build_mnist_loaders(batch_size=batch_size, augment=augment)


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
    extra_tensors: Dict[str, torch.nn.Parameter] | None = None,
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
    history: Dict[str, List[float]],
    cfg: Dict[str, object],
    extra_state: Dict[str, torch.Tensor] | None = None,
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
    if extra_state:
        for key, tensor in extra_state.items():
            if isinstance(tensor, torch.Tensor):
                ckpt_obj[key] = tensor.detach().cpu()
    torch.save(ckpt_obj, path)


def _evaluate(
    val_loader,
    device: torch.device,
    phase_system: PhaseSystem,
    bridge: OpticalBridge,
    tile_cache: TileMaskCache,
    readout_cfg: Dict[str, object],
    margin_cfg: Dict[str, float],
    expected_batch: int,
    *,
    temperature_param: torch.Tensor | None = None,
    channel_num: int | None = None,
    sigma_vals: torch.Tensor | None = None,
    group_size: int | None = None,
) -> Tuple[float, float, Optional[List[float]], Optional[float]]:
    eval_losses: List[float] = []
    eval_accuracies: List[float] = []
    eval_acc_per_channel_batches: List[np.ndarray] = []
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            labels_val = y_val.to(device)
            B_val = x_val.size(0)
            if B_val != expected_batch:
                raise ValueError(
                    f"Validation batch size {B_val} must equal expected {expected_batch} "
                    f"(group_size * channel_num)."
                )
            ch_num = channel_num
            if ch_num is None and sigma_vals is not None:
                ch_num = int(sigma_vals.numel())
            if ch_num is not None:
                ch_num = int(ch_num)
                if ch_num <= 0 or B_val % ch_num != 0:
                    raise ValueError(
                        f"Batch size {B_val} must be divisible by channel_num {ch_num} for eval."
                    )
                gs = group_size if group_size is not None else (B_val // ch_num)
            else:
                gs = None
            y_final = bridge.forward_eval(x_val, phase_system)
            output_hw = tuple(y_final.shape[-2:])
            tile_masks = tile_cache.get_masks(output_hw)
            if tile_masks.device != y_final.device:
                tile_masks = tile_masks.to(y_final.device)
            mask_area = tile_cache.get_mask_area(output_hw).unsqueeze(0)
            if mask_area.device != y_final.device:
                mask_area = mask_area.to(y_final.device)
            tile_scores = compute_tile_scores(y_final, tile_masks, mask_area)
            logits = apply_readout_scaling(
                tile_scores,
                readout_cfg["use_log_scaling"],
                readout_cfg["temperature"],
                readout_cfg["log_eps"],
                temperature_param=temperature_param,
                channel_num=ch_num,
            )
            if ch_num is not None and gs is not None and sigma_vals is not None:
                ce_loss_all = F.cross_entropy(logits, labels_val, reduction="none")
                ce_loss_grouped = ce_loss_all.view(gs, ch_num)
                channel_loss_terms = ce_loss_grouped.mean(dim=0)
                if margin_cfg["weight"] > 0:
                    tile_scores_grouped = tile_scores.view(gs, ch_num, -1)
                    labels_grouped = labels_val.view(gs, ch_num)
                    margin_losses: List[torch.Tensor] = []
                    for ch in range(ch_num):
                        aux_loss = energy_margin_ratio_loss(
                            tile_scores_grouped[:, ch, :],
                            labels_grouped[:, ch],
                            gap_margin=margin_cfg["gap"],
                            ratio_target=margin_cfg["ratio"],
                        )
                        margin_losses.append(aux_loss)
                    margin_loss_tensor = torch.stack(margin_losses)
                    channel_loss_terms = channel_loss_terms + margin_cfg["weight"] * margin_loss_tensor
                sigma_eval = sigma_vals.to(device)
                if sigma_eval.numel() == 1:
                    sigma_eval = sigma_eval.expand(ch_num)
                channel_weighted_loss = channel_loss_terms / (2.0 * sigma_eval.pow(2)) + torch.log(sigma_eval)
                total_loss = channel_weighted_loss.sum()
                loss_val = total_loss / ch_num
                eval_losses.append(float(loss_val.detach().cpu()))
            else:
                ce_loss = F.cross_entropy(logits, labels_val)
                if margin_cfg["weight"] > 0:
                    aux_loss = energy_margin_ratio_loss(
                        tile_scores,
                        labels_val,
                        gap_margin=margin_cfg["gap"],
                        ratio_target=margin_cfg["ratio"],
                    )
                    loss_val = ce_loss + margin_cfg["weight"] * aux_loss
                else:
                    loss_val = ce_loss
                eval_losses.append(float(loss_val.cpu()))
            preds_val = torch.argmax(logits, dim=1)
            acc_val = float((preds_val == labels_val).float().mean().item())
            eval_accuracies.append(acc_val)
            if ch_num is not None and gs is not None:
                preds_grouped = preds_val.reshape(gs, ch_num)
                labels_grouped = labels_val.reshape(gs, ch_num)
                acc_per_channel = (preds_grouped == labels_grouped).float().mean(dim=0)
                eval_acc_per_channel_batches.append(acc_per_channel.detach().cpu().numpy())
    eval_loss = float(np.mean(eval_losses)) if eval_losses else float("nan")
    eval_acc = float(np.mean(eval_accuracies)) if eval_accuracies else float("nan")
    eval_acc_per_channel = None
    eval_acc_channel_mean = None
    if eval_acc_per_channel_batches:
        acc_stack = np.stack(eval_acc_per_channel_batches, axis=0)
        eval_acc_per_channel = np.mean(acc_stack, axis=0).tolist()
        eval_acc_channel_mean = float(np.mean(eval_acc_per_channel))
    return eval_loss, eval_acc, eval_acc_per_channel, eval_acc_channel_mean


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
    batch_size = int(data_cfg.get("batch_size", channel_num))
    if batch_size % channel_num != 0:
        raise ValueError(f"data.batch_size {batch_size} must be divisible by channel_num {channel_num}.")
    group_size = batch_size // channel_num  # define G = B // channels
    expected_batch = batch_size
    data_cfg["batch_size"] = batch_size  # keep cfg aligned
    cfg["data"] = data_cfg
    cfg["group_size"] = group_size
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

    tiles_cfg = cfg.get("tiles", {})
    out_hw = _resolve_output_hw(tiles_cfg, enc_canvas_hw, optical_layer)
    tiles_cfg_for_cache = dict(tiles_cfg)
    tiles_cfg_for_cache["out_hw"] = out_hw
    tile_cache = TileMaskCache(tiles_cfg_for_cache, device, out_hw)

    n_layers = int(onn_cfg.get("n_layers", 3))
    phase_mask_shape_list = list(
        onn_cfg.get(
            "phase_mask_shape",
            [channel_num, enc_canvas_hw[0], enc_canvas_hw[1]],
        )
    )
    phase_mask_shape_list[0] = channel_num  # enforce consistency with channel_num
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
    margin_cfg = {
        "weight": float(onn_cfg.get("margin_loss_weight", 0.0)),
        "gap": float(onn_cfg.get("margin_loss_gap", 0.05)),
        "ratio": float(onn_cfg.get("margin_loss_ratio", 1.5)),
    }
    structural_nonlinearity = bool(onn_cfg.get("structural_nonlinearity", False))
    alpha_init = float(onn_cfg.get("alpha_init", 1.0))
    alpha_max = float(onn_cfg.get("alpha_max", 1.0))

    phase_proc_cfg = cfg.get("phase_processing", {})
    configure_phase_unit(
        normalize_before=bool(phase_proc_cfg.get("normalize_before", False)),
        norm_eps=float(phase_proc_cfg.get("norm_eps", 1e-6)),
        squash_mode=str(phase_proc_cfg.get("squash", "tanh")),
        temperature=float(phase_proc_cfg.get("temperature", 1.0)),
        smooth_kernel=int(phase_proc_cfg.get("smooth_kernel", 0)),
    )

    readout_cfg = {
        "use_log_scaling": bool(cfg.get("readout", {}).get("use_log_scaling", True)),
        "temperature": float(cfg.get("readout", {}).get("temperature", 1.0)),
        "log_eps": float(cfg.get("readout", {}).get("log_eps", 1e-6)),
        "temperature_learnable": bool(cfg.get("readout", {}).get("temperature_learnable", False)),
        "temperature_per_channel": bool(cfg.get("readout", {}).get("temperature_per_channel", False)),
    }
    if readout_cfg["temperature"] <= 0:
        raise ValueError("readout.temperature must be positive")
    sigma_init = float(onn_cfg.get("channel_sigma_init", 1.0))
    if sigma_init <= 0:
        raise ValueError("onn.channel_sigma_init must be positive")

    max_epochs = int(onn_cfg.get("max_epochs", 10))
    eval_every = int(onn_cfg.get("eval_every", 200))
    ckpt_auto_mode, ckpt_every = _parse_ckpt_every(onn_cfg.get("ckpt_every", 1000))
    viz_every = int(onn_cfg.get("viz_every", 200))
    phase_viz_every = max(0, int(onn_cfg.get("phase_viz_every", viz_every)))
    metrics_update_every = max(1, int(onn_cfg.get("metrics_viz_every", 5)))

    optical_default_dir = PROJECT_ROOT / "pre_trained_model_save" / "Optical_neural_net"
    run_dir = Path(onn_cfg.get("save_dir", str(optical_default_dir)))
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_layer_path = checkpoint_dir / f"_mul_{n_layers}.pt"

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
        alpha_map_hw=tuple(onn_cfg.get("alpha_map_hw", [])) if onn_cfg.get("alpha_map_hw", None) else None,
    ).to(device)
    if phase_system.num_channels != channel_num:
        raise ValueError(
            f"Configured channel_num={channel_num} but PhaseSystem uses {phase_system.num_channels}; "
            "ensure phase_mask_shape[0] matches channel_num."
        )
    temperature_param: torch.Tensor | None = None
    if readout_cfg["temperature_learnable"]:
        temp_shape = (channel_num,) if readout_cfg["temperature_per_channel"] else (1,)
        temp_raw_init = _softplus_inverse(readout_cfg["temperature"])
        temperature_param = nn.Parameter(torch.full(temp_shape, temp_raw_init, device=device))
    channel_sigma_raw = nn.Parameter(torch.full((channel_num,), _softplus_inverse(sigma_init), device=device))

    modes = ["tile"] + ["interp"] * (n_layers - 1)
    x_norm_modes = ["percentile"] + ["percentile"] * (n_layers - 1)
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
    optim_module.channel_sigma_raw = channel_sigma_raw
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
        "eval_loss": [],
        "iters": [],
        "train_acc": [],
        "eval_acc": [],
        "eval_iters": [],
        "train_acc_channels": [],
        "eval_acc_channels": [],
        "train_acc_channel_mean": [],
        "eval_acc_channel_mean": [],
    }
    extra_ckpt_tensors: Dict[str, torch.nn.Parameter] = {"channel_sigma_raw": channel_sigma_raw}
    if temperature_param is not None:
        extra_ckpt_tensors["readout_temperature_raw"] = temperature_param
    if bridge.gain_raw is not None:
        extra_ckpt_tensors["bridge_gain_raw"] = bridge.gain_raw
    checkpoint_extra_state = {k: v for k, v in extra_ckpt_tensors.items() if v is not None}
    resume_epoch, it_counter, best_eval = _load_checkpoint(
        checkpoint_layer_path,
        device,
        phase_system,
        history,
        extra_tensors=extra_ckpt_tensors,
    )
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

    train_loss_buffer = _MetricBuffer(metrics_update_every)
    train_acc_buffer = _MetricBuffer(metrics_update_every)
    eval_loss_buffer = _MetricBuffer(metrics_update_every)
    eval_acc_buffer = _MetricBuffer(metrics_update_every)
    train_acc_channel_buffers = [_MetricBuffer(metrics_update_every) for _ in range(channel_num)]
    last_eval_acc_channels: Optional[List[float]] = None

    def _compute_channel_metrics(
        logits_batch: torch.Tensor,
        labels_batch: torch.Tensor,
    ) -> List[float]:
        logits_grouped = logits_batch.reshape(group_size, channel_num, -1)
        labels_grouped = labels_batch.reshape(group_size, channel_num)
        acc_vals: List[float] = []
        for ch in range(channel_num):
            logits_ch = logits_grouped[:, ch, :]
            labels_ch = labels_grouped[:, ch]
            acc_ch = (logits_ch.argmax(dim=1) == labels_ch).float().mean()
            acc_vals.append(float(acc_ch.detach().cpu()))
        return acc_vals

    def _emit_train_metrics(force: bool = False) -> None:
        it_loss, avg_loss = train_loss_buffer.pop_average(force=force)
        it_acc, avg_acc = train_acc_buffer.pop_average(force=force)
        if avg_loss is None and avg_acc is None:
            return
        iter_for_plot = it_loss or it_acc or it_counter
        channel_accs: Optional[List[float]] = None
        if train_acc_channel_buffers:
            channel_accs = []
            for buf in train_acc_channel_buffers:
                _, avg_val = buf.pop_average(force=force)
                channel_accs.append(float("nan") if avg_val is None else float(avg_val))
        update_metrics_plot(
            metrics_plot_state,
            iter_for_plot,
            metrics_update_every,
            train_loss=avg_loss,
            train_acc=avg_acc,
            train_acc_per_channel=channel_accs,
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
            metrics_update_every,
            eval_loss=avg_loss,
            eval_acc=avg_acc,
            eval_acc_per_channel=last_eval_acc_channels,
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
            if steps_per_epoch is not None and steps_per_epoch > 0:
                postfix["batch"] = f"{idx}/{steps_per_epoch}"
            else:
                postfix["batch"] = idx
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
                B_batch = x_batch.size(0)
                if B_batch != expected_batch:
                    raise ValueError(
                        f"Batch size {B_batch} must equal expected {expected_batch} "
                        f"(group_size {group_size} * channel_num {channel_num})."
                    )

                with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                    (
                        y_final,
                        layer_inputs,
                        layer_phases,
                        layer_outputs,
                        layer_inputs_pre,
                    ) = bridge.run(x_batch, phase_system)
                    output_hw = tuple(y_final.shape[-2:])
                    tile_masks = tile_cache.get_masks(output_hw)
                    if tile_masks.device != y_final.device:
                        tile_masks = tile_masks.to(y_final.device)
                    mask_area = tile_cache.get_mask_area(output_hw).unsqueeze(0)
                    if mask_area.device != y_final.device:
                        mask_area = mask_area.to(y_final.device)
                    tile_scores = compute_tile_scores(y_final, tile_masks, mask_area)
                    learned_temp = _positive_param(temperature_param) if temperature_param is not None else None
                    logits = apply_readout_scaling(
                        tile_scores,
                        readout_cfg["use_log_scaling"],
                        readout_cfg["temperature"],
                        readout_cfg["log_eps"],
                        temperature_param=learned_temp,
                        channel_num=channel_num,
                    )
                    ce_loss_all = F.cross_entropy(logits, labels, reduction="none")
                    ce_loss_grouped = ce_loss_all.view(group_size, channel_num)
                    channel_loss_terms = ce_loss_grouped.mean(dim=0)
                    if margin_cfg["weight"] > 0:
                        tile_scores_grouped = tile_scores.view(group_size, channel_num, -1)
                        labels_grouped = labels.view(group_size, channel_num)
                        margin_losses: List[torch.Tensor] = []
                        for ch in range(channel_num):
                            aux_loss = energy_margin_ratio_loss(
                                tile_scores_grouped[:, ch, :],
                                labels_grouped[:, ch],
                                gap_margin=margin_cfg["gap"],
                                ratio_target=margin_cfg["ratio"],
                            )
                            margin_losses.append(aux_loss)
                        margin_loss_tensor = torch.stack(margin_losses)
                        channel_loss_terms = channel_loss_terms + margin_cfg["weight"] * margin_loss_tensor

                    # Encourage phase masks across channels to stay similar.
                    reg_loss = torch.tensor(0.0, device=device)
                    for phi in phase_system.phase_parameters():
                        if not isinstance(phi, torch.Tensor):
                            continue
                        if phi.dim() < 3 or phi.size(0) <= 1:
                            continue
                        phi_mean = phi.mean(dim=0, keepdim=True)
                        reg_loss = reg_loss + ((phi - phi_mean) ** 2).mean()
                    sigma_vals = _positive_param(channel_sigma_raw)
                    channel_weighted_loss = channel_loss_terms / (2.0 * sigma_vals.pow(2)) + torch.log(sigma_vals)
                    total_loss = channel_weighted_loss.sum() + 0 * reg_loss

                loss_full = total_loss
                loss = loss_full / float(accum_steps)
                scaler.scale(loss).backward()

                is_accum_step = (it_counter % accum_steps == 0)
                is_last_in_epoch = steps_per_epoch is not None and batch_idx == steps_per_epoch
                if is_accum_step or is_last_in_epoch:
                    if grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(phase_system.parameters(), grad_clip_norm)
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = scaler.get_scale()
                    if scheduler is not None:
                        # Skip LR step if AMP skipped optimizer step.
                        if scale_after >= scale_before:
                            scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                loss_val = float((loss_full.detach() / channel_num).cpu())
                preds_train = torch.argmax(logits.detach(), dim=1)
                acc_train = float((preds_train == labels).float().mean().item())
                channel_accs = _compute_channel_metrics(
                    logits.detach(),
                    labels,
                )
                channel_acc_mean = (
                    float(sum(channel_accs) / len(channel_accs)) if channel_accs else float("nan")
                )
                history["train_loss"].append(loss_val)
                history["train_acc"].append(acc_train)
                history["train_acc_channels"].append(channel_accs)
                history["train_acc_channel_mean"].append(channel_acc_mean)
                history["iters"].append(it_counter)
                train_loss_buffer.add(loss_val, it_counter)
                train_acc_buffer.add(acc_train, it_counter)
                for ch_idx, acc_ch in enumerate(channel_accs):
                    train_acc_channel_buffers[ch_idx].add(acc_ch, it_counter)
                if train_loss_buffer.ready() and train_acc_buffer.ready():
                    _emit_train_metrics()
                last_train_loss = loss_val
                last_train_acc = acc_train
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
                    eval_loss, eval_acc, eval_acc_channels, eval_acc_channel_mean = _evaluate(
                        val_loader,
                        device,
                        phase_system,
                        bridge,
                        tile_cache,
                        readout_cfg,
                        margin_cfg,
                        expected_batch,
                        temperature_param=_positive_param(temperature_param).detach()
                        if temperature_param is not None
                        else None,
                        channel_num=channel_num,
                        sigma_vals=_positive_param(channel_sigma_raw).detach(),
                        group_size=group_size,
                    )
                    if eval_acc_channels is None:
                        eval_acc_channels = [float("nan")] * channel_num
                    if eval_acc_channel_mean is None:
                        eval_acc_channel_mean = float("nan")
                    history["eval_loss"].append(eval_loss)
                    history["eval_acc"].append(eval_acc)
                    history["eval_iters"].append(it_counter)
                    history["eval_acc_channels"].append(eval_acc_channels)
                    history["eval_acc_channel_mean"].append(eval_acc_channel_mean)
                    eval_loss_buffer.add(eval_loss, it_counter)
                    eval_acc_buffer.add(eval_acc, it_counter)
                    last_eval_acc_channels = eval_acc_channels
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
                    num_examples = min(2, x_batch.size(0))
                    labels_cpu = labels[:num_examples].detach().cpu()
                    y_vis = y_final[:num_examples].detach()
                    encoded_samples = bridge.encoded_cache[0]
                    if encoded_samples is not None:
                        encoded_vis = encoded_samples[:num_examples].detach()
                    else:
                        x_canvas_for_phase = None
                        if structural_nonlinearity:
                            dummy_phase = torch.zeros(
                                num_examples, 1, *enc_canvas_hw, device=x_batch.device, dtype=x_batch.dtype
                            )
                            x_canvas_for_phase, _ = encoding_x_phase_physical(
                                x_batch[:num_examples],
                                dummy_phase,
                                canvas_hw=enc_canvas_hw,
                                x_mode=modes[0],
                                x_norm_mod=x_norm_modes[0],
                            )
                        first_phase = phase_system.expand_to_batch(
                            phase_system.phase_parameters()[0],
                            num_examples,
                            x_input=x_canvas_for_phase,
                            layer_idx=0,
                        ).detach()
                        encoded_vis, _ = encoding_x_phase_physical(
                            x_batch[:num_examples],
                            first_phase,
                            canvas_hw=enc_canvas_hw,
                            x_mode=modes[0],
                        )
                    tile_masks_vis = tile_cache.get_masks(tuple(y_vis.shape[-2:]))
                    if tile_masks_vis.device != y_vis.device:
                        tile_masks_vis = tile_masks_vis.to(y_vis.device)
                    mask_area_vis = tile_cache.get_mask_area(tuple(y_vis.shape[-2:])).unsqueeze(0)
                    if mask_area_vis.device != y_vis.device:
                        mask_area_vis = mask_area_vis.to(y_vis.device)
                    tile_scores_vis = compute_tile_scores(y_vis, tile_masks_vis, mask_area_vis)
                    temp_for_viz = None
                    if temperature_param is not None and tile_scores_vis.size(0) % channel_num == 0:
                        temp_for_viz = _positive_param(temperature_param).detach()
                    logits_batch = apply_readout_scaling(
                        tile_scores_vis,
                        readout_cfg["use_log_scaling"],
                        readout_cfg["temperature"],
                        readout_cfg["log_eps"],
                        temperature_param=temp_for_viz,
                        channel_num=channel_num if temp_for_viz is not None else None,
                    )
                    preds_cpu = torch.argmax(logits_batch, dim=1).detach().cpu()
                    tile_boxes = compute_tile_boxes(tile_masks_vis)
                    update_sample_viz(
                        sample_viz_state,
                        encoded_vis,
                        y_vis,
                        labels_cpu,
                        preds_cpu,
                        tile_boxes,
                    )

                if phase_viz_every > 0 and it_counter % phase_viz_every == 0:
                    # Prefer the last encoded phase (post encoding_x_phase_physical) if it is cached.
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
                                pass  # fall back to phase_to_unit(phase_param) if reshape fails
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
    parser = argparse.ArgumentParser(description="Train an optical neural network (ONN).")
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
