"""
Training for surrogate optical neural networks.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict
from collections import deque

from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from pytorch_msssim import ssim, ms_ssim
from focal_frequency_loss import FocalFrequencyLoss as FFL

ffl= FFL(
    loss_weight=1.0,  # internal scaling
    alpha=1.0,        # scale for weights
    patch_factor=1,   # 1 = full image FFL
    ave_spectrum=False,
    log_matrix=False,
    batch_matrix=False,
)

from .utils import (
    generate_phase_mask,
    encoding_x_phase_physical,
    percentile_scale_spatial,
    channel_relative_mse,
)
from .training_viz import (
    NotebookProgressBar,
    USE_NOTEBOOK_PROGRESS,
    ensure_dark_tqdm_theme,
)
from .config_loader import load_training_config
import tempfile


def _atomic_save(obj: Dict[str, object], path: Path) -> None:
    """Atomic torch.save to avoid partial/corrupt checkpoints."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as f:
        tmp = Path(f.name)
    try:
        torch.save(obj, tmp)
        try:
            tmp.chmod(0o644)
        except Exception:
            pass
        tmp.replace(path)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            raise

def _count_params(m: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


# -----------------------------------------------------------------------------
# Simple notebook visualisations (loss curve + one example)
# -----------------------------------------------------------------------------

CARD_BG = "#1e1e1e"
TEXT_PRIMARY = "#f5f7ff"
PHASE_CMAP = "magma"

_LOSS_PLOT_STATE: Dict[str, object] = {}
_SAMPLE_PLOT_STATE: Dict[str, object] = {}
_SURROGATE_SECTION_CREATED = False
_SURROGATE_CSS_INJECTED = False


def _ensure_surrogate_section() -> None:
    global _SURROGATE_SECTION_CREATED, _SURROGATE_CSS_INJECTED
    if not _SURROGATE_CSS_INJECTED:
        display(
            HTML(
                f"""
                <style>
                    .jp-OutputArea,
                    .jp-OutputArea-output,
                    .output_wrapper,
                    .output_area,
                    .jp-Cell-outputWrapper,
                    .output_subarea,
                    .jp-OutputArea-child {{
                        background-color: {CARD_BG} !important;
                        border: none !important;
                        box-shadow: none !important;
                    }}
                </style>
                """
            )
        )
        _SURROGATE_CSS_INJECTED = True
    if _SURROGATE_SECTION_CREATED:
        return
    display(
        HTML(
            f"""
            <div style="margin:16px 0 18px 0; padding:20px 24px; border-radius:12px; background:{CARD_BG}; box-shadow:0 18px 34px rgba(2,4,10,0.45); text-align:center;">
                <h2 style="margin:0; font-size:24px; color:{TEXT_PRIMARY};">Surrogate Model Training</h2>
            </div>
            """
        )
    )
    _SURROGATE_SECTION_CREATED = True


def _ensure_loss_plot() -> Dict[str, object]:
    _ensure_surrogate_section()
    if _LOSS_PLOT_STATE:
        return _LOSS_PLOT_STATE
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.2))
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(CARD_BG)
    ax.set_title("Surrogate Training Loss", color=TEXT_PRIMARY)
    ax.set_xlabel("Iteration", color=TEXT_PRIMARY)
    ax.set_ylabel("Loss values", color=TEXT_PRIMARY)
    ax.tick_params(colors=TEXT_PRIMARY)
    ax.grid(True, alpha=0.35, color="#2e394f")
    (line,) = ax.plot(
        [],
        [],
        label="train",
        color="#4dd0e1",
        linewidth=1.8,
        marker="o",
        markersize=3,
    )
    ax.set_ylim(0.0, 1.0)
    leg = ax.legend(loc="upper right")
    if leg:
        for text in leg.get_texts():
            text.set_color(TEXT_PRIMARY)
            text.set_fontsize(11)
            text.set_fontweight("bold")
        frame = leg.get_frame()
        frame.set_facecolor(CARD_BG)
        frame.set_edgecolor("#2e394f")
    fig.tight_layout()
    handle = display(fig, display_id=True)
    state = {"fig": fig, "ax": ax, "line": line, "handle": handle}
    _LOSS_PLOT_STATE.update(state)
    return state


def _update_loss_plot(iters: List[int], losses: List[float]) -> None:
    if not iters or not losses:
        return
    finite_pairs = [(i, l) for i, l in zip(iters, losses) if np.isfinite(l)]
    if not finite_pairs:
        return
    xs, ys = zip(*finite_pairs)
    state = _ensure_loss_plot()
    line = state["line"]
    ax = state["ax"]
    fig = state["fig"]
    handle = state.get("handle")
    line.set_data(xs, ys)
    ax.relim()
    ax.autoscale_view(scalex=True, scaley=False)
    ax.set_ylim(0.0, 1.0)
    fig.canvas.draw_idle()
    if handle is not None and hasattr(handle, "update"):
        try:
            handle.update(fig)
        except Exception:
            pass


def _ensure_sample_plot() -> Dict[str, object]:
    # Backward-compatible wrapper kept for older callers.
    return _ensure_sample_plot_multi(num_samples=1)


def _ensure_sample_plot_multi(num_samples: int) -> Dict[str, object]:
    _ensure_surrogate_section()
    state = _SAMPLE_PLOT_STATE
    if state and state.get("num_samples") == num_samples:
        return state
    # Rebuild figure if sample count changed
    old_fig = state.get("fig") if state else None
    if old_fig is not None:
        try:
            plt.close(old_fig)
        except Exception:
            pass
    fig, axes = plt.subplots(num_samples, 4, figsize=(14, 3.2 * num_samples))
    fig.patch.set_facecolor(CARD_BG)
    axes = np.array(axes).reshape(num_samples, 4)
    titles = ["Input", "Phase", "Ground Truth", "Surrogate Pred"]
    images: List[List[object]] = []
    colorbars: List[List[object]] = []
    blank = np.zeros((16, 16), dtype=np.float32)
    for row in range(num_samples):
        row_images = []
        row_cbars = []
        for col, title in enumerate(titles):
            ax = axes[row, col]
            ax.set_facecolor(CARD_BG)
            cmap = "gray" if "Input" in title else PHASE_CMAP if title == "Phase" else "magma"
            im = ax.imshow(blank, cmap=cmap, vmin=0.0, vmax=1.0)
            ax.set_title(title, color=TEXT_PRIMARY)
            ax.axis("off")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_facecolor(CARD_BG)
            cbar.ax.tick_params(labelsize=8, colors=TEXT_PRIMARY)
            row_images.append(im)
            row_cbars.append(cbar)
        images.append(row_images)
        colorbars.append(row_cbars)
    fig.tight_layout()
    handle = display(fig, display_id=True)
    new_state = {
        "fig": fig,
        "axes": axes,
        "images": images,
        "colorbars": colorbars,
        "handle": handle,
        "num_samples": num_samples,
    }
    _SAMPLE_PLOT_STATE.clear()
    _SAMPLE_PLOT_STATE.update(new_state)
    return new_state


def _update_sample_plot(
    x_exp: torch.Tensor,
    phase_exp: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    sample_indices: Tuple[int, ...] = (0,),
) -> None:
    num_samples = max(1, len(sample_indices))
    state = _ensure_sample_plot_multi(num_samples)
    fig = state["fig"]
    images = state["images"]
    colorbars = state["colorbars"]
    handle = state["handle"]

    tensors = [
        (x_exp, "Input"),
        (phase_exp, "Phase"),
        (target, "Ground Truth"),
        (pred, "Surrogate Pred"),
    ]

    def _to_numpy(t: torch.Tensor) -> np.ndarray:
        if t.dim() == 4:
            t = t[0]
        if t.dim() == 3:
            t = t[0]
        arr = t.detach().cpu().numpy()
        if arr.ndim == 0:
            arr = np.array([[arr]])
        return arr

    for row_idx, b_idx in enumerate(sample_indices):
        if b_idx >= x_exp.size(0):
            continue
        for (tensor, title), im, cbar in zip(tensors, images[row_idx], colorbars[row_idx]):
            if tensor.dim() == 4:
                tensor_sel = tensor[b_idx : b_idx + 1]
            else:
                tensor_sel = tensor
            arr = _to_numpy(tensor_sel)
            im.set_data(arr)
            vmin = float(np.min(arr))
            vmax = float(np.max(arr))
            if vmin == vmax:
                vmax = vmin + 1e-6
            im.set_clim(vmin, vmax)
            im.axes.set_title(f"{title} [idx {b_idx}]", color=TEXT_PRIMARY)
            cbar.update_normal(im)
            cbar.ax.tick_params(colors=TEXT_PRIMARY)
    fig.canvas.draw_idle()
    if handle is not None and hasattr(handle, "update"):
        try:
            handle.update(fig)
        except Exception:
            pass


def reset_live_plot() -> None:
    """Close/clear surrogate training figures."""
    global _LOSS_PLOT_STATE, _SAMPLE_PLOT_STATE, _SURROGATE_SECTION_CREATED
    for state in (_LOSS_PLOT_STATE, _SAMPLE_PLOT_STATE):
        fig = state.get("fig") if state else None
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass
    _LOSS_PLOT_STATE = {}
    _SAMPLE_PLOT_STATE = {}
    _SURROGATE_SECTION_CREATED = False


# -----------------------------------------------------------------------------
# physical_optical_sys
# -----------------------------------------------------------------------------
def make_optical_sys(layer, multi_shot_avg: int = 1, *, clamp_output: bool = True):
    def _resolve_device(layer, x: torch.Tensor) -> torch.device:
        # 1) nn.Module (incl. DDP .module)
        mod = layer
        if isinstance(layer, nn.Module) and hasattr(layer, "parameters"):
            for p in layer.parameters():
                return p.device
            for b in layer.buffers():
                return b.device
        if hasattr(layer, "module") and isinstance(layer.module, nn.Module):
            mod = layer.module
            for p in mod.parameters():
                return p.device
            for b in mod.buffers():
                return b.device
        # 2) Custom attribute
        for attr in ("device", "dev", "_device"):
            d = getattr(layer, attr, None)
            if isinstance(d, torch.device):
                return d
        # 3) Fallback: stick to x's device
        return x.device

    @torch.no_grad()
    def _optical_sys(x: torch.Tensor, phase_mask: torch.Tensor) -> torch.Tensor:
        shots = max(1, int(multi_shot_avg))
        # ensure 4D
        if x.dim() == 3: x = x.unsqueeze(0)
        if phase_mask.dim() == 3: phase_mask = phase_mask.unsqueeze(0)
        if x.dim() != 4 or phase_mask.dim() != 4:
            raise ValueError("x and phase_mask must be 4D (B,C,H,W) or 3D (C,H,W).")

        # match spatial sizes
        Ht, Wt = phase_mask.shape[-2:]
        if x.shape[-2:] != (Ht, Wt):
            x = F.interpolate(x, (Ht, Wt), mode="bilinear", align_corners=False)
        if phase_mask.shape[-2:] != (Ht, Wt):
            phase_mask = F.interpolate(phase_mask, (Ht, Wt), mode="nearest")

        # batch align (allow 1->B broadcast)
        Bx, Bp = x.size(0), phase_mask.size(0)
        if Bx != Bp:
            if Bp == 1:
                phase_mask = phase_mask.expand(Bx, -1, -1, -1)
            elif Bx == 1:
                x = x.expand(Bp, -1, -1, -1)
            else:
                raise ValueError(f"Batch mismatch: x.B={Bx}, phase.B={Bp}")

        # pick device robustly
        dev = _resolve_device(layer, x)
        x, phase_mask = x.to(dev), phase_mask.to(dev)

        def _single_pass() -> torch.Tensor:
            if hasattr(layer, "forward"):
                out_local = layer.forward(x, phase_mask)
            else:
                out_local = layer(x, phase_mask)
            if out_local.dim() == 5 and out_local.size(2) == 1:
                out_local = out_local.squeeze(2)
            if clamp_output:
                return percentile_scale_spatial(torch.clamp(out_local, 0.0, 1.0))
            return out_local
        if shots == 1:
            return _single_pass()

        acc = None
        for _ in range(shots):
            out_local = _single_pass()
            acc = out_local if acc is None else (acc + out_local)
        return acc / float(shots)
    return _optical_sys




# -----------------------------------------------------------------------------
# Training function
# -----------------------------------------------------------------------------
@dataclass
class TrainHistory:
    train_loss: List[float]
    eval_loss: List[float]
    train_iters: List[int]
    eval_iters: List[int]

def train_optical(
    model: nn.Module,
    layer: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    device: Union[str, torch.device] = "cuda",
    epochs: int = 1,
    max_iters: Optional[int] = None,
    eval_every: int = 20,  # kept for API compatibility; ignored
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    multi_shot_avg: int = 1,
    plot_live: bool = True,          # controls simple loss plot
    run_name: Optional[str] = None,
    ckpt_dir: Optional[str] = "./checkpoints",
    save_best: bool = True,
    resume_from: Optional[str] = None,
    phase_mask_dim: Tuple[int, int] = (42, 42),
    phase_mask_sigma: float = 2.0,
    HW_input_limit: Optional[Tuple[int, int]] = None,
    visualize_samples: bool = True,
    visualize_every: Optional[int] = None,  # if None: only at the end
    viz_sample_indices: Tuple[int, ...] = (1, 7, 13),
    ckpt_every: int = 5000,
    quiet_mode: bool = False,
    group_channels: bool = True,
    group_channels_first_only: bool = True,
    group_channels_alternate: bool = True,
    num_channels: int = 16,
    group_size: Optional[int] = None,  # if None: inferred as batch_size // num_channels
    grad_clip_norm: Optional[float] = None,
    accum_steps: Optional[int] = None,
) -> TrainHistory:
    device = torch.device(device)
    model = model.to(device)
    num_channels = int(getattr(model, "num_channels", num_channels))
    group_size = int(group_size) if group_size is not None else None
    total_params, trainable_params = _count_params(model)
    #if not quiet_mode:
    #    print(f"[params] total={total_params:,} trainable={trainable_params:,}")
    model.train()
    reset_live_plot()
    _ensure_surrogate_section()
    optical_sys = make_optical_sys(layer, multi_shot_avg=multi_shot_avg)

    # canvas
    if HW_input_limit is not None:
        canvas_hw = tuple(HW_input_limit)
    elif hasattr(model, 'enc_canvas_hw'):
        canvas_hw = tuple(getattr(model, 'enc_canvas_hw'))
    else:
        raise ValueError("HW_input_limit must be provided if model lacks 'enc_canvas_hw'.")
    if (canvas_hw[0] % 8 != 0) or (canvas_hw[1] % 8 != 0):
        raise ValueError(f"canvas_hw {canvas_hw} must be divisible by 8.")

    if HW_input_limit is not None and hasattr(model, 'enc_canvas_hw'):
        model.enc_canvas_hw = canvas_hw

    phase_mask_dim = tuple(phase_mask_dim)
    viz_interval = max(1, int(visualize_every)) if (visualize_every is not None) else None
    viz_indices = tuple(int(i) for i in viz_sample_indices) if viz_sample_indices else (0,)
    viz_indices = tuple(i for i in viz_indices if i >= 0) or (0,)
    max_viz_idx = max(viz_indices)

    def _draw_random_gain(batch_size: int, dtype: torch.dtype) -> torch.Tensor:
        """Per-sample random gain in [0.3, 1.0] to diversify amplitude."""
        return torch.empty((batch_size, 1, 1, 1), device=device, dtype=dtype).uniform_(0.3, 1.0)

    # optim & loss
    if accum_steps is None:
        try:
            cfg, _ = load_training_config()
            pre_cfg = cfg.get("pre_training_surrogate_model", {})
            accum_steps = pre_cfg.get("accum_steps", 8)
        except Exception:
            accum_steps = 8

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    mse_loss = nn.MSELoss()
    accum_steps = max(1, int(accum_steps))

    # simple histories
    train_loss_hist: List[float] = []
    train_iters: List[int] = []
    eval_loss_hist: List[float] = []   # kept for API compatibility (remains empty)
    eval_iters: List[int] = []         # kept for API compatibility (remains empty)
    mse_window = deque(maxlen=50)
    last_avg_mse50: Optional[float] = None

    best_mse: float = float('inf')
    current_iter: int = 0

    # checkpoint dirs
    model_name = model.__class__.__name__
    ckpt_root = None
    if ckpt_dir is not None:
        module_dir = Path(__file__).resolve().parent
        ckpt_root = Path(ckpt_dir)
        if not ckpt_root.is_absolute():
            ckpt_root = module_dir / ckpt_root
        ckpt_root.mkdir(parents=True, exist_ok=True)

    if ckpt_root is not None:
        ckpt_subdir = ckpt_root / model_name / (run_name or "")
        ckpt_subdir.mkdir(parents=True, exist_ok=True)
        last_path = ckpt_subdir / "ckpt_last.pt"
        best_path = ckpt_subdir / "ckpt_best.pt"
    else:
        last_path = None
        best_path = None

    # resume
    start_epoch_from_ckpt = 0
    start_iter_from_ckpt = 0
    if resume_from is not None and ckpt_root is not None:
        target = None
        if resume_from.lower() == 'last' and last_path is not None:
            target = last_path
        elif resume_from.lower() == 'best' and best_path is not None:
            target = best_path
        if target is not None and target.exists():
            try:
                try:
                    ckpt = torch.load(target, map_location=device, weights_only=True)
                except TypeError:
                    ckpt = torch.load(target, map_location=device)

                if isinstance(ckpt, dict) and "model_state" in ckpt:
                    model.load_state_dict(ckpt["model_state"])
                    if "opt_state" in ckpt:
                        opt.load_state_dict(ckpt["opt_state"])
                    if "scaler_state" in ckpt:
                        scaler.load_state_dict(ckpt["scaler_state"])
                    best_mse = float(ckpt.get("best_mse", ckpt.get("best_train_loss", best_mse)))
                else:
                    model.load_state_dict(ckpt)
                start_epoch_from_ckpt = int(ckpt.get("epoch", 0))
                start_iter_from_ckpt  = int(ckpt.get("iter", 0))
                current_iter = start_iter_from_ckpt
            except Exception as exc:
                if not quiet_mode:
                    print(f"[resume] Failed to load checkpoint from {target}: {exc}")

    # main loop
    try:
        steps_per_epoch = len(train_loader)
    except TypeError:
        steps_per_epoch = None

    def _build_channel_groups(
        x_base: torch.Tensor,
        phase_base: torch.Tensor,
        target_groups: Optional[int] = None,
        *,
        group_channels: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Expand a batch of size G to grouped batches of size (G * num_channels),
        repeating each (x, phase) pair across all channels in order.
        If target_groups is provided, tile/truncate the base batch to that count.
        If group_channels is False, return the inputs without channel grouping,
        only aligning batch sizes if needed.
        """
        B_in = x_base.size(0)
        if not group_channels:
            if phase_base.size(0) == 1 and B_in > 1:
                phase_base = phase_base.expand(B_in, -1, -1, -1)
            elif phase_base.size(0) != B_in:
                raise ValueError(f"Batch mismatch: x.B={B_in}, phase.B={phase_base.size(0)}")
            return x_base, phase_base, B_in

        groups = max(1, int(target_groups)) if target_groups is not None else group_size

        # Tile/truncate to exactly `groups`
        idx = torch.arange(groups, device=x_base.device) % B_in
        x_sel = x_base[idx]
        phase_sel = phase_base[idx]
        x_rep = x_sel.repeat_interleave(num_channels, dim=0)
        phase_rep = phase_sel.repeat_interleave(num_channels, dim=0)
        return x_rep, phase_rep, groups

    use_notebook_bar = (not quiet_mode and USE_NOTEBOOK_PROGRESS)
    total_iters = None
    if steps_per_epoch is not None and epochs > 0:
        total_iters = steps_per_epoch * epochs
    progress_bar = None
    if use_notebook_bar:
        ensure_dark_tqdm_theme()
        progress_bar = NotebookProgressBar(total=total_iters, desc=f"Epoch {start_epoch_from_ckpt + 1}/{epochs}")

    stop_training = False
    opt.zero_grad(set_to_none=True)
    for epoch in range(start_epoch_from_ckpt, epochs):
        if progress_bar is not None:
            progress_bar.set_description(f"Epoch {epoch + 1}/{epochs}")
            epoch_iterator = train_loader
            epoch_tqdm = None
        else:
            epoch_iterator = (
                tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True, colour="cyan")
                if not quiet_mode
                else train_loader
            )
            epoch_tqdm = epoch_iterator if not quiet_mode else None

        for batch_idx, batch in enumerate(epoch_iterator, start=1):
            current_iter += 1
            x_input, _ = batch
            x_input = x_input.to(device, non_blocking=True)
            B = x_input.size(0)
            if B % num_channels != 0:
                raise ValueError(
                    f"Batch size {B} must be divisible by num_channels {num_channels}."
                )
            # Infer group_size from current batch (definition: G = B // channels)
            inferred_group = B // num_channels
            group_size = inferred_group
            # grouping strategy: optional alternating (even iters grouped, odd iters ungrouped)
            if group_channels_alternate:
                group_flag = (current_iter % 2 == 0)
            else:
                group_flag = group_channels if not group_channels_first_only else (current_iter == 1)
            target_groups = getattr(layer, "batch_stacks", None)
            if target_groups is None:
                target_groups = group_size if group_flag else B
            phase_mask_base = generate_phase_mask(
                target_groups,
                phase_mask_dim,
                device=device,
                sigma_rel_range=(2/168, 24/168),
                randomize=True,
            )
            x_group, p_group, _ = _build_channel_groups(
                x_input,
                phase_mask_base,
                target_groups=target_groups,
                group_channels=group_flag,
            )
            x_exp, p_exp = encoding_x_phase_physical(x_group, p_group, canvas_hw=canvas_hw, x_mode="mix")
            # Per-sample random gain applied after encoding (shared with viz for consistency)
            scale = _draw_random_gain(x_exp.size(0), x_exp.dtype)
            #x_exp = x_exp * scale

            exp_target = optical_sys(x_exp, p_exp)

            with torch.amp.autocast(device.type, enabled=(device.type == 'cuda')):
                y_pred = model(x_exp, p_exp, exp_target)
                if y_pred.dim() == 5 and y_pred.size(2) == 1:
                    y_pred = y_pred.squeeze(2)
                mse_term = mse_loss(y_pred, exp_target)
                ffl_loss = ffl(y_pred, exp_target)
                # Dynamically pick ms-ssim window/levels to satisfy size constraint: (win-1)*2^(levels-1) < min_hw
                min_hw = int(min(y_pred.shape[-2], y_pred.shape[-1]))
                max_levels = 5  # default ms-ssim levels
                win_size = min(11, max(3, min_hw // (2 ** (max_levels - 1))))
                if win_size % 2 == 0:
                    win_size = max(3, win_size - 1)
                levels = max_levels
                while (win_size - 1) * (2 ** (levels - 1)) >= min_hw and levels > 1:
                    levels -= 1
                    win_size = min(win_size, max(3, min_hw // (2 ** (levels - 1))))
                    if win_size % 2 == 0:
                        win_size = max(3, win_size - 1)
                default_weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
                                               device=y_pred.device, dtype=y_pred.dtype)
                weights = default_weights[:levels]
                weights = weights / weights.sum()
                # Pass list to avoid pytorch_msssim warning from new_tensor(tensor)
                ms_ssim_loss = 1 - ms_ssim(y_pred, exp_target, data_range=1.0, win_size=win_size, weights=weights.detach().cpu().tolist())
                total_loss = 5 * mse_term + 0.5 * ms_ssim_loss + 1.0 * ffl_loss 
                loss = total_loss / float(accum_steps)

            scaler.scale(loss).backward()

            is_accum_step = (current_iter % accum_steps == 0)
            is_last_in_epoch = steps_per_epoch is not None and batch_idx == steps_per_epoch
            hit_max_iters = (max_iters is not None and current_iter >= max_iters)
            should_step = is_accum_step or is_last_in_epoch or hit_max_iters

            if should_step:
                if grad_clip_norm is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            loss_val = float(total_loss.detach().item())
            train_loss_hist.append(loss_val)
            train_iters.append(current_iter)

            mse_val = float(mse_term.detach().item())
            mse_window.append(mse_val)
            if len(mse_window) == mse_window.maxlen and (current_iter % 50 == 0):
                last_avg_mse50 = sum(mse_window) / float(len(mse_window))

            if plot_live:
                _update_loss_plot(train_iters, train_loss_hist)

            if visualize_samples and viz_interval is not None and (current_iter % viz_interval == 0):
                x_viz_batch = x_exp.detach()
                p_viz_batch = p_exp.detach()
                target_viz = exp_target.detach()
                pred_viz = y_pred.detach()
                valid_indices = tuple(idx for idx in viz_indices if idx < x_viz_batch.size(0))
                _update_sample_plot(
                    x_viz_batch.detach(),
                    p_viz_batch.detach(),
                    target_viz.detach(),
                    pred_viz.detach(),
                    sample_indices=valid_indices,
                )

            if progress_bar is not None:
                postfix = {
                    "iter": current_iter,
                    "loss": f"{loss_val:.4g}",
                    "batch": f"{batch_idx}/{steps_per_epoch}" if steps_per_epoch else batch_idx,
                }
                if last_avg_mse50 is not None:
                    postfix["avg_mse50"] = f"{last_avg_mse50:.4g}"
                progress_bar.set_postfix(postfix, refresh=True)
                progress_bar.update(1)
            elif not quiet_mode and epoch_tqdm is not None and hasattr(epoch_tqdm, "set_postfix"):
                postfix = {"iter": current_iter, "loss": f"{loss_val:.4g}"}
                if last_avg_mse50 is not None:
                    postfix["avg_mse50"] = f"{last_avg_mse50:.4g}"
                epoch_tqdm.set_postfix(**postfix)

            # update best-on-training checkpoint (based on MSE term)
            if save_best and best_path is not None and mse_val < best_mse:
                best_mse = mse_val
                ckpt_obj_best: Dict[str, object] = {
                    "model_state": model.state_dict(),
                    "opt_state": opt.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "epoch": epoch + 1,
                    "iter": current_iter,
                    "best_mse": best_mse,
                    "best_train_loss": best_mse,  # backward compatibility
                    "history": {
                        "train_loss": train_loss_hist,
                        "train_iters": train_iters,
                    },
                    "meta": {
                        "saved_at": datetime.utcnow().isoformat(),
                        "kind": "best_mse",
                    },
                }
                _atomic_save(ckpt_obj_best, best_path)

            # checkpoint
            if ckpt_every and last_path is not None and (current_iter % ckpt_every) == 0:
                ckpt_obj: Dict[str, object] = {
                    "model_state": model.state_dict(),
                    "opt_state": opt.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "epoch": epoch + 1,
                    "iter": current_iter,
                    "best_mse": best_mse,
                    "best_train_loss": best_mse,
                    "history": {
                        "train_loss": train_loss_hist,
                        "train_iters": train_iters,
                    },
                    "meta": {
                        "saved_at": datetime.utcnow().isoformat(),
                        "kind": "last",
                    },
                }
                _atomic_save(ckpt_obj, last_path)

            # early stop
            if max_iters is not None and current_iter >= max_iters:
                stop_training = True
                if last_path is not None:
                    ckpt_obj = {
                        "model_state": model.state_dict(),
                        "opt_state": opt.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "epoch": epoch + 1,
                        "iter": current_iter,
                        "best_mse": best_mse,
                        "best_train_loss": best_mse,
                        "history": {
                            "train_loss": train_loss_hist,
                            "train_iters": train_iters,
                        },
                        "meta": {
                            "saved_at": datetime.utcnow().isoformat(),
                            "kind": "last",
                        },
                    }
                    _atomic_save(ckpt_obj, last_path)
                break

        # end of epoch checkpoint
        if ckpt_every and last_path is not None:
            ckpt_obj = {
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "scaler_state": scaler.state_dict(),
                "epoch": epoch + 1,
                "iter": current_iter,
                "best_mse": best_mse,
                "best_train_loss": best_mse,
                "history": {
                    "train_loss": train_loss_hist,
                    "train_iters": train_iters,
                },
                "meta": {
                    "saved_at": datetime.utcnow().isoformat(),
                    "kind": "last",
                },
            }
            _atomic_save(ckpt_obj, last_path)

        if epoch_tqdm is not None and hasattr(epoch_tqdm, "close"):
            epoch_tqdm.close()
        if stop_training:
            break

    if progress_bar is not None:
        progress_bar.close()

    # final example visualisation (one batch) if requested
    if visualize_samples and not quiet_mode:
        model.eval()
        try:
            try:
                fallback_batch, _ = next(iter(train_loader))
            except Exception:
                fallback_batch = torch.zeros(1, 1, canvas_hw[0], canvas_hw[1], device=device)
            fallback_batch = fallback_batch.to(device, non_blocking=True)
            group_flag_viz = group_channels if not group_channels_first_only else False
            target_groups = getattr(layer, "batch_stacks", None)
            if target_groups is None:
                target_groups = max(1, fallback_batch.size(0) // num_channels)
            if not group_flag_viz:
                target_groups = fallback_batch.size(0)
            phase_mask_base = generate_phase_mask(
                target_groups,
                phase_mask_dim,
                device=device,
                sigma_rel_range=(2/168, 24/168),
                randomize=True,
            )
            x_group, p_group, _ = _build_channel_groups(
                fallback_batch,
                phase_mask_base,
                target_groups=target_groups,
                group_channels=group_flag_viz,
            )
            x_vis_exp, p_vis_exp = encoding_x_phase_physical(x_group, p_group, canvas_hw=canvas_hw, x_mode="mix")
            scale_viz = _draw_random_gain(x_vis_exp.size(0), x_vis_exp.dtype)
            # x_vis_exp = x_vis_exp * scale_viz
            with torch.no_grad():
                target_vis = optical_sys(x_vis_exp, p_vis_exp)
                if target_vis.dim() == 5 and target_vis.size(2) == 1:
                    target_vis = target_vis.squeeze(2)
                pred_vis = model(x_vis_exp, p_vis_exp, target_vis)
                if pred_vis.dim() == 5 and pred_vis.size(2) == 1:
                    pred_vis = pred_vis.squeeze(2)
            valid_indices = tuple(idx for idx in viz_indices if idx < x_vis_exp.size(0))
            _update_sample_plot(
                x_vis_exp,
                p_vis_exp,
                target_vis,
                pred_vis,
                sample_indices=valid_indices,
            )
        except Exception as exc:
            if not quiet_mode:
                print(f"[warn] Failed to create sample visualisation: {exc}")

    return TrainHistory(
        train_loss=list(train_loss_hist),
        eval_loss=list(eval_loss_hist),
        train_iters=list(train_iters),
        eval_iters=list(eval_iters),
    )
