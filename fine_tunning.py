"""
Utility helpers for buffering optical layer samples and fine-tuning the surrogate model.
"""
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from IPython.display import display, HTML
from torch.utils.data import DataLoader, TensorDataset

from model_training.onn_online_training.models import Surrogate_OpticalNet_Unet


PHASE_CMAP = "magma"
# Match VS Code dark background to avoid mismatched navy cards
CARD_BG = "#1e1e1e"
SURFACE_BG = CARD_BG
SURFACE_BORDER = "#2b2b2b"
TEXT_PRIMARY = "#f5f7ff"
TEXT_MUTED = "#9cb3d8"
_USE_NOTEBOOK_VIZ = False  # legacy flag kept for compatibility


@dataclass
class FineTuneState:
    """Holds per-layer buffers used during surrogate fine-tuning.

    Each layer buffer tracks:
        - ``x``: encoded activations fed into the corresponding optical layer.
        - ``phase``: phase map delivered to the layer (matching ``x`` spatially).
        - ``y``: optical layer outputs.
        - ``x_raw``: pre-encoded activations before canvas expansion (optional diagnostic).
    Stored tensors are kept on CPU to minimise GPU memory pressure.
    """

    buffer_max: int
    buffers: List[Dict[str, List[torch.Tensor]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.buffers:
            raise ValueError("FineTuneState requires non-empty buffer list.")

    def total_samples(self) -> int:
        """Return the total number of buffered items across all layers."""
        total = 0
        for layer in self.buffers:
            for x in layer["x"]:
                if isinstance(x, torch.Tensor):
                    total += x.size(0)
                else:
                    total += 0
        return total

    def clear(self) -> None:
        """Reset all buffers."""
        for layer in self.buffers:
            layer["x"].clear()
            layer["x_raw"].clear()
            layer["phase"].clear()
            layer["y"].clear()


def init_fine_tune_state(n_layers: int, buffer_max: int) -> FineTuneState:
    """Create an empty FineTuneState for the requested number of optical layers."""
    if n_layers <= 0:
        raise ValueError("n_layers must be positive.")
    buffers = [{"x": [], "x_raw": [], "phase": [], "y": []} for _ in range(n_layers)]
    return FineTuneState(buffer_max=buffer_max, buffers=buffers)


def accumulate_fine_tune_samples(
    state: FineTuneState,
    layer_inputs: Iterable[torch.Tensor],
    layer_phases: Iterable[torch.Tensor],
    layer_outputs: Iterable[torch.Tensor],
    layer_raw_inputs: Optional[Iterable[Optional[torch.Tensor]]] = None,
) -> None:
    
    raw_batches = list(layer_raw_inputs) if layer_raw_inputs is not None else None
    for idx, (x_layer, phase_layer, y_layer) in enumerate(
        zip(layer_inputs, layer_phases, layer_outputs)
    ):
        if not isinstance(x_layer, torch.Tensor):
            continue
        if not isinstance(phase_layer, torch.Tensor):
            continue
        if not isinstance(y_layer, torch.Tensor):
            continue
        layer_buf = state.buffers[idx]
        if len(layer_buf["x"]) >= state.buffer_max:
            continue
        layer_buf["x"].append(x_layer.detach().cpu())
        layer_buf["phase"].append(phase_layer.detach().cpu())
        layer_buf["y"].append(y_layer.detach().cpu())
        raw_tensor: Optional[torch.Tensor]
        if raw_batches is not None and idx < len(raw_batches):
            raw_tensor = raw_batches[idx]
        else:
            raw_tensor = None
        if raw_tensor is None or not isinstance(raw_tensor, torch.Tensor):
            raw_tensor = x_layer
        layer_buf["x_raw"].append(raw_tensor.detach().cpu())


def maybe_fine_tune_surrogate(
    state: FineTuneState,
    *,
    iteration: int,
    buffer_freq: int,
    surrogate: nn.Module,
    device: torch.device,
    enc_canvas_hw: Tuple[int, int],
    fine_steps: int,
    fine_lr: float,
    fine_weight_decay: float,
    use_amp: bool,
    batch_size: int = 64,
    print_losses: bool = False,
    visualize: bool = True,
    viz_every: int = 0,
    sample_every: int = 0,
    show_sample: bool = True,
    surrogate_ckpt_path: Optional[Union[str, Path]] = None,
) -> bool:
    """
    Launch a short fine-tuning run for the surrogate when enough samples are buffered.

    Returns True if fine-tuning was executed, otherwise False.
    """
    if buffer_freq <= 0:
        return False
    if iteration % buffer_freq != 0:
        return False
    total_items = state.total_samples()
    if total_items == 0:
        return False
    num_channels = getattr(surrogate, "num_channels", 16)
    if total_items < num_channels:
        return False

    samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for layer_idx, layer_buf in enumerate(state.buffers):
        for x_buf, phase_buf, y_buf in zip(
            layer_buf["x"], layer_buf["phase"], layer_buf["y"]
        ):
            if not isinstance(x_buf, torch.Tensor):
                continue
            if not isinstance(phase_buf, torch.Tensor):
                continue
            if not isinstance(y_buf, torch.Tensor):
                continue
            if x_buf.dim() != 4 or y_buf.dim() != 4:
                continue
            if phase_buf.dim() == 3:
                phase_batch = phase_buf.unsqueeze(0).expand(x_buf.size(0), -1, -1, -1)
            elif phase_buf.dim() == 4:
                phase_batch = phase_buf
            else:
                continue
            batch_len = min(x_buf.size(0), phase_batch.size(0), y_buf.size(0))
            if batch_len <= 0:
                continue
            x_use = x_buf[:batch_len]
            phase_use = phase_batch[:batch_len]
            y_use = y_buf[:batch_len]
            # Keep channel ordering intact; append whole batch (not individual items).
            samples.append((x_use.clone(), phase_use.clone(), y_use.clone()))

    if not samples:
        return False

    # Concatenate batches; ensure total rows divisible by num_channels
    xs_stack = torch.cat([item[0] for item in samples], dim=0)
    phase_stack = torch.cat([item[1] for item in samples], dim=0)
    ys_stack = torch.cat([item[2] for item in samples], dim=0)
    total_rows = xs_stack.size(0)
    usable_rows = (total_rows // num_channels) * num_channels
    if usable_rows == 0:
        return False
    xs_stack = xs_stack[:usable_rows]
    phase_stack = phase_stack[:usable_rows]
    ys_stack = ys_stack[:usable_rows]
    fine_ds = TensorDataset(xs_stack, phase_stack, ys_stack)
    # Ensure dataloader batches are multiples of num_channels
    batch_size = max(num_channels, (batch_size // num_channels) * num_channels)
    fine_loader = DataLoader(fine_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    ft_model = Surrogate_OpticalNet_Unet(enc_canvas_hw=enc_canvas_hw).to(device)
    ft_model.load_state_dict(surrogate.state_dict())
    ft_model.train()

    optimizer_ft = torch.optim.AdamW(ft_model.parameters(), lr=fine_lr, weight_decay=fine_weight_decay)
    loss_ft = nn.MSELoss()
    scaler_ft = torch.amp.GradScaler("cuda", enabled=use_amp)

    steps_done = 0
    max_steps = max(0, int(fine_steps))
    loss_history: Deque[float] = deque(maxlen=1000)
    loss_steps: Deque[int] = deque(maxlen=1000)

    viz_every = max(0, int(viz_every))
    sample_every = max(0, int(sample_every))
    loss_state = _get_loss_plot_state() if visualize else None

    def _run_fine_tune_step(x_ft: torch.Tensor, phase_ft: torch.Tensor, y_ft: torch.Tensor) -> None:
        nonlocal steps_done
        x_ft = x_ft.to(device)
        phase_ft = phase_ft.to(device)
        y_ft = y_ft.to(device)
        optimizer_ft.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            preds_ft = ft_model(x_ft, phase_ft, target=y_ft)
            loss_val_ft = loss_ft(preds_ft, y_ft)
        scaler_ft.scale(loss_val_ft).backward()
        scaler_ft.step(optimizer_ft)
        scaler_ft.update()
        steps_done += 1
        loss_value = loss_val_ft.detach().item()
        loss_history.append(loss_value)
        loss_steps.append(steps_done)
        if print_losses:
            print(
                f"\r[fine_tune] step {steps_done}/{max_steps or '?'} "
                f"loss {loss_value:.5f}",
                end="",
                flush=True,
            )
        should_update_loss = loss_state is not None and viz_every > 0 and (steps_done % viz_every == 0)
        if should_update_loss:
            _update_loss_plot(loss_state, loss_history, loss_steps)

    try:
        if len(fine_loader) == 0:
            return False
        if max_steps > 0:
            loader_iter = iter(fine_loader)
            while steps_done < max_steps:
                try:
                    x_ft, phase_ft, y_ft = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(fine_loader)
                    continue
                _run_fine_tune_step(x_ft, phase_ft, y_ft)
        else:
            for x_ft, phase_ft, y_ft in fine_loader:
                _run_fine_tune_step(x_ft, phase_ft, y_ft)

        if print_losses and steps_done > 0:
            print(flush=True)

        surrogate.load_state_dict(ft_model.state_dict())
        surrogate.eval()

        if surrogate_ckpt_path is not None:
            ckpt_path = Path(surrogate_ckpt_path)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            loss_vals = list(loss_history)
            final_loss = float(loss_vals[-1]) if loss_vals else float("nan")
            payload = {
                "model_state": surrogate.state_dict(),
                "meta": {
                    "saved_at": datetime.utcnow().isoformat(),
                    "steps": steps_done,
                    "final_loss": final_loss,
                },
            }
            torch.save(payload, ckpt_path)

        final_step_reached = (max_steps == 0) or (steps_done >= max_steps)

        if visualize and samples:
            total_rows = xs_stack.size(0)
            sample_idx = random.randrange(total_rows)
            x_vis = xs_stack[sample_idx].detach().cpu()
            phase_vis = phase_stack[sample_idx].detach().cpu()
            y_vis = ys_stack[sample_idx].detach().cpu()
            x_vis_disp = torch.clamp(x_vis, 0.0, 1.0)
            ft_model.eval()
            with torch.no_grad():
                # Repeat a single sample to satisfy channel-conditioning requirement
                reps = num_channels
                x_vis_rep = x_vis.unsqueeze(0).repeat(reps, 1, 1, 1).to(device)
                phase_vis_rep = phase_vis.unsqueeze(0).repeat(reps, 1, 1, 1).to(device)
                y_vis_rep = y_vis.unsqueeze(0).repeat(reps, 1, 1, 1).to(device)
                pred_rep = ft_model(
                    x_vis_rep,
                    phase_vis_rep,
                    target=y_vis_rep,
                )
                pred_vis = pred_rep[0].detach().cpu()
            ft_model.train()

            should_show_sample = show_sample and (final_step_reached or (sample_every > 0 and steps_done % sample_every == 0))
            if should_show_sample:
                sample_state = _get_sample_plot_state()
                _update_sample_plot(
                    sample_state,
                    [
                        (x_vis_disp, "Input"),
                        (phase_vis, "Phase"),
                        (y_vis, "Ground Truth"),
                        (pred_vis.squeeze(0), "Surrogate Pred"),
                    ],
                )
            # ensure final refresh of loss plot
            if loss_state is not None:
                _update_loss_plot(loss_state, loss_history, loss_steps)
        state.clear()
        return True
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Fine-tuning surrogate failed: {exc}")
        return False


_LOSS_PLOT_STATE: Dict[str, object] = {}
_SAMPLE_PLOT_STATE: Dict[str, object] = {}
_FINE_TUNE_TITLE_PRINTED: bool = False
_FINE_TUNE_SECTION_CREATED: bool = False


def reset_fine_tune_viz() -> None:
    """Clear cached matplotlib figures and force the fine-tune section to re-render."""
    global _LOSS_PLOT_STATE, _SAMPLE_PLOT_STATE
    global _FINE_TUNE_SECTION_CREATED, _FINE_TUNE_TITLE_PRINTED

    def _close_fig(state: Dict[str, object]) -> None:
        fig = state.get("fig")
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass

    if _LOSS_PLOT_STATE:
        _close_fig(_LOSS_PLOT_STATE)
        _LOSS_PLOT_STATE = {}
    if _SAMPLE_PLOT_STATE:
        _close_fig(_SAMPLE_PLOT_STATE)
        _SAMPLE_PLOT_STATE = {}

    _FINE_TUNE_SECTION_CREATED = False
    _FINE_TUNE_TITLE_PRINTED = False


def _ensure_fine_tune_section() -> Dict[str, Optional[object]]:
    global _FINE_TUNE_SECTION_CREATED, _FINE_TUNE_TITLE_PRINTED
    if not _FINE_TUNE_SECTION_CREATED:
        display(
            HTML(
                f"""
                <div style="margin:16px 0 18px 0; padding:20px 24px; border-radius:12px; background:{CARD_BG}; box-shadow:0 18px 34px rgba(2,4,10,0.45); text-align:center;">
                    <h2 style="margin:0; font-size:24px; color:#f5f7ff;">Fine-tuning of Surrogate Model</h2>
                </div>
                """
            )
        )
        _FINE_TUNE_SECTION_CREATED = True
    return {"loss": None, "samples": None}


def _get_loss_plot_state() -> Dict[str, object]:
    outputs = _ensure_fine_tune_section()
    output_widget = outputs.get("loss")
    if _LOSS_PLOT_STATE:
        return _LOSS_PLOT_STATE
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.0))
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(CARD_BG)
    ax.set_title("Fine-tune Loss", color=TEXT_PRIMARY)
    ax.set_xlabel("Step", color=TEXT_PRIMARY)
    ax.set_ylabel("MSE", color=TEXT_PRIMARY)
    ax.tick_params(colors=TEXT_PRIMARY)
    ax.grid(True, alpha=0.35, color="#2e394f")
    line, = ax.plot([], [], color="#4dd0e1", linewidth=1.8, marker="o", markersize=3)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.4, top=0.9, bottom=0.1)
    if output_widget is not None:
        with output_widget:
            handle = display(fig, display_id=True)
    else:
        handle = display(fig, display_id=True)
    state = {"fig": fig, "ax": ax, "line": line, "handle": handle, "output": output_widget}
    _LOSS_PLOT_STATE.update(state)
    return state


def _update_loss_plot(
    state: Dict[str, object],
    loss_history: Iterable[float],
    steps_history: Iterable[int],
) -> None:
    line = state["line"]
    ax = state["ax"]
    fig = state["fig"]
    handle = state.get("handle")
    history_list = list(loss_history)
    steps_list = list(steps_history)
    if not history_list:
        return
    if len(steps_list) != len(history_list):
        steps_list = list(range(1, len(history_list) + 1))
    line.set_data(steps_list, history_list)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()
    if handle is not None and hasattr(handle, "update"):
        handle.update(fig)


def _get_sample_plot_state() -> Dict[str, object]:
    _ensure_fine_tune_section()
    output_widget = None
    if _SAMPLE_PLOT_STATE:
        return _SAMPLE_PLOT_STATE
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    fig.patch.set_facecolor(CARD_BG)
    images = []
    colorbars = []
    titles = ["Input", "Phase", "Ground Truth", "Surrogate Pred"]
    blank = np.zeros((16, 16), dtype=np.float32)
    for ax, title in zip(axes, titles):
        ax.set_facecolor(CARD_BG)
        if title == "Phase":
            cmap = PHASE_CMAP
        elif "Input" in title:
            cmap = "gray"
        else:
            cmap = "magma"
        im = ax.imshow(blank, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(title, color=TEXT_PRIMARY)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_facecolor(CARD_BG)
        cbar.ax.tick_params(labelsize=8, colors=TEXT_PRIMARY)
        colorbars.append(cbar)
        images.append(im)
    fig.tight_layout()
    handle = display(fig, display_id=True)
    state = {
        "fig": fig,
        "axes": axes,
        "images": images,
        "colorbars": colorbars,
        "handle": handle,
        "output": output_widget,
    }
    _SAMPLE_PLOT_STATE.update(state)
    return state


def _update_sample_plot(state: Dict[str, object], tensors: List[Tuple[torch.Tensor, str]]) -> None:
    fig = state["fig"]
    images = state["images"]
    colorbars = state["colorbars"]
    handle = state.get("handle")
    for (tensor, title), im, cbar in zip(tensors, images, colorbars):
        arr = tensor.squeeze().detach().cpu().numpy()
        if arr.ndim == 0:
            arr = np.array([[arr]])
        elif arr.ndim > 2:
            arr = arr[0]
        im.set_data(arr)
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        if vmin == vmax:
            vmax = vmin + 1e-6
        im.set_clim(vmin, vmax)
        im.axes.set_title(title, color=TEXT_PRIMARY)
        cbar.update_normal(im)
        cbar.ax.tick_params(colors=TEXT_PRIMARY)
    fig.canvas.draw_idle()
    if handle is not None and hasattr(handle, "update"):
        handle.update(fig)
