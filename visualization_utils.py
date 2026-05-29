"""
Visualization utilities for live loss plotting and example rendering.
"""

from __future__ import annotations

import random
from typing import List, Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from IPython.display import display, DisplayHandle
try:
    from .utils import encoding_x_phase_physical
except ImportError:
    from utils import encoding_x_phase_physical


# You can tweak these two to taste.
_BG = "#1e1e1e"          # figure/axes background (VSCode Dark+)
_GRID = "#444444"        # subtle grid
_CMAP = "magma"          # high-contrast on dark (alternatives: "inferno", "cividis")

plt.rcParams.update({
    "figure.facecolor": _BG,
    "axes.facecolor": _BG,
    "savefig.facecolor": _BG,

    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "axes.titlecolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "text.color": "white",

    "grid.color": _GRID,
    "grid.alpha": 0.4,
    "axes.grid": True,

    # lines/markers a bit brighter by default
    "axes.prop_cycle": plt.cycler(color=["#4dd0e1", "#ffd54f", "#90ee90", "#ff8a65", "#b39ddb"]),
})

def _style_axes(ax):
    """Ensure axes match our dark theme (use after creating lines/labels)."""
    ax.set_facecolor(_BG)
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.4, color=_GRID)

def _style_legend(leg):
    if leg is None:
        return
    frame = leg.get_frame()
    frame.set_facecolor(_BG)
    frame.set_edgecolor("#555555")
    for text in leg.get_texts():
        text.set_color("white")

def _style_colorbar(cb):
    """Dark-friendly colorbar ticks & frame."""
    cb.outline.set_edgecolor("white")
    cb.ax.tick_params(colors="white")
    for lab in cb.ax.get_yticklabels():
        lab.set_color("white")

# =====================================================


class LiveLossPlot:
    """
    Single-axes live plot (inline-safe, remote-friendly).
    """
    def __init__(
        self,
        title: str = "Training & Eval MSE",
        log_scale: bool = False,
        plot_rate: int = 1,
        dynamic_log: bool = True,
        log_switch_threshold: float = 1.0,
    ) -> None:
        self.plot_rate = max(1, int(plot_rate))
        self.dynamic_log = bool(dynamic_log)
        self.log_switch_threshold = float(log_switch_threshold)

        self._use_log = bool(log_scale)      # current y-scale state
        self._force_log = bool(log_scale)    # if True, never switch back
        self.train_x: List[int] = []
        self.train_y: List[float] = []
        self.eval_x: List[int] = []
        self.eval_y: List[float] = []
        self._last_update_it = -1

        self.fig, self.ax = plt.subplots(figsize=(7, 4.6))
        self.ax.set_title(title)
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("MSE loss")
        (self.train_line,) = self.ax.plot([], [], label="train", marker="o", markersize=3, linewidth=1.8)
        (self.eval_line,)  = self.ax.plot([], [], label="eval",  marker="s", markersize=3, linewidth=1.8)
        if self._use_log:
            self.ax.set_yscale("log")
            self.ax.set_ylabel("MSE loss (log)")
        _style_axes(self.ax)
        _style_legend(self.ax.legend(loc="upper right"))

        self.fig.canvas.draw()
        self.handle: DisplayHandle = display(self.fig, display_id=True)

    @staticmethod
    def _limits(xs: List[int], ys: List[float], log_scale: bool = False) -> Tuple[float, float, float, float]:
        if not xs or not ys:
            return 0.0, 1.0, (1e-8 if log_scale else 0.0), 1.0
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        if log_scale:
            pos_vals = [v for v in ys if v > 0]
            if not pos_vals:
                ymin, ymax = 1e-8, 1.0
            else:
                ymin = max(min(pos_vals), 1e-8)
                ymax = max(pos_vals)
                ymin /= 1.25
                ymax *= 1.25
        else:
            if ymin == ymax:
                ymax = ymin + 1.0
            pad = 0.1 * (ymax - ymin)
            ymin -= 0.05 * (ymax - ymin)
            ymax += pad

        return float(xmin), float(xmax) + 1.0, float(ymin), float(ymax)

    def _maybe_switch_to_log(self) -> None:
        if self._force_log or not self.dynamic_log or self._use_log:
            return
        all_y = self.train_y + self.eval_y
        if not all_y:
            return
        min_y = min(all_y)
        if 0.0 < min_y < self.log_switch_threshold:
            self._use_log = True
            self.ax.set_yscale("log")
            self.ax.set_ylabel("MSE loss (log)")

    def _maybe_update(self, it: int) -> None:
        if (it - self._last_update_it) < self.plot_rate:
            return
        self._last_update_it = it

        self._maybe_switch_to_log()

        self.train_line.set_data(self.train_x, self.train_y)
        self.eval_line.set_data(self.eval_x, self.eval_y)

        xs = (self.train_x + self.eval_x) or [0, 1]
        ys = (self.train_y + self.eval_y) or [1.0]
        xmin, xmax, ymin, ymax = self._limits(xs, ys, log_scale=self._use_log)
        if xmin == xmax:
            pad = 1.0
            xmin -= pad; xmax += pad
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)

        self.fig.canvas.draw()
        if self.handle is not None and hasattr(self.handle, "update"):
            try:
                self.handle.update(self.fig)
            except Exception:
                pass

    def update_train(self, it: int, loss: float) -> None:
        self.train_x.append(it)
        self.train_y.append(float(loss))
        self._maybe_update(it)

    def update_eval(self, it: int, loss: float) -> None:
        self.eval_x.append(it)
        self.eval_y.append(float(loss))
        self._maybe_update(it)

    def close(self) -> None:
        plt.close(self.fig)


class DualLiveLossPlot:
    """
    Two-panel live plot for training and evaluation loss.
    """
    def __init__(
        self,
        title_train: str = "Training MSE",
        title_eval: str = "Eval MSE",
        log_scale: bool = False,
        plot_rate: int = 1,
        dynamic_log: bool = True,
        log_switch_threshold: float = 1.0,
    ) -> None:
        self.plot_rate = max(1, int(plot_rate))
        self.dynamic_log = bool(dynamic_log)
        self.log_switch_threshold = float(log_switch_threshold)

        self._use_log = bool(log_scale)
        self._force_log = bool(log_scale)

        self.train_x: List[int] = []
        self.train_y: List[float] = []
        self.eval_x: List[int] = []
        self.eval_y: List[float] = []
        self._last_update_train = -1
        self._last_update_eval = -1

        self.fig, (self.ax_train, self.ax_eval) = plt.subplots(1, 2, figsize=(12, 4.2))
        self.ax_train.set_title(title_train)
        self.ax_eval.set_title(title_eval)

        for ax in (self.ax_train, self.ax_eval):
            ax.set_xlabel("Iteration")
            ax.set_ylabel("MSE loss")
            if self._use_log:
                ax.set_yscale("log")
                ax.set_ylabel("MSE loss (log)")
            _style_axes(ax)

        (self.train_line,) = self.ax_train.plot([], [], label="train", marker="o", markersize=3, linewidth=1.8)
        (self.eval_line,)  = self.ax_eval.plot([], [], label="eval",  marker="s", markersize=3, linewidth=1.8)
        _style_legend(self.ax_train.legend(loc="upper right"))
        _style_legend(self.ax_eval.legend(loc="upper right"))

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.handle: DisplayHandle = display(self.fig, display_id=True)

    @staticmethod
    def _limits(xs: List[int], ys: List[float], log_scale: bool = False) -> Tuple[float, float, float, float]:
        if not xs or not ys:
            return 0.0, 1.0, (1e-8 if log_scale else 0.0), 1.0
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        if log_scale:
            pos = [v for v in ys if v > 0]
            if not pos:
                ymin, ymax = 1e-8, 1.0
            else:
                ymin = max(min(pos), 1e-8)
                ymax = max(pos)
                ymin /= 1.25
                ymax *= 1.25
        else:
            if ymin == ymax:
                ymax = ymin + 1.0
            pad = 0.1 * (ymax - ymin)
            ymin -= 0.05 * (ymax - ymin)
            ymax += pad

        return float(xmin), float(xmax) + 1.0, float(ymin), float(ymax)

    def _maybe_switch_to_log(self) -> None:
        if self._force_log or not self.dynamic_log or self._use_log:
            return
        all_y = self.train_y + self.eval_y
        if not all_y:
            return
        m = min(all_y)
        if 0.0 < m < self.log_switch_threshold:
            self._use_log = True
            for ax in (self.ax_train, self.ax_eval):
                ax.set_yscale("log")
                ax.set_ylabel("MSE loss (log)")

    def _apply_limits(self) -> None:
        xs_all = (self.train_x or []) + (self.eval_x or [])
        if xs_all:
            xmin_fixed = float(min(xs_all)); xmax_fixed = float(max(xs_all))
            if xmin_fixed == xmax_fixed:
                xmin_fixed -= 1.0; xmax_fixed += 1.0
        else:
            xmin_fixed, xmax_fixed = 0.0, 1.0

        _, _, ymin_t, ymax_t = self._limits(self.train_x, self.train_y, self._use_log)
        self.ax_train.set_xlim(xmin_fixed, xmax_fixed)
        self.ax_train.set_ylim(ymin_t, ymax_t)

        self.ax_eval.set_xlim(xmin_fixed, xmax_fixed)
        if self.eval_x and self.eval_y:
            _, _, ymin_e, ymax_e = self._limits(self.eval_x, self.eval_y, self._use_log)
            self.ax_eval.set_ylim(ymin_e, ymax_e)

    def update_train(self, it: int, loss: float) -> None:
        self.train_x.append(it)
        self.train_y.append(float(loss))
        if (it - self._last_update_train) < self.plot_rate:
            return
        self._last_update_train = it

        self._maybe_switch_to_log()
        self.train_line.set_data(self.train_x, self.train_y)
        self._apply_limits()
        self.fig.canvas.draw()
        if self.handle is not None and hasattr(self.handle, "update"):
            try:
                self.handle.update(self.fig)
            except Exception:
                pass

    def update_eval(self, it: int, loss: float) -> None:
        self.eval_x.append(it)
        self.eval_y.append(float(loss))
        if (it - self._last_update_eval) < self.plot_rate:
            return
        self._last_update_eval = it

        self._maybe_switch_to_log()
        self.eval_line.set_data(self.eval_x, self.eval_y)
        self._apply_limits()
        self.fig.canvas.draw()
        if self.handle is not None and hasattr(self.handle, "update"):
            try:
                self.handle.update(self.fig)
            except Exception:
                pass

    def close(self) -> None:
        plt.close(self.fig)


def visualize_random_examples(
    model: nn.Module,
    optical_sys: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    phase_mask_generator,
    loader: DataLoader,
    device: str = "cuda",
    num_examples: int = 4,
    phase_mask_dim: Tuple[int, int] = (42, 42),
) -> None:
    # persistent display handle stored on the function itself
    if not hasattr(visualize_random_examples, "_handle"):
        visualize_random_examples._handle = None

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    xs, _ = next(iter(loader))
    B = xs.size(0)
    num_examples = min(num_examples, B)
    idxs = random.sample(range(B), num_examples)

    x_input = xs.to(device, non_blocking=True)

    phase_mask = phase_mask_generator(B, phase_mask_dim, device)

    with torch.no_grad():
        enc_hw = tuple(getattr(model, "enc_canvas_hw", ())) or None
        if enc_hw is None:
            H_t = max(x_input.shape[-2], phase_mask.shape[-2])
            W_t = max(x_input.shape[-1], phase_mask.shape[-1])
            H_t = ((H_t + 7) // 8) * 8
            W_t = ((W_t + 7) // 8) * 8
            enc_hw = (H_t, W_t)

        x_exp, p_exp = _expand_inputs_for_layer_dynamic(x_input, phase_mask, enc_canvas_hw=enc_hw)

        ref_out = optical_sys(x_exp, p_exp)
        exp_target = ref_out.squeeze(2) if (ref_out.dim() == 5 and ref_out.size(2) == 1) else ref_out

        y_pred = model(x_exp, p_exp, exp_target)
        if y_pred.dim() == 5 and y_pred.size(2) == 1:
            y_pred = y_pred.squeeze(2)

    n_cols = 4
    fig, axes = plt.subplots(num_examples, n_cols, figsize=(3.6 * n_cols, 3.0 * num_examples))
    fig.patch.set_facecolor(_BG)
    if num_examples == 1:
        axes = [axes]

    for r, idx in enumerate(idxs):
        def ch1(t: torch.Tensor) -> torch.Tensor:
            return t[0] if t.dim() == 3 else t

        x_exp_img = ch1(x_exp[idx].detach().cpu())
        p_exp_img = p_exp[idx, 0].detach().cpu()
        gt_img    = exp_target[idx, 0].detach().cpu()
        pred_img  = y_pred[idx, 0].detach().cpu()

        axs = axes[r]
        for a in axs:
            _style_axes(a)  # ensure dark axes even if Matplotlib changes defaults
            a.axis("off")

        im0 = axs[0].imshow(x_exp_img, cmap=_CMAP); axs[0].set_title("x_exp")
        im1 = axs[1].imshow(p_exp_img, cmap=_CMAP); axs[1].set_title("p_exp")
        im2 = axs[2].imshow(gt_img,    cmap=_CMAP); axs[2].set_title("Ground truth")
        im3 = axs[3].imshow(pred_img,  cmap=_CMAP); axs[3].set_title("Prediction")

        for im, ax in zip([im0, im1, im2, im3], axs):
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            _style_colorbar(cb)

    plt.tight_layout()
    if visualize_random_examples._handle is None:
        visualize_random_examples._handle = display(fig, display_id=True)
    elif hasattr(visualize_random_examples._handle, "update"):
        try:
            visualize_random_examples._handle.update(fig)
        except Exception:
            pass
    plt.close(fig)


def _expand_inputs_for_layer_dynamic(
    x: torch.Tensor,
    phase: torch.Tensor,
    enc_canvas_hw: Optional[Tuple[int, int]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    Hx, Wx = x.shape[-2:]
    Hp, Wp = phase.shape[-2:]

    if enc_canvas_hw is not None:
        canvas_hw = enc_canvas_hw
    else:
        H_t = max(Hx, Hp)
        W_t = max(Wx, Wp)
        H_t = ((H_t + 3) // 4) * 4
        W_t = ((W_t + 3) // 4) * 4
        canvas_hw = (H_t, W_t)

    return encoding_x_phase_physical(x, phase, canvas_hw=canvas_hw)
