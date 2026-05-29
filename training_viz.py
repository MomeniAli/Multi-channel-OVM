"""
Visualization helpers shared by the ONN training script.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import HTML, clear_output, display

try:
    from ipywidgets import HTML as WidgetHTML
except Exception:  # pragma: no cover - ipywidgets may be unavailable
    WidgetHTML = None

_PHASE_CMAP = "magma"
# Align notebook widgets with VS Code dark theme so cards match the editor
# background instead of the previous navy tone.
_VS_CODE_BG = "#1e1e1e"
_CARD_BG = _VS_CODE_BG
_SURFACE_BG = _VS_CODE_BG
_SURFACE_BORDER = "#2b2b2b"
_BAR_BG = "#2d2d2d"
_TEXT_PRIMARY = "#e7ebff"
_TEXT_MUTED = "#9cb3d8"
_TQDM_CSS_INJECTED = False


def _style_axes(
    ax,
    *,
    tick_labelsize: Optional[int] = None,
    tick_labelweight: Optional[str] = None,
):
    ax.set_facecolor(_CARD_BG)
    for spine in ax.spines.values():
        spine.set_color(_TEXT_PRIMARY)
    if tick_labelsize is None:
        ax.tick_params(colors=_TEXT_PRIMARY)
    else:
        ax.tick_params(colors=_TEXT_PRIMARY, labelsize=tick_labelsize)
    if tick_labelweight:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight(tick_labelweight)
    ax.grid(True, alpha=0.35, color="#2e394f")


def _style_legend(leg):
    if leg is None:
        return
    frame = leg.get_frame()
    frame.set_facecolor(_CARD_BG)
    frame.set_edgecolor("#2e394f")
    for txt in leg.get_texts():
        txt.set_color(_TEXT_PRIMARY)


def _style_colorbar(cb):
    cb.outline.set_edgecolor(_TEXT_PRIMARY)
    cb.ax.tick_params(colors=_TEXT_PRIMARY, labelsize=8)


def _detect_notebook_env() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        shell_name = shell.__class__.__name__
        return shell_name in {"ZMQInteractiveShell", "TerminalInteractiveShell"}
    except Exception:
        return False


USE_NOTEBOOK_PROGRESS = _detect_notebook_env()


def create_training_viz_layout(
    n_layers: Optional[int] = None,
    in_silico_mode: Optional[bool] = None,
    fine_tuning_enabled: Optional[bool] = None,
    n_optical_params: Optional[int] = None,
    training_mode_title: Optional[str] = None,
    training_mode_caption: Optional[str] = None,
    depth_label: Optional[str] = None,
    depth_caption: Optional[str] = None,
) -> Dict[str, Optional[object]]:
    display(
        HTML(
            f"""
            <div style="margin:12px 0 10px 0; padding:22px 26px; border-radius:12px; background:{_CARD_BG}; box-shadow:0 18px 34px rgba(2,4,10,0.45); text-align:center;">
                <h1 style="margin:0; font-size:28px; color:#f5f7ff;">Training of Optical Neural Net.</h1>
            </div>
            """
        )
    )
    if n_layers is not None and in_silico_mode is not None and fine_tuning_enabled is not None:
        show_onn_training_summary_card(
            n_layers=n_layers,
            in_silico_mode=in_silico_mode,
            fine_tuning_enabled=fine_tuning_enabled,
            n_optical_params=n_optical_params,
            training_mode_title=training_mode_title,
            training_mode_caption=training_mode_caption,
            depth_label=depth_label,
            depth_caption=depth_caption,
        )
    return {"progress": None, "metrics": None, "samples": None, "confusion": None, "phase": None}


def show_onn_training_summary_card(
    *,
    n_layers: int,
    in_silico_mode: bool,
    fine_tuning_enabled: bool,
    n_optical_params: Optional[int] = None,
    training_mode_title: Optional[str] = None,
    training_mode_caption: Optional[str] = None,
    depth_label: Optional[str] = None,
    depth_caption: Optional[str] = None,
) -> None:
    if training_mode_title is None:
        mode_title = "In-silico training" if in_silico_mode else "Physics-aware training"
    else:
        mode_title = training_mode_title
    if training_mode_caption is None:
        mode_caption = (
            "Forward and backward passes are performed entirely within the differentiable surrogate model."
            if in_silico_mode
            else "The physical optical layer is kept in-the-loop for the forward pass while gradients are provided by the surrogate model."
        )
    else:
        mode_caption = training_mode_caption
    fine_title = "Enabled" if fine_tuning_enabled else "Disabled"
    fine_caption = (
        "Surrogate parameters are periodically updated using hardware measurements."
        if fine_tuning_enabled
        else "Surrogate remains frozen; only the optical phase parameters are optimized."
    )
    if n_optical_params is not None and n_optical_params >= 0:
        grid_template_cols = "repeat(4,minmax(0,1fr))"
        optical_params_value = f"{int(n_optical_params):,}"
        optical_params_block = f"""
                        <div>
                            <div style="text-transform:uppercase; letter-spacing:0.12em; font-size:11px; color:{_TEXT_MUTED}; margin-bottom:4px;">
                                Optical parameters
                            </div>
                            <div style="font-size:18px; font-weight:650; margin-bottom:2px;">
                                {optical_params_value}
                            </div>
                            <div style="font-size:12px; color:{_TEXT_MUTED};">
                                Total trainable phase values across all modulation planes.
                            </div>
                        </div>"""
    else:
        grid_template_cols = "repeat(3,minmax(0,1fr))"
        optical_params_block = ""
    if depth_label is None:
        depth_label = "Optical layers"
    if depth_caption is None:
        depth_caption = "Number of cascaded phase modulation planes used in the optical stack."
    display(
        HTML(
            f"""
            <div style="margin:20px 0 8px 0; display:flex; justify-content:center;">
                <div style="
                    max-width:980px;
                    width:100%;
                    border:1px solid #ffffff;
                    border-radius:14px;
                    padding:18px 26px 20px 26px;
                    background:{_CARD_BG};
                    box-shadow:0 20px 40px rgba(2, 4, 10, 0.55);
                    font-family:-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Inter', 'Segoe UI', system-ui, sans-serif;
                    color:{_TEXT_PRIMARY};
                ">
                    <div style="text-transform:uppercase; letter-spacing:0.16em; font-size:11px; font-weight:600; color:{_TEXT_MUTED}; margin-bottom:4px;">
                        Optical Neural Network
                    </div>
                    <div style="display:flex; flex-wrap:wrap; align-items:flex-end; justify-content:space-between; gap:8px; margin-bottom:14px;">
                        <div style="font-size:22px; font-weight:650;">Training Summary</div>
                    </div>
                    <div style="display:grid; grid-template-columns:{grid_template_cols}; gap:16px; font-size:13px;">
                        <div>
                            <div style="text-transform:uppercase; letter-spacing:0.12em; font-size:11px; color:{_TEXT_MUTED}; margin-bottom:4px;">
                                {depth_label}
                            </div>
                            <div style="font-size:18px; font-weight:650; margin-bottom:2px;">
                                {int(n_layers)}
                            </div>
                            <div style="font-size:12px; color:{_TEXT_MUTED};">
                                {depth_caption}
                            </div>
                        </div>
                        <div>
                            <div style="text-transform:uppercase; letter-spacing:0.12em; font-size:11px; color:{_TEXT_MUTED}; margin-bottom:4px;">
                                Training mode
                            </div>
                            <div style="font-size:16px; font-weight:600; margin-bottom:2px;">
                                {mode_title}
                            </div>
                            <div style="font-size:12px; color:{_TEXT_MUTED};">
                                {mode_caption}
                            </div>
                        </div>
                        <div>
                            <div style="text-transform:uppercase; letter-spacing:0.12em; font-size:11px; color:{_TEXT_MUTED}; margin-bottom:4px;">
                                Surrogate fine-tuning
                            </div>
                            <div style="font-size:16px; font-weight:600; margin-bottom:2px;">
                                {fine_title}
                            </div>
                            <div style="font-size:12px; color:{_TEXT_MUTED};">
                                {fine_caption}
                            </div>
                        </div>
                        {optical_params_block}
                    </div>
                </div>
            </div>
            """
        )
    )


class NotebookProgressBar:
    def __init__(self, total: Optional[int], desc: str, output: Optional[object] = None):
        self.total = total if (total is None or total > 0) else None
        self.desc = desc
        self.n = 0
        self.postfix: Dict[str, str] = {}
        self.start_time = time.time()
        self.output = output if hasattr(output, "clear_output") else None
        self.widget = None
        self._display = None
        if self.output is not None and WidgetHTML is not None:
            try:
                with self.output:
                    clear_output(wait=True)
                    self.widget = WidgetHTML(self._render())
                    display(self.widget)
            except Exception:
                self.widget = None
        if self.widget is None:
            self._display = display(HTML(self._render()), display_id=True)

    def set_description(self, desc: str) -> None:
        self.desc = desc
        self._update()

    def update(self, n: int = 1) -> None:
        self.n += n
        self._update()

    def set_postfix(self, values: Dict[str, str], refresh: bool = False) -> None:
        self.postfix = {k: str(v) for k, v in values.items()}
        if refresh:
            self._update()

    def refresh(self) -> None:
        self._update()

    def close(self) -> None:
        if self.widget is not None and self.output is not None:
            try:
                with self.output:
                    clear_output(wait=True)
            except Exception:
                pass
        elif self._display is not None:
            try:
                self._display.update(HTML(""))
            except Exception:
                pass
            if self.output is not None:
                try:
                    with self.output:
                        clear_output(wait=True)
                except Exception:
                    pass

    def _render(self) -> str:
        total = self.total
        progress = self.n
        pct = min(max((progress / total) if total else 0.0, 0.0), 1.0)
        pct_text = f"{pct * 100:5.1f}%" if total else ""
        elapsed = time.time() - self.start_time
        rate = progress / elapsed if elapsed > 0 else 0.0
        eta = (total - progress) / rate if rate > 0 and total else math.inf
        eta_text = format_duration(eta)
        elapsed_text = format_duration(elapsed)
        total_text = f"{progress}/{total}" if total else f"{progress}"
        postfix_parts = [f"{key}={value}" for key, value in self.postfix.items()]
        postfix_html = " · ".join(postfix_parts)
        bar_inner_width = pct * 100.0
        return f"""
        <div style="font-family: 'IBM Plex Mono', 'Menlo', monospace; color: {_TEXT_PRIMARY}; background: {_CARD_BG}; border-radius: 6px; padding: 8px 12px;">
            <div style="font-size: 13px; margin-bottom: 6px; color:{_TEXT_MUTED};"><strong>{self.desc}</strong> {pct_text}</div>
            <div style="position: relative; width: 100%; height: 12px; background: {_BAR_BG}; border-radius: 4px; overflow: hidden;">
                <div style="position: absolute; left: 0; top: 0; height: 100%; width: {bar_inner_width:.2f}%; background: linear-gradient(90deg, #4dd0e1, #5df5c9);"></div>
            </div>
            <div style="margin-top: 6px; font-size: 12px; color: {_TEXT_MUTED};">
                {total_text} · elapsed {elapsed_text} · eta {eta_text}
                {"· " + postfix_html if postfix_html else ""}
            </div>
        </div>
        """

    def _update(self) -> None:
        if self.widget is not None:
            try:
                self.widget.value = self._render()
            except Exception:
                pass
            return
        if self._display is not None:
            try:
                self._display.update(HTML(self._render()))
            except Exception:
                pass


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "--:--"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def ensure_dark_tqdm_theme() -> None:
    global _TQDM_CSS_INJECTED
    if _TQDM_CSS_INJECTED:
        return
    css = """
    <style>
        .dark-tqdm-container {{
            background-color: {surface_bg} !important;
            border: none !important;
            box-shadow: none !important;
            border-radius: 4px !important;
            padding: 2px 6px !important;
        }}
        .jp-OutputArea .widget-hbox,
        .jp-OutputArea .widget-inline-hbox,
        .jp-OutputArea .widget-box {{
            border: none !important;
            box-shadow: none !important;
            background-color: transparent !important;
        }}
        .jp-OutputArea {{
            border: none !important;
            box-shadow: none !important;
            background-color: {card_bg} !important;
            padding: 0 !important;
        }}
        .jp-OutputArea-output {{
            border: none !important;
            box-shadow: none !important;
            background-color: {card_bg} !important;
        }}
        .jp-Cell-outputWrapper {{
            border: none !important;
            box-shadow: none !important;
            background-color: {card_bg} !important;
            padding: 0 !important;
        }}
        .output_area, .output_wrapper, .output {{
            border: none !important;
            box-shadow: none !important;
            background-color: {card_bg} !important;
        }}
        .dark-tqdm-container .widget-html,
        .dark-tqdm-container .widget-label {{
            color: {text_muted} !important;
        }}
        .dark-tqdm-progress {{
            background-color: transparent !important;
        }}
        .dark-tqdm-progress .progress {{
            background-color: {bar_bg} !important;
            border: none !important;
            border-radius: 4px !important;
        }}
        .dark-tqdm-progress .progress-bar {{
            background-color: #4dd0e1 !important;
        }}
        .onn-dark-card {{
            background-color: {card_bg} !important;
            border: none !important;
            border-radius: 12px !important;
            box-shadow: 0 18px 34px rgba(2, 4, 10, 0.45) !important;
        }}
        .onn-dark-card .widget-box {{
            border: none !important;
            box-shadow: none !important;
            background-color: transparent !important;
        }}
        .onn-dark-card .widget-html {{
            color: {text_muted} !important;
        }}
        .onn-dark-card .widget-output {{
            background-color: {surface_bg} !important;
            border: none !important;
            border-radius: 8px !important;
        }}
        .widget-output {{
            border: none !important;
            background-color: transparent !important;
            padding: 0 !important;
        }}
    </style>
    """.format(
        surface_bg=_SURFACE_BG,
        card_bg=_CARD_BG,
        text_muted=_TEXT_MUTED,
        surface_border=_SURFACE_BORDER,
        bar_bg=_BAR_BG,
    )
    try:
        display(HTML(css))
        _TQDM_CSS_INJECTED = True
    except Exception:
        pass


def init_metrics_plot_state(
    log_scale: bool = False,
    output_widget: Optional[object] = None,
    channel_num: Optional[int] = None,
    show_accuracy: bool = True,
) -> Dict[str, object]:
    return {
        "handle": None,
        "fig": None,
        "axes": None,
        "ax_loss": None,
        "ax_acc": None,
        "loss_train_line": None,
        "loss_eval_line": None,
        "acc_train_line": None,
        "acc_eval_line": None,
        "loss_train_x": [],
        "loss_train_y": [],
        "loss_eval_x": [],
        "loss_eval_y": [],
        "acc_train_x": [],
        "acc_train_y": [],
        "acc_eval_x": [],
        "acc_eval_y": [],
        "last_update": -1,
        "log_scale": log_scale,
        "show_accuracy": bool(show_accuracy),
        "output": output_widget,
    }


def update_metrics_plot(
    state: Dict[str, object],
    iteration: int,
    update_every: int,
    *,
    train_loss: Optional[float] = None,
    eval_loss: Optional[float] = None,
    train_acc: Optional[float] = None,
    eval_acc: Optional[float] = None,
    train_acc_per_channel: Optional[List[float]] = None,
    eval_acc_per_channel: Optional[List[float]] = None,
    force: bool = False,
) -> None:
    output_widget = state.get("output")
    if train_loss is not None:
        state["loss_train_x"].append(iteration)
        state["loss_train_y"].append(train_loss)
    if eval_loss is not None:
        state["loss_eval_x"].append(iteration)
        state["loss_eval_y"].append(eval_loss)
    if train_acc is not None:
        state["acc_train_x"].append(iteration)
        state["acc_train_y"].append(train_acc * 100.0)
    if eval_acc is not None:
        state["acc_eval_x"].append(iteration)
        state["acc_eval_y"].append(eval_acc * 100.0)

    if state["handle"] is None:
        show_accuracy = bool(state.get("show_accuracy", True))
        if show_accuracy:
            fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
            ax_loss, ax_acc = axes
            axes_arr = np.atleast_1d(axes)
        else:
            fig, ax_loss = plt.subplots(1, 1, figsize=(6.4, 4.2))
            ax_acc = None
            axes = np.array([ax_loss], dtype=object)
            axes_arr = np.atleast_1d(axes)
        fig.patch.set_facecolor(_CARD_BG)
        for ax in axes_arr:
            ax.set_facecolor(_CARD_BG)
            ax.tick_params(colors=_TEXT_PRIMARY, labelsize=11)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight("bold")
            ax.grid(True, alpha=0.35, color="#2e394f")
        ax_loss.set_title("ONN Training & Eval Loss", color=_TEXT_PRIMARY, fontsize=14, fontweight="bold")
        ax_loss.set_xlabel("Iteration", color=_TEXT_PRIMARY, fontsize=12, fontweight="bold")
        ax_loss.set_ylabel("Loss", color=_TEXT_PRIMARY, fontsize=12, fontweight="bold")
        if state["log_scale"]:
            ax_loss.set_yscale("log")
        (loss_train_line,) = ax_loss.plot(
            [], [], label="train", color="#4dd0e1", linewidth=1.8, marker="o", markersize=3
        )
        (loss_eval_line,) = ax_loss.plot(
            [], [], label="eval", color="#ffd54f", linewidth=1.8, marker="s", markersize=3
        )
        leg_loss = ax_loss.legend(loc="upper right")
        if leg_loss:
            frame = leg_loss.get_frame()
            frame.set_facecolor(_CARD_BG)
            frame.set_edgecolor("#2e394f")
            for txt in leg_loss.get_texts():
                txt.set_color(_TEXT_PRIMARY)
                txt.set_fontweight("bold")
                txt.set_fontsize(11)

        acc_train_line = None
        acc_eval_line = None
        if show_accuracy and ax_acc is not None:
            ax_acc.set_title("Classification Accuracy", color=_TEXT_PRIMARY, fontsize=14, fontweight="bold")
            ax_acc.set_xlabel("Iteration", color=_TEXT_PRIMARY, fontsize=12, fontweight="bold")
            ax_acc.set_ylabel("Accuracy (%)", color=_TEXT_PRIMARY, fontsize=12, fontweight="bold")
            (acc_train_line,) = ax_acc.plot(
                [], [], label="train", color="#81d4fa", linewidth=1.8, marker="o", markersize=3
            )
            (acc_eval_line,) = ax_acc.plot(
                [], [], label="eval", color="#ffe082", linewidth=1.8, marker="s", markersize=3
            )
            leg_acc = ax_acc.legend(loc="lower right")
            if leg_acc:
                frame = leg_acc.get_frame()
                frame.set_facecolor(_CARD_BG)
                frame.set_edgecolor("#2e394f")
                for txt in leg_acc.get_texts():
                    txt.set_color(_TEXT_PRIMARY)
                    txt.set_fontweight("bold")
                    txt.set_fontsize(11)

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
                "axes": axes,
                "ax_loss": ax_loss,
                "ax_acc": ax_acc,
                "loss_train_line": loss_train_line,
                "loss_eval_line": loss_eval_line,
                "acc_train_line": acc_train_line,
                "acc_eval_line": acc_eval_line,
                "output": output_widget,
            }
        )
        state["last_update"] = -1

    last = state.get("last_update", -1)
    if last == -1:
        force = True
    if not force and iteration - last < update_every:
        return

    ax_loss = state["ax_loss"]
    ax_acc = state["ax_acc"]
    state["loss_train_line"].set_data(state["loss_train_x"], state["loss_train_y"])
    state["loss_eval_line"].set_data(state["loss_eval_x"], state["loss_eval_y"])
    if state.get("show_accuracy", True):
        if state["acc_train_line"] is not None:
            state["acc_train_line"].set_data(state["acc_train_x"], state["acc_train_y"])
        if state["acc_eval_line"] is not None:
            state["acc_eval_line"].set_data(state["acc_eval_x"], state["acc_eval_y"])

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

    if state.get("show_accuracy", True) and ax_acc is not None:
        acc_x = state["acc_train_x"] + state["acc_eval_x"]
        if acc_x:
            xmin, xmax = min(acc_x), max(acc_x)
            if xmin == xmax:
                xmin -= 1
                xmax += 1
            ax_acc.set_xlim(xmin, xmax + 1)
        acc_y = state["acc_train_y"] + state["acc_eval_y"]
        if acc_y:
            ax_acc.relim()
            ax_acc.autoscale_view(scalex=False, scaley=True)

    state["fig"].canvas.draw_idle()
    handle = state.get("handle")
    if handle is not None and hasattr(handle, "update"):
        handle.update(state["fig"])
    state["last_update"] = iteration


def compute_tile_boxes(mask_tensor: torch.Tensor) -> List[Tuple[int, int, int, int, int]]:
    mask_np = mask_tensor.squeeze(1).detach().cpu().numpy()
    boxes: List[Tuple[int, int, int, int, int]] = []
    for tile_idx, mask in enumerate(mask_np):
        coords = np.argwhere(mask > 0.5)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        boxes.append((tile_idx, int(r0), int(c0), int(r1), int(c1)))
    return boxes


def init_sample_viz_state(output_widget: Optional[object] = None) -> Dict[str, object]:
    return {
        "handle": None,
        "fig": None,
        "axes": None,
        "input_images": [],
        "output_images": [],
        "rects": [],
        "num_examples": 0,
        "tile_boxes": [],
        "output": output_widget,
    }


def update_sample_viz(
    state: Dict[str, object],
    encoded_samples: torch.Tensor,
    y_samples: torch.Tensor,
    labels_cpu: torch.Tensor,
    preds_cpu: torch.Tensor,
    tile_boxes: List[Tuple[int, int, int, int, int]],
    output_overlay: Optional[torch.Tensor] = None,
    sample_roles: Optional[List[str]] = None,
    input_grid_shape: Optional[Tuple[int, int]] = None,
    output_channel_ids: Optional[List[int]] = None,
) -> None:
    output_widget = state.get("output")
    if encoded_samples.dim() == 3:
        encoded_samples = encoded_samples.unsqueeze(1)
    if y_samples.dim() == 3:
        y_samples = y_samples.unsqueeze(1)
    num_examples = encoded_samples.size(0)
    if num_examples <= 0:
        return
    n_out = int(y_samples.size(1))
    if output_channel_ids is None or len(output_channel_ids) < n_out:
        output_channel_ids = list(range(n_out))
    output_channel_ids = list(output_channel_ids)[:n_out]
    def _limits(arr: np.ndarray) -> Tuple[float, float]:
        vmin = float(arr.min())
        vmax = float(arr.max())
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1e-6
        return vmin, vmax

    if (
        state["fig"] is None
        or state["num_examples"] != num_examples
        or state.get("n_out") != n_out
        or state.get("input_grid_shape") != input_grid_shape
    ):
        fig = state.get("fig")
        if fig is None:
            fig, axes = plt.subplots(
                1, num_examples * (1 + n_out), figsize=(3.2 * num_examples * (1 + n_out), 3.0)
            )
        else:
            fig.clf()
            axes = fig.subplots(1, num_examples * (1 + n_out), figsize=(3.2 * num_examples * (1 + n_out), 3.0))
        axes = np.atleast_1d(axes)
        fig.patch.set_facecolor(_CARD_BG)
        input_handles: List[plt.AxesImage] = []
        output_handles: List[List[plt.AxesImage]] = []
        input_colorbars: List[plt.colorbar] = []
        output_colorbars: List[List[plt.colorbar]] = []
        rects_all: List[List[List[plt.Rectangle]]] = []
        overlay_handles: List[List[Optional[plt.AxesImage]]] = []
        grid_lines_all: List[List[plt.Line2D]] = []
        highlight_col = "#ffb347"
        default_col = "#66cdaa"
        for idx in range(num_examples):
            role_raw = sample_roles[idx] if sample_roles and idx < len(sample_roles) else "sample"
            role = str(role_raw).lower()
            if role in ("neg", "negative"):
                role_name = "neg"
                label_name = "wrong"
                show_pred = False
            elif role in ("pos", "positive"):
                role_name = "pos"
                label_name = "true"
                show_pred = True
            else:
                role_name = "sample"
                label_name = "label"
                show_pred = True
            base_col = idx * (1 + n_out)
            ax_in = axes[base_col]
            ax_out_list = [axes[base_col + 1 + out_idx] for out_idx in range(n_out)]
            for ax in [ax_in, *ax_out_list]:
                ax.set_facecolor(_CARD_BG)
                ax.axis("off")
            in_arr = encoded_samples[idx, 0].detach().cpu().numpy()
            vmin_in, vmax_in = _limits(in_arr)
            im_in = ax_in.imshow(in_arr, cmap="gray", vmin=vmin_in, vmax=vmax_in)
            cbar_in = fig.colorbar(im_in, ax=ax_in, fraction=0.046, pad=0.02)
            _style_colorbar(cbar_in)
            grid_lines: List[plt.Line2D] = []
            if input_grid_shape is not None:
                grid_rows, grid_cols = input_grid_shape
                H_in, W_in = in_arr.shape
                for r in range(1, int(grid_rows)):
                    y = (r * H_in / grid_rows) - 0.5
                    grid_lines.append(ax_in.axhline(y, color=default_col, linewidth=1.0, alpha=0.7))
                for c in range(1, int(grid_cols)):
                    x = (c * W_in / grid_cols) - 0.5
                    grid_lines.append(ax_in.axvline(x, color=default_col, linewidth=1.0, alpha=0.7))
            out_imgs: List[plt.AxesImage] = []
            out_cbars: List[plt.colorbar] = []
            out_overlays: List[Optional[plt.AxesImage]] = []
            rects_per_out: List[List[plt.Rectangle]] = []
            pred_label = int(preds_cpu[idx])
            for out_idx, ax_out in enumerate(ax_out_list):
                out_arr = y_samples[idx, out_idx].detach().cpu().numpy()
                vmin_out, vmax_out = _limits(out_arr)
                im_out = ax_out.imshow(out_arr, cmap=_PHASE_CMAP, vmin=vmin_out, vmax=vmax_out)
                cbar_out = fig.colorbar(im_out, ax=ax_out, fraction=0.046, pad=0.02)
                _style_colorbar(cbar_out)
                im_overlay: Optional[plt.AxesImage] = None
                if output_overlay is not None:
                    overlay_arr = None
                    if output_overlay.dim() == 4:
                        if output_overlay.size(1) == n_out:
                            overlay_arr = output_overlay[idx, out_idx].detach().cpu().numpy()
                        else:
                            overlay_arr = output_overlay[idx, 0].detach().cpu().numpy()
                    else:
                        overlay_arr = output_overlay[idx].detach().cpu().numpy()
                    if overlay_arr is None:
                        overlay_arr = np.zeros_like(out_arr)
                    im_overlay = ax_out.imshow(
                        overlay_arr,
                        cmap="Reds",
                        alpha=0.5,
                        vmin=0.0,
                        vmax=1.0,
                        interpolation="nearest",
                        zorder=3,
                    )
                channel_id = int(output_channel_ids[out_idx]) if out_idx < len(output_channel_ids) else out_idx
                if show_pred and out_idx == 0:
                    ax_out.set_title(
                        f"ONN output ({role_name}) #{idx+1}\nch={channel_id} pred={pred_label}",
                        color=_TEXT_PRIMARY,
                    )
                else:
                    ax_out.set_title(
                        f"ONN output ({role_name}) #{idx+1}\nch={channel_id}",
                        color=_TEXT_PRIMARY,
                    )
                rects: List[plt.Rectangle] = []
                for tile_idx, r0, c0, r1, c1 in tile_boxes:
                    color = highlight_col if tile_idx == pred_label else default_col
                    rect = plt.Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, edgecolor=color, linewidth=1.5)
                    ax_out.add_patch(rect)
                    rects.append(rect)
                out_imgs.append(im_out)
                out_overlays.append(im_overlay)
                out_cbars.append(cbar_out)
                rects_per_out.append(rects)
            ax_in.set_title(
                f"Optical input ({role_name}) #{idx+1}\n{label_name}={int(labels_cpu[idx])}",
                color=_TEXT_PRIMARY,
            )
            input_handles.append(im_in)
            input_colorbars.append(cbar_in)
            output_handles.append(out_imgs)
            output_colorbars.append(out_cbars)
            overlay_handles.append(out_overlays)
            rects_all.append(rects_per_out)
            grid_lines_all.append(grid_lines)
        fig.tight_layout(pad=1.0)
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
                "input_images": input_handles,
                "output_images": output_handles,
                "input_colorbars": input_colorbars,
                "output_colorbars": output_colorbars,
                "rects": rects_all,
                "overlay_images": overlay_handles,
                "num_examples": num_examples,
                "n_out": n_out,
                "tile_boxes": tile_boxes,
                "output": output_widget,
                "input_grid_shape": input_grid_shape,
                "input_grid_lines": grid_lines_all,
            }
        )
    else:
        highlight_col = "#ffb347"
        default_col = "#66cdaa"
        for idx in range(num_examples):
            role_raw = sample_roles[idx] if sample_roles and idx < len(sample_roles) else "sample"
            role = str(role_raw).lower()
            if role in ("neg", "negative"):
                role_name = "neg"
                label_name = "wrong"
                show_pred = False
            elif role in ("pos", "positive"):
                role_name = "pos"
                label_name = "true"
                show_pred = True
            else:
                role_name = "sample"
                label_name = "label"
                show_pred = True
            im_in = state["input_images"][idx]
            cbar_in = state["input_colorbars"][idx] if "input_colorbars" in state else None
            in_arr = encoded_samples[idx, 0].detach().cpu().numpy()
            vmin_in, vmax_in = _limits(in_arr)
            im_in.set_data(in_arr)
            im_in.set_clim(vmin_in, vmax_in)
            pred_label = int(preds_cpu[idx])
            if cbar_in is not None:
                cbar_in.mappable.set_clim(vmin_in, vmax_in)
                cbar_in.update_normal(im_in)
            out_imgs = state["output_images"][idx]
            out_cbars = state["output_colorbars"][idx]
            out_overlays = state["overlay_images"][idx] if "overlay_images" in state else []
            rects_per_out = state["rects"][idx]
            for out_idx, im_out in enumerate(out_imgs):
                out_arr = y_samples[idx, out_idx].detach().cpu().numpy()
                vmin_out, vmax_out = _limits(out_arr)
                im_out.set_data(out_arr)
                im_out.set_clim(vmin_out, vmax_out)
                if out_idx < len(out_cbars):
                    out_cbars[out_idx].mappable.set_clim(vmin_out, vmax_out)
                    out_cbars[out_idx].update_normal(im_out)
                if out_idx < len(out_overlays) and out_overlays[out_idx] is not None:
                    if output_overlay is not None:
                        if output_overlay.dim() == 4:
                            if output_overlay.size(1) == n_out:
                                overlay_arr = output_overlay[idx, out_idx].detach().cpu().numpy()
                            else:
                                overlay_arr = output_overlay[idx, 0].detach().cpu().numpy()
                        else:
                            overlay_arr = output_overlay[idx].detach().cpu().numpy()
                    else:
                        overlay_arr = np.zeros_like(out_arr)
                    out_overlays[out_idx].set_data(overlay_arr)
                if out_idx < len(rects_per_out):
                    for rect, (tile_idx, _, _, _, _) in zip(rects_per_out[out_idx], state["tile_boxes"]):
                        color = highlight_col if tile_idx == pred_label else default_col
                        rect.set_edgecolor(color)
            base_col = idx * (1 + n_out)
            ax_in = state["axes"][base_col]
            ax_in.set_title(
                f"Optical input ({role_name}) #{idx+1}\n{label_name}={int(labels_cpu[idx])}",
                color=_TEXT_PRIMARY,
            )
            for out_idx in range(n_out):
                ax_out = state["axes"][base_col + 1 + out_idx]
                channel_id = int(output_channel_ids[out_idx]) if out_idx < len(output_channel_ids) else out_idx
                if show_pred and out_idx == 0:
                    ax_out.set_title(
                        f"ONN output ({role_name}) #{idx+1}\nch={channel_id} pred={pred_label}",
                        color=_TEXT_PRIMARY,
                    )
                else:
                    ax_out.set_title(
                        f"ONN output ({role_name}) #{idx+1}\nch={channel_id}",
                        color=_TEXT_PRIMARY,
                    )
        state["fig"].canvas.draw_idle()
        handle = state.get("handle")
        if handle is not None and hasattr(handle, "update"):
            handle.update(state["fig"])


def init_phase_viz_state(output_widget: Optional[object] = None) -> Dict[str, object]:
    return {
        "handle": None,
        "fig": None,
        "axes": None,
        "images": [],
        "colorbars": [],
        "num_layers": 0,
        "rows": 0,
        "cols": 0,
        "vmin": 0.0,
        "vmax": 1.0,
        "output": output_widget,
    }


def init_confusion_viz_state(output_widget: Optional[object] = None) -> Dict[str, object]:
    return {
        "handle": None,
        "fig": None,
        "ax": None,
        "image": None,
        "colorbar": None,
        "shape": None,
        "output": output_widget,
    }


def update_confusion_viz(
    state: Dict[str, object],
    c_ema: torch.Tensor,
    iteration: int,
    *,
    normalize: bool = True,
) -> None:
    if c_ema is None:
        return
    mat = c_ema.detach().cpu().float().numpy()
    if mat.size == 0:
        return
    if normalize:
        row_sum = mat.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            mat = np.divide(mat, row_sum, out=np.zeros_like(mat), where=row_sum > 0)
    n = mat.shape[0]
    vmax = 1.0 if normalize else float(np.max(mat)) if np.isfinite(mat).any() else 1.0
    if vmax <= 0:
        vmax = 1.0
    output_widget = state.get("output")
    if state["fig"] is None or state["shape"] != mat.shape:
        fig, ax = plt.subplots(1, 1, figsize=(4.4, 4.0))
        fig.patch.set_facecolor(_CARD_BG)
        ax.set_facecolor(_CARD_BG)
        im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(f"Hard-negative EMA (iter {iteration})", color=_TEXT_PRIMARY)
        ax.set_xlabel("hard negative label", color=_TEXT_PRIMARY)
        ax.set_ylabel("true label", color=_TEXT_PRIMARY)
        step = max(1, n // 10)
        ticks = list(range(0, n, step))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_yticklabels([str(t) for t in ticks])
        _style_axes(ax, tick_labelsize=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        _style_colorbar(cb)
        fig.tight_layout(pad=0.8)
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
                "ax": ax,
                "image": im,
                "colorbar": cb,
                "shape": mat.shape,
                "output": output_widget,
            }
        )
    else:
        im = state["image"]
        cb = state["colorbar"]
        im.set_data(mat)
        im.set_clim(0.0, vmax)
        if cb is not None:
            cb.mappable.set_clim(0.0, vmax)
            cb.update_normal(im)
        ax = state["ax"]
        if ax is not None:
            ax.set_title(f"Hard-negative EMA (iter {iteration})", color=_TEXT_PRIMARY)
        state["fig"].canvas.draw_idle()
        handle = state.get("handle")
        if handle is not None and hasattr(handle, "update"):
            handle.update(state["fig"])


def update_phase_viz(
    state: Dict[str, object],
    phase_params: List[torch.Tensor],
    enc_canvas_hw: Tuple[int, int],
    iteration: int,
) -> None:
    """
    Visualise three representative phase channels (ch 1, ch 2, ch 3) for each
    layer. Each layer gets its own row; columns correspond to the chosen views.
    """
    output_widget = state.get("output")
    num_layers = len(phase_params)
    rows = max(1, num_layers)
    cols = 3  # fixed: show three channels per layer
    channel_indices = (0, 1, 2)  # 1-based titles: 1, 2, 3
    phase_titles: List[List[str]] = []
    phase_images: List[List[np.ndarray]] = []
    for layer_idx, tensor in enumerate(phase_params):
        arr = tensor.detach().cpu()
        if arr.dim() == 4:
            arr = arr[0]  # first batch element
        if arr.dim() == 2:
            arr = arr.unsqueeze(0)
        if arr.dim() != 3:
            continue
        imgs = []
        C = arr.size(0)
        idxs = [min(idx, C - 1) for idx in channel_indices]
        ch_imgs = [arr[idxs[0]].numpy(), arr[idxs[1]].numpy(), arr[idxs[2]].numpy()]
        imgs.extend(ch_imgs)
        phase_titles.append(
            [
                f"Phase mask L{layer_idx + 1}_1",
                f"Phase mask L{layer_idx + 1}_2",
                f"Phase mask L{layer_idx + 1}_3",
            ]
        )
        phase_images.append(imgs)
    # Phase channels stay in [0,1].
    vmins = (0.0, 0.0, 0.0)
    vmaxs = (1.0, 1.0, 1.0)
    if (
        state["fig"] is None
        or state["num_layers"] != num_layers
        or state["rows"] != rows
        or state["cols"] != cols
    ):
        fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.6 * rows), squeeze=False)
        fig.patch.set_facecolor(_CARD_BG)
        axes_grid = axes.reshape(rows, cols)
        img_handles: List[plt.AxesImage] = []
        colorbars: List[plt.colorbar] = []
        for layer_idx, img_list in enumerate(phase_images):
            for col_idx, img_arr in enumerate(img_list):
                ax = axes_grid[layer_idx, col_idx]
                ax.set_facecolor(_CARD_BG)
                im = ax.imshow(
                    img_arr,
                    cmap=_PHASE_CMAP,
                    interpolation="nearest",
                    vmin=vmins[col_idx],
                    vmax=vmaxs[col_idx],
                )
                ax.set_title(phase_titles[layer_idx][col_idx], color=_TEXT_PRIMARY)
                ax.axis("off")
                img_handles.append(im)
                cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
                cbar.ax.set_facecolor(_CARD_BG)
                cbar.ax.tick_params(labelsize=8, colors=_TEXT_PRIMARY)
                colorbars.append(cbar)
        for ax in axes_grid.flatten()[len(phase_images) * cols:]:
            ax.axis("off")
        fig.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.08, wspace=0.25, hspace=0.35)
        fig.canvas.draw_idle()
        if output_widget is not None:
            with output_widget:
                handle = display(fig, display_id=True)
        else:
            handle = display(fig, display_id=True)
        state.update(
            {
                "handle": handle,
                "fig": fig,
                "axes": axes_grid,
                "images": img_handles,
                "colorbars": colorbars,
                "num_layers": num_layers,
                "rows": rows,
                "cols": cols,
                "vmin": vmins,
                "vmax": vmaxs,
                "output": output_widget,
            }
        )
    else:
        img_idx = 0
        for layer_idx, img_list in enumerate(phase_images):
            for col_idx, img_arr in enumerate(img_list):
                im = state["images"][img_idx]
                im.set_data(img_arr)
                im.set_clim(vmins[col_idx], vmaxs[col_idx])
                im.axes.set_title(phase_titles[layer_idx][col_idx], color=_TEXT_PRIMARY)
                if img_idx < len(state.get("colorbars", [])):
                    cbar = state["colorbars"][img_idx]
                    cbar.mappable.set_clim(vmins[col_idx], vmaxs[col_idx])
                    cbar.update_normal(im)
                    cbar.ax.tick_params(labelsize=8, colors=_TEXT_PRIMARY)
                img_idx += 1
        state["fig"].canvas.draw_idle()
        handle = state.get("handle")
        if handle is not None and hasattr(handle, "update"):
            handle.update(state["fig"])
        state["vmin"] = vmins
        state["vmax"] = vmaxs
        state["num_layers"] = num_layers
        state["rows"] = rows
        state["cols"] = cols
