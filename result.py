import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
try:
    from ipywidgets import interact, SelectionSlider
except Exception:  # ipywidgets may be unavailable
    SelectionSlider = None
    interact = None
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def _apply_plot_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 16,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 14,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "figure.titlesize": 15,
            "axes.linewidth": 1.0,
            "lines.linewidth": 2.0,
            "axes.grid": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

_apply_plot_style()

TRAIN_COLOR = "#4C78A8"
TEST_COLOR = "#E45756"
ACCENT_COLOR = "#54A24B"
NEUTRAL_COLOR = "#6C6F7D"
PAPER_FIGSIZE = (14.0, 5.0)
PAPER_SUMMARY_FIGSIZE = (5.4, 3.2)
PAPER_3D_FIGSIZE = (17.2, 7.0)
ACCURACY_CMAP = "magma"


def _style_axes(ax):
    ax.grid(False)
    ax.spines["left"].set_color("#B0B0B0")
    ax.spines["bottom"].set_color("#B0B0B0")
    ax.spines["top"].set_color("#B0B0B0")
    ax.spines["right"].set_color("#B0B0B0")
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.tick_params(color="#7F7F7F", labelcolor="black", width=0.8, length=5)

_THIS_DIR = Path(__file__).resolve().parent
ckpt_path = str(
    _THIS_DIR / "pre_trained_model_save" / "Optical_neural_net" / "checkpoints" / "_mul_5_final.pt"
)
default_avg_batch = 50
avg_batch_options = [1, 8, 16, 32, 64, 128, 256, 512, 1024]

ckpt = {}
history = {}
iters = []
train_loss = []
test_loss = []
train_acc = []
test_acc = []
test_iters = []
train_acc_channels = []
test_acc_channels = []
test_loss_x = []
test_acc_x = []
_history_loaded = False

def _test_x_from_history(series):
    if not series:
        return []
    if test_iters:
        return test_iters[:len(series)]
    return list(range(len(series)))


def _set_history(new_history):
    global history, iters, train_loss, test_loss, train_acc, test_acc, test_iters
    global train_acc_channels, test_acc_channels, test_loss_x, test_acc_x
    history = new_history or {}
    iters = history.get("iters", [])
    train_loss = history.get("train_loss", [])
    test_loss = history.get("eval_loss", [])
    train_acc = history.get("train_acc", [])
    test_acc = history.get("eval_acc", [])
    test_iters = history.get("eval_iters", [])
    train_acc_channels = history.get("train_acc_channels", [])
    test_acc_channels = history.get("eval_acc_channels", [])
    test_loss_x = _test_x_from_history(test_loss)
    test_acc_x = _test_x_from_history(test_acc)


def load_checkpoint(path=None):
    global ckpt, _history_loaded
    p = Path(path or ckpt_path)
    if not p.is_absolute():
        p = _THIS_DIR / p
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    ckpt = torch.load(p, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Expected checkpoint dict, got {type(ckpt)}")
    _set_history(ckpt.get("history", {}))
    _history_loaded = True
    return ckpt

def _batch_average(values, x_vals=None, window=1):
    if not values:
        return [], []
    if window <= 1:
        return list(values), list(x_vals) if x_vals is not None else list(range(len(values)))
    out_vals = []
    out_x = []
    n = len(values)
    for start in range(0, n, window):
        chunk = values[start:start + window]
        finite = [v for v in chunk if v is not None and np.isfinite(v)]
        out_vals.append(float(np.mean(finite)) if finite else float("nan"))
        if x_vals is not None:
            x_chunk = x_vals[start:start + window]
            out_x.append(float(np.mean(x_chunk)) if x_chunk else float(start))
        else:
            out_x.append(float(start + (len(chunk) - 1) / 2))
    return out_vals, out_x

def _batch_average_x(x_vals, window=1):
    if not x_vals:
        return []
    if window <= 1:
        return list(x_vals)
    out_x = []
    n = len(x_vals)
    for start in range(0, n, window):
        chunk = x_vals[start:start + window]
        out_x.append(float(np.mean(chunk)))
    return out_x


def _truncate_series_by_iter(values, x_vals, end_iter=None):
    vals = list(values)
    xs = list(x_vals)[:len(vals)]
    vals = vals[:len(xs)]
    if end_iter is None:
        return vals, xs
    keep_n = 0
    for x in xs:
        if x <= float(end_iter):
            keep_n += 1
        else:
            break
    return vals[:keep_n], xs[:keep_n]


def _stack_channel_series(series):
    """Pad ragged per-channel series into (steps, channels) numpy array."""
    if not series:
        return None
    rows = []
    max_ch = 0
    for row in series:
        if row is None:
            rows.append(None)
            continue
        arr = np.array(row, dtype=float).reshape(-1)
        rows.append(arr)
        max_ch = max(max_ch, arr.size)
    if max_ch == 0:
        return None
    out = np.full((len(rows), max_ch), np.nan, dtype=float)
    for i, row in enumerate(rows):
        if row is None:
            continue
        out[i, :row.size] = row
    return out

def _batch_average_matrix(mat, window=1):
    if mat is None:
        return None
    if window <= 1:
        return mat
    rows = []
    n = mat.shape[0]
    for start in range(0, n, window):
        chunk = mat[start:start + window, :]
        with np.errstate(all="ignore"):
            rows.append(np.nanmean(chunk, axis=0))
    return np.vstack(rows)

def _infer_channel_count(*series_list):
    for series in series_list:
        if not series:
            continue
        for row in series:
            if row is None:
                continue
            arr = np.array(row, dtype=float).reshape(-1)
            if arr.size:
                return int(arr.size)
    return 0

def _heatmap_extent_from_x(x_vals, channel_num):
    if not x_vals:
        return (0.0, 1.0, -0.5, max(channel_num - 0.5, 0.5))
    if len(x_vals) == 1:
        step = 1.0
    else:
        diffs = np.diff(np.array(x_vals, dtype=float))
        step = float(np.median(diffs)) if diffs.size else 1.0
        if not np.isfinite(step) or step <= 0:
            step = 1.0
    return (
        float(x_vals[0]) - 0.5 * step,
        float(x_vals[-1]) + 0.5 * step,
        -0.5,
        float(channel_num) - 0.5,
    )

def _heatmap_ticks(x_vals, max_ticks=6):
    if not x_vals:
        return []
    if len(x_vals) <= max_ticks:
        return list(x_vals)
    idx = np.linspace(0, len(x_vals) - 1, num=max_ticks, dtype=int)
    return [x_vals[i] for i in idx]

def _set_heatmap_ticks(ax, x_vals, channel_num):
    xticks = _heatmap_ticks(x_vals)
    if xticks:
        ax.set_xticks(xticks)
    yticks = list(range(channel_num))
    if channel_num > 20:
        step = max(1, channel_num // 16)
        yticks = list(range(0, channel_num, step))
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(i + 1) for i in yticks])

def _style_3d_axes(ax):
    pane_color = (1.0, 1.0, 1.0, 1.0)
    grid_color = (0.72, 0.72, 0.72, 0.35)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(pane_color)
        axis.pane.set_edgecolor("#B0B0B0")
        axis._axinfo["grid"]["color"] = grid_color
        axis._axinfo["grid"]["linewidth"] = 0.6
    ax.tick_params(colors="black", width=0.8, length=4, pad=1)
    ax.tick_params(axis="x", labelsize=11, pad=0)
    ax.tick_params(axis="y", labelsize=12, pad=1)
    ax.tick_params(axis="z", labelsize=12, pad=1)

def _prepare_train_acc_channel_matrix(end_iter=None):
    train_acc_mat = _stack_channel_series(train_acc_channels)
    if train_acc_mat is None:
        return None, []

    train_x_raw = iters[:train_acc_mat.shape[0]]
    if len(train_x_raw) < train_acc_mat.shape[0]:
        train_x_raw = list(range(train_acc_mat.shape[0]))

    if end_iter is not None:
        keep_n = 0
        for x in train_x_raw:
            if x <= float(end_iter):
                keep_n += 1
            else:
                break
        train_acc_mat = train_acc_mat[:keep_n, :]
        train_x_raw = train_x_raw[:keep_n]

    if train_acc_mat.size == 0:
        return None, []
    return train_acc_mat * 100.0, train_x_raw

def _prepare_test_acc_channel_matrix(end_iter=None):
    test_acc_mat = _stack_channel_series(test_acc_channels)

    if test_acc_mat is None:
        channel_num = _infer_channel_count(train_acc_channels, test_acc_channels)
        if channel_num <= 0 or not test_acc:
            return None, []
        vals = np.array(test_acc, dtype=float).reshape(-1, 1)
        test_acc_mat = np.repeat(vals, channel_num, axis=1)

    test_x_raw = test_acc_x[:test_acc_mat.shape[0]]
    if len(test_x_raw) < test_acc_mat.shape[0]:
        test_x_raw = test_iters[:test_acc_mat.shape[0]] if test_iters else list(range(test_acc_mat.shape[0]))

    if end_iter is not None:
        keep_n = 0
        for x in test_x_raw:
            if x <= float(end_iter):
                keep_n += 1
            else:
                break
        test_acc_mat = test_acc_mat[:keep_n, :]
        test_x_raw = test_x_raw[:keep_n]

    if test_acc_mat.size == 0:
        return None, []
    return test_acc_mat * 100.0, test_x_raw

def _set_channel_ticks_3d(ax, channel_num):
    if channel_num <= 0:
        return
    if channel_num <= 20:
        ticks = np.arange(1, channel_num + 1)
    else:
        step = max(1, channel_num // 10)
        ticks = np.arange(1, channel_num + 1, step)
        if ticks[-1] != channel_num:
            ticks = np.append(ticks, channel_num)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(tick)) for tick in ticks])

def _nice_iteration_step(max_value, target_intervals=5):
    if not np.isfinite(max_value) or max_value <= 0:
        return 1.0
    raw_step = float(max_value) / float(target_intervals)
    exponent = np.floor(np.log10(raw_step))
    base = 10.0 ** exponent
    fraction = raw_step / base
    if fraction <= 1.0:
        nice_fraction = 1.0
    elif fraction <= 2.0:
        nice_fraction = 2.0
    elif fraction <= 3.0:
        nice_fraction = 2.5
    elif fraction <= 7.5:
        nice_fraction = 5.0
    else:
        nice_fraction = 10.0
    return nice_fraction * base

def _set_iteration_ticks_3d(ax, y_min, y_max):
    if not np.isfinite(y_max) or y_max <= 0:
        ax.set_yticks([0])
        return
    step = _nice_iteration_step(y_max)
    upper = float(np.ceil(y_max / step) * step)
    ticks = np.arange(0.0, upper + 0.5 * step, step)
    ax.set_ylim(min(0.0, float(y_min)), upper)
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(int(round(tick))) for tick in ticks])

def _coerce_metric_matrix(metric_mat, x_vals):
    if metric_mat is None or not x_vals:
        return None, None

    z_vals = np.array(metric_mat, dtype=float)
    if z_vals.ndim == 1:
        z_vals = z_vals.reshape(-1, 1)
    if z_vals.size == 0:
        return None, None

    step_count = z_vals.shape[0]
    y_vals = np.array(x_vals[:step_count], dtype=float)
    if y_vals.size != step_count:
        step_count = min(step_count, y_vals.size)
        z_vals = z_vals[:step_count, :]
        y_vals = y_vals[:step_count]
    if step_count == 0:
        return None, None
    return z_vals, y_vals

def _plot_metric_surface(
    ax,
    metric_mat,
    x_vals,
    z_label,
    cmap,
    z_limits=None,
):
    z_vals, y_vals = _coerce_metric_matrix(metric_mat, x_vals)
    if z_vals is None:
        return None

    step_count, channel_num = z_vals.shape
    channels = np.arange(1, channel_num + 1, dtype=float)
    channel_grid, iter_grid = np.meshgrid(channels, y_vals)
    finite_vals = z_vals[np.isfinite(z_vals)]
    if finite_vals.size == 0:
        return None

    z_plot = np.ma.masked_invalid(z_vals)
    zmin = float(np.min(finite_vals))
    zmax = float(np.max(finite_vals))
    if z_limits is not None:
        zmin, zmax = z_limits
    if np.isclose(zmin, zmax):
        pad = max(abs(zmax) * 0.05, 1e-3)
        zmin -= pad
        zmax += pad

    surface = ax.plot_surface(
        channel_grid,
        iter_grid,
        z_plot,
        cmap=cmap,
        vmin=zmin,
        vmax=zmax,
        rcount=min(step_count, 900),
        ccount=channel_num,
        linewidth=0.0,
        antialiased=True,
        shade=True,
    )

    ax.set_xlabel("Channel", fontsize=17, labelpad=10)
    ax.set_ylabel("Iteration", fontsize=17, labelpad=12)
    ax.set_zlabel("")
    _set_channel_ticks_3d(ax, channel_num)
    _set_iteration_ticks_3d(ax, float(np.nanmin(y_vals)), float(np.nanmax(y_vals)))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    ax.zaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.view_init(elev=26, azim=-60)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.6, 1.9, 0.72))
    ax.set_xlim(0.5, channel_num + 0.5)
    if z_limits is not None:
        ax.set_zlim(*z_limits)
    _style_3d_axes(ax)
    return surface

def _add_3d_zlabel(ax, label="Accuracy", x=1.02, y=0.60):
    ax.text2D(
        x,
        y,
        label,
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=14,
        color="black",
        clip_on=False,
    )

def _annotate_best_channel_accuracy(
    ax,
    edge_metric_mat,
    edge_x_vals,
    best_metric_mat=None,
    best_x_vals=None,
    summary="best",
):
    edge_vals, edge_y_vals = _coerce_metric_matrix(edge_metric_mat, edge_x_vals)
    if edge_vals is None:
        return

    best_vals_mat, _ = _coerce_metric_matrix(
        best_metric_mat if best_metric_mat is not None else edge_metric_mat,
        best_x_vals if best_x_vals is not None else edge_x_vals,
    )
    if best_vals_mat is None:
        return

    if summary == "last":
        values = np.full(best_vals_mat.shape[1], np.nan, dtype=float)
        for ch_idx in range(best_vals_mat.shape[1]):
            col = best_vals_mat[:, ch_idx]
            finite_idx = np.flatnonzero(np.isfinite(col))
            if finite_idx.size:
                values[ch_idx] = col[finite_idx[-1]]
    else:
        with np.errstate(all="ignore"):
            values = np.nanmax(best_vals_mat, axis=0)
    if values.size == 0 or not np.any(np.isfinite(values)):
        return

    channel_num = values.size
    channels = np.arange(1, channel_num + 1, dtype=float)
    y_span = float(np.nanmax(edge_y_vals) - np.nanmin(edge_y_vals)) if edge_y_vals.size > 1 else 1.0
    if not np.isfinite(y_span) or y_span <= 0:
        y_span = 1.0
    y_edge = float(edge_y_vals[-1]) + 0.035 * y_span

    for channel, value in zip(channels, values):
        if not np.isfinite(value):
            continue
        label = f"{value:.1f}"
        z_text = min(float(value) + 2.0, 101.5)
        ax.text(
            channel,
            y_edge,
            z_text,
            label,
            zdir="z",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="semibold",
            color="#111111",
        )

    y_min = float(np.nanmin(edge_y_vals))
    _set_iteration_ticks_3d(ax, y_min, y_edge + 0.045 * y_span)
    ax.set_zlim(0.0, 108.0)
    ax.set_zticks([0, 20, 40, 60, 80, 100])

def _plot_accuracy_surfaces(
    train_acc_mat,
    train_acc_x_vals,
    test_acc_mat,
    test_acc_x_vals,
    test_acc_best_mat=None,
    test_acc_best_x_vals=None,
):
    has_acc = (train_acc_mat is not None and train_acc_x_vals) or (test_acc_mat is not None and test_acc_x_vals)
    if not has_acc:
        return

    fig = plt.figure(figsize=PAPER_3D_FIGSIZE)
    axes = [
        fig.add_axes([0.000, 0.020, 0.540, 0.900], projection="3d"),
        fig.add_axes([0.430, 0.020, 0.510, 0.900], projection="3d"),
    ]

    train_surface = _plot_metric_surface(
        axes[0],
        train_acc_mat,
        train_acc_x_vals,
        z_label="Accuracy",
        cmap=ACCURACY_CMAP,
        z_limits=(0.0, 100.0),
    )
    axes[0].set_title("Train accuracy", fontsize=18, pad=10)
    _add_3d_zlabel(axes[0], x=1.075)

    test_surface = _plot_metric_surface(
        axes[1],
        test_acc_mat,
        test_acc_x_vals,
        z_label="Accuracy",
        cmap=ACCURACY_CMAP,
        z_limits=(0.0, 100.0),
    )
    _annotate_best_channel_accuracy(
        axes[1],
        test_acc_mat,
        test_acc_x_vals,
        best_metric_mat=test_acc_best_mat,
        best_x_vals=test_acc_best_x_vals,
        summary="best",
    )
    axes[1].set_title("Test accuracy", fontsize=18, pad=10)
    _add_3d_zlabel(axes[1], x=1.075)

    if train_surface is None:
        axes[0].set_axis_off()
    if test_surface is None:
        axes[1].set_axis_off()

    color_surface = train_surface if train_surface is not None else test_surface
    if color_surface is not None:
        cax = fig.add_axes([0.392, 0.928, 0.216, 0.024])
        cbar = fig.colorbar(color_surface, cax=cax, orientation="horizontal")
        cbar.set_label("Accuracy (%)", fontsize=13, labelpad=9)
        cbar.ax.xaxis.set_label_position("top")
        cbar.ax.tick_params(labelsize=10, length=3, pad=1)

    plt.show()

def plot_history(avg_batch=1, end_iter=None):
    if not _history_loaded:
        load_checkpoint()

    # Average train loss/acc over avg_batch
    train_loss_use, train_loss_x_raw = _truncate_series_by_iter(train_loss, iters[:len(train_loss)], end_iter)
    train_acc_use, train_acc_x_raw = _truncate_series_by_iter(train_acc, iters[:len(train_acc)], end_iter)
    train_loss_avg, train_loss_x = _batch_average(train_loss_use, train_loss_x_raw, avg_batch)
    train_acc_avg, train_acc_x = _batch_average(train_acc_use, train_acc_x_raw, avg_batch)

    # Test series use eval_iters from checkpoint (no inference)
    test_loss_use, test_loss_x_use = _truncate_series_by_iter(
        test_loss[:len(test_loss_x)],
        test_loss_x,
        end_iter,
    )
    test_acc_use, test_acc_x_use = _truncate_series_by_iter(
        test_acc[:len(test_acc_x)],
        test_acc_x,
        end_iter,
    )

    fig, ax_loss = plt.subplots(figsize=PAPER_SUMMARY_FIGSIZE)
    ax_acc = ax_loss.twinx()

    train_loss_markevery = max(1, len(train_loss_x) // 80) if train_loss_x else 1
    train_acc_markevery = max(1, len(train_acc_x) // 80) if train_acc_x else 1

    loss_train_line, = ax_loss.plot(
        train_loss_x,
        train_loss_avg,
        label="train loss",
        color="#78AADD",
        linestyle="-",
        marker="o",
        markersize=2.8,
        markevery=train_loss_markevery,
        linewidth=1.7,
        alpha=0.95,
    )
    loss_test_line, = ax_loss.plot(
        test_loss_x_use,
        test_loss_use,
        label="test loss",
        color="#FF7A73",
        linestyle="-",
        marker="s",
        markersize=3.0,
        linewidth=1.7,
        alpha=0.95,
    )
    acc_train_line, = ax_acc.plot(
        train_acc_x,
        train_acc_avg,
        label="train accuracy",
        color=TRAIN_COLOR,
        linestyle="-",
        marker="o",
        markersize=2.8,
        markevery=train_acc_markevery,
        linewidth=1.6,
        alpha=0.88,
    )
    acc_test_line, = ax_acc.plot(
        test_acc_x_use,
        test_acc_use,
        label="test accuracy",
        color=TEST_COLOR,
        linestyle="-",
        marker="s",
        markersize=3.0,
        linewidth=1.6,
        alpha=0.88,
    )

    ax_loss.set_xlabel("Iteration", fontsize=12)
    ax_loss.set_ylabel("Loss", fontsize=12)
    ax_acc.set_ylabel("Accuracy", fontsize=12)
    _style_axes(ax_loss)
    _style_axes(ax_acc)
    ax_acc.spines["left"].set_visible(False)
    ax_loss.spines["right"].set_visible(False)
    ax_loss.tick_params(axis="both", labelsize=11)
    ax_acc.tick_params(axis="y", color="#7F7F7F", labelcolor="black", width=0.8, length=5, labelsize=11)

    lines = [loss_train_line, loss_test_line, acc_train_line, acc_test_line]
    ax_loss.legend(
        lines,
        [line.get_label() for line in lines],
        frameon=False,
        loc="center right",
        fontsize=9,
        handlelength=1.7,
        borderaxespad=0.25,
        labelspacing=0.45,
    )

    fig.subplots_adjust(left=0.14, right=0.85, bottom=0.22, top=0.94)
    plt.show()

    train_acc_mat, train_acc_channel_x = _prepare_train_acc_channel_matrix(end_iter)
    test_acc_mat, test_acc_channel_x = _prepare_test_acc_channel_matrix(end_iter)
    train_acc_mat_avg = _batch_average_matrix(train_acc_mat, avg_batch)
    train_acc_channel_x_avg = _batch_average_x(train_acc_channel_x, avg_batch)
    _plot_accuracy_surfaces(
        train_acc_mat_avg,
        train_acc_channel_x_avg,
        test_acc_mat,
        test_acc_channel_x,
        test_acc_best_mat=test_acc_mat,
        test_acc_best_x_vals=test_acc_channel_x,
    )

    # Per-channel accuracy heatmaps (if present)
    train_mat = _stack_channel_series(train_acc_channels)
    test_mat = _stack_channel_series(test_acc_channels)
    if train_mat is None and test_mat is None:
        return

    fig_hm, (ax_train, ax_test) = plt.subplots(1, 2, figsize=PAPER_FIGSIZE)

    if train_mat is not None:
        train_x_raw = iters[:train_mat.shape[0]]
        if end_iter is not None:
            keep_n = 0
            for x in train_x_raw:
                if x <= float(end_iter):
                    keep_n += 1
                else:
                    break
            train_mat = train_mat[:keep_n, :]
            train_x_raw = train_x_raw[:keep_n]
        if train_mat.size == 0:
            train_mat = None
    if train_mat is not None:
        train_mat_avg = _batch_average_matrix(train_mat, avg_batch)
        train_x_avg = _batch_average_x(train_x_raw, avg_batch)
        extent = _heatmap_extent_from_x(train_x_avg, train_mat_avg.shape[1])
        im = ax_train.imshow(
            train_mat_avg.T * 100.0,
            origin="lower",
            aspect="auto",
            cmap="YlGnBu",
            extent=extent,
            vmin=0.0,
            vmax=100.0,
        )
        ax_train.set_xlabel("Iteration")
        ax_train.set_ylabel("Channel")
        _set_heatmap_ticks(ax_train, train_x_avg, train_mat_avg.shape[1])
        fig_hm.colorbar(im, ax=ax_train, fraction=0.046, pad=0.02, label="Accuracy(%)")
        _style_axes(ax_train)
    else:
        ax_train.text(0.5, 0.5, "train per-channel accuracy not found", ha="center", va="center")
        ax_train.set_axis_off()

    if test_mat is not None:
        test_x_raw = test_iters[:test_mat.shape[0]] if test_iters else list(range(test_mat.shape[0]))
        if end_iter is not None:
            keep_n = 0
            for x in test_x_raw:
                if x <= float(end_iter):
                    keep_n += 1
                else:
                    break
            test_mat = test_mat[:keep_n, :]
            test_x_raw = test_x_raw[:keep_n]
        if test_mat.size > 0:
            test_mat_avg = _batch_average_matrix(test_mat, avg_batch)
            test_x_avg = _batch_average_x(test_x_raw, avg_batch)
        else:
            test_mat_avg = None
            test_x_avg = []
    else:
        test_mat_avg = None
        test_x_avg = []
    if test_mat_avg is not None:
        extent = _heatmap_extent_from_x(test_x_avg, test_mat_avg.shape[1])
        im = ax_test.imshow(
            test_mat_avg.T * 100.0,
            origin="lower",
            aspect="auto",
            cmap="YlGnBu",
            extent=extent,
            vmin=0.0,
            vmax=100.0,
        )
        ax_test.set_xlabel("Iteration")
        ax_test.set_ylabel("Channel")
        _set_heatmap_ticks(ax_test, test_x_avg, test_mat_avg.shape[1])
        fig_hm.colorbar(im, ax=ax_test, fraction=0.046, pad=0.02, label="Accuracy(%)")
        _style_axes(ax_test)
    else:
        ax_test.text(0.5, 0.5, "test per-channel accuracy not found", ha="center", va="center")
        ax_test.set_axis_off()

    plt.tight_layout()
    plt.show()

    # Best test accuracy per channel (max over eval steps)
    if test_mat is not None and test_mat.size > 0:
        with np.errstate(all="ignore"):
            best_test = np.nanmax(test_mat, axis=0)
        if np.ndim(best_test) == 0:
            best_test = np.array([best_test], dtype=float)
        finite_mask = np.isfinite(best_test)
        avg_best = float(np.mean(best_test[finite_mask])) if np.any(finite_mask) else float("nan")

        channel_count = best_test.size
        labels = [str(i + 1) for i in range(channel_count)] + ["avg"]
        values = np.concatenate([best_test, [avg_best]]) * 100.0
        plot_vals = np.where(np.isfinite(values), values, 0.0)

        colors = []
        avg_color = TRAIN_COLOR
        for idx, val in enumerate(values):
            if idx == channel_count:
                colors.append(avg_color)
                continue
            if not np.isfinite(val):
                colors.append("#dddddd")
                continue
            norm = min(max(val / 100.0, 0.0), 1.0)
            colors.append(plt.cm.YlGnBu(0.25 + 0.75 * norm))

        fig_bar, ax_bar = plt.subplots(figsize=(12.5, 5.2))
        bars = ax_bar.bar(
            range(channel_count + 1),
            plot_vals,
            color=colors,
            edgecolor="#2b2b2b",
            linewidth=0.6,
        )

        for idx, val in enumerate(values):
            if not np.isfinite(val):
                bars[idx].set_alpha(0.35)
                bars[idx].set_hatch("//")
            label = f"{val:.1f}%" if np.isfinite(val) else "n/a"
            y = plot_vals[idx] + 1.8
            ax_bar.text(
                idx,
                y,
                label,
                ha="center",
                va="bottom",
                fontsize=12,
            )

        ax_bar.set_xlabel("Channel")
        ax_bar.set_ylabel("Accuracy(%)")
        ax_bar.set_xticks(range(channel_count + 1))
        ax_bar.set_xticklabels(labels)
        ax_bar.set_ylim(0.0, 112.0)
        _style_axes(ax_bar)

        fig_bar.subplots_adjust(left=0.07, right=0.99, bottom=0.18, top=0.92)
        plt.show()

def main():
    if interact and SelectionSlider:
        interact(
            plot_history,
            avg_batch=SelectionSlider(
                options=avg_batch_options,
                value=default_avg_batch,
                description="avg_batch",
                continuous_update=False,
            ),
        )
    else:
        plot_history(default_avg_batch)

if __name__ == "__main__":
    main()
