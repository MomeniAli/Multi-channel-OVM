import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from ipywidgets import interact, SelectionSlider
except Exception:  # ipywidgets may be unavailable
    SelectionSlider = None
    interact = None


def _apply_plot_style() -> None:
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
PLOT_LINEWIDTH = 3.0
TRAIN_MARKERSIZE = 4.2
TEST_MARKERSIZE = 4.8
PAPER_FIGSIZE = (14.0, 5.0)


def _style_axes(ax) -> None:
    ax.grid(False)
    ax.spines["left"].set_color("#B0B0B0")
    ax.spines["bottom"].set_color("#B0B0B0")
    ax.spines["top"].set_color("#B0B0B0")
    ax.spines["right"].set_color("#B0B0B0")
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.tick_params(color="#7F7F7F", labelcolor="black", width=0.8, length=5)


_THIS_DIR = Path(__file__).resolve().parent

# Default checkpoint (change as needed)
ckpt_path = str(
    _THIS_DIR / "pre_trained_model_save" / "Optical_neural_net" / "checkpoints" / "_mul_mix_5_Facial.pt"
)


default_avg_batch = 50
avg_batch_options = [1, 5, 10, 16, 20, 32, 64, 128, 256, 512, 1024]
default_ignore_train_outliers = True
default_outlier_mad_z = 10.0
PIXELS_PER_AXIS = 96.0


def _load_checkpoint(path: Optional[str] = None) -> Dict[str, Any]:
    p = Path(path or ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    obj = torch.load(p, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Expected checkpoint dict, got {type(obj)}")
    return obj


def _get_history(path: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ckpt_obj = _load_checkpoint(path)
    history = ckpt_obj.get("history", {})
    if not isinstance(history, dict):
        history = {}
    cfg = ckpt_obj.get("cfg", {})
    if not isinstance(cfg, dict):
        cfg = {}
    return history, cfg


def _batch_average(values: List[float], x_vals: Optional[List[float]] = None, window: int = 1):
    if not values:
        return [], []
    if window <= 1:
        return list(values), list(x_vals) if x_vals is not None else list(range(len(values)))
    out_vals = []
    out_x = []
    n = len(values)
    for start in range(0, n, window):
        chunk = values[start : start + window]
        finite = [v for v in chunk if v is not None and np.isfinite(v)]
        out_vals.append(float(np.mean(finite)) if finite else float("nan"))
        if x_vals is not None:
            x_chunk = x_vals[start : start + window]
            out_x.append(float(np.mean(x_chunk)) if x_chunk else float(start))
        else:
            out_x.append(float(start + (len(chunk) - 1) / 2))
    return out_vals, out_x


def _mask_outliers_mad(
    values: List[float],
    *,
    mad_z: float = default_outlier_mad_z,
) -> List[float]:
    if not values:
        return []

    y = np.asarray(values, dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < 4:
        return y.tolist()

    y_finite = y[finite]
    median = float(np.median(y_finite))
    mad = float(np.median(np.abs(y_finite - median)))
    if mad <= 0:
        return y.tolist()

    robust_z = 0.6745 * (y - median) / mad
    keep = finite & (np.abs(robust_z) <= float(mad_z))
    y_masked = y.copy()
    y_masked[~keep] = np.nan
    return y_masked.tolist()


def _truncate_series_by_iter(
    values: List[float],
    x_vals: List[float],
    end_iter: Optional[float] = None,
) -> Tuple[List[float], List[float]]:
    vals = list(values)
    xs = list(x_vals)[: len(vals)]
    vals = vals[: len(xs)]
    if end_iter is None:
        return vals, xs
    keep_n = 0
    for x in xs:
        if x <= float(end_iter):
            keep_n += 1
        else:
            break
    return vals[:keep_n], xs[:keep_n]


def _best_finite_point(
    values: List[float],
    x_vals: List[float],
    *,
    mode: str,
) -> Optional[Tuple[float, float]]:
    if not values or not x_vals:
        return None
    n = min(len(values), len(x_vals))
    y = np.asarray(values[:n], dtype=float)
    x = np.asarray(x_vals[:n], dtype=float)
    finite_mask = np.isfinite(y) & np.isfinite(x)
    if not np.any(finite_mask):
        return None
    y_f = y[finite_mask]
    x_f = x[finite_mask]
    idx = int(np.nanargmin(y_f) if mode == "min" else np.nanargmax(y_f))
    return float(x_f[idx]), float(y_f[idx])


def plot_history(
    avg_batch: int = 1,
    *,
    checkpoint_path: Optional[str] = None,
    ignore_train_outliers: bool = default_ignore_train_outliers,
    outlier_mad_z: float = default_outlier_mad_z,
    end_iter: Optional[float] = None,
    y_log: bool = False,
) -> None:
    """
    Plot train/test loss and MSE from a facial regression checkpoint.
    """
    history, cfg = _get_history(checkpoint_path)

    iters = history.get("iters", [])
    train_loss = history.get("train_loss", [])
    train_mse = history.get("train_mse", history.get("train_mse_loc", []))
    test_iters = history.get("eval_iters", [])
    test_loss = history.get("eval_loss", [])
    test_mse = history.get("eval_mse", history.get("eval_mse_loc", []))

    if ignore_train_outliers:
        train_loss = _mask_outliers_mad(train_loss, mad_z=outlier_mad_z)
        train_mse = _mask_outliers_mad(train_mse, mad_z=outlier_mad_z)

    train_loss_x = iters[: len(train_loss)] if iters else list(range(1, len(train_loss) + 1))
    train_mse_x = iters[: len(train_mse)] if iters else list(range(1, len(train_mse) + 1))
    train_loss, train_loss_x = _truncate_series_by_iter(train_loss, train_loss_x, end_iter)
    train_mse, train_mse_x = _truncate_series_by_iter(train_mse, train_mse_x, end_iter)
    train_loss_avg, train_loss_x_avg = _batch_average(train_loss, train_loss_x, avg_batch)
    train_mse_avg, train_mse_x_avg = _batch_average(train_mse, train_mse_x, avg_batch)
    test_loss_plot, test_loss_x = _truncate_series_by_iter(test_loss, test_iters, end_iter)
    test_mse_plot, test_mse_x = _truncate_series_by_iter(test_mse, test_iters, end_iter)
    # Labels are normalized to [0,1]; convert MSE to pixel error (RMSE in px).
    train_err_px_avg = [
        float(np.sqrt(max(float(v), 0.0))) * PIXELS_PER_AXIS if np.isfinite(v) else float("nan")
        for v in train_mse_avg
    ]
    test_err_px_plot = [
        float(np.sqrt(max(float(v), 0.0))) * PIXELS_PER_AXIS if np.isfinite(v) else float("nan")
        for v in test_mse_plot
    ]
    best_train_loss = _best_finite_point(train_loss_avg, train_loss_x_avg, mode="min")
    best_test_loss = _best_finite_point(test_loss_plot, test_loss_x, mode="min")
    best_train_px = _best_finite_point(train_err_px_avg, train_mse_x_avg, mode="min")
    best_test_px = _best_finite_point(test_err_px_plot, test_mse_x, mode="min")

    print("Best metrics:")
    if best_train_loss is not None:
        print(f"  train_loss:       {best_train_loss[1]:.6g} @ iter {best_train_loss[0]:.2f}")
    else:
        print("  train_loss:       N/A")
    if best_test_loss is not None:
        print(f"  test_loss:        {best_test_loss[1]:.6g} @ iter {best_test_loss[0]:.2f}")
    else:
        print("  test_loss:        N/A")
    if best_train_px is not None:
        print(f"  train_pixel_err:  {best_train_px[1]:.6g} @ iter {best_train_px[0]:.2f}")
    else:
        print("  train_pixel_err:  N/A")
    if best_test_px is not None:
        print(f"  test_pixel_err:   {best_test_px[1]:.6g} @ iter {best_test_px[0]:.2f}")
    else:
        print("  test_pixel_err:   N/A")

    fig, (ax_loss, ax_mse) = plt.subplots(1, 2, figsize=PAPER_FIGSIZE)

    train_loss_markevery = max(1, len(train_loss_x_avg) // 80) if train_loss_x_avg else 1
    ax_loss.plot(
        train_loss_x_avg,
        train_loss_avg,
        label="train_loss",
        color=TRAIN_COLOR,
        linestyle="-",
        marker="o",
        markersize=TRAIN_MARKERSIZE,
        markevery=train_loss_markevery,
        linewidth=PLOT_LINEWIDTH,
        alpha=0.95,
    )
    if test_loss_plot and test_loss_x:
        ax_loss.plot(
            test_loss_x,
            test_loss_plot,
            label="test_loss",
            color=TEST_COLOR,
            linestyle="-",
            marker="s",
            markersize=TEST_MARKERSIZE,
            linewidth=PLOT_LINEWIDTH,
            alpha=0.95,
        )
    else:
        test_every = None
        try:
            test_every = cfg.get("onn", {}).get("eval_every", None)
        except Exception:
            test_every = None
        ax_loss.text(
            0.02,
            0.98,
            "No test_loss found in checkpoint\n"
            + (f"(test_every={test_every} > total iters)\n" if test_every is not None else ""),
            transform=ax_loss.transAxes,
            va="top",
        )
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(frameon=False)
    _style_axes(ax_loss)
    if y_log:
        ax_loss.set_yscale("log")

    train_mse_markevery = max(1, len(train_mse_x_avg) // 80) if train_mse_x_avg else 1
    ax_mse.plot(
        train_mse_x_avg,
        train_err_px_avg,
        label="train_pixel_error",
        color=TRAIN_COLOR,
        linestyle="-",
        marker="o",
        markersize=TRAIN_MARKERSIZE,
        markevery=train_mse_markevery,
        linewidth=PLOT_LINEWIDTH,
        alpha=0.95,
    )
    if test_err_px_plot and test_mse_x:
        ax_mse.plot(
            test_mse_x,
            test_err_px_plot,
            label="test_pixel_error",
            color=TEST_COLOR,
            linestyle="-",
            marker="s",
            markersize=TEST_MARKERSIZE,
            linewidth=PLOT_LINEWIDTH,
            alpha=0.95,
        )
    else:
        ax_mse.text(
            0.02,
            0.98,
            "No test_mse found in checkpoint\n",
            transform=ax_mse.transAxes,
            va="top",
        )
    ax_mse.set_xlabel("Iteration")
    ax_mse.set_ylabel("Pixel error (px)")
    ax_mse.legend(frameon=False)
    _style_axes(ax_mse)
    if y_log:
        ax_mse.set_yscale("log")

    plt.tight_layout()
    plt.show()


def _widget_plot() -> None:
    if interact is None or SelectionSlider is None:
        raise RuntimeError("ipywidgets is not available in this environment.")

    slider = SelectionSlider(
        options=avg_batch_options,
        value=default_avg_batch,
        description="avg_batch",
        continuous_update=False,
    )
    interact(
        lambda avg_batch: plot_history(
            avg_batch=avg_batch,
            ignore_train_outliers=default_ignore_train_outliers,
            outlier_mad_z=default_outlier_mad_z,
            y_log=False,
        ),
        avg_batch=slider,
    )


__all__ = [
    "ckpt_path",
    "default_avg_batch",
    "avg_batch_options",
    "default_ignore_train_outliers",
    "default_outlier_mad_z",
    "plot_history",
    "_widget_plot",
]
