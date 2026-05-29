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
    _THIS_DIR / "pre_trained_model_save" / "Optical_neural_net" / "checkpoints" / "_code_class_readout_5.pt"
)

default_avg_batch = 50
avg_batch_options = [1, 5, 10, 16, 20, 32, 64, 128, 256, 512, 1024]
default_ignore_train_outliers = False
default_outlier_mad_z = 10.0


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
) -> None:
    """
    Plot train/test loss and accuracy from a code-class readout ONN checkpoint.

    This intentionally plots only:
      - train_loss (averaged over avg_batch)
      - test_loss (raw)
      - train_acc  (averaged over avg_batch)
      - test_acc  (raw)

    If test series are not present, it will annotate the plot.
    """
    history, cfg = _get_history(checkpoint_path)

    iters = history.get("iters", [])
    train_loss = history.get("train_loss", [])
    train_acc = history.get("train_acc", [])
    test_iters = history.get("eval_iters", [])
    test_loss = history.get("eval_loss", [])
    test_acc = history.get("eval_acc", [])

    if ignore_train_outliers:
        train_loss = _mask_outliers_mad(train_loss, mad_z=outlier_mad_z)
        train_acc = _mask_outliers_mad(train_acc, mad_z=outlier_mad_z)

    # Train x axes align to their series length.
    train_loss_x = iters[: len(train_loss)] if iters else list(range(1, len(train_loss) + 1))
    train_acc_x = iters[: len(train_acc)] if iters else list(range(1, len(train_acc) + 1))
    train_loss, train_loss_x = _truncate_series_by_iter(train_loss, train_loss_x, end_iter)
    train_acc, train_acc_x = _truncate_series_by_iter(train_acc, train_acc_x, end_iter)

    train_loss_avg, train_loss_x_avg = _batch_average(train_loss, train_loss_x, avg_batch)
    train_acc_avg, train_acc_x_avg = _batch_average(train_acc, train_acc_x, avg_batch)
    test_loss_plot, test_loss_x = _truncate_series_by_iter(test_loss, test_iters, end_iter)
    test_acc_plot, test_acc_x = _truncate_series_by_iter(test_acc, test_iters, end_iter)
    test_loss_had_points = bool(test_loss_plot and test_loss_x)
    test_acc_had_points = bool(test_acc_plot and test_acc_x)
    test_loss_plot, test_loss_x = test_loss_plot[1:], test_loss_x[1:]
    test_acc_plot, test_acc_x = test_acc_plot[1:], test_acc_x[1:]
    best_train_loss = _best_finite_point(train_loss_avg, train_loss_x_avg, mode="min")
    best_test_loss = _best_finite_point(test_loss_plot, test_loss_x, mode="min")
    best_train_acc = _best_finite_point(train_acc_avg, train_acc_x_avg, mode="max")
    best_test_acc = _best_finite_point(test_acc_plot, test_acc_x, mode="max")

    print("Best metrics:")
    if best_train_loss is not None:
        print(f"  train_loss: {best_train_loss[1]:.6g} @ iter {best_train_loss[0]:.2f}")
    else:
        print("  train_loss: N/A")
    if best_test_loss is not None:
        print(f"  test_loss:  {best_test_loss[1]:.6g} @ iter {best_test_loss[0]:.2f}")
    else:
        print("  test_loss:  N/A")
    if best_train_acc is not None:
        print(f"  train_acc:  {best_train_acc[1]:.6g} @ iter {best_train_acc[0]:.2f}")
    else:
        print("  train_acc:  N/A")
    if best_test_acc is not None:
        print(f"  test_acc:   {best_test_acc[1]:.6g} @ iter {best_test_acc[0]:.2f}")
    else:
        print("  test_acc:   N/A")

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=PAPER_FIGSIZE)

    ax_loss.plot(train_loss_x_avg, train_loss_avg, label="train_loss", linewidth=2, color=TRAIN_COLOR, marker="o", markersize=4)
    if test_loss_plot and test_loss_x:
        ax_loss.plot(test_loss_x, test_loss_plot, label="test_loss", linewidth=2, color=TEST_COLOR, marker="s", markersize=4)
    else:
        test_every = None
        try:
            test_every = cfg.get("onn", {}).get("eval_every", None)
        except Exception:
            test_every = None
        message = "Only first test_loss point present\n(first point hidden)\n" if test_loss_had_points else "No test_* found in checkpoint\n"
        ax_loss.text(
            0.02,
            0.98,
            message
            + (f"(test_every={test_every} > total iters)\n" if test_every is not None else ""),
            transform=ax_loss.transAxes,
            va="top",
        )
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(frameon=False)
    _style_axes(ax_loss)

    ax_acc.plot(train_acc_x_avg, train_acc_avg, label="train_acc", linewidth=2, color=TRAIN_COLOR, marker="o", markersize=4)
    if test_acc_plot and test_acc_x:
        ax_acc.plot(test_acc_x, test_acc_plot, label="test_acc", linewidth=2, color=TEST_COLOR, marker="s", markersize=4)
    elif test_acc_had_points:
        ax_acc.text(0.02, 0.98, "Only first test_acc point present\n(first point hidden)", transform=ax_acc.transAxes, va="top")
    ax_acc.set_xlabel("Iteration")
    ax_acc.set_ylabel("Accuracy(%)")
    ax_acc.legend(frameon=False)
    _style_axes(ax_acc)

    plt.tight_layout()
    plt.show()


def _prepare_train_series(
    history: Dict[str, Any],
    key: str,
    *,
    avg_batch: int,
    end_iter: Optional[float] = None,
    ignore_outliers: bool = False,
    outlier_mad_z: float = default_outlier_mad_z,
) -> Tuple[List[float], List[float]]:
    values = history.get(key, [])
    if not isinstance(values, list):
        values = list(values) if values is not None else []
    iters = history.get("iters", [])
    x_vals = iters[: len(values)] if iters else list(range(1, len(values) + 1))
    if ignore_outliers:
        values = _mask_outliers_mad(values, mad_z=outlier_mad_z)
    values, x_vals = _truncate_series_by_iter(values, x_vals, end_iter)
    return _batch_average(values, x_vals, avg_batch)


def _plot_optional_series(
    ax,
    history: Dict[str, Any],
    specs: List[Tuple[str, str, str]],
    *,
    avg_batch: int,
    end_iter: Optional[float],
    ignore_outliers: bool,
    outlier_mad_z: float,
) -> bool:
    plotted = False
    for key, label, color in specs:
        y_vals, x_vals = _prepare_train_series(
            history,
            key,
            avg_batch=avg_batch,
            end_iter=end_iter,
            ignore_outliers=ignore_outliers,
            outlier_mad_z=outlier_mad_z,
        )
        if y_vals and x_vals:
            ax.plot(x_vals, y_vals, label=label, linewidth=2, marker="o", markersize=3, color=color)
            plotted = True
    return plotted


def plot_code_class_readout_diagnostics(
    avg_batch: int = 1,
    *,
    checkpoint_path: Optional[str] = None,
    ignore_train_outliers: bool = default_ignore_train_outliers,
    outlier_mad_z: float = default_outlier_mad_z,
    end_iter: Optional[float] = None,
) -> None:
    """
    Plot code-class readout-specific train metrics when they exist in the checkpoint.

    These series are produced by optical_onn_training_code_class_readout.py:
      - train_loss_cls / train_loss_bin
      - train_true_yes_score / train_true_no_score
      - train_true_margin / train_max_negative_margin / train_margin_gap
    """
    history, _ = _get_history(checkpoint_path)
    fig, axes = plt.subplots(1, 3, figsize=(21.0, 5.0))
    ax_loss, ax_scores, ax_margin = axes

    plotted_loss = _plot_optional_series(
        ax_loss,
        history,
        [
            ("train_loss", "total_loss", "#4C78A8"),
            ("train_loss_cls", "loss_cls", "#54A24B"),
            ("train_loss_bin", "loss_bin", "#F58518"),
            ("train_loss_spill", "loss_spill", "#B279A2"),
        ],
        avg_batch=avg_batch,
        end_iter=end_iter,
        ignore_outliers=ignore_train_outliers,
        outlier_mad_z=outlier_mad_z,
    )
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Auxiliary Loss Terms")
    if plotted_loss:
        ax_loss.legend(frameon=False)
    else:
        ax_loss.text(0.02, 0.98, "No auxiliary loss metrics found", transform=ax_loss.transAxes, va="top")
    _style_axes(ax_loss)

    plotted_scores = _plot_optional_series(
        ax_scores,
        history,
        [
            ("train_true_yes_score", "true_yes", "#54A24B"),
            ("train_true_no_score", "true_no", "#E45756"),
        ],
        avg_batch=avg_batch,
        end_iter=end_iter,
        ignore_outliers=ignore_train_outliers,
        outlier_mad_z=outlier_mad_z,
    )
    ax_scores.set_xlabel("Iteration")
    ax_scores.set_ylabel("Scaled Score")
    ax_scores.set_title("True Channel YES/NO")
    if plotted_scores:
        ax_scores.legend(frameon=False)
    else:
        ax_scores.text(0.02, 0.98, "No YES/NO score metrics found", transform=ax_scores.transAxes, va="top")
    _style_axes(ax_scores)

    plotted_margin = _plot_optional_series(
        ax_margin,
        history,
        [
            ("train_true_margin", "true_margin", "#4C78A8"),
            ("train_max_negative_margin", "max_negative_margin", "#E45756"),
            ("train_margin_gap", "margin_gap", "#F58518"),
        ],
        avg_batch=avg_batch,
        end_iter=end_iter,
        ignore_outliers=ignore_train_outliers,
        outlier_mad_z=outlier_mad_z,
    )
    ax_margin.axhline(0.0, color="#808080", linewidth=1, linestyle="--")
    ax_margin.set_xlabel("Iteration")
    ax_margin.set_ylabel("Margin")
    ax_margin.set_title("Class Margin Separation")
    if plotted_margin:
        ax_margin.legend(frameon=False)
    else:
        ax_margin.text(0.02, 0.98, "No margin metrics found", transform=ax_margin.transAxes, va="top")
    _style_axes(ax_margin)

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
        ),
        avg_batch=slider,
    )


def _widget_code_class_readout_diagnostics() -> None:
    if interact is None or SelectionSlider is None:
        raise RuntimeError("ipywidgets is not available in this environment.")

    slider = SelectionSlider(
        options=avg_batch_options,
        value=default_avg_batch,
        description="avg_batch",
        continuous_update=False,
    )
    interact(
        lambda avg_batch: plot_code_class_readout_diagnostics(
            avg_batch=avg_batch,
            ignore_train_outliers=default_ignore_train_outliers,
            outlier_mad_z=default_outlier_mad_z,
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
    "plot_code_class_readout_diagnostics",
    "_widget_plot",
    "_widget_code_class_readout_diagnostics",
]
