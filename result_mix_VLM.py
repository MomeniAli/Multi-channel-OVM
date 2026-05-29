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
BLEU_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
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
    _THIS_DIR / "pre_trained_model_save" / "Optical_neural_net" / "checkpoints" / "_mul_mix_6_VLM_caption.pt"
)

default_avg_batch = 32
avg_batch_options = [1, 5, 10, 16, 20, 32, 64, 128, 256, 512, 1024]
default_ignore_train_outliers = True
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


def _finite_pairs(x_vals: List[float], y_vals: List[float]) -> Tuple[List[float], List[float]]:
    if not x_vals or not y_vals:
        return [], []
    n = min(len(x_vals), len(y_vals))
    xs_out: List[float] = []
    ys_out: List[float] = []
    for x, y in zip(x_vals[:n], y_vals[:n]):
        try:
            xf = float(x)
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if np.isfinite(xf) and np.isfinite(yf):
            xs_out.append(xf)
            ys_out.append(yf)
    return xs_out, ys_out


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
    Plot train/test caption loss (left) and train/test BLEU-1..4 (right) from a VLM checkpoint.
    """
    history, cfg = _get_history(checkpoint_path)

    # Prefer raw loss histories so plots start from the first available iteration.
    # If raw history is missing, fall back to averaged histories stored in checkpoints.
    train_loss = history.get(
        "train_caption_loss",
        history.get("train_caption_loss_avg", history.get("train_loss", [])),
    )
    train_loss_iters = history.get("iters", history.get("train_loss_avg_iters", []))
    test_loss = history.get(
        "eval_caption_loss",
        history.get("eval_caption_loss_avg", history.get("eval_loss", [])),
    )
    test_loss_iters = history.get("eval_iters", history.get("eval_loss_avg_iters", []))

    train_bleu_iters = history.get(
        "train_bleu_iters",
        history.get("train_bleu_avg_iters", history.get("iters", [])),
    )
    test_bleu_iters = history.get(
        "eval_bleu_iters",
        history.get("eval_bleu_avg_iters", history.get("eval_iters", [])),
    )
    train_bleu1 = history.get("train_bleu1", history.get("train_bleu1_avg", []))
    train_bleu2 = history.get("train_bleu2", history.get("train_bleu2_avg", []))
    train_bleu3 = history.get("train_bleu3", history.get("train_bleu3_avg", []))
    train_bleu4 = history.get("train_bleu4", history.get("train_bleu4_avg", []))
    test_bleu1 = history.get("eval_bleu1", history.get("eval_bleu1_avg", []))
    test_bleu2 = history.get("eval_bleu2", history.get("eval_bleu2_avg", []))
    test_bleu3 = history.get("eval_bleu3", history.get("eval_bleu3_avg", []))
    test_bleu4 = history.get("eval_bleu4", history.get("eval_bleu4_avg", []))

    if ignore_train_outliers:
        train_loss = _mask_outliers_mad(train_loss, mad_z=outlier_mad_z)

    train_loss_x = (
        train_loss_iters[: len(train_loss)]
        if train_loss_iters
        else list(range(1, len(train_loss) + 1))
    )
    test_loss_x = (
        test_loss_iters[: len(test_loss)]
        if test_loss_iters
        else list(range(1, len(test_loss) + 1))
    )

    train_loss, train_loss_x = _truncate_series_by_iter(train_loss, train_loss_x, end_iter)
    test_loss, test_loss_x = _truncate_series_by_iter(test_loss, test_loss_x, end_iter)
    train_bleu1, train_bleu_x_1 = _truncate_series_by_iter(
        train_bleu1,
        train_bleu_iters[: len(train_bleu1)] if train_bleu_iters else list(range(1, len(train_bleu1) + 1)),
        end_iter,
    )
    train_bleu2, train_bleu_x_2 = _truncate_series_by_iter(
        train_bleu2,
        train_bleu_iters[: len(train_bleu2)] if train_bleu_iters else list(range(1, len(train_bleu2) + 1)),
        end_iter,
    )
    train_bleu3, train_bleu_x_3 = _truncate_series_by_iter(
        train_bleu3,
        train_bleu_iters[: len(train_bleu3)] if train_bleu_iters else list(range(1, len(train_bleu3) + 1)),
        end_iter,
    )
    train_bleu4, train_bleu_x_4 = _truncate_series_by_iter(
        train_bleu4,
        train_bleu_iters[: len(train_bleu4)] if train_bleu_iters else list(range(1, len(train_bleu4) + 1)),
        end_iter,
    )
    test_bleu1, bleu_x_1 = _truncate_series_by_iter(
        test_bleu1,
        test_bleu_iters[: len(test_bleu1)] if test_bleu_iters else list(range(1, len(test_bleu1) + 1)),
        end_iter,
    )
    test_bleu2, bleu_x_2 = _truncate_series_by_iter(
        test_bleu2,
        test_bleu_iters[: len(test_bleu2)] if test_bleu_iters else list(range(1, len(test_bleu2) + 1)),
        end_iter,
    )
    test_bleu3, bleu_x_3 = _truncate_series_by_iter(
        test_bleu3,
        test_bleu_iters[: len(test_bleu3)] if test_bleu_iters else list(range(1, len(test_bleu3) + 1)),
        end_iter,
    )
    test_bleu4, bleu_x_4 = _truncate_series_by_iter(
        test_bleu4,
        test_bleu_iters[: len(test_bleu4)] if test_bleu_iters else list(range(1, len(test_bleu4) + 1)),
        end_iter,
    )

    # Apply smoothing window to train curves only.
    # Keep test curves raw so sparse checkpoint evaluations stay visible.
    train_loss_avg, train_loss_x_avg = _batch_average(train_loss, train_loss_x, avg_batch)
    test_loss_plot, test_loss_x_plot = _batch_average(test_loss, test_loss_x, 1)
    train_bleu1_plot, train_bleu1_x_plot = _batch_average(train_bleu1, train_bleu_x_1, avg_batch)
    train_bleu2_plot, train_bleu2_x_plot = _batch_average(train_bleu2, train_bleu_x_2, avg_batch)
    train_bleu3_plot, train_bleu3_x_plot = _batch_average(train_bleu3, train_bleu_x_3, avg_batch)
    train_bleu4_plot, train_bleu4_x_plot = _batch_average(train_bleu4, train_bleu_x_4, avg_batch)
    test_bleu1_plot, test_bleu1_x = _batch_average(test_bleu1, bleu_x_1, 1)
    test_bleu2_plot, test_bleu2_x = _batch_average(test_bleu2, bleu_x_2, 1)
    test_bleu3_plot, test_bleu3_x = _batch_average(test_bleu3, bleu_x_3, 1)
    test_bleu4_plot, test_bleu4_x = _batch_average(test_bleu4, bleu_x_4, 1)

    best_train_loss = _best_finite_point(train_loss_avg, train_loss_x_avg, mode="min")
    best_test_loss = _best_finite_point(test_loss_plot, test_loss_x_plot, mode="min")
    best_train_bleu1 = _best_finite_point(train_bleu1_plot, train_bleu1_x_plot, mode="max")
    best_train_bleu2 = _best_finite_point(train_bleu2_plot, train_bleu2_x_plot, mode="max")
    best_train_bleu3 = _best_finite_point(train_bleu3_plot, train_bleu3_x_plot, mode="max")
    best_train_bleu4 = _best_finite_point(train_bleu4_plot, train_bleu4_x_plot, mode="max")
    best_test_bleu1 = _best_finite_point(test_bleu1_plot, test_bleu1_x, mode="max")
    best_test_bleu2 = _best_finite_point(test_bleu2_plot, test_bleu2_x, mode="max")
    best_test_bleu3 = _best_finite_point(test_bleu3_plot, test_bleu3_x, mode="max")
    best_test_bleu4 = _best_finite_point(test_bleu4_plot, test_bleu4_x, mode="max")

    print("Best metrics:")
    if best_train_loss is not None:
        print(f"  train_caption_loss: {best_train_loss[1]:.6g} @ iter {best_train_loss[0]:.2f}")
    else:
        print("  train_caption_loss: N/A")
    if best_test_loss is not None:
        print(f"  test_caption_loss:  {best_test_loss[1]:.6g} @ iter {best_test_loss[0]:.2f}")
    else:
        print("  test_caption_loss:  N/A")
    if best_train_bleu1 is not None:
        print(f"  train_bleu1:        {best_train_bleu1[1]:.6g} @ iter {best_train_bleu1[0]:.2f}")
    else:
        print("  train_bleu1:        N/A")
    if best_test_bleu1 is not None:
        print(f"  test_bleu1:         {best_test_bleu1[1]:.6g} @ iter {best_test_bleu1[0]:.2f}")
    else:
        print("  test_bleu1:         N/A")
    if best_train_bleu2 is not None:
        print(f"  train_bleu2:        {best_train_bleu2[1]:.6g} @ iter {best_train_bleu2[0]:.2f}")
    else:
        print("  train_bleu2:        N/A")
    if best_test_bleu2 is not None:
        print(f"  test_bleu2:         {best_test_bleu2[1]:.6g} @ iter {best_test_bleu2[0]:.2f}")
    else:
        print("  test_bleu2:         N/A")
    if best_train_bleu3 is not None:
        print(f"  train_bleu3:        {best_train_bleu3[1]:.6g} @ iter {best_train_bleu3[0]:.2f}")
    else:
        print("  train_bleu3:        N/A")
    if best_test_bleu3 is not None:
        print(f"  test_bleu3:         {best_test_bleu3[1]:.6g} @ iter {best_test_bleu3[0]:.2f}")
    else:
        print("  test_bleu3:         N/A")
    if best_train_bleu4 is not None:
        print(f"  train_bleu4:        {best_train_bleu4[1]:.6g} @ iter {best_train_bleu4[0]:.2f}")
    else:
        print("  train_bleu4:        N/A")
    if best_test_bleu4 is not None:
        print(f"  test_bleu4:         {best_test_bleu4[1]:.6g} @ iter {best_test_bleu4[0]:.2f}")
    else:
        print("  test_bleu4:         N/A")

    fig, (ax_loss, ax_bleu) = plt.subplots(1, 2, figsize=PAPER_FIGSIZE)

    train_loss_x_finite, train_loss_y_finite = _finite_pairs(train_loss_x_avg, train_loss_avg)
    test_loss_x_finite, test_loss_y_finite = _finite_pairs(test_loss_x_plot, test_loss_plot)
    if train_loss_x_finite and train_loss_y_finite:
        ax_loss.plot(train_loss_x_finite, train_loss_y_finite, label="train_caption_loss", linewidth=2, color=TRAIN_COLOR, marker="o", markersize=4)
    if test_loss_x_finite and test_loss_y_finite:
        ax_loss.plot(
            test_loss_x_finite,
            test_loss_y_finite,
            label="test_caption_loss",
            linewidth=2,
            color=TEST_COLOR,
            marker="s",
            markersize=4,
        )
    else:
        test_every = None
        try:
            test_every = cfg.get("onn", {}).get("eval_every", None)
        except Exception:
            test_every = None
        reason = "No finite test_caption_loss found in checkpoint"
        if not test_loss_plot:
            reason = "No test_caption_loss found in checkpoint"
        ax_loss.text(
            0.02,
            0.98,
            reason + "\n"
            + (f"(test_every={test_every} > total iters)\n" if test_every is not None else ""),
            transform=ax_loss.transAxes,
            va="top",
        )
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Caption loss")
    ax_loss.legend(frameon=False)
    _style_axes(ax_loss)
    if y_log:
        ax_loss.set_yscale("log")

    has_bleu = False
    has_train_bleu = False
    has_test_bleu = False
    c1, c2, c3, c4 = BLEU_COLORS
    train_bleu1_x_finite, train_bleu1_y_finite = _finite_pairs(train_bleu1_x_plot, train_bleu1_plot)
    train_bleu2_x_finite, train_bleu2_y_finite = _finite_pairs(train_bleu2_x_plot, train_bleu2_plot)
    train_bleu3_x_finite, train_bleu3_y_finite = _finite_pairs(train_bleu3_x_plot, train_bleu3_plot)
    train_bleu4_x_finite, train_bleu4_y_finite = _finite_pairs(train_bleu4_x_plot, train_bleu4_plot)
    test_bleu1_x_finite, test_bleu1_y_finite = _finite_pairs(test_bleu1_x, test_bleu1_plot)
    test_bleu2_x_finite, test_bleu2_y_finite = _finite_pairs(test_bleu2_x, test_bleu2_plot)
    test_bleu3_x_finite, test_bleu3_y_finite = _finite_pairs(test_bleu3_x, test_bleu3_plot)
    test_bleu4_x_finite, test_bleu4_y_finite = _finite_pairs(test_bleu4_x, test_bleu4_plot)

    if train_bleu1_x_finite and train_bleu1_y_finite:
        ax_bleu.plot(train_bleu1_x_finite, train_bleu1_y_finite, label="train BLEU-1", linewidth=1.8, linestyle="-", color=c1, marker="o", markersize=3.5)
        has_bleu = True
        has_train_bleu = True
    if train_bleu2_x_finite and train_bleu2_y_finite:
        ax_bleu.plot(train_bleu2_x_finite, train_bleu2_y_finite, label="train BLEU-2", linewidth=1.8, linestyle="-", color=c2, marker="o", markersize=3.5)
        has_bleu = True
        has_train_bleu = True
    if train_bleu3_x_finite and train_bleu3_y_finite:
        ax_bleu.plot(train_bleu3_x_finite, train_bleu3_y_finite, label="train BLEU-3", linewidth=1.8, linestyle="-", color=c3, marker="o", markersize=3.5)
        has_bleu = True
        has_train_bleu = True
    if train_bleu4_x_finite and train_bleu4_y_finite:
        ax_bleu.plot(train_bleu4_x_finite, train_bleu4_y_finite, label="train BLEU-4", linewidth=1.8, linestyle="-", color=c4, marker="o", markersize=3.5)
        has_bleu = True
        has_train_bleu = True
    if test_bleu1_x_finite and test_bleu1_y_finite:
        ax_bleu.scatter(test_bleu1_x_finite, test_bleu1_y_finite, label="test BLEU-1", marker="s", s=42, color=c1, zorder=3)
        has_bleu = True
        has_test_bleu = True
    if test_bleu2_x_finite and test_bleu2_y_finite:
        ax_bleu.scatter(test_bleu2_x_finite, test_bleu2_y_finite, label="test BLEU-2", marker="s", s=42, color=c2, zorder=3)
        has_bleu = True
        has_test_bleu = True
    if test_bleu3_x_finite and test_bleu3_y_finite:
        ax_bleu.scatter(test_bleu3_x_finite, test_bleu3_y_finite, label="test BLEU-3", marker="s", s=42, color=c3, zorder=3)
        has_bleu = True
        has_test_bleu = True
    if test_bleu4_x_finite and test_bleu4_y_finite:
        ax_bleu.scatter(test_bleu4_x_finite, test_bleu4_y_finite, label="test BLEU-4", marker="s", s=42, color=c4, zorder=3)
        has_bleu = True
        has_test_bleu = True
    if has_bleu:
        ax_bleu.set_ylim(0.0, 1.0)
        if has_train_bleu and not has_test_bleu:
            ax_bleu.text(
                0.02,
                0.98,
                "Test BLEU not found in checkpoint",
                transform=ax_bleu.transAxes,
                va="top",
            )
        if has_test_bleu and not has_train_bleu:
            ax_bleu.text(
                0.02,
                0.98,
                "Train BLEU not found in checkpoint",
                transform=ax_bleu.transAxes,
                va="top",
            )
    else:
        ax_bleu.text(
            0.02,
            0.98,
            "No train/test BLEU history found in checkpoint",
            transform=ax_bleu.transAxes,
            va="top",
        )
    ax_bleu.set_xlabel("Iteration")
    ax_bleu.set_ylabel("BLEU score")
    ax_bleu.legend(frameon=False)
    _style_axes(ax_bleu)

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
