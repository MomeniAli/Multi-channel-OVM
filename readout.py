"""
Tile-based readout utilities for optical ONN classification.
"""

from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn.functional as F


# -------------------------------------------------------------------------
# Tile mask generation
# -------------------------------------------------------------------------
def build_tile_masks(
    out_hw: Tuple[int, int],
    tile_hw: Tuple[int, int],
    rows: int,
    cols: int,
    gap_hw: Tuple[int, int],
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Builds spatial masks for tile-based optical readouts.

    Layout (rows=3, cols=4) leaves two middle slots empty on the center row:
        0 1 2 3
        4 X X 5
        6 7 8 9
    For other shapes, tiles fill left-to-right, top-to-bottom without skips.
    """
    H, W = out_hw
    th, tw = tile_hw
    gap_h, gap_w = gap_hw

    grid_h = rows * th + (rows - 1) * gap_h
    grid_w = cols * tw + (cols - 1) * gap_w

    top = (H - grid_h) // 2
    left = (W - grid_w) // 2

    masks = torch.zeros((num_classes, H, W), dtype=torch.float32, device=device)

    # Skip two middle slots on the center row when using 3x4 layout
    if rows == 3 and cols == 4:
        skip_positions = {(1, 1), (1, 2)}
    else:
        skip_positions = set()

    available_slots = rows * cols - len(skip_positions)
    if num_classes > available_slots:
        raise ValueError(f"num_classes {num_classes} exceeds available slots {available_slots} with current layout.")

    idx = 0
    done = False
    for i in range(rows):
        for j in range(cols):
            if (i, j) in skip_positions:
                continue
            if idx >= num_classes:
                done = True
                break
            r0 = top + i * (th + gap_h)
            c0 = left + j * (tw + gap_w)
            masks[idx, r0:r0 + th, c0:c0 + tw] = 1.0
            idx += 1
        if done:
            break

    return masks


# -------------------------------------------------------------------------
# Tile mask cache
# -------------------------------------------------------------------------
class TileMaskCache:
    """Keeps resized tile masks/areas for different spatial resolutions encountered during training."""

    def __init__(
        self,
        tile_cfg: Dict[str, object],
        device: torch.device,
        fallback_hw: Tuple[int, int],
    ) -> None:
        tile_hw = tuple(tile_cfg.get("tile_hw", (30, 30)))
        rows = int(tile_cfg.get("rows", 4))
        cols = int(tile_cfg.get("cols", 3))
        gap_cfg = tile_cfg.get("gap", (10, 10))
        if isinstance(gap_cfg, (int, float)):
            gap_hw = (int(gap_cfg), int(gap_cfg))
        else:
            gap_list = list(gap_cfg)
            if len(gap_list) != 2:
                raise ValueError("tile.gap must be a scalar or length-2 iterable")
            gap_hw = (int(gap_list[0]), int(gap_list[1]))
        num_classes = int(tile_cfg.get("num_classes", 10))
        desired_hw = tuple(tile_cfg.get("out_hw", fallback_hw))

        base_masks = build_tile_masks(
            out_hw=desired_hw,
            tile_hw=tile_hw,
            rows=rows,
            cols=cols,
            gap_hw=gap_hw,
            num_classes=num_classes,
            device=device,
        )

        self.device = device
        self.base_hw = tuple(base_masks.shape[-2:])
        self.tile_masks_exp = base_masks.unsqueeze(1)  # (num_classes, 1, H, W)

        # Cache for resized versions
        self.tile_mask_cache: Dict[Tuple[int, int], torch.Tensor] = {
            self.base_hw: self.tile_masks_exp
        }

        area = self.tile_masks_exp.sum(dim=(1, 2, 3)).clamp_min(1e-6)
        self.tile_mask_area_cache: Dict[Tuple[int, int], torch.Tensor] = {
            self.base_hw: area
        }

    # ---------------------------------------------------------------------
    def get_masks(self, hw: Tuple[int, int]) -> torch.Tensor:
        """Returns tile masks resized to the given spatial resolution."""
        if hw not in self.tile_mask_cache:
            resized = F.interpolate(self.tile_masks_exp, size=hw, mode="nearest")
            self.tile_mask_cache[hw] = resized
        return self.tile_mask_cache[hw]

    # ---------------------------------------------------------------------
    def get_mask_area(self, hw: Tuple[int, int]) -> torch.Tensor:
        """Returns cached or computed mask areas for the given resolution."""
        if hw not in self.tile_mask_area_cache:
            masks = self.get_masks(hw)
            self.tile_mask_area_cache[hw] = masks.sum(dim=(1, 2, 3)).clamp_min(1e-6)
        return self.tile_mask_area_cache[hw]


# -------------------------------------------------------------------------
# Tile-based score computation
# -------------------------------------------------------------------------
def compute_tile_scores(
    outputs: torch.Tensor,
    tile_masks: torch.Tensor,
    mask_area: torch.Tensor,
) -> torch.Tensor:
    """Computes normalized energy per tile for ONN outputs."""
    if mask_area.dim() == 1:
        mask_area = mask_area.unsqueeze(0)  # kept for interface compatibility

    total_power = outputs.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    outputs_norm = outputs / total_power.detach()
    spatial_energy = outputs_norm.sum(dim=1)  # (B, H, W)

    if tile_masks.dim() == 4:
        tile_masks_2d = tile_masks.squeeze(1)
    else:
        tile_masks_2d = tile_masks

    tile_masks_exp = tile_masks_2d.unsqueeze(0)
    masked_energy = spatial_energy.unsqueeze(1) * tile_masks_exp
    tile_power = masked_energy.flatten(-2).sum(dim=-1)
    tile_scores = tile_power / mask_area

    return tile_scores

# def compute_tile_scores(
#     outputs: torch.Tensor,
#     tile_masks: torch.Tensor,
#     mask_area: torch.Tensor,
# ) -> torch.Tensor:
#     """Computes normalized energy per tile for ONN outputs."""
#     if mask_area.dim() == 1:
#         mask_area = mask_area.unsqueeze(0)  # kept for interface compatibility

#     total_power = outputs.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
#     outputs_norm = outputs / total_power.detach() 
#     spatial_energy = outputs_norm.sum(dim=1)  # (B, H, W)

#     if tile_masks.dim() == 4:
#         tile_masks_2d = tile_masks.squeeze(1)
#     else:
#         tile_masks_2d = tile_masks

#     tile_masks_exp = tile_masks_2d.unsqueeze(0)
#     masked_energy = spatial_energy.unsqueeze(1) * tile_masks_exp
#     tile_power = masked_energy.flatten(-2).sum(dim=-1)
#     tile_scores = tile_power / mask_area

#     return tile_scores
# -------------------------------------------------------------------------
# Readout scaling
# -------------------------------------------------------------------------
def apply_readout_scaling(
    tile_scores: torch.Tensor,
    use_log_scaling: bool,
    temperature: float,
    log_eps: float,
    temperature_param: torch.Tensor | None = None,
    channel_num: int | None = None,
) -> torch.Tensor:
    """Applies logarithmic or temperature scaling to tile scores.

    If ``temperature_param`` is provided, it overrides the scalar ``temperature`` and can
    be per-channel (shape [channel_num]) or global (shape [1]). The batch dimension is
    expected to be ``group_size * channel_num`` when using per-channel scaling.
    """
    scaled = tile_scores
    if use_log_scaling:
        scaled = torch.log(tile_scores.clamp_min(log_eps))

    if temperature_param is not None:
        temp = temperature_param
        if temp.dim() == 0:
            temp = temp.view(1)
        if channel_num is None:
            channel_num = int(temp.numel())
        channel_num = int(channel_num)
        B, num_classes = scaled.shape
        if channel_num <= 0 or B % channel_num != 0:
            raise ValueError(
                f"Batch size {B} must be divisible by channel_num {channel_num} when using per-channel temperature."
            )
        group_size = B // channel_num
        temp_expanded = (
            temp.view(1, channel_num, 1)
            .expand(group_size, channel_num, num_classes)
            .reshape(B, num_classes)
        )
        scaled = scaled * temp_expanded
    elif temperature != 1.0:
        scaled = scaled * temperature
    return scaled


# -------------------------------------------------------------------------
# Energy margin and ratio loss
# -------------------------------------------------------------------------
def energy_margin_ratio_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gap_margin: float,
    ratio_target: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Computes combined energy margin and ratio-based classification loss."""
    if logits.ndim != 2:
        raise ValueError("Expected logits with shape (B, num_classes) for margin loss.")

    true_scores = logits.gather(1, labels.unsqueeze(1)).squeeze(1)

    competitor_mask = torch.ones_like(logits, dtype=torch.bool)
    competitor_mask.scatter_(1, labels.unsqueeze(1), False)

    competitor_scores = logits.masked_fill(~competitor_mask, float("-inf")).max(dim=1).values
    competitor_scores = torch.where(
        torch.isfinite(competitor_scores),
        competitor_scores,
        torch.zeros_like(competitor_scores),
    )

    diff = true_scores - competitor_scores
    gap_loss = torch.relu(gap_margin - diff)

    ratio = true_scores / (competitor_scores + eps)
    ratio_loss = torch.relu(ratio_target - ratio)

    return (gap_loss + ratio_loss).mean()
