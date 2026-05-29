import math
import torch
import torch.nn.functional as F
from typing import Callable, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


def build_mnist_loaders(
    batch_size: int = 64,
    augment: bool = False,
    num_workers: int = 2,
    pin_memory: bool = True,
    root: str = "./data",
    val_split: float = 0.1,
    drop_last: bool = True,
    pair_negatives: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    # --- Transforms that keep data in [0,1] ---
    if augment:
        train_tfms = transforms.Compose([
            transforms.RandomAffine(
                degrees=5,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=4
            ),
            transforms.ToTensor(),  # -> [0,1]
            transforms.RandomErasing(
                p=0.15,
                scale=(0.01, 0.10),
                ratio=(0.1, 2.0),
                value=0.0,         # erase to 0.0 in [0,1] space
                inplace=True
            ),
        ])
    else:
        train_tfms = transforms.ToTensor()  # -> [0,1]

    val_tfms = transforms.ToTensor()        # -> [0,1]

    # Dataset and split
    full_train = datasets.MNIST(root=root, train=True, download=True, transform=train_tfms)
    val_len = int(len(full_train) * val_split)
    train_len = len(full_train) - val_len
    train_ds, val_ds = random_split(full_train, [train_len, val_len])

    # Override val transform to ensure no train-time augs leak into val
    val_ds.dataset.transform = val_tfms

    if pair_negatives:
        train_ds = _PairedMNIST(train_ds)
        val_ds = _PairedMNIST(val_ds)

    # Loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last
    )
    return train_loader, val_loader


class _PairedMNIST(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.targets = self._extract_targets(dataset)
        self.label_to_indices = {}
        for idx, label in enumerate(self.targets):
            self.label_to_indices.setdefault(int(label), []).append(idx)
        self.labels = list(self.label_to_indices.keys())

    @staticmethod
    def _extract_targets(dataset: Dataset) -> list:
        if isinstance(dataset, Subset):
            base = dataset.dataset
            targets = getattr(base, "targets", None)
            if targets is None:
                targets = getattr(base, "labels", None)
            if targets is None:
                raise ValueError("PairedMNIST requires dataset targets or labels.")
            return [int(targets[i]) for i in dataset.indices]
        targets = getattr(dataset, "targets", None)
        if targets is None:
            targets = getattr(dataset, "labels", None)
        if targets is None:
            raise ValueError("PairedMNIST requires dataset targets or labels.")
        return [int(t) for t in targets]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        x_pos, y_pos = self.dataset[idx]
        y_pos_int = int(y_pos) if not isinstance(y_pos, torch.Tensor) else int(y_pos.item())
        if len(self.labels) <= 1:
            neg_idx = idx
        else:
            neg_label = y_pos_int
            while neg_label == y_pos_int:
                label_idx = int(torch.randint(0, len(self.labels), (1,)).item())
                neg_label = self.labels[label_idx]
            candidates = self.label_to_indices[neg_label]
            cand_idx = int(torch.randint(0, len(candidates), (1,)).item())
            neg_idx = candidates[cand_idx]
        x_neg, _ = self.dataset[neg_idx]
        return x_pos, x_neg, y_pos

# -----------------------------------------------------------------------------
# Dynamic input expansion
# -----------------------------------------------------------------------------

def _ensure_nchw(t: torch.Tensor) -> Tuple[torch.Tensor, int, int, int]:
    """
    Ensure 4D NCHW.
    Returns: (t4d, orig_ndim, orig_B, orig_C)
    """
    if t.dim() == 2:            # (H,W)
        H, W = t.shape
        return t.unsqueeze(0).unsqueeze(0), 2, 1, 1
    elif t.dim() == 3:          # (C,H,W)
        C, H, W = t.shape
        return t.unsqueeze(0), 3, 1, C
    elif t.dim() == 4:          # (B,C,H,W)
        B, C, H, W = t.shape
        return t, 4, B, C
    else:
        raise ValueError(f"_ensure_nchw: expected 2D/3D/4D, got {tuple(t.shape)}")


def _tile_to_canvas(
    x4: torch.Tensor,
    canvas_hw: tuple[int,int],
    tile_factor: Union[int, Tuple[int,int]] = 2,
    rotate_sequence: Optional[Tuple[int, ...]] = (0,1,2,3),
    variant_builder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """
    1) Pixel-repeat upsample by tile_factor (nearest-like).
    2) Tile the upsampled block to cover canvas (optionally rotating/augmenting tiles deterministically).
    3) Center-crop to canvas.
    """
    B, C, H, W = x4.shape
    Ht, Wt = canvas_hw

    # --- parse tile_factor as pixel-repeat upsample ---
    if tile_factor is None:
        tf_h = tf_w = 1
    elif isinstance(tile_factor, int):
        tf_h = tf_w = max(1, tile_factor)
    else:
        tf_h, tf_w = max(1, tile_factor[0]), max(1, tile_factor[1])

    # 1) pixel-repeat upsample once (nearest, integer factor)
    x_scaled = x4.repeat_interleave(tf_h, dim=-2).repeat_interleave(tf_w, dim=-1)
    Hs, Ws = x_scaled.shape[-2:]

    # 2) compute how many tiles needed to cover canvas
    rep_h = max(1, math.ceil(Ht / Hs))
    rep_w = max(1, math.ceil(Wt / Ws))

    rotate_sequence = tuple(int(k) % 4 for k in rotate_sequence) if rotate_sequence else ()
    if not rotate_sequence and variant_builder is None:
        x_tiled = x_scaled.repeat(1, 1, rep_h, rep_w)
    else:
        if variant_builder is None:
            variants = [torch.rot90(x_scaled, k=k, dims=(-2, -1)) for k in (0, 1, 2, 3)]
            x_stack = torch.stack(variants, dim=1)  # (B, 4, C, Hs, Ws)
        else:
            x_stack = variant_builder(x_scaled)  # expected (B, K, C, Hs, Ws)
            if not isinstance(x_stack, torch.Tensor) or x_stack.dim() != 5:
                raise ValueError("variant_builder must return a Tensor shaped (B, K, C, H, W)")
            if x_stack.size(0) != B or x_stack.size(2) != C or x_stack.size(3) != Hs or x_stack.size(4) != Ws:
                raise ValueError("variant_builder output batch/shape must match input x_scaled")
        device = x4.device
        n_variants = x_stack.size(1)
        if rotate_sequence:
            order = torch.tensor(rotate_sequence, device=device, dtype=torch.long) % n_variants
        else:
            order = torch.arange(n_variants, device=device, dtype=torch.long)
        tile_ids = torch.arange(rep_h * rep_w, device=device) % order.numel()
        tile_ids = order[tile_ids].view(rep_h, rep_w)
        tile_ids = tile_ids.unsqueeze(0).expand(B, -1, -1)  # (B, rep_h, rep_w)

        H_big = rep_h * Hs
        W_big = rep_w * Ws
        x_tiled = x_scaled.new_zeros(B, C, H_big, W_big)
        batch_idx = torch.arange(B, device=device)
        for i in range(rep_h):
            for j in range(rep_w):
                sel = tile_ids[:, i, j]
                picked = x_stack[batch_idx, sel]
                r0, r1 = i * Hs, (i + 1) * Hs
                c0, c1 = j * Ws, (j + 1) * Ws
                x_tiled[:, :, r0:r1, c0:c1] = picked

    # 4) center-crop to canvas
    H_big, W_big = x_tiled.shape[-2:]
    off_h = (H_big - Ht) // 2
    off_w = (W_big - Wt) // 2
    return x_tiled[..., off_h:off_h+Ht, off_w:off_w+Wt]


def _resize_to_canvas(
    t: torch.Tensor,
    canvas_hw: Tuple[int, int],
    is_mask: bool,
    antialias: bool = True,  # harmless for nearest; improves bilinear/bicubic
) -> torch.Tensor:
    t4 = t if t.dim() == 4 else t.view(t.size(0), t.size(1), t.size(-2), t.size(-1))
    Ht, Wt = canvas_hw
    if t4.shape[-2:] == (Ht, Wt):
        return t4
    if is_mask:
        return F.interpolate(t4, size=(Ht, Wt), mode="nearest")
    try:
        return F.interpolate(t4, size=(Ht, Wt), mode="bilinear",
                             align_corners=False, antialias=antialias)
    except TypeError:
        return F.interpolate(t4, size=(Ht, Wt), mode="bilinear",
                             align_corners=False)
def _norm01_spatial(t: torch.Tensor) -> torch.Tensor:
        mn = t.amin(dim=(-2, -1), keepdim=True)
        mx = t.amax(dim=(-2, -1), keepdim=True)
        return ((t - mn) / (mx - mn + 1e-8))
    
def percentile_scale_spatial(
    x: torch.Tensor,
    peak: float = 0.99,
    q: float = 0.99,
    eps: float = 1e-6,
) -> torch.Tensor:
    orig_dtype = x.dtype
    x_float = x if orig_dtype in (torch.float32, torch.float64) else x.float()
    H, W = x_float.shape[-2], x_float.shape[-1]
    flat = x_float.reshape(*x_float.shape[:-2], H * W)            # (..., HW)
    p = torch.quantile(flat, q, dim=-1)                           # (...,)
    p = p.clamp_min(eps)                                          # keep positivity
    while p.dim() < x_float.dim():
        p = p.unsqueeze(-1)                                       # (..., 1, 1)

    y = x_float * (peak / p)
    y = y.clamp(0, 1)
    if orig_dtype not in (torch.float32, torch.float64):
        y = y.to(orig_dtype)
    return y



def encoding_x_phase_physical(
    x: torch.Tensor,
    phase: torch.Tensor,
    canvas_hw: tuple[int, int],
    x_mode: str = "tile",  # "interp" | "tile" | "mix"
    x_norm_mod: Optional[str] = "percentile",  # "none" | "norm01" | "percentile"
    tile_factor: Union[int, Tuple[int, int], None] = None,
    tile_variant_builder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    *,
    mix_random_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Expand x and phase to canvas_hw.
    - x_mode="interp": bilinear resize
    - x_mode="tile": replicate input by tile_factor (or auto-fill if None)
    - x_mode="mix": per-sample choose between tiled input and random phase-mask amplitude
      (probability controlled by mix_random_ratio)
    - phase is always resized with nearest (mask semantics)
    - tile_factor=None → auto derived from canvas + input size (≈ canvas / (2 * H_in))
    """
    if tile_variant_builder is None and x_mode == "mix":
        def tile_variant_builder(t: torch.Tensor) -> torch.Tensor:
            base = make_jittered_variants(  # small affine jitter per tile
                t,
                n_variants=4,
                max_translate=0.01,
                scale_range=(0.99, 1.01),
                max_rotate_deg=0.1,  # no random small rotation; use 90° steps below
                padding_mode="zeros",
            )  # (B, 4, C, H, W)
            rotated = [
                torch.rot90(base[:, i], k=i, dims=(-2, -1))  # i → 0°,90°,180°,270°
                for i in range(base.size(1))
            ]
            return torch.stack(rotated, dim=1)
    # ensure 4D
    x4 = x if x.dim() == 4 else x.view(x.size(0), x.size(1), x.size(-2), x.size(-1))
    p4 = phase if phase.dim() == 4 else phase.view(phase.size(0), phase.size(1), phase.size(-2), phase.size(-1))

    # derive tile_factor if not provided (scales with canvas and current input size)
    if tile_factor is None:
        H_in, W_in = x4.shape[-2:]
        tf_h = max(1, int(canvas_hw[0] // max(1, 2 * H_in)))
        tf_w = max(1, int(canvas_hw[1] // max(1, 2 * W_in)))
        tile_factor_resolved: Union[int, Tuple[int, int]] = (tf_h, tf_w) if tf_h != tf_w else tf_h
    else:
        tile_factor_resolved = tile_factor

    if x_mode == "interp":
        x_res = _resize_to_canvas(x4, canvas_hw, is_mask=False)
    elif x_mode == "tile":
        x_res = _tile_to_canvas(
            x4, canvas_hw,
            tile_factor=tile_factor_resolved,
            rotate_sequence=(0,1,2,3),
            variant_builder=tile_variant_builder,
        )
    elif x_mode == "mix":
        x_tile = _tile_to_canvas(
            x4, canvas_hw,
            tile_factor=tile_factor_resolved,
            rotate_sequence=(0,1,2,3),
            variant_builder=tile_variant_builder,
        )
        ratio = float(max(0.0, min(1.0, mix_random_ratio)))
        B = x_tile.size(0)
        random_amp = generate_phase_mask(
            batch_size=B,
            phase_mask_dim=tuple(canvas_hw),
            device=x_tile.device,
            randomize=True,
        ).to(dtype=x_tile.dtype)
        if random_amp.size(1) == 1 and x_tile.size(1) > 1:
            random_amp = random_amp.expand(-1, x_tile.size(1), -1, -1)
        elif random_amp.size(1) != x_tile.size(1):
            random_amp = random_amp[:, :x_tile.size(1)]
        mix_choice = (torch.rand(B, 1, 1, 1, device=x_tile.device) < ratio)
        x_res = torch.where(mix_choice, random_amp, x_tile)
    else:
        raise ValueError(f"Unknown x_mode '{x_mode}', use 'interp', 'tile', or 'mix'.")

    p_res = _resize_to_canvas(p4, canvas_hw, is_mask=True)      
    # batch match
    if x_res.size(0) != p_res.size(0):
        if p_res.size(0) == 1 and x_res.size(0) > 1:
            p_res = p_res.repeat(x_res.size(0), 1, 1, 1)
        else:
            raise ValueError(f"Batch mismatch after resize/tile: x={x_res.size(0)}, phase={p_res.size(0)}")
    
    
    norm_key = "none"
    if x_norm_mod is not None:
        if isinstance(x_norm_mod, str):
            norm_key = x_norm_mod.strip().lower()
        else:
            raise TypeError("x_norm_mod must be a string or None.")

    if norm_key == "norm01":
        x_res = _norm01_spatial(x_res)
    elif norm_key == "percentile":
        x_res = percentile_scale_spatial(x_res)
    elif norm_key == "none":
        pass
    else:
        raise ValueError(f"Unknown x_norm_mod '{x_norm_mod}', use 'norm01', 'percentile', or 'none'.")
    
    # p_res = _norm01_spatial(p_res)
    
    return x_res, p_res


def make_jittered_variants(
    x_scaled: torch.Tensor,
    *,
    n_variants: int = 4,
    max_translate: float = 0.01,
    scale_range: Tuple[float, float] = (0.99, 1.01),
    max_rotate_deg: float = 0.1,
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """
    Build jittered per-tile variants of an upsampled block.
    Returns a tensor of shape (B, n_variants, C, H, W).
    """
    if n_variants <= 0:
        raise ValueError("n_variants must be positive")
    B, C, Hs, Ws = x_scaled.shape
    device = x_scaled.device
    variants = []
    max_translate = float(max(0.0, max_translate))
    smin, smax = scale_range
    smin, smax = float(smin), float(smax)
    if smin <= 0 or smax <= 0:
        raise ValueError("scale_range must be positive")
    if smin > smax:
        smin, smax = smax, smin
    rot_rad = float(max_rotate_deg) * math.pi / 180.0

    for _ in range(int(n_variants)):
        dx = torch.empty(B, device=device).uniform_(-max_translate, max_translate)
        dy = torch.empty(B, device=device).uniform_(-max_translate, max_translate)
        scale = torch.empty(B, device=device).uniform_(smin, smax)
        theta = torch.empty(B, device=device).uniform_(-rot_rad, rot_rad)
        cos_t, sin_t = torch.cos(theta) * scale, torch.sin(theta) * scale

        A = torch.zeros(B, 2, 3, device=device)
        A[:, 0, 0] = cos_t
        A[:, 0, 1] = -sin_t
        A[:, 1, 0] = sin_t
        A[:, 1, 1] = cos_t
        A[:, 0, 2] = dx * 2  # normalized translation
        A[:, 1, 2] = dy * 2

        grid = F.affine_grid(A, size=(B, C, Hs, Ws), align_corners=False)
        jittered = F.grid_sample(
            x_scaled,
            grid,
            mode="bilinear",
            padding_mode=padding_mode,
            align_corners=False,
        )
        variants.append(jittered)

    return torch.stack(variants, dim=1)



def _odd(k: int) -> int:
    k = int(round(k))
    return k if (k % 2 == 1) else (k + 1)

def _gaussian_kernel(ks: int, sigma: float, device) -> torch.Tensor:
    ax = torch.arange(ks, device=device) - (ks - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    g = torch.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
    g = g / g.sum().clamp_min(1e-12)
    return g

def _gaussian_kernel_aniso(ks: int, sx: float, sy: float, theta: float, device) -> torch.Tensor:
    ax = torch.arange(ks, device=device) - (ks - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    c, s = math.cos(theta), math.sin(theta)
    xr =  c*xx + s*yy
    yr = -s*xx + c*yy
    g = torch.exp(-(xr**2)/(2*sx*sx) - (yr**2)/(2*sy*sy))
    g = g / g.sum().clamp_min(1e-12)
    return g

def generate_phase_mask(
    batch_size: int,
    phase_mask_dim: Tuple[int, int],
    device: torch.device,
    # —— new defaults tuned for 168×168 ——
    sigma_rel_range: Tuple[float, float] = (2/168, 24/168),  
    aniso_prob: float = 0.5,
    aniso_ratio_range: Tuple[float, float] = (1.2, 4.0),
    dog_ratio_range: Tuple[float, float] = (math.sqrt(2.0), 3.0),
    ks_cap: int = 151,                # cap kernel size for speed (odd)
    # modes
    randomize: Optional[bool] = None,
    mode: str = "mixed",             # "identity"|"gaussian"|"random"|"dog"|"laplacian"|"mixed"
    mixed_probs: Tuple[float, float, float, float, float] = (0.05, 0.5, 0.5, 0.2, 0.2),
    # random kernel style
    random_signed: bool = True,     # if True → zero-mean signed random kernel (edge-y)
) -> torch.Tensor:
    """
    Per-sample kernel via grouped conv, then per-sample normalization to [-1,1].
    Tuned for 168x168 by choosing σ relative to H and deriving ks ≈ 6σ (odd, capped).
    """
    H, W = phase_mask_dim
    B = int(batch_size)

    # Start with zero-mean noise so signed filters have something to center
    masks = torch.rand(B, 1, H, W, device=device)  # ~N(0,1)

    # legacy flag -> mode
    if randomize is not None:
        mode = "mixed" if randomize else "gaussian"
    mode = mode.lower()
    if mode not in {"identity", "gaussian", "random", "dog", "laplacian", "mixed"}:
        raise ValueError(f"Unknown mode '{mode}'")

    # Per-sample mode selection
    if mode == "mixed":
        probs = torch.tensor(mixed_probs, device=device)
        probs = probs / probs.sum()
        picks = torch.multinomial(probs, B, replacement=True).tolist()
        mode_list = [["identity", "gaussian", "random", "dog", "laplacian"][i] for i in picks]
    else:
        mode_list = [mode] * B

    def _draw_sigma_and_ks():
        # sample sigma relative to height
        smin, smax = sigma_rel_range
        sigma = float(torch.empty((), device=device).uniform_(smin, smax).item() * H)
        # 6σ covers ≈99.7%; ensure odd, apply cap
        ks = _odd(min(int(math.ceil(6.0 * sigma)), ks_cap))
        ks = max(ks, 3)  # minimum useful size
        sigma = max(sigma, 0.5)  # avoid degenerate
        return sigma, ks

    kernels, sizes = [], []
    for i in range(B):
        m = mode_list[i]

        if m == "identity":
            ks_i = 1
            k = torch.ones(1, 1, device=device)

        elif m == "gaussian":
            sigma, ks_i = _draw_sigma_and_ks()
            # possibly anisotropic
            if torch.rand((), device=device).item() < aniso_prob:
                ratio = float(torch.empty((), device=device).uniform_(*aniso_ratio_range).item())
                if torch.rand((), device=device).item() < 0.5:
                    sx, sy = sigma, sigma * ratio
                else:
                    sx, sy = sigma * ratio, sigma
                theta = float(torch.empty((), device=device).uniform_(0.0, math.pi).item())
                k = _gaussian_kernel_aniso(ks_i, sx, sy, theta, device)
            else:
                k = _gaussian_kernel(ks_i, sigma, device)

        elif m == "random":
            ks_i = _odd(min(ks_cap, max(7, int(torch.empty((), device=device).uniform_(7, 31).item()))))
            k = torch.rand(ks_i, ks_i, device=device)
            if random_signed:
                k = k * 2 - 1   # signed
                k = k - k.mean()
                k = k / k.abs().sum().clamp_min(1e-12)
            else:
                k = k / k.sum().clamp_min(1e-12)

        elif m == "dog":
            sigma1, ks_i = _draw_sigma_and_ks()
            r = float(torch.empty((), device=device).uniform_(*dog_ratio_range).item())
            sigma2 = sigma1 * r
            # Ensure both kernels fit into capped ks; if sigma2 larger, recompute ks_i
            ks_i = min(ks_i, _odd(min(int(math.ceil(6.0 * sigma2)), ks_cap)))
            ks_i = max(ks_i, 5)
            g1 = _gaussian_kernel(ks_i, sigma1, device)
            g2 = _gaussian_kernel(ks_i, sigma2, device)
            k  = g1 - g2
            k  = k / k.abs().sum().clamp_min(1e-12)

        elif m == "laplacian":
            # Keep small, classic 3x3 (fast, strong edges)
            k = torch.tensor([[0., -1., 0.],
                              [-1., 4., -1.],
                              [0., -1., 0.]], device=device)
            ks_i = 3
            k = k / k.abs().sum().clamp_min(1e-12)

        else:
            raise RuntimeError("Unhandled mode.")

        kernels.append(k)
        sizes.append(ks_i)

    # Pad kernels to common size and stack as (B,1,ks,ks)
    ks_common = max(sizes)
    pad_kernels = []
    for k, ks_i in zip(kernels, sizes):
        if ks_i == ks_common:
            pad_kernels.append(k)
        else:
            pad = (ks_common - ks_i) // 2
            kpad = torch.zeros(ks_common, ks_common, device=device)
            kpad[pad:pad + ks_i, pad:pad + ks_i] = k
            pad_kernels.append(kpad)
    K = torch.stack(pad_kernels, dim=0).unsqueeze(1)  # (B,1,ks_common,ks_common)

    # grouped conv per-sample
    pad = ks_common // 2
    masks_padded = F.pad(masks, (pad, pad, pad, pad), mode="reflect")    # (B,1,H+2p,W+2p)
    x = masks_padded.reshape(1, B, masks_padded.shape[-2], masks_padded.shape[-1])  # (1,B,*,*)
    y = F.conv2d(x, K, groups=B)                                         # (1,B,H,W)
    smoothed = y.reshape(B, 1, H, W)

    # normalize to [0,1] per sample
    mn = smoothed.amin(dim=(-2, -1), keepdim=True)
    mx = smoothed.amax(dim=(-2, -1), keepdim=True)
    smoothed = 1 * (smoothed - mn) / (mx - mn + 1e-8) 
    return smoothed


def channel_relative_mse(y_pred: torch.Tensor,
                         y_true: torch.Tensor,
                         num_channels: int = 16) -> torch.Tensor:
    """
    Relative-channel MSE loss.
    """
    assert y_pred.shape == y_true.shape, "y_pred and y_true must have same shape"
    B_total, C_out, H, W = y_true.shape
    assert B_total % num_channels == 0, (
        f"Batch ({B_total}) must be divisible by num_channels ({num_channels})"
    )
    G = B_total // num_channels

    # Reshape into (G, num_channels, C_out, H, W)
    y_true_g = y_true.view(G, num_channels, C_out, H, W)
    y_pred_g = y_pred.view(G, num_channels, C_out, H, W)

    # Remove per-group channel mean
    y_true_rel = y_true_g - y_true_g.mean(dim=1, keepdim=True)
    y_pred_rel = y_pred_g - y_pred_g.mean(dim=1, keepdim=True)

    # Back to (B_total, C_out, H, W) if you like (optional)
    y_true_rel = y_true_rel.view(B_total, C_out, H, W)
    y_pred_rel = y_pred_rel.view(B_total, C_out, H, W)

    # Relative-channel MSE
    rel_loss = F.mse_loss(y_pred_rel, y_true_rel)
    return rel_loss


def adaptive_gain_clip_safe(x, k=8.0, min_mean=0.05):
    mean_val = torch.mean(x, dim=(-2, -1), keepdim=True)
    safe_mean = torch.clamp(mean_val, min=min_mean)
    x_scaled = x / (safe_mean * k + 1e-8)
    return torch.clamp(x_scaled, 0.0, 1.0)
