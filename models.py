import os
import pickle
import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import numpy as np
from tqdm import tqdm   
import torch.nn.functional as F
from typing import Optional


def SafeGroupNorm(num_channels: int, target_groups: int = 32):
    """
    Auto-selects a valid number of groups close to target_groups.
    If num_channels is 45, target 32 -> will pick 15 groups (3 channels/group)
    or 9 groups (5 channels/group) depending on divisibility.
    """
    if num_channels % target_groups == 0:
        return nn.GroupNorm(target_groups, num_channels)
    best_g = 1
    for g in range(target_groups, 0, -1):
        if num_channels % g == 0:
            best_g = g
            break
    return nn.GroupNorm(best_g, num_channels)
 
class PSFNet(nn.Module):
    def __init__(self, data_shapes, device=None, cache_path="cached_psf.pt"):
        super().__init__()
        self.input_shape = data_shapes['input']
        self.out_shape = data_shapes['output']
        self.no_run = True 
        # Try loading PSF from file, else generate and save
        if os.path.exists(cache_path):
            psf_tensor = torch.load(cache_path, map_location=device)
        else:
            # generate_psf should be defined elsewhere or mocked
            # You can replace generate_psf with an appropriate function to compute the PSF
            from .psf import generate_psf  # type: ignore
            psf_tensor = generate_psf(device = device)
            torch.save(psf_tensor.cpu(), cache_path)

        # psf_tensor has shape (1,1,ph,pw)
        self.register_buffer("psf", psf_tensor.to(device))

    def forward(self, x, phase_mask):
        phase_weight = F.interpolate(phase_mask, scale_factor = 4, mode='nearest')
        ref_size = phase_weight.shape[-2:]
        if x.shape[-2:] != ref_size:
            x = F.interpolate(x, size=ref_size, mode='bilinear', align_corners=False)

        x = x.to(self.psf.device)
        phase_weight = phase_weight.to(self.psf.device)

        # apply phase modulation
        self.x_phase = x * torch.exp(1j * 2 * torch.pi * phase_weight)

        if self.no_run: #only at first use
            B, C, H, W = self.x_phase.shape
            ph, pw = self.psf.shape[-2:]  # original PSF height/width
    
            # compute symmetric padding to go from (ph,pw) → (H,W)
            pad_h = H - ph
            pad_w = W - pw
            # F.pad format: (left, right, top, bottom)
            pad = (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2)
    
            # zero-pad PSF to (1,1,H,W), then expand to (B,1,H,W)
            psf_padded = F.pad(self.psf, pad)
            psf_padded = psf_padded.expand(B, -1, -1, -1).contiguous()
            psf_padded = torch.fft.ifftshift(psf_padded, dim=(-2, -1))
            self.PSF_fft = torch.fft.fft2(psf_padded)
            self.no_run= False #change flag when the PSF is computed once and for all

        # FFT-based convolution
        X_fft   = torch.fft.fft2(self.x_phase)
        Y_fft   = X_fft * self.PSF_fft
        y       = torch.fft.ifft2(Y_fft)
        #y       = torch.fft.fftshift(y, dim=(-2, -1))

        # crop to desired output shape
        h, w = y.shape[-2], y.shape[-1]
        crop_top  = (h - self.out_shape[0]) // 2
        crop_left = (w - self.out_shape[1]) // 2
        y_cropped = y[:, :, crop_top:crop_top + self.out_shape[0], crop_left:crop_left + self.out_shape[1]]

        # intensity and normalize to [0,1]
        intensity = torch.abs(y_cropped) ** 2
        mn = intensity.amin(dim=(-2, -1), keepdim=True)
        mx = intensity.amax(dim=(-2, -1), keepdim=True)
        intensity = (intensity - mn) / (mx - mn + 1e-8)
        return intensity


# ---------------- Small-k residual block ----------------
class ResidualBlock3x3(nn.Module):
    """
    Compact residual block with 3x3 kernels.
    - stride=2 halves H,W
    - stride=1 keeps H,W
    norm: 'bn' | 'gn' | 'none'
    """
    def __init__(self, in_ch, out_ch=None, stride=1, norm='bn'):
        super().__init__()
        out_ch = out_ch or in_ch
        if norm == 'bn':
            Norm = lambda c: SafeGroupNorm(c)
        elif norm == 'gn':
            Norm = lambda c: SafeGroupNorm(c)
        else:
            Norm = lambda c: nn.Identity()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            Norm(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            Norm(out_ch)
        )
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
            if (in_ch != out_ch or stride != 1) else nn.Identity()
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.skip(x) + self.conv(x))

# --------------- SE & Spatial attention (unchanged) ---------------
class SEBlock(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // r, 1),
            nn.GELU(),
            nn.Conv2d(channels // r, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        att = self.sig(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att

# --------- Lightweight learnable upsampler (refine -> resize -> refine) ----------
class LearnedUpsampler(nn.Module):
    """
    A size-agnostic, learnable upsampler that maps any (B,C,H,W) to (B,C_out,H_t,W_t).
    - Uses shallow residual refinement before and after a robust resize.
    - Works for arbitrary scale factors (not constrained to powers of 2).
    """
    def __init__(self, c_in: int, c_out: int, c_mid: int = 32, norm: str = 'bn'):
        super().__init__()
        if norm == 'bn':
            Norm = lambda c: SafeGroupNorm(c)
        elif norm == 'gn':
            Norm = lambda c: SafeGroupNorm(c)
        else:
            Norm = lambda c: nn.Identity()

        self.pre = nn.Sequential(
            nn.Conv2d(c_in, c_mid, 3, padding=1, bias=False), Norm(c_mid), nn.GELU(),
            nn.Conv2d(c_mid, c_mid, 3, padding=1, bias=False), Norm(c_mid), nn.GELU(),
        )
        self.post = nn.Sequential(
            nn.Conv2d(c_mid, c_mid, 3, padding=1, bias=False), Norm(c_mid), nn.GELU(),
            nn.Conv2d(c_mid, c_out, 3, padding=1, bias=False)
        )

    def forward(self, x: torch.Tensor, canvas_hw: tuple[int, int]) -> torch.Tensor:
        h_t, w_t = canvas_hw
        z = self.pre(x)
        # robust, anti-aliased resize; bilinear is stable and fast
        z = F.interpolate(z, size=(h_t, w_t), mode='bilinear', align_corners=False)
        z = self.post(z)
        return z

# --------------- Optional: dilated RF in bottleneck ----------------
class BottleneckDilated(nn.Module):
    """
    Three 3x3 convs with dilations 1,2,3 to recover large receptive field.
    """
    def __init__(self, ch, norm='bn', p_drop=0.01):
        super().__init__()
        if norm == 'bn':
            Norm = lambda c: SafeGroupNorm(c)
        elif norm == 'gn':
            Norm = lambda c: SafeGroupNorm(c)
        else:
            Norm = lambda c: nn.Identity()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, dilation=1, bias=False), Norm(ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=2, dilation=2, bias=False), Norm(ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=3, dilation=3, bias=False), Norm(ch), nn.GELU(),
            nn.Dropout(p=p_drop),
        )
    def forward(self, x): return self.net(x)

class SoftClip01(nn.Module):
    def __init__(self, beta: float = 2.0):
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.beta
        # softplus(z) ≈ relu(z) but smooth
        return x - F.softplus(b * (x - 1.0)) / b + F.softplus(-b * x) / b
    

class ChannelMidResBlock(nn.Module):
    """
    Small nonlinear residual block used per channel at mid resolution.
    Conv -> GELU -> Conv -> GELU -> Conv, returns residual only.
    """
    def __init__(self, channels: int, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=3, padding=1, bias=True),
        )
        # Start with small residual scaling for stability; it will learn its own gain.
        self.res_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_scale * self.net(x)


class ChannelBottleneckBlock(nn.Module):
    """
    Compact per-channel residual applied at the bottleneck (Hc/8) to inject
    coarse-scale corrections before decoding.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_scale * self.net(x)


class ChannelHead(nn.Module):
    def __init__(self, in_ch, out_ch, hidden=64):
        super().__init__()
        
        # 1) Local mixing: captures basic per-pixel differences 
        self.conv1 = nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1, bias=True)
        self.act1  = nn.GELU()
        
        # 2) Dilated conv: captures medium-range correlation (PSF, blur envelope)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=2, dilation=2, bias=True)
        self.act2  = nn.GELU()
        
        # 3) Depthwise conv: captures high-frequency speckle grain structure
        self.conv3 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1,
                               groups=hidden, bias=True)
        self.act3  = nn.GELU()
        
        # 4) Bottleneck fusion
        self.conv4 = nn.Conv2d(hidden, hidden // 2, kernel_size=1, bias=True)
        self.act4  = nn.GELU()
        
        # 5) Final projection to 1-channel residual
        self.conv_out = nn.Conv2d(hidden // 2, out_ch, kernel_size=1, bias=True)
        
        # Optional: learnable residual scaling for stability
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, z):
        h = self.act1(self.conv1(z))
        h = self.act2(self.conv2(h))
        h = self.act3(self.conv3(h))
        h = self.act4(self.conv4(h))
        
        # Residual output
        return self.res_scale * self.conv_out(h)

# --------------------------- Model --------------------------------

class Surrogate_OpticalNet_Unet(nn.Module):
    """
    Surrogate optical neural network using a U-Net-like architecture.

    This model accepts an arbitrary-sized input image ``x`` and a phase mask ``phase_mask``,
    upsamples both inputs to a common canvas resolution ``enc_canvas_hw`` via learnable
    upsamplers, and then processes the combined channels through an encoder-decoder
    architecture with multi-harmonic phase encoding.  The output is matched to a
    target resolution via either resizing or center padding/cropping.

    """
    def __init__(self, in_channels: int = 1, base_channels: int = 45, out_channels: int = 1,
                 norm='bn',
                 default_out_hw=(182, 182),
                 match_mode: str = 'resize',
                 upsample_mode: str = 'nearest',
                 enc_canvas_hw: tuple[int, int] = (168, 168),     # shared encoder canvas
                 phase_harmonics: int = 20,   # multi-harmonic phase encoding
                 num_channels: int = 16,
                 channel_mid_hidden: int = 45):  # width of per-channel mid block
        super().__init__()
        self.in_channels = in_channels
        self.default_out_hw = tuple(default_out_hw)
        self.match_mode = match_mode
        self.upsample_mode = upsample_mode
        self.phase_harmonics = phase_harmonics
        self.enc_canvas_hw = tuple(enc_canvas_hw)  # must be divisible by 8 for 3 downsamples
        self.channel_mid_hidden = channel_mid_hidden
        self.num_channels = int(num_channels)

        Hc, Wc = self.enc_canvas_hw
        if (Hc % 8 != 0) or (Wc % 8 != 0):
            raise ValueError(f"enc_canvas_hw must be divisible by 8, got {self.enc_canvas_hw}")

        # ---- Learnable upsamplers to the shared canvas
        # Keep per-branch heads simple and fast; refine -> resize -> refine
        self.x_upsampler     = LearnedUpsampler(c_in=in_channels, c_out=in_channels, c_mid=32, norm=norm)
        self.phase_upsampler = LearnedUpsampler(c_in=1,          c_out=1,          c_mid=32, norm=norm)

        # ---- Encoder input channels: image + 2*harmonics (cos/sin) + CoordConv(3: x,y,r)
        enc_in = in_channels + 2 * phase_harmonics + 3
        self.enc_in = enc_in
        self.base = base_channels
        self.out_channels = out_channels

        # ---------------- Encoder: Hc -> Hc/2 -> Hc/4 -> Hc/8 ----------------
        self.enc1 = ResidualBlock3x3(enc_in, base_channels,     stride=2, norm=norm)
        self.enc2 = ResidualBlock3x3(base_channels, base_channels*2, stride=2, norm=norm)
        self.enc3 = ResidualBlock3x3(base_channels*2, base_channels*4, stride=2, norm=norm)

        # Bottleneck at Hc/8 with richer channels
        self.bottleneck = nn.Sequential(
            ResidualBlock3x3(base_channels*4, base_channels*4, stride=1, norm=norm),
            BottleneckDilated(base_channels*4, norm=norm, p_drop=0.01),
            SEBlock(base_channels * 4),
            SpatialAttention(),
        )

        # ---------------- Decoder (PixelShuffle): Hc/8 -> Hc/4 -> Hc/2 -> Hc ----------------
        self.dec3_up = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*8, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2)  # -> channels: base_channels*2
        )
        self.fuse3 = nn.Conv2d(base_channels*2 + base_channels*2, base_channels*2, kernel_size=1)
        self.dec3_rb = ResidualBlock3x3(base_channels*2, base_channels*2, stride=1, norm='none')

        self.dec2_up = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2)  # -> channels: base_channels
        )
        self.fuse2 = nn.Conv2d(base_channels + base_channels, base_channels, kernel_size=1)
        self.dec2_rb = ResidualBlock3x3(base_channels, base_channels, stride=1, norm='none')

        self.dec1_up = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2)  # -> channels: base_channels
        )
        self.fuse1 = nn.Conv2d(base_channels + enc_in, base_channels, kernel_size=1)
        self.dec1_rb = ResidualBlock3x3(base_channels, base_channels, stride=1, norm='none')

        # ---------------- Receptive-field head (depthwise refinement) ----------------
        self.final = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(base_channels, out_channels, 1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            nn.Conv2d(out_channels, out_channels, 1),
            SoftClip01(beta=3.0),
        )
        self.emb_dim = 512           # higher-rank embedding for finer channel-specific structure
        # Per-channel nonlinear mid-resolution block applied at d2 (Hc/2, Wc/2)
        self.channel_mid_blocks = nn.ModuleList([
            ChannelMidResBlock(base_channels, hidden=channel_mid_hidden)
            for _ in range(self.num_channels)
        ])
        # Per-channel coarse residual at bottleneck (Hc/8)
        self.channel_bottleneck_blocks = nn.ModuleList([
            ChannelBottleneckBlock(base_channels * 4) for _ in range(self.num_channels)
        ])

        # Low-rank spatial basis maps: (emb_dim, Hc, Wc)
        # Initialized small to avoid overpowering U-Net output

        self.spatial_basis = nn.Parameter(
            0.005 * torch.randn(self.emb_dim, Hc, Wc)
        )

        # Learnable channel identity embedding
        self.channel_emb = nn.Embedding(self.num_channels, self.emb_dim)

        # Correction head that adjusts output per channel
        self.channel_head = ChannelHead(
            in_ch = self.out_channels + 1,
            out_ch = self.out_channels,
            hidden = 64  # try 48, 64, 96
        )

        # Parameter accounting: with default mid-width 32 we get ≈57/43 shared/per-channel
        # (≈54/46 if channel_mid_hidden=48).
        self.param_stats = self._parameter_split()
    @staticmethod
    def _center_pad_to(x: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        """Center-pad (or crop if negative pad) to target H, W."""
        B, C, H, W = x.shape
        Ht, Wt = out_hw
        pad_h = max(Ht - H, 0)
        pad_w = max(Wt - W, 0)
        x = F.pad(x, (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2))
        H2, W2 = x.shape[-2:]
        off_h = max((H2 - Ht)//2, 0)
        off_w = max((W2 - Wt)//2, 0)
        return x[..., off_h:off_h+Ht, off_w:off_w+Wt]

    def _match_to_target(self, out: torch.Tensor, target: Optional[torch.Tensor] = None, target_size: Optional[tuple[int, int]] = None) -> torch.Tensor:
        if target is not None:
            tgt_hw = target.shape[-2:]
        elif target_size is not None:
            tgt_hw = tuple(target_size)
        else:
            tgt_hw = self.default_out_hw

        if out.shape[-2:] == tgt_hw:
            return out

        if self.match_mode == 'resize':
            return F.interpolate(out, size=tgt_hw, mode=self.upsample_mode,
                                 align_corners=False if self.upsample_mode in ('bilinear','bicubic') else None)
        elif self.match_mode == 'padcrop':
            return self._center_pad_to(out, tgt_hw)
        else:
            raise ValueError(f"Unknown match_mode: {self.match_mode}")

    def _apply_channel_mid(self, feat: torch.Tensor, chan_ids: torch.Tensor) -> torch.Tensor:
        """
        Dispatch mid-resolution residual CNNs per channel.
        feat: (B, C_feat, H, W), chan_ids: (B,) channel indices for each sample.
        Returns residual with same shape as feat.
        """
        residual = torch.zeros_like(feat)
        for ch, block in enumerate(self.channel_mid_blocks):
            idx = (chan_ids == ch)
            if idx.any():
                residual[idx] = block(feat[idx]).to(feat.dtype)
        return residual

    def _apply_channel_bottleneck(self, feat: torch.Tensor, chan_ids: torch.Tensor) -> torch.Tensor:
        """
        Dispatch bottleneck-scale residuals per channel.
        """
        residual = torch.zeros_like(feat)
        for ch, block in enumerate(self.channel_bottleneck_blocks):
            idx = (chan_ids == ch)
            if idx.any():
                residual[idx] = block(feat[idx]).to(feat.dtype)
        return residual

    def _parameter_split(self) -> dict[str, float]:
        """
        Estimate shared vs channel-conditioned parameter counts.
        We treat the mid-resolution blocks, channel embeddings, and spatial_basis
        (used exclusively by the channel head) as the per-channel bucket.
        """
        shared, per_channel = 0, 0
        for name, param in self.named_parameters():
            if name.startswith(("channel_mid_blocks", "channel_bottleneck_blocks", "channel_emb", "spatial_basis")):
                per_channel += param.numel()
            else:
                shared += param.numel()
        total = shared + per_channel
        pct_shared = 100.0 * shared / total if total > 0 else 0.0
        pct_channel = 100.0 * per_channel / total if total > 0 else 0.0
        return {
            "shared": shared,
            "per_channel": per_channel,
            "total": total,
            "pct_shared": pct_shared,
            "pct_per_channel": pct_channel,
        }

    def forward(self, x: torch.Tensor, phase_mask: torch.Tensor,
            target: Optional[torch.Tensor] = None,
            target_size: Optional[tuple[int, int]] = None) -> torch.Tensor:
        # Shared encoder canvas
        Hc, Wc = self.enc_canvas_hw
        want_hw = (Hc, Wc)
    
        # ---- X branch: accept pre-expanded x (x_exp), else upsample
        if x.shape[-2:] == want_hw:
            x_up = x
        else:
            x_up = self.x_upsampler(x, canvas_hw=want_hw)
    
        # ---- Phase branch: always upsample (you pass raw phase_mask here)
        phase_up = self.phase_upsampler(phase_mask, canvas_hw=want_hw)
    
        # If mask batch is 1, broadcast to match x batch (common case)
        if phase_up.size(0) == 1 and x_up.size(0) > 1:
            phase_up = phase_up.expand(x_up.size(0), -1, -1, -1).contiguous()

        # Channel ids for dispatching per-channel modules
        B_total = x_up.size(0)
        C = self.num_channels
        assert B_total % C == 0, (
            f"Surrogate batch ({B_total}) must be divisible by num_channels ({C})."
        )
        chan_ids = torch.arange(C, device=x_up.device, dtype=torch.long).repeat(B_total // C)
    
        # --- Multi-harmonic phase encoding
        phi = phase_up * (2 * torch.pi)
        phis = [phi * m for m in range(1, self.phase_harmonics + 1)]
        phase_feats = torch.cat([torch.cos(p) for p in phis] + [torch.sin(p) for p in phis], dim=1)
    
        # --- CoordConv channels (x,y in [-1,1]) on the actual canvas
        B, _, H, W = x_up.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=x_up.device, dtype=x_up.dtype),
            torch.linspace(-1.0, 1.0, W, device=x_up.device, dtype=x_up.dtype),
            indexing='ij'
        )
        r = torch.sqrt(xx**2 + yy**2)
        coords = torch.stack([xx, yy, r], dim=0).unsqueeze(0).expand(B, -1, H, W)
    
        # Encoder input at canvas size
        x_cat = torch.cat([x_up, phase_feats, coords], dim=1)
    
        # ---------------- Encoder path ----------------
        x1 = self.enc1(x_cat)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        # Bottleneck
        x_mid = self.bottleneck(x3)
        x_mid = x_mid + self._apply_channel_bottleneck(x_mid, chan_ids)
    
        # ---------------- Decoder path ----------------
        d3_up = self.dec3_up(x_mid)
        d3 = self.dec3_rb(self.fuse3(torch.cat([d3_up, x2], dim=1)))
    
        d2_up = self.dec2_up(d3)
        d2 = self.dec2_rb(self.fuse2(torch.cat([d2_up, x1], dim=1)))
        # Insert per-channel nonlinear block at 84x84 (Hc/2) resolution
        d2 = d2 + self._apply_channel_mid(d2, chan_ids)
    
        d1_up = self.dec1_up(d2)
        d1 = self.dec1_rb(self.fuse1(torch.cat([d1_up, x_cat], dim=1)))
    
        out_canvas = self.final(d1)  # (B, out_ch, Hc, Wc)
    
        # Match to target resolution
        #out = self._match_to_target(out_canvas, target=target, target_size=target_size)
        
        # ----------------------- Channel-conditioned correction -----------------------
        # out_canvas: (B, 1, Hc, Wc) — same shape as before

        B_check = out_canvas.size(0)
        assert B_check == B_total, "Batch size changed between shared decoder and channel head."
        Hc, Wc = out_canvas.shape[-2:]

        # Embeddings: (B, emb_dim)
        mix = self.channel_emb(chan_ids)  # (B_total, emb_dim)
        mix = F.layer_norm(mix, [self.emb_dim])
        # Spatial basis: (K, Hc, Wc)
        basis = self.spatial_basis        # (emb_dim, Hc, Wc)

        # Compute per-sample spatial embedding:
        # emb_map[b,h,w] = Σ_k mix[b,k] * basis[k,h,w]
        emb_map = torch.einsum("bk,khw->bhw", mix, basis)  # (B, Hc, Wc)
        emb_map = emb_map.unsqueeze(1)                     # (B,1,Hc,Wc)

        # Concatenate U-Net output with channel conditioning map
        z = torch.cat([out_canvas, emb_map], dim=1)        

        # Predict channel-specific correction
        delta = self.channel_head(z)                  # (B, 1, Hc, Wc)

        # Apply correction
        out_corrected = out_canvas + delta

        # Match output size exactly as before
        out = self._match_to_target(out_corrected, target=target, target_size=target_size)

        return out  # (B,1,H,W)
