"""
Physical-forward/surrogate-backward bridge helpers.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import torch
from torch import nn

from .phase_system import PhaseSystem
from .utils import adaptive_gain_clip_safe, encoding_x_phase_physical, percentile_scale_spatial

class _PhysForwardSurrogateBackward(torch.autograd.Function):
    """
    Custom autograd bridge: physical forward, surrogate backward.
    """

    @staticmethod
    def forward(
        ctx,
        phase: torch.Tensor,
        x: torch.Tensor,
        model: nn.Module,
        phys_forward_callable: Callable,
        canvas_hw: Tuple[int, int],
        mode_str: str,
        norm_mode_str: str,
    ):
        ctx.model = model
        ctx.phys_forward_callable = phys_forward_callable
        ctx.canvas_hw = canvas_hw
        ctx.mode_str = mode_str
        ctx.norm_mode_str = norm_mode_str
        ctx.save_for_backward(phase.detach(), x.detach())
        y_phys = phys_forward_callable(x, phase)
        ctx.output_hw = tuple(y_phys.shape[-2:]) if isinstance(y_phys, torch.Tensor) and y_phys.dim() >= 2 else None
        ctx.output_template = y_phys.detach() if isinstance(y_phys, torch.Tensor) else None
        return y_phys

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        phase_saved, x_saved = ctx.saved_tensors
        model = ctx.model
        target_hw = getattr(ctx, "output_hw", None)
        target_template = getattr(ctx, "output_template", None)
        canvas_hw = getattr(ctx, "canvas_hw", None)
        mode_str = getattr(ctx, "mode_str", "mix")
        norm_mode_str = getattr(ctx, "norm_mode_str", None)

        phase_saved = phase_saved.detach().requires_grad_(True)
        x_base = x_saved.detach().requires_grad_(True)
        with torch.enable_grad():
            x_enc, phase_enc = encoding_x_phase_physical(
                x_base,
                phase_saved,
                canvas_hw=canvas_hw,
                x_mode=mode_str,
                x_norm_mod=norm_mode_str,
            )
            if target_template is not None:
                y_model = model(x_enc, phase_enc, target=target_template)
            elif target_hw is not None:
                y_model = model(x_enc, phase_enc, target_size=target_hw)
            else:
                y_model = model(x_enc, phase_enc)
            grad_phase, grad_x = torch.autograd.grad(
                outputs=y_model,
                inputs=(phase_saved, x_base),
                grad_outputs=grad_out,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
        return grad_phase, grad_x, None, None, None, None, None


class OpticalBridge:
    """
    Coordinates canvasing and the custom autograd bridge for each optical layer.
    """

    def __init__(
        self,
        optical_sys: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
        surrogate: nn.Module,
        enc_canvas_hw: Tuple[int, int],
        modes: List[str],
        x_norm_modes: Optional[List[Optional[str]]] = None,
        *,
        in_silico_mode: bool = False,
        channel_num: Optional[int] = None,
        learnable_gain: bool = False,
        gain_init: float = 8.0,
        apply_output_gain: bool = True,
    ) -> None:
        self.in_silico_mode = bool(in_silico_mode)
        if (not self.in_silico_mode) and optical_sys is None:
            raise ValueError("optical_sys must be provided unless in_silico_mode=True.")
        self.optical_sys = optical_sys
        self.surrogate = surrogate
        self.enc_canvas_hw = tuple(enc_canvas_hw)
        self.modes = list(modes)
        if x_norm_modes is None:
            x_norm_modes = ["percentile"] * len(self.modes)
        if len(x_norm_modes) != len(self.modes):
            raise ValueError("x_norm_modes must match the number of layers.")
        self.x_norm_modes = list(x_norm_modes)
        self.encoded_cache: List[Optional[torch.Tensor]] = [None for _ in self.modes]
        self.phase_cache: List[Optional[torch.Tensor]] = [None for _ in self.modes]
        self.phys_forward_funcs: List[Callable] = []
        if not self.in_silico_mode:
            self.phys_forward_funcs = [
                self._make_phys_forward(idx, mode, self.x_norm_modes[idx])
                for idx, mode in enumerate(self.modes)
            ]

        self.channel_num = int(channel_num) if channel_num is not None else None
        self.learnable_gain = bool(learnable_gain)
        self.gain_init = float(gain_init)
        self.apply_output_gain = bool(apply_output_gain)
        self.gain_min_mean = 0.05
        self.gain_raw: Optional[nn.Parameter] = None
        if self.learnable_gain:
            if self.channel_num is None or self.channel_num <= 0:
                raise ValueError("channel_num must be provided and >0 when learnable_gain=True.")
            log_gain = math.log(max(self.gain_init, 1e-6))
            try:
                first_param = next(self.surrogate.parameters())
                param_device = first_param.device
            except StopIteration:
                param_device = torch.device("cpu")
            self.gain_raw = nn.Parameter(
                torch.full((len(self.modes), self.channel_num), log_gain, device=param_device)
            )

    def _make_phys_forward(
        self,
        idx: int,
        mode_str: str,
        norm_mode_str: Optional[str],
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        def phys_forward(x_in: torch.Tensor, phase_in: torch.Tensor) -> torch.Tensor:
            x_enc, phase_enc = self._encode_inputs(idx, x_in, phase_in, mode_str, norm_mode_str)
            with torch.no_grad():
                y_phys = self.optical_sys(x_enc, phase_enc)
            return y_phys

        return phys_forward

    def _encode_inputs(
        self,
        idx: int,
        x_in: torch.Tensor,
        phase_in: torch.Tensor,
        mode_str: str,
        norm_mode_str: Optional[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_enc, phase_enc = encoding_x_phase_physical(
            x_in,
            phase_in,
            canvas_hw=self.enc_canvas_hw,
            x_mode=mode_str,
            x_norm_mod=norm_mode_str,
        )
        self.encoded_cache[idx] = x_enc.detach()
        self.phase_cache[idx] = phase_enc.detach()
        return x_enc, phase_enc

    def gain_parameters(self) -> List[nn.Parameter]:
        return [self.gain_raw] if isinstance(self.gain_raw, nn.Parameter) else []

    def _gain_k(self, layer_idx: int, batch: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.gain_raw is None:
            return torch.as_tensor(self.gain_init, device=device, dtype=dtype)
        if self.channel_num is None or self.channel_num <= 0:
            raise ValueError("channel_num must be set when learnable_gain=True.")
        if layer_idx < 0 or layer_idx >= self.gain_raw.size(0):
            raise IndexError(f"layer_idx {layer_idx} out of range for gain_raw with {self.gain_raw.size(0)} layers.")
        if batch % self.channel_num != 0:
            raise ValueError(f"Batch size {batch} must be divisible by channel_num {self.channel_num}.")
        gain_layer = self.gain_raw[layer_idx].exp().clamp_min(1e-6).to(device=device)
        ch_idx = torch.arange(batch, device=device) % self.channel_num
        return gain_layer[ch_idx].view(batch, 1, 1, 1).to(dtype=dtype)

    def _apply_gain(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if not self.apply_output_gain:
            return x
        if not self.learnable_gain and self.gain_raw is None:
            return percentile_scale_spatial(x)
        k = self._gain_k(layer_idx, x.size(0), device=x.device, dtype=x.dtype)
        return adaptive_gain_clip_safe(x, k=k, min_mean=self.gain_min_mean)

    def _surrogate_forward(
        self,
        idx: int,
        x_in: torch.Tensor,
        phase_in: torch.Tensor,
        mode_str: str,
        norm_mode_str: Optional[str],
        *,
        track_grads: bool,
    ) -> torch.Tensor:
        x_enc, phase_enc = self._encode_inputs(idx, x_in, phase_in, mode_str, norm_mode_str)
        if track_grads:
            return self.surrogate(x_enc, phase_enc)
        with torch.no_grad():
            return self.surrogate(x_enc, phase_enc)

    def _apply_layer(
        self,
        idx: int,
        x_in: torch.Tensor,
        phase_in: torch.Tensor,
        mode_str: str,
        norm_mode_str: Optional[str],
        *,
        track_grads: bool,
    ) -> torch.Tensor:
        if self.in_silico_mode:
            return self._surrogate_forward(idx, x_in, phase_in, mode_str, norm_mode_str, track_grads=track_grads)
        if track_grads:
            return _PhysForwardSurrogateBackward.apply(
                phase_in,
                x_in,
                self.surrogate,
                self.phys_forward_funcs[idx],
                self.enc_canvas_hw,
                mode_str,
                norm_mode_str,
            )
        return self.phys_forward_funcs[idx](x_in, phase_in)

    def run(
        self,
        x: torch.Tensor,
        phase_system: PhaseSystem,
    ) -> Tuple[
        torch.Tensor,
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        x_current = x
        layer_inputs: List[torch.Tensor] = []
        layer_phases: List[torch.Tensor] = []
        layer_outputs: List[torch.Tensor] = []
        layer_inputs_pre: List[torch.Tensor] = []
        base_canvas_for_phase: Optional[torch.Tensor] = None
        if getattr(phase_system, "structural_nonlinearity", False):
            B0 = x.size(0)
            dummy_phase = torch.zeros(B0, 1, *self.enc_canvas_hw, device=x.device, dtype=x.dtype)
            base_canvas_for_phase, _ = encoding_x_phase_physical(
                x,
                dummy_phase,
                canvas_hw=self.enc_canvas_hw,
                x_mode=self.modes[0],
                x_norm_mod=self.x_norm_modes[0],
            )
        for idx, phase_param in enumerate(phase_system.phase_parameters()):
            B = x_current.size(0)
            layer_inputs_pre.append(x_current.detach())
            layer_mode = self.modes[idx]
            layer_norm_mode = self.x_norm_modes[idx]
            x_canvas_for_phase: Optional[torch.Tensor] = None
            if getattr(phase_system, "structural_nonlinearity", False):
                x_canvas_for_phase = base_canvas_for_phase
            phase_batch = phase_system.expand_to_batch(
                phase_param,
                B,
                x_input=x_canvas_for_phase,
                layer_idx=idx,
            )
            x_current = self._apply_layer(
                idx,
                x_current,
                phase_batch,
                layer_mode,
                layer_norm_mode,
                track_grads=True,
            )
            x_current = self._apply_gain(x_current, idx)
            x_encoded = self.encoded_cache[idx] if self.encoded_cache[idx] is not None else x_current
            phase_encoded = self.phase_cache[idx] if self.phase_cache[idx] is not None else phase_batch
            layer_inputs.append(x_encoded.detach())
            layer_phases.append(phase_encoded.detach())
            layer_outputs.append(x_current.detach())
        return x_current, layer_inputs, layer_phases, layer_outputs, layer_inputs_pre

    def forward_eval(
        self,
        x: torch.Tensor,
        phase_system: PhaseSystem,
    ) -> torch.Tensor:
        x_current = x
        base_canvas_for_phase: Optional[torch.Tensor] = None
        if getattr(phase_system, "structural_nonlinearity", False):
            B0 = x.size(0)
            dummy_phase = torch.zeros(B0, 1, *self.enc_canvas_hw, device=x.device, dtype=x.dtype)
            base_canvas_for_phase, _ = encoding_x_phase_physical(
                x,
                dummy_phase,
                canvas_hw=self.enc_canvas_hw,
                x_mode=self.modes[0],
                x_norm_mod=self.x_norm_modes[0],
            )
        for idx, phase_param in enumerate(phase_system.phase_parameters()):
            B = x_current.size(0)
            layer_mode = self.modes[idx]
            layer_norm_mode = self.x_norm_modes[idx]
            x_canvas_for_phase: Optional[torch.Tensor] = None
            if getattr(phase_system, "structural_nonlinearity", False):
                x_canvas_for_phase = base_canvas_for_phase
            phase_batch = phase_system.expand_to_batch(
                phase_param,
                B,
                x_input=x_canvas_for_phase,
                layer_idx=idx,
            )
            x_current = self._apply_layer(
                idx,
                x_current,
                phase_batch,
                layer_mode,
                layer_norm_mode,
                track_grads=False,
            )
            x_current = self._apply_gain(x_current, idx)
        return x_current
