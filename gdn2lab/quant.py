"""Native 8-bit training for GDN-2.

Two genuinely different 8-bit paths are supported:

``int8``  — INT8 QAT.  Weights are quantised per output channel, activations
per token, both to signed int8.  On CUDA/CPU the forward matmul is a *real*
integer GEMM (``torch._int_mm`` -> int32 accumulate), i.e. it uses INT8
Tensor Cores on a free Colab T4.  Gradients use a straight-through estimator
and an fp32 master copy of the weights (standard QAT).

``fp8``   — FP8 (e4m3) fake quantisation with per-tensor scaling.  On Hopper /
Ada GPUs real FP8 matmuls are possible via torchao; a free T4 (Turing) has no
FP8 Tensor Cores, so this path is emulated.  It is kept because it produces a
valid FP8-quantised checkpoint for newer hardware.

Optimiser states can additionally be 8-bit (bitsandbytes AdamW8bit), which is
where most of the VRAM saving comes from when training from scratch.

IMPORTANT: the GDN-2 recurrence itself (decay/gates/state) always stays in
fp32.  Only the linear projections are 8-bit — quantising the cumulative
log-decay destroys long-range memory.
"""
from __future__ import annotations

import contextlib
import warnings
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


@contextlib.contextmanager
def _no_autocast(device: torch.device):
    """Disable AMP around the quantisation math (we do our own fp32 casts)."""
    try:
        with torch.autocast(device_type=device.type, enabled=False):
            yield
    except (RuntimeError, ValueError, TypeError):
        with contextlib.nullcontext():
            yield


def _ste_fake_quant(x: Tensor, scale: Tensor, scheme: str) -> Tensor:
    """Fake quant with straight-through gradients."""
    xf, sf = x.float(), scale.float()
    if scheme == "fp8_e4m3":
        q = (xf / sf).to(torch.float8_e4m3fn).float()
    else:
        q = torch.round((xf / sf).clamp(-127.0, 127.0)).clamp(-127.0, 127.0)
    deq = q * sf
    return x + (deq - x).detach()


class _Int8Gemm(torch.autograd.Function):
    """Real int8 GEMM forward + STE backward."""

    @staticmethod
    def forward(ctx, x: Tensor, w: Tensor, sx: Tensor, sw: Tensor):
        xf, wf = x.float(), w.float()
        xq = torch.round((xf / sx.float()).clamp(-127, 127)).clamp(-127, 127)
        wq = torch.round((wf / sw.float()).clamp(-127, 127)).clamp(-127, 127)
        xd, wd = xq * sx.float(), wq * sw.float()
        m = xd.reshape(-1, xd.shape[-1])
        int_result = torch._int_mm(xq.reshape(-1, xq.shape[-1]).to(torch.int8),
                                   wq.to(torch.int8).t())  # int32 [M, out]
        out = int_result.float() * (sx.float().reshape(-1, 1) * sw.float().t().reshape(1, -1))
        ctx.save_for_backward(xd, wd)
        ctx.x_shape = x.shape
        return out.reshape(*x.shape[:-1], w.shape[0]).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        xd, wd = ctx.saved_tensors
        go = grad_out.float().reshape(-1, grad_out.shape[-1])
        x2 = xd.reshape(-1, xd.shape[-1])
        gx = (go @ wd).reshape(*grad_out.shape[:-1], wd.shape[1])
        gw = go.t() @ x2
        return gx.to(grad_out.dtype), gw.to(grad_out.dtype), None, None


class QuantLinear(nn.Module):
    """Drop-in ``nn.Linear`` replacement with 8-bit weights/activations.

    ``weight`` is kept as an fp32 master parameter, so tiny AdamW updates are
    not lost; the *forward* only ever sees 8-bit values.  Use
    :func:`fold_quantization` before export to bake the quantised values into a
    standard checkpoint.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 scheme: str = "int8", int8_gemm: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scheme = scheme
        self.int8_gemm = int8_gemm
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @classmethod
    def from_linear(cls, linear: nn.Linear, scheme: str = "int8",
                    int8_gemm: bool = True) -> "QuantLinear":
        q = cls(linear.in_features, linear.out_features,
                bias=linear.bias is not None, scheme=scheme, int8_gemm=int8_gemm)
        q.weight = linear.weight  # keep the Parameter identity (weight tying survives)
        q.bias = linear.bias
        return q

    def scales(self, x: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
        eps = 1e-8
        if self.scheme == "fp8_e4m3":
            sx = x.float().abs().amax().clamp(min=eps) / 448.0
            sw = w.float().abs().amax().clamp(min=eps) / 448.0
            return sx.detach().reshape(1), sw.detach().reshape(1)
        sx = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=eps) / 127.0
        sw = w.float().abs().amax(dim=-1, keepdim=True).clamp(min=eps) / 127.0
        return sx.detach(), sw.detach()

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        with _no_autocast(x.device):
            xf, wf = x.float(), w.float()
            sx, sw = self.scales(xf, wf)
            use_gemm = (
                self.scheme == "int8" and self.int8_gemm and x.is_cuda
                and hasattr(torch, "_int_mm")
            )
            if use_gemm:
                try:
                    out = _Int8Gemm.apply(xf, wf, sx, sw)
                except (RuntimeError, NotImplementedError):
                    out = F.linear(_ste_fake_quant(xf, sx, self.scheme),
                                   _ste_fake_quant(wf, sw, self.scheme))
            else:
                out = F.linear(_ste_fake_quant(xf, sx, self.scheme),
                               _ste_fake_quant(wf, sw, self.scheme))
            if self.bias is not None:
                out = out + self.bias.float()
        return out.to(x.dtype)

    def folded_weight(self) -> Tensor:
        """The exact 8-bit weight values (dequantised) for export."""
        w = self.weight.detach().float()
        if self.scheme == "fp8_e4m3":
            sx = w.abs().amax().clamp(min=1e-8) / 448.0
            return (w / sx).to(torch.float8_e4m3fn).float() * sx
        sw = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
        return torch.round((w / sw).clamp(-127, 127)).clamp(-127, 127) * sw

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"scheme={self.scheme}, int8_gemm={self.int8_gemm}")


def _iter_linears(module: nn.Module, prefix: str = ""):
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, (nn.Linear, QuantLinear)):
            yield module, name, full, child
        else:
            yield from _iter_linears(child, full)


def convert_linears(model: nn.Module, scheme: str = "int8", int8_gemm: bool = True,
                    quantize_lm_head: bool = False,
                    quantize_embeddings: bool = False) -> int:
    """Replace trainable ``nn.Linear`` projections with :class:`QuantLinear`.

    Embeddings and the tied LM head stay in higher precision by default.
    Returns the number of converted layers.
    """
    n = 0
    for parent, name, full, linear in list(_iter_linears(model)):
        if full == "head" and not quantize_lm_head:
            continue
        if full.startswith("embed") and not quantize_embeddings:
            continue
        setattr(parent, name, QuantLinear.from_linear(linear, scheme, int8_gemm))
        n += 1
    return n


def fold_quantization(model: nn.Module) -> dict[str, Tensor]:
    """Replace QuantLinear by plain Linear and return a *folded* state dict.

    After this call the model is a vanilla GDN-2 whose weights are exactly the
    8-bit values used during training — no scale tensors needed at inference.
    """
    for parent, name, full, qlinear in list(_iter_linears(model)):
        if not isinstance(qlinear, QuantLinear):
            continue
        lin = nn.Linear(qlinear.in_features, qlinear.out_features,
                        bias=qlinear.bias is not None)
        with torch.no_grad():
            lin.weight.copy_(qlinear.folded_weight())
            if qlinear.bias is not None:
                lin.bias.copy_(qlinear.bias.detach())
        setattr(parent, name, lin)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return sd


def has_quantized(model: nn.Module) -> bool:
    return any(isinstance(m, QuantLinear) for m in model.modules())


def make_optimizer(params: Iterable[Tensor], name: str, lr: float,
                   weight_decay: float = 0.1, betas=(0.9, 0.95)):
    params = list(params)
    if name == "adamw8bit":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=lr, betas=betas,
                                       weight_decay=weight_decay)
        except Exception as exc:  # pragma: no cover - depends on Colab image
            warnings.warn(
                f"bitsandbytes недоступен ({exc}); используем обычный AdamW. "
                "8-битное обучение весов при этом сохраняется."
            )
            name = "adamw"
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    fused = torch.cuda.is_available()
    try:
        return torch.optim.AdamW(params, lr=lr, betas=betas,
                                 weight_decay=weight_decay, fused=fused)
    except TypeError:
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
