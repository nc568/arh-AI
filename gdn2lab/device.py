"""One small abstraction over CUDA / TPU (torch-xla) / CPU.

Colab free tier: T4 (sm75, INT8 + FP16 Tensor Cores, no FP8/BF16 compute) or
TPU v2/v3 (bf16).  The notebook and CLI only ever talk to this module.
"""
from __future__ import annotations

import contextlib
import os
import warnings
from typing import Any

import torch


def detect_runtime() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch": torch.__version__,
        "kind": "cpu",
        "name": "CPU",
        "mem_gb": 0.0,
        "tpu_cores": 0,
        "notes": [],
    }
    if torch.cuda.is_available():
        info["kind"] = "cuda"
        info["name"] = torch.cuda.get_device_name(0)
        info["mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        cap = torch.cuda.get_device_capability(0)
        info["compute_capability"] = f"{cap[0]}.{cap[1]}"
        if cap[0] < 8:
            info["notes"].append("GPU старше Ampere: нет BF16/FP8 Tensor Cores — используем FP16 + INT8.")
    else:
        try:
            os.environ.setdefault("PJRT_DEVICE", "TPU")
            import torch_xla
            import torch_xla.core.xla_model as xm
            info["kind"] = "tpu"
            try:
                info["name"] = str(xm.xla_device_hw(xm.xla_device()))
            except Exception:
                info["name"] = "TPU"
            info["tpu_cores"] = len(xm.get_xla_supported_devices())
            info["mem_gb"] = 8.0 * max(1, info["tpu_cores"])
            info["notes"].append("TPU: bf16, фиксированные формы, xm.optimizer_step().")
        except Exception:
            info["notes"].append("CPU: обучение возможно, но медленное. Проверьте Среда выполнения -> T4/TPU.")
    return info


def setup_device(prefer: str = "auto", use_all_tpu_cores: bool = False) -> tuple[torch.device, dict[str, Any]]:
    info = detect_runtime()
    kind = info["kind"] if prefer == "auto" else prefer
    if kind == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA запрошена, но недоступна — используем CPU.")
        kind = "cpu"
    if kind == "tpu":
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        try:
            import torch_xla.core.xla_model as xm
            dev = xm.xla_device()
            info["xla"] = xm
            return dev, info
        except Exception as exc:
            raise RuntimeError(
                "torch_xla недоступен. В Colab выберите «Среда выполнения -> Сменить тип -> TPU»."
            ) from exc
    if kind == "cuda":
        return torch.device("cuda", 0), info
    return torch.device("cpu"), info


def resolve_precision(precision: str, device: torch.device) -> str:
    """Return the compute dtype actually used, with honest fallbacks."""
    if precision in ("int8", "fp8"):
        return "fp16" if device.type == "cuda" else ("bf16" if device.type == "xla" else "fp32")
    if precision == "bf16" and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            warnings.warn("На этом GPU нет BF16 — переключаемся на FP16.")
            return "fp16"
    if precision == "fp16" and device.type == "xla":
        warnings.warn("На TPU используем BF16 вместо FP16.")
        return "bf16"
    if device.type == "cpu" and precision in ("fp16", "bf16"):
        return "fp32"
    return precision


def autocast_ctx(device: torch.device, compute_dtype: str):
    if device.type == "cpu" or compute_dtype == "fp32":
        return contextlib.nullcontext()
    dtype = torch.float16 if compute_dtype == "fp16" else torch.bfloat16
    try:
        return torch.autocast(device_type=device.type, dtype=dtype)
    except (RuntimeError, ValueError, TypeError):
        return contextlib.nullcontext()


def make_scaler(compute_dtype: str, device: torch.device):
    enabled = compute_dtype == "fp16" and device.type == "cuda"
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError, ValueError):
        try:
            return torch.cuda.amp.GradScaler(enabled=enabled)
        except Exception:
            class _NoScaler:
                enabled = False
                def scale(self, x): return x
                def step(self, opt): opt.step()
                def update(self): pass
                def unscale_(self, opt): pass
            return _NoScaler()


def optimizer_step(optimizer, scaler, model, grad_clip: float,
                   device: torch.device, info: dict[str, Any] | None = None):
    if scaler is not None and getattr(scaler, "enabled", False):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if device.type == "xla":
            xm = (info or {}).get("xla")
            if xm is None:
                import torch_xla.core.xla_model as xm
            xm.optimizer_step(optimizer, barrier=True)
        else:
            optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def synchronize(device: torch.device, info: dict[str, Any] | None = None):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xla":
        xm = (info or {}).get("xla")
        if xm is None:
            try:
                import torch_xla.core.xla_model as xm
            except ImportError:
                return
        xm.mark_step()


def empty_cache(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def auto_micro_batch(model, cfg, device, info, batch_fn, start: int = 1, cap: int = 64) -> int:
    """Probe the largest micro-batch that fits; used when cfg.micro_batch_size=0."""
    best = 1
    b = start
    while b <= cap:
        try:
            x, y = batch_fn(b)
            with autocast_ctx(device, resolve_precision(cfg.train.precision, device)):
                logits, loss = model(x, y)
            (loss if loss is not None else logits.mean()).backward()
            model.zero_grad(set_to_none=True)
            synchronize(device, info)
            best = b
            lo = b
            b *= 2
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() or "resource exhausted" in str(exc).lower():
                empty_cache(device)
                break
            raise
        finally:
            empty_cache(device)
    return max(1, best)
