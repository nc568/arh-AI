"""LiteRT (.tflite) export with per-NPU profiles.

Exported graph — a *stateful single-token decode step*::

    (tokens[1,1] int32, state[L,1,S] float32) -> (logits[1,1,V], state[L,1,S])

Fixed shapes, constant-size recurrent state (no growing KV cache) and only
MUL/SUM/ADD/EXP/LOGISTIC/GATHER/FULLY_CONNECTED ops: exactly what mobile NPUs
accelerate.  Optionally a fixed-length ``prefill`` signature is added.

Pipeline (verified against litert-torch 0.10 / ai-edge-quantizer 0.10):
  1. ``litert_torch.convert`` (torch.export -> StableHLO -> TFLite) for a
     float32 decode graph;
  2. post-training quantisation of the resulting flatbuffer with
     ``ai-edge-quantizer`` (dynamic int8 / fp16 weight cast / static int8);
  3. interpreter smoke test + op/NPU compatibility report.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

from .config import NPU_PROFILES, LabConfig


# --------------------------------------------------------------------------- #
# Export-friendly wrappers
# --------------------------------------------------------------------------- #


class DecodeWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor, state: torch.Tensor):
        logits, new_state = self.model.step(tokens, state)
        return logits.float(), new_state.float()


class PrefillWrapper(nn.Module):
    def __init__(self, model, length: int):
        super().__init__()
        self.model = model
        self.length = length

    def forward(self, tokens: torch.Tensor, state: torch.Tensor):
        outs = []
        for i in range(self.length):
            logits, state = self.model.step(tokens[:, i:i + 1], state)
            outs.append(logits)
        return torch.cat(outs, dim=1), state


# --------------------------------------------------------------------------- #
# Loading & converter
# --------------------------------------------------------------------------- #


def _resolve(cfg: LabConfig, ckpt_dir: Path):
    if (ckpt_dir / "config.json").exists():
        try:
            cfg = LabConfig.from_json(ckpt_dir / "config.json")
        except Exception:
            pass
    for name in ("model.safetensors", "sft_final.safetensors", "base_final.safetensors"):
        p = ckpt_dir / name
        if p.exists():
            return cfg, p, ckpt_dir
    raise FileNotFoundError(f"В {ckpt_dir} нет чекпоинта — сначала обучите модель.")


def load_export_model(cfg: LabConfig, ckpt_dir: str | Path | None = None):
    from .model import build_model
    ckpt_dir = Path(ckpt_dir or cfg.path(cfg.train.out_dir))
    cfg, ckpt, tok_dir = _resolve(cfg, ckpt_dir)
    model = build_model(cfg.model)
    sd = load_file(str(ckpt))
    if cfg.model.tie_embeddings and "head.weight" not in sd and "embed.weight" in sd:
        sd["head.weight"] = sd["embed.weight"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Чекпоинт не совпадает с конфигом: {missing=} {unexpected=}")
    return cfg, model.float().cpu().eval(), tok_dir


def _import_converter():
    try:
        import litert_torch as conv
        return conv, "litert-torch"
    except ImportError:
        pass
    try:
        import ai_edge_torch as conv
        return conv, "ai-edge-torch"
    except ImportError as exc:
        raise RuntimeError(
            "Нужен конвертер PyTorch -> LiteRT. В Colab:\n"
            "  !pip install -q 'litert-torch>=0.10' ai-edge-litert\n"
            "(litert-torch 0.7+ — новое имя ai-edge-torch; оба импорта поддерживаются.)"
        ) from exc


def _interpreter(path: str | Path):
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter(str(path))
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
        return Interpreter(str(path))


def _signature_io(path: str | Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        it = _interpreter(path)
        for name in it.get_signature_list():
            r = it.get_signature_runner(name)
            out[name] = {"inputs": list(r.get_input_details()),
                         "outputs": list(r.get_output_details())}
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- #
# Post-training quantisation of the flatbuffer
# --------------------------------------------------------------------------- #


def _real_states(model, state_shape: tuple[int, ...], vocab_size: int,
                 n: int = 10) -> list[np.ndarray]:
    """Collect representative recurrent states by actually running the model."""
    out = []
    with torch.no_grad():
        state = torch.zeros(state_shape, dtype=torch.float32)
        for i in range(n * 3 + 1):
            tok = torch.randint(0, vocab_size, (1, 1), dtype=torch.int32)
            _, state = model.step(tok, state)
            if i % 3 == 0:
                out.append(state.numpy().copy())
    return out


def _calibration_data(float_path: Path, states: list[np.ndarray],
                      vocab_size: int) -> dict[str, list[dict[str, np.ndarray]]]:
    it = _interpreter(float_path)
    rng = np.random.default_rng(0)
    calib: dict[str, list[dict[str, np.ndarray]]] = {}
    for sig in it.get_signature_list():
        det = it.get_signature_runner(sig).get_input_details()
        samples = []
        for s in states:
            sample: dict[str, np.ndarray] = {}
            for name, d in det.items():
                shape, dtype = tuple(int(x) for x in d["shape"]), np.dtype(d["dtype"])
                if np.issubdtype(dtype, np.integer):
                    sample[name] = rng.integers(0, vocab_size, shape, dtype=np.int32).astype(dtype)
                else:
                    # recurrent state: use a real one if shapes match
                    if s.shape == shape:
                        sample[name] = s.astype(dtype)
                    else:
                        sample[name] = np.zeros(shape, dtype=dtype)
            samples.append(sample)
        calib[sig] = samples
    return calib


def _quantize_flatbuffer(float_path: Path, out_path: Path, quant_mode: str,
                         state_shape: tuple[int, ...], vocab_size: int,
                         signature_io: dict[str, Any],
                         model=None) -> tuple[str, list[str]]:
    """Returns (actually used quant mode, warnings)."""
    warns: list[str] = []
    if quant_mode == "float32":
        shutil.copyfile(float_path, out_path)
        return "float32", warns
    try:
        from ai_edge_quantizer import Quantizer, recipe, recipe_manager
    except ImportError:
        shutil.copyfile(float_path, out_path)
        return "float32", ["ai-edge-quantizer не установлен: сохранена float32-модель. "
                           "pip install ai-edge-quantizer"]

    if quant_mode == "fp16":
        rp = recipe_manager.RecipeManager()
        rp.add_weight_only_config(
            regex=".*", operation_name=recipe.TFLOperationName.ALL_SUPPORTED,
            num_bits=16, algorithm_key=recipe.AlgorithmName.FLOAT_CASTING)
        rec = rp.get_quantization_recipe()
        q = Quantizer(str(float_path), rec)
    elif quant_mode == "full_int8":
        # Static int8 is what EdgeTPU needs.  Calibration can fail on some
        # runtimes (XNNPACK), so we always have a dynamic-int8 fallback.
        q = Quantizer(str(float_path), recipe.static_wi8_ai8())
        try:
            states = _real_states(model, state_shape, vocab_size) if model is not None else [
                (np.random.default_rng(0).standard_normal(state_shape) * 0.02).astype(np.float32)
                for _ in range(10)]
            calib = _calibration_data(float_path, states, vocab_size)
            cal = q.calibrate(calib)
            result = q.quantize(cal)
            result.export_model(str(out_path), overwrite=True)
            warns.append("Статическая int8-калибровка выполнена на реальных "
                         "рекуррентных состояниях модели. Для Edge TPU обязателен "
                         "edgetpu_compiler; неподдерживаемые ops уйдут на CPU.")
            return "full_int8", warns
        except Exception as exc:
            warns.append(
                f"Статическая int8-калибровка не удалась ({type(exc).__name__}: "
                f"{str(exc)[:150]}). Для Edge TPU всё равно нужен edgetpu_compiler; "
                "сохраняем dynamic int8 как рабочую альтернативу.")
            q = Quantizer(str(float_path), recipe.dynamic_wi8_afp32())
            result = q.quantize()
            result.export_model(str(out_path), overwrite=True)
            return "dynamic_int8", warns
    else:  # dynamic_int8
        q = Quantizer(str(float_path), recipe.dynamic_wi8_afp32())

    result = q.quantize()
    result.export_model(str(out_path), overwrite=True)
    return quant_mode, warns


def _verify(out_path: Path, ref_logits: np.ndarray, state_shape: tuple[int, ...],
            vocab_size: int) -> dict[str, Any]:
    smoke: dict[str, Any] = {"passed": False}
    try:
        it = _interpreter(out_path)
        it.allocate_tensors()
        sigs = list(it.get_signature_list())
        name = "decode" if "decode" in sigs else (sigs[0] if sigs else None)
        r = it.get_signature_runner(name) if name else None
        if r is None:
            ins = it.get_input_details()
            for d in ins:
                it.set_tensor(d["index"], np.zeros(d["shape"], dtype=d["dtype"]))
            it.invoke()
            det = it.get_output_details()[0]
            logits = it.get_tensor(det["index"])
            qdet = det
        else:
            inputs = {}
            for n, d in r.get_input_details().items():
                if np.dtype(d["dtype"]) == np.int32:
                    inputs[n] = np.zeros(d["shape"], dtype=np.int32)
                else:
                    inputs[n] = np.zeros(d["shape"], dtype=d["dtype"])
            out = r(**inputs)
            key = next((k for k, v in out.items()
                        if np.asarray(v).shape[-1:] == (vocab_size,)), list(out)[0])
            logits = np.asarray(out[key])
            qdet = r.get_output_details()[key]
        qp = qdet.get("quantization") or (0.0, 0)
        scale, zero = (qp if isinstance(qp, (tuple, list)) else (0.0, 0))[:2]
        if scale:
            logits = (logits.astype(np.float32) - int(zero)) * float(scale)
        smoke["max_abs_diff"] = float(np.abs(
            logits.reshape(-1)[:64].astype(np.float64) -
            np.asarray(ref_logits).reshape(-1)[:64].astype(np.float64)).max())
        smoke["passed"] = bool(np.isfinite(logits).all())
        smoke["signature"] = name
    except Exception as exc:
        smoke["error"] = str(exc)
    return smoke


def _ops_report(path: Path, profile: dict[str, Any]) -> dict[str, Any]:
    try:
        it = _interpreter(path)
        it.allocate_tensors()
        details = it._get_ops_details()
        ops = sorted({d.get("op_name", "?") for d in details})
        risky = [o for o in ops if any(o.upper() == r.upper()
                                      for r in profile.get("risky_ops", []))]
        ins = [{"name": d["name"], "dtype": str(d["dtype"]),
                "shape": [int(x) for x in d["shape"]],
                "quantization": str(d.get("quantization"))}
               for d in it.get_input_details()]
        outs = [{"name": d["name"], "dtype": str(d["dtype"]),
                 "shape": [int(x) for x in d["shape"]],
                 "quantization": str(d.get("quantization"))}
                for d in it.get_output_details()]
        return {"ops": ops, "risky_ops": risky, "num_ops": len(details),
                "inputs": ins, "outputs": outs}
    except Exception as exc:
        return {"ops": [], "error": str(exc)}


# --------------------------------------------------------------------------- #
# Main export
# --------------------------------------------------------------------------- #


def export_litert(cfg: LabConfig, ckpt_dir: str | Path | None = None,
                  out_path: str | Path | None = None,
                  profile_name: str | None = None,
                  quant_mode: str | None = None,
                  prefill_len: int | None = None,
                  export_prefill: bool | None = None) -> dict[str, Any]:
    converter, converter_name = _import_converter()
    cfg, model, tok_dir = load_export_model(cfg, ckpt_dir)
    profile_name = profile_name or cfg.export.litert_profile
    profile = NPU_PROFILES.get(profile_name)
    if profile is None:
        raise ValueError(f"Неизвестный NPU-профиль {profile_name}. Доступно: {list(NPU_PROFILES)}")
    if cfg.model.swa_layers:
        raise RuntimeError(
            "В модели есть SWA-слои (model.hybrid_every > 0). Для LiteRT decode-step "
            "нужна чисто рекуррентная модель: выключите гибридность и пересохраните.")

    quant_mode = quant_mode or cfg.export.litert_quant
    if quant_mode == "auto":
        quant_mode = profile["quant"]
    prefill_len = int(prefill_len or cfg.export.prefill_len or profile["prefill_len"])
    do_prefill = cfg.export.export_prefill if export_prefill is None else export_prefill

    L, S = cfg.model.n_layer, model.state_size()
    state_shape = (L, 1, S)
    decode = DecodeWrapper(model).eval()
    d_args = (torch.zeros((1, 1), dtype=torch.int32),
              torch.zeros(state_shape, dtype=torch.float32))
    warns: list[str] = []

    # 1) torch.export sanity check (catches unsupported Python before converting)
    with torch.no_grad():
        torch.export.export(decode, d_args)
        ref_logits, _ = decode(*d_args)
        ref_logits = ref_logits.numpy()

    # 2) convert float32 graph (optionally with a prefill signature)
    signatures: dict[str, Any] = {"decode": {"inputs": ["tokens", "state"],
                                              "outputs": ["logits", "state"]}}
    with torch.no_grad():
        if do_prefill and hasattr(converter, "signature"):
            try:
                prefill = PrefillWrapper(model, prefill_len).eval()
                p_args = (torch.zeros((1, prefill_len), dtype=torch.int32),
                          torch.zeros(state_shape, dtype=torch.float32))
                edge = (converter.signature("decode", decode, d_args)
                        .signature("prefill", prefill, p_args).convert())
                signatures["prefill"] = {"inputs": ["tokens", "state"],
                                         "outputs": ["logits", "state"],
                                         "length": prefill_len}
            except Exception as exc:  # pragma: no cover
                warns.append(f"Prefill-подпись не добавлена ({type(exc).__name__}): {exc}")
                edge = converter.convert(decode, d_args)
        else:
            edge = converter.convert(decode, d_args)

    out_dir = cfg.path(cfg.export.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = (Path(out_path) if out_path else
           out_dir / f"{cfg.run_name}-{profile_name}-{quant_mode}.tflite")
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="gdn2float-") as td:
        float_path = Path(td) / "float.tflite"
        edge.export(str(float_path))
        io = _signature_io(float_path)
        used_quant, qwarns = _quantize_flatbuffer(
            float_path, out, quant_mode, state_shape, cfg.model.vocab_size, io,
            model=model)
        warns.extend(qwarns)

    smoke = _verify(out, ref_logits, state_shape, cfg.model.vocab_size)
    ops = _ops_report(out, profile)
    final_io = _signature_io(out)

    tok_info = {}
    if (tok_dir / "tokenizer_info.json").exists():
        tok_info = json.loads((tok_dir / "tokenizer_info.json").read_text(encoding="utf-8"))
    signature = {
        "tflite": str(out), "converter": converter_name,
        "profile": profile_name, "quant": used_quant,
        "requested_quant": quant_mode,
        "state_shape": list(state_shape), "state_dtype": "float32",
        "vocab_size": cfg.model.vocab_size,
        "bos_id": tok_info.get("bos_id", 1), "eos_id": tok_info.get("eos_id", 2),
        "pad_id": tok_info.get("pad_id", 3),
        "signatures": {k: {kk: vv for kk, vv in v.items() if kk != "length"}
                       for k, v in signatures.items()},
        "io": final_io,
        "prefill_len": prefill_len if "prefill" in signatures else 0,
        "tokenizer_dir": str(tok_dir),
        "smoke_test": smoke,
    }
    sig_path = out.with_suffix(".signature.json")
    sig_path.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "path": str(out), "signature_file": str(sig_path),
        "size_mb": round(out.stat().st_size / 1e6, 2),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest()[:16],
        "profile": profile_name, "profile_label": profile["label"],
        "quant": used_quant, "requested_quant": quant_mode,
        "static_shapes": profile["static_shapes"],
        "state_shape": list(state_shape), "state_kb": round(S * 4 / 1024, 1),
        "signatures": list(signatures), "smoke_test": smoke, "ops": ops,
        "delegate_snippet": profile["delegate"],
        "notes": profile["notes"] + warns,
        "deployment": (
            f"python -m gdn2lab.cli chat --from-litert {out}  # CPU smoke test\n"
            "См. litert_runtime.py: передайте delegate вашего NPU в LiteRTGenerator."
        ),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for f in ("tokenizer.json", "tokenizer_info.json", "hf_name.txt"):
        if (tok_dir / f).exists():
            shutil.copyfile(tok_dir / f, out.parent / f)
    report["signature"] = signature
    return report
