"""GGUF export (v3 container) + a portable PyTorch runtime for it.

Honest caveat: llama.cpp today knows Gated DeltaNet / Qwen3-Next but *not* the
new GDN-2 update rule with separate erase/write gates (arXiv:2605.22791).  So
this writer produces a **valid GGUF file with a custom ``gdn2`` architecture**
and this repository ships a small runtime that loads and runs it.  The metadata
contains a complete tensor map for anyone adding a llama.cpp backend.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from .config import GGUF_QUANTS, LabConfig, ModelConfig

QUANT_ALIASES = {
    "Q4_K_M": "Q4_0", "Q4_K_S": "Q4_0", "Q5_K_M": "Q5_0",
    "Q5_K_S": "Q5_0", "Q6_K": "Q5_1", "Q8_K": "Q8_0",
    "IQ4_NL": "Q5_0", "Q2_K": "Q4_0",
}
FILE_TYPES = {"F32": 0, "F16": 1, "Q4_0": 2, "Q4_1": 3,
              "Q5_0": 6, "Q5_1": 9, "Q8_0": 7, "BF16": 30}


def _gguf_imports():
    try:
        import gguf
        from gguf import GGUFWriter, GGMLQuantizationType
        from gguf.constants import GGML_QUANT_SIZES
        return gguf, GGUFWriter, GGMLQuantizationType, GGML_QUANT_SIZES
    except ImportError as exc:
        raise RuntimeError("Нужен пакет gguf: pip install gguf") from exc


def _resolve_checkpoint(cfg: LabConfig, ckpt_dir: Path) -> tuple[LabConfig, Path, Path]:
    if (ckpt_dir / "config.json").exists():
        try:
            cfg = LabConfig.from_json(ckpt_dir / "config.json")
        except Exception:
            pass
    for name in ("model.safetensors", "sft_final.safetensors", "base_final.safetensors"):
        p = ckpt_dir / name
        if p.exists():
            return cfg, p, ckpt_dir
    raise FileNotFoundError(f"В {ckpt_dir} нет model.safetensors — сначала обучите модель.")


def export_gguf(cfg: LabConfig, ckpt_dir: str | Path | None = None,
                out_path: str | Path | None = None,
                quant: str | None = None) -> dict[str, Any]:
    gguf, GGUFWriter, Q, SIZES = _gguf_imports()
    ckpt_dir = Path(ckpt_dir or cfg.path(cfg.train.out_dir))
    cfg, ckpt, tok_dir = _resolve_checkpoint(cfg, ckpt_dir)
    quant = (quant or cfg.export.gguf_quant or "F16").upper()
    alias_note = None
    if quant in QUANT_ALIASES:
        alias_note = f"{quant} недоступен в python-gguf → используем {QUANT_ALIASES[quant]}"
        quant = QUANT_ALIASES[quant]
    if quant not in Q.__members__:
        raise ValueError(f"Неизвестный тип квантования {quant}. Доступно: {list(GGUF_QUANTS)}")
    qtype = Q[quant]

    state = load_file(str(ckpt))
    if cfg.model.tie_embeddings and "head.weight" not in state:
        state["head.weight"] = state["embed.weight"]

    out_dir = cfg.path(cfg.export.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = Path(out_path) if out_path else out_dir / f"{cfg.run_name}-{quant.lower()}.gguf"
    out.parent.mkdir(parents=True, exist_ok=True)

    tok_info = {}
    tok_json = ""
    if (tok_dir / "tokenizer_info.json").exists():
        tok_info = json.loads((tok_dir / "tokenizer_info.json").read_text(encoding="utf-8"))
    tok_json = ""
    if (tok_dir / "tokenizer.json").exists():
        tok_json = (tok_dir / "tokenizer.json").read_text(encoding="utf-8")
    elif tok_info:
        tok_json = json.dumps(tok_info, ensure_ascii=False)
    tokens = [t if isinstance(t, str) else "" for t in (tok_info.get("tokens") or [])]

    w = GGUFWriter(str(out), arch="gdn2")
    w.add_name(cfg.run_name)
    w.add_description("Gated DeltaNet-2 (arXiv:2605.22791) trained with gdn2lab")
    w.add_string("general.license", "MIT")
    w.add_string("general.source_url", "https://github.com/nc568/arh-AI")
    w.add_uint32("general.file_type", FILE_TYPES.get(quant, 0))
    w.add_uint32("gdn2.version", 1)
    w.add_uint32("gdn2.block_count", cfg.model.n_layer)
    w.add_uint32("gdn2.head_count", cfg.model.n_head)
    w.add_uint32("gdn2.head_dim", cfg.model.head_dim)
    w.add_uint32("gdn2.chunk_size", cfg.model.chunk_size)
    w.add_uint32("gdn2.conv_size", cfg.model.conv_size if cfg.model.use_short_conv else 0)
    w.add_uint32("gdn2.hybrid_every", cfg.model.hybrid_every)
    w.add_context_length(cfg.model.seq_len)
    w.add_embedding_length(cfg.model.d_model)
    w.add_uint32("gdn2.vocab_size", cfg.model.vocab_size)
    w.add_string("gdn2.config_json", json.dumps(asdict(cfg.model)))
    w.add_string("gdn2.tokenizer_info_json", json.dumps(tok_info, ensure_ascii=False))
    w.add_string("gdn2.tokenizer_json", tok_json)
    w.add_string("gdn2.precision", cfg.train.precision)
    w.add_string("gdn2.update_rule",
                 "S_t = (I - k_t (b_t*k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t*v_t)^T")
    w.add_array("gdn2.tensor_names", list(state.keys()))
    if tokens:
        w.add_tokenizer_model("gpt2")
        w.add_token_list(tokens)
        w.add_uint32("tokenizer.ggml.bos_token_id", int(tok_info.get("bos_id", 1)))
        w.add_uint32("tokenizer.ggml.eos_token_id", int(tok_info.get("eos_id", 2)))

    notes: list[str] = []
    if alias_note:
        notes.append(alias_note)
    block_size, _ = SIZES[qtype]
    n_quantized = 0
    for name, tensor in state.items():
        arr = tensor.detach().cpu().float().numpy()
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        use = qtype
        if quant not in ("F32", "F16", "BF16") and (arr.shape[-1] % block_size):
            use = Q.F16
            notes.append(f"{name}: размер {arr.shape[-1]} не кратен {block_size} → F16")
        if use in (Q.F32, Q.F16):
            data = arr.astype(np.float32 if use == Q.F32 else np.float16)
            w.add_tensor(name, data)
        else:
            q = gguf.quantize(arr, use)
            w.add_tensor(name, q, raw_shape=list(q.shape), raw_dtype=use)
            n_quantized += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    # verify by reading back
    from gguf import GGUFReader
    reader = GGUFReader(str(out))
    if len(reader.tensors) != len(state):
        raise RuntimeError("GGUF: количество тензоров не совпадает — файл повреждён.")
    max_err = 0.0
    for t in list(reader.tensors)[:3]:
        d = gguf.dequantize(t.data, t.tensor_type)
        ref = state[t.name].float().numpy()
        if d.shape == ref.shape:
            max_err = max(max_err, float(np.abs(d.astype(np.float64) - ref).max()))
    report = {
        "path": str(out),
        "size_mb": round(out.stat().st_size / 1e6, 2),
        "quant": quant,
        "tensors": len(state),
        "quantized_tensors": n_quantized,
        "params": int(sum(v.numel() for v in state.values())),
        "verify_max_abs_error": round(max_err, 5),
        "architecture": "gdn2",
        "notes": notes,
        "runtime": "python -m gdn2lab.cli chat --from-gguf <file.gguf>",
        "llama_cpp": (
            "Кастомная архитектура GDN-2: stock llama.cpp её не знает. "
            "Файл корректен как GGUF v3; для llama.cpp нужен backend под "
            "Gated Delta Rule-2 (см. gdn2.tensor_names + gdn2.update_rule). "
            "Пока используйте встроенный runtime из этого репозитория."
        ),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    report_path = out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for f in ("tokenizer.json", "tokenizer_info.json", "hf_name.txt"):
        if (tok_dir / f).exists():
            shutil.copyfile(tok_dir / f, out.parent / f)
    return report
