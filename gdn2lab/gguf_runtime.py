"""Load a GDN-2 GGUF file back into PyTorch and generate text.

This is the "it actually runs" half of the GGUF export: stock llama.cpp does
not implement Gated Delta Rule-2 yet, so we ship a compact reader that uses
``gguf`` for dequantisation and the same pure-PyTorch model for inference.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ModelConfig
from .data import LabTokenizer, SPECIAL_TOKENS, build_byte_tokenizer
from .model import build_model


def _field(reader, key: str, default=None):
    try:
        f = reader.get_field(key)
        if f is None:
            return default
        return f.contents()
    except Exception:
        return default


def _tokenizer_from_gguf(reader) -> LabTokenizer:
    info = _field(reader, "gdn2.tokenizer_info_json", "{}")
    info = json.loads(info) if isinstance(info, str) else {}
    tok_json = _field(reader, "gdn2.tokenizer_json", "")
    kind = info.get("kind", "byte")
    if kind == "byte" or not tok_json:
        return build_byte_tokenizer()
    try:
        maybe_meta = json.loads(tok_json)
        if isinstance(maybe_meta, dict) and "kind" in maybe_meta and "model" not in maybe_meta:
            # это метаданные, а не tokenizer.json (например HF AutoTokenizer)
            if kind == "hf" and maybe_meta.get("name"):
                from .data import build_hf_tokenizer
                return build_hf_tokenizer(maybe_meta["name"])
            return build_byte_tokenizer()
    except (ValueError, TypeError):
        pass
    tmp = Path(tempfile.mkdtemp(prefix="gdn2tok-"))
    (tmp / "tokenizer.json").write_text(tok_json, encoding="utf-8")
    from tokenizers import Tokenizer
    impl = Tokenizer.from_file(str(tmp / "tokenizer.json"))
    return LabTokenizer(impl, "scratch" if kind == "scratch" else kind,
                        int(info.get("vocab_size", impl.get_vocab_size())),
                        info.get("tokens", []), name=info.get("name"))


def load_gguf(path: str | Path, device: str | torch.device = "cpu"):
    try:
        import gguf
        from gguf import GGUFReader
    except ImportError as exc:
        raise RuntimeError("Нужен пакет gguf: pip install gguf") from exc
    path = Path(path)
    reader = GGUFReader(str(path))
    arch = _field(reader, "general.architecture", "")
    if str(arch) != "gdn2":
        raise ValueError(f"Ожидалась архитектура gdn2, получено {arch!r}")
    cfg_json = _field(reader, "gdn2.config_json", "{}")
    mcfg = ModelConfig(**json.loads(cfg_json))
    model = build_model(mcfg)
    sd = {}
    for t in reader.tensors:
        arr = gguf.dequantize(t.data, t.tensor_type)
        sd[t.name] = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    if mcfg.tie_embeddings and "head.weight" not in sd and "embed.weight" in sd:
        sd["head.weight"] = sd["embed.weight"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"GGUF: несовпадение тензоров: missing={missing}, unexpected={unexpected}")
    model = model.to(torch.device(device)).eval()
    tok = _tokenizer_from_gguf(reader)
    return model, tok


def generate_text(model, tok: LabTokenizer, prompt: str,
                  max_new_tokens: int = 100, temperature: float = 0.8,
                  top_k: int = 40, top_p: float = 0.95) -> str:
    ids = [tok.bos_id] + tok.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=max_new_tokens,
                             temperature=temperature, top_k=top_k, top_p=top_p,
                             eos_id=tok.eos_id)
    return tok.decode(out[0].tolist())
