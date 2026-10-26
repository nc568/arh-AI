"""Run an exported GDN-2 .tflite decode-step model (CPU smoke test or NPU).

On a device, pass a delegate from your vendor SDK; see the ``delegate_snippet``
in ``*.report.json`` produced by :mod:`gdn2lab.export_litert`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _interpreter():
    try:
        from ai_edge_litert.interpreter import Interpreter, load_delegate
        return Interpreter, load_delegate
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter, load_delegate
            return Interpreter, load_delegate
        except ImportError as exc:
            raise RuntimeError(
                "Нужен LiteRT runtime: pip install ai-edge-litert "
                "(или tensorflow для CPU-проверки)."
            ) from exc


def _quantize(arr: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    q = detail.get("quantization") or (0.0, 0)
    scale, zero = (q if isinstance(q, (tuple, list)) else (0.0, 0))[:2]
    if not scale:
        return arr.astype(detail["dtype"])
    out = np.round(arr / float(scale)) + int(zero)
    info = np.iinfo(detail["dtype"])
    return np.clip(out, info.min, info.max).astype(detail["dtype"])


def _dequantize(arr: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    q = detail.get("quantization") or (0.0, 0)
    scale, zero = (q if isinstance(q, (tuple, list)) else (0.0, 0))[:2]
    if not scale:
        return arr.astype(np.float32)
    return (arr.astype(np.float32) - int(zero)) * float(scale)


def _sample(logits: np.ndarray, temperature: float, top_k: int, top_p: float,
            rng: np.random.Generator) -> int:
    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    if temperature <= 0:
        return int(np.argmax(x))
    x = x / temperature
    if top_k > 0 and top_k < x.size:
        idx = np.argpartition(x, -top_k)[-top_k:]
        mask = np.full(x.size, -np.inf)
        mask[idx] = x[idx]
        x = mask
    if 0 < top_p < 1.0:
        order = np.argsort(-x)
        probs = np.exp(x[order] - x[order].max())
        probs /= probs.sum()
        cum = np.cumsum(probs)
        keep = order[:max(1, int(np.searchsorted(cum, top_p)) + 1)]
        mask = np.full(x.size, -np.inf)
        mask[keep] = x[keep]
        x = mask
    p = np.exp(x - x.max())
    p /= p.sum()
    return int(rng.choice(x.size, p=p))


class LiteRTGenerator:
    def __init__(self, tflite_path: str | Path, signature_path: str | Path,
                 tokenizer, delegates: list | None = None):
        Interpreter, _ = _interpreter()
        self.sig = json.loads(Path(signature_path).read_text(encoding="utf-8"))
        self.tok = tokenizer
        self.interp = Interpreter(str(tflite_path),
                                  experimental_delegates=delegates or [])
        self.interp.allocate_tensors()
        self.state_shape = tuple(self.sig["state_shape"])
        self.vocab = int(self.sig["vocab_size"])
        self.runner = None
        try:
            self.runner = self.interp.get_signature_runner("decode")
            self.in_details = self.runner.get_input_details()
            self.out_details = self.runner.get_output_details()
        except (ValueError, RuntimeError, AttributeError):
            self.runner = None
            self.in_details = {d["name"]: d for d in self.interp.get_input_details()}
            self.out_details = {d["name"]: d for d in self.interp.get_output_details()}
        self.in_names = list(self.in_details)
        self.out_names = list(self.out_details)
        io = self.sig.get("io", {}).get("decode", {})
        self.io_in = io.get("inputs") or self.in_names
        self.io_out = io.get("outputs") or self.out_names

    def _find_index(self, names, detail, want: str) -> str:
        if want == "logits" and len(self.io_out) >= 1:
            return self.io_out[0]
        if want == "state" and len(self.io_out) >= 2:
            return self.io_out[-1]
        if want == "logits":
            # logits output: last dim == vocab
            for n in names:
                if tuple(self.out_details[n].get("shape", []))[-1:] == (self.vocab,):
                    return n
        else:
            for n in names:
                if tuple(self.out_details[n].get("shape", [])) == self.state_shape:
                    return n
        return names[0] if want == "logits" else names[-1]

    def step(self, token_id: int, state: np.ndarray):
        token_name = self._input_name_for("tokens")
        state_name = self._input_name_for("state")
        tok_detail = self.in_details[token_name]
        st_detail = self.in_details[state_name]
        tok_arr = np.array([[token_id]], dtype=np.int32)
        if np.dtype(tok_detail["dtype"]) != np.int32:
            tok_arr = tok_arr.astype(tok_detail["dtype"])
        st_arr = _quantize(state.astype(np.float32), st_detail)
        inputs = {token_name: tok_arr, state_name: st_arr}
        if self.runner is not None:
            out = self.runner(**inputs)
        else:
            for name, value in inputs.items():
                idx = self.in_details[name]["index"]
                self.interp.set_tensor(idx, value)
            self.interp.invoke()
            out = {d["name"]: self.interp.get_tensor(d["index"])
                   for d in self.interp.get_output_details()}
        names = list(out)
        logits_name = self._find_index(names, None, "logits")
        state_name_out = self._find_index(names, None, "state")
        logits = _dequantize(np.asarray(out[logits_name]), self.out_details[logits_name])
        new_state = _dequantize(np.asarray(out[state_name_out]), self.out_details[state_name_out])
        return logits, new_state.astype(np.float32)

    def _input_name_for(self, want: str) -> str:
        if len(self.io_in) >= 2 and all(n in self.in_details for n in self.io_in):
            return self.io_in[0] if want == "tokens" else self.io_in[1]
        for n in self.in_names:
            low = n.lower()
            if want == "tokens" and any(k in low for k in ("token", "input_0", "args_0", "input_ids")):
                return n
            if want == "state" and any(k in low for k in ("state", "input_1", "args_1")):
                return n
        # fall back by shape
        for n in self.in_names:
            sh = tuple(self.in_details[n].get("shape", []))
            if want == "tokens" and sh[-1:] == (1,) and np.prod(sh) == 1:
                return n
            if want == "state" and sh == self.state_shape:
                return n
        return self.in_names[0] if want == "tokens" else self.in_names[-1]

    def generate(self, prompt: str, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 40, top_p: float = 0.95,
                 seed: int = 0) -> str:
        rng = np.random.default_rng(seed)
        state = np.zeros(self.state_shape, dtype=np.float32)
        ids = [int(getattr(self.tok, "bos_id", 1))] + self.tok.encode(prompt)
        out_ids = list(ids)
        logits = None
        for t in ids:
            logits, state = self.step(int(t), state)
        for _ in range(max_new_tokens):
            nxt = _sample(logits, temperature, top_k, top_p, rng)
            out_ids.append(nxt)
            if nxt == int(self.sig.get("eos_id", 2)):
                break
            logits, state = self.step(nxt, state)
        return self.tok.decode(out_ids)
