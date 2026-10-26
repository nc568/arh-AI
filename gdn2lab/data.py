"""Datasets and tokenizers: upload / local / Google Drive / HuggingFace / URL.

Base corpus -> packed token stream (.bin).  SFT corpus -> (input_ids, labels)
arrays with -100 masking of the prompt.  Nothing here needs a GPU.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import LabConfig

SPECIAL_TOKENS = ["<unk>", "<bos>", "<eos>", "<pad>", "<|im_start|>", "<|im_end|>"]
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #


class LabTokenizer:
    def __init__(self, impl, kind: str, vocab_size: int, tokens: list[str] | None = None,
                 name: str | None = None):
        self.impl = impl
        self.kind = kind
        self.vocab_size = vocab_size
        self.tokens = tokens or []
        self.name = name

    # -- ids ---------------------------------------------------------------- #
    def _tid(self, token: str, default: int = 0) -> int:
        if self.kind == "byte":
            return SPECIAL_TOKENS.index(token)
        try:
            i = self.impl.token_to_id(token)
            return int(i) if i is not None else default
        except Exception:
            return default

    @property
    def unk_id(self) -> int:
        return self._tid("<unk>", 0)

    @property
    def bos_id(self) -> int:
        return self._tid("<bos>", 1)

    @property
    def eos_id(self) -> int:
        return self._tid("<eos>", 2)

    @property
    def pad_id(self) -> int:
        return self._tid("<pad>", 3)

    @property
    def im_start_id(self) -> int:
        return self._tid("<|im_start|>", 4)

    @property
    def im_end_id(self) -> int:
        return self._tid("<|im_end|>", 5)

    # -- encode / decode ---------------------------------------------------- #
    def encode(self, text: str) -> list[int]:
        if self.kind == "byte":
            return [b + len(SPECIAL_TOKENS) for b in text.encode("utf-8")]
        enc = self.impl.encode(text, add_special_tokens=False)
        ids = enc.ids if hasattr(enc, "ids") else enc["input_ids"]
        return [int(i) for i in ids]

    def decode(self, ids: Iterable[int]) -> str:
        ids = [int(i) for i in ids if 0 <= int(i) < self.vocab_size]
        if self.kind == "byte":
            off = len(SPECIAL_TOKENS)
            raw = bytes(max(0, i - off) for i in ids if i >= off)
            return raw.decode("utf-8", errors="replace")
        return self.impl.decode(ids, skip_special_tokens=False)

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        info = {
            "kind": self.kind,
            "vocab_size": self.vocab_size,
            "tokens": self.tokens,
            "special_tokens": SPECIAL_TOKENS,
            "name": self.name,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
        }
        (out / "tokenizer_info.json").write_text(
            json.dumps(info, ensure_ascii=False), encoding="utf-8")
        if self.kind == "scratch":
            self.impl.save(str(out / "tokenizer.json"))
        elif self.kind == "hf" and hasattr(self.impl, "save"):
            self.impl.save(str(out / "tokenizer.json"))
        if self.kind == "hf" and self.name:
            (out / "hf_name.txt").write_text(self.name, encoding="utf-8")

    @classmethod
    def load(cls, out_dir: str | Path) -> "LabTokenizer":
        out = Path(out_dir)
        info = json.loads((out / "tokenizer_info.json").read_text(encoding="utf-8"))
        kind = info["kind"]
        if kind == "byte":
            return cls(None, kind, info["vocab_size"], info.get("tokens", []))
        if kind == "hf":
            name = info.get("name") or (out / "hf_name.txt").read_text(encoding="utf-8").strip()
            return build_hf_tokenizer(name)
        from tokenizers import Tokenizer
        return cls(Tokenizer.from_file(str(out / "tokenizer.json")), kind,
                   info["vocab_size"], info.get("tokens", []))
    
    def json_text(self) -> str:
        """Full tokenizer.json as a string (stored inside GGUF)."""
        if self.kind == "byte":
            return json.dumps({"kind": "byte", "vocab_size": self.vocab_size})
        if not hasattr(self.impl, "save"):
            return json.dumps({"kind": self.kind, "name": self.name,
                               "vocab_size": self.vocab_size})
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            p = f.name
        self.impl.save(p)
        text = Path(p).read_text(encoding="utf-8")
        os.unlink(p)
        return text


def train_bpe(texts: Iterable[str], vocab_size: int) -> "LabTokenizer":
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    tokens = [tok.id_to_token(i) for i in range(tok.get_vocab_size())]
    return LabTokenizer(tok, "scratch", tok.get_vocab_size(), tokens)


def build_hf_tokenizer(name: str) -> "LabTokenizer":
    from tokenizers import Tokenizer
    try:
        tok = Tokenizer.from_pretrained(name)
    except Exception:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
    size = int(max(tok.get_vocab_size() if hasattr(tok, "get_vocab_size") else tok.vocab_size,
                   getattr(tok, "vocab_size", 0)))
    try:
        tokens = [tok.id_to_token(i) if hasattr(tok, "id_to_token")
                  else tok.convert_ids_to_tokens(i) for i in range(size)]
    except Exception:
        tokens = []
    return LabTokenizer(tok, "hf", size, tokens, name=name)


def build_byte_tokenizer() -> "LabTokenizer":
    n = len(SPECIAL_TOKENS) + 256
    tokens = SPECIAL_TOKENS + [f"<byte_{i}>" for i in range(256)]
    return LabTokenizer(None, "byte", n, tokens)


def build_tokenizer(cfg: LabConfig, texts: Iterable[str]) -> "LabTokenizer":
    spec = cfg.data.tokenizer
    kind = spec.get("kind", "scratch")
    if kind == "byte":
        tok = build_byte_tokenizer()
    elif kind == "hf":
        tok = build_hf_tokenizer(spec.get("name", "gpt2"))
    else:
        tok = train_bpe(texts, int(spec.get("vocab_size", cfg.model.vocab_size)))
    # keep the model in sync with the actual tokenizer vocabulary
    cfg.model.vocab_size = int(tok.vocab_size)
    return tok


# --------------------------------------------------------------------------- #
# Source resolution and text iteration
# --------------------------------------------------------------------------- #


def resolve_source(spec: dict[str, Any], out_dir: Path, tag: str) -> list[Path]:
    kind = spec.get("kind", "demo")
    dst = out_dir / "raw" / tag
    dst.mkdir(parents=True, exist_ok=True)

    if kind == "demo":
        name = "toy_sft.jsonl" if tag == "sft" else "toy_base.txt"
        target = dst / name
        shutil.copyfile(EXAMPLES_DIR / name, target)
        return [target]

    if kind == "path":
        pattern = str(spec.get("path", ""))
        paths = [Path(p) for p in glob.glob(pattern, recursive=True)] if any(
            ch in pattern for ch in "*?[") else [Path(pattern)]
        root = Path(pattern)
        if root.is_dir():
            paths = [p for p in root.rglob("*") if p.is_file()]
        found = [p for p in paths if p.exists() and p.is_file()]
        if not found:
            raise FileNotFoundError(f"Не найдено файлов по пути: {pattern}")
        return found

    if kind == "upload":
        found = [p for p in dst.iterdir() if p.is_file() and not p.name.startswith(".")]
        if not found:
            raise FileNotFoundError(
                f"Папка {dst} пуста. Загрузите файлы кнопкой «Загрузить данные» "
                "или выберите другой источник.")
        return found

    if kind == "paste":
        text = spec.get("text", "")
        if not text.strip():
            raise ValueError("Поле с текстом пустое.")
        target = dst / "pasted.txt"
        target.write_text(text, encoding="utf-8")
        return [target]

    if kind == "url":
        url = spec.get("url", "")
        target = dst / (spec.get("filename") or Path(url.split("?")[0]).name or "download.txt")
        if not target.exists():
            urllib.request.urlretrieve(url, target)
        return [target]

    if kind == "hf":
        from datasets import load_dataset
        ds = load_dataset(spec.get("id", ""), spec.get("subset"),
                          split=spec.get("split", "train"), streaming=True)
        target = dst / "hf.jsonl"
        limit = int(spec.get("limit", 20000))
        with target.open("w", encoding="utf-8") as f:
            for i, row in enumerate(ds):
                if i >= limit:
                    break
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return [target]

    raise ValueError(f"Неизвестный источник данных: {kind}")


def iter_texts(paths: list[Path], field: str = "text") -> Iterator[str]:
    for p in paths:
        suf = p.suffix.lower()
        if suf in (".txt", ".md", ".text"):
            text = p.read_text(encoding="utf-8", errors="replace")
            for block in text.split("\n\n"):
                if block.strip():
                    yield block.strip()
        elif suf in (".jsonl", ".ndjson"):
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    yield _obj_to_text(obj, field)
        elif suf == ".json":
            obj = json.loads(p.read_text(encoding="utf-8"))
            rows = obj if isinstance(obj, list) else obj.get("data", [obj])
            for row in rows:
                yield _obj_to_text(row, field)
        elif suf == ".csv":
            with p.open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    yield _obj_to_text(row, field)
        elif suf == ".parquet":
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(p)
                for row in table.to_pylist():
                    yield _obj_to_text(row, field)
            except ImportError as exc:
                raise RuntimeError("Для .parquet нужен pyarrow: pip install pyarrow") from exc
        else:
            yield p.read_text(encoding="utf-8", errors="replace")


def _obj_to_text(obj: Any, field: str) -> str:
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return str(obj)
    if field in obj:
        return str(obj[field])
    if "messages" in obj:
        return "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in obj["messages"])
    parts = []
    for key in ("instruction", "input", "output", "prompt", "completion", "content", "title", "text"):
        if obj.get(key):
            parts.append(str(obj[key]))
    return "\n\n".join(parts) if parts else json.dumps(obj, ensure_ascii=False)


def iter_records(paths: list[Path]) -> Iterator[dict[str, Any]]:
    for p in paths:
        suf = p.suffix.lower()
        if suf in (".jsonl", ".ndjson"):
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        elif suf == ".json":
            obj = json.loads(p.read_text(encoding="utf-8"))
            for row in (obj if isinstance(obj, list) else obj.get("data", [obj])):
                yield row
        elif suf == ".csv":
            with p.open("r", encoding="utf-8") as f:
                yield from csv.DictReader(f)
        elif suf == ".parquet":
            import pyarrow.parquet as pq
            yield from pq.read_table(p).to_pylist()
        else:
            for block in p.read_text(encoding="utf-8").split("\n\n"):
                if block.strip():
                    yield {"text": block.strip()}


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #


def prepare_base(cfg: LabConfig, tok: LabTokenizer, out_dir: Path) -> dict[str, Any]:
    paths = resolve_source(cfg.data.base_source, out_dir, "base")
    max_tokens = int(cfg.data.max_base_tokens)
    ids: list[int] = []
    for text in iter_texts(paths, cfg.data.text_field):
        ids.append(tok.bos_id)
        ids.extend(tok.encode(text))
        ids.append(tok.eos_id)
        if len(ids) >= max_tokens:
            break
    arr = np.asarray(ids[:max_tokens], dtype=np.uint16 if tok.vocab_size < 65536 else np.int32)
    n_val = max(1, int(len(arr) * cfg.data.val_fraction))
    n_val = min(n_val, max(1, len(arr) // 10))
    train, val = arr[:-n_val], arr[-n_val:]
    train.tofile(str(out_dir / "base_train.bin"))
    val.tofile(str(out_dir / "base_val.bin"))
    meta = {"train_tokens": int(len(train)), "val_tokens": int(len(val)),
            "dtype": str(arr.dtype), "vocab_size": tok.vocab_size}
    (out_dir / "base_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _chatml_prompt(row: dict[str, Any], template: str) -> tuple[str, str]:
    msgs = row.get("messages")
    if msgs:
        parts = []
        for m in msgs:
            parts.append(f"<|im_start|>{m.get('role', 'user')}\n{m.get('content', '')}<|im_end|>\n")
        body = "".join(parts)
        # last assistant message is the target
        if msgs[-1].get("role") == "assistant":
            head = f"<|im_start|>assistant\n{msgs[-1].get('content', '')}"
            prompt = "".join(parts[:-1]) + "<|im_start|>assistant\n"
            return prompt, msgs[-1].get("content", "") + "<|im_end|>"
        return body, ""
    instruction = row.get("instruction", "")
    inp = row.get("input", "")
    output = row.get("output", row.get("completion", ""))
    if template == "alpaca" or (template == "chatml" and instruction):
        prompt = f"### Instruction:\n{instruction}\n\n"
        if inp:
            prompt += f"### Input:\n{inp}\n\n"
        prompt += "### Response:\n"
        return prompt, str(output)
    if "prompt" in row:
        return str(row["prompt"]), str(row.get("completion", output))
    return "", str(row.get("text", output))


def prepare_sft(cfg: LabConfig, tok: LabTokenizer, out_dir: Path) -> dict[str, Any]:
    paths = resolve_source(cfg.data.sft_source, out_dir, "sft")
    seq_len = int(cfg.data.sft_max_seq_len or cfg.model.seq_len)
    template = cfg.data.sft_template
    inputs: list[list[int]] = []
    labels: list[list[int]] = []
    for row in iter_records(paths):
        prompt, answer = _chatml_prompt(row, template)
        p_ids = tok.encode(prompt)
        a_ids = tok.encode(answer) + [tok.eos_id]
        # Always keep the full answer: trim the *left* part of a long prompt.
        max_prompt = max(1, seq_len - len(a_ids) - 1)
        if len(p_ids) > max_prompt:
            p_ids = p_ids[-max_prompt:]
        a_ids = a_ids[:max(1, seq_len - 1 - len(p_ids))]
        seq = ([tok.bos_id] + p_ids + a_ids)[:seq_len]
        lab = ([-100] * (1 + len(p_ids)) + a_ids)[:seq_len]
        pad = seq_len - len(seq)
        seq = seq + [tok.pad_id] * pad
        lab = lab + [-100] * pad
        inputs.append(seq)
        labels.append(lab)
    if not inputs:
        raise RuntimeError("SFT-датасет пуст или имеет неожиданный формат.")
    x = np.asarray(inputs, dtype=np.int32)
    y = np.asarray(labels, dtype=np.int32)
    rng = np.random.default_rng(cfg.data.seed)
    perm = rng.permutation(len(x))
    x, y = x[perm], y[perm]
    n_val = min(max(1, int(len(x) * cfg.data.val_fraction)), max(1, len(x) // 5))
    np.savez(out_dir / "sft_val.npz", input_ids=x[:n_val], labels=y[:n_val])
    np.savez(out_dir / "sft_train.npz", input_ids=x[n_val:], labels=y[n_val:])
    meta = {"train": int(len(x) - n_val), "val": int(n_val), "seq_len": seq_len,
            "template": template, "supervised_tokens": int((y != -100).sum())}
    (out_dir / "sft_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def prepare_all(cfg: LabConfig) -> dict[str, Any]:
    out = cfg.path(cfg.data.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base_paths = resolve_source(cfg.data.base_source, out, "base")
    tok = build_tokenizer(cfg, iter_texts(base_paths, cfg.data.text_field))
    tok.save(out)
    base = prepare_base(cfg, tok, out)
    sft = prepare_sft(cfg, tok, out)
    cfg.to_json(cfg.path("config.json"))
    return {"base": base, "sft": sft, "vocab_size": tok.vocab_size,
            "data_dir": str(out)}


# --------------------------------------------------------------------------- #
# Torch datasets
# --------------------------------------------------------------------------- #


class PackedBaseDataset(Dataset):
    def __init__(self, path: str | Path, seq_len: int, dtype=np.uint16, stride: int = 0):
        self.data = np.fromfile(path, dtype=dtype)
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.n = max(0, (len(self.data) - seq_len) // self.stride)

    def __len__(self) -> int:
        return max(1, self.n)

    def __getitem__(self, i: int):
        s = (i % max(1, self.n)) * self.stride
        chunk = self.data[s:s + self.seq_len].astype(np.int64)
        x = torch.from_numpy(chunk.copy())
        return x, x.clone()  # labels aligned with inputs; model shifts internally


class SFTDataset(Dataset):
    def __init__(self, path: str | Path):
        z = np.load(path)
        self.x = z["input_ids"]
        self.y = z["labels"]

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return (torch.from_numpy(self.x[i].astype(np.int64)),
                torch.from_numpy(self.y[i].astype(np.int64)))


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool = True,
                seed: int = 42, device: torch.device | None = None,
                use_all_tpu_cores: bool = False) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        generator=g, num_workers=0, drop_last=False,
                        pin_memory=bool(device and device.type == "cuda"))
    if use_all_tpu_cores and device is not None and device.type == "xla":
        try:
            from torch_xla.distributed.parallel_loader import MpDeviceLoader
            import torch_xla.core.xla_model as xm
            loader = MpDeviceLoader(loader, xm.get_xla_supported_devices())
        except Exception as exc:  # pragma: no cover
            print(f"[warn] MpDeviceLoader недоступен ({exc}); используется один TPU-кор.")
    return loader
