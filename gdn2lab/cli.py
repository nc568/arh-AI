"""Command-line entry point.  The Colab UI is a thin wrapper around these commands."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from .config import (GGUF_QUANTS, NPU_PROFILES, PRESETS, LabConfig, ModelConfig,
                     count_params, estimate_memory, human_params, solve_architecture,
                     state_size)


def _load(args) -> LabConfig:
    cfg_path = Path(args.config) if args.config else Path(args.workdir or "/content/work") / "config.json"
    cfg = LabConfig.from_json(cfg_path) if cfg_path.exists() else LabConfig()
    if args.workdir:
        cfg.workdir = args.workdir
    for pair in args.set or []:
        if "=" not in pair:
            raise ValueError(f"--set ожидает ключ=значение, получено {pair!r}")
        key, raw = pair.split("=", 1)
        parts = key.split(".")
        if len(parts) != 2:
            raise ValueError("--set поддерживает только один уровень, например model.n_layer=6")
        section, field = parts
        obj = getattr(cfg, section)
        cur = getattr(obj, field)
        if isinstance(cur, bool):
            val = raw.lower() in ("1", "true", "yes", "y", "да")
        elif isinstance(cur, int):
            val = int(raw)
        elif isinstance(cur, float):
            val = float(raw)
        else:
            val = raw
        setattr(obj, field, val)
        if section == "model" and field == "vocab_size":
            cfg.data.tokenizer["vocab_size"] = int(val)
    return cfg


def _save(cfg: LabConfig, args) -> None:
    cfg.path().mkdir(parents=True, exist_ok=True)
    cfg.to_json(cfg.path("config.json"))


def cmd_info(args):
    from .device import detect_runtime
    cfg = _load(args)
    dev = detect_runtime()
    p = count_params(cfg.model)
    mem = estimate_memory(cfg.model, cfg.train.micro_batch_size or 4,
                          cfg.train.precision, cfg.train.optimizer, dev["mem_gb"] or 15.0)
    print(json.dumps({
        "runtime": dev,
        "model": cfg.model.to_dict() if hasattr(cfg.model, "to_dict") else vars(cfg.model),
        "params": p, "params_human": human_params(p),
        "state_size": state_size(cfg.model),
        "memory_estimate": mem,
        "warnings": cfg.model.validate(),
        "npu_profiles": {k: v["label"] for k, v in NPU_PROFILES.items()},
        "gguf_quants": GGUF_QUANTS,
        "presets": PRESETS,
    }, ensure_ascii=False, indent=2, default=str))


def cmd_solve(args):
    c = solve_architecture(args.params, vocab_size=args.vocab, head_dim=args.head_dim,
                           mlp_ratio=args.mlp_ratio, expand_v=args.expand_v,
                           seq_len=args.seq_len, hybrid_every=args.hybrid_every)
    print(json.dumps({"config": vars(c), "params": count_params(c),
                      "state_size": state_size(c)}, ensure_ascii=False, indent=2))


def cmd_prepare(args):
    from .data import prepare_all
    cfg = _load(args)
    cfg.path().mkdir(parents=True, exist_ok=True)
    info = prepare_all(cfg)  # сохраняет config.json с ФАКТИЧЕСКИМ vocab_size
    print(json.dumps(info, ensure_ascii=False, indent=2))


def cmd_train(args):
    from .trainer import run_training
    cfg = _load(args)
    phases = {"all": ("base", "sft"), "base": ("base",), "sft": ("sft",)}[args.phase]
    res = run_training(cfg, phases=phases, force_prepare=args.force_prepare)
    print(json.dumps(res, ensure_ascii=False, indent=2))


def cmd_export_gguf(args):
    from .export_gguf import export_gguf
    cfg = _load(args)
    rep = export_gguf(cfg, args.ckpt_dir, args.out, args.quant)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


def cmd_export_litert(args):
    from .export_litert import export_litert
    cfg = _load(args)
    rep = export_litert(cfg, args.ckpt_dir, args.out, args.profile, args.quant,
                        args.prefill_len, args.export_prefill)
    rep.pop("signature", None)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


def cmd_chat(args):
    cfg = _load(args)
    if args.from_gguf:
        from .gguf_runtime import generate_text, load_gguf
        model, tok = load_gguf(args.from_gguf, args.device)
        print(generate_text(model, tok, args.prompt, args.max_new_tokens,
                            args.temperature, args.top_k, args.top_p))
        return
    if args.from_litert:
        from .data import LabTokenizer
        from .litert_runtime import LiteRTGenerator
        sig = Path(args.from_litert).with_suffix(".signature.json")
        tok = LabTokenizer.load(cfg.path(cfg.data.out_dir))
        gen = LiteRTGenerator(args.from_litert, sig, tok)
        print(gen.generate(args.prompt, args.max_new_tokens, args.temperature,
                           args.top_k, args.top_p))
        return
    from .data import LabTokenizer
    from .trainer import load_checkpoint
    ckpt = Path(args.ckpt) if args.ckpt else cfg.path(cfg.train.out_dir) / "model.safetensors"
    model = load_checkpoint(cfg, ckpt, torch.device(args.device), quantized=False)
    model.eval()
    tok = LabTokenizer.load(cfg.path(cfg.data.out_dir))
    from .gguf_runtime import generate_text
    print(generate_text(model, tok, args.prompt, args.max_new_tokens,
                        args.temperature, args.top_k, args.top_p))


def cmd_selftest(args):
    from .selftest import run_all
    ok = run_all(verbose=True)
    print("SELFTEST:", "OK" if ok else "FAILED")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gdn2lab", description="GDN-2 для бесплатного Colab")
    p.add_argument("--config", help="путь к config.json")
    p.add_argument("--workdir", default=None, help="рабочая папка (по умолчанию из config.json или /content/work)")
    p.add_argument("--set", action="append", help="переопределить поле, напр. model.n_layer=6")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="runtime, параметры, оценка памяти").set_defaults(fn=cmd_info)
    s = sub.add_parser("solve", help="подобрать архитектуру под число параметров")
    s.add_argument("--params", type=int, default=15_000_000)
    s.add_argument("--vocab", type=int, default=8192)
    s.add_argument("--head-dim", type=int, default=64)
    s.add_argument("--mlp-ratio", type=float, default=2.0)
    s.add_argument("--expand-v", type=float, default=1.0)
    s.add_argument("--seq-len", type=int, default=256)
    s.add_argument("--hybrid-every", type=int, default=0)
    s.set_defaults(fn=cmd_solve)

    sub.add_parser("prepare", help="подготовить данные и токенизатор").set_defaults(fn=cmd_prepare)
    t = sub.add_parser("train", help="обучение")
    t.add_argument("--phase", choices=["all", "base", "sft"], default="all")
    t.add_argument("--force-prepare", action="store_true")
    t.set_defaults(fn=cmd_train)

    g = sub.add_parser("export-gguf", help="экспорт в GGUF")
    g.add_argument("--quant", default=None, help=f"один из {list(GGUF_QUANTS)}")
    g.add_argument("--out", default=None)
    g.add_argument("--ckpt-dir", default=None)
    g.set_defaults(fn=cmd_export_gguf)

    l = sub.add_parser("export-litert", help="экспорт в LiteRT (.tflite)")
    l.add_argument("--profile", default=None, choices=list(NPU_PROFILES))
    l.add_argument("--quant", default=None,
                   choices=["auto", "float32", "fp16", "dynamic_int8", "full_int8"])
    l.add_argument("--prefill-len", type=int, default=None)
    l.add_argument("--export-prefill", action="store_true", default=None)
    l.add_argument("--out", default=None)
    l.add_argument("--ckpt-dir", default=None)
    l.set_defaults(fn=cmd_export_litert)

    c = sub.add_parser("chat", help="генерация из чекпоинта / GGUF / LiteRT")
    c.add_argument("--prompt", default="Привет! Кто ты?")
    c.add_argument("--max-new-tokens", type=int, default=100)
    c.add_argument("--temperature", type=float, default=0.8)
    c.add_argument("--top-k", type=int, default=40)
    c.add_argument("--top-p", type=float, default=0.95)
    c.add_argument("--device", default="cpu")
    c.add_argument("--ckpt", default=None)
    c.add_argument("--from-gguf", default=None)
    c.add_argument("--from-litert", default=None)
    c.set_defaults(fn=cmd_chat)

    sub.add_parser("selftest", help="проверить ядро (рекуррентность, 8 бит, GGUF)").set_defaults(fn=cmd_selftest)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = args.fn(args)
    return int(result or 0)


if __name__ == "__main__":
    sys.exit(main())
