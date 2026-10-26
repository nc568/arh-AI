"""From-scratch pretraining + SFT on CUDA / TPU / CPU."""
from __future__ import annotations

import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors.torch import load_file, save_file

from .config import LabConfig, count_params, human_params
from .data import (LabTokenizer, PackedBaseDataset, SFTDataset, make_loader,
                   prepare_all)
from .device import (auto_micro_batch, autocast_ctx, detect_runtime, empty_cache,
                     make_scaler, optimizer_step, resolve_precision,
                     setup_device, synchronize)
from .model import build_model
from .quant import convert_linears, fold_quantization, has_quantized, make_optimizer

IGNORE_INDEX = -100


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #


def _model_state_dict(model, cfg: LabConfig) -> dict[str, torch.Tensor]:
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    if cfg.model.tie_embeddings and "head.weight" in sd and "embed.weight" in sd:
        # safetensors refuses shared storages; tied head is rebuilt from embed.
        sd.pop("head.weight", None)
    return sd


def save_checkpoint(cfg: LabConfig, model, path: str | Path, step: int,
                    phase: str, tokenizer_dir: Path | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(_model_state_dict(model, cfg), str(path))
    meta = {"step": step, "phase": phase, "params": sum(p.numel() for p in model.parameters()),
            "precision": cfg.train.precision, "config": cfg.to_dict()}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    if tokenizer_dir is not None and tokenizer_dir.exists():
        for f in tokenizer_dir.glob("tokenizer*"):
            shutil.copyfile(f, path.parent / f.name)
        for f in ("hf_name.txt",):
            if (tokenizer_dir / f).exists():
                shutil.copyfile(tokenizer_dir / f, path.parent / f)
    return path


def load_checkpoint(cfg: LabConfig, path: str | Path, device: torch.device,
                    quantized: bool = False, strict: bool = True):
    path = Path(path)
    model = build_model(cfg.model)
    if quantized:
        convert_linears(model, scheme="fp8_e4m3" if cfg.train.precision == "fp8" else "int8",
                        int8_gemm=cfg.train.int8_gemm,
                        quantize_lm_head=cfg.train.quantize_lm_head,
                        quantize_embeddings=cfg.train.quantize_embeddings)
    sd = load_file(str(path))
    if cfg.model.tie_embeddings and "head.weight" not in sd:
        sd["head.weight"] = sd["embed.weight"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    unexpected = [k for k in unexpected if not k.endswith("inv_freq")]
    if strict and (missing or unexpected):
        raise RuntimeError(f"Ошибка загрузки {path}: missing={missing}, unexpected={unexpected}")
    return model.to(device)


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


def _scheduler(optimizer, total_steps: int, warmup: int, kind: str, min_ratio: float):
    def fn(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max(1e-3, (step + 1) / warmup)
        t = (step - warmup) / max(1, total_steps - warmup)
        t = min(1.0, max(0.0, t))
        if kind == "linear":
            cur = 1.0 - t
        elif kind == "constant":
            cur = 1.0
        else:
            cur = 0.5 * (1.0 + math.cos(math.pi * t))
        return min_ratio + (1.0 - min_ratio) * cur
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


class Trainer:
    def __init__(self, cfg: LabConfig, model, device: torch.device, info: dict[str, Any],
                 tokenizer: LabTokenizer | None = None, log_fn: Callable[[str], None] | None = None):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.info = info
        self.tokenizer = tokenizer
        self.log = log_fn or print
        self.compute_dtype = resolve_precision(cfg.train.precision, device)
        self.scaler = make_scaler(self.compute_dtype, device)
        self.history: list[dict[str, Any]] = []
        self.wandb = None
        if cfg.train.wandb:
            try:
                import wandb
                self.wandb = wandb
                wandb.init(project="gdn2lab", name=cfg.run_name, config=cfg.to_dict())
            except Exception as exc:
                self.log(f"[warn] wandb недоступен: {exc}")

    def _to_device(self, batch):
        return tuple(t.to(self.device, non_blocking=True) for t in batch)

    @torch.no_grad()
    def evaluate(self, dataset, max_batches: int = 10) -> float:
        self.model.eval()
        loader = make_loader(dataset, max(1, self.cfg.train.micro_batch_size or 1),
                             shuffle=False, seed=0, device=self.device)
        total, n = 0.0, 0
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x, y = self._to_device(batch)
            with autocast_ctx(self.device, self.compute_dtype):
                _, loss = self.model(x, y)
            if loss is not None:
                total += float(loss.detach().float())
                n += 1
            if self.device.type == "xla":
                synchronize(self.device, self.info)
        self.model.train()
        return total / max(1, n)

    @torch.no_grad()
    def sample(self, prompt: str, max_new_tokens: int = 60) -> str:
        if self.tokenizer is None:
            return ""
        self.model.eval()
        ids = [self.tokenizer.bos_id] + self.tokenizer.encode(prompt)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(x, max_new_tokens=max_new_tokens,
                                  eos_id=self.tokenizer.eos_id)
        self.model.train()
        return self.tokenizer.decode(out[0].tolist())

    def train_phase(self, phase: str, train_ds, val_ds, lr: float,
                    total_steps: int | None = None, epochs: int | None = None,
                    save_name: str | None = None) -> Path:
        cfg = self.cfg
        bs = int(cfg.train.micro_batch_size or 1)
        accum = max(1, int(cfg.train.grad_accum))
        steps = total_steps or max(1, epochs * max(1, len(train_ds) // bs))
        warmup = max(1, int(steps * cfg.train.warmup_ratio))
        opt = make_optimizer(self.model.parameters(), cfg.train.optimizer, lr,
                             cfg.train.weight_decay)
        sched = _scheduler(opt, steps, warmup, cfg.train.scheduler,
                           cfg.train.min_lr_ratio)
        loader = make_loader(train_ds, bs, shuffle=True, seed=cfg.train.seed,
                             device=self.device,
                             use_all_tpu_cores=cfg.train.use_all_tpu_cores)
        it = iter(loader)
        tok_dir = cfg.path(cfg.data.out_dir)
        out_dir = cfg.path(cfg.train.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_name = save_name or f"{phase}_final.safetensors"

        self.log(f"▶ {phase.upper()}: {steps} шагов, lr={lr:g}, micro_bs={bs}, "
                 f"grad_accum={accum}, precision={cfg.train.precision}/{self.compute_dtype}")
        self.model.train()
        t0 = time.time()
        running = 0.0
        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(accum):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)
                    batch = next(it)
                x, y = self._to_device(batch)
                with autocast_ctx(self.device, self.compute_dtype):
                    logits, loss = self.model(x, y)
                    if loss is None:
                        loss = torch.nn.functional.cross_entropy(
                            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                            y[:, 1:].reshape(-1), ignore_index=IGNORE_INDEX)
                loss = loss / accum
                if getattr(self.scaler, "enabled", False):
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                step_loss += float(loss.detach().float())
                if self.device.type == "xla" and accum > 1:
                    synchronize(self.device, self.info)
            optimizer_step(opt, self.scaler, self.model, cfg.train.grad_clip,
                           self.device, self.info)
            sched.step()
            synchronize(self.device, self.info)
            running += step_loss

            if step % max(1, cfg.train.log_every) == 0 or step == 1:
                avg = running / min(step, max(1, cfg.train.log_every))
                running = 0.0
                elapsed = time.time() - t0
                tps = step * bs * accum * cfg.model.seq_len / max(1e-6, elapsed)
                msg = (f"[{phase} {step}/{steps}] loss={avg:.4f} "
                       f"lr={sched.get_last_lr()[0]:.2e} tok/s={tps:,.0f} "
                       f"elapsed={elapsed:.0f}s")
                self.log(msg)
                rec = {"phase": phase, "step": step, "loss": avg,
                       "lr": sched.get_last_lr()[0], "tokens_per_s": tps}
                self.history.append(rec)
                if self.wandb:
                    self.wandb.log(rec, step=step)
            if cfg.train.eval_every and step % cfg.train.eval_every == 0 and val_ds is not None:
                val = self.evaluate(val_ds)
                self.log(f"[{phase} {step}] val_loss={val:.4f}")
                if self.wandb:
                    self.wandb.log({f"{phase}/val_loss": val}, step=step)
            if cfg.train.sample_every and step % cfg.train.sample_every == 0:
                self.log("-sample- " + self.sample(self._sample_prompt())[:300].replace("\n", " ⏎ "))
            if cfg.train.save_every and step % cfg.train.save_every == 0:
                save_checkpoint(cfg, self.model, out_dir / f"{phase}_step{step}.safetensors",
                                step, phase, tok_dir)

        path = save_checkpoint(cfg, self.model, out_dir / save_name, steps, phase, tok_dir)
        self._write_log(out_dir / "train_log.csv")
        self.log(f"✅ Сохранено: {path}")
        return path

    def _sample_prompt(self) -> str:
        if self.cfg.data.sft_template == "alpaca":
            return "### Instruction:\nРасскажи коротко о себе\n\n### Response:\n"
        if self.cfg.data.sft_template == "chatml":
            return "<|im_start|>user\nПривет! Кто ты?<|im_end|>\n<|im_start|>assistant\n"
        return "Однажды "

    def _write_log(self, path: Path):
        if not self.history:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
            w.writeheader()
            w.writerows(self.history)


# --------------------------------------------------------------------------- #
# Orchestration used by the CLI / notebook buttons
# --------------------------------------------------------------------------- #


def ensure_data(cfg: LabConfig, force: bool = False) -> LabTokenizer:
    out = cfg.path(cfg.data.out_dir)
    if force or not (out / "tokenizer_info.json").exists() or not (out / "base_train.bin").exists():
        print("📚 Готовим данные и токенизатор...")
        info = prepare_all(cfg)
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        # config.json on disk is the source of truth after preparation
        cfg = LabConfig.from_json(cfg.path("config.json"))
    return LabTokenizer.load(out)


def build_for_training(cfg: LabConfig, device: torch.device, quantized: bool = True):
    model = build_model(cfg.model)
    if quantized and cfg.train.precision in ("int8", "fp8"):
        if cfg.train.quantize_lm_head and cfg.model.tie_embeddings:
            print("[warn] quantize_lm_head вместе с tie_embeddings: "
                  "при экспорте голова будет привязана к эмбеддингам.")
        n = convert_linears(
            model, scheme="fp8_e4m3" if cfg.train.precision == "fp8" else "int8",
            int8_gemm=cfg.train.int8_gemm,
            quantize_lm_head=cfg.train.quantize_lm_head,
            quantize_embeddings=cfg.train.quantize_embeddings,
        )
        print(f"🔢 8-битный режим: {n} линейных слоёв ({cfg.train.precision}).")
    model = model.to(device)
    return model


def auto_batch(cfg: LabConfig, model, device, info, train_ds) -> int:
    if cfg.train.micro_batch_size and cfg.train.micro_batch_size > 0:
        return int(cfg.train.micro_batch_size)
    seq_len = cfg.model.seq_len

    def batch_fn(b):
        x = torch.randint(0, cfg.model.vocab_size, (b, seq_len), device=device)
        return x, x.clone()

    print("🔎 Подбираем максимальный micro_batch_size...")
    bs = auto_micro_batch(model, cfg, device, info, batch_fn, start=1, cap=32)
    cfg.train.micro_batch_size = bs
    print(f"✅ micro_batch_size = {bs}")
    return bs


def run_training(cfg: LabConfig, phases: tuple[str, ...] = ("base", "sft"),
                 force_prepare: bool = False) -> dict[str, str]:
    torch.manual_seed(cfg.train.seed)
    device, info = setup_device(cfg.train.device, cfg.train.use_all_tpu_cores)
    print("🖥 ", json.dumps(info, ensure_ascii=False, default=str))
    tok = ensure_data(cfg, force_prepare)
    cfg = LabConfig.from_json(cfg.path("config.json"))
    data_dir = cfg.path(cfg.data.out_dir)
    base_meta = json.loads((data_dir / "base_meta.json").read_text(encoding="utf-8"))
    sft_meta = json.loads((data_dir / "sft_meta.json").read_text(encoding="utf-8"))
    print(f"🧠 Модель: {human_params(count_params(cfg.model))} параметров, "
          f"состояние={cfg.model.n_layer} слоя")

    model = build_for_training(cfg, device)
    base_train = PackedBaseDataset(data_dir / "base_train.bin", cfg.model.seq_len,
                                   dtype=np_dtype(base_meta["dtype"]))
    base_val = PackedBaseDataset(data_dir / "base_val.bin", cfg.model.seq_len,
                                 dtype=np_dtype(base_meta["dtype"]))
    auto_batch(cfg, model, device, info, base_train)
    out_dir = cfg.path(cfg.train.out_dir)
    trainer = Trainer(cfg, model, device, info, tok)
    results: dict[str, str] = {}

    if "base" in phases:
        path = trainer.train_phase("base", base_train, base_val, cfg.train.base_lr,
                                   total_steps=cfg.train.base_steps)
        results["base"] = str(path)
    else:
        base_path = out_dir / "base_final.safetensors"
        if not base_path.exists():
            raise FileNotFoundError(f"Нет {base_path}: сначала запустите pretrain.")
        model = load_checkpoint(cfg, base_path, device,
                                quantized=cfg.train.precision in ("int8", "fp8"))
        trainer = Trainer(cfg, model, device, info, tok)

    if "sft" in phases:
        sft_train = SFTDataset(data_dir / "sft_train.npz")
        sft_val = SFTDataset(data_dir / "sft_val.npz")
        path = trainer.train_phase("sft", sft_train, sft_val, cfg.train.sft_lr,
                                   epochs=cfg.train.sft_epochs,
                                   save_name="sft_final.safetensors")
        results["sft"] = str(path)

    # portable, folded checkpoint used by both exporters
    if has_quantized(model):
        sd = fold_quantization(model)
    else:
        sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    if cfg.model.tie_embeddings:
        sd.pop("head.weight", None)
    final = out_dir / "model.safetensors"
    save_file(sd, str(final))
    cfg.to_json(out_dir / "config.json")
    for name in ("tokenizer.json", "tokenizer_info.json", "hf_name.txt"):
        if (data_dir / name).exists():
            shutil.copyfile(data_dir / name, out_dir / name)
    results["final"] = str(final)
    print(f"📦 Итоговая модель: {final}")
    return results


def np_dtype(name: str):
    import numpy as np
    return np.uint16 if "uint16" in name else np.int32
