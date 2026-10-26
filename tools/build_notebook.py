"""Generate colab/GDN2_Lab.ipynb from these cell sources.

Kept in the repo so the notebook can be regenerated/editable without hand-
editing JSON:  python tools/build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "colab" / "GDN2_Lab.ipynb"

CELLS: list[tuple[str, str]] = []
CELLS.append(("markdown", r'''# 🧠 GDN-2 Lab — маленькая модель с нуля в бесплатном Colab

**Gated DeltaNet-2** ([arXiv:2605.22791](https://arxiv.org/abs/2605.22791)) — линейное внимание с
раздельным *стиранием* и *записью*. Этот ноутбук позволяет:

1. задать **число параметров** (или точную архитектуру) в удобной форме;
2. положить **base**-корпус и **SFT**-датасет (загрузка файлов, Google Drive, HuggingFace, URL, просто текст);
3. обучить модель **с нуля** на бесплатном **T4 GPU** или **TPU**;
4. учиться в **16 бит** (FP16/BF32) или **нативно в 8 бит** — INT8 QAT с настоящим int8-GEMM
   (`torch._int_mm`) + 8-битный оптимизатор; FP8-эмуляция для новых GPU;
5. в пару кликов получить **GGUF** (F16/Q8_0/Q4_0/...) или **LiteRT `.tflite` под конкретный NPU**
   (Qualcomm QNN/HTP, MediaTek NeuroPilot, Google Edge TPU, GPU, CPU) с отчётом совместимости.

> ⚠️ Честно про экспорт: llama.cpp пока не знает *Gated Delta Rule-2*, поэтому GGUF пишется
> с кастомной архитектурой `gdn2` и запускается встроенным PyTorch-runtime (в комплекте).
> LiteRT-граф — это пошаговый decode с состоянием фиксированного размера; он реально
> конвертируется и квантуется, но delegate конкретного NPU ставится на устройстве.
> Подробности — в `docs/EXPORT.md`.

> 📜 Код реализации написан независимо по уравнениям статьи. Официальный репозиторий NVIDIA
> распространяется под **NVIDIA Source Code License-NC**, мы его код не копируем.

**Как пользоваться:** запускайте ячейки по порядку. Ячейка 4 — вся форма настроек и кнопки.
Если виджеты не отображаются — внизу есть запасной вариант на `#@param`.
'''))

CELLS.append(("code", r'''#@title 1️⃣ Установка и окружение (запустить первым) { display-mode: "form" }
BRANCH    = "arena/fde0d6d5-arh-ai"                  #@param {type:"string"}
REPO_URL  = "https://github.com/nc568/arh-AI.git"   #@param {type:"string"}
REPO_DIR  = "/content/arh-AI"                       #@param {type:"string"}
WORKDIR   = "/content/work"                         #@param {type:"string"}
USE_DRIVE = False                                   #@param {type:"boolean"}
INSTALL_LITERT_NOW = False                          #@param {type:"boolean"}

import os, sys, subprocess, json, shutil, textwrap, glob, time
from pathlib import Path

def sh(cmd, check=True):
    print("$", " ".join(cmd))
    return subprocess.run(cmd, check=check, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout

# --- репозиторий -----------------------------------------------------------
if (Path(REPO_DIR) / ".git").exists():
    try:
        print(sh(["git", "-C", REPO_DIR, "pull", "-q", "--ff-only"]))
    except Exception as e:
        print("[warn] git pull:", e)
else:
    print(sh(["git", "clone", "-q", "-b", BRANCH, "--depth", "1", REPO_URL, REPO_DIR]))
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)
os.makedirs(WORKDIR, exist_ok=True)

# --- базовые зависимости (torch в Colab уже установлен) ---------------------
BASE_DEPS = ["numpy", "tokenizers", "safetensors", "gguf", "tqdm", "ipywidgets"]
print(sh([sys.executable, "-m", "pip", "install", "-q"] + BASE_DEPS, check=False)[-1500:])
try:
    from google.colab import output
    output.enable_custom_widget_manager()
except Exception as e:
    print("[info] кастомный менеджер виджетов недоступен:", e)

# --- Google Drive (необязательно, чтобы чекпоинты не пропали) ---------------
if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        WORKDIR = "/content/drive/MyDrive/gdn2lab"
        os.makedirs(WORKDIR, exist_ok=True)
        print("💾 Рабочая папка перенесена на Drive:", WORKDIR)
    except Exception as e:
        print("[warn] Drive не смонтирован:", e)

# --- тяжёлый конвертер LiteRT: ставим только по запросу --------------------
if INSTALL_LITERT_NOW:
    print("⏳ Ставим litert-torch (2–4 минуты)...")
    print(sh([sys.executable, "-m", "pip", "install", "-q",
              "litert-torch>=0.10", "ai-edge-litert", "ai-edge-quantizer"],
             check=False)[-2000:])
    print("Если версия torch изменилась — перезапустите среду (Runtime -> Restart) и повторите ячейку 1.")

from gdn2lab.device import detect_runtime
RUNTIME = detect_runtime()
print("\n🖥  Обнаружено:", json.dumps(RUNTIME, ensure_ascii=False, indent=2, default=str))
print("\n📁 WORKDIR =", WORKDIR)
'''))

CELLS.append(("code", r'''#@title 2️⃣ Быстрая проверка ядра (рекомендуется) { display-mode: "form" }
# Проверяет рекуррентность GDN-2, 8-битный путь и GGUF roundtrip. ~10 секунд.
import subprocess, sys
p = subprocess.run([sys.executable, "-m", "gdn2lab.cli", "selftest"],
                   text=True, capture_output=True)
print(p.stdout[-4000:])
if p.returncode:
    print(p.stderr[-2000:])
'''))

CELLS.append(("code", r'''#@title 3️⃣ Все настройки в одном месте { display-mode: "form" }
import ipywidgets as W
from IPython.display import display, HTML, clear_output
from collections import deque
import subprocess, sys, json, os, glob, threading
from pathlib import Path

from gdn2lab.config import (LabConfig, ModelConfig, DataConfig, TrainConfig,
                            ExportConfig, PRESETS, NPU_PROFILES, GGUF_QUANTS,
                            solve_architecture, count_params, estimate_memory,
                            human_params, state_size)
from gdn2lab.device import detect_runtime

RUNTIME = detect_runtime()
BOX = W.Layout(width="98%")
def S(desc, value, kind="int", **kw):
    if kind == "int":   w = W.IntText(value=int(value), description=desc, layout=BOX, **kw)
    elif kind == "float": w = W.FloatText(value=float(value), description=desc, layout=BOX, **kw)
    elif kind == "text": w = W.Text(value=str(value), description=desc, layout=BOX, **kw)
    elif kind == "bool": w = W.Checkbox(value=bool(value), description=desc, layout=BOX, **kw)
    else: w = W.Dropdown(options=list(kw.pop("options")), value=value, description=desc, layout=BOX, **kw)
    w.style.description_width = "190px"
    return w
def SEC(title, *ws):
    return W.VBox([W.HTML(f"<h4 style='margin:8px 0 2px 0'>{title}</h4>"), *ws],
                  layout=W.Layout(border="1px solid #4444", padding="6px", margin="3px"))

# ------------------------------ вкладки ---------------------------------- #
preset   = S("Пресет", "Крошка · ~5M", "dd", options=PRESETS)
params_m = S("Млн параметров", 5.0, "float")
run_name = S("Имя запуска", "gdn2-tiny", "text")
precision = S("Точность", "fp16", "dd",
              options=["fp16", "bf16", "fp32", "int8", "fp8"])
device   = S("Устройство", "auto", "dd", options=["auto", "cuda", "tpu", "cpu"])
int8_gemm = S("Реальный int8 GEMM", True, "bool")
opt8     = S("8-битный оптимизатор", True, "bool")
qlmhead  = S("Квантовать LM-голову", False, "bool")
tpu_all  = S("Все ядра TPU", False, "bool")
seed     = S("Seed", 42, "int")
tab_start = SEC("🚀 Старт", preset, params_m, run_name, precision, device,
                int8_gemm, opt8, qlmhead, tpu_all, seed)

vocab    = S("Размер словаря", 8192, "int")
head_dim = S("head_dim", 64, "int")
mlp_ratio= S("MLP ratio", 2.0, "float")
expand_v = S("expand_v", 1.0, "float")
hybrid   = S("SWA каждые N слоёв (0=выкл)", 0, "int")
seq_len  = S("Длина последовательности", 256, "int")
chunk    = S("Размер чанка", 16, "int")
decay_blk= S("decay_block", 16, "int")
shortconv= S("Short conv", True, "bool")
conv_size= S("conv_size", 4, "int")
neg_eig  = S("Отриц. собств. значения", False, "bool")
max_decay= S("max |log decay| за токен", 8.0, "float")
dropout  = S("Dropout", 0.0, "float")
tie_emb  = S("Связать эмбеддинги", True, "bool")
manual   = S("✋ Ручная архитектура", False, "bool")
n_layer  = S("Слоёв (вручную)", 6, "int")
d_model  = S("d_model (вручную)", 384, "int")
n_head   = S("Голов (вручную)", 6, "int")
tab_arch = SEC("🧠 Архитектура", vocab, head_dim, mlp_ratio, expand_v, hybrid,
               seq_len, chunk, decay_blk, shortconv, conv_size, neg_eig,
               max_decay, dropout, tie_emb, manual, n_layer, d_model, n_head)

SRC = ["demo", "upload", "path", "hf", "url", "paste"]
base_kind = S("Base: источник", "demo", "dd", options=SRC)
base_path = S("Base: путь/шаблон", "/content/work/data/raw/base", "text")
base_hf   = S("Base: HF dataset id", "roneneldan/TinyStories", "text")
base_url  = S("Base: URL", "", "text")
base_text = S("Base: вставить текст", "", "text")
sft_kind  = S("SFT: источник", "demo", "dd", options=SRC)
sft_path  = S("SFT: путь/шаблон", "/content/work/data/raw/sft", "text")
sft_hf    = S("SFT: HF dataset id", "yahma/alpaca-cleaned", "text")
sft_url   = S("SFT: URL", "", "text")
sft_text  = S("SFT: вставить jsonl", "", "text")
tok_kind  = S("Токенизатор", "scratch", "dd", options=["scratch", "hf", "byte"])
tok_hf    = S("HF токенизатор", "gpt2", "text")
text_field= S("Поле с текстом", "text", "text")
sft_templ = S("SFT шаблон", "chatml", "dd", options=["chatml", "alpaca", "plain"])
max_tok   = S("Макс. токенов корпуса", 5000000, "int")
val_frac  = S("Доля валидации", 0.02, "float")
sft_seq   = S("SFT длина (0=как base)", 0, "int")
upload_base = W.Button(description="⬆️ Загрузить base-файлы", button_style="info")
upload_sft  = W.Button(description="⬆️ Загрузить SFT-файлы", button_style="info")
tab_data = SEC("📚 Данные", base_kind, base_path, base_hf, base_url, base_text,
               sft_kind, sft_path, sft_hf, sft_url, sft_text,
               W.HBox([upload_base, upload_sft]),
               tok_kind, tok_hf, text_field, sft_templ, max_tok, val_frac, sft_seq)

micro    = S("micro_batch (0=авто)", 0, "int")
accum    = S("grad_accum", 4, "int")
bsteps   = S("Шагов pretrain", 300, "int")
blr      = S("LR pretrain", 0.003, "float")
sepochs  = S("Эпох SFT", 3, "int")
slr      = S("LR SFT", 0.001, "float")
warmup   = S("Warmup, доля", 0.05, "float")
wd       = S("Weight decay", 0.1, "float")
clip     = S("Grad clip", 1.0, "float")
sched    = S("Scheduler", "cosine", "dd", options=["cosine", "linear", "constant"])
eval_ev  = S("Eval каждые N шагов", 50, "int")
log_ev   = S("Лог каждые N шагов", 10, "int")
save_ev  = S("Сохранять каждые N (0=нет)", 0, "int")
sample_ev= S("Пример текста каждые N", 0, "int")
resume   = S("Resume", False, "bool")
wandb_on = S("W&B логирование", False, "bool")
tab_train= SEC("⚙️ Обучение", micro, accum, bsteps, blr, sepochs, slr,
               warmup, wd, clip, sched, eval_ev, log_ev, save_ev, sample_ev,
               resume, wandb_on)

gguf_q   = S("GGUF квантование", "Q8_0", "dd", options=list(GGUF_QUANTS))
npu      = S("NPU-профиль", "qualcomm_qnn_htp", "dd", options=list(NPU_PROFILES))
litert_q = S("LiteRT квантование", "auto", "dd",
             options=["auto", "float32", "fp16", "dynamic_int8", "full_int8"])
prefill_n= S("Prefill длина (0=из профиля)", 0, "int")
prefill_on=S("Добавить prefill-подпись", False, "bool")
chat_src = S("Чат из", "checkpoint", "dd", options=["checkpoint", "GGUF", "LiteRT"])
prompt   = S("Промпт", "Привет! Кто ты?", "text")
tab_export=SEC("📦 Экспорт и чат", gguf_q, npu, litert_q, prefill_n, prefill_on,
               chat_src, prompt)

tabs = W.Tab(children=[tab_start, tab_arch, tab_data, tab_train, tab_export])
for i, name in enumerate(["🚀 Старт", "🧠 Архитектура", "📚 Данные", "⚙️ Обучение", "📦 Экспорт"]):
    tabs.set_title(i, name)

summary = W.HTML(value="<i>Расчёт параметров...</i>")
out = W.Output(layout=W.Layout(border="1px solid #8884", max_height="360px",
                               overflow_y="auto", padding="6px"))
BTN = lambda label, style="": W.Button(description=label, button_style=style,
                                      layout=W.Layout(width="auto", margin="2px"))
b_calc  = BTN("🧮 Пересчитать")
b_save  = BTN("💾 Сохранить конфиг")
b_data  = BTN("📚 Подготовить данные", "info")
b_all   = BTN("🚀 Обучить ВСЁ", "success")
b_base  = BTN("🧱 Только pretrain")
b_sft   = BTN("🎯 Только SFT")
b_gguf  = BTN("📤 Экспорт GGUF", "warning")
b_lite  = BTN("📱 Экспорт LiteRT", "warning")
b_chat  = BTN("💬 Чат с моделью")
b_info  = BTN("ℹ️ Инфо / память")
buttons = W.HBox([b_calc, b_save, b_data, b_all, b_base, b_sft, b_gguf, b_lite, b_chat, b_info],
                 layout=W.Layout(flex_flow="row wrap"))
display(tabs, summary, buttons, out)

# ------------------------------ конфиг ----------------------------------- #
def source_spec(kind, path, hf, url, text):
    k = kind.value
    if k == "demo":   return {"kind": "demo"}
    if k == "upload": return {"kind": "upload"}
    if k == "path":   return {"kind": "path", "path": path.value}
    if k == "hf":     return {"kind": "hf", "id": hf.value, "split": "train", "limit": 20000}
    if k == "url":    return {"kind": "url", "url": url.value}
    return {"kind": "paste", "text": text.value}

def build_config(save=True):
    target = int(max(0.2, params_m.value) * 1_000_000)
    if manual.value:
        mc = ModelConfig(vocab_size=vocab.value, d_model=d_model.value,
                         n_layer=n_layer.value, n_head=n_head.value,
                         head_dim=head_dim.value, seq_len=seq_len.value,
                         mlp_ratio=mlp_ratio.value, expand_v=expand_v.value,
                         hybrid_every=hybrid.value, tie_embeddings=tie_emb.value)
    else:
        mc = solve_architecture(target, vocab_size=vocab.value, head_dim=head_dim.value,
                                mlp_ratio=mlp_ratio.value, expand_v=expand_v.value,
                                seq_len=seq_len.value, hybrid_every=hybrid.value,
                                tie_embeddings=tie_emb.value)
    mc.seq_len, mc.chunk_size, mc.decay_block = seq_len.value, chunk.value, decay_blk.value
    mc.use_short_conv, mc.conv_size = shortconv.value, conv_size.value
    mc.allow_neg_eigval, mc.max_log_decay = neg_eig.value, max_decay.value
    mc.dropout = dropout.value
    cfg = LabConfig(model=mc, run_name=run_name.value, workdir=WORKDIR)
    cfg.data = DataConfig(
        base_source=source_spec(base_kind, base_path, base_hf, base_url, base_text),
        sft_source=source_spec(sft_kind, sft_path, sft_hf, sft_url, sft_text),
        tokenizer={"kind": tok_kind.value,
                   "vocab_size": mc.vocab_size,
                   "name": tok_hf.value},
        text_field=text_field.value, sft_template=sft_templ.value,
        max_base_tokens=max_tok.value, val_fraction=val_frac.value,
        sft_max_seq_len=sft_seq.value, seed=seed.value)
    cfg.train = TrainConfig(
        precision=precision.value, int8_gemm=int8_gemm.value,
        optimizer="adamw8bit" if opt8.value else "adamw",
        quantize_lm_head=qlmhead.value,
        micro_batch_size=micro.value, grad_accum=accum.value,
        base_steps=bsteps.value, base_lr=blr.value,
        sft_epochs=sepochs.value, sft_lr=slr.value,
        warmup_ratio=warmup.value, weight_decay=wd.value, grad_clip=clip.value,
        scheduler=sched.value, eval_every=eval_ev.value, log_every=log_ev.value,
        save_every=save_ev.value, sample_every=sample_ev.value,
        resume=resume.value, wandb=wandb_on.value, seed=seed.value,
        device=device.value, use_all_tpu_cores=tpu_all.value)
    cfg.export = ExportConfig(
        gguf_quant=gguf_q.value, litert_profile=npu.value,
        litert_quant=litert_q.value, prefill_len=prefill_n.value,
        export_prefill=prefill_on.value)
    if save:
        cfg.path().mkdir(parents=True, exist_ok=True)
        cfg.to_json(cfg.path("config.json"))
    return cfg

def update_summary(_=None):
    try:
        cfg = build_config(save=False)
        p = count_params(cfg.model)
        mem = estimate_memory(cfg.model, cfg.train.micro_batch_size or 4,
                              cfg.train.precision, cfg.train.optimizer,
                              RUNTIME.get("mem_gb") or 15.0)
        warn = cfg.model.validate()
        color = "#8f8" if mem["fits"] else "#f88"
        warn_html = ("<br><span style='color:#fbb'>⚠ " + "</span><br><span style='color:#fbb'>⚠ ".join(warn) + "</span>") if warn else ""
        summary.value = (
            f"<b>Параметры:</b> {human_params(p)} ({p:,}) &nbsp;|&nbsp; "
            f"<b>слоёв:</b> {cfg.model.n_layer}, <b>d_model:</b> {cfg.model.d_model}, "
            f"<b>голов:</b> {cfg.model.n_head}×{cfg.model.head_dim}<br>"
            f"<b>Состояние decode:</b> {state_size(cfg.model)*2/1024:.1f} КБ (фиксированное)<br>"
            f"<b>Память (оценка):</b> <span style='color:{color}'>{mem['estimated_gb']:.2f} ГБ</span> "
            f"из {mem['device_mem_gb']} ГБ; веса {mem['weights_gb']:.2f}, "
            f"активации {mem['activations_gb']:.2f}"
            + warn_html)
    except Exception as e:
        summary.value = f"<span style='color:#f88'>Ошибка конфига: {e}</span>"

def on_preset(change):
    v = PRESETS.get(change["new"])
    if v and v["params"]:
        params_m.value = round(v["params"] / 1e6, 2)
        seq_len.value, bsteps.value, micro.value = v["seq_len"], v["base_steps"], v["micro"]
    update_summary()
preset.observe(on_preset, names="value")
for wdg in (params_m, vocab, head_dim, mlp_ratio, expand_v, hybrid, seq_len,
            manual, n_layer, d_model, n_head, micro, opt8, precision):
    wdg.observe(update_summary, names="value")
update_summary()

# ------------------------------ запуск CLI ------------------------------- #
def run_cli(args, title=""):
    build_config(save=True)
    cmd = [sys.executable, "-m", "gdn2lab.cli", "--workdir", WORKDIR] + [str(a) for a in args]
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_DIR + os.pathsep + env.get("PYTHONPATH", "")
    tail = deque(maxlen=70)
    with out:
        print(f"\n▶ {title or ' '.join(args)}\n$ {' '.join(cmd[3:])}\n" + "-" * 60)
    try:
        proc = subprocess.Popen(cmd, cwd=REPO_DIR, env=env, text=True, bufsize=1,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            tail.append(line.rstrip())
            with out:
                clear_output(wait=True)
                print("\n".join(tail))
        rc = proc.wait()
        with out:
            clear_output(wait=True)
            print("\n".join(tail))
            print("-" * 60, f"\n{'✅ Готово' if rc == 0 else '❌ Ошибка'} (код {rc})")
    except Exception as e:
        with out:
            print("❌", repr(e))
    return None

def latest(pattern):
    files = sorted(glob.glob(str(Path(WORKDIR) / "exports" / pattern)), key=os.path.getmtime)
    return files[-1] if files else None

b_calc.on_click(lambda _: update_summary())
b_save.on_click(lambda _: (build_config(True), with_out_print("💾 config.json сохранён")))
b_data.on_click(lambda _: run_cli(["prepare"], "Подготовка данных"))
b_all.on_click(lambda _: run_cli(["train", "--phase", "all"], "Pretrain + SFT"))
b_base.on_click(lambda _: run_cli(["train", "--phase", "base"], "Pretrain"))
b_sft.on_click(lambda _: run_cli(["train", "--phase", "sft"], "SFT"))
b_gguf.on_click(lambda _: run_cli(["export-gguf", "--quant", gguf_q.value], "Экспорт GGUF"))
b_info.on_click(lambda _: run_cli(["info"], "Информация"))

def with_out_print(msg):
    with out:
        print(msg)

def do_upload(tag):
    try:
        from google.colab import files
        up = files.upload()
        dst = Path(WORKDIR) / "data" / "raw" / tag
        dst.mkdir(parents=True, exist_ok=True)
        for name, data in up.items():
            (dst / name).write_bytes(data)
        with out:
            print(f"⬆️ {tag}: загружено {list(up)} -> {dst}")
    except Exception as e:
        with out:
            print("❌ Загрузка доступна только в Colab:", e)
upload_base.on_click(lambda _: do_upload("base"))
upload_sft.on_click(lambda _: do_upload("sft"))

def do_litert(_):
    try:
        import litert_torch  # noqa
    except ImportError:
        with out:
            print("⏳ litert-torch не установлен — ставим (2–4 минуты). "
                  "Если torch обновился, перезапустите среду и повторите.")
        p = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "litert-torch>=0.10", "ai-edge-litert", "ai-edge-quantizer"],
                           text=True, capture_output=True)
        with out:
            print((p.stdout + p.stderr)[-1200:])
    args = ["export-litert", "--profile", npu.value, "--quant", litert_q.value]
    if prefill_n.value:
        args += ["--prefill-len", str(prefill_n.value)]
    if prefill_on.value:
        args += ["--export-prefill"]
    run_cli(args, "Экспорт LiteRT / NPU")
b_lite.on_click(do_litert)

def do_chat(_):
    src = chat_src.value
    args = ["chat", "--prompt", prompt.value, "--max-new-tokens", "100"]
    if src == "GGUF":
        f = latest("*.gguf")
        if not f:
            with out: print("❌ Сначала сделайте экспорт GGUF"); return
        args += ["--from-gguf", f]
    elif src == "LiteRT":
        f = latest("*.tflite")
        if not f:
            with out: print("❌ Сначала сделайте экспорт LiteRT"); return
        args += ["--from-litert", f]
    run_cli(args, f"Чат ({src})")
b_chat.on_click(do_chat)
print("✅ Форма готова. Настройте параметры и нажмите «🚀 Обучить ВСЁ».")
'''))

CELLS.append(("markdown", r'''## 🆘 Запасной режим (без виджетов)

Если ipywidgets не отображаются (например, при просмотре ноутбука на GitHub), используйте
нативные формы Colab `#@param` ниже. Они пишут тот же `config.json`, а команды запускаются
обычными ячейками.
'''))

CELLS.append(("code", r'''#@title 4️⃣ (необязательно) Конфиг через # @param { display-mode: "form" }
P_PARAMS_M   = 5.0        #@param {type:"number"}
P_PRECISION  = "fp16"     #@param ["fp16","bf16","fp32","int8","fp8"]
P_VOCAB      = 8192       #@param {type:"integer"}
P_SEQ_LEN    = 256        #@param {type:"integer"}
P_CHUNK      = 16         #@param {type:"integer"}
P_HEAD_DIM   = 64         #@param {type:"integer"}
P_MLP_RATIO  = 2.0        #@param {type:"number"}
P_BASE_STEPS = 300        #@param {type:"integer"}
P_SFT_EPOCHS = 3          #@param {type:"integer"}
P_MICRO_BS   = 0          #@param {type:"integer"}
P_GRAD_ACCUM = 4          #@param {type:"integer"}
P_BASE_SRC   = "demo"     #@param ["demo","upload","path","hf","url","paste"]
P_BASE_PATH  = "/content/work/data/raw/base"  #@param {type:"string"}
P_SFT_SRC    = "demo"     #@param ["demo","upload","path","hf","url","paste"]
P_SFT_PATH   = "/content/work/data/raw/sft"   #@param {type:"string"}
P_GGUF_Q     = "Q8_0"     #@param ["F16","Q8_0","Q5_1","Q5_0","Q4_1","Q4_0","F32"]
P_NPU        = "qualcomm_qnn_htp"  #@param ["generic_cpu","gpu_fp16","qualcomm_qnn_htp","mediatek_neuropilot","google_edgetpu","npu_auto"]

import json, os
from pathlib import Path
from gdn2lab.config import LabConfig, ModelConfig, solve_architecture, count_params, human_params

mc = solve_architecture(int(P_PARAMS_M * 1_000_000), vocab_size=P_VOCAB,
                        head_dim=P_HEAD_DIM, mlp_ratio=P_MLP_RATIO, seq_len=P_SEQ_LEN)
mc.chunk_size = P_CHUNK
cfg = LabConfig(model=mc, run_name="gdn2-form", workdir=WORKDIR)
cfg.data.base_source = {"kind": P_BASE_SRC, "path": P_BASE_PATH,
                        "url": P_BASE_SRC == "url" and P_BASE_PATH or "", "text": ""}
cfg.data.sft_source  = {"kind": P_SFT_SRC, "path": P_SFT_PATH,
                        "url": P_SFT_SRC == "url" and P_SFT_PATH or "", "text": ""}
cfg.data.tokenizer = {"kind": "scratch", "vocab_size": mc.vocab_size}
cfg.train.precision = P_PRECISION
cfg.train.base_steps, cfg.train.sft_epochs = P_BASE_STEPS, P_SFT_EPOCHS
cfg.train.micro_batch_size, cfg.train.grad_accum = P_MICRO_BS, P_GRAD_ACCUM
cfg.train.optimizer = "adamw8bit" if P_PRECISION in ("int8", "fp8") else "adamw"
cfg.export.gguf_quant, cfg.export.litert_profile = P_GGUF_Q, P_NPU
cfg.path().mkdir(parents=True, exist_ok=True)
cfg.to_json(cfg.path("config.json"))
print("💾", cfg.path("config.json"), "|", human_params(count_params(mc)), "параметров")
'''))

CELLS.append(("code", r'''#@title 5️⃣ (необязательно) Команды без виджетов — раскомментируйте нужные { display-mode: "form" }
import subprocess, sys, os
os.chdir(REPO_DIR)

def cli(*args):
    cmd = [sys.executable, "-m", "gdn2lab.cli", "--workdir", WORKDIR, *map(str, args)]
    print("$", " ".join(cmd[3:]), flush=True)
    p = subprocess.run(cmd, text=True, capture_output=True)
    print(p.stdout[-6000:])
    if p.returncode: print("STDERR:", p.stderr[-3000:])
    return p.returncode

# cli("selftest")
# cli("prepare")
# cli("train", "--phase", "all")
# cli("export-gguf", "--quant", P_GGUF_Q)
# cli("export-litert", "--profile", P_NPU)
# cli("chat", "--prompt", "Привет! Кто ты?")
print("Раскомментируйте нужные строки выше и запустите ячейку ещё раз.")
'''))

CELLS.append(("markdown", r'''## ❓ Частые вопросы

**«Out of memory» на T4.** Уменьшите `micro_batch` (или поставьте `0` — авто-подбор),
`seq_len`, `chunk_size`, число параметров. Включите `int8` + 8-битный оптимизатор —
это экономит до ~2× памяти на состояния оптимизатора.

**TPU.** Выберите «Среда выполнения → TPU» до запуска ноутбука; `torch_xla` подхватится автоматически,
точность станет `bf16`. INT8 GEMM на TPU эмулируется (веса всё равно остаются 8-битными в экспорте).

**GGUF не открывается в llama.cpp.** Это ожидаемо: архитектура `gdn2` кастомная.
Файл валиден как GGUF v3; для запуска используйте `python -m gdn2lab.cli chat --from-gguf ...`
или `gdn2lab.gguf_runtime`. См. `docs/EXPORT.md`.

**LiteRT-конвертер ругается на версии.** `litert-torch` требует совместимый `torch`.
В Colab обычно достаточно `pip install -q 'litert-torch>=0.10' ai-edge-litert ai-edge-quantizer`
и перезапуска среды. Проверено на `litert-torch 0.10`.

**Свои данные.** Base: `.txt`, `.jsonl`, `.json`, `.csv`, `.parquet` (поле `text`).
SFT: `.jsonl` с `messages` (ChatML), либо `instruction/input/output`, либо `prompt/completion`.
Промпт маскируется (`-100`),_loss считается только по ответу.

**Где лежат результаты.** `WORKDIR/data`, `WORKDIR/checkpoints`, `WORKDIR/exports`.
Включите `USE_DRIVE` в первой ячейке, чтобы сохранить их на Google Drive.

📄 Подробности: `README.md`, `docs/EXPORT.md`, `docs/NPU.md`.
'''))


def build():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": [], "toc_visible": True},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for kind, src in CELLS:
        lines = src.splitlines(keepends=True)
        cell = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb["cells"].append(cell)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT, len(nb["cells"]), "cells")


if __name__ == "__main__":
    build()
