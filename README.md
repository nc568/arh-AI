# 🧠 GDN-2 Lab — обучаем маленькую Gated DeltaNet-2 с нуля в бесплатном Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nc568/arh-AI/blob/arena/fde0d6d5-arh-ai/colab/GDN2_Lab.ipynb)

**GDN-2 Lab** — ноутбук и небольшая библиотека на чистом PyTorch, которые позволяют в
бесплатном Google Colab (T4 GPU **или** TPU):

| Что | Как |
| --- | --- |
| Задать модель | «хочу ~5M параметров» — солвер сам подберёт число слоёв/ширину, либо ручная архитектура |
| Данные | base-корпус + SFT-датасет: загрузка файлов, Google Drive, HuggingFace, URL, вставка текста |
| Обучение | from scratch → pretrain → SFT, в одной кнопке; 16 бит (FP16/BF16) или **нативные 8 бит** |
| Экспорт | **GGUF** (F16/Q8_0/Q4_0/…) и **LiteRT `.tflite`** под конкретный NPU, в пару кликов |
| Проверка | встроенный self-test, GGUF-runtime, LiteRT-runtime, отчёты совместимости |

Реализация сделана по статье [Gated DeltaNet-2: Decoupling Erase and Write in Linear
Attention](https://arxiv.org/abs/2605.22791v1) (arXiv:2605.22791) **независимо** от
официального кода NVIDIA (у того [NC-лицензия](NOTICE.md)). Triton-ядра не нужны:
одна и та же модель работает на CUDA, TPU (torch-xla) и CPU.

> ⚠️ **Честно про ограничения.** Рецепт статьи (1.3B параметров, 100B токенов, батч 0.5M)
> в бесплатном Colab не воспроизводится. Здесь — маленькие модели ≈1–80M параметров.
> llama.cpp пока не знает Gated Delta Rule-2, поэтому GGUF пишется с кастомной
> архитектурой `gdn2` и запускается [встроенным runtime](#gguf). LiteRT-граф
> конвертируется по-настоящему, но delegate NPU ставится на устройстве.
> Подробности: [docs/EXPORT.md](docs/EXPORT.md) и [docs/NPU.md](docs/NPU.md).

---

## 🚀 Быстрый старт

1. Откройте [ноутбук в Colab](https://colab.research.google.com/github/nc568/arh-AI/blob/arena/fde0d6d5-arh-ai/colab/GDN2_Lab.ipynb).
2. **Среда выполнения → Сменить тип → T4 GPU** (или **TPU**).
3. Запустите ячейку 1 (установка), ячейку 2 (`selftest`), ячейку 3 (форма).
4. В форме: выберите пресет, точность, источники данных.
5. `📚 Подготовить данные` → `🚀 Обучить ВСЁ` → `📤 Экспорт GGUF` / `📱 Экспорт LiteRT`.

Первый запуск можно сделать вообще без своих данных: в комплекте есть
`examples/toy_base.txt` и `examples/toy_sft.jsonl` (демо-режим).

Локально (без Colab):

```bash
git clone -b arena/fde0d6d5-arh-ai https://github.com/nc568/arh-AI.git
cd arh-AI
pip install -r requirements.txt
python -m gdn2lab.cli selftest

# Конфиг создаётся ноутбуком или берётся готовый пример:
mkdir -p /tmp/work && cp examples/config.example.json /tmp/work/config.json
python -m gdn2lab.cli --workdir /tmp/work info
python -m gdn2lab.cli --workdir /tmp/work prepare
python -m gdn2lab.cli --workdir /tmp/work train --phase all
python -m gdn2lab.cli --workdir /tmp/work chat --prompt "Привет!"
python -m gdn2lab.cli --workdir /tmp/work export-gguf --quant Q8_0
pip install -r requirements-litert.txt
python -m gdn2lab.cli --workdir /tmp/work export-litert --profile qualcomm_qnn_htp
```

---

## 🧮 Архитектура

Ядро — **Gated Delta Rule-2** (уравнение (10) статьи):

```
z_t = w_t ⊙ v_t,   e_t = b_t ⊙ k_t
S_t = (I − k_t e_tᵀ) · Diag(exp(g_t)) · S_{t−1} + k_t z_tᵀ
o_t = S_tᵀ q_t
```

* `g_t = −exp(a) ⊙ softplus(W_f x_t + δ)` — канальный логарифмический decay;
* `b_t` — **гейт стирания** (ключевая ось, по умолчанию [0,1], опционально [0,2]);
* `w_t` — **гейт записи** ( value-ось, [0,1]).
  При `b=w=β·1` получается KDA, при скалярном decay — обычная Gated DeltaNet.

Блок: `RMSNorm → GDN-2 mixer → MLP (SwiGLU)` (+ опционально sliding-window attention
через каждые N слоёв, как в гибридной модели статьи). q/k проходят causal short conv,
SiLU и L2-нормализацию; выход — `RMSNorm(o) ⊙ SiLU(gate)`.

**Обучение** идёт через чанковую WY-форму (уравнения (18)–(25)):

```
Ē = γ ⊙ (b ⊙ k),  Z = w ⊙ v,  T = tril(Ē K̄ᵀ, −1),  A = (I+T)⁻¹
Y = A Ē,  U = A Z,  O = Q_γ S + A_qk (U − Y S)
```

Всё считается в fp32 и в логарифмическом пространстве: `exp(G_r − G_s)` при `r≥s`
всегда ≤ 1, поэтому сильный decay не даёт `inf`/`NaN`. Реализация в
[`gdn2lab/recurrence.py`](gdn2lab/recurrence.py) проверяется на совпадение с
наивной токен-за-токеном рекурсией (вперёд, состояние **и градиенты**).

**Decode** — отдельный пошаговый путь с состоянием фиксированного размера
`[n_layer, batch, S]`: он численно совпадает с полным forward, но не растёт с длиной
контекста. Именно он экспортируется в LiteRT.

### Солвер размера

`python -m gdn2lab.cli solve --params 15000000 --vocab 8192`

Солвер перебирает `d_model = n_head·head_dim` и `n_layer`, считая **точное** число
параметров (совпадает с реальной моделью), и выбирает конфигурацию с лёгким приоритетом
в сторону ≈8 слоёв. Ограничение: при `vocab_size=8192` одни только эмбеддинги занимают
`8192·d_model`, поэтому модель меньше ~1M параметров требует уменьшить словарь.

| Пресет | Параметры | seq_len | Шагов pretrain | micro batch |
| --- | --- | --- | --- | --- |
| Пробный запуск | ~2M | 128 | 50 | 8 |
| Крошка | ~5M | 256 | 300 | 8 |
| Маленькая | ~15M | 256 | 800 | 4 |
| Средняя | ~40M | 512 | 1500 | 2 |
| Максимум T4 | ~80M | 512 | 2000 | 1 |

Это **демо-объёмы**: на 300 шагах модель не станет ChatGPT. Для осмысленного результата
нужны свои данные и тысячи шагов.

---

## 🔢 Нативные 8 бит

`precision = "int8"` — это **INT8 QAT**, а не просто «сохранить веса в int8»:

* веса квантуются **по выходному каналу**, активации — **по токену**, оба в signed int8;
* на CUDA/CPU прямой matmul — настоящий целочисленный GEMM
  (`torch._int_mm` → int32-аккумулятор, т.е. INT8 Tensor Cores на T4);
* обратный проход — straight-through estimator, **мастер-веса fp32**, чтобы мелкие
  обновления AdamW не терялись;
* оптимизатор — `bitsandbytes.AdamW8bit` (8-битные состояния, основная экономия VRAM);
* рекуррентность/decay/gates **всегда fp32** — квантование кумулятивного decay ломает
  долгую память;
* перед экспортом веса «складываются» (`fold_quantization`): получается обычная модель
  с ровно теми 8-битными значениями, которые использовались в обучении.

`precision = "fp8"` — FP8-e4m3 fake-quant (per-tensor scale). На T4 нет FP8 Tensor Cores,
поэтому это эмуляция; на Hopper/Ada можно заменить на torchao FP8 GEMM.
`precision = "fp16" | "bf16" | "fp32"` — обычный AMP (GradScaler для fp16).

| Устройство | Рекомендуемая точность | Комментарий |
| --- | --- | --- |
| Colab T4 (sm75) | `fp16` или `int8` | INT8 + FP16 Tensor Cores; BF16/FP8 compute отсутствует |
| Colab TPU v2/v3 | `bf16` | `xm.optimizer_step`, фиксированные формы; int8 GEMM эмулируется |
| CPU | `fp32` / `int8` | медленно, но всё работает; используется в self-test |

---

## 📚 Данные

**Base-корпус** (`.txt`, `.md`, `.jsonl`, `.json`, `.csv`, `.parquet`) токенизируется и
упаковывается в непрерывный поток токенов (`data/base_train.bin`, `base_val.bin`).

**SFT** (`.jsonl` / `.json` / `.csv`) — один из форматов:

```jsonl
{"messages":[{"role":"user","content":"Привет!"},{"role":"assistant","content":"Привет! Я GDN-2."}]}
{"instruction":"Сложи 2 и 2","input":"","output":"Четыре."}
{"prompt":"Кто ты?","completion":"Я небольшая модель."}
{"text":"произвольный текст без маскировки промпта"}
```

Шаблоны `chatml` / `alpaca` / `plain`. Лосс считается **только по ответу** (промпт
маскируется `-100`), слишком длинный промпт обрезается слева, чтобы ответ всегда помещался.

**Источники:** `demo` (встроенные примеры), `upload` (кнопка в ноутбуке), `path`
(файл/папка/glob, в т.ч. Google Drive), `hf` (HF `datasets` streaming), `url`, `paste`.
**Токенизатор:** `scratch` (свой byte-level BPE на корпусе), `hf` (готовый, например
`gpt2`), `byte` (256 байт + спецтокены). Фактический размер словаря сохраняется в
`config.json` после подготовки данных.

---

## 📦 Экспорт

### GGUF

```bash
python -m gdn2lab.cli --workdir /tmp/work export-gguf --quant Q8_0
python -m gdn2lab.cli --workdir /tmp/work chat --from-gguf /tmp/work/exports/*.gguf
```

Пишется **валидный GGUF v3** с архитектурой `gdn2`, полным `ModelConfig`, списком
тензоров, токенизатором и всеми весами. Доступные кванты: `F32`, `F16`, `BF16`, `Q8_0`,
`Q5_1`, `Q5_0`, `Q4_1`, `Q4_0`, `TQ2_0`, `MXFP4` (чистый `gguf` не умеет K-кванты —
`Q4_K_M` автоматически мапится в `Q4_0` с пометкой в отчёте). Тензоры, чей размер строки
не кратен блоку кванта, остаются в F16.

> llama.cpp поддерживает Gated DeltaNet для Qwen3-Next/3.5/3.8, но **не** новое правило
> GDN-2 с раздельными erase/write-гейтами. Поэтому запускать файл нужно через
> `gdn2lab.gguf_runtime` (деквантизация + та же модель на PyTorch), а для llama.cpp
> потребуется отдельный backend. Метаданные `gdn2.tensor_names` + `gdn2.update_rule`
> содержат всё необходимое для такого патча. Это не «универсальный конвертер в GGUF».

### LiteRT / NPU

```bash
pip install -r requirements-litert.txt
python -m gdn2lab.cli --workdir /tmp/work export-litert --profile qualcomm_qnn_htp
python -m gdn2lab.cli --workdir /tmp/work chat --from-litert /tmp/work/exports/*.tflite
```

Экспортируется **пошаговый decode-граф** с фиксированными формами:

```
(tokens[1,1] int32, state[L,1,S] float32) → (logits[1,1,V], state[L,1,S])
```

Опционально добавляется `prefill`-подпись фиксированной длины. Граф содержит только
`ADD/MUL/SUM/EXP/LOG/LOGISTIC/RSQRT/GATHER/FULLY_CONNECTED/SLICE/RESHAPE/...` —
без `CUMSUM` и `BATCH_MATMUL` внутри рекурсии (специально для NPU).

Пайплайн проверен на `litert-torch 0.10` + `ai-edge-quantizer 0.10`:

1. `torch.export` → `litert_torch.convert` → float32 `.tflite`;
2. post-training квантование: `dynamic_int8` (int8 веса, fp32 активации),
   `fp16` (weight-only FLOAT_CASTING), `full_int8` (статическая калибровка на
   **реальных рекуррентных состояниях**), либо `float32`;
3. запуск через `ai_edge_litert.Interpreter` (smoke-test), отчёт по операциям.

Профили: `generic_cpu`, `gpu_fp16`, `qualcomm_qnn_htp`, `mediatek_neuropilot`,
`google_edgetpu`, `npu_auto`. Для каждого генерируются `*.report.json`
(ops, risky ops, размер, SHA, delegate-сниппет) и `*.signature.json` (имена входа/выхода,
форма состояния, токенизатор). См. [docs/NPU.md](docs/NPU.md).

> Гибридные SWA-слои (`hybrid_every > 0`) не экспортируются в decode-step: для NPU
> используйте чисто рекуррентную модель (значение по умолчанию).

---

## 🧪 Проверки

```bash
python -m gdn2lab.cli selftest     # ~5 с на CPU
pytest -q                          # те же проверки через pytest
```

Проверяется: совпадение чанковой и наивной рекуррентности (forward **и** gradients),
устойчивость при сильном decay, точный подсчёт параметров, солвер, равенство decode-step
и полного forward, INT8 QAT + fold, `torch.export` decode-графа, GGUF roundtrip.
Локально также были выполнены реальные `litert_torch.convert` + dynamic int8 / fp16 /
full int8 квантования и генерация из `.tflite`.

---

## 🗂 Структура

```
colab/GDN2_Lab.ipynb        — ноутбук с формой (RU) и запасным режимом # @param
tools/build_notebook.py     — генератор ноутбука
gdn2lab/
  config.py                 — конфиги, солвер архитектуры, NPU-профили
  recurrence.py             — GDN-2: чанковая WY-форма, наивная рекурсия, step
  model.py                  — RMSNorm, short conv, mixer, SWA, MLP, GDN2LM
  quant.py                  — INT8/FP8 QAT, int8 GEMM, fold, AdamW8bit
  device.py                 — CUDA / TPU / CPU, AMP, GradScaler, auto micro-batch
  data.py                   — источники, BPE-токенизатор, packing, SFT-маски
  trainer.py                — pretrain + SFT, чекпоинты, логи, сэмплинг
  export_gguf.py            — GGUF v3 + метаданные + верификация
  gguf_runtime.py           — загрузка GGUF обратно в PyTorch, генерация
  export_litert.py          — decode-граф, litert-torch, PTQ, отчёты
  litert_runtime.py         — запуск .tflite (CPU / delegate), сэмплинг
  cli.py / selftest.py      — CLI и self-test
docs/EXPORT.md, docs/NPU.md — подробности экспорта
examples/                   — демо base-корпус и SFT-датасет
```

---

## ⚖️ Лицензия

Код — MIT, см. [LICENSE](LICENSE). Сторонние материалы и дисклеймеры — [NOTICE.md](NOTICE.md).
Официальный код NVIDIA (NC-лицензия) не включён и не копируется.
