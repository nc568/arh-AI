# Экспорт: GGUF и LiteRT

Оба экспортера работают с одним и тем же чекпоинтом
`WORKDIR/checkpoints/model.safetensors` и конфигом `WORKDIR/checkpoints/config.json`.
`model.safetensors` уже содержит **свёрнутые** 8-битные веса, если обучение шло в
`int8`/`fp8` — поэтому экспорт не требует никаких scale-тензоров.

```bash
python -m gdn2lab.cli --workdir /tmp/work export-gguf --quant Q8_0
python -m gdn2lab.cli --workdir /tmp/work export-litert --profile qualcomm_qnn_htp
```

Результаты: `WORKDIR/exports/*.gguf` / `*.tflite` плюс `*.report.json` и
`*.signature.json`. Рядом копируется токенизатор.

---

## 1. GGUF

### Что внутри

Контейнер GGUF v3 (`pip install gguf`), `general.architecture = "gdn2"`.

| Ключ | Смысл |
| --- | --- |
| `gdn2.version` | версия формата (=1) |
| `gdn2.config_json` | полный `ModelConfig` (JSON) — позволяет собрать модель заново |
| `gdn2.tensor_names` | порядок всех тензоров |
| `gdn2.block_count`, `gdn2.head_count`, `gdn2.head_dim`, `gdn2.chunk_size`, `gdn2.conv_size`, `gdn2.hybrid_every` | архитектура |
| `gdn2.update_rule` | формула обновления состояния (шпаргалка для разработчика runtime) |
| `gdn2.tokenizer_info_json`, `gdn2.tokenizer_json` | токенизатор целиком внутри файла |
| `general.context_length`, `general.embedding_length`, `general.file_type` | стандартные поля |

Имена тензоров совпадают с `state_dict()` PyTorch-модели, например:

```
embed.weight
blocks.0.norm1.weight
blocks.0.mixer.q_proj.weight
blocks.0.mixer.k_proj.weight
blocks.0.mixer.v_proj.weight
blocks.0.mixer.b_proj.weight      # erase gate
blocks.0.mixer.w_proj.weight      # write gate
blocks.0.mixer.f_proj.0.weight    # low-rank decay
blocks.0.mixer.f_proj.1.weight
blocks.0.mixer.g_proj.0.weight    # output gate
blocks.0.mixer.g_proj.1.weight / .bias
blocks.0.mixer.o_norm.weight
blocks.0.mixer.o_proj.weight
blocks.0.mixer.q_conv.weight / k_conv.weight / v_conv.weight
blocks.0.mixer.A_log / dt_bias
blocks.0.norm2.weight, blocks.0.mlp.w1/w2/w3.weight
norm.weight, head.weight
```

При `tie_embeddings=true` `head.weight` повторяет `embed.weight` (в файл пишутся оба).

### Квантование

`--quant` принимает: `F32`, `F16`, `BF16`, `Q8_0`, `Q5_1`, `Q5_0`, `Q4_1`, `Q4_0`,
`TQ2_0`, `MXFP4`. Это типы, которые умеет **python-пакет `gguf`**. K-кванты
(`Q4_K_M`, `Q6_K`, …) в нём не реализованы — они автоматически мапятся в
`Q4_0`/`Q5_1`, и это явно написано в `*.report.json`.

Если последняя размерность тензора не кратна блоку кванта (например `A_log` длиной
`n_head`), такой тензор остаётся в F16 — опять же с пометкой в отчёте. У нас
`d_model` кратен 64, поэтому все большие матрицы квантуются нормально.

Рекомендация: для модели, обученной в `int8`, используйте `Q8_0` — веса останутся
8-битными без дополнительной потери точности.

### Верификация

После записи файл читается обратно через `GGUFReader`, сравнивается число тензоров и
значения первых трёх (деквантованных) с исходными. В `*.report.json` лежат
`verify_max_abs_error`, размер, число квантованных тензоров.

### ⚠️ Про llama.cpp

llama.cpp умеет Gated DeltaNet (Qwen3-Next / 3.5 / 3.8, GATED_LINEAR_ATTN, Vulkan/HMX),
но **не умеет GDN-2**: у него два разных канальных гейта (`b_t` для стирания по ключу и
`w_t` для записи в value) вместо одного `β_t`. Поэтому:

* файл **корректен как GGUF**, но `llama-cli` скажет «unknown architecture gdn2»;
* запускайте его встроенным runtime:

```bash
python -m gdn2lab.cli --workdir /tmp/work chat --from-gguf /tmp/work/exports/*.gguf
```

```python
from gdn2lab.gguf_runtime import load_gguf, generate_text
model, tok = load_gguf("model.gguf", device="cpu")
print(generate_text(model, tok, "Привет!", max_new_tokens=100))
```

* чтобы добавить поддержку в llama.cpp, нужен новый оператор под правило
  `S_t = (I - k_t (b_t⊙k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t⊙v_t)^T`:
  карта тензоров есть в `gdn2.tensor_names`, формула — в `gdn2.update_rule`,
  а эталонная реализация decode-шага — в `gdn2lab/recurrence.py::gdn2_step`.
  Простой «связать b=w» недостаточно: это превратит GDN-2 обратно в KDA.

---

## 2. LiteRT (`.tflite`)

### Граф

Основная подпись — `decode` (или `serving_default`, если конвертер не сохранил имя):

```
tokens : int32  [1, 1]
state  : float32 [n_layer, 1, S]
  ↓
logits : float32 [1, 1, vocab_size]
state  : float32 [n_layer, 1, S]
```

`S` — размер состояния **одного** слоя:

```
S = n_v_heads · head_dim · head_v_dim
  + (conv_size-1) · (2·key_dim + value_dim)     # если use_short_conv
```

Внутри состояния лежат (в таком порядке): рекуррентная матрица `S_l` формы
`[n_v_heads, head_dim, head_v_dim]`, затем три сдвиговых регистра short-conv для
q/k/v. Порядок фиксирован в `GDN2LM.state_shapes()`; `initial_state()` создаёт
нулевое состояние, `step()` обновляет его.

Опция `--export-prefill --prefill-len 32` добавляет вторую подпись `prefill`:
`tokens[1,T] + state → logits[1,T,V] + state`. Это ускоряет обработку промпта на
устройстве, но граф получается в T раз больше (конвертация занимает минуты).

### Пайплайн

```
GDN2LM (fp32, eval, CPU)
   └─ torch.export.export(DecodeWrapper)          # проверка экспортируемости
        └─ litert_torch.convert(...)              # StableHLO → float32 .tflite
             └─ ai_edge_quantizer.Quantizer       # PTQ (см. ниже)
                  └─ ai_edge_litert.Interpreter   # smoke-test + список ops
```

Поддерживаются оба имени конвертера: новое `litert_torch` (0.10+) и старое
`ai_edge_torch` (0.7 — это deprecation-stub). Проверено на `litert-torch 0.10.0`,
`ai-edge-quantizer 0.10`, `torch 2.14`.

### Режимы квантования

| `--quant` | Что делает | Входы/выходы |
| --- | --- | --- |
| `float32` | без квантования | fp32 |
| `fp16` | weight-only FLOAT_CASTING (16 бит), вычисления fp32 | fp32 |
| `dynamic_int8` | int8 веса (channel-wise), fp32 активации | fp32 |
| `full_int8` | статическая int8-калибровка на реальных состояниях модели | int8 (state/logits), int32 tokens |

`full_int8` нужен для Edge TPU. Если калибровка не удалась (например, XNNPACK ругается
на reshape), экспортёр честно откатывается на `dynamic_int8` и пишет причину в отчёт.
Для Edge TPU после этого всё равно обязателен `edgetpu_compiler`.

### Отчёты

`*.report.json`:

* `quant` — что реально применилось (может отличаться от запрошенного);
* `smoke_test` — результат реального запуска `.tflite` через Interpreter и
  `max_abs_diff` логитов против PyTorch;
* `ops` — список операций графа, `risky_ops` (сверено с профилем NPU),
  точные типы/формы/квантизация входов и выходов;
* `delegate_snippet` — готовый код подключения delegate;
* `notes` — предупреждения профиля + предупреждения конвертации.

Реальный список операций для 2-слойной модели (dynamic int8):
`ADD, BROADCAST_TO, CONCATENATION, DELEGATE, EMBEDDING_LOOKUP, EXP, FULLY_CONNECTED,
LOG, LOGISTIC, MAXIMUM, MEAN, MINIMUM, MUL, RESHAPE, RSQRT, SLICE, SUB, SUM` —
нет `CUMSUM`, `BATCH_MATMUL`, `SOFTPLUS`, `SILU` (мы намеренно разложили SiLU/softplus
на `MUL+LOGISTIC` и `LOG(1+EXP)`).

`*.signature.json` — всё, что нужно runtime: имена входов/выходов каждой подписи,
`state_shape`, `state_dtype`, `vocab_size`, id спецтокенов, путь к токенизатору.

### Запуск

```bash
python -m gdn2lab.cli --workdir /tmp/work chat --from-litert /tmp/work/exports/*.tflite
```

```python
from gdn2lab.data import LabTokenizer
from gdn2lab.litert_runtime import LiteRTGenerator
from ai_edge_litert.interpreter import load_delegate

tok = LabTokenizer.load("/tmp/work/data")
delegate = load_delegate("libQnnTFLiteDelegate.so",
                         options={"backend_type": "htp"})
gen = LiteRTGenerator("model.tflite", "model.signature.json", tok,
                      delegates=[delegate])
print(gen.generate("Привет!", max_new_tokens=80))
```

`LiteRTGenerator` сам определяет имена входов/выходов, квантует/деквантует
входы-выходы по `quantization` из интерпретатора, выполняет top-k/top-p сэмплинг
и пошагово обновляет состояние.

---

## 3. Чек-лист перед деплоем

1. `python -m gdn2lab.cli selftest` — зелёный.
2. `*.report.json → smoke_test.passed == true`.
3. `risky_ops` сверен с вашим NPU (см. [NPU.md](NPU.md)).
4. `state_shape` в `signature.json` совпадает с тем, что выделяет ваш код.
5. Токенизатор (`tokenizer.json`) лежит рядом с моделью и имеет тот же
   `vocab_size`, что и `ModelConfig`.
6. Для Edge TPU — выполнен `edgetpu_compiler -s model_full_int8.tflite`.
