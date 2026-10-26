# Профили NPU и делегаты

Профиль выбирается в ноутбуке (вкладка «📦 Экспорт») или через
`--profile`. Он определяет тип квантования, длину prefill и текст подсказки по
delegate. **Профиль не гарантирует 100% делегирование**: итоговое решение принимает
runtime на устройстве. Всегда читайте `*.report.json → ops.risky_ops`.

| Профиль | Квантование по умолчанию | Delegate | Особенности |
| --- | --- | --- | --- |
| `generic_cpu` | `dynamic_int8` | не нужен (XNNPACK) | самый совместимый; prefill 64 |
| `gpu_fp16` | `fp16` (weight-only) | `GpuDelegate` | FP16-веса, fp32-вычисления; prefill 64 |
| `qualcomm_qnn_htp` | `dynamic_int8` | `libQnnTFLiteDelegate.so`, `backend_type=htp` | статичные формы обязательны; нужен QNN SDK; prefill 32 |
| `mediatek_neuropilot` | `dynamic_int8` | `libneuron_delegate.so` (Neuron) | имя `.so`/опции зависят от версии SDK; prefill 32 |
| `google_edgetpu` | `full_int8` | `libedgetpu.so.1` + `edgetpu_compiler` | только full-integer int8; EXP/LOG/RSQRT уйдут на CPU; prefill 16 |
| `npu_auto` | `dynamic_int8` | ваш | консервативный вариант для неизвестного NPU |

Пример переопределения:

```bash
python -m gdn2lab.cli --workdir /tmp/work export-litert \
    --profile qualcomm_qnn_htp --quant full_int8 --export-prefill --prefill-len 32
```

---

## Какие операции реально получаются

Реальный запуск на 2-слойной модели (`dynamic_int8`, 292 узла):

```
ADD, BROADCAST_TO, CONCATENATION, DELEGATE, EMBEDDING_LOOKUP, EXP,
FULLY_CONNECTED, LOG, LOGISTIC, MAXIMUM, MEAN, MINIMUM, MUL, RESHAPE,
RSQRT, SLICE, SUB, SUM
```

Почему так:

* `SOFTPLUS` заменён на `LOG(1 + EXP(clamp(x)))`;
* `SILU` заменён на `x * SIGMOID(x)` → `MUL + LOGISTIC`;
* рекуррентный шаг использует `MUL + SUM` вместо `einsum`/`BATCH_MATMUL`;
* `cumsum` есть только в **чанковой** обучающей реализации — в decode-графе его нет;
* `DELEGATE` — это технические узлы `ai-edge-quantizer` (dequantize), не ошибка.

**Потенциально проблемные для NPU:** `EXP`, `LOG`, `RSQRT` (трансцендентные функции).
Большинство современных NPU (HTP, Neuron) их либо умеют, либо выполняют на CPU с
маленькими тензорами. На Edge TPU ожидайте частичного делегирования.

`risky_ops` в отчёте считается **точным** совпадением имени операции (не подстрокой),
поэтому `BROADCAST_TO` больше не путается с `CAST`.

---

## Qualcomm QNN / HTP

```python
from ai_edge_litert.interpreter import Interpreter, load_delegate
from gdn2lab.litert_runtime import LiteRTGenerator
from gdn2lab.data import LabTokenizer

qnn = load_delegate("libQnnTFLiteDelegate.so", options={
    "backend_type": "htp",
    "htp_performance_mode": "burst",      # или "sustained_high_performance"
    "htp_precision": "default",           # "f16" / "int8"
    "htp_graph_finalization_optimization_mode": "3",
})
gen = LiteRTGenerator("model.tflite", "model.signature.json",
                      LabTokenizer.load("..."), delegates=[qnn])
print(gen.generate("Привет!"))
```

Требуются `libQnnTFLiteDelegate.so`, `libQnnHtp.so`, `libQnnHtpV*` skel и
`libQnnSystem.so` из **Qualcomm AI Engine Direct (QNN) SDK**, а также разрешение
`android.permission.READ_EXTERNAL_STORAGE`/доступ к `/vendor/lib/rfsa/adsp/`.
В Colab delegate не запустить — только экспорт и CPU smoke-test.

Полезно: `pip install "ai-edge-litert[npu-sdk]"` подтягивает SDK-пакеты, но на
устройстве всё равно нужны библиотеки вендора.

---

## MediaTek NeuroPilot / Neuron

```python
neuron = load_delegate("libneuron_delegate.so", options={
    "allowed_builtin_code": 0,          # 0 = все операции
    "inference_preference": 2,          # prefer sustained speed
})
```

Имя `.so` и опции отличаются между NeuroPilot 7/8/9 — сверьтесь с документацией
вашего SDK. Если delegate падает на `EXP`/`RSQRT`, попробуйте `--quant dynamic_int8`
(вместо `full_int8`) или запустите модель без delegate и посмотрите лог
`Interpreter` — неделегированные подграфы выполнятся на CPU.

---

## Google Edge TPU (Coral)

`full_int8` даёт модель с int8-входами/выходами, но **этого недостаточно**:

```bash
edgetpu_compiler -s model-google_edgetpu-full_int8.tflite
# → model-google_edgetpu-full_int8_edgetpu.tflite + .log с числом операций на TPU
```

Edge TPU поддерживает фиксированный набор операций (в основном свёртки и
полносвязные слои). Наши `EXP`, `LOG`, `RSQRT`, `MEAN`, `SLICE` почти наверняка
останутся на CPU, поэтому модель запустится, но ускорение будет частичным.
В логе компилятора смотрите `Number of operations that will run on Edge TPU`.

Калибровка `full_int8` в экспортёре выполняется на **реальных рекуррентных
состояниях** (мы прогоняем модель на нескольких токенах), а не на случайном шуме —
это важно, иначе scale для состояния будет неверным.

---

## Если delegate не подхватил граф

1. Проверьте `report.json`: `static_shapes`, `quant`, `smoke_test.passed`.
2. Убедитесь, что форма `state` в точности `[n_layer, 1, S]` и dtype float32
   (для full_int8 — int8 с scale из `input_details`).
3. Попробуйте `--quant dynamic_int8` и убрать prefill (`--export-prefill` не нужен).
4. Включите подробный лог delegate (`ADSP_LOG_LEVEL`, `NEURON_DEBUG`,
   `TFLITE_LOG_LEVEL`) — обычно там прямо написано, какой op «not supported».
5. Если NPU не умеет `EXP` — единственный радикальный вариант: заменить
   `softplus`/`exp(decay)` на табличную аппроксимацию в `gdn2lab/model.py` и
   переэкспортировать. Мы оставили точные функции ради числовой корректности.

---

## Ссылки

* [LiteRT Torch (PyTorch converter)](https://github.com/google-ai-edge/litert-torch)
* [PyTorch → LiteRT quickstart](https://ai.google.dev/edge/litert/conversion/pytorch/overview)
* [AI Edge Quantizer](https://github.com/google-ai-edge/ai-edge-torch/tree/main/ai_edge_quantizer)
* [LiteRT runtime / delegates](https://github.com/google-ai-edge/LiteRT)
