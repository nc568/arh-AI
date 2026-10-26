"""Configuration objects, architecture solver and deployment profiles.

Everything is a plain dataclass serialised to JSON, so the Colab UI, the CLI
and the exported model all share exactly one source of truth.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# NPU / deployment profiles for LiteRT export
# --------------------------------------------------------------------------- #

NPU_PROFILES: dict[str, dict[str, Any]] = {
    "generic_cpu": {
        "label": "Универсально (CPU / XNNPACK / NNAPI)",
        "quant": "dynamic_int8",
        "static_shapes": True,
        "prefill_len": 64,
        "delegate": "# delegates not required; XNNPACK is used automatically",
        "notes": [
            "Самый совместимый профиль: float/int8-веса, динамическая квантизация.",
            "Работает везде, но без ускорения NPU.",
        ],
        "risky_ops": [],
    },
    "gpu_fp16": {
        "label": "Мобильный GPU (FP16 delegate)",
        "quant": "fp16",
        "static_shapes": True,
        "prefill_len": 64,
        "delegate": "interpreter = Interpreter(path, experimental_delegates=[GpuDelegate()])",
        "notes": ["FP16-веса, float32-вычисления на GPU.", "Хороший вариант, если NPU недоступен."],
        "risky_ops": [],
    },
    "qualcomm_qnn_htp": {
        "label": "Qualcomm Snapdragon (QNN / HTP NPU)",
        "quant": "dynamic_int8",
        "static_shapes": True,
        "prefill_len": 32,
        "delegate": (
            "from ai_edge_litert.interpreter import Interpreter, load_delegate\n"
            "qnn = load_delegate('libQnnTFLiteDelegate.so', options={\n"
            "    'backend_type': 'htp', 'htp_performance_mode': 'burst',\n"
            "    'htp_precision': 'default',\n"
            "})\n"
            "interpreter = Interpreter(model_path, experimental_delegates=[qnn])"
        ),
        "notes": [
            "HTP быстрее всего на int8: используйте dynamic_int8 или full_int8.",
            "Фиксированные формы (batch=1, seq=1 у decode-подписи) обязательны.",
            "Нужны библиотеки QNN из Snapdragon Neural Processing Engine SDK на устройстве.",
        ],
        "risky_ops": ["CUMSUM", "CAST"],
    },
    "mediatek_neuropilot": {
        "label": "MediaTek Dimensity (NeuroPilot / Neuron)",
        "quant": "dynamic_int8",
        "static_shapes": True,
        "prefill_len": 32,
        "delegate": (
            "from ai_edge_litert.interpreter import Interpreter, load_delegate\n"
            "neuron = load_delegate('libneuron_delegate.so',\n"
            "                       options={'allowed_builtin_code': 0, 'inference_preference': 2})\n"
            "interpreter = Interpreter(model_path, experimental_delegates=[neuron])"
        ),
        "notes": [
            "NeuroPilot предпочитает int8 и статичные формы.",
            "Имя .so и опции зависят от версии NeuroPilot SDK — уточните у вендора.",
        ],
        "risky_ops": ["CUMSUM", "CAST"],
    },
    "google_edgetpu": {
        "label": "Google Edge TPU (Coral)",
        "quant": "full_int8",
        "static_shapes": True,
        "prefill_len": 16,
        "delegate": (
            "# required extra step on the DEVICE / with edgetpu_compiler:\n"
            "edgetpu_compiler -s your_model_full_int8.tflite\n"
            "interpreter = Interpreter('your_model_full_int8_edgetpu.tflite',\n"
            "                          experimental_delegates=[load_delegate('libedgetpu.so.1')])"
        ),
        "notes": [
            "Edge TPU принимает ТОЛЬКО full-integer int8 модель (входы/выходы int8).",
            "После экспорта обязателен edgetpu_compiler; неподдерживаемые операции уйдут на CPU.",
            "Рекуррентный GDN-2 содержит EXP/SIGMOID — ожидайте частичного делегирования.",
        ],
        "risky_ops": ["EXP", "SIGMOID", "SOFTPLUS", "RSQRT", "CUMSUM"],
    },
    "npu_auto": {
        "label": "Другой NPU (универсальный int8)",
        "quant": "dynamic_int8",
        "static_shapes": True,
        "prefill_len": 32,
        "delegate": "# подставьте delegate вашего вендора в litert_runtime.py",
        "notes": [
            "Консервативный int8-профиль: статичные формы, минимум экзотических операций.",
            "Проверьте npu_report.json на предмет операций, которые ваш NPU не умеет.",
        ],
        "risky_ops": ["CUMSUM", "EXP"],
    },
}

GGUF_QUANTS: dict[str, str] = {
    "F32": "без квантования, 32 бита",
    "F16": "без квантования, 16 бит (по умолчанию)",
    "BF16": "bfloat16",
    "Q8_0": "8 бит — лучший выбор для int8-обученной модели",
    "Q5_1": "5 бит + scale/min",
    "Q5_0": "5 бит + scale",
    "Q4_1": "4 бита + scale/min",
    "Q4_0": "4 бита + scale",
    "TQ2_0": "~2 бита, тернарный",
    "MXFP4": "4-битный MXFP4 (экспериментально)",
}


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 256
    n_layer: int = 4
    n_head: int = 4
    head_dim: int = 64
    expand_v: float = 1.0
    n_v_heads: int = 0  # 0 => n_head
    mlp_ratio: float = 2.0
    intermediate_size: int = 0  # 0 => round64(mlp_ratio * d_model)
    seq_len: int = 256
    chunk_size: int = 16
    decay_block: int = 16
    use_short_conv: bool = True
    conv_size: int = 4
    hybrid_every: int = 0  # 0 => pure recurrent (best for NPU export)
    swa_window: int = 128
    allow_neg_eigval: bool = False
    tie_embeddings: bool = True
    dropout: float = 0.0
    norm_eps: float = 1e-5
    max_log_decay: float = 8.0  # clamp |g_t| per token; 0 disables clamping

    @property
    def head_v_dim(self) -> int:
        return max(1, int(round(self.head_dim * self.expand_v)))

    @property
    def key_dim(self) -> int:
        return self.n_head * self.head_dim

    @property
    def value_dim(self) -> int:
        return (self.n_v_heads or self.n_head) * self.head_v_dim

    @property
    def n_v_heads_eff(self) -> int:
        return self.n_v_heads or self.n_head

    @property
    def hidden_ffn(self) -> int:
        return self.intermediate_size or _round64(self.mlp_ratio * self.d_model)

    @property
    def swa_layers(self) -> list[int]:
        if self.hybrid_every <= 0:
            return []
        return [i for i in range(self.n_layer - 1, -1, -self.hybrid_every)]

    def validate(self) -> list[str]:
        w: list[str] = []
        if self.d_model % 8:
            w.append("d_model желательно кратно 64 (квантование GGUF / Tensor Cores).")
        if self.chunk_size < 2 or self.chunk_size > 128:
            w.append("chunk_size должен быть от 2 до 128 (обычно 16 или 32).")
        if self.seq_len % self.chunk_size:
            w.append("seq_len не кратен chunk_size — будет дополнен паддингом.")
        if self.n_v_heads and self.n_v_heads % self.n_head:
            w.append("n_v_heads должен быть кратен n_head (GVA).")
        if self.head_v_dim != self.head_dim * self.expand_v:
            w.append("expand_v округлён до целого head_v_dim.")
        if self.hybrid_every > 0:
            w.append("Гибридные SWA-слои нельзя экспортировать в LiteRT decode-step: выключите hybrid_every для NPU.")
        return w


@dataclass
class DataConfig:
    # source dicts: {"kind": "demo"|"upload"|"path"|"hf"|"url"|"paste", ...}
    base_source: dict[str, Any] = field(default_factory=lambda: {"kind": "demo"})
    sft_source: dict[str, Any] = field(default_factory=lambda: {"kind": "demo"})
    tokenizer: dict[str, Any] = field(
        default_factory=lambda: {"kind": "scratch", "vocab_size": 8192}
    )
    text_field: str = "text"
    val_fraction: float = 0.02
    max_base_tokens: int = 5_000_000
    sft_template: str = "chatml"  # chatml | alpaca | plain
    sft_max_seq_len: int = 0  # 0 => model.seq_len
    pack_base: bool = True
    seed: int = 42
    out_dir: str = "data"


@dataclass
class TrainConfig:
    precision: str = "fp16"  # fp16 | bf16 | fp32 | int8 | fp8
    int8_mode: str = "int8"  # int8 | fp8_e4m3 (used when precision == int8/fp8)
    int8_gemm: bool = True  # use real torch._int_mm when available
    quantize_embeddings: bool = False
    quantize_lm_head: bool = False
    optimizer: str = "adamw8bit"  # adamw | adamw8bit | sgd
    micro_batch_size: int = 0  # 0 => auto-probe
    grad_accum: int = 4
    base_steps: int = 200
    base_lr: float = 3e-3
    sft_epochs: int = 3
    sft_lr: float = 1e-3
    warmup_ratio: float = 0.05
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    scheduler: str = "cosine"  # cosine | linear | constant
    min_lr_ratio: float = 0.1
    eval_every: int = 50
    log_every: int = 10
    save_every: int = 0
    sample_every: int = 0
    seed: int = 42
    resume: bool = False
    device: str = "auto"  # auto | cuda | tpu | cpu
    use_all_tpu_cores: bool = False
    wandb: bool = False
    out_dir: str = "checkpoints"


@dataclass
class ExportConfig:
    gguf_quant: str = "Q8_0"
    litert_profile: str = "qualcomm_qnn_htp"
    litert_quant: str = "auto"  # auto | float32 | fp16 | dynamic_int8 | full_int8
    prefill_len: int = 0  # 0 => from profile
    export_prefill: bool = False  # decode-step only is the most NPU-compatible
    out_dir: str = "exports"


@dataclass
class LabConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    run_name: str = "gdn2-tiny"
    workdir: str = "/content/work"  # Colab default; overridden by CLI

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path, indent: int = 2) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=indent, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LabConfig":
        d = copy.deepcopy(d)

        def build(klass, raw):
            if not isinstance(raw, dict):
                raw = {}
            names = {f.name for f in fields(klass)}
            return klass(**{k: v for k, v in raw.items() if k in names})

        cfg = cls(
            model=build(ModelConfig, d.get("model", {})),
            data=build(DataConfig, d.get("data", {})),
            train=build(TrainConfig, d.get("train", {})),
            export=build(ExportConfig, d.get("export", {})),
            run_name=d.get("run_name", "gdn2-tiny"),
            workdir=d.get("workdir", "/content/work"),
        )
        # keep the tokenizer vocab in sync with the model vocab
        if cfg.data.tokenizer.get("kind") == "scratch":
            cfg.data.tokenizer["vocab_size"] = cfg.model.vocab_size
        return cfg

    @classmethod
    def from_json(cls, path: str | Path) -> "LabConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def path(self, *parts: str) -> Path:
        return Path(self.workdir).joinpath(*parts)


# --------------------------------------------------------------------------- #
# Parameter counting and architecture solver
# --------------------------------------------------------------------------- #


def _round64(x: float) -> int:
    return max(64, int(math.ceil(float(x) / 64.0) * 64))


def _lin(i: int, o: int, bias: bool = False) -> int:
    return i * o + (o if bias else 0)


def count_params(c: ModelConfig) -> int:
    """Exact parameter count of :class:`gdn2lab.model.GDN2LM`."""
    d, k, v = c.d_model, c.key_dim, c.value_dim
    hv = c.head_v_dim
    ff = c.hidden_ffn
    total = 0

    # embeddings / output head
    total += c.vocab_size * d
    if not c.tie_embeddings:
        total += _lin(d, c.vocab_size)

    for layer in range(c.n_layer):
        # GDN-2 token mixer
        total += _lin(d, k) * 2                    # q_proj, k_proj
        total += _lin(d, v)                       # v_proj
        total += _lin(d, k)                       # b_proj (erase gate)
        total += _lin(d, v)                       # w_proj (write gate)
        total += _lin(d, hv) + _lin(hv, k)        # f_proj (low-rank decay)
        total += _lin(d, hv) + _lin(hv, v, True)  # g_proj (output gate)
        total += _lin(v, d)                       # o_proj
        total += hv                               # o_norm weight
        total += c.n_head + k                     # A_log + dt_bias
        if c.use_short_conv:
            total += c.conv_size * (2 * k + v)    # q/k/v depthwise convs
        total += d                                # norm1
        # SwiGLU MLP
        total += _lin(d, ff) * 2 + _lin(ff, d)
        total += d                                # norm2
        # optional sliding-window attention (hybrid)
        if layer in c.swa_layers:
            total += 4 * _lin(d, d) + d           # q,k,v,o + norm3

    total += d  # final norm
    return int(total)


def state_size(c: ModelConfig) -> int:
    """Number of recurrent state floats per sequence (decode cache)."""
    per_layer = c.n_head * c.head_dim * c.head_v_dim
    if c.use_short_conv:
        per_layer += (c.conv_size - 1) * (2 * c.key_dim + c.value_dim)
    return int(c.n_layer * per_layer)


def solve_architecture(
    target_params: int,
    vocab_size: int = 8192,
    head_dim: int = 64,
    mlp_ratio: float = 2.0,
    expand_v: float = 1.0,
    seq_len: int = 256,
    max_layers: int = 48,
    max_heads: int = 32,
    hybrid_every: int = 0,
    tie_embeddings: bool = True,
) -> ModelConfig:
    """Pick (n_layer, d_model) whose exact parameter count is closest to target.

    d_model = n_head * head_dim, so every candidate is a multiple of head_dim.
    A mild shape prior prefers reasonably deep models instead of one huge layer.
    """
    target_params = int(max(20_000, target_params))
    best: tuple[float, ModelConfig] | None = None
    for n_head in range(2, max_heads + 1):
        d_model = n_head * head_dim
        for n_layer in range(1, max_layers + 1):
            c = ModelConfig(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layer=n_layer,
                n_head=n_head,
                head_dim=head_dim,
                expand_v=expand_v,
                mlp_ratio=mlp_ratio,
                seq_len=seq_len,
                hybrid_every=hybrid_every,
                tie_embeddings=tie_embeddings,
            )
            p = count_params(c)
            err = abs(math.log(p / target_params))
            # Mild depth prior: small LMs work better with ~8 layers than with
            # one giant layer at the same parameter count.
            shape_penalty = 0.08 * abs(math.log(n_layer / 8.0))
            score = err + shape_penalty
            if best is None or score < best[0]:
                best = (score, c)
    assert best is not None
    return best[1]


def estimate_memory(
    c: ModelConfig,
    batch: int,
    precision: str,
    optimizer: str = "adamw8bit",
    device_mem_gb: float = 15.0,
) -> dict[str, Any]:
    """Rough training-memory estimate (GB). Deliberately conservative."""
    p = count_params(c)
    master_bytes = 4 if precision in ("fp32", "int8", "fp8") else 4  # master weights are fp32
    amp_bytes = 2 if precision in ("fp16", "bf16") else 0
    opt_bytes = 2 if optimizer == "adamw8bit" else (8 if optimizer == "adamw" else 0)
    weights = p * (master_bytes + amp_bytes + opt_bytes + 4) / 1e9  # + grads
    # dominant activations: chunkwise ratio matrices + per-layer hidden states
    b, t, d = batch, c.seq_len, c.d_model
    ratio = c.n_head * c.chunk_size * c.chunk_size * min(c.decay_block, c.head_dim)
    acts = b * t * (c.n_layer * d * 14 + c.vocab_size * 2 + ratio) * 2 / 1e9
    total = weights + acts + 0.6  # CUDA context / fragmentation
    return {
        "params": p,
        "weights_gb": round(weights, 3),
        "activations_gb": round(acts, 3),
        "estimated_gb": round(total, 3),
        "device_mem_gb": device_mem_gb,
        "fits": total < device_mem_gb * 0.92,
        "state_kb": round(state_size(c) * 2 / 1024, 1),
    }


PRESETS: dict[str, dict[str, Any]] = {
    "Пробный запуск · ~2M": {"params": 2_000_000, "seq_len": 128, "base_steps": 50, "micro": 8},
    "Крошка · ~5M": {"params": 5_000_000, "seq_len": 256, "base_steps": 300, "micro": 8},
    "Маленькая · ~15M": {"params": 15_000_000, "seq_len": 256, "base_steps": 800, "micro": 4},
    "Средняя · ~40M": {"params": 40_000_000, "seq_len": 512, "base_steps": 1500, "micro": 2},
    "Максимум T4 · ~80M": {"params": 80_000_000, "seq_len": 512, "base_steps": 2000, "micro": 1},
    "Своя (ввести вручную)": {"params": 0, "seq_len": 256, "base_steps": 200, "micro": 0},
}


def human_params(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)
