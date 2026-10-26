"""Small GDN-2 language model, written from the paper's block design (fig. 1).

Pure PyTorch, no Triton / flash-attn / custom CUDA extensions, so the *same*
checkpoint trains and runs on a free Colab T4, a free TPU (torch-xla) and CPU.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .recurrence import gdn2_chunkwise, gdn2_step, gdn2_naive

Tensor = torch.Tensor


def _silu(x: Tensor) -> Tensor:
    """SiLU written out — TFLite has LOGISTIC and MUL, not a SILU opcode."""
    return x * torch.sigmoid(x)


def _softplus(x: Tensor) -> Tensor:
    """Stable softplus via LOG(1+EXP) — no SOFTPLUS opcode on most NPUs."""
    return torch.log(1.0 + torch.exp(x.clamp(max=20.0)))


def _l2norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True).clamp(min=eps))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).to(dtype)


class CausalShortConv(nn.Module):
    """Depthwise causal conv, implemented as explicit shifts (TFLite-friendly).

    ``y[t] = SiLU(sum_j w[j] * x[t-K+1+j])``.  No ``conv1d`` op is needed,
    which matters a lot for NPU delegates that only support CONV_2D.
    """

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.zeros(kernel_size, channels))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.weight[-1].fill_(1.0)  # start close to identity

    def forward(self, x: Tensor) -> Tensor:
        b, t, c = x.shape
        xp = F.pad(x, (0, 0, self.kernel_size - 1, 0))
        y = 0.0
        for j in range(self.kernel_size):
            y = y + self.weight[j] * xp[:, j:j + t, :]
        return _silu(y)

    def step(self, x_t: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        # x_t [B,1,C], state [B,K-1,C]
        window = torch.cat([state, x_t], dim=1)
        y = 0.0
        for j in range(self.kernel_size):
            y = y + self.weight[j] * window[:, j:j + 1, :]
        return _silu(y), window[:, 1:, :]


class GatedRMSNormGate(nn.Module):
    """RMSNorm(o) * SiLU(gate) — the output path in fig. 1."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, o: Tensor, gate: Tensor) -> Tensor:
        dtype = o.dtype
        of = o.float()
        normed = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight.float() * _silu(gate.float())).to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, c: ModelConfig, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(c.d_model, c.hidden_ffn, bias=False)
        self.w3 = nn.Linear(c.d_model, c.hidden_ffn, bias=False)
        self.w2 = nn.Linear(c.hidden_ffn, c.d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.w2(_silu(self.w1(x)) * self.w3(x)))


def _rotate_half(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


class SlidingWindowAttention(nn.Module):
    """Optional local softmax attention for the hybrid model (paper fig. 1)."""

    def __init__(self, c: ModelConfig):
        super().__init__()
        self.n_head = c.n_head
        self.head_dim = c.head_dim
        self.window = c.swa_window
        self.q = nn.Linear(c.d_model, c.key_dim, bias=False)
        self.k = nn.Linear(c.d_model, c.key_dim, bias=False)
        self.v = nn.Linear(c.d_model, c.key_dim, bias=False)
        self.o = nn.Linear(c.key_dim, c.d_model, bias=False)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, c.head_dim, 2).float() / c.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope(self, x: Tensor) -> Tensor:
        t = x.shape[2]
        pos = torch.arange(t, device=x.device, dtype=torch.float32)
        ang = torch.einsum("t,d->td", pos, self.inv_freq.float())
        emb = torch.cat([ang, ang], dim=-1)  # [T, head_dim]
        cos, sin = emb.cos(), emb.sin()
        return x * cos[None, None] + _rotate_half(x) * sin[None, None]

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        h, dh = self.n_head, self.head_dim
        q = self._rope(self.q(x).view(b, t, h, dh).transpose(1, 2))
        k = self._rope(self.k(x).view(b, t, h, dh).transpose(1, 2))
        v = self.v(x).view(b, t, h, dh).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dh)
        pos = torch.arange(t, device=x.device)
        causal = (pos[:, None] >= pos[None, :]) & ((pos[:, None] - pos[None, :]) < self.window)
        scores = scores.masked_fill(~causal[None, None], float("-inf"))
        out = torch.matmul(F.softmax(scores.float(), dim=-1).to(v.dtype), v)
        return self.o(out.transpose(1, 2).reshape(b, t, h * dh))


class GDN2Mixer(nn.Module):
    """Gated DeltaNet-2 token mixer: projections -> Gated Delta Rule-2 -> gate."""

    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c
        d, k, v, hv = c.d_model, c.key_dim, c.value_dim, c.head_v_dim
        self.nv = c.n_v_heads_eff
        self.group = max(1, self.nv // c.n_head)
        self.q_proj = nn.Linear(d, k, bias=False)
        self.k_proj = nn.Linear(d, k, bias=False)
        self.v_proj = nn.Linear(d, v, bias=False)
        self.b_proj = nn.Linear(d, k, bias=False)  # erase gate (key axis)
        self.w_proj = nn.Linear(d, v, bias=False)  # write gate (value axis)
        self.f_proj = nn.Sequential(nn.Linear(d, hv, bias=False), nn.Linear(hv, k, bias=False))
        self.g_proj = nn.Sequential(nn.Linear(d, hv, bias=False), nn.Linear(hv, v, bias=True))
        self.o_norm = GatedRMSNormGate(hv, c.norm_eps)
        self.o_proj = nn.Linear(v, d, bias=False)
        self.drop = nn.Dropout(c.dropout)

        if c.use_short_conv:
            self.q_conv = CausalShortConv(k, c.conv_size)
            self.k_conv = CausalShortConv(k, c.conv_size)
            self.v_conv = CausalShortConv(v, c.conv_size)

        self.A_log = nn.Parameter(torch.log(torch.empty(c.n_head).uniform_(1, 16)))
        dt = torch.exp(
            torch.rand(k) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

    # -- shared helpers ---------------------------------------------------- #
    def _decay_log(self, x: Tensor) -> Tensor:
        """eq. (12): g_t = -exp(a) * softplus(W_f x_t + delta), in fp32."""
        a = self.A_log.float().exp().repeat_interleave(self.c.head_dim)
        bias = self.dt_bias.float()
        f = self.f_proj(x).float()
        if self.group > 1:
            a = a.repeat_interleave(self.group)
            bias = bias.repeat_interleave(self.group)
            f = f.repeat_interleave(self.group, dim=-1)
        return -a * _softplus(f + bias)

    def _heads(self, x: Tensor, dim: int, value_side: bool = False,
               repeated: bool = False) -> Tensor:
        b, t, _ = x.shape
        h = self.nv if (value_side or repeated) else self.c.n_head
        d = self.c.head_v_dim if value_side else dim
        y = x.reshape(b, t, h, d).transpose(1, 2).contiguous()  # [B,H,T,D]
        if not value_side and not repeated and self.group > 1:
            y = y.repeat_interleave(self.group, dim=1)
        return y

    def _out_path(self, o: Tensor, x: Tensor) -> Tensor:
        # o: [B,H,T,Dv] -> [B,T,H,Dv]
        b, _, t, dv = o.shape
        oo = o.transpose(1, 2).contiguous()
        gate = self.g_proj(x).reshape(b, t, self.nv, dv)
        y = self.o_norm(oo, gate).reshape(b, t, self.nv * dv)
        return self.drop(self.o_proj(y))

    def _prepare(self, x: Tensor, conv_state=None, step: bool = False):
        c = self.c
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        new_conv = None
        if c.use_short_conv:
            if step:
                assert conv_state is not None
                cq, ck, cv = conv_state
                q, nq = self.q_conv.step(q, cq)
                k, nk = self.k_conv.step(k, ck)
                v, nv = self.v_conv.step(v, cv)
                new_conv = (nq, nk, nv)
            else:
                q, k, v = self.q_conv(q), self.k_conv(k), self.v_conv(v)
        else:
            q, k, v = _silu(q), _silu(k), _silu(v)

        q = _l2norm(q.float())
        k = _l2norm(k.float())
        v = v.float()
        g = self._decay_log(x)
        b = torch.sigmoid(self.b_proj(x).float())
        w = torch.sigmoid(self.w_proj(x).float())
        if c.allow_neg_eigval:
            b = b * 2.0
        q = self._heads(q, c.head_dim)
        k = self._heads(k, c.head_dim)
        g = self._heads(g, c.head_dim, repeated=self.group > 1)
        b = self._heads(b, c.head_dim)
        v = self._heads(v, c.head_v_dim, value_side=True)
        w = self._heads(w, c.head_v_dim, value_side=True)
        return q, k, v, g, b, w, new_conv

    # -- training / prefill ------------------------------------------------ #
    def forward(self, x: Tensor, state: Tensor | None = None) -> Tensor:
        q, k, v, g, b, w, _ = self._prepare(x)
        c = self.c
        if state is None:
            o, _ = gdn2_chunkwise(
                q, k, v, g, b, w,
                chunk_size=c.chunk_size,
                decay_block=c.decay_block,
                max_log_decay=c.max_log_decay,
            )
        else:
            o, _ = gdn2_naive(q, k, v, g, b, w, initial_state=state,
                              max_log_decay=c.max_log_decay)
        return self._out_path(o, x)

    # -- single-token decode ---------------------------------------------- #
    def forward_step(self, x: Tensor, layer_state: tuple[Tensor, ...]):
        c = self.c
        conv_state = layer_state[1:] if c.use_short_conv else None
        q, k, v, g, b, w, new_conv = self._prepare(x, conv_state, step=True)
        S = layer_state[0]
        o, Snew = gdn2_step(
            q[:, :, 0], k[:, :, 0], v[:, :, 0], g[:, :, 0], b[:, :, 0], w[:, :, 0],
            S, max_log_decay=c.max_log_decay,
        )
        o = o.unsqueeze(2)  # [B,H,1,Dv]
        y = self._out_path(o, x)
        if c.use_short_conv:
            assert new_conv is not None
            return y, (Snew,) + new_conv
        return y, (Snew,)


class GDN2Block(nn.Module):
    def __init__(self, c: ModelConfig, idx: int):
        super().__init__()
        self.norm1 = RMSNorm(c.d_model, c.norm_eps)
        self.mixer = GDN2Mixer(c)
        self.norm2 = RMSNorm(c.d_model, c.norm_eps)
        self.mlp = SwiGLU(c, c.dropout)
        self.swa: SlidingWindowAttention | None = None
        if idx in c.swa_layers:
            self.norm3 = RMSNorm(c.d_model, c.norm_eps)
            self.swa = SlidingWindowAttention(c)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        if self.swa is not None:
            x = x + self.swa(self.norm3(x))
        return x

    def forward_step(self, x: Tensor, layer_state):
        if self.swa is not None:
            raise RuntimeError(
                "Гибридные SWA-слои не поддерживаются в пошаговом decode / LiteRT. "
                "Поставьте model.hybrid_every = 0."
            )
        y, st = self.mixer.forward_step(self.norm1(x), layer_state)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x, st


class GDN2LM(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList([GDN2Block(c, i) for i in range(c.n_layer)])
        self.norm = RMSNorm(c.d_model, c.norm_eps)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        if c.tie_embeddings:
            self.head.weight = self.embed.weight
        self.apply(self._init_weights)
        self._scale_residual()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _scale_residual(self):
        std = 0.02 / math.sqrt(2 * max(1, self.c.n_layer))
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and (name.endswith("o_proj") or name.endswith("w2") or name.endswith(".o")):
                nn.init.normal_(m.weight, mean=0.0, std=std)

    # -- full-sequence forward (training) ---------------------------------- #
    def forward(self, idx: Tensor, labels: Tensor | None = None):
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                labels[:, 1:].reshape(-1).long(),
                ignore_index=-100,
            )
        return logits, loss

    # -- recurrent state helpers ------------------------------------------- #
    def state_shapes(self) -> list[tuple[str, tuple[int, ...]]]:
        c = self.c
        shapes = [("S", (self._h, c.head_dim, c.head_v_dim))]
        if c.use_short_conv:
            shapes += [
                ("cq", (c.conv_size - 1, c.key_dim)),
                ("ck", (c.conv_size - 1, c.key_dim)),
                ("cv", (c.conv_size - 1, c.value_dim)),
            ]
        return shapes

    @property
    def _h(self) -> int:
        return self.c.n_v_heads_eff

    def state_size(self) -> int:
        """Number of floats per layer in the flat [L, B, S] decode state."""
        n = 0
        for _, shape in self.state_shapes():
            p = 1
            for s in shape:
                p *= s
            n += p
        return int(n)

    def initial_state(self, batch: int = 1, device=None, dtype=torch.float32) -> Tensor:
        """Flat state [N_LAYER, STATE_PER_LAYER] — the LiteRT-friendly layout."""
        device = device or next(self.parameters()).device
        vals = []
        for _, shape in self.state_shapes():
            p = 1
            for s in shape:
                p *= s
            vals.append(torch.zeros(batch, p, device=device, dtype=dtype))
        per = torch.cat(vals, dim=-1)  # [B, S]
        return per.unsqueeze(0).expand(self.c.n_layer, -1, -1).contiguous()

    @staticmethod
    def _split_state(flat: Tensor, shapes) -> list[tuple[Tensor, ...]]:
        """flat: [L, B, S] -> per-layer tuple of reshaped tensors."""
        layers = []
        offsets = {}
        off = 0
        for name, shape in shapes:
            p = 1
            for s in shape:
                p *= s
            offsets[name] = (off, off + p, shape)
            off += p
        for li in range(flat.shape[0]):
            row = flat[li]  # [B, S]
            parts = []
            for name, _ in shapes:
                a, b, shape = offsets[name]
                parts.append(row[:, a:b].reshape(row.shape[0], *shape))
            layers.append(tuple(parts))
        return layers

    def step(self, idx: Tensor, flat_state: Tensor):
        """One decode step. idx: [B,1], flat_state: [L,B,S]. -> logits, new_state."""
        if self.c.hybrid_every > 0 and self.c.swa_layers:
            raise RuntimeError("SWA layers: use forward() for generation or set hybrid_every=0.")
        parts = self._split_state(flat_state, self.state_shapes())
        x = self.embed(idx)
        new_layers = []
        for blk, st in zip(self.blocks, parts):
            x, new_st = blk.forward_step(x, st)
            new_layers.append(new_st)
        logits = self.head(self.norm(x))
        rows = [torch.cat([p.reshape(p.shape[0], -1) for p in layer], dim=-1) for layer in new_layers]
        return logits, torch.stack(rows, dim=0)

    # -- generation --------------------------------------------------------- #
    @torch.no_grad()
    def generate(
        self,
        prompt: Tensor,
        tokenizer=None,
        max_new_tokens: int = 80,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        eos_id: int | None = None,
        callback=None,
    ) -> Tensor:
        self.eval()
        device = prompt.device
        ids = prompt.clone()
        if self.c.hybrid_every > 0 and self.c.swa_layers:
            # slow but correct fallback: re-run full forward each step
            for _ in range(max_new_tokens):
                logits, _ = self.forward(ids[:, -self.c.seq_len:])
                nxt = _sample(logits[:, -1], temperature, top_k, top_p)
                ids = torch.cat([ids, nxt], dim=1)
                if eos_id is not None and (nxt == eos_id).all():
                    break
            return ids
        state = self.initial_state(prompt.shape[0], device, torch.float32)
        # feed the prompt
        for t in range(prompt.shape[1] - 1):
            _, state = self.step(prompt[:, t:t + 1], state)
        cur = prompt[:, -1:]
        for i in range(max_new_tokens):
            logits, state = self.step(cur, state)
            nxt = _sample(logits[:, -1], temperature, top_k, top_p)
            ids = torch.cat([ids, nxt], dim=1)
            cur = nxt
            if callback is not None:
                callback(nxt)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return ids


def _sample(logits: Tensor, temperature: float, top_k: int, top_p: float) -> Tensor:
    logits = logits.float()
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    if top_k > 0:
        k = min(top_k, logits.shape[-1])
        vals, _ = torch.topk(logits, k)
        logits = logits.masked_fill(logits < vals[:, -1:], float("-inf"))
    if 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)


def build_model(c: ModelConfig) -> GDN2LM:
    return GDN2LM(c)
