"""Gated Delta Rule-2 (GDN-2) in pure PyTorch.

Reference: arXiv:2605.22791v1, eq. (10), (12) and the chunkwise WY form
eq. (18)-(25).  Written independently from the paper, without using the
NVIDIA reference kernels.

Recurrent form (one head), with channel-wise decay ``alpha_t = exp(g_t)``,
erase gate ``b_t`` and write gate ``w_t``::

    z_t = w_t * v_t
    e_t = b_t * k_t
    S_t = alpha_t[:, None] * S_{t-1} + k_t[:, None] * (z_t - S_{t-1}^T @ e_t)
    o_t = S_t^T @ q_t

The chunkwise form absorbs the cumulative decay into the two factors of the
rank-one erase and needs only small triangular solves plus dense matmuls, so
it trains in parallel on CUDA, TPU and CPU.  All decay arithmetic is done in
log space / fp32: we never form ``1/gamma`` directly (that would overflow).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Tuple

Tensor = torch.Tensor


def _lower_masks(c: int, device, dtype):
    lower = torch.ones(c, c, device=device, dtype=dtype).tril(0)       # r >= s
    strict = torch.ones(c, c, device=device, dtype=dtype).tril(-1)     # r > s
    return lower, strict


def _solve_unit_lower(t: Tensor, rhs: Tensor) -> Tensor:
    """Forward substitution for ``(I + T) X = rhs`` with strictly lower ``T``.

    A Python loop (not ``linalg.solve_triangular``) keeps the graph portable to
    torch-xla / torch.export.  ``t``: [..., C, C], ``rhs``: [..., C, M].
    """
    c = t.shape[-1]
    rows: list[Tensor] = []
    for i in range(c):
        row = rhs[..., i, :]
        if i:
            prev = torch.stack(rows, dim=-2)                 # [..., i, M]
            coef = t[..., i, :i].unsqueeze(-1)               # [..., i, 1]
            row = row - (coef * prev).sum(dim=-2)
        rows.append(row)
    return torch.stack(rows, dim=-2)


def gdn2_step(
    q: Tensor, k: Tensor, v: Tensor, g: Tensor, b: Tensor, w: Tensor,
    state: Tensor, max_log_decay: float = 8.0,
) -> Tuple[Tensor, Tensor]:
    """One recurrent token.  All inputs are [B, H, D]; state is [B,H,Dk,Dv]."""
    q, k, v, g, b, w = (x.float() for x in (q, k, v, g, b, w))
    state = state.float()
    if max_log_decay > 0:
        g = g.clamp(min=-max_log_decay, max=0.0)
    alpha = torch.exp(g)                                    # [B,H,Dk]
    z = w * v                                              # [B,H,Dv]
    e = b * k                                              # [B,H,Dk]
    # Written as multiply + reduce instead of einsum/bmm so the decode graph
    # contains only MUL / SUM / ADD — the ops every NPU delegate understands.
    decayed = alpha.unsqueeze(-1) * state                  # [B,H,Dk,Dv]
    read = (e.unsqueeze(-1) * decayed).sum(dim=-2)         # S^T e  [B,H,Dv]
    new_state = decayed + k.unsqueeze(-1) * (z - read).unsqueeze(-2)
    o = (q.unsqueeze(-1) * new_state).sum(dim=-2)          # S^T q  [B,H,Dv]
    return o, new_state


def gdn2_naive(
    q: Tensor, k: Tensor, v: Tensor, g: Tensor, b: Tensor, w: Tensor,
    initial_state: Tensor | None = None, max_log_decay: float = 8.0,
) -> Tuple[Tensor, Tensor]:
    """Straightforward token loop, used as the numerical reference and for
    generation / LiteRT decode.  q,k,g,b: [B,H,T,Dk]; v,w: [B,H,T,Dv]."""
    q, k, v, g, b, w = (x.float() for x in (q, k, v, g, b, w))
    B, H, T, dk = q.shape
    dv = v.shape[-1]
    if initial_state is None:
        state = q.new_zeros((B, H, dk, dv))
    else:
        state = initial_state.float()
    outs = []
    for i in range(T):
        o, state = gdn2_step(
            q[:, :, i], k[:, :, i], v[:, :, i], g[:, :, i], b[:, :, i], w[:, :, i],
            state, max_log_decay,
        )
        outs.append(o)
    return torch.stack(outs, dim=2), state


def gdn2_chunkwise(
    q: Tensor, k: Tensor, v: Tensor, g: Tensor, b: Tensor, w: Tensor,
    chunk_size: int = 16, initial_state: Tensor | None = None,
    decay_block: int = 16, max_log_decay: float = 8.0,
) -> Tuple[Tensor, Tensor]:
    """Chunkwise WY forward, eq. (18)-(25).  Returns ``(o, final_state)``.

    q, k, g, b : [B, H, T, Dk]
    v, w       : [B, H, T, Dv]
    o          : [B, H, T, Dv]
    state      : [B, H, Dk, Dv]
    """
    out_dtype = q.dtype
    q, k, v, g, b, w = (x.float() for x in (q, k, v, g, b, w))
    B, H, T, dk = q.shape
    dv = v.shape[-1]
    c = int(chunk_size)
    if T % c:
        pad = c - T % c
        q, k, v, g, b, w = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v, g, b, w))
    tp = q.shape[2]
    n = tp // c
    if max_log_decay > 0:
        g = g.clamp(min=-max_log_decay, max=0.0)

    def chunks(x, d):
        return x.reshape(B, H, n, c, d)

    qc, kc, gc, bc = chunks(q, dk), chunks(k, dk), chunks(g, dk), chunks(b, dk)
    vc, wc = chunks(v, dv), chunks(w, dv)

    # G_r = sum_{i<=r} g_i  (within each chunk); gamma_r = exp(G_r) <= 1.
    G = torch.cumsum(gc, dim=3)                                    # [B,H,N,C,Dk]
    gamma = torch.exp(G)
    lower, strict = _lower_masks(c, q.device, q.dtype)
    lower = lower.view(1, 1, 1, c, c, 1)
    strict = strict.view(1, 1, 1, c, c)

    # T = tril(E_bar K_bar^T, -1),  A_qk = tril(Q_gamma K_inv^T)
    # Both are built from exp(G_r - G_s) with r >= s, which is <= 1 and stable.
    Ebase = bc * kc                                               # b ⊙ k
    tmat = q.new_zeros((B, H, n, c, c))
    aqk = q.new_zeros((B, H, n, c, c))
    dblk = max(1, int(decay_block))
    for d0 in range(0, dk, dblk):
        d1 = min(dk, d0 + dblk)
        # r indexes dim 3, s indexes dim 4, channel d indexes the last dim.
        Gr = G[..., d0:d1].unsqueeze(4)                            # [B,H,N,C,1,db]
        Gs = G[..., d0:d1].unsqueeze(3)                            # [B,H,N,1,C,db]
        ratio = torch.exp((Gr - Gs).clamp(max=0.0)) * lower
        er = Ebase[..., d0:d1].unsqueeze(4)                        # [B,H,N,C,1,db]
        qr = qc[..., d0:d1].unsqueeze(4)
        ks = kc[..., d0:d1].unsqueeze(3)                           # [B,H,N,1,C,db]
        tmat = tmat + (er * ks * ratio).sum(-1) * strict
        aqk = aqk + (qr * ks * ratio).sum(-1)

    # WY auxiliaries: Y = A E_bar, U = A Z with A = (I + T)^{-1}.
    Ebar = gamma * Ebase                                          # gamma ⊙ (b⊙k)
    Z = wc * vc                                                   # w ⊙ v
    rhs = torch.cat([Ebar, Z], dim=-1)                             # [B,H,N,C,Dk+Dv]
    sol = _solve_unit_lower(tmat, rhs)
    Y, U = sol[..., :dk], sol[..., dk:]

    # K_tail row r = (gamma_C / gamma_r) ⊙ k_r, again in log space.
    ktail = kc * torch.exp((G[..., -1:, :] - G).clamp(max=0.0))
    qgamma = qc * gamma

    if initial_state is None:
        state = q.new_zeros((B, H, dk, dv))
    else:
        state = initial_state.float()

    out_blocks: list[Tensor] = []
    for i in range(n):
        delta = U[:, :, i] - torch.matmul(Y[:, :, i], state)      # [B,H,C,Dv]
        o = torch.matmul(qgamma[:, :, i], state) + torch.matmul(aqk[:, :, i], delta)
        out_blocks.append(o)
        state = gamma[:, :, i, -1].unsqueeze(-1) * state + torch.matmul(
            ktail[:, :, i].transpose(-1, -2), delta
        )

    o = torch.cat(out_blocks, dim=2)[:, :, :T]
    return o.to(out_dtype), state.to(out_dtype)
