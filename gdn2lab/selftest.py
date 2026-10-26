"""Fast self-checks that run on CPU (no Colab / GPU needed).

    python -m gdn2lab.cli selftest
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch


def _check(name: str, cond: bool, detail: str = "", verbose: bool = True) -> bool:
    if verbose:
        print(f"{'✅' if cond else '❌'} {name}" + (f" — {detail}" if detail else ""))
    return bool(cond)


def run_all(verbose: bool = True) -> bool:
    from .config import ModelConfig, count_params, solve_architecture
    from .model import build_model
    from .recurrence import gdn2_chunkwise, gdn2_naive

    ok = True
    torch.manual_seed(0)

    # 1. chunkwise WY == naive recurrence (forward + final state)
    B, H, T, dk, dv = 2, 2, 9, 8, 8
    q = torch.randn(B, H, T, dk)
    k = torch.randn(B, H, T, dk)
    v = torch.randn(B, H, T, dv)
    g = -torch.rand(B, H, T, dk) * 1.2
    b, w = torch.rand(B, H, T, dk), torch.rand(B, H, T, dv)
    s0 = torch.randn(B, H, dk, dv) * 0.1
    for c in (1, 3, 9, 16):
        o1, s1 = gdn2_chunkwise(q, k, v, g, b, w, chunk_size=c,
                                initial_state=s0, decay_block=4)
        o2, s2 = gdn2_naive(q, k, v, g, b, w, initial_state=s0)
        ok &= _check(f"chunkwise C={c} совпадает с рекуррентной формой",
                     torch.allclose(o1, o2, atol=2e-4, rtol=2e-4)
                     and torch.allclose(s1, s2, atol=2e-4, rtol=2e-4),
                     f"maxΔ={(o1 - o2).abs().max().item():.2e}", verbose)

    # 2. strong decay must not overflow / NaN
    gg = -torch.ones(B, H, 32, dk) * 7.0
    o, s = gdn2_chunkwise(torch.randn(B, H, 32, dk), torch.randn(B, H, 32, dk),
                          torch.randn(B, H, 32, dv), gg,
                          torch.rand(B, H, 32, dk), torch.rand(B, H, 32, dv),
                          chunk_size=16, decay_block=4)
    ok &= _check("сильный decay без NaN/Inf",
                 bool(torch.isfinite(o).all() and torch.isfinite(s).all()), "", verbose)

    # 3. exact parameter counting
    for cfg in [
        ModelConfig(vocab_size=257, d_model=128, n_layer=2, n_head=2, head_dim=64,
                    seq_len=8, chunk_size=4, use_short_conv=False),
        ModelConfig(vocab_size=257, d_model=128, n_layer=3, n_head=2, head_dim=64,
                    seq_len=8, chunk_size=4, use_short_conv=True),
    ]:
        actual = sum(p.numel() for p in build_model(cfg).parameters())
        ok &= _check("count_params точный", actual == count_params(cfg),
                     f"{actual} == {count_params(cfg)}", verbose)

    # 4. architecture solver hits the target
    sol = solve_architecture(5_000_000, vocab_size=4096)
    p = count_params(sol)
    ok &= _check("solve_architecture попадает в цель ±10%",
                 abs(p - 5_000_000) / 5_000_000 < 0.10,
                 f"{p:,} ({p / 1e6:.2f}M)", verbose)

    # 5. decode step == full sequence forward
    cfg = ModelConfig(vocab_size=101, d_model=64, n_layer=2, n_head=2,
                      head_dim=32, seq_len=8, chunk_size=4, use_short_conv=True)
    m = build_model(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        ref, _ = m(idx)
        state = m.initial_state(1, torch.device("cpu"), torch.float32)
        outs = []
        for t in range(8):
            lg, state = m.step(idx[:, t:t + 1], state)
            outs.append(lg)
        got = torch.cat(outs, dim=1)
    ok &= _check("decode step == prefill forward",
                 torch.allclose(ref, got, atol=2e-4, rtol=2e-4),
                 f"maxΔ={(ref - got).abs().max().item():.2e}", verbose)

    # 6. 8-bit training path + folding
    from .quant import convert_linears, fold_quantization, has_quantized
    m2 = build_model(cfg)
    n = convert_linears(m2, "int8", int8_gemm=False)
    logits, loss = m2(idx, idx)
    loss.backward()
    grad_ok = m2.blocks[0].mixer.q_proj.weight.grad is not None
    sd = fold_quantization(m2)
    m3 = build_model(cfg)
    m3.load_state_dict(sd, strict=True)
    ok &= _check("int8 QAT: градиенты и fold", n > 0 and grad_ok and not has_quantized(m2),
                 f"{n} слоёв", verbose)

    # 7. torch.export of the LiteRT decode graph
    from .export_litert import DecodeWrapper
    w = DecodeWrapper(m3.eval())
    args = (torch.zeros((1, 1), dtype=torch.int32),
            torch.zeros((cfg.n_layer, 1, m3.state_size()), dtype=torch.float32))
    try:
        with torch.no_grad():
            ep = torch.export.export(w, args)
            out = ep.module()(*args)
        ok &= _check("torch.export decode-графа", out[0].shape[-1] == cfg.vocab_size,
                     f"{len(list(ep.graph.nodes))} узлов", verbose)
    except Exception as exc:
        ok &= _check("torch.export decode-графа", False, repr(exc), verbose)

    # 8. GGUF write/read roundtrip
    try:
        import gguf  # noqa: F401
        from safetensors.torch import save_file
        from .config import LabConfig
        from .export_gguf import export_gguf
        from .gguf_runtime import load_gguf
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            lab = LabConfig(model=cfg, workdir=str(root))
            lab.train.out_dir = "ckpt"
            ck = root / "ckpt"
            ck.mkdir(parents=True, exist_ok=True)
            state = {k: v.detach().cpu().contiguous() for k, v in m3.state_dict().items()
                     if k != "head.weight"}
            save_file(state, str(ck / "model.safetensors"))
            lab.to_json(ck / "config.json")
            rep = export_gguf(lab, ck, root / "test.gguf", quant="Q8_0")
            loaded, _ = load_gguf(root / "test.gguf", "cpu")
            with torch.no_grad():
                a, _ = loaded(idx)
                b_, _ = m3(idx)
            good = rep["tensors"] == len(state) + 1 and torch.allclose(a, b_, atol=0.05, rtol=0.05)
            ok &= _check("GGUF Q8_0 roundtrip", good,
                         f"{rep['size_mb']} MB, maxΔ={(a - b_).abs().max().item():.3f}", verbose)
    except Exception as exc:
        ok &= _check("GGUF Q8_0 roundtrip", False, repr(exc), verbose)

    return bool(ok)
