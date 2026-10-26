import torch

from gdn2lab.config import ModelConfig
from gdn2lab.model import build_model
from gdn2lab.quant import (QuantLinear, _Int8Gemm, convert_linears,
                           fold_quantization, has_quantized)


def test_int8_gemm_matches_dequantized_matmul():
    torch.manual_seed(0)
    x = torch.randn(5, 32)
    w = torch.randn(11, 32)
    sx = x.abs().amax(-1, keepdim=True) / 127.0
    sw = w.abs().amax(-1, keepdim=True) / 127.0
    got = _Int8Gemm.apply(x, w, sx, sw)
    xq = torch.round(x / sx).clamp(-127, 127)
    wq = torch.round(w / sw).clamp(-127, 127)
    ref = (xq * sx) @ (wq * sw).T
    assert torch.allclose(got, ref, atol=1e-4)


def test_qat_gradients_and_fold():
    c = ModelConfig(vocab_size=101, d_model=64, n_layer=2, n_head=2,
                    head_dim=32, seq_len=8, chunk_size=4)
    m = build_model(c)
    n = convert_linears(m, scheme="int8", int8_gemm=False)
    assert n > 0 and has_quantized(m)
    assert isinstance(m.blocks[0].mixer.q_proj, QuantLinear)
    idx = torch.randint(0, c.vocab_size, (2, 8))
    _, loss = m(idx, idx)
    loss.backward()
    assert m.blocks[0].mixer.q_proj.weight.grad is not None
    sd = fold_quantization(m)
    assert not has_quantized(m)
    build_model(c).load_state_dict(sd, strict=True)


def test_folded_weight_is_actually_8bit():
    lin = torch.nn.Linear(32, 16, bias=False)
    with torch.no_grad():
        lin.weight.normal_()
    q = QuantLinear.from_linear(lin, scheme="int8", int8_gemm=False)
    w = q.folded_weight()
    sw = q.weight.detach().float().abs().amax(-1, keepdim=True) / 127.0
    levels = torch.round(q.weight.detach().float() / sw).clamp(-127, 127)
    assert torch.allclose(w / sw, levels, atol=1e-3)


def test_fp8_fake_quant_runs():
    c = ModelConfig(vocab_size=101, d_model=64, n_layer=1, n_head=2,
                    head_dim=32, seq_len=8, chunk_size=4)
    m = build_model(c)
    convert_linears(m, scheme="fp8_e4m3", int8_gemm=False)
    idx = torch.randint(0, c.vocab_size, (1, 8))
    _, loss = m(idx, idx)
    loss.backward()
    assert torch.isfinite(loss)
