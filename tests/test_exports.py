import json
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from gdn2lab.config import LabConfig, ModelConfig
from gdn2lab.export_litert import DecodeWrapper, PrefillWrapper
from gdn2lab.model import build_model


def tiny():
    return ModelConfig(vocab_size=101, d_model=64, n_layer=2, n_head=2,
                       head_dim=32, seq_len=16, chunk_size=8, use_short_conv=True)


def test_torch_export_decode_and_prefill_graphs():
    c = tiny()
    m = build_model(c).eval()
    S = m.state_size()
    args = (torch.zeros((1, 1), dtype=torch.int32),
            torch.zeros((c.n_layer, 1, S), dtype=torch.float32))
    with torch.no_grad():
        ep = torch.export.export(DecodeWrapper(m).eval(), args)
        o = ep.module()(*args)
        assert o[0].shape == (1, 1, c.vocab_size)
        assert o[1].shape == (c.n_layer, 1, S)

        pargs = (torch.zeros((1, 4), dtype=torch.int32),
                 torch.zeros((c.n_layer, 1, S), dtype=torch.float32))
        ep2 = torch.export.export(PrefillWrapper(m, 4).eval(), pargs)
        assert ep2.module()(*pargs)[0].shape == (1, 4, c.vocab_size)


def test_gguf_q8_roundtrip():
    gguf = pytest.importorskip("gguf")
    from gdn2lab.export_gguf import export_gguf
    from gdn2lab.gguf_runtime import load_gguf

    c = tiny()
    m = build_model(c).eval()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        lab = LabConfig(model=c, workdir=str(root))
        ck = root / "ckpt"
        ck.mkdir(parents=True, exist_ok=True)
        sd = {k: v.detach().cpu().contiguous() for k, v in m.state_dict().items()
              if k != "head.weight"}
        save_file(sd, str(ck / "model.safetensors"))
        lab.to_json(ck / "config.json")
        rep = export_gguf(lab, ck, root / "m.gguf", quant="Q8_0")
        assert rep["tensors"] == len(sd) + 1
        assert rep["verify_max_abs_error"] < 0.1
        loaded, _ = load_gguf(root / "m.gguf", "cpu")
        idx = torch.randint(0, c.vocab_size, (1, 8))
        with torch.no_grad():
            a, _ = loaded(idx)
            b, _ = m(idx)
        assert torch.allclose(a, b, atol=0.08, rtol=0.08)


def test_selftest_entrypoint():
    from gdn2lab.selftest import run_all
    assert run_all(verbose=False)
