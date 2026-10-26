import json

import numpy as np
import torch

from gdn2lab.config import DataConfig, LabConfig, ModelConfig
from gdn2lab.data import (LabTokenizer, PackedBaseDataset, SFTDataset,
                          iter_records, prepare_all)


def test_prepare_demo_data(tmp_path):
    cfg = LabConfig(
        model=ModelConfig(vocab_size=1024, d_model=64, n_layer=1, n_head=2,
                          head_dim=32, seq_len=32, chunk_size=8),
        data=DataConfig(tokenizer={"kind": "scratch", "vocab_size": 1024},
                        max_base_tokens=3000, sft_max_seq_len=32),
        workdir=str(tmp_path),
    )
    info = prepare_all(cfg)
    d = tmp_path / "data"
    assert (d / "base_train.bin").exists() and (d / "sft_train.npz").exists()
    assert info["base"]["train_tokens"] > 500
    assert info["sft"]["supervised_tokens"] > 300
    cfg = LabConfig.from_json(tmp_path / "config.json")
    assert cfg.model.vocab_size == info["vocab_size"]

    tok = LabTokenizer.load(d)
    ids = tok.encode("Привет, мир!")
    assert ids and "Привет" in tok.decode(ids)

    base = PackedBaseDataset(d / "base_train.bin", cfg.model.seq_len, dtype=np.uint16)
    x, y = base[0]
    assert x.shape == y.shape == (cfg.model.seq_len,)
    assert torch.equal(x, y)

    sft = SFTDataset(d / "sft_train.npz")
    sx, sy = sft[0]
    assert sx.shape == sy.shape == (cfg.model.seq_len,)
    assert (sy != -100).sum() > 0          # ответ не замаскирован
    assert (sy == -100).sum() > 0          # промпт замаскирован
    assert (sx[sy != -100] == sy[sy != -100]).all()


def test_chatml_records():
    from gdn2lab.data import _chatml_prompt
    row = {"messages": [
        {"role": "user", "content": "Кто ты?"},
        {"role": "assistant", "content": "Я GDN-2."},
    ]}
    p, a = _chatml_prompt(row, "chatml")
    assert "assistant" in p and a == "Я GDN-2.<|im_end|>"
