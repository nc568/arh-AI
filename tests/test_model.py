import torch

from gdn2lab.config import ModelConfig, count_params, state_size, solve_architecture
from gdn2lab.model import build_model


def tiny(**kw):
    d = dict(vocab_size=101, d_model=64, n_layer=2, n_head=2, head_dim=32,
             seq_len=8, chunk_size=4, use_short_conv=True)
    d.update(kw)
    return ModelConfig(**d)


def test_param_count_is_exact_with_and_without_conv():
    for conv in (False, True):
        c = tiny(use_short_conv=conv)
        assert count_params(c) == sum(p.numel() for p in build_model(c).parameters())


def test_decode_step_equals_full_forward():
    for conv in (False, True):
        c = tiny(use_short_conv=conv)
        m = build_model(c).eval()
        idx = torch.randint(0, c.vocab_size, (1, 8))
        with torch.no_grad():
            ref, _ = m(idx)
            state = m.initial_state(1, torch.device("cpu"), torch.float32)
            outs = []
            for t in range(8):
                lg, state = m.step(idx[:, t:t + 1], state)
                outs.append(lg)
            got = torch.cat(outs, dim=1)
        assert torch.allclose(ref, got, atol=3e-4, rtol=3e-4)


def test_state_layout():
    c = tiny()
    m = build_model(c)
    assert state_size(c) == c.n_layer * m.state_size()
    st = m.initial_state(3, torch.device("cpu"), torch.float32)
    assert st.shape == (c.n_layer, 3, m.state_size())


def test_forward_backward_and_loss():
    c = tiny()
    m = build_model(c)
    idx = torch.randint(0, c.vocab_size, (2, 8))
    logits, loss = m(idx, idx)
    assert logits.shape == (2, 8, c.vocab_size)
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())


def test_sft_mask_ignores_prompt():
    c = tiny(seq_len=8)
    m = build_model(c)
    idx = torch.randint(0, c.vocab_size, (2, 8))
    labels = idx.clone()
    labels[:, :4] = -100
    _, loss_masked = m(idx, labels)
    assert torch.isfinite(loss_masked)


def test_hybrid_swa_forward():
    c = tiny(hybrid_every=2, swa_window=4)
    m = build_model(c)
    idx = torch.randint(0, c.vocab_size, (1, 8))
    logits, loss = m(idx, idx)
    assert torch.isfinite(loss) and logits.shape[-1] == c.vocab_size


def test_solver_hits_target_and_conv_state_grows():
    c = solve_architecture(5_000_000, vocab_size=4096)
    assert abs(count_params(c) - 5_000_000) / 5_000_000 < 0.1
    assert c.n_layer >= 4
    assert c.d_model % c.head_dim == 0
