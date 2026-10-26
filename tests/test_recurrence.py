import torch

from gdn2lab.recurrence import gdn2_chunkwise, gdn2_naive, gdn2_step


def _data(B=2, H=2, T=9, dk=8, dv=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    q, k, v = r(B, H, T, dk), r(B, H, T, dk), r(B, H, T, dv)
    gd = -torch.rand(B, H, T, dk, generator=g) * 1.2
    b = torch.rand(B, H, T, dk, generator=g)
    w = torch.rand(B, H, T, dv, generator=g)
    s0 = r(B, H, dk, dv) * 0.1
    return q, k, v, gd, b, w, s0


def test_chunkwise_matches_naive_for_many_chunk_sizes():
    q, k, v, g, b, w, s0 = _data()
    ref_o, ref_s = gdn2_naive(q, k, v, g, b, w, initial_state=s0)
    for c in (1, 2, 3, 5, 9, 16):
        o, s = gdn2_chunkwise(q, k, v, g, b, w, chunk_size=c,
                              initial_state=s0, decay_block=4)
        assert torch.allclose(o, ref_o, atol=3e-4, rtol=3e-4), c
        assert torch.allclose(s, ref_s, atol=3e-4, rtol=3e-4), c


def test_gradients_match_naive():
    a = list(_data(T=8, seed=3))
    b = [x.clone().requires_grad_(True) for x in a[:6]]
    a = [x.requires_grad_(True) for x in a[:6]] + [a[6]]
    o1, s1 = gdn2_chunkwise(*a[:6], chunk_size=4, initial_state=a[6], decay_block=4)
    o2, s2 = gdn2_naive(*b[:6], initial_state=a[6].clone())
    (o1.square().sum() + s1.square().sum()).backward()
    (o2.square().sum() + s2.square().sum()).backward()
    for x, y in zip(a[:6], b[:6]):
        assert torch.allclose(x.grad, y.grad, atol=2e-3, rtol=2e-3)


def test_strong_decay_is_finite():
    B, H, T, dk, dv = 1, 2, 32, 8, 8
    r = lambda *s: torch.randn(*s)
    g = -torch.ones(B, H, T, dk) * 7.0
    o, s = gdn2_chunkwise(r(B, H, T, dk), r(B, H, T, dk), r(B, H, T, dv),
                          g, torch.rand(B, H, T, dk), torch.rand(B, H, T, dv),
                          chunk_size=16, decay_block=4)
    assert torch.isfinite(o).all() and torch.isfinite(s).all()


def test_single_step_shapes():
    q, k, v, g, b, w, s0 = _data(T=1)
    o, s = gdn2_step(q[:, :, 0], k[:, :, 0], v[:, :, 0], g[:, :, 0],
                     b[:, :, 0], w[:, :, 0], s0)
    assert o.shape == (2, 2, 8) and s.shape == (2, 2, 8, 8)
