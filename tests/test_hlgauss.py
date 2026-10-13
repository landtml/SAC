import pytest
import torch

from softedge.agent.hlgauss import CategoricalCritic, HLGauss


@pytest.fixture
def hl():
    return HLGauss()  # 101 bins on [-10, 10]


def test_centers_are_bin_midpoints():
    hl = HLGauss(n_bins=4, v_min=-2.0, v_max=2.0)
    assert hl.width == pytest.approx(1.0)
    assert torch.allclose(hl.centers, torch.tensor([-1.5, -0.5, 0.5, 1.5]))


def test_projection_is_a_distribution(hl):
    torch.manual_seed(0)
    y = torch.rand(8, 5) * 30 - 15  # includes targets outside the support
    p = hl.project(y)
    assert p.shape == (8, 5, hl.n_bins)
    assert (p >= 0).all()
    assert torch.allclose(p.sum(-1), torch.ones(8, 5), atol=1e-6)


def test_projection_mean_matches_target_inside_support(hl):
    # Renormalized truncation shifts the mean of targets near an edge, so the bin_width / 10
    # bound is checked on targets at least 0.5 from either edge.
    y = torch.linspace(-9.5, 9.5, 401)
    mean = hl.project(y) @ hl.centers
    assert (mean - y).abs().max() < hl.width / 10


def test_value_recovers_target_from_log_projection(hl):
    y = torch.linspace(-9.5, 9.5, 201)
    recovered = hl.value(torch.log(hl.project(y)))
    assert (recovered - y).abs().max() < hl.width / 10


def test_out_of_support_targets_clip_to_edge_bins(hl):
    low = hl.project(torch.tensor([-50.0]))[0]
    high = hl.project(torch.tensor([50.0]))[0]
    assert torch.allclose(low, hl.project(torch.tensor([-10.0]))[0])
    assert torch.allclose(high, hl.project(torch.tensor([10.0]))[0])
    assert low.argmax().item() == 0
    assert high.argmax().item() == hl.n_bins - 1


def test_loss_is_minimal_at_the_projected_target(hl):
    torch.manual_seed(0)
    targets = torch.rand(6) * 10 - 5
    # Clamp before the log: exact zeros in the projection would give -inf logits.
    at_target = torch.log(hl.project(targets).clamp_min(1e-30))[None]
    random = torch.randn(1, 6, hl.n_bins)
    assert hl.loss(at_target, targets) < hl.loss(random, targets)


def test_loss_sums_over_members(hl):
    torch.manual_seed(0)
    logits = torch.randn(2, 6, hl.n_bins)
    targets = torch.rand(6) * 10 - 5
    total = hl.loss(logits, targets)
    parts = sum(hl.loss(logits[e : e + 1], targets) for e in range(2))
    assert torch.allclose(total, parts)


def test_loss_is_finite_with_gradients_only_to_logits(hl):
    torch.manual_seed(0)
    logits = torch.randn(2, 6, hl.n_bins, requires_grad=True)
    targets = (torch.rand(6) * 10 - 5).requires_grad_()
    loss = hl.loss(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
    assert targets.grad is None


def test_critic_shapes_and_independent_members():
    torch.manual_seed(0)
    critic = CategoricalCritic(obs_dim=3, act_dim=2, members=2, hidden=32, blocks=1)
    obs, act = torch.randn(5, 3), torch.randn(5, 2)
    logits = critic(obs, act)
    assert logits.shape == (2, 5, critic.hl.n_bins)
    assert critic.q(obs, act).shape == (2, 5)
    assert torch.allclose(critic.q(obs, act), critic.hl.value(logits))
    assert not torch.allclose(logits[0], logits[1])


def test_critic_fits_fixed_targets_with_adam():
    torch.manual_seed(0)
    obs = torch.randn(64, 4)
    act = torch.tanh(torch.randn(64, 2))
    target = torch.rand(64) * 10 - 5  # inside the support, away from the edges
    critic = CategoricalCritic(obs_dim=4, act_dim=2, members=2, hidden=128)
    opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    for _ in range(500):
        loss = critic.hl.loss(critic(obs, act), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        err = (critic.q(obs, act) - target).abs().mean()
    assert err < 0.1
