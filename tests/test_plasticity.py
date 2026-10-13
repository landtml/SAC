import pytest
import torch
from torch import nn

from softedge.agent.networks import RunningNorm, SimbaNet
from softedge.agent.plasticity import _dormant_masks, dormant_fraction, redo_reset, shrink_perturb

MEMBERS, D_IN, D_OUT, HIDDEN = 2, 6, 3, 16
UNITS = 4 * HIDDEN  # SimbaBlock expansion width


def _net(seed: int, blocks: int = 2) -> SimbaNet:
    torch.manual_seed(seed)
    return SimbaNet(MEMBERS, D_IN, D_OUT, hidden=HIDDEN, blocks=blocks)


def _probe(seed: int = 0, batch: int = 64) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(MEMBERS, batch, D_IN, generator=g)


def _kill(net: SimbaNet, units_per_block: dict[int, list[int]]) -> None:
    # A large negative bias keeps the pre-activation below zero, so ReLU emits exact zeros.
    with torch.no_grad():
        for b, units in units_per_block.items():
            net.blocks[b].fc1.bias[:, :, units] = -1e3


def _reference_dormant_count(net: SimbaNet, x: torch.Tensor, tau: float) -> int:
    """Count dormant units with the ratio test written out, without hooks."""
    count = 0
    with torch.no_grad():
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(MEMBERS, -1, -1)
        h = net.inp(x)
        for blk in net.blocks:
            score = torch.relu(blk.fc1(blk.norm(h))).mean(dim=1)  # (members, units)
            ratio = score / score.mean(dim=1, keepdim=True)
            count += int((ratio <= tau).sum())
            h = blk(h)
    return count


def test_shrink_perturb_alpha_one_keeps_parameters():
    model = _net(0)
    before = [p.detach().clone() for p in model.parameters()]
    shrink_perturb(model, _net(1), alpha=1.0)
    for p, b in zip(model.parameters(), before, strict=True):
        assert torch.equal(p, b)


def test_shrink_perturb_alpha_zero_copies_fresh():
    model, fresh = _net(0), _net(1)
    shrink_perturb(model, fresh, alpha=0.0)
    for p, q in zip(model.parameters(), fresh.parameters(), strict=True):
        assert torch.equal(p, q)


def test_shrink_perturb_blends_parameters_linearly():
    model, fresh = _net(0), _net(1)
    before = [p.detach().clone() for p in model.parameters()]
    shrink_perturb(model, fresh, alpha=0.8)
    for p, b, q in zip(model.parameters(), before, fresh.parameters(), strict=True):
        torch.testing.assert_close(p, 0.8 * b + 0.2 * q.detach())


def test_shrink_perturb_leaves_buffers_alone():
    model = nn.Sequential(nn.Linear(3, 2), RunningNorm(2))
    model[1].update(torch.randn(50, 2))
    weight_before = model[0].weight.detach().clone()
    buffers_before = {k: v.clone() for k, v in model.named_buffers()}
    shrink_perturb(model, nn.Sequential(nn.Linear(3, 2), RunningNorm(2)), alpha=0.5)
    assert not torch.equal(model[0].weight, weight_before)
    for k, v in model.named_buffers():
        assert torch.equal(v, buffers_before[k])


@pytest.mark.parametrize("kwargs", [{"hidden": 32, "blocks": 2}, {"hidden": HIDDEN, "blocks": 3}])
def test_shrink_perturb_rejects_mismatched_architecture(kwargs):
    model = _net(0)
    before = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(ValueError):
        shrink_perturb(model, SimbaNet(MEMBERS, D_IN, D_OUT, **kwargs), alpha=0.5)
    for p, b in zip(model.parameters(), before, strict=True):
        assert torch.equal(p, b)


def test_shrink_perturb_rejects_alpha_outside_unit_interval():
    with pytest.raises(ValueError):
        shrink_perturb(_net(0), _net(1), alpha=1.5)


def test_dormant_fraction_flags_units_with_negative_bias():
    net = _net(0)
    _kill(net, {0: [0, 1, 2, 3], 1: [5]})
    masks = _dormant_masks(net, _probe(), 0.025)
    assert masks[0][:, :4].all()
    assert masks[1][:, 5].all()


@pytest.mark.parametrize("tau", [0.025, 0.3])
@pytest.mark.parametrize("two_d", [True, False])
def test_dormant_fraction_matches_reference_definition(tau, two_d):
    net = _net(0)
    _kill(net, {0: [0, 1, 2, 3], 1: [5]})
    x = _probe()[0] if two_d else _probe()
    expected = _reference_dormant_count(net, x, tau) / (2 * MEMBERS * UNITS)
    assert dormant_fraction(net, x, tau=tau) == pytest.approx(expected)


def test_dormant_fraction_counts_all_zero_layer_as_dormant():
    net = _net(0, blocks=1)
    _kill(net, {0: list(range(UNITS))})
    assert dormant_fraction(net, _probe()) == 1.0


@pytest.mark.parametrize("train", [True, False])
def test_dormant_fraction_keeps_mode_and_removes_hooks(train):
    net = _net(0).train(train)
    dormant_fraction(net, _probe())
    assert net.training is train
    assert all(len(blk.fc1._forward_hooks) == 0 for blk in net.blocks)
    assert all(p.grad is None for p in net.parameters())


def test_redo_reset_revives_silent_units_and_keeps_output():
    net = _net(0)
    _kill(net, {0: [0, 1, 2, 3], 1: [5]})
    x = _probe()
    out_before = net(x).detach().clone()
    frac_before = dormant_fraction(net, x, tau=0.0)
    count = redo_reset(net, x, tau=0.0, generator=torch.Generator().manual_seed(0))
    assert count == round(frac_before * 2 * MEMBERS * UNITS)
    assert count >= 5 * MEMBERS
    assert dormant_fraction(net, x, tau=0.0) < frac_before
    # Only exactly silent units were reset, so their outgoing rows contributed nothing.
    torch.testing.assert_close(net(x).detach(), out_before)


def test_redo_reset_redraws_incoming_at_mean_norm_and_zeroes_outgoing():
    net = _net(0, blocks=1)
    _kill(net, {0: [0, 1]})
    x = _probe()
    fc1, fc2 = net.blocks[0].fc1, net.blocks[0].fc2
    mean_norm = fc1.weight.detach().norm(dim=1).mean(dim=1)
    w_before, b_before = fc1.weight.detach().clone(), fc1.bias.detach().clone()
    out_before = fc2.weight.detach().clone()
    count = redo_reset(net, x, generator=torch.Generator().manual_seed(0))
    assert count == MEMBERS * 2
    norms = fc1.weight.detach().norm(dim=1)
    torch.testing.assert_close(norms[:, :2], mean_norm[:, None].expand(-1, 2))
    assert torch.equal(fc1.bias.detach()[:, :, :2], torch.zeros(MEMBERS, 1, 2))
    assert torch.equal(fc2.weight.detach()[:, :2, :], torch.zeros(MEMBERS, 2, HIDDEN))
    assert torch.equal(fc1.weight.detach()[:, :, 2:], w_before[:, :, 2:])
    assert torch.equal(fc1.bias.detach()[:, :, 2:], b_before[:, :, 2:])
    assert torch.equal(fc2.weight.detach()[:, 2:, :], out_before[:, 2:, :])


@pytest.mark.parametrize("two_d", [True, False])
def test_redo_reset_count_matches_reference(two_d):
    net = _net(0)
    x = _probe()[0] if two_d else _probe()
    expected = _reference_dormant_count(net, x, 0.5)
    count = redo_reset(net, x, tau=0.5, generator=torch.Generator().manual_seed(1))
    assert expected > 0
    assert count == expected


def test_redo_reset_at_tau_zero_resets_exactly_silent_units():
    net = _net(0)
    _kill(net, {0: [0, 1, 2]})
    x = _probe()
    expected = _reference_dormant_count(net, x, 0.0)
    assert expected >= 3 * MEMBERS
    assert redo_reset(net, x, tau=0.0, generator=torch.Generator().manual_seed(0)) == expected
