import pytest
import torch
from torch.nn import functional as F

from sactrader.agent.hyperspherical import HyperNet

torch.set_num_threads(1)


def _hidden_layers(net):
    layers = [net.inp]
    for blk in net.blocks:
        layers += [blk.fc1, blk.fc2]
    return layers


@pytest.mark.parametrize("members", [1, 3])
@pytest.mark.parametrize("ndim", [2, 3])
def test_output_shape(members, ndim):
    torch.manual_seed(0)
    net = HyperNet(members, d_in=5, d_out=2, hidden=32)
    x = torch.randn(7, 5) if ndim == 2 else torch.randn(members, 7, 5)
    assert net(x).shape == (members, 7, 2)


def test_2d_input_matches_replicated_3d_input():
    torch.manual_seed(0)
    net = HyperNet(3, d_in=5, d_out=2, hidden=32)
    x = torch.randn(7, 5)
    assert torch.allclose(net(x), net(x.expand(3, -1, -1)))


def test_hidden_state_has_unit_norm():
    torch.manual_seed(0)
    net = HyperNet(3, d_in=5, d_out=1, hidden=32, blocks=2)
    h = net.encode(torch.randn(3, 11, 5))
    assert h.shape == (3, 11, 32)
    assert torch.allclose(h.norm(dim=-1), torch.ones(3, 11), atol=1e-5)


def test_project_normalizes_hidden_rows_and_leaves_output_alone():
    torch.manual_seed(0)
    net = HyperNet(2, d_in=5, d_out=1, hidden=32, blocks=2)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p))
    out_before = net.out.weight.clone()
    net.project_()
    for layer in _hidden_layers(net):
        norms = layer.weight.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert torch.equal(net.out.weight, out_before)
    assert torch.isfinite(net(torch.randn(4, 5))).all()


def test_members_are_independent():
    torch.manual_seed(0)
    net = HyperNet(2, d_in=5, d_out=1, hidden=32, blocks=2)
    x = torch.randn(9, 5)
    before = net(x).detach().clone()
    with torch.no_grad():
        for p in net.parameters():
            p[0].add_(torch.randn_like(p[0]))
    after = net(x).detach()
    assert not torch.allclose(after[0], before[0])
    assert torch.allclose(after[1], before[1], atol=1e-6)


def test_dropout_zero_is_deterministic_in_train_mode():
    torch.manual_seed(0)
    net = HyperNet(2, d_in=5, d_out=1, hidden=32, dropout=0.0).train()
    x = torch.randn(2, 6, 5)
    assert torch.equal(net(x), net(x))


def test_adam_regression_with_projection_reduces_mse():
    torch.manual_seed(0)
    net = HyperNet(1, d_in=4, d_out=1, hidden=64, blocks=1)
    x = torch.randn(1, 32, 4)
    target = torch.randn(1, 32, 1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    with torch.no_grad():
        start = F.mse_loss(net(x), target).item()
    for _ in range(300):
        opt.zero_grad()
        F.mse_loss(net(x), target).backward()
        opt.step()
        net.project_()
    with torch.no_grad():
        end = F.mse_loss(net(x), target).item()
    assert end <= 0.2 * start
