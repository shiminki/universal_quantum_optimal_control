import torch

from uqoc.errors import HBN_DETUNINGS_MHZ, make_sampler


def test_ore_mhz_normalised_std():
    torch.manual_seed(0)
    # δ ~ N(0, 90² MHz²) with Ω = 30 MHz → δ/Ω ~ N(0, 3²)
    sampler = make_sampler("ore_mhz", {"delta_std_mhz": 90.0}, omega_mhz=30.0)
    err = sampler(200_000)
    assert err.shape == (2, 200_000)
    assert abs(err[0].std().item() - 3.0) < 0.05
    assert torch.all(err[1] == 0.0)


def test_ore_mhz_curriculum_param_overrides_omega():
    sampler = make_sampler("ore_mhz", {"delta_std_mhz": 90.0, "omega_mhz": 45.0},
                           omega_mhz=30.0)
    err = sampler(100_000)
    assert abs(err[0].std().item() - 2.0) < 0.05


def test_hbn_discrete_hits_only_lines():
    torch.manual_seed(0)
    sampler = make_sampler("hbn_discrete", {}, omega_mhz=30.0)
    err = sampler(10_000)
    lines = torch.tensor(HBN_DETUNINGS_MHZ) / 30.0
    dists = (err[0][:, None] - lines[None, :]).abs().min(dim=1).values
    assert dists.max() < 1e-6
    # all four lines drawn
    assert torch.unique(err[0]).numel() == 4


def test_dimensionless_samplers_ignore_omega():
    # `ore`/`ore_ple` don't declare omega_mhz — injection must not break them.
    sampler = make_sampler("ore_ple", {"delta_std": 0.4, "epsilon_std": 0.05},
                           omega_mhz=30.0)
    err = sampler(128)
    assert err.shape == (2, 128)
