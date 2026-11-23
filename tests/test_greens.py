"""Response-table forward model against the direct multi-RHS solve."""
import numpy as np

from prince_analysis_tools.greens import GreensWalker, _hat
from prince_analysis_tools.multirhs_walker import MultiRHSWalker

PIDS = (2212, 1000020040, 1000070140)
FRACTIONS = (0.2, 0.3, 0.5)


def _params(m):
    return [{"gamma": 1.0, "rmax": 5e9, "m": ("simple", m), "fractions": FRACTIONS},
            {"gamma": 2.0, "rmax": 2e10, "m": ("simple", m), "fractions": FRACTIONS}]


def _tables(run, n_z, m, dz=5e-3):
    gw = GreensWalker(run, PIDS, n_znodes=n_z, e_min_total=1.0, rmax_max=1e13).build(dz, chunk=500, progress=None)
    direct = MultiRHSWalker(run, None, None, None).compute_models_batch_combined(
        PIDS, _params(m), sclass="auger-sech", max_step=dz)
    table = gw.compute_models_batch_combined(PIDS, _params(m), sclass="auger-sech")
    return direct, table


def test_hats_partition_unity():
    nodes = np.linspace(0.0, np.log(2.0), 7)
    for x in np.linspace(0.0, np.log(2.0), 23):
        assert abs(sum(_hat(x, nodes, k) for k in range(len(nodes))) - 1.0) < 1e-12


def test_flat_evolution_exact(prince_run_m14):
    # (1+z)^0 is reproduced exactly by the hats: the table equals the direct solve to round-off
    direct, table = _tables(prince_run_m14, 3, 0.0)
    for d, t in zip(direct, table):
        np.testing.assert_allclose(t.state, d.state, rtol=1e-8, atol=1e-10 * np.abs(d.state).max())


def test_strong_evolution_converges(prince_run_m14):
    # (1+z)^m is interpolated between nodes; 41 nodes keep the flux within 1e-3
    direct, table = _tables(prince_run_m14, 41, 4.0)
    for d, t in zip(direct, table):
        fd, ft = d.get_solution_group("CR")[1], t.get_solution_group("CR")[1]
        sel = fd > 1e-3 * fd.max()
        np.testing.assert_allclose(ft[sel], fd[sel], rtol=1e-3)
