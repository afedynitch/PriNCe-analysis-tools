"""Forward model on Prince Ultra: solver selection and the combined-source path."""
import numpy as np
import pytest

from prince_analysis_tools.multirhs_walker import MultiRHSWalker

PIDS = (2212, 1000020040, 1000070140)
FRACTIONS = (0.2, 0.3, 0.5)
PARAMS = [
    {"gamma": 1.0, "rmax": 5e9, "m": ("simple", 0.0), "fractions": FRACTIONS},
    {"gamma": 2.0, "rmax": 2e10, "m": ("simple", 3.0), "fractions": FRACTIONS},
]


def _flux(result):
    """All-particle E^3 J on the run's grid."""
    return result.get_solution_group("CR")[1]


def test_unknown_solver_raises():
    with pytest.raises(ValueError):
        MultiRHSWalker(None, None, None, None, solver="bdf")


def test_combined_source_equals_weighted_species(prince_run_m14):
    walker = MultiRHSWalker(prince_run_m14, None, None, None)
    combined = walker.compute_models_batch_combined(PIDS, PARAMS, sclass="auger", max_step=5e-3)
    species = walker.compute_models_batch(PIDS, PARAMS, sclass="auger", max_step=5e-3)
    assert len(combined) == len(PARAMS)
    for comb, per_species in zip(combined, species):
        # linear transport: exact on the state; the interpolated group spectra are not linear in it
        mixed = sum(f * r.state for f, r in zip(FRACTIONS, per_species))
        np.testing.assert_allclose(comb.state, mixed, rtol=1e-9, atol=1e-12 * np.abs(mixed).max())
        assert np.isfinite(_flux(comb)).all()


def test_etd2_and_trbdf2_agree(prince_run_m14):
    etd2 = MultiRHSWalker(prince_run_m14, None, None, None, solver="etd2")
    trbdf2 = MultiRHSWalker(prince_run_m14, None, None, None, solver="trbdf2")
    a = etd2.compute_models_batch_combined(PIDS, PARAMS, sclass="auger-sech", max_step=1e-3)
    b = trbdf2.compute_models_batch_combined(PIDS, PARAMS, sclass="auger-sech", max_step=5e-3)
    for ra, rb in zip(a, b):
        fa, fb = _flux(ra), _flux(rb)
        sel = fa > 1e-3 * fa.max()
        np.testing.assert_allclose(fb[sel], fa[sel], rtol=0.03)
