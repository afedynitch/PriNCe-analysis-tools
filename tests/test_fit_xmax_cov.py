"""Covariance treatment inside the fit classes: batched and scalar paths agree."""
import numpy as np
import pytest

from prince_analysis_tools.spectra import XRMS2026FD, Xmax2026FD, auger2025
from prince_analysis_tools.ultranest_fit import UltraNestFractionsFit

PIDS = (2212, 1000020040, 1000070140)
# gamma, log10_rmax, m, deltaE, f_p, f_He (f_N = remainder)
THETAS = np.array([[1.0, 9.8, 0.0, 0.0, 0.2, 0.3], [1.8, 10.2, 2.0, 0.05, 0.5, 0.2]])


def _fit(run, **kw):
    return UltraNestFractionsFit(prince_run=run, spectrum=auger2025, xmax=Xmax2026FD, xrms=XRMS2026FD,
                                 particle_ids=PIDS, xmax_model="sibyll", use_corr=True, use_corr_residual=True,
                                 sclass="auger-sech", max_step=5e-3, solver="trbdf2", **kw)


@pytest.mark.parametrize("sigma_sys", ["correlated", "uncorrelated"])
def test_batched_equals_scalar(prince_run_m14, sigma_sys):
    fit = _fit(prince_run_m14, xmax_sys="cov", sigma_sys=sigma_sys)
    assert fit.n_params == THETAS.shape[1]
    chi2_batch = -2.0 * fit.loglike(THETAS)
    opts = [opt for _, opt in fit.recompute_fit_batch(THETAS)]
    np.testing.assert_allclose(chi2_batch, [o._inferred_chi2_total for o in opts], rtol=1e-8)
    pulls = opts[0].get_xmax_pulls()
    assert pulls["side"] in ("Up", "Low") and np.isfinite(pulls["modes"]).all()


def test_cov_is_below_stat_only(prince_run_m14):
    cov = -2.0 * _fit(prince_run_m14, xmax_sys="cov").loglike(THETAS)
    stat = -2.0 * _fit(prince_run_m14).loglike(THETAS)
    assert np.all(cov < stat)


def test_cov_excludes_sampled_shift(prince_run_m14):
    with pytest.raises(ValueError):
        _fit(prince_run_m14, xmax_sys="cov", sample_xmax_shift=True)
