"""Covariance treatment of the Xmax systematics."""
import numpy as np
import pytest
from scipy.optimize import minimize

from prince_analysis_tools.spectra import XRMS2026FD, Xmax2026FD
from prince_analysis_tools.xmax_sys import XmaxSystematics

SEL = Xmax2026FD["energy"] > 6e9
RNG = np.random.default_rng(3)


@pytest.fixture(scope="module")
def xsys():
    return XmaxSystematics(Xmax2026FD, XRMS2026FD, SEL)


def test_release_tables():
    x = Xmax2026FD
    for side in ("Up", "Low"):
        c = x["corr_" + side]
        assert c.shape == (20, 20)
        np.testing.assert_allclose(c, c.T)
        np.testing.assert_allclose(np.diag(c), 1.0)
        assert np.all(x["scale_sys_" + side] > 0)
        assert np.all(x["scale_sys_" + side] <= x["sys_" + side] + 0.05)


def _explicit_mean(xsys, delta, side):
    """Minimum over the mode pulls of the nuisance form, for one side."""
    m = xsys._modes[side]
    d = Xmax2026FD["stat"][SEL] ** 2 + np.clip(
        Xmax2026FD["sys_" + side][SEL] ** 2 - Xmax2026FD["scale_sys_" + side][SEL] ** 2, 0, None)
    fun = lambda a: ((delta - m @ a) ** 2 / d).sum() + a @ a
    res = minimize(fun, np.zeros(m.shape[1]), method="BFGS", options={"gtol": 1e-10})
    return res.fun, res.x


@pytest.mark.parametrize("offset", [12.0, -12.0, 3.0, -3.0])
def test_mean_equals_explicit_nuisances(xsys, offset):
    delta = offset + 2.0 * RNG.standard_normal(SEL.sum())
    side = "Up" if offset > 0 else "Low"
    chi2, a = _explicit_mean(xsys, delta, side)
    assert xsys.chi2_mean(delta) == pytest.approx(chi2, rel=1e-7)
    pulls = xsys.pulls(delta, np.zeros(SEL.sum()))
    assert pulls["side"] == side
    np.testing.assert_allclose(pulls["modes"], a, atol=1e-5)
    assert np.sign(pulls["modes"][0]) == np.sign(offset)


def test_one_sigma_scale_shift_costs_about_one(xsys):
    # exactly 1 for fully correlated bins; the published correlation is slightly below 1
    up = Xmax2026FD["scale_sys_Up"][SEL]
    low = Xmax2026FD["scale_sys_Low"][SEL]
    assert 0.9 < xsys.chi2_mean(up) < 1.1
    assert 0.9 < xsys.chi2_mean(-low) < 1.1
    stat_only = ((up / Xmax2026FD["stat"][SEL]) ** 2).sum()
    assert stat_only > 50


def test_stat_only_limit():
    x = {k: (v * 1e-6 if k.startswith(("scale_sys", "sys_")) else v) for k, v in Xmax2026FD.items()}
    r = {k: (v * 1e-6 if k.startswith("sys_") else v) for k, v in XRMS2026FD.items()}
    xs = XmaxSystematics(x, r, SEL)
    delta = RNG.standard_normal(SEL.sum())
    assert xs.chi2_mean(delta) == pytest.approx(((delta / x["stat"][SEL]) ** 2).sum(), rel=1e-6)
    assert xs.chi2_sigma(delta) == pytest.approx(((delta / r["stat"][SEL]) ** 2).sum(), rel=1e-6)


@pytest.mark.parametrize("offset", [2.0, -2.0])
def test_sigma_correlated_equals_explicit(xsys, offset):
    delta = offset + RNG.standard_normal(SEL.sum())
    s = XRMS2026FD["sys_Up" if offset > 0 else "sys_Low"][SEL]
    stat2 = XRMS2026FD["stat"][SEL] ** 2
    cov = np.diag(stat2) + np.outer(s, s)
    assert xsys.chi2_sigma(delta) == pytest.approx(delta @ np.linalg.solve(cov, delta), rel=1e-10)


def test_sigma_uncorrelated_is_quadrature():
    xs = XmaxSystematics(Xmax2026FD, XRMS2026FD, SEL, sigma_sys="uncorrelated")
    delta = RNG.standard_normal(SEL.sum())
    s = np.where(delta >= 0, XRMS2026FD["sys_Up"][SEL], XRMS2026FD["sys_Low"][SEL])
    expect = (delta**2 / (XRMS2026FD["stat"][SEL] ** 2 + s**2)).sum()
    assert xs.chi2_sigma(delta) == pytest.approx(expect)


def test_batched_equals_scalar(xsys):
    delta = 5.0 * RNG.standard_normal((50, SEL.sum())) + 4.0 * RNG.standard_normal((50, 1))
    np.testing.assert_allclose(xsys.chi2_mean(delta), [xsys.chi2_mean(d) for d in delta], rtol=1e-12)
    np.testing.assert_allclose(xsys.chi2_sigma(delta), [xsys.chi2_sigma(d) for d in delta], rtol=1e-12)


def test_missing_correlation_raises():
    x = {k: v for k, v in Xmax2026FD.items() if not k.startswith("corr_")}
    with pytest.raises(KeyError):
        XmaxSystematics(x, XRMS2026FD, SEL)
