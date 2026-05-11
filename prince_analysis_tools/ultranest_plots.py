"""Posterior plotting helpers for :class:`UltraNestFit`.

Adopts the figure style of ``examples/evaluate.ipynb``:

- ``plot_corner`` — corner / triangle plot of the posterior over
  ``(gamma, log10_rmax, m)`` (UltraNest replacement for the Δχ² contour panel).
- ``plot_bestfit_spectrum`` — combined CR spectrum at the MAP with
  per-mass-group traces (A=1, 2≤A≤4, 5≤A≤14, 15≤A≤28, 29≤A≤56),
  Auger 2017 errorbars, gray Emin band — matching ``plot_cosmic_rays`` in
  the grid-scan notebook.
- ``plot_bestfit_xmax`` / ``plot_bestfit_xrms`` — mean Xmax(E) and RMS Xmax(E)
  with reference curves for A=1,4,14,56, systematic error boxes around the
  data, stat errorbars — matching ``plot_xmax_mean`` / ``plot_xmax_sigma``
  in the grid-scan notebook.
- ``plot_posterior_neutrino_band`` — neutrino flux band built from a random
  subsample of equal-weight posterior points (UltraNest's Bayesian
  replacement for the grid-derived Δχ² band).
"""

from __future__ import annotations

import numpy as np


# Mass-group bands used by evaluate.ipynb's ``plot_cosmic_rays``.
# Each tuple is (Amin, Amax, color, latex-label).
_MASS_GROUPS = [
    (1, 1,   "red",    r"$A = 1$"),
    (2, 4,   "gray",   r"$2 \leq A \leq 4$"),
    (5, 14,  "green",  r"$5 \leq A \leq 14$"),
    (15, 28, "orange", r"$15 \leq A \leq 28$"),
    (29, 56, "blue",   r"$29 \leq A \leq 56$"),
]

# Reference single-mass curves on the Xmax / RMS panels.
_XMAX_REFS = [
    (1,  "red",   "H"),
    (4,  "gray",  "He"),
    (14, "green", "N"),
    (56, "blue",  "Fe"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _combined_result(fit, theta):
    """Re-run forward model + inner fit at ``theta``; build the linear-
    combination :class:`UHECRPropagationResult` and pull the nuisance params.

    Returns ``(combined_result, deltaE, xshift, norms, minres, optimizer)``.
    """
    minres, opt = fit.recompute_fit(theta)
    deltaE = float(minres.values["deltaE"]) if "deltaE" in minres.values else 0.0
    xshift = float(minres.values["xmax_shift"]) if "xmax_shift" in minres.values else 0.0
    norms = np.array([float(minres.values["norm{}".format(p)]) for p in fit.particle_ids])

    combined = opt.lst_res[0] * float(norms[0])
    for r, n in zip(opt.lst_res[1:], norms[1:]):
        combined = combined + r * float(n)
    return combined, deltaE, xshift, norms, minres, opt


def _combined_result_batch(fit, theta_array):
    """Batched forward model + inner fit on a stack of ``theta`` points.

    Packs all forward solves into a single ``MultiRHSPropagationSolverETD2``
    call (K = N_θ × N_species), then loops the inner iMinuit fit. The
    multi-RHS bench at K=32 is the sweet spot but K up to ~200 is still
    fine (`runs/2026-05-11_multi-rhs-bench-mac/`).

    Yields ``(combined_result, deltaE, xshift, norms, minres, opt)`` per
    input theta, in the same order.
    """
    batch_params = [fit._theta_row_to_params(theta) for theta in theta_array]
    results_per_live = fit.walker.compute_models_batch(
        fit.particle_ids,
        batch_params,
        sclass=fit.sclass,
        initial_z=fit.initial_z,
        final_z=fit.final_z,
        max_step=fit.max_step,
    )
    for models in results_per_live:
        minres, opt = fit._inner_fit(models)
        deltaE = float(minres.values["deltaE"]) if "deltaE" in minres.values else 0.0
        xshift = (
            float(minres.values["xmax_shift"]) if "xmax_shift" in minres.values else 0.0
        )
        norms = np.array(
            [float(minres.values["norm{}".format(p)]) for p in fit.particle_ids]
        )
        combined = opt.lst_res[0] * float(norms[0])
        for r, n in zip(opt.lst_res[1:], norms[1:]):
            combined = combined + r * float(n)
        yield combined, deltaE, xshift, norms, minres, opt


def _data_stat_sys(data):
    """Pull out (stat_up, stat_low, sys_up, sys_low) from a data dict.

    Falls back gracefully when only the legacy combined `upper_err` /
    `lower_err` (stat ⊕ sys quadrature) are available. In that case the
    returned ``sys_*`` arrays are zeros and the combined error is
    reported as the stat side — the plotting picks up a single set of
    errorbars and skips the sys ribbon.
    """
    if "stat_up" in data and "stat_low" in data and "sys_Up" in data:
        return data["stat_up"], data["stat_low"], data["sys_Up"], data["sys_Low"]
    if "upper_err" in data and "lower_err" in data:
        zeros = np.zeros_like(data["upper_err"])
        return data["upper_err"], data["lower_err"], zeros, zeros
    raise KeyError("data dict has neither stat_up/stat_low + sys_Up/sys_Low nor upper_err/lower_err")


def _mass_group_select(amin, amax):
    """Build the (selector, vmin, vmax) tuple that PriNCe's
    ``get_solution_group`` understands for a mass range."""
    from prince_cr.util import get_AZN
    return (lambda pid: get_AZN(pid)[0], int(amin), int(amax))


def _find_nearest(array, value):
    array = np.asarray(array)
    return int((np.abs(array - value)).argmin())


def _make_error_boxes(ax, xdata, ydata, xerr, yerr, facecolor="gray", alpha=0.4):
    """Rectangular systematic-error boxes around each data point. Mirrors
    the ``make_error_boxes`` helper in the grid-scan notebook."""
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    xerr = np.asarray(xerr)
    yerr = np.asarray(yerr)
    rects = []
    for x, y, xe, ye in zip(xdata, ydata, xerr.T, yerr.T):
        rects.append(
            Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
        )
    pc = PatchCollection(rects, facecolor=facecolor, alpha=alpha, edgecolor="none")
    ax.add_collection(pc)


# ---------------------------------------------------------------------------
# Corner plot
# ---------------------------------------------------------------------------
def plot_corner(fit, savepath=None, labels=None, **corner_kwargs):
    """Corner plot of the posterior over outer parameters.

    Uses ``corner`` if installed, otherwise falls back to a plain matplotlib
    hist2d grid. Passes the prior bounds as ``range=`` so tiny / under-converged
    chains don't trigger corner.py's "no dynamic range" rejection.
    """
    import matplotlib.pyplot as plt

    samples = fit.posterior_samples(equal_weight=True)
    if isinstance(samples, tuple):
        pts, weights = samples
        idx = np.random.default_rng(0).choice(
            len(pts), size=min(len(pts), 10000), p=weights / weights.sum()
        )
        samples = pts[idx]

    if labels is None:
        labels = list(fit.param_names)

    try:
        import corner

        kw = dict(
            labels=labels,
            show_titles=True,
            title_fmt=".2f",
            quantiles=[0.16, 0.5, 0.84],
        )
        if "range" not in corner_kwargs:
            kw["range"] = [fit.bounds[name] for name in fit.param_names]
        kw.update(corner_kwargs)
        fig = corner.corner(samples, **kw)
    except ImportError:
        D = samples.shape[1]
        fig, axes = plt.subplots(D, D, figsize=(2.5 * D, 2.5 * D))
        for i in range(D):
            for j in range(D):
                ax = axes[i, j]
                if i == j:
                    ax.hist(samples[:, i], bins=40, histtype="step", color="k")
                elif j < i:
                    ax.hist2d(samples[:, j], samples[:, i], bins=40, cmap="Greys")
                else:
                    ax.set_visible(False)
                if i == D - 1:
                    ax.set_xlabel(labels[j])
                if j == 0 and i > 0:
                    ax.set_ylabel(labels[i])
        fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Best-fit spectrum — mass-group decomposition (evaluate.ipynb style)
# ---------------------------------------------------------------------------
def plot_bestfit_spectrum(fit, theta=None, savepath=None, ax=None,
                           figsize=(5, 3.5), data_label=None,
                           n_band=0, band_seed=0):
    """Combined CR spectrum at θ_MAP with per-mass-group decomposition.

    Mirrors ``plot_cosmic_rays`` in ``evaluate.ipynb`` and adds:

    - Data shown as **stat errorbars + asymmetric sys ribbon** (when
      `fit.spectrum` carries split `stat_up/stat_low` + `sys_Up/sys_Low`,
      as `auger2025` does). For legacy `auger2017`/`auger2019` dicts the
      single combined errorbar is shown.
    - Optional **posterior band** on the model curve(s): set
      ``n_band > 0`` to resample ``n_band`` equal-weight posterior
      points, re-run forward+inner fit at each (batched via one
      multi-RHS solve), and overlay 68/95 % bands on the combined
      curve. ``n_band=0`` (default) shows only the MAP curve.

    The inner-fit `deltaE` pull is applied to the model curve (the
    spectrum is shifted by ``E → E·(1+deltaE)``), so the displayed
    fit already accounts for the inferred energy-scale shift.
    """
    import matplotlib.pyplot as plt

    if theta is None:
        theta = fit.best_fit_theta()
    combined, deltaE, _xshift, _norms, _minres, _opt = _combined_result(fit, theta)
    if data_label is None:
        nbins = len(fit.spectrum["energy"])
        if nbins == 18:
            data_label = "Auger 2025"
        elif nbins == 27:
            data_label = "Auger 2017"
        else:
            data_label = "Auger"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for (amin, amax, color, label) in _MASS_GROUPS:
        sel = _mass_group_select(amin, amax)
        energy, spectrum = combined.get_solution_group(sel)
        energy = energy * (1.0 + deltaE)
        ax.loglog(energy, spectrum, color=color, alpha=0.6)
        spec_arr = np.asarray(spectrum)
        if spec_arr.size and np.nanmax(spec_arr) > 0:
            i = int(np.nanargmax(spec_arr))
            ax.annotate(
                label,
                (energy[i], spec_arr[i] / 1.5),
                color=color, weight="bold", fontsize=10,
                ha="center", va="top",
            )

    # Combined MAP curve
    energy_map, spectrum_map = combined.get_solution_group("CR")
    energy_map = energy_map * (1.0 + deltaE)

    # Posterior band on the combined curve
    if n_band > 0:
        samples = fit.posterior_samples(equal_weight=True)
        if isinstance(samples, tuple):
            pts, weights = samples
            rng_w = np.random.default_rng(band_seed)
            idx = rng_w.choice(len(pts), size=min(len(pts), 10000), p=weights / weights.sum())
            samples = pts[idx]
        rng = np.random.default_rng(band_seed)
        pick = rng.choice(len(samples), size=min(n_band, len(samples)), replace=False)
        chosen = samples[pick]
        curves = []
        for combined_i, deltaE_i, _, _, _, _ in _combined_result_batch(fit, chosen):
            e_i, s_i = combined_i.get_solution_group("CR")
            # Interpolate onto the MAP energy grid (in log-log space so
            # the energy shift doesn't introduce edge artefacts).
            log_e_i = np.log10(e_i * (1.0 + deltaE_i))
            with np.errstate(divide="ignore", invalid="ignore"):
                log_s = np.where(s_i > 0, np.log10(s_i), -np.inf)
            interp = np.interp(np.log10(energy_map), log_e_i, log_s,
                               left=-np.inf, right=-np.inf)
            curves.append(10.0 ** interp)
        band = np.array(curves)
        q = np.quantile(band, [0.025, 0.16, 0.84, 0.975], axis=0)
        ax.fill_between(energy_map, q[0], q[3], color="saddlebrown", alpha=0.10)
        ax.fill_between(energy_map, q[1], q[2], color="saddlebrown", alpha=0.25)

    ax.loglog(energy_map, spectrum_map, color="saddlebrown", lw=1.5)

    # Data: stat errorbars + sys ribbon if available
    data = fit.spectrum
    stat_up, stat_low, sys_up, sys_low = _data_stat_sys(data)
    sys_band = np.any(sys_up > 0) or np.any(sys_low > 0)
    if sys_band:
        ax.fill_between(
            data["energy"],
            data["spectrum"] - sys_low,
            data["spectrum"] + sys_up,
            color="black", alpha=0.10, step="mid", label="sys",
        )
    ax.errorbar(
        data["energy"], data["spectrum"],
        yerr=(stat_low, stat_up),
        fmt="o", color="black", markersize=3,
        label=data_label + (" (stat)" if sys_band else ""),
    )
    ax.annotate(
        data_label,
        (data["energy"][-4], data["spectrum"][-4] * 1.5),
        fontsize=12, ha="left", va="bottom",
    )

    # Sub-Emin gray band
    ax.fill_between([1e8, fit.Emin], 1e-1, 1e3, color="gray", alpha=0.4)

    ax.set_xlim(5e8, 5e11)
    ax.set_ylim(1e0, 1e3)
    ax.set_xlabel("E [GeV]")
    ax.set_ylabel(r"$E^3 J$  [GeV$^2$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    title = r"Best-fit @ $\theta_{\rm MAP}$" + r"   ($\Delta_E=${:+.3f}".format(deltaE)
    if n_band > 0:
        title += r", $n_{{\rm band}}={}$".format(n_band)
    title += ")"
    ax.set_title(title, fontsize=10)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


# ---------------------------------------------------------------------------
# Best-fit Xmax — reference curves + error boxes (evaluate.ipynb style)
# ---------------------------------------------------------------------------
def _build_xmax_model(name):
    from .xmax import XmaxSimple
    name = (name or "epos").lower()
    if name == "epos":
        return XmaxSimple(model=XmaxSimple.EPOSLHC)
    if name == "sibyll":
        return XmaxSimple(model=XmaxSimple.Sibyll23)
    if name == "qgsjet":
        return XmaxSimple(model=XmaxSimple.QGSJetII04)
    raise ValueError("Unknown Xmax model: {!r}".format(name))


def _xmax_panel(fit, theta, kind, savepath, ax, figsize, data_label,
                 n_band, band_seed):
    """Shared engine for ``plot_bestfit_xmax`` / ``plot_bestfit_xrms``.

    ``kind`` ∈ {"mean", "rms"}. Picks the appropriate data dict
    (`fit.xmax` / `fit.xrms`) and the ``XmaxSimple`` accessor
    (``get_mean_Xmax`` / ``get_var_Xmax``).
    """
    import matplotlib.pyplot as plt

    if theta is None:
        theta = fit.best_fit_theta()
    combined, deltaE, xshift, _norms, _minres, _opt = _combined_result(fit, theta)
    model = _build_xmax_model(fit.xmax_model)

    def curve_for(combined_i, deltaE_i):
        nuclei = [el for el in combined_i.known_species if el >= 100]
        egrid, mean_lnA, var_lnA = combined_i.get_lnA(nuclei)
        e_shift = egrid * (1.0 + deltaE_i)
        if kind == "mean":
            return e_shift, model.get_mean_Xmax(mean_lnA, e_shift)
        var, _ = model.get_var_Xmax(mean_lnA, var_lnA, e_shift)
        return e_shift, np.sqrt(var)

    e_map, y_map = curve_for(combined, deltaE)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Single-mass reference curves on the MAP energy grid
    for A, color, name in _XMAX_REFS:
        if kind == "mean":
            ref = model.get_mean_Xmax(np.log(A), e_map)
        else:
            v, _ = model.get_var_Xmax(np.log(A), 0.0, e_map)
            ref = np.sqrt(v)
        ax.semilogx(e_map, ref, color=color)
        idx = _find_nearest(e_map, 1e11)
        idx2 = min(idx + 1, e_map.size - 1)
        ax.annotate(name, (e_map[idx2], ref[idx]),
                    color=color, annotation_clip=False)

    # Posterior band on the combined curve
    if n_band > 0:
        samples = fit.posterior_samples(equal_weight=True)
        if isinstance(samples, tuple):
            pts, weights = samples
            rng_w = np.random.default_rng(band_seed)
            idx = rng_w.choice(len(pts), size=min(len(pts), 10000), p=weights / weights.sum())
            samples = pts[idx]
        rng = np.random.default_rng(band_seed)
        pick = rng.choice(len(samples), size=min(n_band, len(samples)), replace=False)
        chosen = samples[pick]
        curves = []
        for combined_i, deltaE_i, _, _, _, _ in _combined_result_batch(fit, chosen):
            e_i, y_i = curve_for(combined_i, deltaE_i)
            curves.append(np.interp(e_map, e_i, y_i,
                                    left=np.nan, right=np.nan))
        band = np.array(curves)
        with np.errstate(all="ignore"):
            q = np.nanquantile(band, [0.025, 0.16, 0.84, 0.975], axis=0)
        ax.fill_between(e_map, q[0], q[3], color="saddlebrown", alpha=0.10)
        ax.fill_between(e_map, q[1], q[2], color="saddlebrown", alpha=0.25)

    # Combined MAP curve on top
    ax.semilogx(e_map, y_map, color="saddlebrown", linestyle="solid", lw=1.5)

    # Data: sys boxes + stat errorbars + xshift correction on the data side
    data = fit.xmax if kind == "mean" else fit.xrms
    xerr = np.array((data["energy_Low"], data["energy_Up"]))
    yerr = np.array((data["sys_Low"], data["sys_Up"]))
    _make_error_boxes(ax, data["energy"], data["val"], xerr, yerr, facecolor="gray")

    if xshift > 0:
        xcorr = xshift * data["sys_Up"]
    else:
        xcorr = xshift * data["sys_Low"]
    ax.errorbar(
        data["energy"], data["val"] + xcorr,
        xerr=(data["energy_Low"], data["energy_Up"]),
        yerr=(data["stat"], data["stat"]),
        fmt="o", markersize=2.5, color="black",
        label=data_label + " (stat)",
    )

    ax.fill_between([1e8, fit.Emin], 1e-1, 1e3, color="gray", alpha=0.4)
    ax.set_xlim(1e9, 1e11)
    if kind == "mean":
        ax.set_ylim(600, 900)
        ax.set_ylabel(r"$\langle X_{\max} \rangle$ [g cm$^{-2}$]")
    else:
        ax.set_ylim(0, 80)
        ax.set_ylabel(r"$\sigma(X_{\max})$ [g cm$^{-2}$]")
    ax.set_xlabel("E [GeV]")
    title_bits = [
        r"$\theta_{\rm MAP}$",
        r"$\Delta_E={:+.3f}$".format(deltaE),
        r"$\Delta_{{X}}={:+.2f}$".format(xshift),
    ]
    if n_band > 0:
        title_bits.append(r"$n_{{\rm band}}={}$".format(n_band))
    ax.set_title(", ".join(title_bits), fontsize=10)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=400, bbox_inches="tight")
    return fig, ax


def plot_bestfit_xmax(fit, theta=None, savepath=None, ax=None,
                       figsize=(4, 3),
                       data_label=r"$\langle X_{\max}\rangle$",
                       n_band=0, band_seed=0):
    """Mean Xmax(E) at θ_MAP with A=1/4/14/56 reference curves, sys boxes
    + stat errorbars, and an optional posterior band on the combined fit.

    The data points are shifted by ``xmax_shift × sys`` (the inner-fit
    pull on the Xmax systematic) — same convention as
    ``plot_xmax_mean`` in ``evaluate.ipynb``.
    """
    return _xmax_panel(fit, theta, "mean", savepath, ax, figsize,
                       data_label, n_band, band_seed)


def plot_bestfit_xrms(fit, theta=None, savepath=None, ax=None,
                       figsize=(4, 3),
                       data_label=r"$\sigma(X_{\max})$",
                       n_band=0, band_seed=0):
    """σ(Xmax)(E) at θ_MAP. Same conventions as :func:`plot_bestfit_xmax`."""
    return _xmax_panel(fit, theta, "rms", savepath, ax, figsize,
                       data_label, n_band, band_seed)


# ---------------------------------------------------------------------------
# Posterior neutrino-flux band
# ---------------------------------------------------------------------------
def plot_posterior_neutrino_band(
    fit,
    n_samples=64,
    epow=2,
    quantiles=(0.025, 0.16, 0.5, 0.84, 0.975),
    savepath=None,
    ax=None,
    rng_seed=0,
):
    """Posterior neutrino flux band built from a random subsample of
    equal-weight posterior points. Median + 68/95% bands."""
    import matplotlib.pyplot as plt

    samples = fit.posterior_samples(equal_weight=True)
    if isinstance(samples, tuple):
        pts, weights = samples
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(pts), size=min(len(pts), 10000), p=weights / weights.sum())
        samples = pts[idx]

    rng = np.random.default_rng(rng_seed)
    pick = rng.choice(len(samples), size=min(n_samples, len(samples)), replace=False)
    chosen = samples[pick]

    nu_egrid = None
    band = []
    for theta in chosen:
        combined, _dE, _xs, _norms, _m, _opt = _combined_result(fit, theta)
        egrid, spec_nu = combined.get_solution_group("nu", epow=epow)
        nu_egrid = egrid
        band.append(spec_nu)
    band = np.array(band)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    q = np.quantile(band, quantiles, axis=0)
    ax.fill_between(nu_egrid, q[0], q[-1], color="C0", alpha=0.15, label="95%")
    ax.fill_between(nu_egrid, q[1], q[-2], color="C0", alpha=0.30, label="68%")
    ax.plot(nu_egrid, q[2], color="C0", lw=2, label="median")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("E [GeV]")
    ax.set_ylabel(r"$E^{{{}}} \Phi_\nu$".format(epow))
    # Clip to the cosmogenic-neutrino window of interest — outside this
    # range the propagated spectrum drops off precipitously.
    ax.set_xlim(1e5, 1e11)
    # Use the band's own dynamic range to set y; avoid the 1e-52 numerical
    # underflow at the tail of the egrid swamping the plot.
    band_q = q[2][(nu_egrid >= 1e5) & (nu_egrid <= 1e11)]
    band_q = band_q[band_q > 0]
    if band_q.size:
        ymax = band_q.max() * 5
        ymin = max(band_q.max() * 1e-6, 1e-20)
        ax.set_ylim(ymin, ymax)
    ax.legend(loc="best", fontsize=9)
    ax.set_title("Posterior neutrino flux band (n={})".format(len(chosen)))
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, ax, nu_egrid, band
