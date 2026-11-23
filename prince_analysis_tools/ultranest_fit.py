"""UltraNest fit driver for UHECR source-parameter inference.

Mirrors the workflow of the existing grid scan (``create_project.py`` +
``recompute_fit.py``) but replaces the brute-force 81 × 61 × 61 grid with
nested sampling over the three outer parameters ``(gamma, rmax, m)``.

The likelihood is the *profile* chi² over the inner nuisance parameters
(``deltaE``, ``xmax_shift`` and the per-species norms), computed by the same
:class:`UHECROptimizer.fit_data_minuit` call that the grid scan uses for each
cell. We keep the chi² definition, the data fixtures and the iMinuit-based
inner fit identical to the grid path, so any difference in the resulting
posterior modes vs the grid maximum reflects the sampler, not the model.

Forward model uses :class:`MultiRHSWalker`, which batches ``K_live × N_species``
RHSs into a single ETD2 multi-RHS solve. At K=32 on Mac the bench (see
``wiki/results/multi-rhs-mac-bench.md``) reports ~32× per-RHS speedup; UltraNest
typically draws live-point batches of a few × 10 to 100 per loglike call, so
this lands near the throughput sweet-spot of the multi-RHS path.

References
----------
- UltraNest: https://johannesbuchner.github.io/UltraNest/
- Existing grid scan: ``examples/create_project.py``, ``examples/recompute_fit.py``
- Forward-model wrapper: :mod:`.multirhs_walker`
"""

from __future__ import annotations

import os
import numpy as np

from .multirhs_walker import MultiRHSWalker
from .optimizer import UHECROptimizer


DEFAULT_PARAM_BOUNDS = {
    "gamma": (-1.5, 2.5),
    "log10_rmax": (8.5, 11.5),  # rmax in GV, log10 spans 3e8..3e11 GV
    "m": (-6.0, 6.0),
}

DEFAULT_PARAM_NAMES = ("gamma", "log10_rmax", "m")


def paramlist_to_bounds(paramlist):
    """Extract ``(low, high)`` bounds for each axis from a grid ``paramlist``.

    ``paramlist`` is the analysis-tools tuple-of-tuples format
    ``(("gamma", linspace), ("rmax", logspace), ("m", linspace))``. For
    ``rmax`` the bounds are converted to ``log10_rmax``.
    """
    bounds = {}
    for name, arr in paramlist:
        arr = np.asarray(arr)
        if name == "rmax":
            bounds["log10_rmax"] = (float(np.log10(arr.min())), float(np.log10(arr.max())))
        else:
            bounds[name] = (float(arr.min()), float(arr.max()))
    return bounds


class UltraNestFit(object):
    """Nested-sampling driver wrapping :class:`MultiRHSWalker` + ``UHECROptimizer``.

    Parameters
    ----------
    prince_run : PriNCeRun
        Pre-built PriNCe kernel object (same one the grid scan loads from
        its pickle).
    spectrum, xmax, xrms : dict
        Auger data fixtures (e.g. ``auger2017``, ``Xmax2017``, ``XRMS2017``
        from :mod:`.spectra`).
    particle_ids : sequence of int
        Injection species PDG ids. Defaults to the modern PDG-encoded 5-species
        set ``(2212, 1000020040, 1000070140, 1000140280, 1000260560)``
        = (p, He-4, N-14, Si-28, Fe-56). The cached pickle must be built at
        ``max_mass ≥ 56`` for all five to be in ``spec_man.pdgid2sref``. (The
        legacy ``A·100+Z`` IDs ``[101, 402, 1407, 2814, 5626]`` used by
        ``examples/create_project.py`` won't resolve in current PriNCe.)
    bounds : dict, optional
        Flat-prior bounds keyed by ``param_names``. Defaults to
        :data:`DEFAULT_PARAM_BOUNDS`.
    param_names : tuple of str
        Outer parameter names. Default ``("gamma", "log10_rmax", "m")``.
    xmax_model : {"epos", "sibyll", "qgsjet"} or None
        Forwarded to :class:`UHECROptimizer`.
    Emin : float
        Energy cut for the chi² sum.
    minimizer_args : dict
        Passed to :meth:`UHECROptimizer.fit_data_minuit`. Default
        ``{"fix_deltaE": False}`` matches the canonical "floating_E_Sibyll"
        recompute-fit of the grid scan.
    spectrum_only : bool or str
        Forwarded to ``fit_data_minuit``. ``False`` (default) uses the full
        spectrum+Xmax+XRMS chi².
    use_corr : bool
        **Opt-in correlation-aware spectrum χ²** (see
        :class:`UHECROptimizer`). Default False keeps the existing
        diagonal χ² used by the grid scan; set True to fold the
        spectrum bin-to-bin correlation matrix into the chi² (e.g. when
        running against the ``auger2025`` dict which carries a `corr`
        key derived from Table VI of PRL 135:241002).
    apply_auger_xmax_shift : bool
        **Opt-in Auger-2024 ΔXmax shift on the model ⟨Xmax⟩.** When True
        (and ``xmax_model`` is one of {"epos","sibyll","qgsjet"}), each
        inner iMinuit fit floats an extra nuisance ``auger_dxmax``
        (g cm^-2) that shifts the predicted ⟨Xmax⟩ only (σ(Xmax) is
        untouched). A per-HIM asymmetric-Gaussian prior centred on the
        Table III measurement of PRD 109, 102001 (2024) — Sibyll-2.3d:
        29 ⁺¹⁰⁻¹³, EPOS-LHC: 22 ⁺¹¹⁻¹⁴, QGSJet-II-04: 47 ⁺⁹⁻¹¹ g cm^-2
        (combining stat ⊕ sys in quadrature on each side) — contributes
        a Gaussian penalty to the χ². Default False reproduces the
        pre-2026-05-12 fits exactly.
    initial_z, final_z, max_step
        Forwarded to the multi-RHS solver.
    solver : {"etd2", "trbdf2", "be"}
        Integrator of the forward model (see :class:`MultiRHSWalker`).
    """

    def __init__(
        self,
        prince_run,
        spectrum,
        xmax,
        xrms,
        particle_ids=(2212, 1000020040, 1000070140, 1000140280, 1000260560),
        bounds=None,
        param_names=DEFAULT_PARAM_NAMES,
        xmax_model="sibyll",
        Emin=6e9,
        minimizer_args=None,
        spectrum_only=False,
        use_corr=False,
        apply_auger_xmax_shift=False,
        sclass="auger",
        m_kind="simple",
        initial_z=1.0,
        final_z=0.0,
        max_step=1e-2,
        solver="etd2",
    ):
        self.prince_run = prince_run
        self.spectrum = spectrum
        self.xmax = xmax
        self.xrms = xrms
        self.particle_ids = list(particle_ids)
        self.param_names = tuple(param_names)
        self.bounds = dict(DEFAULT_PARAM_BOUNDS)
        if bounds is not None:
            self.bounds.update(bounds)
        for name in self.param_names:
            if name not in self.bounds:
                raise KeyError(
                    "No prior bound provided for parameter {!r}".format(name)
                )
        self.xmax_model = xmax_model
        self.Emin = Emin
        self.minimizer_args = (
            {"fix_deltaE": False} if minimizer_args is None else dict(minimizer_args)
        )
        self.spectrum_only = spectrum_only
        self.use_corr = bool(use_corr)
        self.apply_auger_xmax_shift = bool(apply_auger_xmax_shift)
        self.sclass = sclass
        self.m_kind = m_kind
        self.initial_z = initial_z
        self.final_z = final_z
        self.max_step = max_step

        self.walker = MultiRHSWalker(
            prince_run, spectrum, xmax, xrms, progressbar=False, solver=solver
        )
        self.sampler = None
        self.result = None
        # Diagnostics — counters incremented across loglike calls.
        self._n_loglike_calls = 0
        self._n_live_evals = 0

    # ------------------------------------------------------------------
    # Prior transform — unit cube → physical
    # ------------------------------------------------------------------
    def prior_transform(self, u):
        """Flat priors over the configured ``bounds``.

        Works for both vectorized (shape ``(N, D)``) and 1-D inputs.
        """
        u = np.asarray(u)
        out = np.empty_like(u, dtype=float)
        for i, name in enumerate(self.param_names):
            lo, hi = self.bounds[name]
            if u.ndim == 1:
                out[i] = lo + (hi - lo) * u[i]
            else:
                out[:, i] = lo + (hi - lo) * u[:, i]
        return out

    # ------------------------------------------------------------------
    # theta → batch_params dict list — converts parameter convention
    # (log10_rmax → rmax, m → ("simple", m), …) into the multi-RHS walker's
    # input.
    # ------------------------------------------------------------------
    def _theta_row_to_params(self, theta_row):
        out = {}
        for i, name in enumerate(self.param_names):
            val = float(theta_row[i])
            if name == "log10_rmax":
                out["rmax"] = 10.0 ** val
            elif name == "m":
                if self.m_kind == "simple":
                    out["m"] = ("simple", val)
                else:
                    out["m"] = val
            else:
                out[name] = val
        # Defaults for parameters that aren't being sampled:
        if "rmax" not in out:
            out["rmax"] = 10.0 ** (
                0.5 * sum(self.bounds.get("log10_rmax", (8.5, 11.5)))
            )
        if "gamma" not in out:
            out["gamma"] = 0.5 * sum(self.bounds.get("gamma", (-1.5, 2.5)))
        if "m" not in out:
            out["m"] = (self.m_kind, 0.0) if self.m_kind == "simple" else 0.0
        return out

    # ------------------------------------------------------------------
    # Inner profile chi² fit for one livepoint
    # ------------------------------------------------------------------
    def _inner_fit(self, models):
        opt = UHECROptimizer(
            models,
            self.spectrum,
            self.xmax,
            self.xrms,
            Emin=self.Emin,
            ncoids=self.particle_ids,
            xmax_model=self.xmax_model,
            use_corr=self.use_corr,
            apply_auger_xmax_shift=self.apply_auger_xmax_shift,
        )
        params = {"fix_deltaE": True, "fix_xmax_shift": True}
        params.update(self.minimizer_args)
        minres = opt.fit_data_minuit(
            spectrum_only=self.spectrum_only, minimizer_args=params
        )
        return minres, opt

    # ------------------------------------------------------------------
    # Vectorized loglike — UltraNest entry point
    # ------------------------------------------------------------------
    def loglike(self, theta_batch):
        theta_batch = np.atleast_2d(np.asarray(theta_batch, dtype=float))
        K_live = theta_batch.shape[0]

        batch_params = [self._theta_row_to_params(theta_batch[j]) for j in range(K_live)]
        results_per_live = self.walker.compute_models_batch(
            self.particle_ids,
            batch_params,
            sclass=self.sclass,
            initial_z=self.initial_z,
            final_z=self.final_z,
            max_step=self.max_step,
        )

        logl = np.empty(K_live, dtype=float)
        for j, models in enumerate(results_per_live):
            try:
                minres, _opt = self._inner_fit(models)
                chi2 = float(minres.fval)
            except Exception:
                # iMinuit can fail on degenerate cells (e.g. all-zero spectra
                # outside the propagation range). Treat as zero-likelihood.
                chi2 = np.inf
            if not np.isfinite(chi2):
                # UltraNest accepts -np.inf; keep finite but very negative for
                # robustness across versions.
                logl[j] = -1e30
            else:
                logl[j] = -0.5 * chi2

        self._n_loglike_calls += 1
        self._n_live_evals += K_live
        return logl

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------
    def run(
        self,
        log_dir=None,
        resume="resume",
        min_num_live_points=400,
        dlogz=0.5,
        max_iters=None,
        max_ncalls=None,
        vectorized=True,
        show_status=True,
        sampler_kwargs=None,
        **run_kwargs,
    ):
        """Run the nested sampler.

        ``sampler_kwargs`` are forwarded to
        :class:`ultranest.ReactiveNestedSampler` (constructor). ``**run_kwargs``
        are forwarded to :meth:`ReactiveNestedSampler.run` and override the
        defaults set here.

        Returns the UltraNest result dict and stores the sampler on
        ``self.sampler`` and the result on ``self.result``.
        """
        import ultranest

        sampler_kwargs = dict(sampler_kwargs or {})
        sampler = ultranest.ReactiveNestedSampler(
            list(self.param_names),
            self.loglike,
            self.prior_transform,
            log_dir=log_dir,
            resume=resume,
            vectorized=vectorized,
            **sampler_kwargs,
        )

        kw = dict(
            min_num_live_points=min_num_live_points,
            dlogz=dlogz,
            show_status=show_status,
        )
        if max_iters is not None:
            kw["max_iters"] = max_iters
        if max_ncalls is not None:
            kw["max_ncalls"] = max_ncalls
        kw.update(run_kwargs)

        result = sampler.run(**kw)
        self.sampler = sampler
        self.result = result
        return result

    # ------------------------------------------------------------------
    # Post-processing helpers — equivalents of ScanPlotter.get_comb_result
    # but driven by posterior samples.
    # ------------------------------------------------------------------
    def best_fit_theta(self):
        """Returns the maximum-likelihood theta from the sampled chain."""
        if self.result is None:
            raise RuntimeError("Call .run() first.")
        # Prefer the explicit MAP if UltraNest exposed it; else take argmax.
        if "maximum_likelihood" in self.result:
            return np.asarray(self.result["maximum_likelihood"]["point"], dtype=float)
        samples = np.asarray(self.result["weighted_samples"]["points"])
        logl = np.asarray(self.result["weighted_samples"]["logl"])
        return samples[int(np.argmax(logl))]

    def recompute_fit(self, theta):
        """Re-run forward model + inner fit at a single point ``theta``.

        Returns ``(minres, optimizer)`` matching
        :meth:`ScanPlotter.recompute_fit`. Useful for pulling out the
        best-fit norms / fractions / shifts at a posterior point.
        """
        theta = np.asarray(theta, dtype=float)
        batch_params = [self._theta_row_to_params(theta)]
        results = self.walker.compute_models_batch(
            self.particle_ids,
            batch_params,
            sclass=self.sclass,
            initial_z=self.initial_z,
            final_z=self.final_z,
            max_step=self.max_step,
        )
        return self._inner_fit(results[0])

    def posterior_samples(self, equal_weight=True):
        """Return posterior samples — equal-weight by default."""
        if self.result is None:
            raise RuntimeError("Call .run() first.")
        if equal_weight and "samples" in self.result:
            return np.asarray(self.result["samples"])
        ws = self.result["weighted_samples"]
        return np.asarray(ws["points"]), np.asarray(ws["weights"])

    def summary(self):
        """One-shot textual summary — logZ, posterior medians, MAP."""
        if self.result is None:
            return "<UltraNestFit: no result yet>"
        r = self.result
        out = []
        out.append("logZ = {:.3f} ± {:.3f}".format(r["logz"], r["logzerr"]))
        if "samples" in r:
            samples = np.asarray(r["samples"])
            for i, name in enumerate(self.param_names):
                q = np.quantile(samples[:, i], [0.16, 0.5, 0.84])
                out.append(
                    "  {} = {:+.3f}  (68%: {:+.3f}, {:+.3f})".format(
                        name, q[1], q[0], q[2]
                    )
                )
        out.append(
            "n_loglike_calls = {}, total_live_evals = {}".format(
                self._n_loglike_calls, self._n_live_evals
            )
        )
        return "\n".join(out)


# ======================================================================
# UltraNestFractionsFit — no inner iMinuit; mass fractions + deltaE
# sampled by UltraNest, total normalization profiled analytically.
# ======================================================================
#
# Motivation
# ----------
# :class:`UltraNestFit` above samples three outer parameters
# (gamma, log10_rmax, m) and runs an iMinuit profile over (deltaE,
# xmax_shift, [auger_dxmax,] per-species norms) per live point. When
# ``fix_deltaE=False`` the 5-start delta-tries grid dominates wall
# time at ~900 ms/livepoint vs ~50 ms for the cupy multi-RHS forward
# solve. The GPU's K-RHS capacity then can't drive throughput — see
# UH-UHECR-Fluka-Prince/wiki/results/ultranest-bigK-prod.md.
#
# Structural observation
# ----------------------
# In :meth:`UHECROptimizer.compute_combined_result`:
#   spectrum = sum_i (norms_i * arr_spectrum_i)         (linear in norms)
#   mean_lnA = sum_i (norms_i * w_i * mean_lnA_i)
#            / sum_i (norms_i * w_i)                    (ratio — invariant
#                                                        to global scaling)
#   var_lnA  = same ratio structure                     (ratio)
# So ``res_xmax`` and ``res_sigma_xmax`` depend on *fractions* only;
# the total normalization N is a single linear scalar in the spectrum
# chi² with a closed-form least-squares profile:
#   correlation-aware:  N* = (m C^-1 d) / (m C^-1 m), exact at fixed sign mask.
#   diagonal asymm err: N* = sum(d·m/σ²) / sum(m²/σ²), 2-pass sign-stabilized.
#
# This lets us move the 5 per-species norms + deltaE into the outer
# UltraNest sampling space (Dirichlet stick-breaking on the fractions
# so they sum to 1) and analytically profile only the total norm. No
# iMinuit anywhere in the per-livepoint loop. Smoke
# (UH-UHECR-Fluka-Prince/runs/2026-05-23_ultranest-fractions-test):
# Δχ²(new−prod) = 0.0000 at the prod-fit MAP; per-livepoint wall
# 74 ms (50 fwd + 24 chi²) vs prod's ~950 ms — ~12.7× speedup per
# likelihood eval, GPU now the bottleneck.

FRACTIONS_PARAM_NAMES = (
    "gamma",
    "log10_rmax",
    "m",
    "deltaE",
)

# Sampling bounds for the source/nuisance dims. Fractions are not in
# this dict — they live on the (n_species-1)-dim unit simplex via
# stick-breaking with a Dirichlet(1,…,1) prior.
FRACTIONS_SOURCE_BOUNDS = {
    "gamma": (-1.5, 2.5),
    "log10_rmax": (8.5, 11.5),
    "m": (-6.0, 6.0),
    "deltaE": (-0.14, 0.14),
}


def stick_break_dirichlet(u_subset, k):
    """Map (k-1) unit-cube draws to a Dirichlet(1,…,1) draw on the k-simplex.

    Inverse-CDF stick-breaking: ``v_j = u_j**(1/(k-1-j))`` is Beta(1, k-1-j)-
    distributed; the j-th simplex component is ``stick * (1 - v_j)`` and the
    remaining mass shrinks by ``stick *= v_j``. Each fraction is monotone in
    its own unit-cube coordinate so UltraNest's region sampling stays stable.

    Parameters
    ----------
    u_subset : array, shape ``(k-1,)`` or ``(M, k-1)``
        Unit-cube draws.
    k : int
        Number of simplex components (species).

    Returns
    -------
    fractions : array, shape ``(k,)`` or ``(M, k)``
        Strictly positive, sums to 1 along the last axis.
    """
    u = np.asarray(u_subset, dtype=float)
    if u.shape[-1] != k - 1:
        raise ValueError(
            "u_subset must have last-axis size k-1; got {} for k={}".format(
                u.shape[-1], k
            )
        )
    out_shape = u.shape[:-1] + (k,)
    out = np.empty(out_shape, dtype=float)
    if u.ndim == 1:
        stick = 1.0
        for j in range(k - 1):
            vj = u[j] ** (1.0 / (k - 1 - j))
            out[j] = stick * (1.0 - vj)
            stick = stick * vj
        out[-1] = stick
    else:
        stick = np.ones(u.shape[:-1])
        for j in range(k - 1):
            vj = u[..., j] ** (1.0 / (k - 1 - j))
            out[..., j] = stick * (1.0 - vj)
            stick = stick * vj
        out[..., -1] = stick
    return out


def profile_norm_diag(model_unit, data, err_up, err_low, indices, max_iter=3):
    """Closed-form profile of total normalization in the diagonal asymmetric-err χ².

    Model is ``N * model_unit`` for a positive scalar ``N``. At fixed
    sign mask ``sign_above = (N*m > d)``, ``σ = err_up`` where above and
    ``err_low`` below, giving
        N* = sum(d·m / σ²) / sum(m² / σ²)
    The sign flips with N, so we iterate (typically 1–2 passes).

    Parameters
    ----------
    model_unit, data, err_up, err_low : array, shape (n_bins_total,)
    indices : 1-D int array or boolean mask
        Spectrum bins included in the chi² (e.g. ``egrid > Emin``).

    Returns
    -------
    chi2 : float
        Profile chi² (minimum over N).
    N_star : float
        The closed-form optimum.
    """
    m = np.asarray(model_unit[indices]).ravel()
    d = np.asarray(data[indices]).ravel()
    eu = np.asarray(err_up[indices]).ravel()
    el = np.asarray(err_low[indices]).ravel()

    sign_above = m > d
    N_prev = -1.0
    for _ in range(max_iter):
        sigma = np.where(sign_above, eu, el)
        sigma_sq = sigma * sigma
        valid = sigma_sq > 0
        num = (m[valid] * d[valid] / sigma_sq[valid]).sum()
        den = (m[valid] * m[valid] / sigma_sq[valid]).sum()
        N = num / den if den > 0 else 0.0
        N = max(N, 1e-300)
        new_sign = (N * m) > d
        if np.array_equal(new_sign, sign_above) and abs(N - N_prev) < 1e-8 * max(N, 1e-30):
            break
        sign_above = new_sign
        N_prev = N
    sigma = np.where(sign_above, eu, el)
    sigma_sq = sigma * sigma
    valid = sigma_sq > 0
    chi2 = ((N * m[valid] - d[valid]) ** 2 / sigma_sq[valid]).sum()
    return float(chi2), float(N)


def _fill_log_interior(fp_K, log_xp):
    """Fill non-positive bins with log-interpolated values from the nearest
    positive neighbors, per walker.

    This is what makes ``_batched_loglog_interp`` agree with PriNCe's
    scalar ``np.interp`` on ``(curr_egrid[mask], curr_spec[mask])``: PriNCe
    drops the non-positive bins entirely before interp, so a query
    between a negative bin and a positive one gets bracketed by the two
    nearest POSITIVE bins on either side. Our batched code (which uses
    the full xp array) instead would treat the negative bin as a hard
    masked point. Filling those bins with the log-linear interp of their
    surrounding positive neighbors makes the bracketing equivalent.

    Edge non-positive values (no positive neighbor on one side) remain
    zero — matches the OOR behavior of ``np.interp(..., left=NaN,
    right=NaN)`` after ``nan_to_num``.

    Parameters
    ----------
    fp_K : ndarray, shape ``(K, N)``
        Linear-space values; non-positive bins are the masked-out ones.
    log_xp : ndarray, shape ``(N,)``
        Log of the native energy grid (shared across K).

    Returns
    -------
    out : ndarray, shape ``(K, N)`` — linear-space, strictly positive on
    the closed convex hull of each row's positive bins, zero outside.
    """
    K, N = fp_K.shape
    valid = fp_K > 0
    arr = np.arange(N)
    valid_idx_fwd = np.where(valid, arr[None, :], -1)
    fwd_idx = np.maximum.accumulate(valid_idx_fwd, axis=1)            # (K, N) or -1
    valid_idx_bwd = np.where(valid, arr[None, :], N)
    bwd_idx = np.minimum.accumulate(valid_idx_bwd[:, ::-1], axis=1)[:, ::-1]  # or N
    has_fwd = fwd_idx >= 0
    has_bwd = bwd_idx < N

    fwd_safe = np.where(has_fwd, fwd_idx, 0)
    bwd_safe = np.where(has_bwd, bwd_idx, N - 1)

    log_fp_safe = np.where(valid, np.log(np.where(valid, fp_K, 1.0)), 0.0)
    log_x_arr = np.broadcast_to(log_xp, (K, N))
    fwd_log_fp = np.take_along_axis(log_fp_safe, fwd_safe, axis=1)
    fwd_log_x = np.take_along_axis(log_x_arr, fwd_safe, axis=1)
    bwd_log_fp = np.take_along_axis(log_fp_safe, bwd_safe, axis=1)
    bwd_log_x = np.take_along_axis(log_x_arr, bwd_safe, axis=1)

    same_idx = fwd_safe == bwd_safe
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = bwd_log_x - fwd_log_x
        w = np.where(same_idx, 0.0, (log_xp[None, :] - fwd_log_x) / np.where(same_idx, 1.0, denom))
    interp_log = fwd_log_fp + w * (bwd_log_fp - fwd_log_fp)

    both = has_fwd & has_bwd
    result_log = np.where(
        valid,
        log_fp_safe,
        np.where(both & ~same_idx, interp_log, -np.inf),
    )
    safe = result_log > -700.0
    with np.errstate(over="ignore", under="ignore"):
        return np.where(safe, np.exp(np.where(safe, result_log, 0.0)), 0.0)


def _batched_loglog_interp(log_x_eval, log_xp, fp_K):
    """Batched log-log linear interpolation matching the scalar log-log interp
    PriNCe uses inside ``_collect_interpolated_spectra``.

    Parameters
    ----------
    log_x_eval : ndarray, shape ``(M,)``
        Evaluation points in log-space.
    log_xp : ndarray, shape ``(N,)``
        Data x-grid in log-space, monotone strictly-increasing (shared
        across K).
    fp_K : ndarray, shape ``(K, N)``
        Linear-space function values; may contain zeros/negatives that
        should be treated as masked-out (returned as 0).

    Returns
    -------
    out : ndarray, shape ``(K, M)``
        Linear-space interpolated values, 0 where either endpoint was
        non-positive (matches PriNCe's
        ``mask = curr_spec > 0`` + ``nan_to_num`` behavior).
    """
    K, N = fp_K.shape
    M = log_x_eval.size

    # Pad zeros / negatives with a tiny positive epsilon. log(eps) is very
    # negative (~−700), so log-log interp through that point yields a tiny
    # positive result that ``exp`` returns as ~0 — numerically equivalent
    # to PriNCe's ``mask = fp > 0`` + ``np.interp`` + ``nan_to_num``
    # convention. Avoids per-walker mask handling.
    LOG_EPS = -700.0
    log_fp = np.where(fp_K > 0,
                      np.log(np.where(fp_K > 0, fp_K, 1.0)),
                      LOG_EPS)

    idx = np.searchsorted(log_xp, log_x_eval) - 1
    idx = np.clip(idx, 0, N - 2)                    # (M,)

    x_lo = log_xp[idx]
    x_hi = log_xp[idx + 1]
    w = (log_x_eval - x_lo) / (x_hi - x_lo)         # (M,)

    idx_K = np.broadcast_to(idx, (K, M))
    fp_lo = np.take_along_axis(log_fp, idx_K, axis=-1)
    fp_hi = np.take_along_axis(log_fp, idx_K + 1, axis=-1)

    # Linear interp in log space; padded LOG_EPS gives ~−700 endpoints
    # for masked bins, which exp'd back is essentially zero.
    log_result = fp_lo + w[None, :] * (fp_hi - fp_lo)
    # Guard exp under/overflow: the LOG_EPS padding gives ~-700 endpoints
    # for masked bins, which compounded can fall further below.
    safe = (log_result > -700.0) & (log_result < 700.0)
    with np.errstate(over="ignore", under="ignore"):
        out = np.where(safe, np.exp(np.where(safe, log_result, 0.0)), 0.0)

    in_range = (log_x_eval >= log_xp[0]) & (log_x_eval <= log_xp[-1])
    return np.where(in_range[None, :], out, 0.0)


def _batched_interp1d(x_eval, xp, fp):
    """Vectorized linear 1-D interpolation with scipy.interp1d's
    ``fill_value=(0.0, 0.0)`` convention (OOR returns 0).

    Parameters
    ----------
    x_eval : ndarray, shape ``(..., M)``
        Evaluation points; trailing axis is per-output-bin.
    xp : ndarray, shape ``(N,)``
        Monotone strictly-increasing grid (shared across all leading dims).
    fp : ndarray, shape ``(..., N)``
        Function values on ``xp``; leading axes broadcast with x_eval.

    Returns
    -------
    out : ndarray, shape ``(..., M)``
    """
    idx = np.searchsorted(xp, x_eval) - 1
    idx = np.clip(idx, 0, xp.size - 2)
    x_lo = xp[idx]
    x_hi = xp[idx + 1]
    w = (x_eval - x_lo) / (x_hi - x_lo)
    fp_lo = np.take_along_axis(fp, idx, axis=-1)
    fp_hi = np.take_along_axis(fp, idx + 1, axis=-1)
    out = fp_lo + w * (fp_hi - fp_lo)
    in_range = (x_eval >= xp[0]) & (x_eval <= xp[-1])
    return np.where(in_range, out, 0.0)


def _profile_norm_diag_batch(model_unit_K, data, err_up, err_low, indices, max_iter=3):
    """Batched version of :func:`profile_norm_diag` for K walkers."""
    m_K = np.asarray(model_unit_K[:, indices], dtype=float)
    d = np.asarray(data[indices], dtype=float).ravel()
    eu = np.asarray(err_up[indices], dtype=float).ravel()
    el = np.asarray(err_low[indices], dtype=float).ravel()

    K = m_K.shape[0]
    sign_above = m_K > d[None, :]
    N_prev = -np.ones(K)
    for _ in range(max_iter):
        sigma = np.where(sign_above, eu[None, :], el[None, :])
        sigma_sq = sigma * sigma
        valid = sigma_sq > 0
        # safe accumulators: zero out invalid bins
        denom_terms = np.where(valid, m_K * m_K / np.where(valid, sigma_sq, 1.0), 0.0)
        num_terms = np.where(valid, m_K * d[None, :] / np.where(valid, sigma_sq, 1.0), 0.0)
        denom = denom_terms.sum(axis=1)
        num = num_terms.sum(axis=1)
        N_K = np.where(denom > 0, num / denom, 0.0)
        N_K = np.maximum(N_K, 1e-300)
        new_sign = (N_K[:, None] * m_K) > d[None, :]
        sign_unchanged = np.array_equal(new_sign, sign_above)
        N_close = np.all(np.abs(N_K - N_prev) < 1e-8 * np.maximum(N_K, 1e-30))
        if sign_unchanged and N_close:
            break
        sign_above = new_sign
        N_prev = N_K
    sigma = np.where(sign_above, eu[None, :], el[None, :])
    sigma_sq = sigma * sigma
    valid = sigma_sq > 0
    delta = N_K[:, None] * m_K - d[None, :]
    chi2_K = np.where(valid, (delta ** 2) / np.where(valid, sigma_sq, 1.0), 0.0).sum(axis=1)
    return chi2_K.astype(float), N_K.astype(float)


def profile_norm_corr_batch(model_unit_K, data, stat_up, stat_low, sys_up, sys_low,
                             corr_sub, indices, max_iter=3, cov_sys_alt=None):
    """Batched version of :func:`profile_norm_corr`.

    Performs the sign-iterated closed-form total-norm profile for K walkers
    simultaneously via numpy batched linear algebra:

      cov_K shape (K, n_eff, n_eff)
      np.linalg.solve(cov_K, b_K[..., None]).squeeze(-1)

    The sign mask `sign_above_K` (shape (K, n_eff)) is updated independently
    per walker; the global iteration breaks when all walkers' masks AND
    N_star values are stable.

    Parameters
    ----------
    model_unit_K : ndarray, shape ``(K, n_spec_total)``
        Per-walker model spectrum (in the data energy frame, ΔE already
        applied via the caller's interpolation).
    data, stat_up, stat_low, sys_up, sys_low : ndarray, shape ``(n_spec_total,)``
        Same as :func:`profile_norm_corr`.
    corr_sub : ndarray, shape ``(n_eff, n_eff)``
        Pre-built Emin-sliced correlation matrix.
    indices : 1-D int array or boolean mask
        Spectrum bins included in the chi² (e.g. ``egrid > Emin``).
    cov_sys_alt : ndarray, shape ``(n_eff, n_eff)``, optional
        Pre-built systematic-only covariance (e.g. residual after dropping
        the energy-scale eigenmode). Replaces the ``sys ⊗ sys · corr`` block
        when set.

    Returns
    -------
    chi2_K : ndarray, shape ``(K,)``
    N_K : ndarray, shape ``(K,)``
    """
    m_K = np.asarray(model_unit_K[:, indices], dtype=float)
    d = np.asarray(data[indices], dtype=float).ravel()
    K = m_K.shape[0]
    n_eff = d.size

    su = np.asarray(stat_up[indices], dtype=float).ravel()
    sl_ = np.asarray(stat_low[indices], dtype=float).ravel()
    if cov_sys_alt is None:
        yu = np.asarray(sys_up[indices], dtype=float).ravel()
        yl = np.asarray(sys_low[indices], dtype=float).ravel()
    else:
        cov_sys_alt = np.asarray(cov_sys_alt, dtype=float)

    sign_above = m_K > d[None, :]      # (K, n_eff)
    N_K = np.ones(K, dtype=float)
    N_prev = -np.ones(K, dtype=float)
    diag_idx = np.arange(n_eff)

    chi2_K = np.zeros(K, dtype=float)
    for _ in range(max_iter):
        stat_K = np.where(sign_above, su[None, :], sl_[None, :])     # (K, n_eff)
        if cov_sys_alt is None:
            sys_K = np.where(sign_above, yu[None, :], yl[None, :])   # (K, n_eff)
            # Outer product per walker, multiplied by the shared corr.
            cov_K = sys_K[:, :, None] * sys_K[:, None, :] * corr_sub[None, :, :]
        else:
            # Broadcast a copy so we can mutate the diagonal per-walker.
            cov_K = np.broadcast_to(cov_sys_alt[None, :, :], (K, n_eff, n_eff)).copy()
        cov_K[:, diag_idx, diag_idx] += stat_K * stat_K

        # Batched solve. We need C⁻¹ m and C⁻¹ d per walker.
        # d is shared → broadcast to (K, n_eff) for the solve.
        d_K = np.broadcast_to(d, (K, n_eff))
        # Stack both RHS into one solve: (K, n_eff, 2)
        rhs = np.stack([m_K, d_K], axis=-1)                          # (K, n_eff, 2)
        sol = np.linalg.solve(cov_K, rhs)                            # (K, n_eff, 2)
        Ci_m_K = sol[..., 0]
        Ci_d_K = sol[..., 1]

        mCm_K = np.einsum("ki,ki->k", m_K, Ci_m_K)
        mCd_K = np.einsum("ki,ki->k", m_K, Ci_d_K)
        dCd_K = np.einsum("ki,ki->k", d_K, Ci_d_K)

        # N* per walker, with the positivity floor
        with np.errstate(divide="ignore", invalid="ignore"):
            N_K = np.where(mCm_K > 0, mCd_K / mCm_K, 0.0)
        N_K = np.maximum(N_K, 1e-300)
        chi2_K = N_K * N_K * mCm_K - 2.0 * N_K * mCd_K + dCd_K

        new_sign = (N_K[:, None] * m_K) > d[None, :]
        sign_unchanged = np.array_equal(new_sign, sign_above)
        N_close = np.all(np.abs(N_K - N_prev) < 1e-8 * np.maximum(N_K, 1e-30))
        if sign_unchanged and N_close:
            break
        sign_above = new_sign
        N_prev = N_K
    return chi2_K.astype(float), N_K.astype(float)


def profile_norm_corr(model_unit, data, stat_up, stat_low, sys_up, sys_low,
                      corr_sub, indices, max_iter=3, cov_sys_alt=None):
    """Closed-form profile of total normalization with bin-to-bin correlation.

    Mirrors :meth:`UHECROptimizer._get_chi2_spectrum_corr` but with a
    single linear total-norm degree of freedom analytically optimized:
    at fixed sign mask,
        N* = (m^T C^-1 d) / (m^T C^-1 m)
        chi2 = N*² (m C^-1 m) − 2 N* (m C^-1 d) + d C^-1 d
    Iterate sign with N* recomputed each pass — usually 1–2 passes.

    Parameters
    ----------
    cov_sys_alt : ndarray, optional
        Pre-built systematic-only covariance for the Emin slice. If
        given, replaces the bin-by-bin asymmetric ``sys ⊗ sys · corr``
        block — used to substitute the residual-sys covariance from
        :data:`auger2025` when the caller is *also* floating ``deltaE``
        and the top energy-scale eigenmode has been dropped to avoid
        double-counting.
    """
    from scipy.linalg import cho_factor, cho_solve

    m = np.asarray(model_unit[indices]).ravel()
    d = np.asarray(data[indices]).ravel()
    su = np.asarray(stat_up[indices]).ravel()
    sl_ = np.asarray(stat_low[indices]).ravel()
    if cov_sys_alt is None:
        yu = np.asarray(sys_up[indices]).ravel()
        yl = np.asarray(sys_low[indices]).ravel()

    sign_above = m > d
    N = 1.0
    N_prev = -1.0
    chi2 = np.inf
    for _ in range(max_iter):
        stat = np.where(sign_above, su, sl_)
        if cov_sys_alt is None:
            sys_ = np.where(sign_above, yu, yl)
            cov = (sys_[:, None] * sys_[None, :]) * corr_sub
        else:
            cov = cov_sys_alt.copy()
        cov.flat[:: cov.shape[0] + 1] += stat * stat
        c, lower = cho_factor(cov, lower=True, check_finite=False)
        Ci_m = cho_solve((c, lower), m, check_finite=False)
        Ci_d = cho_solve((c, lower), d, check_finite=False)
        mCm = float(m @ Ci_m)
        mCd = float(m @ Ci_d)
        dCd = float(d @ Ci_d)
        N = mCd / mCm if mCm > 0 else 0.0
        N = max(N, 1e-300)
        chi2 = N * N * mCm - 2.0 * N * mCd + dCd
        new_sign = (N * m) > d
        if np.array_equal(new_sign, sign_above) and abs(N - N_prev) < 1e-8 * max(N, 1e-30):
            break
        sign_above = new_sign
        N_prev = N
    return float(chi2), float(N)


class UltraNestFractionsFit(object):
    """Nested-sampling driver with no inner iMinuit.

    Samples ``(gamma, log10_rmax, m, deltaE)`` + ``n_species − 1`` mass
    fractions on the unit simplex (Dirichlet(1,…,1) via stick-breaking).
    The total normalization is profiled analytically inside ``loglike``
    by closed-form least-squares.

    ``xmax_shift`` is held at 0.0 in this driver (matches the prod-fit's
    ``fix_xmax_shift=True``). Opening it as a sampled dimension would
    add a single flat-prior parameter — see :class:`UltraNestFit` for
    the corresponding inner-fit version.

    Parameters
    ----------
    prince_run, spectrum, xmax, xrms, particle_ids, xmax_model, Emin,
    use_corr, sclass, m_kind, initial_z, final_z, max_step, solver
        Same semantics as :class:`UltraNestFit`.
    xmax_sys, sigma_sys
        Xmax systematics as in :class:`UHECROptimizer`; ``xmax_sys="cov"``
        profiles them in closed form and excludes ``sample_xmax_shift``.
    source_bounds : dict, optional
        Override flat-prior bounds for ``gamma`` / ``log10_rmax`` / ``m``
        / ``deltaE``. Defaults to :data:`FRACTIONS_SOURCE_BOUNDS`.

    Notes
    -----
    ``apply_auger_xmax_shift`` is intentionally not supported here — the
    Auger-2024 ΔXmax shift requires a Gaussian-prior nuisance that's
    cleaner to introduce as another UltraNest dim than to fold into the
    closed-form profile. Filed as a follow-up on the fractions-test
    result page.
    """

    def __init__(
        self,
        prince_run,
        spectrum, xmax, xrms,
        particle_ids=(2212, 1000020040, 1000070140, 1000140280, 1000260560),
        source_bounds=None,
        xmax_model="sibyll",
        Emin=6e9,
        use_corr=False,
        sclass="auger",
        m_kind="simple",
        initial_z=1.0,
        final_z=0.0,
        max_step=1e-2,
        sample_xmax_shift=False,
        use_corr_residual=False,
        solver="etd2",
        xmax_sys="shift",
        sigma_sys="correlated",
    ):
        self.prince_run = prince_run
        self.spectrum = spectrum
        self.xmax = xmax
        self.xrms = xrms
        self.particle_ids = list(particle_ids)
        self.n_species = len(particle_ids)
        if self.n_species < 2:
            raise ValueError(
                "Need ≥2 species for a Dirichlet prior on fractions; got {}".format(
                    self.n_species
                )
            )
        self.xmax_model = xmax_model
        self.Emin = Emin
        self.use_corr = bool(use_corr)
        self.use_corr_residual = bool(use_corr_residual)
        self.sample_xmax_shift = bool(sample_xmax_shift)
        if xmax_sys == "cov" and self.sample_xmax_shift:
            raise ValueError("xmax_sys='cov' profiles the Xmax systematics; set sample_xmax_shift=False")
        self.xmax_sys = xmax_sys
        self.sigma_sys = sigma_sys
        self.sclass = sclass
        self.m_kind = m_kind
        self.initial_z = initial_z
        self.final_z = final_z
        self.max_step = max_step

        self._bounds = dict(FRACTIONS_SOURCE_BOUNDS)
        if self.sample_xmax_shift:
            # ``xmax_shift`` is in units of the per-bin sys_Up/sys_Low arrays
            # the existing UHECROptimizer chi² applies; the [-1, +1] bound
            # is a flat 1σ-window prior. Matches the iMinuit-pipeline
            # convention from fit_data_minuit.
            self._bounds["xmax_shift"] = (-1.0, 1.0)
        if source_bounds is not None:
            self._bounds.update(source_bounds)
        for name in FRACTIONS_PARAM_NAMES:
            if name not in self._bounds:
                raise KeyError(
                    "No prior bound provided for parameter {!r}".format(name)
                )

        # Param names: 4 source [+ xmax_shift] + (n_species-1) free fractions.
        # xmax_shift sits BETWEEN the source nuisances and the fractions so
        # the implicit-last-fraction convention stays unchanged.
        param_names_base = list(FRACTIONS_PARAM_NAMES)
        if self.sample_xmax_shift:
            param_names_base.append("xmax_shift")
        frac_names = tuple(
            "frac_{}".format(_short_id(pid))
            for pid in self.particle_ids[:-1]
        )
        self.param_names = tuple(param_names_base) + frac_names
        # The last species' fraction is implicit (1 − Σ others) and
        # exposed via :meth:`fractions_from_theta` for plotting.
        self.implicit_fraction_name = "frac_{}".format(_short_id(self.particle_ids[-1]))
        # Number of "non-fraction" leading dims for prior_transform / loglike.
        self._n_source_dims = len(param_names_base)

        self.walker = MultiRHSWalker(
            prince_run, spectrum, xmax, xrms, progressbar=False, solver=solver
        )
        self.sampler = None
        self.result = None
        self._n_loglike_calls = 0
        self._n_live_evals = 0
        self._t_forward = 0.0
        self._t_chi2 = 0.0

    @property
    def n_params(self):
        return len(self.param_names)

    # ---------- Prior transform ----------
    def prior_transform(self, u):
        """Map unit cube → physical theta. Vectorized in shape ``(M, D)`` or 1-D."""
        u = np.asarray(u, dtype=float)
        out = np.empty_like(u)
        # Source / nuisance dims (first n_source): flat priors
        source_names = list(FRACTIONS_PARAM_NAMES)
        if self.sample_xmax_shift:
            source_names.append("xmax_shift")
        for i, name in enumerate(source_names):
            lo, hi = self._bounds[name]
            if u.ndim == 1:
                out[i] = lo + (hi - lo) * u[i]
            else:
                out[:, i] = lo + (hi - lo) * u[:, i]
        # Stick-breaking Dirichlet on the remaining dims
        ns = self._n_source_dims
        n = self.n_species
        u_frac = u[..., ns : ns + n - 1]
        fr_all = stick_break_dirichlet(u_frac, n)
        if u.ndim == 1:
            out[ns : ns + n - 1] = fr_all[: n - 1]
        else:
            out[:, ns : ns + n - 1] = fr_all[:, : n - 1]
        return out

    # ---------- Helpers ----------
    def _theta_to_source_params(self, theta_row):
        """Returns the dict consumed by :meth:`MultiRHSWalker.compute_models_batch_combined`.

        Includes the full ``fractions`` vector (length ``n_species``).
        """
        gamma = float(theta_row[0])
        rmax = 10.0 ** float(theta_row[1])
        m_val = float(theta_row[2])
        m_arg = (self.m_kind, m_val) if self.m_kind == "simple" else m_val
        fractions = self.fractions_from_theta(theta_row)
        return {
            "gamma": gamma, "rmax": rmax, "m": m_arg,
            "fractions": fractions,
        }

    def fractions_from_theta(self, theta_row):
        """Return the full (n_species,) fraction vector for one theta row.

        The last component is the implicit ``1 − Σ(others)``; we clamp
        infinitesimal numerical negatives to a tiny positive floor.
        """
        ns = self._n_source_dims
        fr = np.empty(self.n_species, dtype=float)
        fr[: self.n_species - 1] = theta_row[ns : ns + self.n_species - 1]
        fr[-1] = 1.0 - fr[: self.n_species - 1].sum()
        return np.clip(fr, 1e-300, None)

    def xmax_shift_from_theta(self, theta_row):
        """Return the sampled xmax_shift if active; else 0.0."""
        if self.sample_xmax_shift:
            return float(theta_row[4])
        return 0.0

    def _build_optimizer(self, combined_result):
        """Wrap a single combined CR result for :class:`UHECROptimizer`.

        The optimizer was built to take a list of per-species results and
        mix them with weights ``norms``. We feed it a length-1 list
        containing the already-combined result; calling
        ``compute_combined_result(norms=[1.0], deltaE)`` then reduces to
        an identity mix (the species-mixing was done during propagation).
        ``ncoids=[0]`` is a harmless placeholder — we never call the
        per-species iMinuit naming code path.

        ``use_corr_residual`` propagates through so the optimizer's
        ``_cov_residual_sub`` is built once at construction and reused on
        every chi² call.
        """
        return UHECROptimizer(
            [combined_result], self.spectrum, self.xmax, self.xrms,
            Emin=self.Emin, ncoids=[0],
            xmax_model=self.xmax_model,
            use_corr=self.use_corr,
            use_corr_residual=self.use_corr_residual,
            apply_auger_xmax_shift=False,
            xmax_sys=self.xmax_sys,
            sigma_sys=self.sigma_sys,
        )

    def _ensure_batch_state(self, sample_result):
        """Lazy-build the cached configuration the batched chi² needs.

        Constructs a single throw-away ``UHECROptimizer`` once on the
        first batch call to harvest its precomputed pieces:

          - the XmaxModel object (e.g. ``XmaxSimple(model=Sibyll23)``)
          - the model energy grid (``egrid_spectrum``, ``egrid_xmax``)
          - the spectrum data grid (``self.spectrum["energy"]``)
          - the correlation matrix slice ``_corr_sub`` and its bin indices
          - the residual covariance ``_cov_residual_sub`` if active

        These are PriNCe-pickle properties (egrid) and data-fixture
        properties (corr) — shared across all walkers, so we build them
        exactly once.
        """
        if getattr(self, "_batch_state", None) is not None:
            return
        opt = self._build_optimizer(sample_result)
        # Capture only what the batched code needs.
        spec_arrs = (
            (opt.spectrum.get("stat_up"), opt.spectrum.get("stat_low"))
            if "stat_up" in opt.spectrum and "stat_low" in opt.spectrum
            else (opt.spectrum["upper_err"], opt.spectrum["lower_err"])
        )
        sys_arrs = (
            (opt.spectrum.get("sys_Up"), opt.spectrum.get("sys_Low"))
            if "sys_Up" in opt.spectrum and "sys_Low" in opt.spectrum
            else (np.zeros_like(spec_arrs[0]), np.zeros_like(spec_arrs[0]))
        )
        # CR species layout (shared across all results from this prince_run):
        # for each pid in "CR" (= all nucleus species in the kernel) we need
        # (lidx, uidx) to slice the state array and A to scale the energy
        # axis. The common output egrid is also fixed by `prince_run`.
        from prince_ultra.util import is_nucleus
        sample_result = sample_result
        cr_pids = [s for s in sample_result.known_species if is_nucleus(s)]
        species_info = []
        for pid in cr_pids:
            spec = sample_result.spec_man.pdgid2sref[pid]
            lidx, uidx = int(spec.lidx()), int(spec.uidx())
            A = float(spec.A)
            native_egrid = A * sample_result.egrid          # (n_native,)
            log_native_egrid = np.log(native_egrid)
            species_info.append(dict(
                pid=int(pid), lidx=lidx, uidx=uidx, A=A,
                log_native_egrid=log_native_egrid,
                lnA=float(np.log(A)),
            ))
        # Common egrid: identical to what _check_id_grid builds. Cache by
        # running _check_id_grid once on the sample result.
        _, com_egrid = sample_result._check_id_grid("CR", None)
        log_com_egrid = np.log(com_egrid)

        self._batch_state = {
            "XmaxModel": opt.XmaxModel,
            "egrid_spec_data": opt.egrid_spectrum,
            "egrid_xmax_data": opt.egrid_xmax,
            "spec_data": opt.spectrum["spectrum"],
            "stat_up": spec_arrs[0],
            "stat_low": spec_arrs[1],
            "sys_up": sys_arrs[0],
            "sys_low": sys_arrs[1],
            "Xmax_val": opt.Xmax["val"],
            "Xmax_stat": opt.Xmax["stat"],
            "Xmax_sys_Up": opt.Xmax["sys_Up"],
            "Xmax_sys_Low": opt.Xmax["sys_Low"],
            "XRMS_val": opt.XRMS["val"],
            "XRMS_stat": opt.XRMS["stat"],
            "XRMS_sys_Up": opt.XRMS["sys_Up"],
            "XRMS_sys_Low": opt.XRMS["sys_Low"],
            "use_corr": opt.use_corr,
            "corr_sub": getattr(opt, "_corr_sub", None),
            "corr_indices": getattr(opt, "_corr_indices", None),
            "cov_residual_sub": getattr(opt, "_cov_residual_sub", None),
            "use_corr_residual": opt.use_corr_residual,
            "xsys": opt._xsys,
            # Model egrid (extracted from the result via the optimizer's interps)
            "egrid_spec_model": opt.intp_spectrum.x.copy(),
            "egrid_xmax_model": opt.intp_mean_lnA.x.copy(),
            # Per-species layout for the batched CR extraction
            "cr_species_info": species_info,
            "com_egrid": com_egrid,
            "log_com_egrid": log_com_egrid,
            "lnA_arr": np.array([s["lnA"] for s in species_info]),  # (n_species,)
        }

    def _chi2_batch(self, results, theta_batch):
        """Vectorized chi² profile over K walkers.

        Bypasses the per-walker ``UHECROptimizer`` loop in
        :meth:`_chi2_with_profile`. Extracts per-walker model arrays
        once, then performs all interpolation, ``XmaxSimple`` evaluation,
        χ² accumulation, and the closed-form total-norm profile as
        batched numpy ops. Returns ``(chi2_K, N_star_K)`` matching the
        scalar code's outputs to ≈ fp64 round-off.

        Parameters
        ----------
        results : sequence of ``UHECRPropagationResult``
            One combined CR result per walker (length K).
        theta_batch : ndarray, shape ``(K, n_params)``
            Per-walker physical theta. ``theta_batch[:, 3]`` is ΔE;
            ``theta_batch[:, 4]`` is xmax_shift if ``sample_xmax_shift``,
            else implicit 0.
        """
        self._ensure_batch_state(results[0])
        bs = self._batch_state
        K = len(results)

        # ----- Batched extraction: per-species J(E) on the common egrid -----
        # Replaces the K × n_species scalar log-log interps inside
        # PriNCe's get_solution_group / get_lnA. Stack states once, then
        # loop over species (small loop, ~50-100 iters) doing batched
        # log-log interp across K results per species.
        state_K = np.stack([np.asarray(r.state) for r in results], axis=0)  # (K, dim_states)
        species_info = bs["cr_species_info"]
        com_egrid = bs["com_egrid"]
        log_com_egrid = bs["log_com_egrid"]
        n_species_cr = len(species_info)
        n_com = com_egrid.size
        # J_K[k, i, e] = J of species i at common energy bin e for walker k.
        # (epow=0 — pure J, no E^k weighting; we apply E^3 explicitly below
        # for the spectrum chi².)
        J_K = np.empty((K, n_species_cr, n_com), dtype=float)
        for i, info in enumerate(species_info):
            # Per-species J on its native grid: state[lidx:uidx] / A
            j_native_K = state_K[:, info["lidx"]:info["uidx"]] / info["A"]  # (K, n_native_i)
            # Pre-fill numerical negatives / zeros with log-interp from
            # nearest positive neighbors — restores PriNCe's mask + interp
            # behavior. Crucial for sech-cutoff source class where the
            # ETD2 solver produces O(10³) negative-state bins per walker.
            j_native_K_filled = _fill_log_interior(j_native_K, info["log_native_egrid"])
            J_K[:, i, :] = _batched_loglog_interp(
                log_com_egrid, info["log_native_egrid"], j_native_K_filled,
            )

        # get_solution_group("CR", epow=3) ≡ com_egrid^3 · sum_i J_i
        # The output is "spectrum on com_egrid". The optimizer's
        # intp_spectrum is then built on com_egrid (= bs["egrid_spec_model"]).
        # We just take the summed-species J·E^3.
        com_E3 = com_egrid ** 3
        arr_spec_K = (J_K.sum(axis=1)) * com_E3[None, :]                  # (K, n_com)

        # get_lnA("CR") ≡ epow=0 spectra → weighted average over species
        lnA = bs["lnA_arr"]                                               # (n_species,)
        total_K = J_K.sum(axis=1)                                         # (K, n_com)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_lnA_K = (lnA[None, :, None] * J_K).sum(axis=1) / total_K
            var_lnA_K = (lnA[None, :, None] ** 2 * J_K).sum(axis=1) / total_K - mean_lnA_K ** 2
        # Replace NaN/Inf with 0 (matches PriNCe's np.nan_to_num convention
        # inside _create_interpolators for the per-species mean_lnA/var_lnA).
        arr_mlnA_K = np.nan_to_num(mean_lnA_K)
        arr_vlnA_K = np.nan_to_num(var_lnA_K)

        # ----- Per-walker ΔE and xmax_shift -----
        deltaE_K = np.asarray(theta_batch[:, 3], dtype=float)
        if self.sample_xmax_shift:
            xshift_K = np.asarray(theta_batch[:, 4], dtype=float)
        else:
            xshift_K = np.zeros(K, dtype=float)

        # ----- Vectorized interp to ΔE-shifted DATA grid -----
        # The intp_spectrum / intp_mean_lnA in the optimizer use linear
        # interp with fill_value=0 outside; we replicate that.
        spec_at_data_K = _batched_interp1d(
            bs["egrid_spec_data"][None, :] * (1.0 - deltaE_K[:, None]),
            bs["egrid_spec_model"], arr_spec_K,
        )
        mlnA_at_xmax_K = _batched_interp1d(
            bs["egrid_xmax_data"][None, :] * (1.0 - deltaE_K[:, None]),
            bs["egrid_xmax_model"], arr_mlnA_K,
        )
        vlnA_at_xmax_K = _batched_interp1d(
            bs["egrid_xmax_data"][None, :] * (1.0 - deltaE_K[:, None]),
            bs["egrid_xmax_model"], arr_vlnA_K,
        )

        # ----- Vectorized XmaxModel application -----
        # XmaxSimple.get_mean_Xmax / get_var_Xmax are polynomial in
        # (lnA, log10(E/E0)) — vectorize naturally over (K, n_xmax_data).
        egrid_xmax_data_K = np.broadcast_to(
            bs["egrid_xmax_data"], (K, bs["egrid_xmax_data"].size)
        )
        res_xmax_K = bs["XmaxModel"].get_mean_Xmax(mlnA_at_xmax_K, egrid_xmax_data_K)
        var_xmax_K, _ = bs["XmaxModel"].get_var_Xmax(
            mlnA_at_xmax_K, vlnA_at_xmax_K, egrid_xmax_data_K
        )
        res_sigma_K = np.sqrt(np.clip(var_xmax_K, 0.0, None))

        # ----- chi²_Xmax (E > Emin slice, sign-asymmetric xmax_shift) -----
        e_xmax = bs["egrid_xmax_data"]
        sl_x = np.flatnonzero(e_xmax > self.Emin)
        Xv = bs["Xmax_val"][sl_x]
        Xs = bs["Xmax_stat"][sl_x]
        XsU = bs["Xmax_sys_Up"][sl_x]
        XsL = bs["Xmax_sys_Low"][sl_x]
        delta_x_K = res_xmax_K[:, sl_x] - Xv[None, :]
        xshift_corr_x = np.where(
            xshift_K[:, None] > 0.0,
            xshift_K[:, None] * XsU[None, :],
            xshift_K[:, None] * XsL[None, :],
        )
        if bs["xsys"] is not None:
            chi2_xmax_K = bs["xsys"].chi2_mean(delta_x_K)
        else:
            delta_x_K = delta_x_K + xshift_corr_x
            chi2_xmax_K = ((delta_x_K / Xs[None, :]) ** 2).sum(axis=1)

        # ----- chi²_VarXmax (same selection, same nuisance) -----
        Rv = bs["XRMS_val"][sl_x]
        Rs = bs["XRMS_stat"][sl_x]
        RsU = bs["XRMS_sys_Up"][sl_x]
        RsL = bs["XRMS_sys_Low"][sl_x]
        delta_r_K = res_sigma_K[:, sl_x] - Rv[None, :]
        xshift_corr_r = np.where(
            xshift_K[:, None] > 0.0,
            xshift_K[:, None] * RsU[None, :],
            xshift_K[:, None] * RsL[None, :],
        )
        if bs["xsys"] is not None:
            chi2_var_K = bs["xsys"].chi2_sigma(delta_r_K)
        else:
            delta_r_K = delta_r_K + xshift_corr_r
            chi2_var_K = ((delta_r_K / Rs[None, :]) ** 2).sum(axis=1)

        # ----- chi²_spectrum with closed-form total-norm profile -----
        if bs["use_corr"]:
            cov_sys_alt = (
                bs["cov_residual_sub"] if bs["use_corr_residual"] else None
            )
            chi2_spec_K, N_star_K = profile_norm_corr_batch(
                spec_at_data_K, bs["spec_data"],
                bs["stat_up"], bs["stat_low"],
                bs["sys_up"], bs["sys_low"],
                bs["corr_sub"], bs["corr_indices"],
                cov_sys_alt=cov_sys_alt,
            )
        else:
            # Diagonal asymm fallback: vectorize the scalar profile_norm_diag.
            sl_spec = np.flatnonzero(bs["egrid_spec_data"] > self.Emin)
            chi2_spec_K, N_star_K = _profile_norm_diag_batch(
                spec_at_data_K, bs["spec_data"],
                bs["stat_up"], bs["stat_low"], sl_spec,
            )

        return chi2_spec_K + chi2_xmax_K + chi2_var_K, N_star_K

    def _chi2_with_profile(self, opt, deltaE, xmax_shift=0.0):
        """Total profile χ² with closed-form total-norm LSQ.

        Returns ``(chi2_total, N_star)``. Decomposition:
            chi2_total = chi2_spec(N*) + chi2_Xmax + chi2_VarXmax
        chi2_Xmax / chi2_VarXmax do not depend on the total norm. When
        ``self.sample_xmax_shift`` is True, ``xmax_shift`` is threaded
        through ``get_chi2_Xmax`` / ``get_chi2_VarXmax`` to add the
        FD-calibration nuisance.

        Mass fractions were baked into the source class at propagation
        time (one combined RHS per livepoint), so ``opt`` already has the
        species-mixed result and we call ``compute_combined_result`` with
        ``norms=[1.0]`` (no-op identity mix).
        """
        opt.compute_combined_result(np.array([1.0]), deltaE)
        chi2_xmax = float(opt.get_chi2_Xmax(xmax_shift=xmax_shift, auger_dxmax=None))
        chi2_var = float(opt.get_chi2_VarXmax(xmax_shift=xmax_shift))

        model_unit = opt.res_spectrum
        spec = opt.spectrum
        if opt.use_corr:
            indices = opt._corr_indices
            if "stat_up" in spec and "stat_low" in spec:
                stat_up_arr = spec["stat_up"]
                stat_low_arr = spec["stat_low"]
            else:
                # Same fallback as _get_chi2_spectrum_corr: legacy combined
                # errors as stat, zero sys.
                stat_up_arr = spec["upper_err"]
                stat_low_arr = spec["lower_err"]
            # If the optimizer was constructed with use_corr_residual, swap
            # the asymmetric (sys ⊗ corr) block for the pre-built residual
            # covariance — keeps stat asymmetry handled in the iterative
            # sign-stabilization loop.
            cov_sys_alt = getattr(opt, "_cov_residual_sub", None) if opt.use_corr_residual else None
            if cov_sys_alt is None and "sys_Up" in spec and "sys_Low" in spec:
                sys_up_arr = spec["sys_Up"]
                sys_low_arr = spec["sys_Low"]
            else:
                sys_up_arr = np.zeros_like(stat_up_arr)
                sys_low_arr = np.zeros_like(stat_low_arr)
            chi2_spec, N_star = profile_norm_corr(
                model_unit, spec["spectrum"],
                stat_up_arr, stat_low_arr, sys_up_arr, sys_low_arr,
                opt._corr_sub, indices, cov_sys_alt=cov_sys_alt,
            )
        else:
            indices = np.flatnonzero(opt.egrid_spectrum > self.Emin)
            chi2_spec, N_star = profile_norm_diag(
                model_unit, spec["spectrum"],
                spec["upper_err"], spec["lower_err"], indices,
            )

        return chi2_spec + chi2_xmax + chi2_var, N_star

    # ---------- Loglike ----------
    def loglike(self, theta_batch):
        """Vectorized log-likelihood.

        Uses :meth:`MultiRHSWalker.compute_models_batch_combined` —
        ``K_RHS = K_live`` (vs ``K_live × N_species`` in the per-species
        path) — because the mass fractions are baked into the source
        class at injection time and the propagation is linear in those
        fractions. Saves an ``N_species`` factor on the forward solve.
        """
        import time as _t

        theta_batch = np.atleast_2d(np.asarray(theta_batch, dtype=float))
        K_live = theta_batch.shape[0]
        batch_params = [
            self._theta_to_source_params(theta_batch[j]) for j in range(K_live)
        ]

        t0 = _t.time()
        combined_results = self.walker.compute_models_batch_combined(
            self.particle_ids, batch_params,
            sclass=self.sclass,
            initial_z=self.initial_z, final_z=self.final_z,
            max_step=self.max_step,
        )
        t1 = _t.time()
        self._t_forward += (t1 - t0)

        # Vectorized chi² across all K walkers in one numpy block. Replaces
        # the per-walker Python loop (~10 ms/walker) with batched
        # interp + XmaxModel + chi² + np.linalg.solve on (K, n, n) covariance.
        # Falls back to the scalar per-walker path on any exception (e.g.
        # cholesky failure on a degenerate cell).
        logl = np.empty(K_live, dtype=float)
        try:
            chi2_K, _N_K = self._chi2_batch(combined_results, theta_batch)
            finite = np.isfinite(chi2_K)
            logl[:] = np.where(finite, -0.5 * chi2_K, -1e30)
        except Exception:
            for j, result in enumerate(combined_results):
                try:
                    opt = self._build_optimizer(result)
                    deltaE = float(theta_batch[j, 3])
                    xshift = self.xmax_shift_from_theta(theta_batch[j])
                    chi2, _N = self._chi2_with_profile(opt, deltaE, xmax_shift=xshift)
                except Exception:
                    chi2 = np.inf
                if not np.isfinite(chi2):
                    logl[j] = -1e30
                else:
                    logl[j] = -0.5 * chi2
        t2 = _t.time()
        self._t_chi2 += (t2 - t1)

        self._n_loglike_calls += 1
        self._n_live_evals += K_live
        return logl

    # ---------- Driver ----------
    def run(
        self,
        log_dir=None,
        resume="resume",
        min_num_live_points=400,
        dlogz=0.5,
        max_iters=None,
        max_ncalls=None,
        vectorized=True,
        show_status=True,
        sampler_kwargs=None,
        **run_kwargs,
    ):
        """Run UltraNest. Defaults match the production convergence target."""
        import ultranest

        sampler_kwargs = dict(sampler_kwargs or {})
        sampler = ultranest.ReactiveNestedSampler(
            list(self.param_names),
            self.loglike,
            self.prior_transform,
            log_dir=log_dir,
            resume=resume,
            vectorized=vectorized,
            **sampler_kwargs,
        )
        kw = dict(
            min_num_live_points=min_num_live_points,
            dlogz=dlogz,
            show_status=show_status,
        )
        if max_iters is not None:
            kw["max_iters"] = max_iters
        if max_ncalls is not None:
            kw["max_ncalls"] = max_ncalls
        kw.update(run_kwargs)
        result = sampler.run(**kw)
        self.sampler = sampler
        self.result = result
        return result

    # ---------- Post-processing ----------
    def best_fit_theta(self):
        """Maximum-likelihood theta from the sampled chain."""
        if self.result is None:
            raise RuntimeError("Call .run() first.")
        if "maximum_likelihood" in self.result:
            return np.asarray(self.result["maximum_likelihood"]["point"], dtype=float)
        samples = np.asarray(self.result["weighted_samples"]["points"])
        logl = np.asarray(self.result["weighted_samples"]["logl"])
        return samples[int(np.argmax(logl))]

    # ---------- Plotting hooks ----------
    def recompute_fit(self, theta):
        """Single-theta forward solve + analytic norm profile.

        Matches the calling contract of :meth:`UltraNestFit.recompute_fit`
        from the iMinuit pipeline so the existing plot helpers can dispatch
        on fit class. Returns ``(None, opt)``:

          ``None`` in the iMinuit-result slot — signals to plotting code
          that this is the fractions pipeline (no per-species norms).

          ``opt`` is a populated :class:`UHECROptimizer` over a
          length-1 list containing the already species-mixed combined
          result. Inferred nuisances are attached as attributes for the
          plotting helpers:

            ``opt._inferred_deltaE``     — sampled ΔE
            ``opt._inferred_N_star``     — closed-form total norm
            ``opt._inferred_fractions``  — length-``n_species`` vector
            ``opt._inferred_chi2_total`` — χ² at the profile minimum
        """
        theta = np.asarray(theta, dtype=float)
        batch_params = [self._theta_to_source_params(theta)]
        combined_results = self.walker.compute_models_batch_combined(
            self.particle_ids, batch_params,
            sclass=self.sclass,
            initial_z=self.initial_z, final_z=self.final_z,
            max_step=self.max_step,
        )
        combined = combined_results[0]
        opt = self._build_optimizer(combined)
        deltaE = float(theta[3])
        xshift = self.xmax_shift_from_theta(theta)
        chi2_total, N_star = self._chi2_with_profile(opt, deltaE, xmax_shift=xshift)
        opt._inferred_deltaE = deltaE
        opt._inferred_xmax_shift = xshift
        opt._inferred_N_star = N_star
        opt._inferred_fractions = self.fractions_from_theta(theta)
        opt._inferred_chi2_total = chi2_total
        return None, opt

    def recompute_fit_batch(self, theta_array):
        """Batched version of :meth:`recompute_fit` — N_theta in one solve.

        Packs all forward solves into a single
        ``MultiRHSPropagationSolverETD2(K=N_theta)`` call; then loops the
        analytic norm profile per theta. Yields ``(None, opt)`` per
        theta, in the same order.

        ``theta_array`` must have shape ``(N_theta, n_params)``.
        """
        theta_array = np.asarray(theta_array, dtype=float)
        if theta_array.ndim != 2 or theta_array.shape[1] != self.n_params:
            raise ValueError(
                "theta_array must be (N_theta, {}); got {}".format(
                    self.n_params, theta_array.shape
                )
            )
        batch_params = [
            self._theta_to_source_params(theta_array[j])
            for j in range(theta_array.shape[0])
        ]
        combined_results = self.walker.compute_models_batch_combined(
            self.particle_ids, batch_params,
            sclass=self.sclass,
            initial_z=self.initial_z, final_z=self.final_z,
            max_step=self.max_step,
        )
        for j, combined in enumerate(combined_results):
            opt = self._build_optimizer(combined)
            deltaE = float(theta_array[j, 3])
            xshift = self.xmax_shift_from_theta(theta_array[j])
            chi2_total, N_star = self._chi2_with_profile(opt, deltaE, xmax_shift=xshift)
            opt._inferred_deltaE = deltaE
            opt._inferred_xmax_shift = xshift
            opt._inferred_N_star = N_star
            opt._inferred_fractions = self.fractions_from_theta(theta_array[j])
            opt._inferred_chi2_total = chi2_total
            yield None, opt

    def posterior_samples(self, equal_weight=True):
        """Return posterior samples — equal-weight by default."""
        if self.result is None:
            raise RuntimeError("Call .run() first.")
        if equal_weight and "samples" in self.result:
            return np.asarray(self.result["samples"])
        ws = self.result["weighted_samples"]
        return np.asarray(ws["points"]), np.asarray(ws["weights"])

    def summary(self):
        if self.result is None:
            return "<UltraNestFractionsFit: no result yet>"
        r = self.result
        out = []
        out.append("logZ = {:.3f} ± {:.3f}".format(r["logz"], r["logzerr"]))
        if "samples" in r:
            samples = np.asarray(r["samples"])
            for i, name in enumerate(self.param_names):
                q = np.quantile(samples[:, i], [0.16, 0.5, 0.84])
                out.append(
                    "  {} = {:+.4f}  (68%: {:+.4f}, {:+.4f})".format(
                        name, q[1], q[0], q[2]
                    )
                )
        out.append(
            "n_loglike_calls = {}, total_live_evals = {}".format(
                self._n_loglike_calls, self._n_live_evals
            )
        )
        return "\n".join(out)


_ELEMENT_BY_Z = {
    1: "H", 2: "He", 6: "C", 7: "N", 8: "O", 10: "Ne", 12: "Mg",
    14: "Si", 16: "S", 18: "Ar", 20: "Ca", 22: "Ti", 24: "Cr",
    26: "Fe", 28: "Ni",
}


def _short_id(pid):
    """Compact label for a PDG id used in fraction parameter names.

    ``2212 → p`` (proton), ``2112 → n`` (neutron); PDG ion code
    ``10LZZZAAAI`` → element symbol if Z is known else mass number.
    """
    p = int(pid)
    if p == 2212:
        return "p"
    if p == 2112:
        return "n"
    if p > 1_000_000_000:
        # PDG ion code layout: 10 L ZZZ AAA I (10 digits).
        Z = (p // 10_000) % 1000
        A = (p // 10) % 1000
        return _ELEMENT_BY_Z.get(Z, str(A))
    return str(p)
