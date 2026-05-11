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
    initial_z, final_z, max_step
        Forwarded to the multi-RHS solver.
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
        sclass="auger",
        m_kind="simple",
        initial_z=1.0,
        final_z=0.0,
        max_step=1e-2,
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
        self.sclass = sclass
        self.m_kind = m_kind
        self.initial_z = initial_z
        self.final_z = final_z
        self.max_step = max_step

        self.walker = MultiRHSWalker(prince_run, spectrum, xmax, xrms, progressbar=False)
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
