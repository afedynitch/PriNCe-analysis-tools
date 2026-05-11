import numpy as np
from .xmax import XmaxSimple


class UHECROptimizer(object):
    def __init__(
        self,
        single_results,
        spectrum,
        Xmax,
        XRMS,
        ncoids=None,
        Emin=6e9,
        norms=None,
        xmax_model=None,
        use_corr=False,
        corr_jitter=1e-3,
    ):
        """Per-cell χ² + iMinuit inner fit for UHECR source-parameter scans.

        Parameters
        ----------
        single_results : list of UHECRPropagationResult
            One PriNCe result per injection species.
        spectrum, Xmax, XRMS : dict
            Auger data fixtures (e.g. ``auger2017`` / ``auger2025``).
        ncoids : sequence of int, optional
            Injection PDG ids; defaults to a positional range.
        Emin : float
            Energy threshold (GeV) below which bins are excluded from χ².
        norms : ndarray, optional
            Initial per-species norms.
        xmax_model : {"epos", "sibyll", "qgsjet"} or None
            Hadronic-interaction model for the Xmax conversion.
        use_corr : bool
            **Opt-in correlation-aware spectrum χ².** When True and
            ``spectrum["corr"]`` is present, the spectrum χ² becomes
            ``Δᵀ C⁻¹ Δ`` with ``C = diag(stat²) + outer(sys, sys) ⊙ corr``.
            The asymmetric error sides are chosen bin-by-bin based on
            whether the model is above or below the data. Default False
            preserves the legacy diagonal χ² used by the grid scan.
        corr_jitter : float
            Small relative jitter added to the diagonal of the
            correlation matrix to keep it positive-definite under
            OCR-rounded inputs. 1e-3 is well below any meaningful
            statistical weight; only matters at chol/inv time.
        """
        self.Emin = Emin

        if xmax_model is None or xmax_model == "epos":
            self.XmaxModel = XmaxSimple(model=XmaxSimple.EPOSLHC)
        elif xmax_model == "qgsjet":
            self.XmaxModel = XmaxSimple(model=XmaxSimple.QGSJetII04)
        elif xmax_model == "sibyll":
            self.XmaxModel = XmaxSimple(model=XmaxSimple.Sibyll23)

        # print self.XmaxModel, self.XmaxModel.model
        # spectral data
        self.spectrum = spectrum
        self.egrid_spectrum = spectrum["energy"]

        # Xmax data
        self.Xmax = Xmax
        self.egrid_xmax = Xmax["energy"]
        self.XRMS = XRMS

        self.lst_res = np.array(single_results)
        if norms is None:
            norms = np.ones_like(self.lst_res, dtype=float)
        if ncoids is None:
            ncoids = list(range(len(self.lst_res)))
        self.ncoids = ncoids

        # Correlation-aware χ² setup. We pre-build the boolean mask + a
        # statics covariance scaffold so the hot path in fit_data_minuit
        # only does a small linear solve per evaluation.
        self.use_corr = bool(use_corr) and "corr" in spectrum
        self._corr_solver = None
        if self.use_corr:
            sl = np.flatnonzero(self.egrid_spectrum > self.Emin)
            corr = np.asarray(spectrum["corr"], dtype=float)
            if corr.shape[0] != self.egrid_spectrum.size:
                raise ValueError(
                    "spectrum['corr'] shape {} does not match energy bins {}".format(
                        corr.shape, self.egrid_spectrum.size
                    )
                )
            self._corr_sub = corr[np.ix_(sl, sl)].copy()
            # OCR-rounded matrices can sit infinitesimally outside the
            # positive-definite cone. Jitter the diagonal so cholesky
            # is robust.
            self._corr_sub.flat[:: self._corr_sub.shape[0] + 1] += corr_jitter
            self._corr_indices = sl
            self._corr_jitter = corr_jitter

        self._create_interpolators()
        self.compute_combined_result(norms)

    def _create_interpolators(self):
        from scipy.interpolate import interp1d

        egrid_spectrum, _ = self.lst_res[0].get_solution_group("CR")
        egrid_xmax, _, _ = self.lst_res[0].get_lnA("CR")
        arr_spectrum = np.zeros((len(self.lst_res), egrid_spectrum.size))
        arr_mean_lnA = np.zeros((len(self.lst_res), egrid_xmax.size))
        arr_var_lnA = np.zeros((len(self.lst_res), egrid_xmax.size))
        for idx, res in enumerate(self.lst_res):
            _, spectrum = res.get_solution_group("CR")
            _, mean_lnA, var_lnA = res.get_lnA("CR")
            arr_spectrum[idx] = spectrum
            # in cases where the spectrum is 0, mean and average will be NaN
            # we replace this by zero, so it does not affect the fit
            # arr_mean_lnA[idx] = mean_lnA
            # arr_var_lnA[idx] = var_lnA
            arr_mean_lnA[idx] = np.nan_to_num(mean_lnA)
            arr_var_lnA[idx] = np.nan_to_num(var_lnA)

        self.intp_spectrum = interp1d(
            egrid_spectrum,
            arr_spectrum,
            axis=1,
            fill_value=(0.0, 0.0),
            bounds_error=False,
        )
        self.intp_mean_lnA = interp1d(
            egrid_xmax, arr_mean_lnA, axis=1, fill_value=(0.0, 0.0), bounds_error=False
        )
        self.intp_var_lnA = interp1d(
            egrid_xmax, arr_var_lnA, axis=1, fill_value=(0.0, 0.0), bounds_error=False
        )

    def _intpolate_scipy(self, deltaE):
        self.arr_spectrum = self.intp_spectrum(self.egrid_spectrum * (1 - deltaE))
        self.arr_spec_lnA = self.intp_spectrum(self.egrid_xmax * (1 - deltaE))
        self.arr_mean_lnA = self.intp_mean_lnA(self.egrid_xmax * (1 - deltaE))
        self.arr_var_lnA = self.intp_var_lnA(self.egrid_xmax * (1 - deltaE))

    def compute_combined_result(self, norms, deltaE=0.0):
        self._intpolate_scipy(deltaE)
        # get the averages from subsets by weighting with the norms
        spectrum = (norms[:, np.newaxis] * self.arr_spectrum).sum(axis=0)
        mean_lnA = (norms[:, np.newaxis] * self.arr_spec_lnA * self.arr_mean_lnA).sum(
            axis=0
        ) / (norms[:, np.newaxis] * self.arr_spec_lnA).sum(axis=0)
        var_lnA = (
            norms[:, np.newaxis]
            * self.arr_spec_lnA
            * (self.arr_var_lnA + self.arr_mean_lnA**2)
        ).sum(axis=0) / (norms[:, np.newaxis] * self.arr_spec_lnA).sum(
            axis=0
        ) - mean_lnA**2

        self.res_spectrum = spectrum
        self.res_xmax = self.XmaxModel.get_mean_Xmax(mean_lnA, self.egrid_xmax)
        self.res_sigma_xmax, _ = np.sqrt(
            self.XmaxModel.get_var_Xmax(mean_lnA, var_lnA, self.egrid_xmax)
        )

    def get_chi2_spectrum(
        self,
        norms=None,
        deltaE=0.0,
    ):
        if self.lst_res is not None and norms is not None:
            self.compute_combined_result(norms, deltaE)

        if self.use_corr:
            return self._get_chi2_spectrum_corr()

        # chi2 for values above Emin
        sl = np.argwhere(self.egrid_spectrum > self.Emin)
        delta = self.res_spectrum[sl] - self.spectrum["spectrum"][sl]
        error = np.where(
            self.res_spectrum[sl] > self.spectrum["spectrum"][sl],
            self.spectrum["upper_err"][sl],
            self.spectrum["lower_err"][sl],
        )

        # #penalty chi2 for values below Emin
        # slp = np.argwhere(self.egrid_spectrum <= self.Emin)
        # delta_penalty = self.res_spectrum[slp] - self.spectrum['spectrum'][slp]
        # delta_penalty = np.where(delta_penalty > 0., delta_penalty, 0.)
        # error_penalty = np.where(self.res_spectrum[slp] > self.spectrum['spectrum'][slp],
        #                  self.spectrum['upper_err'][slp],
        #                  self.spectrum['lower_err'][slp])

        return np.sum(
            (delta / error) ** 2
        )  # + np.sum((delta_penalty / error_penalty)**2)

    def _get_chi2_spectrum_corr(self):
        """Correlation-aware spectrum χ². Folds the bin-to-bin correlation
        matrix (``spectrum["corr"]``) into the residual:

            Δ = model - data, σ_i = stat_side_i ⊕ sys_side_i (per bin)
            C_ij = (stat_i if i==j else 0) + sys_i sys_j corr_ij
            χ² = Δᵀ C⁻¹ Δ

        Asymmetric error sides are picked per bin from the sign of Δ.
        The correlation matrix's diagonal carries a small jitter
        (``corr_jitter``) to defend against the OCR-rounded inputs not
        being strictly positive-definite.
        """
        sl = self._corr_indices
        delta = self.res_spectrum[sl] - self.spectrum["spectrum"][sl]
        model_above = delta > 0
        spec = self.spectrum
        if "stat_up" in spec and "stat_low" in spec:
            stat = np.where(model_above, spec["stat_up"][sl], spec["stat_low"][sl])
        else:
            # Fall back: legacy upper_err / lower_err are combined stat+sys; use them
            # as stat and zero out sys to recover the diagonal χ² when no split is
            # available.
            stat = np.where(model_above, spec["upper_err"][sl], spec["lower_err"][sl])
        if "sys_Up" in spec and "sys_Low" in spec:
            sys_ = np.where(model_above, spec["sys_Up"][sl], spec["sys_Low"][sl])
        else:
            sys_ = np.zeros_like(stat)

        # Covariance: diagonal stat² + outer-product sys correlated by `corr`.
        cov = (sys_[:, None] * sys_[None, :]) * self._corr_sub
        cov.flat[:: cov.shape[0] + 1] += stat ** 2
        # Solve C x = Δ via cholesky (cov is positive-definite up to OCR-rounded jitter).
        from scipy.linalg import cho_factor, cho_solve

        c, lower = cho_factor(cov, lower=True, check_finite=False)
        x = cho_solve((c, lower), delta, check_finite=False)
        return float(np.dot(delta, x))

    def get_chi2_Xmax(self, norms=None, deltaE=0.0, xmax_shift=0.0):
        if self.lst_res is not None and norms is not None:
            self.compute_combined_result(norms, deltaE)

        # chi2 for values above Emin
        sl = np.argwhere(self.egrid_xmax > self.Emin)

        if xmax_shift == 0.0:
            delta = self.res_xmax[sl] - self.Xmax["val"][sl]
        elif xmax_shift > 0.0:
            delta = (
                self.res_xmax[sl]
                - self.Xmax["val"][sl]
                + xmax_shift * self.Xmax["sys_Up"][sl]
            )
        elif xmax_shift < 0.0:
            delta = (
                self.res_xmax[sl]
                - self.Xmax["val"][sl]
                + xmax_shift * self.Xmax["sys_Low"][sl]
            )
        error = self.Xmax["stat"][sl]
        # print '\n \n'
        # print delta
        # print error
        # print delta / error
        # print (delta / error)**2
        # print '\n \n'

        return np.sum((delta / error) ** 2)

    def get_chi2_VarXmax(self, norms=None, deltaE=0.0, xmax_shift=0.0):
        if self.lst_res is not None and norms is not None:
            self.compute_combined_result(norms, deltaE)

        # chi2 for values above Emin
        sl = np.argwhere(self.egrid_xmax > self.Emin)
        if xmax_shift == 0.0:
            delta = self.res_sigma_xmax[sl] - self.XRMS["val"][sl]
        elif xmax_shift > 0.0:
            delta = (
                self.res_sigma_xmax[sl]
                - self.XRMS["val"][sl]
                + xmax_shift * self.XRMS["sys_Up"][sl]
            )
        elif xmax_shift < 0.0:
            delta = (
                self.res_sigma_xmax[sl]
                - self.XRMS["val"][sl]
                + xmax_shift * self.XRMS["sys_Low"][sl]
            )
        error = self.XRMS["stat"][sl]
        # print '\n \n'
        # print delta
        # print error
        # print delta / error
        # print (delta / error)**2
        # print '\n \n'

        return np.sum((delta / error) ** 2)

    def get_chi2_total(self, norms=None, deltaE=0.0, xmax_shift=0):
        if self.lst_res is not None and norms is not None:
            self.compute_combined_result(norms, deltaE)

        return sum(
            [
                self.get_chi2_spectrum(),
                self.get_chi2_Xmax(xmax_shift=xmax_shift),
                self.get_chi2_VarXmax(xmax_shift=xmax_shift),
            ]
        )

    def fit_data_minuit(self, spectrum_only=False, minimizer_args=None):
        """Profile chi² fit over (deltaE, xmax_shift, per-species norms).

        Rewritten 2026-05-11 for iminuit 2.x (kwargs ``forced_parameters``,
        ``error_<name>``, ``limit_<name>`` were removed; ``errordef`` /
        ``print_level`` are now attributes, not kwargs). The grid logic
        (5×5 multi-start when both shifts float; single-start otherwise) is
        preserved verbatim.
        """
        from iminuit import Minuit

        if minimizer_args is None:
            minimizer_args = {}

        n_pids = len(self.ncoids)

        def chi2(*args):
            deltaE = args[0]
            xmax_shift = args[1]
            norms = np.array(args[2:])
            if spectrum_only is True:
                return self.get_chi2_spectrum(norms=norms, deltaE=deltaE)
            if spectrum_only == "xmax":
                return self.get_chi2_spectrum(
                    norms=norms, deltaE=deltaE
                ) + self.get_chi2_Xmax(
                    norms=norms, deltaE=deltaE, xmax_shift=xmax_shift
                )
            return self.get_chi2_total(
                norms=norms, deltaE=deltaE, xmax_shift=xmax_shift
            )

        init_norm = (
            self.spectrum["spectrum"][14] / self.res_spectrum[14] / n_pids
        )
        # If model spectrum is zero at bin 14, fall back to unit norm.
        init_norm = init_norm if np.isfinite(init_norm) else 1.0
        m_best = None

        # Multi-start when a shift is allowed to float; otherwise single start at 0.
        fix_deltaE_default = True
        fix_xmax_shift_default = True
        fix_deltaE = bool(minimizer_args.get("fix_deltaE", fix_deltaE_default))
        fix_xmax_shift = bool(minimizer_args.get("fix_xmax_shift", fix_xmax_shift_default))
        delta_tries = [0.0] if fix_deltaE else [-0.13, -0.08, 0.0, 0.08, 0.13]
        shift_tries = [0.0] if fix_xmax_shift else [-0.9, -0.5, 0.0, 0.5, 0.9]

        arg_names = (
            ["deltaE", "xmax_shift"]
            + ["norm{:}".format(pid) for pid in self.ncoids]
        )

        # Same bounds / step sizes as the original 1.x invocation.
        norm_lo = init_norm / 1e6
        norm_hi = init_norm * 1e6
        errors = [0.1, 0.2] + [init_norm / 100.0] * n_pids
        limits = [(-0.14, 0.14), (-1.0, 1.0)] + [(norm_lo, norm_hi)] * n_pids

        for delta_start in delta_tries:
            for shift_start in shift_tries:
                start = [delta_start, shift_start] + [init_norm] * n_pids
                m = Minuit(chi2, *start, name=arg_names)
                m.errordef = 1.0
                m.print_level = int(minimizer_args.get("print_level", 0))
                for name, err in zip(arg_names, errors):
                    m.errors[name] = err
                for name, lim in zip(arg_names, limits):
                    m.limits[name] = lim
                m.fixed["deltaE"] = fix_deltaE
                m.fixed["xmax_shift"] = fix_xmax_shift
                # Norms always free.
                for nm in arg_names[2:]:
                    m.fixed[nm] = False
                m.migrad(ncall=100000)

                if m_best is None or m.fval < m_best.fval:
                    m_best = m

        return m_best


class UHECRWalker(object):
    def __init__(self, prince_run, spectrum, xmax, xrms, progressbar=False):
        self.prince_run = prince_run
        self.spectrum = spectrum
        self.xmax = xmax
        self.xrms = xrms
        self.progressbar = progressbar

    def compute_models(
        self,
        particle_ids,
        rmax=5.0e9,
        gamma=1.0,
        m="flat",
        sclass="auger",
        rscale=1.0,
        initial_z=1.0,
        final_z=0.0,
        max_step=1e-3,
        atol=1e40,
    ):
        """
        Compute the results corresponding to source_params for each particle id individually and return a list
        """
        from prince.solvers import UHECRPropagationSolverBDF
        from prince.cr_sources import AugerFitSource, SimpleSource, RigidityFlexSource

        lst_models = []
        for ncoid in particle_ids:

            solver = UHECRPropagationSolverBDF(
                initial_z=initial_z,
                final_z=final_z,
                prince_run=self.prince_run,
                enable_partial_diff_jacobian=True,
                atol=atol,
            )

            if sclass == "auger":
                params = {
                    ncoid: (gamma, rmax, 1.0),
                }
                source = AugerFitSource(self.prince_run, params=params, m=m, norm=1.0)
            elif sclass == "simple":
                params = {
                    ncoid: (gamma, rmax, 1.0),
                }
                source = SimpleSource(self.prince_run, params=params, m=m, norm=1.0)
            elif sclass == "rflex":
                params = {
                    ncoid: (gamma, rmax, rscale, 1.0),
                }
                source = RigidityFlexSource(
                    self.prince_run, params=params, m=m, norm=1.0
                )
            else:
                raise Exception("Unknown source class: {:}".format(sclass))
            solver.add_source_class(source)
            # solver.set_initial_condition()
            solver.solve(
                dz=max_step,
                verbose=False,
                full_reset=False,
                progressbar=self.progressbar,
            )

            lst_models.append(solver.res)

        # return the results only
        return lst_models

    def compute_gridpoint(
        self, particle_ids, spectrum_only=False, Emin=6e9, **source_params
    ):
        """
        Compute the Model on a single gridpoint and fit to data
        """
        # print 'computing with source parameters :'
        # print source_params

        lst_models = self.compute_models(particle_ids, **source_params)

        optimizer = UHECROptimizer(
            lst_models,
            self.spectrum,
            self.xmax,
            self.xrms,
            Emin=Emin,
            ncoids=particle_ids,
        )
        minres = optimizer.fit_data_minuit(spectrum_only=spectrum_only)
        lst_res = [res.to_dict() for res in optimizer.lst_res]
        mindetail = (
            minres.parameters,
            list(minres.args),
            list(minres.values.items()),
            list(minres.errors.items()),
        )
        return minres.fval, mindetail, lst_res

    def compute_lnprob_mc(
        self,
        source_params,
        particle_ids,
        return_blob=False,
        spectrum_only=False,
        Emin=6e9,
    ):
        """
        Return the chi2 for the fitted fractions in a format need by the emcee module
        """
        lst_models = self.compute_models(source_params, particle_ids)

        optimizer = UHECROptimizer(
            lst_models,
            self.spectrum,
            self.xmax,
            self.xrms,
            Emin=Emin,
            ncoids=particle_ids,
        )
        minres = optimizer.fit_data_minuit(spectrum_only=spectrum_only)

        # return either only the chi2 for MCMC, or also the computation result
        # if the computation result is returned, it will be saved in the MCMC chain
        lst_res = [res.to_dict() for res in optimizer.lst_res]
        mindetail = minres.parameters, minres.args, minres.values, minres.errors

        if return_blob:
            # return minres.fval, (minres, optimizer.lst_res)
            return minres.fval, (mindetail, lst_res)
        else:
            return minres.fval

    def __call__(self, params, pids):
        return self.compute_lnprob_mc(params, pids)

    def run_mcmc(self, params, pids, nwalkers=100, nsamples=100, mpi=False, threads=1):
        """
        Runs an MCMC chain
        """
        # Setup the pool, to map lnprob
        import schwimmbad
        import emcee

        pool = schwimmbad.choose_pool(mpi=mpi, processes=threads)

        ndim = len(params)
        pos0 = [params + params * np.random.randn(ndim) * 0.01 for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self, args=(pids,), pool=pool)
        sampler.run_mcmc(pos0, nsamples)

        pool.close()
        return sampler.chain
