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
    ):
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
            norms = np.ones_like(self.lst_res, dtype=np.float)
        if ncoids is None:
            ncoids = list(range(len(self.lst_res)))
        self.ncoids = ncoids

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

    def fit_data_minuit(self, spectrum_only=False, minimizer_args={}):
        from iminuit import Minuit

        def chi2(deltaE, xmax_shift, *norms):
            norms = np.array(norms)
            # norms = np.array(norms)
            if spectrum_only is True:
                result = self.get_chi2_spectrum(norms=norms, deltaE=deltaE)
            elif spectrum_only == "xmax":
                result = self.get_chi2_spectrum(
                    norms=norms, deltaE=deltaE
                ) + self.get_chi2_Xmax(
                    norms=norms, deltaE=deltaE, xmax_shift=xmax_shift
                )
            else:
                result = self.get_chi2_total(
                    norms=norms, deltaE=deltaE, xmax_shift=xmax_shift
                )
            return result

        init_norm = (
            self.spectrum["spectrum"][14] / self.res_spectrum[14] / len(self.ncoids)
        )
        # trick if spectrum is zero, will iMinuit will give nan anyway
        init_norm = init_norm if np.isfinite(init_norm) else 1.0
        m_best = None

        if "fix_deltaE" in minimizer_args and not minimizer_args["fix_deltaE"]:
            delta_tries = [-0.13, -0.8, 0.0, 0.8, 0.13]
            # delta_tries = [-0.12, 0., 0.12]

        else:
            delta_tries = [0.0]

        if "fix_xmax_shift" in minimizer_args and not minimizer_args["fix_xmax_shift"]:
            pass
            shift_tries = [-0.9, -0.5, 0.0, 0.5, 0.9]
            # shift_tries = [-0.9, 0., 0.9]
        else:
            shift_tries = [0.0]

        for delta_start in delta_tries:
            for shift_start in shift_tries:

                arg_names = (
                    ["deltaE"]
                    + ["xmax_shift"]
                    + ["norm{:}".format(pid) for pid in self.ncoids]
                )
                start = [delta_start] + [shift_start] + [init_norm] * len(self.ncoids)
                error = [0.1] + [0.2] + [init_norm / 100] * len(self.ncoids)
                limit = (
                    [(-0.14, 0.14)]
                    + [(-1.0, 1.0)]
                    + [(init_norm / 1e6, init_norm * 1e6)] * len(self.ncoids)
                )

                params = {"fix_deltaE": True, "fix_xmax_shift": True, "print_level": 0}
                params.update({name: val for name, val in zip(arg_names, start)})
                params.update(
                    {"error_" + name: val for name, val in zip(arg_names, error)}
                )
                params.update(
                    {"limit_" + name: val for name, val in zip(arg_names, limit)}
                )

                params.update(minimizer_args)
                m = Minuit(chi2, forced_parameters=arg_names, errordef=1.0, **params)
                # m.print_param()
                m.migrad(ncall=100000)

                if m_best is None:
                    m_best = m
                if m.fval < m_best.fval:
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
