import h5py
import numpy as np


class ScanPlotter(object):

    def __init__(self, filepath, input_spec, paramlist, fit=None):
        self.filepath = filepath

        with h5py.File(self.filepath, "r") as f:
            self.available = list(f.keys())

            if fit is None:
                if "fixed E" in f:
                    fit = "fixed E"
                elif "default fit" in f:
                    fit = "default fit"
                else:
                    raise Exception("No fit in file!")

            self.chi2_array = f[fit]["chi2"][:]
            self.norm_array = f[fit]["norm"][:]
            self.deltaE_array = f[fit]["delta E"][:]

            if "xmax_shift" in f[fit]:
                self.xshift_array = f[fit]["xmax_shift"][:]
            else:
                self.xshift_array = np.zeros_like(self.deltaE_array)

            self.fractions_array = f[fit]["fractions"][:]

            self.egrid = f["egrid"][:]
            self.known_spec = f["known_spec"][:]

        self.input_spec = input_spec
        self.paramlist = paramlist

    def reload_fit(self, name):
        with h5py.File(self.filepath, "r") as f:
            self.chi2_array = f[name]["chi2"][:]
            self.norm_array = f[name]["norm"][:]
            self.deltaE_array = f[name]["delta E"][:]
            self.fractions_array = f[name]["fractions"][:]

    def print_summary(self, index=None):
        index = self.minindex if index is None else index
        print("-----------------------------")
        print("| Summary:                  |")
        print("-----------------------------")
        print(("| Best fit chi2: ", self.chi2_array[index]))
        print(("| at parameters: ", self.index2params(index)))
        print(("| norm:          ", self.norm_array[index]))
        print(("| E-shift        ", self.deltaE_array[index]))
        print(("| xmax-shift     ", self.xshift_array[index]))
        print(
            "| fractions:     "
        )  # , ['{:.2f}'.format(f*100) for f in self.fractions_array[index]]
        nocontr = []
        for spec, percent in zip(self.input_spec, self.fractions_array[index]):
            if percent > 0.001:
                print(("| {:} with {:.2f} %".format(spec, percent * 100)))
            else:
                nocontr.append(spec)
        print(("< 0.1 %:", nocontr))

        if hasattr(self, "lum_fractions"):
            print("-----------------------------")

            print(
                "| lum fractions:     "
            )  # , ['{:.2f}'.format(f*100) for f in self.fractions_array[index]]
            nocontr = []
            for spec, percent in zip(self.input_spec, self.lum_fractions[index]):
                if percent > 0.001:
                    print(("| {:} with {:.2f} %".format(spec, percent * 100)))
                else:
                    nocontr.append(spec)
            print(("< 0.1 %:", nocontr))
            print("-----------------------------")

    @property
    def paramnames(self):
        return [p[0] for p in self.paramlist]

    @property
    def paramvalues(self):
        return [p[1] for p in self.paramlist]

    @property
    def minchi2(self):
        return self.chi2_array.min()

    @property
    def minindex(self):
        return np.unravel_index(self.chi2_array.argmin(), self.chi2_array.shape)

    def index2params(self, index):
        return tuple(p[i] for p, i in zip(self.paramvalues, index))

    def closest_params(self, params):
        return tuple(
            (np.abs(array - value)).argmin()
            for array, value in zip(self.paramvalues, params)
        )

    @property
    def permutations(self):
        import itertools as it

        # Create a list of all permutations of the scan parameters
        permutations = it.product(*[list(range(arr[1].size)) for arr in self.paramlist])
        return list(permutations)

    def get_states(self, index):
        with h5py.File(self.filepath, "r") as f:
            print((f["states"].shape))
            states = f["states"][index]
        return states

    def get_results(self, index):
        with h5py.File(self.filepath, "r") as f:
            states = f["states"][index]

        dicts = [
            {"egrid": self.egrid, "known_spec": self.known_spec, "state": state}
            for state in states
        ]
        from prince.solvers import UHECRPropagationResult

        results = [UHECRPropagationResult.from_dict(d) for d in dicts]
        return results

    def get_comb_result(self, index, frac="best"):
        results = self.get_results(index)
        norm = self.norm_array[index]
        if frac == "best":
            fractions = self.fractions_array[index]
        else:
            fractions = np.array(frac)

        return np.sum([res * f for f, res in zip(fractions, results)]) * norm

    def comp_neutrino_band(
        self, chi_max=14.16, pbar=False, fix_m=None, exclude_idx=[], epow=2, pids="nu"
    ):
        """Reads out the neutrino fluxes within a range defined by chi_max"""

        if fix_m is None:
            sig = np.argwhere(self.chi2_array - self.chi2_array.min() < chi_max)
            sig = [tuple(idx) for idx in sig]
        else:
            sig = np.argwhere(
                self.chi2_array[:, :, fix_m] - self.chi2_array[:, :, fix_m].min()
                < chi_max
            )
            sig = [(idx[0], idx[1], fix_m) for idx in sig]

        sig = [tup for tup in sig if tup not in exclude_idx]

        if pbar:
            from tqdm import tqdm_notebook as tqdm
        else:
            # if no pbar, return an empty function as placeholder
            def tqdm(a):
                return a

        band = []
        indices = []
        for idx in tqdm(sig):
            res = self.get_comb_result(tuple(idx))
            egrid, spec = res.get_solution_group(pids, epow=epow)
            band.append(spec)
            indices.append(idx)
        band = np.array(band)
        indices = np.array(indices)
        return egrid, band, indices

    def comp_lum_integral(self, index, prince_run):
        gamma, Rcut, m = self.index2params(index)
        norm = self.norm_array[index]
        fractions = self.fractions_array[index]
        spec = self.input_spec

        params = {}
        for s, f in zip(spec, fractions):
            params[s] = (gamma, Rcut, f)

        from prince.cr_sources import AugerFitSource

        source = AugerFitSource(prince_run, params=params, m=float(m), norm=norm)

        return source.integrated_lum(Emin=1e9)

    def comp_lum_frac(self, prince_run, pbar=False):
        self.num_fractions = np.zeros_like(self.fractions_array)
        self.lum_fractions = np.zeros_like(self.fractions_array)

        if pbar:
            from tqdm import tqdm_notebook as tqdm
        else:
            # if no pbar, return an empty function as placeholder
            def tqdm(a):
                return a

        for index in tqdm(self.permutations):
            f1, f2 = self.comp_lum_integral(index, prince_run)
            self.num_fractions[index] = f1 / f1.sum()
            self.lum_fractions[index] = f2 / f2.sum()

        # import cPickle as pickle
        # import os.path as path
        # lustre = path.expanduser("~/lustre/")
        # with open(lustre + 'prince_run_PSB.ppo', 'rb') as thefile:
        #     prince_run = pickle.load(thefile)
        # self.prince_run = prince_run

        # self.num_fractions = np.zeros_like(self.fractions_array)
        # self.lum_fractions = np.zeros_like(self.fractions_array)

        # from tqdm import tqdm_notebook as tqdm
        # for index in tqdm(self.permutations):
        #     f1,f2 = self.comp_lum_integral(index)
        #     self.num_fractions[index] = f1 / f1.sum()
        #     self.lum_fractions[index] = f2 / f2.sum()

    def recompute_fit(
        self,
        index,
        minimizer_args={},
        Emin=6e9,
        xmax_model=None,
        spectrum_only=False,
        dataset=2017,
    ):
        if dataset == 2019:
            from .spectra import auger2019 as spec
            from .spectra import Xmax2019 as xmax
            from .spectra import XRMS2019 as xrms
        elif dataset == 2017:
            from .spectra import auger2017 as spec
            from .spectra import Xmax2017 as xmax
            from .spectra import XRMS2017 as xrms
        elif dataset == 2015:
            from .spectra import auger2015 as spec
            from .spectra import Xmax2015 as xmax
            from .spectra import XRMS2015 as xrms
        else:
            raise Exception("Unknown dataset from year {:}".format(dataset))

        lst_results = self.get_results(index)
        from .optimizer import UHECROptimizer

        optimizer = UHECROptimizer(
            lst_results,
            spec,
            xmax,
            xrms,
            Emin=Emin,
            ncoids=self.input_spec,
            xmax_model=xmax_model,
        )
        params = {"fix_deltaE": True, "fix_xmax_shift": True}
        params.update(minimizer_args)
        minres = optimizer.fit_data_minuit(
            spectrum_only=spectrum_only, minimizer_args=params
        )

        return minres, optimizer

    def recompute_fit_proton_component(
        self,
        index,
        proton_idx,
        minimizer_args={},
        Emin=6e9,
        xmax_model=None,
        spectrum_only=False,
        dataset=2017,
    ):
        if dataset == 2017:
            from .spectra import auger2017 as spec
            from .spectra import Xmax2017 as xmax
            from .spectra import XRMS2017 as xrms
        elif dataset == 2015:
            from .spectra import auger2015 as spec
            from .spectra import Xmax2015 as xmax
            from .spectra import XRMS2015 as xrms
        else:
            raise Exception("Unknown dataset from year {:}".format(dataset))

        lst_results = self.get_results(index)
        proton_result = self.get_results(proton_idx)[0]

        lst_results.append(proton_result)

        from .optimizer import UHECROptimizer

        ncoids = np.append(self.input_spec, 10100)

        optimizer = UHECROptimizer(
            lst_results,
            spec,
            xmax,
            xrms,
            Emin=Emin,
            ncoids=ncoids,
            xmax_model=xmax_model,
        )
        params = {"fix_deltaE": False, "fix_xmax_shift": True}
        params.update(minimizer_args)
        minres = optimizer.fit_data_minuit(
            spectrum_only=spectrum_only, minimizer_args=params
        )

        return minres, optimizer

    def recompute_scan(
        self,
        name="new fit",
        minimizer_args={},
        Emin=6e9,
        spectrum_only=False,
        dataset=2017,
        xmax_model=None,
    ):
        chi2_new = np.zeros_like(self.chi2_array)
        norm_new = np.zeros_like(self.norm_array)
        deltaE_new = np.zeros_like(self.deltaE_array)
        xshift_new = np.zeros_like(self.xshift_array)
        fractions_new = np.zeros_like(self.fractions_array)

        from tqdm import tqdm as tqdm

        for index in tqdm(self.permutations):
            params = {"print_level": 0.0}
            params.update(minimizer_args)
            m, _ = self.recompute_fit(
                index,
                minimizer_args=params,
                Emin=Emin,
                spectrum_only=spectrum_only,
                dataset=dataset,
                xmax_model=xmax_model,
            )
            mindetail = (
                m.parameters,
                list(m.args),
                list(m.values.items()),
                list(m.errors.items()),
            )

            chi2_new[index] = m.fval
            norm_new[index] = sum(mindetail[1][2:])
            deltaE_new[index] = mindetail[1][0]
            xshift_new[index] = mindetail[1][0]
            fractions_new[index] = [f / norm_new[index] for f in mindetail[1][2:]]

        with h5py.File(self.filepath, "r+") as f:
            grp = f.require_group(name)
            d_chi2 = grp.require_dataset("chi2", chi2_new.shape, dtype=np.float64)
            d_chi2[:] = chi2_new
            d_norm = grp.require_dataset("norm", norm_new.shape, dtype=np.float64)
            d_norm[:] = norm_new
            d_deltaE = grp.require_dataset(
                "delta E", deltaE_new.shape, dtype=np.float64
            )
            d_deltaE[:] = deltaE_new
            xshift_new = grp.require_dataset(
                "xmax_shift", xshift_new.shape, dtype=np.float64
            )
            xshift_new[:] = xshift_new
            d_fractions = grp.require_dataset(
                "fractions", fractions_new.shape, dtype=np.float64
            )
            d_fractions[:] = fractions_new

        self.chi2_array = chi2_new
        self.norm_array = norm_new
        self.deltaE_array = deltaE_new
        self.fractions_array = fractions_new
