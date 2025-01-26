import os.path as path


template_submit = """#!/bin/zsh
#$ -N {project_tag}
#$ -l h_rt={hours}:00:00
#$ -l h_rss={mem}G
#$ -j y
#$ -m ae
#$ -o {folder_log}/{project_tag}$TASK_ID.log

OUTFILE={folder_out}/{project_tag}$SGE_TASK_ID.out
TMPOUT=$TMPDIR/tmp.out

echo Starting job with options on
echo `hostname`. Now is `date`

source ~/.zshrc
python {runfile} -r --jobid $SGE_TASK_ID --outfile $TMPOUT

#Copy output to destination
mv $TMPOUT $OUTFILE
"""


class PropagationProject(object):
    def __init__(self, conf, dryrun=False):
        self.conf = conf
        if "fit_only" in conf:
            self.fit_only = conf["fit_only"]
        else:
            self.fit_only = False

        # tag used for filenames and basefolder
        self.project_tag = conf["project_tag"]
        if "fit_tag" in conf:
            self.fit_tag = conf["fit_tag"]
        # basefolder where to create the project tag
        self.targetdir = path.join(conf["targetdir"], self.project_tag)
        # path where the input file is located
        self.inputpath = conf["inputpath"]

        # the subfolders, log files and output
        if self.fit_only:
            self.folder_log = path.join(self.targetdir, "log_fit")
            self.folder_out = path.join(self.targetdir, "out_fit")
        else:
            self.folder_log = path.join(self.targetdir, "log")
            self.folder_out = path.join(self.targetdir, "out")

        # list of parameters to run the prog with
        self.paramlist = conf["paramlist"]
        self.njobs = conf["njobs"]

        if "run_subset" in conf and conf["run_subset"] is True:
            self._perm_subset = conf["perm_subset"]
        else:
            self._perm_subset = None

        self.max_memory = conf["max memory GB"] if "max memory GB" in conf else 2
        self.hours_per_job = conf["hours per job"] if "hours per job" in conf else 3

    # shortcuts for parameters
    @property
    def param_names(self):
        if type(self.paramlist) is dict:
            return list(self.paramlist.keys())
        elif type(self.paramlist) is tuple:
            return tuple(param[0] for param in self.paramlist)

    @property
    def param_values(self):
        if type(self.paramlist) is dict:
            return list(self.paramlist.values())
        elif type(self.paramlist) is tuple:
            return tuple(param[1] for param in self.paramlist)

    @property
    def permutations(self):
        if self._perm_subset is not None:
            return self._perm_subset
        else:
            import itertools as it

            # Create a list of all permutations of the scan parameters
            permutations = it.product(
                *[list(range(arr.size)) for arr in self.param_values]
            )
            return list(permutations)

    def index_to_params(self, index):
        values = self.param_values
        return [v[i] for i, v in zip(index, values)]

    def params_to_index(self, params):
        import numpy as np

        values = self.param_values
        names = self.param_names
        res = []
        for par, na, arr in zip(params, names, values):
            loc = np.argwhere(arr == par)
            if len(loc) < 1:
                raise Exception(
                    "Error: could not find value ({:}) for the parameter ({:})".format(
                        par, na
                    )
                )
            elif len(loc) > 1:
                raise Exception(
                    "Error: found value {:} for parameter ({:}) {:} times".format(
                        par, na, len(loc)
                    )
                )
            else:
                res.append(int(loc[0]))
        return tuple(res)

    def index_to_jobid(self, idx):
        import numpy as np

        perms = np.array(self.permutations)

        loc = np.argwhere((perms == idx).all(axis=1)).flatten()
        if len(loc) < 1:
            raise Exception(
                "Error: could not find index ({:}) in permutations".format(idx)
            )
        elif len(loc) > 1:
            raise Exception(
                "Error: found index {:} in permutations {:} times".format(idx, len(loc))
            )
        loc = loc[0]
        div = loc / self.njobs
        rest = loc % self.njobs
        # The jobid is given by the rest, the location in the job results list is given by the devision
        jobid = rest + 1  # plus 1 here, as job indexing starts at 1 and not zero
        jobloc = div

        # test by forwarding again to perm_slice
        if not idx == self.perm_slice(jobid)[jobloc]:
            raise Exception(
                "Error: Wrong result for index {:}, found jobid {:} and jobloc {:}, which leads to {:}".format(
                    idx, jobid, jobloc, self.perm_slice(jobid)[jobloc]
                )
            )
        else:
            return jobid, jobloc

    def perm_slice(self, jobid):
        return self.permutations[jobid - 1 :: self.njobs]

    def values_slice(self, jobid):
        index_list = self.perm_slice(jobid)
        return [self.index_to_params(idx) for idx in index_list]

    @property
    def runfile(self):
        if self.fit_only:
            return path.join(self.targetdir, "fit_" + self.fit_tag + ".py")
        else:
            return path.join(self.targetdir, "run.py")

    @property
    def subfile(self):
        if self.fit_only:
            return path.join(self.targetdir, "fit_" + self.fit_tag + ".sh")
        else:
            return path.join(self.targetdir, "sub.sh")

    def logfile(self, num):
        if self.fit_only:
            return self.fit_tag + "{:}.log".format(num)
        else:
            return self.project_tag + "{:}.log".format(num)

    def outfile(self, num):
        if self.fit_only:
            return self.fit_tag + "{:}.out".format(num)
        else:
            return self.project_tag + "{:}.out".format(num)

    def setup_project(self):
        """Sets up the standard folders and files in the project folder"""
        from os import makedirs

        # step 1: create the project folders
        try:
            print("making directories:")
            print((self.folder_log))
            print((self.folder_out))
            makedirs(self.folder_log)
            makedirs(self.folder_out)
        except:  # noqa: E722
            raise Exception(
                "_setup_project():: folders already exists, delete old files first!!"
            )

        # step 2: create files in the target folder
        from shutil import copyfile

        copyfile(self.inputpath, self.runfile)
        # step 3: create a submit file from template
        with open(self.subfile, "w") as subfile:
            subfile.write(
                template_submit.format(
                    project_tag=self.project_tag,
                    runfile=self.runfile,
                    folder_log=self.folder_log,
                    folder_out=self.folder_out,
                    hours=self.hours_per_job,
                    mem=self.max_memory,
                )
            )

    def setup_fit(self):
        """Sets up the standard folders and files in the project folder"""
        from os import makedirs

        # step 1: create the project folders
        try:
            print("making directories:")
            print((self.folder_log))
            print((self.folder_out))
            makedirs(self.folder_log)
            makedirs(self.folder_out)
        except:  # noqa: E722
            pass

        # step 2: create files in the target folder
        try:
            from shutil import copyfile

            copyfile(self.inputpath, self.runfile)
        except:  # noqa: E722
            # we will assume, the file is already in the correct folder
            pass

        # step 3: create a submit file from template
        with open(self.subfile, "w") as subfile:
            subfile.write(
                template_submit.format(
                    project_tag=self.fit_tag,
                    runfile=self.runfile,
                    folder_log=self.folder_log,
                    folder_out=self.folder_out,
                    hours=self.hours_per_job,
                    mem=self.max_memory,
                )
            )

    def scan_logfiles(self):
        """Scans the log folder for missing files"""
        import os

        import itertools

        def ranges(i):
            for a, b in itertools.groupby(enumerate(i), lambda x_y: x_y[1] - x_y[0]):
                b = list(b)
                yield b[0][1], b[-1][1]

        expected = list(range(1, self.njobs + 1))
        existing = os.listdir(self.folder_log)
        found = [idx for idx in expected if self.logfile(idx) in existing]
        found = list(ranges(found))
        missing = [idx for idx in expected if self.logfile(idx) not in existing]
        num_missing = len(missing)
        missing = list(ranges(missing))
        print("------------------------------")
        print("missing logfiles:")
        print(
            (
                ",\n".join(
                    [
                        (
                            "{:}-{:}".format(*tup)
                            if not tup[0] == tup[1]
                            else "{:}".format(tup[0])
                        )
                        for tup in missing
                    ]
                )
            )
        )
        print(("total missing files:", num_missing))
        print("------------------------------")
        return found, missing

    def scan_output(self):
        """Scans the output folder for missing files"""
        import os

        import itertools

        def ranges(i):
            for a, b in itertools.groupby(enumerate(i), lambda x_y1: x_y1[1] - x_y1[0]):
                b = list(b)
                yield b[0][1], b[-1][1]

        expected = list(range(1, self.njobs + 1))
        existing = os.listdir(self.folder_out)
        found = [idx for idx in expected if self.outfile(idx) in existing]
        found = list(ranges(found))
        missing = [idx for idx in expected if self.outfile(idx) not in existing]
        num_missing = len(missing)
        missing = list(ranges(missing))
        print("------------------------------")
        print("missing outputfiles:")
        print(
            (
                ",\n".join(
                    [
                        (
                            "{:}-{:}".format(*tup)
                            if not tup[0] == tup[1]
                            else "{:}".format(tup[0])
                        )
                        for tup in missing
                    ]
                )
            )
        )
        print(("total missing files:", num_missing))
        print("------------------------------")
        return found, missing

    def submit_all_jobs(self):
        """Submits a job array"""
        import subprocess

        _ = subprocess.call(["qsub", "-t", "1:{:}".format(self.njobs), self.subfile])

    def run_subset(self, jobid, outputfile):
        """Run the calculations for a subset of the parameter space"""

        # Runs the function supplied by config on a a fraction of the parameter space
        # Fraction depends on the number of total jobs
        setup = self.conf["setup_func"]()
        results = []

        for perm in self.perm_slice(jobid):
            # inp = {}
            # for key, arr, idx in zip(self.param_names, self.param_values,
            #                          perm):
            #     inp[key] = arr[idx]

            # func = self.conf['single_run_func']
            # results.append(func(setup, **inp))
            perm = tuple(perm)
            func = self.conf["single_run_func"]
            results.append(func(setup, perm))

        # Save the list of results to pickle
        import pickle as pickle

        with open(outputfile, "wb") as thefile:
            pickle.dump(results, thefile, protocol=pickle.HIGHEST_PROTOCOL)
        print(("collected results dumped to ", outputfile))

    def submit_missing_jobs(self):
        import subprocess
        import os

        _, missing = self.scan_output()
        for jobid in missing:
            for sid in range(jobid[0], jobid[1] + 1):
                if os.path.exists("./log/{:}".format(self.logfile(sid))):
                    os.remove("./log/{:}".format(self.logfile(sid)))
            _ = subprocess.call(
                ["qsub", "-t", "{:}:{:}".format(*jobid), self.subfile]
            )

    def submit_single_job(self, jobid):
        import subprocess

        _ = subprocess.call(
            ["qsub", "-t", "{:}:{:}".format(jobid, jobid), self.subfile]
        )

    def check_job_results(self):
        """Check the computed output pickle files for integrity"""

        import pickle as pickle
        import os.path as path

        # Loop over the single output files
        from tqdm import tqdm

        print("reading output files:")
        for jobid in tqdm(list(range(1, self.njobs + 1))):
            outputfile = path.join(self.folder_out, self.outfile(jobid))
            with open(outputfile, "rb") as thefile:
                try:
                    results = pickle.load(thefile)  # noqa: F841
                except:  # noqa: E722
                    print(("Error reading jobfile {:}".format(jobid)))

    def collect_job_results(self):
        """Collect the computed results to a single array"""
        _, missing = self.scan_output()
        if len(missing) != 0:
            raise Exception(
                "Cannot collect results, not all results were computed yet!"
            )

        import numpy as np
        import pickle as pickle
        import os.path as path

        # Create an array of the needed size
        shape = tuple(arr.size for arr in self.param_values)

        # create a hdf5 file to store the data intensive stuff
        import h5py

        h5file = h5py.File(path.join(self.targetdir, "collected.hdf5"), "w")

        # read first output to get the grid dimensions
        outputfile = path.join(self.folder_out, self.outfile(1))
        with open(outputfile, "rb") as thefile:
            try:
                results = pickle.load(thefile)
            except:
                print(("Error reading jobfile {:}".format(1)))
                raise

        chi2, minres, results = results[0]
        egrid = results[0]["egrid"]
        state = results[0]["state"]
        known_spec = np.array(results[0]["known_spec"], dtype=np.int)
        frac = np.array(minres[1][2:])
        injected = len(results)

        # create datasets on hdf5
        dset = h5file.create_dataset("egrid", (egrid.size,), dtype=np.float64)
        dset[:] = egrid
        dset = h5file.create_dataset("known_spec", (known_spec.size,), dtype=np.int32)
        dset[:] = known_spec
        d_states = h5file.create_dataset(
            "states", shape + (injected, state.size), dtype=np.float64
        )
        grp = h5file.create_group("default fit")
        d_chi2 = grp.create_dataset("chi2", shape, dtype=np.float64)
        d_norm = grp.create_dataset("norm", shape, dtype=np.float64)
        d_deltaE = grp.create_dataset("delta E", shape, dtype=np.float64)
        d_xshift = grp.create_dataset("xmax_shift", shape, dtype=np.float64)
        d_fractions = grp.create_dataset(
            "fractions", shape + (frac.size,), dtype=np.float64
        )

        # Loop over the single output files
        from tqdm import tqdm

        print("reading output files:")
        for jobid in tqdm(list(range(1, self.njobs + 1))):
            outputfile = path.join(self.folder_out, self.outfile(jobid))
            with open(outputfile, "rb") as thefile:
                try:
                    results = pickle.load(thefile)
                except:
                    print(("Error reading jobfile {:}".format(jobid)))
                    raise

            # write to arrays
            for res, perm in zip(results, self.perm_slice(jobid)):
                perm = tuple(perm)
                chi2, minres, results = res
                if chi2 == minres == results == np.inf:
                    # Something went wrong in this case, no data there, just continue
                    d_chi2[perm] = np.inf
                    continue
                stack = np.vstack([r["state"] for r in results])
                dE = minres[1][0]
                xshift = minres[1][1]
                norm = sum(minres[1][2:])
                frac = [f / norm for f in minres[1][2:]]

                d_states[perm] = stack
                d_chi2[perm] = chi2
                d_norm[perm] = norm
                d_deltaE[perm] = dE
                d_xshift[perm] = xshift
                d_fractions[perm] = frac
                h5file.flush()

        h5file.flush()
        h5file.close()

    def collect_fit_results(self):
        """Collect the computed results to a single array"""
        _, missing = self.scan_output()
        if len(missing) != 0:
            raise Exception(
                "Cannot collect results, not all results were computed yet!"
            )

        import numpy as np
        import pickle as pickle
        import os.path as path

        # Create an array of the needed size
        shape = tuple(arr.size for arr in self.param_values)

        # create a hdf5 file to store the data intensive stuff
        import h5py

        h5file = h5py.File(path.join(self.targetdir, "collected.hdf5"), "r+")

        # read first output to get the grid dimensions
        outputfile = path.join(self.folder_out, self.outfile(1))
        with open(outputfile, "rb") as thefile:
            results = pickle.load(thefile)

        chi2, minres = results[0]
        frac = np.array(minres[1][2:])

        # create datasets on hdf5
        grp = h5file.require_group(self.fit_tag)
        d_chi2 = grp.require_dataset("chi2", shape, dtype=np.float64)
        d_norm = grp.require_dataset("norm", shape, dtype=np.float64)
        d_deltaE = grp.require_dataset("delta E", shape, dtype=np.float64)
        d_xshift = grp.require_dataset("xmax_shift", shape, dtype=np.float64)
        d_fractions = grp.require_dataset(
            "fractions", shape + (frac.size,), dtype=np.float64
        )

        # Loop over the single output files
        from tqdm import tqdm

        print("reading output files:")
        for jobid in tqdm(list(range(1, self.njobs + 1))):
            outputfile = path.join(self.folder_out, self.outfile(jobid))
            with open(outputfile, "rb") as thefile:
                results = pickle.load(thefile)

            # write to arrays
            for res, perm in zip(results, self.perm_slice(jobid)):
                chi2, minres = res
                dE = minres[1][0]
                xshift = minres[1][1]
                norm = sum(minres[1][2:])
                frac = [f / norm for f in minres[1][2:]]

                perm = tuple(perm)
                d_chi2[perm] = chi2
                d_norm[perm] = norm
                d_deltaE[perm] = dE
                d_xshift[perm] = xshift
                d_fractions[perm] = frac
                h5file.flush()

        h5file.flush()
        h5file.close()

    def collect_fireball_results(self, superphotos=False):
        """Collect the computed results to a single array"""
        _, missing = self.scan_output()
        if len(missing) != 0:
            raise Exception(
                "Cannot collect results, not all results were computed yet!"
            )

        print(
            ("Will only collect superphotospheric collisions: {:}".format(superphotos))
        )

        import numpy as np
        import pickle as pickle
        import os.path as path

        # Create an array of the needed size
        shape = tuple(arr.size for arr in self.param_values)

        # create a hdf5 file to store the data intensive stuff
        import h5py

        if superphotos:
            h5file = h5py.File(
                path.join(self.targetdir, "collected_super_photos.hdf5"), "a"
            )
        else:
            h5file = h5py.File(path.join(self.targetdir, "collected.hdf5"), "a")

        # read first output to get the grid dimensions
        outputfile = path.join(self.folder_out, self.outfile(1))
        with open(outputfile, "rb") as thefile:
            try:
                results = pickle.load(thefile)
            except:  # noqa: E722
                print(("Error reading jobfile {:}".format(1)))
                raise

        if superphotos:
            try:
                egrid = results[0].specCosmicRays.source_energy
            except:  # noqa: E722
                egrid = results[0].specAllCosmicRays.source_energy
        else:
            egrid = results[0].specAllCosmicRays.source_energy

        grp = h5file.require_group("source")
        dset_egrid = grp.require_dataset("egrid", (egrid.size,), dtype=np.float64)
        dset_egrid[:] = egrid

        # use first two parameters to determine array shape, last one assumed to be composition
        shape = tuple(len(ls) for ls in self.param_values[:-1])

        # Loop over the single output files
        from tqdm import tqdm as tqdm

        print("reading output files:")
        for jobid in tqdm(list(range(1, self.njobs + 1))):
            outputfile = path.join(self.folder_out, self.outfile(jobid))
            with open(outputfile, "rb") as thefile:
                try:
                    fireballs = pickle.load(thefile)
                except:
                    print(("Error reading jobfile {:}".format(jobid)))
                    raise
            # write to arrays
            for fb, perm in zip(fireballs, self.perm_slice(jobid)):
                perm = tuple(perm)
                params = tuple(p[i] for p, i in zip(self.param_values, perm))
                # comp = params[-1]
                if len(params[-1]) > 1:
                    tag = "mixed"
                else:
                    tag = str(list(params[-1].keys())[0])
                subgrp = grp.require_group(tag)

                # TODO: max idx 400: workaround for faulty files, remove when files are fixed
                if superphotos:
                    try:
                        indices = fb.specCosmicRays.pids
                    except:  # noqa: E722
                        indices = fb.specAllCosmicRays.pids
                else:
                    indices = fb.specAllCosmicRays.pids
                nonzero_ids = np.nonzero(indices)
                indices = indices[nonzero_ids]
                if superphotos:
                    try:
                        spectra = fb.specCosmicRays.source_spectrum().T[nonzero_ids]
                        neutrinos = fb.specNeutrinos.source_spectrum().T
                    except:  # noqa: E722
                        spectra = fb.specAllCosmicRays.source_spectrum().T[nonzero_ids]
                        neutrinos = fb.specAllNeutrinos.source_spectrum().T
                        spectra = np.zeros_like(spectra)
                        neutrinos = np.zeros_like(neutrinos)
                else:
                    spectra = fb.specAllCosmicRays.source_spectrum().T[nonzero_ids]
                    neutrinos = fb.specAllNeutrinos.source_spectrum().T
                dset_indices = subgrp.require_dataset(
                    "indices", shape + indices.shape, dtype=np.int64
                )
                dset_spectra = subgrp.require_dataset(
                    "spectra", shape + spectra.shape, dtype=np.float64
                )
                dset_neutrinos = subgrp.require_dataset(
                    "neutrinos", shape + neutrinos.shape, dtype=np.float64
                )
                dset_indices[perm[:-1]] = indices
                dset_spectra[perm[:-1]] = spectra
                dset_neutrinos[perm[:-1]] = neutrinos

        h5file.flush()
        h5file.close()

    def run_from_terminal(self):
        from optparse import OptionParser, OptionGroup

        usage = """usage: %prog [options] args"""
        parser = OptionParser(usage=usage)

        parser.add_option(
            "-c",
            "--create",
            dest="create",
            action="store_true",
            help="If this is set, the project folder will be created",
        )
        parser.add_option(
            "-s",
            "--submit",
            dest="submit",
            action="store_true",
            help="If this is set, the project will be submitted to the cluster",
        )
        parser.add_option(
            "-m",
            "--miss",
            dest="missing",
            action="store_true",
            help="If this is set, the project search for missing output and logfiles",
        )
        parser.add_option(
            "-r",
            "--run",
            dest="run",
            action="store_true",
            help="If this is set, a single calculations from the project will be run",
        )
        parser.add_option(
            "--check",
            dest="check",
            action="store_true",
            help="If this is set, the project results checked for integrity",
        )
        parser.add_option(
            "--collect",
            dest="collect",
            action="store_true",
            help="If this is set, the project results will be collected into a single folder",
        )
        parser.add_option(
            "--fit",
            dest="fit",
            action="store_true",
            help="If this is set, will assume that the model is already computed and only do a new fit",
        )

        parser.add_option(
            "--fireball",
            dest="fireball",
            action="store_true",
            help="If this is set, will assume that the objects to collect are ReMuS fireball objects",
        )

        parser.add_option(
            "--superphotos",
            dest="superphotos",
            action="store_true",
            help="If this is set and the objects to collect are ReMuS fireballs, will only account for superphotospheric collisions",
        )

        parser.add_option(
            "--single",
            dest="single",
            action="store_true",
            help="Submit only a single job",
        )

        run_group = OptionGroup(
            parser, "Options for a single calculations, need -r to be set"
        )
        run_group.add_option(
            "--outfile",
            dest="outputfile",
            type="str",
            help="ouput will be written to this file",
        )
        run_group.add_option(
            "--jobid",
            dest="jobid",
            type="int",
            help="ID of the job in distributed scan",
        )
        parser.add_option_group(run_group)
        options, args = parser.parse_args()

        if options.create:
            if self.fit_only:
                self.setup_fit()
            else:
                self.setup_project()
        elif options.submit and options.single:
            self.submit_single_job(options.jobid)
        elif options.submit and options.missing:
            self.submit_missing_jobs()
        elif options.submit:
            self.submit_all_jobs()
        elif options.missing:
            self.scan_logfiles()
            self.scan_output()
        elif options.run:
            self.run_subset(options.jobid, options.outputfile)
        elif options.check:
            self.check_job_results()
        elif options.collect:
            if options.fireball:
                self.collect_fireball_results(superphotos=options.superphotos)
            elif self.fit_only:
                self.collect_fit_results()
            else:
                self.collect_job_results()
        else:
            raise Exception("No valid options specified, set either -s -r -c")
