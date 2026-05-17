"""Re-render figures from a finished UltraNest run without re-sampling.

Loads the chain dump (UltraNest's ``log_dir``) and the originating pickle,
re-instantiates :class:`UltraNestFit`, replays the chain into ``fit.result``,
and re-runs the plotting helpers. Useful after tweaking
:mod:`.ultranest_plots` styling without paying the sampler cost again.

Usage::

    python examples/replot_ultranest.py \
        --pickle /path/to/prince_run_max56.pkl \
        --chain  /path/to/out/chain \
        --figdir /path/to/out/figures \
        --species 2212,1000020040,1000070140,1000140280,1000260560 \
        --nu-samples 16
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pickle", required=True)
    p.add_argument("--chain", required=True,
                   help="UltraNest log_dir (the dir that contains 'info/results.points.npy' etc.)")
    p.add_argument("--figdir", required=True)
    p.add_argument("--species", default="auto",
                   help="'auto' for the standard 5-species set (if present in pickle), "
                        "or comma-separated PDG IDs.")
    p.add_argument("--dataset", choices=("2015", "2017", "2019", "2025", "2026FD"), default="2017")
    p.add_argument("--xmax-model", default="sibyll")
    p.add_argument("--nu-samples", type=int, default=16)
    p.add_argument("--n-band", type=int, default=0,
                   help="Posterior band size on spectrum/Xmax/RMS curves (0 = MAP only). "
                        "Each band point costs one forward solve + inner fit; batched.")
    p.add_argument("--only", default="all",
                   help="Comma-separated subset of {corner,spectrum,xmax,xrms,nu}; default all.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print("Loading pickle:", args.pickle, flush=True)
    with open(args.pickle, "rb") as f:
        prince_run = pickle.load(f)

    if args.dataset == "2017":
        from prince_analysis_tools.spectra import auger2017 as spec, Xmax2017 as xm, XRMS2017 as xr
    elif args.dataset == "2015":
        from prince_analysis_tools.spectra import auger2015 as spec, Xmax2015 as xm, XRMS2015 as xr
    elif args.dataset == "2019":
        from prince_analysis_tools.spectra import auger2019 as spec, Xmax2019 as xm, XRMS2019 as xr
    elif args.dataset == "2025":
        from prince_analysis_tools.spectra import auger2025 as spec, Xmax2025 as xm, XRMS2025 as xr
    elif args.dataset == "2026FD":
        from prince_analysis_tools.spectra import auger2025 as spec
        from prince_analysis_tools.spectra import Xmax2026FD as xm, XRMS2026FD as xr

    if args.species == "auto":
        five = (2212, 1000020040, 1000070140, 1000140280, 1000260560)
        pids = tuple(p for p in five if p in prince_run.spec_man.pdgid2sref)
    else:
        pids = tuple(int(s.strip()) for s in args.species.split(","))
    print("species:", pids, flush=True)

    from prince_analysis_tools.ultranest_fit import UltraNestFit
    fit = UltraNestFit(
        prince_run=prince_run, spectrum=spec, xmax=xm, xrms=xr,
        particle_ids=pids, xmax_model=args.xmax_model,
        minimizer_args={"fix_deltaE": False},
    )

    # Reconstruct the result dict from UltraNest's hdf5 chain dump.
    import h5py
    points_npy = os.path.join(args.chain, "info", "results.json")
    pts_hdf5 = os.path.join(args.chain, "results", "points.hdf5")
    equal_npy = os.path.join(args.chain, "chains", "equal_weighted_post.txt")
    samples = None
    if os.path.exists(equal_npy):
        # Standard UltraNest dump: header line + columns
        arr = np.loadtxt(equal_npy, comments="#", skiprows=1)
        samples = arr[:, :len(fit.param_names)]
    elif os.path.exists(pts_hdf5):
        with h5py.File(pts_hdf5, "r") as fh:
            samples = fh["points"][...][:, :len(fit.param_names)]
    if samples is None:
        sys.exit("Couldn't find equal-weighted samples under {}".format(args.chain))

    # Pull logZ + theta_max from results.json if present.
    import json
    res = {"samples": samples}
    rj = os.path.join(args.chain, "info", "results.json")
    if os.path.exists(rj):
        with open(rj) as fh:
            j = json.load(fh)
        res["logz"] = j.get("logz", float("nan"))
        res["logzerr"] = j.get("logzerr", float("nan"))
        if "maximum_likelihood" in j:
            res["maximum_likelihood"] = j["maximum_likelihood"]
    else:
        res["logz"] = float("nan")
        res["logzerr"] = float("nan")
    fit.result = res

    theta_map = fit.best_fit_theta()
    print("θ_MAP =", dict(zip(fit.param_names, theta_map)), flush=True)
    print("logZ  =", res["logz"], "±", res["logzerr"], flush=True)

    os.makedirs(args.figdir, exist_ok=True)
    only = set(args.only.split(","))
    if "all" in only:
        only = {"corner", "spectrum", "xmax", "xrms", "nu"}

    from prince_analysis_tools import ultranest_plots as up
    if "corner" in only:
        up.plot_corner(fit, savepath=os.path.join(args.figdir, "fig_corner.png"))
    if "spectrum" in only:
        up.plot_bestfit_spectrum(fit, theta=theta_map, n_band=args.n_band,
                                 savepath=os.path.join(args.figdir, "fig_bestfit_spectrum.png"))
    if "xmax" in only:
        up.plot_bestfit_xmax(fit, theta=theta_map, n_band=args.n_band,
                             savepath=os.path.join(args.figdir, "fig_bestfit_xmax.png"))
    if "xrms" in only:
        up.plot_bestfit_xrms(fit, theta=theta_map, n_band=args.n_band,
                             savepath=os.path.join(args.figdir, "fig_bestfit_xrms.png"))
    if "nu" in only:
        up.plot_posterior_neutrino_band(fit, n_samples=args.nu_samples,
                                        savepath=os.path.join(args.figdir, "fig_neutrino_band.png"))
    print("figures →", args.figdir, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
