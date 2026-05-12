"""End-to-end UltraNest fit for the 3D source-parameter space.

Mirrors ``examples/evaluate.ipynb`` but replaces the grid scan + per-cell
local fit with nested sampling driven by
:class:`prince_analysis_tools.ultranest_fit.UltraNestFit`. The same chi²
definition, the same Auger 2017 data fixtures and the same iMinuit-based
inner fit are reused — only the *outer* parameter exploration changes.

Run as::

    python examples/run_ultranest.py \
        --prince-pickle /path/to/prince_run_max24.pkl \
        --log-dir ./ultranest_out \
        --min-live 400 --dlogz 0.5

Pickle path defaults to the file referenced by the existing grid scan;
override it for your environment.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prince-pickle",
        required=True,
        help="Path to a pre-built PriNCeRun pickle (matches what the grid scan loads).",
    )
    p.add_argument(
        "--log-dir",
        default="./ultranest_out",
        help="UltraNest output directory (chains, run state, plots).",
    )
    p.add_argument(
        "--resume",
        default="resume",
        choices=("resume", "overwrite", "resume-similar", "subfolder"),
        help="UltraNest resume mode.",
    )
    p.add_argument("--min-live", type=int, default=400, help="min_num_live_points")
    p.add_argument("--dlogz", type=float, default=0.5, help="dlogz convergence target")
    p.add_argument(
        "--max-ncalls", type=int, default=None, help="hard cap on likelihood calls"
    )
    p.add_argument(
        "--dataset",
        choices=("2015", "2017", "2019"),
        default="2017",
        help="Auger data vintage (matches recompute_fit.py).",
    )
    p.add_argument(
        "--xmax-model",
        choices=("epos", "sibyll", "qgsjet"),
        default="sibyll",
        help="Hadronic interaction model for Xmax/RMS.",
    )
    p.add_argument(
        "--apply-auger-xmax-shift",
        action="store_true",
        help=(
            "Float a per-HIM ΔXmax nuisance with the Auger-2024 prior "
            "(PRD 109, 102001 Table III) on the predicted ⟨Xmax⟩ — "
            "model-side shift only, σ(Xmax) untouched."
        ),
    )
    p.add_argument(
        "--n-nu-samples",
        type=int,
        default=64,
        help="Number of posterior draws for the neutrino-flux band.",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip the plotting pass (run fit only).",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print("Loading PriNCe pickle: {}".format(args.prince_pickle), flush=True)
    with open(args.prince_pickle, "rb") as f:
        prince_run = pickle.load(f)
    print("  dim_states = {}, max_mass = {}".format(
        prince_run.dim_states, max(s.A for s in prince_run.spec_man.species_refs)
    ), flush=True)

    if args.dataset == "2017":
        from prince_analysis_tools.spectra import auger2017 as spec
        from prince_analysis_tools.spectra import Xmax2017 as xm
        from prince_analysis_tools.spectra import XRMS2017 as xr
    elif args.dataset == "2015":
        from prince_analysis_tools.spectra import auger2015 as spec
        from prince_analysis_tools.spectra import Xmax2015 as xm
        from prince_analysis_tools.spectra import XRMS2015 as xr
    elif args.dataset == "2019":
        from prince_analysis_tools.spectra import auger2019 as spec
        from prince_analysis_tools.spectra import Xmax2019 as xm
        from prince_analysis_tools.spectra import XRMS2019 as xr
    else:
        raise ValueError(args.dataset)

    from prince_analysis_tools.ultranest_fit import UltraNestFit

    fit = UltraNestFit(
        prince_run=prince_run,
        spectrum=spec,
        xmax=xm,
        xrms=xr,
        xmax_model=args.xmax_model,
        apply_auger_xmax_shift=args.apply_auger_xmax_shift,
        minimizer_args={"fix_deltaE": False},
    )

    print("Running UltraNest (log_dir={})".format(args.log_dir), flush=True)
    fit.run(
        log_dir=args.log_dir,
        resume=args.resume,
        min_num_live_points=args.min_live,
        dlogz=args.dlogz,
        max_ncalls=args.max_ncalls,
    )

    print("\n" + fit.summary(), flush=True)

    np.savez(
        os.path.join(args.log_dir, "summary.npz"),
        param_names=np.array(fit.param_names),
        samples=np.asarray(fit.result.get("samples", fit.result["weighted_samples"]["points"])),
        logz=fit.result["logz"],
        logzerr=fit.result["logzerr"],
        best_fit_theta=fit.best_fit_theta(),
    )

    if args.no_plots:
        return 0

    print("Producing figures...", flush=True)
    from prince_analysis_tools import ultranest_plots as up

    figdir = os.path.join(args.log_dir, "figures")
    os.makedirs(figdir, exist_ok=True)

    up.plot_corner(fit, savepath=os.path.join(figdir, "fig_corner.png"))
    theta_map = fit.best_fit_theta()
    print("  θ_MAP = {}".format(dict(zip(fit.param_names, theta_map))), flush=True)
    up.plot_bestfit_spectrum(
        fit, theta=theta_map, savepath=os.path.join(figdir, "fig_bestfit_spectrum.png")
    )
    up.plot_bestfit_xmax(
        fit, theta=theta_map, savepath=os.path.join(figdir, "fig_bestfit_xmax.png")
    )
    up.plot_bestfit_xrms(
        fit, theta=theta_map, savepath=os.path.join(figdir, "fig_bestfit_xrms.png")
    )
    up.plot_posterior_neutrino_band(
        fit,
        n_samples=args.n_nu_samples,
        savepath=os.path.join(figdir, "fig_neutrino_band.png"),
    )
    print("Figures in {}".format(figdir), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
