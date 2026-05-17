"""Tiny end-to-end smoke test for the UltraNest pipeline.

Loads the cached PriNCe pickle, runs the forward model on a 2-livepoint batch
through :class:`MultiRHSWalker`, evaluates :meth:`UltraNestFit.loglike`, and
prints the chi² and inner-fit values. No nested sampler is invoked — this is
just a sanity check that the entire forward + inner-fit path executes.

Usage::

    python examples/smoke_test_ultranest.py [--pickle /path/to/prince_run.pkl]
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np


DEFAULT_PICKLE = (
    "/Users/anatoli/devel_mac/UH-UHECR-Fluka-Prince/runs/_cache/prince_run_max24.pkl"
)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pickle", default=DEFAULT_PICKLE)
    ap.add_argument("--max-step", type=float, default=1e-2,
                    help="dz for the ETD2 multi-RHS solver. 1e-2 is fine on ETD2.")
    ap.add_argument("--out-dir", default=None,
                    help="Persistent output directory; defaults to a tempdir (no plots saved).")
    ap.add_argument("--with-plots", action="store_true",
                    help="Produce corner / spectrum / Xmax / XRMS / neutrino-band figures.")
    ap.add_argument("--species", default="auto",
                    help="'auto' picks 5-species if pickle has Si+Fe, else 3-species; "
                         "or a comma-separated PDG list like '2212,1000020040,1000070140'.")
    ap.add_argument("--min-live", type=int, default=40,
                    help="UltraNest min_num_live_points (≥40).")
    ap.add_argument("--max-ncalls", type=int, default=400,
                    help="Hard cap on likelihood calls.")
    ap.add_argument("--dlogz", type=float, default=10.0,
                    help="Loose convergence target; lower → run longer.")
    ap.add_argument("--ndraw-min", type=int, default=None,
                    help="UltraNest ndraw_min — batch size per loglike call. "
                         "Setting this >> 1 amortises the multi-RHS overhead.")
    ap.add_argument("--ndraw-max", type=int, default=None,
                    help="UltraNest ndraw_max (default 65536).")
    ap.add_argument("--nu-samples", type=int, default=16,
                    help="Subsample size for the posterior neutrino-flux band.")
    ap.add_argument("--n-band", type=int, default=0,
                    help="Posterior band size on spectrum/Xmax/RMS curves (0 = MAP only).")
    ap.add_argument("--dataset", default="2017", choices=("2015", "2017", "2019", "2025", "2026FD"),
                    help="Auger data vintage. 2025 enables the new spectrum + Xmax/RMS "
                         "from PierreAuger:2025eun and PierreAuger:2024nzw. 2026FD pairs the "
                         "2025 combined spectrum with the new 17-year FD Xmax/σ from "
                         "PierreAuger:2026qbt (arXiv:2605.12598).")
    ap.add_argument("--use-corr", action="store_true",
                    help="Fold the spectrum bin-to-bin correlation matrix into the χ² "
                         "(only meaningful with --dataset 2025 or 2026FD).")
    args = ap.parse_args(argv)

    print("[smoke] loading pickle: {}".format(args.pickle), flush=True)
    t0 = time.time()
    with open(args.pickle, "rb") as f:
        prince_run = pickle.load(f)
    print("[smoke]   loaded in {:.1f}s, dim_states={}".format(
        time.time() - t0, prince_run.dim_states), flush=True)

    if args.dataset == "2026FD":
        # Pair the 2025 combined spectrum (PierreAuger:2025eun) with the new
        # 17-year FD Xmax/σ (PierreAuger:2026qbt) — the 2026 paper does not
        # release a spectrum dataset of its own.
        from prince_analysis_tools.spectra import auger2025 as spec_d
        from prince_analysis_tools.spectra import Xmax2026FD as xm_d
        from prince_analysis_tools.spectra import XRMS2026FD as xr_d
    elif args.dataset == "2025":
        from prince_analysis_tools.spectra import auger2025 as spec_d
        from prince_analysis_tools.spectra import Xmax2025 as xm_d
        from prince_analysis_tools.spectra import XRMS2025 as xr_d
    elif args.dataset == "2019":
        from prince_analysis_tools.spectra import auger2019 as spec_d
        from prince_analysis_tools.spectra import Xmax2019 as xm_d
        from prince_analysis_tools.spectra import XRMS2019 as xr_d
    elif args.dataset == "2015":
        from prince_analysis_tools.spectra import auger2015 as spec_d
        from prince_analysis_tools.spectra import Xmax2015 as xm_d
        from prince_analysis_tools.spectra import XRMS2015 as xr_d
    else:
        from prince_analysis_tools.spectra import auger2017 as spec_d
        from prince_analysis_tools.spectra import Xmax2017 as xm_d
        from prince_analysis_tools.spectra import XRMS2017 as xr_d
    print("[smoke] dataset = auger{}".format(args.dataset), flush=True)
    if args.use_corr:
        if "corr" not in spec_d:
            print("[smoke] WARNING: --use-corr passed but the selected spectrum has no corr key; falling back to diagonal", flush=True)
        else:
            print("[smoke] use_corr=True (folds bin-to-bin correlation into spectrum χ²)", flush=True)

    from prince_analysis_tools.ultranest_fit import UltraNestFit

    if args.species == "auto":
        five = (2212, 1000020040, 1000070140, 1000140280, 1000260560)
        have_all = all(p in prince_run.spec_man.pdgid2sref for p in five)
        if have_all:
            pids = five
        else:
            pids = (2212, 1000020040, 1000070140)
    else:
        pids = tuple(int(s.strip()) for s in args.species.split(","))
    print("[smoke] particle_ids = {}".format(pids), flush=True)

    fit = UltraNestFit(
        prince_run=prince_run,
        spectrum=spec_d,
        xmax=xm_d,
        xrms=xr_d,
        particle_ids=pids,
        xmax_model="sibyll",
        minimizer_args={"fix_deltaE": False},
        use_corr=args.use_corr,
        max_step=args.max_step,
    )

    # Two livepoints near the published best-fit region (Heinze 2019 Sibyll):
    # gamma ≈ -1, log10_rmax ≈ 9.7, m ≈ 4.
    theta_batch = np.array([
        [-1.0, 9.7, 4.0],
        [ 0.5, 10.0, 0.0],
    ])
    print("[smoke] loglike on batch of shape {}".format(theta_batch.shape), flush=True)

    t1 = time.time()
    logl = fit.loglike(theta_batch)
    dt = time.time() - t1
    print("[smoke] loglike returned in {:.2f}s: {}".format(dt, logl), flush=True)
    print("[smoke] chi²  = {}".format(-2.0 * logl), flush=True)

    if not np.all(np.isfinite(logl)):
        print("[smoke] FAIL: non-finite logl entries", file=sys.stderr)
        return 1

    # Re-run inner fit at the first point and dump details
    print("[smoke] recompute_fit at theta[0]:", theta_batch[0], flush=True)
    minres, opt = fit.recompute_fit(theta_batch[0])
    print("[smoke]   fval = {:.3f}".format(minres.fval), flush=True)
    print("[smoke]   chi²_spec = {:.3f}".format(opt.get_chi2_spectrum()), flush=True)
    print("[smoke]   chi²_Xmax = {:.3f}".format(opt.get_chi2_Xmax()), flush=True)
    print("[smoke]   chi²_VarXmax = {:.3f}".format(opt.get_chi2_VarXmax()), flush=True)
    for name, val in minres.values.to_dict().items() if hasattr(minres.values, "to_dict") else []:
        print("[smoke]     {} = {:.3e}".format(name, val), flush=True)
    # Tiny UltraNest live run — exercises the sampler glue; not enough to converge.
    try:
        import ultranest  # noqa
    except ImportError:
        print("[smoke] ultranest not installed — skipping sampler glue test", flush=True)
        print("[smoke] OK (loglike only)", flush=True)
        return 0

    import tempfile
    out_ctx = (
        tempfile.TemporaryDirectory()
        if args.out_dir is None
        else _PersistDir(args.out_dir)
    )

    with out_ctx as out_dir:
        log_dir = os.path.join(out_dir, "chain")
        os.makedirs(log_dir, exist_ok=True)
        print("[smoke] UltraNest run (live={}, max_ncalls={}, dlogz={}, ndraw_min={}) in {}".format(
            args.min_live, args.max_ncalls, args.dlogz, args.ndraw_min, log_dir), flush=True)
        sampler_kwargs = {}
        if args.ndraw_min is not None:
            sampler_kwargs["ndraw_min"] = int(args.ndraw_min)
        if args.ndraw_max is not None:
            sampler_kwargs["ndraw_max"] = int(args.ndraw_max)
        t2 = time.time()
        result = fit.run(
            log_dir=log_dir,
            resume="overwrite",
            min_num_live_points=args.min_live,
            dlogz=args.dlogz,
            max_ncalls=args.max_ncalls,
            show_status=False,
            sampler_kwargs=sampler_kwargs,
        )
        dt = time.time() - t2
        print("[smoke] sampler returned in {:.1f}s".format(dt), flush=True)
        print("[smoke] logZ = {:.3f} ± {:.3f}".format(result["logz"], result["logzerr"]), flush=True)
        if "samples" in result:
            print("[smoke] equal-weight samples shape: {}".format(
                np.asarray(result["samples"]).shape), flush=True)
        theta_map = fit.best_fit_theta()
        print("[smoke] θ_MAP = {}".format(
            dict(zip(fit.param_names, theta_map))), flush=True)

        if args.out_dir is not None:
            np.savez(
                os.path.join(out_dir, "summary.npz"),
                param_names=np.array(fit.param_names),
                samples=np.asarray(result.get("samples",
                    result["weighted_samples"]["points"])),
                logz=result["logz"],
                logzerr=result["logzerr"],
                best_fit_theta=theta_map,
                particle_ids=np.asarray(pids),
            )
            print("[smoke] summary saved to {}/summary.npz".format(out_dir), flush=True)

        if args.with_plots:
            print("[smoke] producing figures...", flush=True)
            from prince_analysis_tools import ultranest_plots as up
            figdir = os.path.join(out_dir, "figures")
            os.makedirs(figdir, exist_ok=True)
            up.plot_corner(
                fit, savepath=os.path.join(figdir, "fig_corner.png"))
            up.plot_bestfit_spectrum(
                fit, theta=theta_map, n_band=args.n_band,
                savepath=os.path.join(figdir, "fig_bestfit_spectrum.png"))
            up.plot_bestfit_xmax(
                fit, theta=theta_map, n_band=args.n_band,
                savepath=os.path.join(figdir, "fig_bestfit_xmax.png"))
            up.plot_bestfit_xrms(
                fit, theta=theta_map, n_band=args.n_band,
                savepath=os.path.join(figdir, "fig_bestfit_xrms.png"))
            up.plot_posterior_neutrino_band(
                fit, n_samples=args.nu_samples,
                savepath=os.path.join(figdir, "fig_neutrino_band.png"))
            print("[smoke] figures saved to {}".format(figdir), flush=True)
    print("[smoke] OK", flush=True)
    return 0


class _PersistDir:
    """Tiny duck-typed context manager that yields a persistent directory.

    Mirrors ``tempfile.TemporaryDirectory`` so the ``with`` block in main()
    can switch between persistent and temp paths transparently.
    """

    def __init__(self, path):
        self.path = path
        os.makedirs(path, exist_ok=True)

    def __enter__(self):
        return self.path

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    sys.exit(main())
