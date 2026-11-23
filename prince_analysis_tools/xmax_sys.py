"""chi2 of the Xmax moments with correlated systematic uncertainties.

The Xmax-scale systematic of <Xmax> enters with its published bin-to-bin correlation, the
systematic of sigma(Xmax) as one separate nuisance. Formulas and numbers: docs/xmax_systematics.md.
"""

import numpy as np

SIGMA_SYS = ("correlated", "uncorrelated")


class XmaxSystematics(object):
    """Profiled chi2 of <Xmax> and sigma(Xmax) residuals for one data set.

    Parameters
    ----------
    xmax, xrms : dict
        Data dicts with ``stat``, ``sys_Up``, ``sys_Low``; ``xmax`` also needs ``scale_sys_Up``,
        ``scale_sys_Low``, ``corr_Up``, ``corr_Low`` (e.g. ``spectra.Xmax2026FD``).
    sel : array of int or bool
        Bins used in the fit.
    sigma_sys : {"correlated", "uncorrelated"}
        sigma(Xmax) systematic as one nuisance common to all bins, or added per bin in quadrature.
    """

    def __init__(self, xmax, xrms, sel, sigma_sys="correlated"):
        if sigma_sys not in SIGMA_SYS:
            raise ValueError("sigma_sys must be one of {}, got {!r}".format(SIGMA_SYS, sigma_sys))
        missing = [k for k in ("scale_sys_Up", "scale_sys_Low", "corr_Up", "corr_Low") if k not in xmax]
        if missing:
            raise KeyError("Xmax data set has no systematic correlation ({})".format(", ".join(missing)))
        sel = np.arange(len(xmax["val"]))[sel]
        self.sel = sel
        self.sigma_sys = sigma_sys
        stat2 = xmax["stat"][sel] ** 2
        self._cinv, self._proj, self._modes = {}, {}, {}
        for side in ("Up", "Low"):
            scale = xmax["scale_sys_" + side][sel]
            rest2 = np.clip(xmax["sys_" + side][sel] ** 2 - scale**2, 0.0, None)
            d = stat2 + rest2
            w, v = np.linalg.eigh(scale[:, None] * scale[None, :] * xmax["corr_" + side][np.ix_(sel, sel)])
            keep = w > 1e-9 * w.max()
            m = (v[:, keep] * np.sqrt(w[keep]))[:, ::-1]
            m *= np.where(m.sum(axis=0) < 0.0, -1.0, 1.0)[None, :]
            self._modes[side] = m
            self._cinv[side] = np.linalg.inv(np.diag(d) + m @ m.T)
            mtd = m.T / d[None, :]
            self._proj[side] = np.linalg.solve(np.eye(m.shape[1]) + mtd @ m, mtd)
        self._sstat2 = xrms["stat"][sel] ** 2
        self._ssys = {side: xrms["sys_" + side][sel] for side in ("Up", "Low")}

    def _mean_side(self, delta):
        """True where the upward systematic applies (model above data along the common shift)."""
        return delta @ self._proj["Up"][0] >= 0.0

    def chi2_mean(self, delta):
        """chi2 of <Xmax> residuals ``delta`` = model - data, shape (..., n_sel)."""
        delta = np.asarray(delta, dtype=float)
        up = np.einsum("...i,ij,...j->...", delta, self._cinv["Up"], delta)
        low = np.einsum("...i,ij,...j->...", delta, self._cinv["Low"], delta)
        return np.where(self._mean_side(delta), up, low)

    def _sigma_pull(self, delta):
        """Pull of the common sigma(Xmax) nuisance and the systematic vector of its side."""
        b = {}
        for side in ("Up", "Low"):
            s = self._ssys[side]
            b[side] = (delta * s / self._sstat2).sum(axis=-1) / (1.0 + (s**2 / self._sstat2).sum())
        up = b["Up"] >= 0.0
        pull = np.where(up, b["Up"], b["Low"])
        s = np.where(np.expand_dims(up, -1), self._ssys["Up"], self._ssys["Low"])
        return pull, s

    def chi2_sigma(self, delta):
        """chi2 of sigma(Xmax) residuals ``delta`` = model - data, shape (..., n_sel)."""
        delta = np.asarray(delta, dtype=float)
        if self.sigma_sys == "uncorrelated":
            s = np.where(delta >= 0.0, self._ssys["Up"], self._ssys["Low"])
            return (delta**2 / (self._sstat2 + s**2)).sum(axis=-1)
        pull, s = self._sigma_pull(delta)
        return ((delta - pull[..., None] * s) ** 2 / self._sstat2).sum(axis=-1) + pull**2

    def pulls(self, delta_mean, delta_sigma):
        """Best-fit nuisances for one pair of residual vectors.

        Returns a dict: ``modes`` (pulls of the <Xmax> modes in sigma, leading mode first),
        ``shift`` (their sum per bin in g/cm2, to be added to the data), ``side`` and, for the
        correlated sigma(Xmax) systematic, ``sigma`` (pull in sigma).
        """
        delta_mean = np.asarray(delta_mean, dtype=float)
        side = "Up" if self._mean_side(delta_mean) else "Low"
        a = self._proj[side] @ delta_mean
        out = {"side": side, "modes": a, "shift": self._modes[side] @ a}
        if self.sigma_sys == "correlated":
            out["sigma"] = float(self._sigma_pull(np.asarray(delta_sigma, dtype=float))[0])
        return out
