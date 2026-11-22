"""Forward model from a precomputed response table (Green's functions) of the linear CR transport.

Without secondaries the propagated state is linear in the injection. With a factorised source
``q_s(E) * evo(z)`` the state at ``final_z`` is

    state = sum_{s, i, k} q_{s,i} * evo(z_k) * G[(s, i, k), :]

where ``q_{s,i}`` is the bin-averaged injection of species ``s`` in bin ``i`` and ``G[(s, i, k)]`` is
the state produced by a unit injection in that bin with the evolution replaced by the hat function of
node ``z_k`` (nodes uniform in ln(1+z)). The only approximation is the piecewise-linear interpolation
of ``evo(z)`` between nodes. :class:`GreensWalker` builds ``G`` once with the multi-RHS solver and then
evaluates any (gamma, R_max, m, fractions) by one matrix product; it has the interface of
:meth:`MultiRHSWalker.compute_models_batch_combined`. Derivation and accuracy: docs/greens_forward.md.
"""

import numpy as np

from .multirhs_walker import _multirhs_solver


def _hat(x, nodes, k):
    """Hat function of node ``k`` on the sorted ``nodes`` at ``x``."""
    y = 0.0
    if k > 0 and nodes[k - 1] <= x <= nodes[k]:
        y = (x - nodes[k - 1]) / (nodes[k] - nodes[k - 1])
    if k < len(nodes) - 1 and nodes[k] <= x <= nodes[k + 1]:
        y = (nodes[k + 1] - x) / (nodes[k + 1] - nodes[k])
    if x == nodes[k]:
        y = 1.0
    return y


def _unit_source_class():
    """CosmicRaySource subclass: unit injection in one state, hat-function evolution."""
    from prince_ultra.cr_sources import CosmicRaySource

    class _UnitHatSource(CosmicRaySource):
        def __init__(self, prince_run, state_index, xnodes, k):
            self._state_index = int(state_index)
            self._xnodes = xnodes
            self._k = int(k)
            # distinct key per node so the solver's injection plan evaluates each node once per step
            super().__init__(prince_run, params={}, norm=1.0, m=("hat", int(k)))

        def _compute_injection_grid(self):
            self.injection_grid = np.zeros(self.dim_states)
            self.injection_grid[self._state_index] = 1.0

        def injection_spectrum(self, pid, energy, params):
            raise NotImplementedError

        def evolution(self, z):
            return _hat(np.log1p(z), self._xnodes, self._k)

    return _UnitHatSource


def _sech_binavg(A, Z, E_lo, E_hi, gamma, rmax, n_quad=8):
    """Bin averages of the ``SechCutoffSource`` spectrum (unit relnorm), shape (K, n_bins).

    Same Gauss-Legendre rule in ln E as ``CosmicRaySource._binned_spectrum``.
    """
    x, w = np.polynomial.legendre.leggauss(n_quad)
    lo, hi = np.log(E_lo), np.log(E_hi)
    half = 0.5 * (hi - lo)
    E = np.exp(0.5 * (hi + lo)[:, None] + half[:, None] * x[None, :])         # (n_bins, n_quad)
    e_k = A * E
    g = np.asarray(gamma, dtype=float)[:, None, None]
    emax = (np.asarray(rmax, dtype=float) * Z)[:, None, None]
    r2 = (e_k[None] / emax) ** 2
    with np.errstate(over="ignore"):
        cut = 1.0 / np.cosh(r2)
    cut = np.where(np.isfinite(cut), cut, 0.0)
    q = A * np.exp(-g * np.log(e_k[None] / 1e9)) * cut                         # (K, n_bins, n_quad)
    integral = half[None] * (q * E[None] * w[None, None, :]).sum(axis=2)
    return integral / (E_hi - E_lo)[None]


class GreensWalker(object):
    """Response-table forward model with the interface of ``MultiRHSWalker.compute_models_batch_combined``.

    Parameters
    ----------
    prince_run : PriNCeRun
        The run (grid, species, photon fields) the table is built for.
    particle_ids : sequence of int
        Injected species.
    initial_z, final_z : float
        Redshift range of the transport.
    n_znodes : int
        Number of hat-function nodes, uniform in ln(1+z) on [final_z, initial_z].
    e_min_total : float
        Lowest injected total energy [GeV] kept in the basis; injection below it cannot reach the
        fitted energies.
    rmax_max : float
        Largest cutoff rigidity the table must cover [GV]; bins above 10 * Z/A * rmax_max per nucleon
        are dropped (the sech cutoff is below 1e-40 there).
    """

    def __init__(self, prince_run, particle_ids, initial_z=1.0, final_z=0.0, n_znodes=41, e_min_total=1e9,
                 rmax_max=10 ** 11.5):
        self.prince_run = prince_run
        self.particle_ids = [int(p) for p in particle_ids]
        self.initial_z, self.final_z = float(initial_z), float(final_z)
        self.xnodes = np.linspace(np.log1p(self.final_z), np.log1p(self.initial_z), int(n_znodes))
        self.znodes = np.expm1(self.xnodes)
        grid, bins = prince_run.cr_grid.grid, prince_run.cr_grid.bins
        self.species = []
        for pid in self.particle_ids:
            s = prince_run.spec_man.pdgid2sref[pid]
            sel = np.flatnonzero((grid * s.A >= e_min_total) & (grid <= 10.0 * s.Z / s.A * rmax_max))
            self.species.append(dict(pid=pid, A=float(s.A), Z=float(s.Z), sel=sel,
                                     states=s.sl.start + sel, E_lo=bins[sel], E_hi=bins[sel + 1]))
        # basis order: species, energy bin, z node (node fastest)
        self.basis = [(si, st, k) for si, sp in enumerate(self.species) for st in sp["states"]
                      for k in range(len(self.xnodes))]
        self.G = None
        self.meta = {}

    @property
    def n_basis(self):
        return len(self.basis)

    def build(self, max_step, solver="etd2", chunk=2000, progress=print):
        """Solve the unit responses; fills ``self.G``, shape (n_basis, dim_states)."""
        import time

        Src = _unit_source_class()
        G = np.empty((self.n_basis, self.prince_run.dim_states))
        t0 = time.time()
        for c0 in range(0, self.n_basis, chunk):
            cols = self.basis[c0:c0 + chunk]
            sv = _multirhs_solver(solver, initial_z=self.initial_z, final_z=self.final_z,
                                  prince_run=self.prince_run, K=len(cols), enable_partial_diff_jacobian=True)
            for j, (_, st, k) in enumerate(cols):
                sv[j].add_source_class(Src(self.prince_run, st, self.xnodes, k))
            sv.solve(dz=max_step, verbose=False, progressbar=False)
            G[c0:c0 + len(cols)] = np.asarray(sv.state).T
            if progress:
                progress(f"[greens] {c0 + len(cols)}/{self.n_basis} columns, {time.time() - t0:.0f} s")
        self.G = G
        self.meta = dict(max_step=max_step, solver=solver, build_s=time.time() - t0)
        self._G_dev = None
        return self

    def save(self, path):
        """Write the table and its definition to ``path`` (npz)."""
        np.savez(path, G=self.G, xnodes=self.xnodes, particle_ids=np.array(self.particle_ids),
                 states=np.concatenate([sp["states"] for sp in self.species]),
                 initial_z=self.initial_z, final_z=self.final_z, dim_states=self.prince_run.dim_states,
                 **{k: v for k, v in self.meta.items()})

    def load(self, path):
        """Read a table written by :meth:`save`; checks it matches this walker's definition."""
        d = np.load(path)
        states = np.concatenate([sp["states"] for sp in self.species])
        if not (np.array_equal(d["states"], states) and np.allclose(d["xnodes"], self.xnodes)
                and int(d["dim_states"]) == self.prince_run.dim_states):
            raise ValueError(f"{path} was built for another grid, species set or z nodes")
        self.G = d["G"]
        self.meta = {k: d[k].item() for k in ("max_step", "solver", "build_s") if k in d.files}
        self._G_dev = None
        return self

    def coefficients(self, batch_params, m_kind="simple"):
        """Basis coefficients, shape (K, n_basis), for ``sech``-cutoff sources with ``(1+z)^m`` evolution."""
        K = len(batch_params)
        gamma = np.array([p["gamma"] for p in batch_params], dtype=float)
        rmax = np.array([p["rmax"] for p in batch_params], dtype=float)
        m = np.array([p["m"][1] if isinstance(p["m"], tuple) else p["m"] for p in batch_params], dtype=float)
        if m_kind != "simple":
            raise NotImplementedError("only (1+z)^m evolution ('simple') is tabulated")
        frac = np.array([p["fractions"] for p in batch_params], dtype=float)
        evo = np.exp(m[:, None] * self.xnodes[None, :])                          # (K, n_nodes) = (1+z_k)^m
        parts = []
        for si, sp in enumerate(self.species):
            q = _sech_binavg(sp["A"], sp["Z"], sp["E_lo"], sp["E_hi"], gamma, rmax) * frac[:, si:si + 1]
            parts.append((q[:, :, None] * evo[:, None, :]).reshape(K, -1))
        return np.concatenate(parts, axis=1)

    def states(self, batch_params):
        """Propagated states at ``final_z``, shape (K, dim_states), on the host."""
        C = self.coefficients(batch_params)
        if self.prince_run.backend.linear_algebra_backend.lower() != "cupy":
            return C @ self.G
        import cupy as cp

        if self._G_dev is None:
            self._G_dev = cp.asarray(self.G)
        return cp.asnumpy(cp.asarray(C) @ self._G_dev)

    def compute_models_batch_combined(self, particle_ids, batch_params, sclass="auger-sech", initial_z=None,
                                      final_z=None, max_step=None, **_):
        """Drop-in for ``MultiRHSWalker.compute_models_batch_combined`` (``sclass="auger-sech"`` only)."""
        from prince_ultra.solvers import UHECRPropagationResult

        if [int(p) for p in particle_ids] != self.particle_ids:
            raise ValueError("particle_ids differ from the table's")
        if sclass != "auger-sech":
            raise NotImplementedError("the table is evaluated for the sech-cutoff source only")
        for name, v, ref in (("initial_z", initial_z, self.initial_z), ("final_z", final_z, self.final_z)):
            if v is not None and abs(v - ref) > 1e-12:
                raise ValueError(f"{name}={v} differs from the table's {ref}")
        S = self.states(batch_params)
        egrid, spec_man = self.prince_run.cr_grid.grid, self.prince_run.spec_man
        return [UHECRPropagationResult(S[j], egrid, spec_man) for j in range(len(batch_params))]
