"""Forward-model wrapper that drives PriNCe's multi-RHS ETD2 solver.

`MultiRHSWalker.compute_models_batch` packs ``K_live × N_species`` independent
propagation problems into a single :class:`MultiRHSPropagationSolverETD2`
solve, then hands back a per-livepoint list of detached
:class:`UHECRPropagationResult` objects that the existing
:class:`UHECROptimizer` consumes unchanged.

Each "live point" is one outer parameter set ``(gamma, rmax, m)``. Within a
live point, each injection species (e.g. p, He, N, Si, Fe) becomes a separate
RHS column with its own ``AugerFitSource``. The k-th column index is laid out
as ``k = j * N_species + i`` (j = live-point index, i = species index).
"""

import numpy as np


class MultiRHSWalker(object):
    """Batched forward-model wrapper backed by ``MultiRHSPropagationSolverETD2``.

    Constructor mirrors :class:`UHECRWalker` so the same data fixtures
    (spectrum, Xmax, XRMS) can be reused.
    """

    def __init__(self, prince_run, spectrum, xmax, xrms, progressbar=False):
        self.prince_run = prince_run
        self.spectrum = spectrum
        self.xmax = xmax
        self.xrms = xrms
        self.progressbar = progressbar

    def compute_models_batch(
        self,
        particle_ids,
        batch_params,
        sclass="auger",
        initial_z=1.0,
        final_z=0.0,
        max_step=1e-2,
        enable_partial_diff_jacobian=True,
    ):
        """Propagate K outer parameter sets × N_species species in one solve.

        Parameters
        ----------
        particle_ids : sequence of int
            Injection species PDG-like ids (e.g. [101, 402, 1407, 2814, 5626]).
        batch_params : sequence of dict
            One dict per live point, with keys ``gamma``, ``rmax``, ``m``
            (and optionally ``rscale`` for ``sclass="rflex"``). ``m`` is
            passed through unchanged to the source class — typically the
            ``("simple", value)`` tuple used by the existing grid scan.
        sclass : {"auger", "simple", "rflex"}
            Source class selector; matches :meth:`UHECRWalker.compute_models`.
        initial_z, final_z, max_step
            Forwarded to the solver. Defaults match the grid scan.
        enable_partial_diff_jacobian : bool
            Forwarded to the solver constructor.

        Returns
        -------
        list[list[UHECRPropagationResult]]
            Outer list has length ``len(batch_params)``. Each entry is a list
            of ``len(particle_ids)`` detached results (state arrays copied out
            of the multi-RHS column views so the caller can safely re-solve).
        """
        from prince_cr.solvers import (
            MultiRHSPropagationSolverETD2,
            UHECRPropagationResult,
        )
        from prince_cr.cr_sources import (
            AugerFitSource,
            SimpleSource,
            RigidityFlexSource,
        )

        N_species = len(particle_ids)
        K_live = len(batch_params)
        K_total = K_live * N_species

        solver = MultiRHSPropagationSolverETD2(
            initial_z=initial_z,
            final_z=final_z,
            prince_run=self.prince_run,
            K=K_total,
            enable_partial_diff_jacobian=enable_partial_diff_jacobian,
        )

        for j, p in enumerate(batch_params):
            gamma = p["gamma"]
            rmax = p["rmax"]
            m = p["m"]
            for i, ncoid in enumerate(particle_ids):
                k = j * N_species + i
                if sclass == "auger":
                    params = {ncoid: (gamma, rmax, 1.0)}
                    source = AugerFitSource(
                        self.prince_run, params=params, m=m, norm=1.0
                    )
                elif sclass == "simple":
                    params = {ncoid: (gamma, rmax, 1.0)}
                    source = SimpleSource(
                        self.prince_run, params=params, m=m, norm=1.0
                    )
                elif sclass == "rflex":
                    rscale = p.get("rscale", 1.0)
                    params = {ncoid: (gamma, rmax, rscale, 1.0)}
                    source = RigidityFlexSource(
                        self.prince_run, params=params, m=m, norm=1.0
                    )
                else:
                    raise ValueError("Unknown source class: {!r}".format(sclass))
                solver[k].add_source_class(source)

        solver.solve(dz=max_step, verbose=False, progressbar=self.progressbar)

        egrid = solver.egrid
        spec_man = solver.spec_man
        results_per_live = []
        for j in range(K_live):
            per_live = []
            for i in range(N_species):
                k = j * N_species + i
                # Detach: copy the column view's state into a fresh result so
                # the optimizer can be called from the caller after the solver
                # falls out of scope or gets re-solved on the next batch.
                col_state = np.asarray(solver[k].res.state).copy()
                per_live.append(UHECRPropagationResult(col_state, egrid, spec_man))
            results_per_live.append(per_live)

        return results_per_live

    def compute_models_batch_combined(
        self,
        particle_ids,
        batch_params,
        sclass="auger",
        initial_z=1.0,
        final_z=0.0,
        max_step=1e-2,
        enable_partial_diff_jacobian=True,
    ):
        """Multi-species single-RHS variant of :meth:`compute_models_batch`.

        Each live point becomes ONE RHS with all ``N_species`` injection
        species combined into a single source class, using per-species
        ``fractions`` as their relative injection norms. The propagation is
        bit-identical to the per-species path (linear transport: the
        combined result equals the fraction-weighted sum of the per-species
        results), but uses ``N_species`` × fewer RHSs.

        Compared to :meth:`compute_models_batch`:
            K_RHS = K_live           (was K_live × N_species)
            output = list[result]    (was list[list[result]])

        Parameters
        ----------
        particle_ids : sequence of int
            Injection species PDG-like ids.
        batch_params : sequence of dict
            One dict per live point, with keys ``gamma``, ``rmax``, ``m``,
            and ``fractions`` (length ``N_species``, in the same order as
            ``particle_ids``; should sum to 1 but the source class doesn't
            enforce it). Optional ``rscale`` for ``sclass="rflex"``.
        sclass : {"auger", "simple", "rflex"}
            Source class selector; matches :meth:`compute_models_batch`.
        initial_z, final_z, max_step
            Forwarded to the solver.
        enable_partial_diff_jacobian : bool
            Forwarded to the solver constructor.

        Returns
        -------
        list[UHECRPropagationResult]
            One detached result per live point, representing the
            propagated CR ensemble injected as the species mixture.
        """
        from prince_cr.solvers import (
            MultiRHSPropagationSolverETD2,
            UHECRPropagationResult,
        )
        from prince_cr.cr_sources import (
            AugerFitSource,
            SechCutoffSource,
            SimpleSource,
            RigidityFlexSource,
        )

        N_species = len(particle_ids)
        K_live = len(batch_params)

        solver = MultiRHSPropagationSolverETD2(
            initial_z=initial_z,
            final_z=final_z,
            prince_run=self.prince_run,
            K=K_live,
            enable_partial_diff_jacobian=enable_partial_diff_jacobian,
        )

        for j, p in enumerate(batch_params):
            gamma = p["gamma"]
            rmax = p["rmax"]
            m = p["m"]
            fractions = p["fractions"]
            if len(fractions) != N_species:
                raise ValueError(
                    "batch_params[{}]['fractions'] has len {}, expected {}".format(
                        j, len(fractions), N_species
                    )
                )
            # Multi-isotope params dict: per-species (gamma, rmax, relative_norm).
            # The source class linearly combines the contributions at injection.
            params = {
                int(ncoid): (gamma, rmax, float(fractions[i]))
                for i, ncoid in enumerate(particle_ids)
            }
            if sclass == "auger":
                source = AugerFitSource(
                    self.prince_run, params=params, m=m, norm=1.0
                )
            elif sclass == "auger-sech":
                # Same (gamma, rmax, norm) parameter schema as AugerFitSource
                # but with the smooth sech[(E/E_cut)²] cutoff from
                # Comisso+ 2024 (functional form only — Rcut and gamma stay
                # free fit parameters).
                source = SechCutoffSource(
                    self.prince_run, params=params, m=m, norm=1.0
                )
            elif sclass == "simple":
                source = SimpleSource(
                    self.prince_run, params=params, m=m, norm=1.0
                )
            elif sclass == "rflex":
                rscale = p.get("rscale", 1.0)
                # rflex needs a 4-tuple; rebuild params with rscale.
                params = {
                    int(ncoid): (gamma, rmax, rscale, float(fractions[i]))
                    for i, ncoid in enumerate(particle_ids)
                }
                source = RigidityFlexSource(
                    self.prince_run, params=params, m=m, norm=1.0
                )
            else:
                raise ValueError("Unknown source class: {!r}".format(sclass))
            solver[j].add_source_class(source)

        solver.solve(dz=max_step, verbose=False, progressbar=self.progressbar)

        egrid = solver.egrid
        spec_man = solver.spec_man
        results = []
        for j in range(K_live):
            col_state = np.asarray(solver[j].res.state).copy()
            results.append(UHECRPropagationResult(col_state, egrid, spec_man))
        return results
