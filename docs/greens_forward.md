# Response-table forward model (`GreensWalker`)

Without secondaries the transport is a linear ODE in the state, `dY/dz = L(z) Y + Q(z)`, and the sources
used in the fits factorise, `Q(z) = evo(z) q`. For a fixed integrator and step size the state at `z = 0` is
therefore a linear function of the injection vector `q` and of the values of `evo` at the solver's steps.

`GreensWalker` expands the evolution in hat functions `h_k` with nodes uniform in `x = ln(1+z)`:

    evo(z) ≈ Σ_k evo(z_k) h_k(z)

and solves once, with the multi-RHS solver, the unit responses `G[(s, i, k)]`: unit injection of species `s`
in energy bin `i` with evolution `h_k`. A model is then

    state = Σ_{s,i,k} f_s q_{s,i}(γ, R_max) evo(z_k) G[(s, i, k)]

one matrix product, with `q_{s,i}` the bin average of the sech-cutoff spectrum (same Gauss-Legendre rule as
`CosmicRaySource._binned_spectrum`). Bins whose total injected energy is below `e_min_total` cannot reach the
fitted energies (photo-disintegration keeps the energy per nucleon, losses lower it) and bins above
`10 Z/A rmax_max` per nucleon carry less than 1e-40 of the cutoff; both are left out.

Accuracy: the table reproduces the direct solve with the same integrator and step size, except for the
linear interpolation of `evo` between nodes. For `(1+z)^m` the relative error of the interpolant is at most
`m² h² / 8` with `h = ln(1+z_max) / (n_nodes − 1)`: 1.3e-3 for |m| = 6 and 41 nodes on z ≤ 1, and its
effect on the flux is smaller (`tests/test_greens.py`: within 1e-3 at m = 4). Step-size errors of the
integrator are the same as in the direct solve, so the table is built at the step size the fit needs.

Cost: (number of kept bins) × (nodes) right-hand sides once; for the 5-species v8 fit at 16 bins/decade
339 bins × 41 nodes = 13 899 columns. The evaluation is a (K × n_basis) · (n_basis × dim_states) product
on the GPU plus the existing batched χ².
