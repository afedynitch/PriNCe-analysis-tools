# Xmax systematics in the fit (`xmax_sys="cov"`)

Code: `prince_analysis_tools/xmax_sys.py` (`XmaxSystematics`), used by `UHECROptimizer` and the batched
χ² of `UltraNestFractionsFit`. Data: `Xmax2026FD` / `XRMS2026FD` with the tables in `data/auger_fd_2026/`.

## ⟨Xmax⟩

The data release gives, per energy bin i, the upward and downward systematic of the Xmax scale σ_i^± and its
bin-to-bin correlation ρ_ij^±. The true scale is modelled as data + Σ_k a_k m_k with unit-Gaussian
nuisances a_k, where the mode vectors m_k are the eigenvectors of σ_i σ_j ρ_ij (restricted to the bins in
the fit) scaled by the square root of their eigenvalues; eigenvalues below 1e-9 of the largest are dropped
(the matrices are rounded to three decimals and not positive definite). With r = model − data,

    χ² = min_a [ (r − M a)ᵀ D⁻¹ (r − M a) + aᵀa ] = rᵀ (D + M Mᵀ)⁻¹ r,
    a* = (1 + Mᵀ D⁻¹ M)⁻¹ Mᵀ D⁻¹ r,

D = diag(stat_i² + rest_i²). `rest` is the part of the ⟨Xmax⟩ systematic of the moments table that exceeds
the scale systematic (0.7–1.9 g/cm² in quadrature), taken as uncorrelated.

Asymmetry: the upward set (σ⁺, ρ⁺) is used when the pull of the leading mode computed with it is ≥ 0, i.e.
when the model lies above the data along the common shift; otherwise the downward set.

Above 6 EeV (9 bins) the leading mode is a common shift of 6.8–7.0 g/cm² (up) and carries 99.6 % of the
trace (96.9 % down); over all 20 bins two modes carry 99.9 %.

## σ(Xmax)

No correlation is published. Default (`sigma_sys="correlated"`): one unit-Gaussian nuisance b common to all
bins with the per-bin systematic s_i^± as vector,

    χ² = Σ_i (r_i − b s_i)² / stat_i² + b²,   b* = (Σ r_i s_i / stat_i²) / (1 + Σ s_i² / stat_i²),

side by the sign of b*. `sigma_sys="uncorrelated"` adds s_i to stat_i in quadrature, side per bin by the sign
of r_i; it brackets the unknown correlation.

## Not covered

The energy-scale uncertainty (`deltaE`) and a hadronic-model shift of the predicted ⟨Xmax⟩ (`auger_dxmax`)
are separate parameters. The common experimental shift and `auger_dxmax` are degenerate: the data constrain
their sum.
