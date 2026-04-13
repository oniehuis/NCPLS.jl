# NCPLS.jl

NCPLS.jl provides a Julia implementation of the N-way Canonical Partial Least Squares 
(NCPLS) algorithm, as proposed by Liland et al. (2022), for both regression and 
discriminant analysis of tensor-valued predictors. This method extends the CPLS approach 
introduced by Indahl et al. (2009), which was originally designed for matrix-shaped 
predictors.

The matrix-based approach is implemented in the Julia package CPPLS.jl, which incorporates 
the "power" parameter extension outlined by Indahl (2009) and Liland & Indahl (2009)—hence 
the additional "P" in the acronym. Much like CPPLS, NCPLS allows users to provide auxiliary 
responses and observation weights to guide and refine the extraction of latent variables.

## Installation

The package is currently unregistered, so install it directly from GitHub:

```julia
julia> ]
pkg> add https://github.com/oniehuis/NCPLS.jl
```

Then load it with:

```julia
julia> using NCPLS
```

## Scope

Model configuration is represented by [`NCPLS.NCPLSModel`](@ref). Fitting returns an
[`NCPLS.NCPLSFit`](@ref), which stores regression coefficients, projection tensors,
scores, loadings, residuals, and preprocessing statistics needed for prediction and
inspection.

Primary responses are supplied through `Yprim`. Optional additional responses `Yadd`
contribute to loading-weight estimation but are not themselves predicted.

## Manual

- Theory: [`NCPLS/theory.md`](NCPLS/theory.md)
- Types and fitted-model accessors: [`NCPLS/types.md`](NCPLS/types.md)
- Fitting: [`NCPLS/fit.md`](NCPLS/fit.md)
- Projection and prediction: [`NCPLS/predict.md`](NCPLS/predict.md)

## Disclaimer

NCPLS is provided "as is," without warranty of any kind. Users are responsible for
independently validating all outputs and interpretations and for determining suitability
for their specific applications. The authors and contributors disclaim any liability for
errors, omissions, or any consequences arising from use of the software, including use
in regulated, clinical, or safety-critical contexts.

## References
- Indahl UG (2005): A twist to partial least squares regression. *Journal of Chemometrics* 
  19: 32–44. https://doi.org/10.1002/cem.904.
- Indahl UG, Liland KH, Naes T (2009): Canonical partial least squares — a unified PLS 
  approach to classification and regression problems. *Journal of Chemometrics* 23: 495-504. 
  https://doi.org/10.1002/cem.1243.
- Liland KH, Indahl UG (2009): Powered partial least squares discriminant analysis. 
  *Journal of Chemometrics* 23: 7-18. https://doi.org/10.1002/cem.1186.
- Liland KH, Indahl UG, Skogholt J, Mishra P (2022): The canonical partial least squares 
  approach to analysing multiway datasets—N-CPLS. *Journal of Chemometrics* 36: e3432. 
  https://doi.org/10.1002/cem.3432.
- Smit S, van Breemen MJ, Hoefsloot HCJ, Smilde AK, Aerts JMFG, de Koster CG (2007): 
  Assessing the statistical validity of proteomics based biomarkers. *Analytica Chimica 
  Acta* 592: 210-217. https://doi.org/10.1016/j.aca.2007.04.043.