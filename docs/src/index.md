# NCPLS.jl

NCPLS.jl provides a Julia implementation of the N-way Canonical Partial Least Squares 
(NCPLS) algorithm, as proposed by Liland et al. (2022), for both regression and 
discriminant analysis of tensor-valued predictors. This method extends the CPLS approach 
introduced by Indahl et al. (2009), which was originally designed for matrix-shaped 
predictors. The matrix-based approach is implemented in the Julia package
[CPPLS.jl](https://github.com/oniehuis/CPPLS.jl), which incorporates the "power" parameter 
extension outlined by Indahl (2009) and Liland & Indahl (2009)—hence the additional "P" in 
the acronym.

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

The package supports supervised latent-variable modelling with numeric response matrices,
categorical class labels, and hybrid response blocks that combine class-indicator columns
with continuous traits. Fitting is controlled through NCPLSModel, which separates model
configuration from the data passed to fit. The resulting fitted models provide
prediction, class assignment where a class block is defined, latent projections,
regression coefficients, fitted values, residuals, and labels or metadata retained from
model fitting.

NCPLS also includes helper functionality for common preprocessing and encoding tasks, 
including one-hot encoding of class labels and utility functions used in chemometric and 
multivariate workflows. In addition, the package provides dedicated cross-validation and
permutation-testing routines for regression and discriminant analysis, together with
visualization helpers for score plots, coefficient landscapes, loading-weight landscapes,
and multilinear weight profiles. Fitting can also include optional additional responses
through `Yadd`, which influence component extraction but are not themselves prediction
targets, and optional observation weights for weighted preprocessing and supervised
fitting. The dedicated discriminant-analysis cross-validation helpers target categorical
labels or pure one-hot class-indicator matrices; hybrid response workflows use the more
general nested cross-validation functions with custom callbacks.

## Why Separate from CPPLS.jl?

NCPLS.jl is kept separate from CPPLS.jl even though the two packages share much of the
same modelling philosophy. CPPLS.jl targets matrix-valued predictors, whereas NCPLS.jl
targets genuinely multiway predictors and therefore needs different tensor
representations, multilinear loading-weight routines, and visualization workflows.
Keeping the packages separate makes the implementation easier to maintain and the
user-facing API easier to understand.

## Related Packages

- [CPPLS.jl](https://github.com/oniehuis/CPPLS.jl): closely related CPLS implementation
  for matrix-valued predictors rather than multiway predictors.

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
