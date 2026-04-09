# NCPLS.jl

NCPLS.jl provides a Julia implementation of NCPLS for matrix- and tensor-valued
predictors. The package supports weighted preprocessing, latent-component regression,
projection of new observations, and an optional multilinear loading-weight branch based
on rank-1 PARAFAC updates.

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

NCPLS.jl is provided "as is", without warranty of any kind. Users remain responsible for
validating outputs and for determining suitability in their own analytical workflows.
