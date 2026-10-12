# NCPLS.jl

| **Documentation**                                                               | **Build Status**                                                                                |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][GHA-img]][GHA-url] [![][codecov-img]][codecov-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://oniehuis.github.io/NCPLS.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://oniehuis.github.io/NCPLS.jl/dev

[GHA-img]: https://github.com/oniehuis/NCPLS.jl/actions/workflows/CI.yml/badge.svg?branch=main
[GHA-url]: https://github.com/oniehuis/NCPLS.jl/actions/workflows/CI.yml?query=branch%3Amain

[codecov-img]: https://codecov.io/gh/oniehuis/NCPLS.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/oniehuis/NCPLS.jl

A Julia package for N-way Canonical Partial Least Squares (N-CPLS) regression and
discriminant analysis of multiway data. Model tensor-valued predictors, such as
sample-by-retention-by-m/z arrays, and inspect latent scores, predictions, and
loading weights along the original predictor axes.

## Features

- Fit regression, classification, or hybrid models that combine class indicators
  with continuous responses.
- Use multilinear loading weights with one vector per predictor mode, optionally
  orthogonalize mode weights across components, or compare with unfolded fits.
- Control centering and scaling, supply observation weights, and include auxiliary
  responses (`Yadd`) that guide component extraction.
- Select component counts and assess predictions with nested cross-validation and
  permutation tests for regression and discriminant analysis.
- Plot latent scores with Makie or PlotlyJS, and explore coefficient landscapes,
  loading-weight landscapes, and multilinear weight profiles.

## Installation

Julia 1.9 or later is required. Install the development version from GitHub in
your Julia environment:

```julia
using Pkg
Pkg.add(url="https://github.com/oniehuis/NCPLS.jl", rev="main")
```

Then load the package with `using NCPLS`.

## Quickstart

Run this in the Julia REPL. The example uses NCPLS's synthetic data generator, so
no measurement files are needed.

```julia
using NCPLS

# Generate 100 samples with two predictor modes (retention and m/z).
data = synthetic_multilinear_hybrid_data(
    nmajor=60,
    nminor=40,
    mode_dims=(40, 30),
    integer_counts=false,
)

# Fit a two-component discriminant model from categorical class labels.
model = NCPLSModel(ncomponents=2, multilinear=true)
mf = fit(
    model,
    data.X,
    categorical(data.sampleclasses_string);
    Yadd=data.Yadd,
    obs_weights=data.obs_weights,
    samplelabels=data.samplelabels,
    predictoraxes=data.predictoraxes,
)

# Inspect latent scores and classify the example samples using two components.
scores = xscores(mf)
predicted_classes = predictclasses(mf, data.X, 2)
```

`data.X` has size `(100, 40, 30)`: samples occupy the first dimension, followed by
the predictor modes. `scores` has one row per sample and one column per component.
`Yadd` and `obs_weights` are optional fitting inputs; only predictor data are
needed for prediction. These predictions use the training samples; see the
[cross-validation guide](docs/src/NCPLS/crossvalidation.md) for component selection
and evaluation on held-out samples.

### Fit continuous responses

Pass a numeric response vector or matrix instead of categorical labels to fit a
regression model. Continue with the synthetic data from the quickstart:

```julia
mf_reg = fit(model, data.X, data.Yprim_reg; predictoraxes=data.predictoraxes)
predictions = predict(mf_reg, data.X, 2)
predicted_responses = predictions[:, 2, :]
```

`predict` returns an array with dimensions `(samples, components, responses)`.
The slice `[:, 2, :]` contains cumulative predictions using the first two
components. Use `project(mf, Xnew)` to obtain latent scores for new samples, or
`predictclasses(mf, Xnew, 2)` to classify them with the discriminant model above.

### Plot latent scores

Install a Makie backend in the same environment to use NCPLS's Makie score plots:

```julia
using Pkg
Pkg.add("CairoMakie")
```

Continue with the discriminant model from the quickstart:

```julia
using CairoMakie

fig = scoreplot(mf; backend=:makie, figure_kwargs=(; size=(900, 600)))
save("ncpls_scores.png", fig)
```

The figure is saved in the current working directory. Score plots also support
PlotlyJS; the dedicated landscape and weight-profile plotting helpers use
PlotlyJS. See the [visualization guide](docs/src/NCPLS/visualization.md) for both
backends and examples of predictor-surface plots.

### Use your own data

Replace `data.X` with a numeric array shaped `(samples, mode1, mode2, ...)` and
supply responses in the same sample order. Use `categorical(labels)` for
classification, a numeric vector for one continuous response, or a numeric matrix
with samples as rows for multiple responses. The model centers predictors and
primary responses by default; set `scale_X=true` in `NCPLSModel` to also scale
predictors. New samples must use the same predictor dimensions and ordering as
the training data.

See the [fitting guide](docs/src/NCPLS/fit.md) for hybrid response blocks, predictor
axis metadata, observation weights, and auxiliary responses. For matrix-valued
predictors, see the related [CPPLS.jl](https://github.com/oniehuis/CPPLS.jl) package.

## Documentation

See the [documentation](https://oniehuis.github.io/NCPLS.jl/dev/) for the full API
and worked examples, including [N-CPLS theory](docs/src/NCPLS/theory.md),
[model fitting](docs/src/NCPLS/fit.md),
[projection and prediction](docs/src/NCPLS/predict.md),
[cross-validation and permutation testing](docs/src/NCPLS/crossvalidation.md), and
[visualization](docs/src/NCPLS/visualization.md).
