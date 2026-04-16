# Types

`NCPLS` relies on two main container types in the [`StatsAPI.fit`](@ref) and
[`StatsAPI.predict`](@ref) workflow. [`NCPLSModel`](@ref) stores the fitting
specification: preprocessing options, the number of extracted components, and the
switches that control the multilinear loading-weight branch. [`NCPLSFit`](@ref) is the
full fitted model returned by [`fit`](@ref); it contains regression coefficients,
predictor scores, multilinear weight information, preprocessing statistics, and the
labels or axis metadata retained from model fitting.

A third container, [`NCPLSFitLight`](@ref), stores only the subset of fitted-model
information needed for fast internal prediction during cross-validation. It is mainly an
internal optimization and is usually not accessed directly.

The package provides getters for quantities that are commonly useful in downstream work.
For fitted models, the most important ones are [`coef`](@ref), [`fitted`](@ref),
[`residuals`](@ref), [`xscores`](@ref), [`predictoraxes`](@ref),
[`responselabels`](@ref), [`samplelabels`](@ref), [`sampleclasses`](@ref), [`xmean`](@ref),
[`xstd`](@ref), and [`ymean`](@ref). `NCPLSModel` does not currently have a separate
getter layer; its options are accessed directly as fields, for example
`model.multilinear`, `model.orthogonalize_mode_weights`, or `model.ncomponents`.

Two stored label vectors are easy to confuse:

- [`sampleclasses`](@ref) lives on the sample axis and stores one label per training
  sample,
- [`responselabels`](@ref) lives on the response axis and stores one label per response
  column.

These vectors interact only when NCPLS tries to infer a class-response block from a
custom response matrix. If the unique sample classes can be matched consistently to a
one-hot sub-block of the responses, classification helpers use that sub-block. If not,
`sampleclasses(mf)` remains sample-level metadata for grouping and visualization.

As with the `CPPLS` documentation, the purpose of this page is to describe the stored
containers and the most useful accessors. The modeling choices controlled through
[`fit`](@ref) are discussed on the [Fit](fit.md) page, and the downstream interpretation
of predictions is discussed on [Projection and Prediction](predict.md).

## NCPLSModel

[`NCPLSModel`](@ref) is the user-facing model specification. It stores:

- the number of latent components to extract,
- predictor preprocessing choices (`center_X`, `scale_X`),
- primary-response centering (`center_Yprim`),
- whether predictor-side weights stay unfolded or are compressed to rank-1 multilinear
  tensors (`multilinear`),
- whether multilinear mode vectors are orthogonalized across components
  (`orthogonalize_mode_weights`),
- and the PARAFAC control parameters used in the multilinear branch.

The default `NCPLSModel()` uses `multilinear=true`, so genuinely multiway predictors are
handled through the multilinear weight representation unless you request the unfolded
comparison branch explicitly.

```@docs
NCPLS.NCPLSModel
```

## PredictorAxis

[`PredictorAxis`](@ref) stores metadata for one non-sample predictor mode. It is useful
when your predictor tensor has meaningful coordinates such as time, wavelength, or `m/z`
values, and you want those coordinates retained on the fitted model for downstream plots
or inspection.

Provide one `PredictorAxis` per non-sample predictor mode through the `predictoraxes`
keyword in [`fit`](@ref). The stored tuple can then be retrieved with
[`predictoraxes`](@ref).

```@docs
NCPLS.PredictorAxis
```

## AbstractNCPLSFit

[`AbstractNCPLSFit`](@ref) is the common supertype of full and reduced fitted NCPLS
models. The generic fitted-model accessors are defined at this level.

[`coef`](@ref) returns the regression coefficients for the final or requested number of
components. These coefficients act on predictors after the fitted model's centering and
optional scaling have been applied. [`xmean`](@ref), [`xstd`](@ref), and [`ymean`](@ref)
expose the stored preprocessing statistics, and [`ncomponents`](@ref) returns the number
of latent components retained in the fitted object.

```@docs
NCPLS.AbstractNCPLSFit
StatsAPI.coef(::AbstractNCPLSFit)
NCPLS.xmean(::AbstractNCPLSFit)
NCPLS.xstd(::AbstractNCPLSFit)
NCPLS.ymean(::AbstractNCPLSFit)
NCPLS.ncomponents(::AbstractNCPLSFit)
```

## NCPLSFit

[`NCPLSFit`](@ref) is the full fitted model returned by [`fit`](@ref). In addition to the
coefficient tensor, it stores predictor scores, loadings, projection objects, response
residuals, and the multilinear diagnostics associated with each extracted component.

The most commonly used fitted-model getters are:

- [`fitted`](@ref) for fitted responses on the original response scale,
- [`residuals`](@ref) for response residuals,
- [`xscores`](@ref) for latent predictor scores,
- [`responselabels`](@ref), [`samplelabels`](@ref), and [`sampleclasses`](@ref) for
  stored labels,
- [`predictoraxes`](@ref) for retained predictor-axis metadata.

For deeper inspection, the full stored fields remain accessible via dot notation. This is
particularly useful for multilinear fits, where `mf.W_modes`,
`mf.W_multilinear_relerr`, `mf.W_multilinear_method`, `mf.W_multilinear_lambda`,
`mf.W_multilinear_niter`, and `mf.W_multilinear_converged` expose the component-wise
PARAFAC approximation details.

```@docs
NCPLS.NCPLSFit
StatsAPI.fitted(::NCPLSFit)
NCPLS.predictoraxes(::NCPLSFit)
StatsAPI.residuals(::NCPLSFit)
NCPLS.responselabels(::NCPLSFit)
NCPLS.sampleclasses(::NCPLSFit)
NCPLS.samplelabels(::NCPLSFit)
NCPLS.xscores(::NCPLSFit)
```

## NCPLSFitLight

[`NCPLSFitLight`](@ref) is a reduced fitted-model container used mainly in cross-
validation, permutation testing, and nested resampling. It stores only what prediction
needs: the coefficient tensor and the preprocessing statistics required to restore the
predictor and response scales.

Most users will interact with it only indirectly through the resampling helpers.

```@docs
NCPLS.NCPLSFitLight
```
