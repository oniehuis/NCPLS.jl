# Fit

Model fitting in `NCPLS` is performed through [`StatsAPI.fit`](@ref) together with an
[`NCPLSModel`](@ref). As in `CPPLS`, the same `fit` entry point is used for regression
and discriminant analysis. The NCPLS-specific modeling choice is how the predictor-side
loading weights are represented:

- `multilinear=false` keeps the predictor-side weight tensor in unfolded form,
- `multilinear=true` compresses each component to a rank-1 multilinear weight tensor,
- `orthogonalize_mode_weights=true` optionally orthogonalizes the multilinear mode
  vectors across components.

!!! note
    The practical motivation for `obs_weights` and `Yadd` is the same as in the
    matrix-valued `CPPLS` workflow, so that discussion is not repeated in full here.
    See the corresponding `CPPLS` pages for the general rationale and longer examples:
    [Fit](https://oniehuis.github.io/CPPLS.jl/dev/CPPLS/fit/) and
    [Fit Examples](https://oniehuis.github.io/CPPLS.jl/dev/CPPLS/fit_examples/).
    In `NCPLS`, these same options act together with the additional multilinear
    predictor-side compression described on the [Theory](theory.md) page.

## Response Conventions

The interpretation of the third positional argument is determined by its type:

1\. `fit(m, X, y::AbstractVector{<:Real})` fits univariate regression.
2\. `fit(m, X, Y::AbstractMatrix{<:Real})` fits a user-defined response block.
3\. `fit(m, X, labels::AbstractCategoricalArray)` fits discriminant analysis by converting
   the labels internally to a one-hot response matrix.

This means that a custom response matrix may contain:

- only continuous columns,
- only one-hot class-indicator columns,
- or a hybrid block such as `[class indicators | continuous traits]`.

For hybrid response matrices, class-related downstream helpers need
`sampleclasses` and `responselabels` so the class block can be inferred correctly.
The actual decoding of those mixed predictions is discussed on
[Projection and Prediction](predict.md).

## Multilinear Versus Unfolded

`NCPLSModel` controls whether the predictor-side loading weights are kept in unfolded form
or approximated by one rank-1 outer product per component.

`multilinear=true` is the default, because `NCPLS` is primarily meant for genuinely
multiway predictors. In that setting, the multilinear branch is usually the most natural
starting point: it returns one weight vector per predictor mode and can be inspected with
[`weightprofiles`](@ref).

`multilinear=false` is mainly useful for comparison and sensitivity analysis. It lets you
fit the same response model without the rank-1 multilinear compression, so you can ask
whether the multilinear structure is helping or whether it is too restrictive for the
current dataset. It also gives an unfolded NCPLS variant that is methodologically easier
to compare with matrix-based CPPLS analyses. Because the NCPLS implementation extracts
components without explicit predictor deflation, this unfolded branch may also be
computationally attractive in some settings, but that should be treated as a practical
possibility rather than a benchmarked package claim.

The additional switch `orthogonalize_mode_weights=true` is only relevant when
`multilinear=true`. It makes the stored mode vectors orthogonal within each predictor mode
across components. This can make multilinear weight profiles easier to compare, but it is
also a stricter model assumption and is therefore not the default.

## Quick Start

The synthetic dataset used below supports all three fitting styles: discriminant
analysis, regression, and hybrid response modeling.

```@example fit_examples
using NCPLS
using Statistics
using CairoMakie

data = synthetic_multilinear_hybrid_data(
    nmajor=60,
    nminor=40,
    mode_dims=(40, 30),
    orthogonal_truth=true,
    integer_counts=false,
)

model_multilinear = NCPLSModel(
    ncomponents=2,
    multilinear=true,
    orthogonalize_mode_weights=false,
)

model_unfolded = NCPLSModel(
    ncomponents=2,
    multilinear=false,
)

model_multilinear_orth = NCPLSModel(
    ncomponents=2,
    multilinear=true,
    orthogonalize_mode_weights=true,
)

blue, orange, green = Makie.wong_colors()[[1, 2, 3]]

trait1 = data.Yprim_reg[:, 1]
q1, q2 = quantile(trait1, [1/3, 2/3])
trait_bins = categorical(ifelse.(trait1 .<= q1, "low",
                         ifelse.(trait1 .<= q2, "mid", "high")))

nothing # hide
```

### Discriminant Analysis

Passing a categorical vector triggers the discriminant-analysis convenience path.

```@example fit_examples
mf_da_unfolded = fit(
    model_unfolded,
    data.X,
    categorical(data.sampleclasses);
    Yadd=data.Yadd,
    obs_weights=data.obs_weights,
    samplelabels=data.samplelabels,
    predictoraxes=data.predictoraxes,
)

mf_da = fit(
    model_multilinear,
    data.X,
    categorical(data.sampleclasses);
    Yadd=data.Yadd,
    obs_weights=data.obs_weights,
    samplelabels=data.samplelabels,
    predictoraxes=data.predictoraxes,
)

nothing # hide
```

```@example fit_examples
fig_da = Figure(size=(1200, 500))

scoreplot(
    mf_da_unfolded;
    backend=:makie,
    figure=fig_da,
    axis=Axis(fig_da[1, 1]),
    title="DA scores: unfolded",
    group_order=["minor", "major"],
    default_marker=(; markersize=12),
    group_marker=Dict("minor" => (; color=orange), "major" => (; color=blue)),
    show_inspector=false,
)

scoreplot(
    mf_da;
    backend=:makie,
    figure=fig_da,
    axis=Axis(fig_da[1, 2]),
    title="DA scores: multilinear",
    group_order=["minor", "major"],
    default_marker=(; markersize=12),
    group_marker=Dict("minor" => (; color=orange), "major" => (; color=blue)),
    show_inspector=false,
)

save("fit_da_multilinear_vs_unfolded.svg", fig_da)
nothing # hide
```

![](fit_da_multilinear_vs_unfolded.svg)

### Regression

For regression, pass a numeric vector or matrix directly. Here the synthetic dataset
provides two continuous target columns in `data.Yprim_reg`.

```@example fit_examples
mf_reg_unfolded = fit(
    model_unfolded,
    data.X,
    data.Yprim_reg;
    Yadd=data.Yadd,
    obs_weights=data.obs_weights,
    responselabels=data.responselabels_reg,
    samplelabels=data.samplelabels,
    predictoraxes=data.predictoraxes,
)

mf_reg = fit(
    model_multilinear,
    data.X,
    data.Yprim_reg;
    Yadd=data.Yadd,
    obs_weights=data.obs_weights,
    responselabels=data.responselabels_reg,
    samplelabels=data.samplelabels,
    predictoraxes=data.predictoraxes,
)

nothing # hide
```

```@example fit_examples
fig_reg = Figure(size=(1200, 500))

scoreplot(
    data.samplelabels,
    trait_bins,
    xscores(mf_reg_unfolded, 1:2);
    backend=:makie,
    figure=fig_reg,
    axis=Axis(fig_reg[1, 1]),
    title="Regression scores: unfolded",
    group_order=["low", "mid", "high"],
    default_marker=(; markersize=12),
    group_marker=Dict("low" => (; color=blue), "mid" => (; color=green), "high" => (; color=orange)),
    show_inspector=false,
)

scoreplot(
    data.samplelabels,
    trait_bins,
    xscores(mf_reg, 1:2);
    backend=:makie,
    figure=fig_reg,
    axis=Axis(fig_reg[1, 2]),
    title="Regression scores: multilinear",
    group_order=["low", "mid", "high"],
    default_marker=(; markersize=12),
    group_marker=Dict("low" => (; color=blue), "mid" => (; color=green), "high" => (; color=orange)),
    show_inspector=false,
)

save("fit_reg_multilinear_vs_unfolded.svg", fig_reg)
nothing # hide
```

![](fit_reg_multilinear_vs_unfolded.svg)

### Switching Between Unfolded and Multilinear Fits

The response handling is unchanged; only the predictor-side representation differs.
The two figures above therefore isolate the main practical question behind the
`multilinear` switch: how much of the fitted latent structure changes when the predictor
weights are compressed to one rank-1 multilinear term per component.

In practice, this gives three common model specifications:

```julia
NCPLSModel(ncomponents=2, multilinear=false)
NCPLSModel(ncomponents=2, multilinear=true,  orthogonalize_mode_weights=false)
NCPLSModel(ncomponents=2, multilinear=true,  orthogonalize_mode_weights=true)
```

The first keeps fully unfolded predictor-side weights. The second uses multilinear rank-1
compression without additional orthogonality constraints. The third adds orthogonality of
the stored mode vectors across components.

```@example fit_examples
mf_reg_multilinear_orth = fit(
    model_multilinear_orth,
    data.X,
    data.Yprim_reg;
    Yadd=data.Yadd,
    obs_weights=data.obs_weights,
    responselabels=data.responselabels_reg,
    samplelabels=data.samplelabels,
    predictoraxes=data.predictoraxes,
)

profiles_no_orth = [weightprofiles(mf_reg; lv=lv) for lv in 1:2]
profiles_with_orth = [weightprofiles(mf_reg_multilinear_orth; lv=lv) for lv in 1:2]

fig_profiles = Figure(size=(1200, 700))

for (j, axis_meta) in enumerate(data.predictoraxes)
    ax_without = Axis(
        fig_profiles[1, j],
        title="$(axis_meta.name): multilinear without orthogonalization",
        xlabel=axis_meta.name,
        ylabel="Weight",
    )
    lines!(ax_without, axis_meta.values, profiles_no_orth[1][j], color=blue, label="LV1")
    lines!(ax_without, axis_meta.values, profiles_no_orth[2][j], color=orange, label="LV2")
    axislegend(ax_without, position=:rt)

    ax_with = Axis(
        fig_profiles[2, j],
        title="$(axis_meta.name): multilinear with orthogonalization",
        xlabel=axis_meta.name,
        ylabel="Weight",
    )
    lines!(ax_with, axis_meta.values, profiles_with_orth[1][j], color=blue, label="LV1")
    lines!(ax_with, axis_meta.values, profiles_with_orth[2][j], color=orange, label="LV2")
    axislegend(ax_with, position=:rt)
end

save("fit_mode_orthogonalization.svg", fig_profiles)
nothing # hide
```

![](fit_mode_orthogonalization.svg)

This last comparison shows the effect of `orthogonalize_mode_weights=true` on the stored
multilinear mode profiles. The latent score space may change only modestly, but the
mode-wise weight vectors become easier to compare because each predictor mode is forced
to carry orthogonal component profiles.

### Hybrid Response Blocks

Hybrid response blocks are fit exactly like any other numeric response matrix. The only
additional requirement is that the class-indicator columns must be identifiable from
`sampleclasses` and `responselabels`.

```julia
mf_hybrid = fit(
    model_multilinear,
    data.X,
    data.Yprim_hybrid;
    Yadd=data.Yadd,
    obs_weights=data.obs_weights,
    sampleclasses=data.sampleclasses,
    responselabels=data.responselabels_hybrid,
    samplelabels=data.samplelabels,
    predictoraxes=data.predictoraxes,
)
```

This is usually more useful once predictions are inspected, because the main question is
how the class block and continuous block are decoded downstream. For that reason, the
worked hybrid usage is better continued on [Projection and Prediction](predict.md).

## API

```@docs
NCPLS.fit
```
