# Visualization

Visualization in NCPLS happens at three different levels:

- `scoreplot` shows how samples are arranged in the latent space.
- `weightlandscape` and `coefficientlandscape` map fitted objects back onto the original
  predictor surface.
- `weightprofiles` keeps the multilinear mode weights separated, one profile per
  predictor axis.

This page uses Makie for static figures embedded in the manual and PlotlyJS for the
interactive plot wrappers. `scoreplot` supports both backends. The dedicated landscape
and profile plot wrappers currently use PlotlyJS only.

## Example Data

The examples below reuse the synthetic multilinear dataset introduced on the
[Fit](fit.md) page. We fit one discriminant-analysis model for score plots and one
multilinear regression model for the predictor-surface and weight-profile views.

```@example visualization_examples
using NCPLS
using CairoMakie
using PlotlyJS
using Statistics

data = synthetic_multilinear_hybrid_data(
    nmajor=48,
    nminor=32,
    mode_dims=(28, 20),
    orthogonal_truth=true,
    integer_counts=false,
)

model_da = NCPLSModel(
    ncomponents=2,
    multilinear=true,
    orthogonalize_mode_weights=false,
)

model_reg = NCPLSModel(
    ncomponents=2,
    multilinear=true,
    orthogonalize_mode_weights=false,
)

mf_da = fit(
    model_da,
    data.X,
    data.sampleclasses;
    Yadd=data.Yadd,
    obs_weights=data.obs_weights,
    samplelabels=data.samplelabels,
    predictoraxes=data.predictoraxes,
)

mf_reg = fit(
    model_reg,
    data.X,
    data.Yprim_reg;
    Yadd=data.Yadd,
    obs_weights=data.obs_weights,
    responselabels=data.responselabels_reg,
    samplelabels=data.samplelabels,
    sampleclasses=data.sampleclasses,
    predictoraxes=data.predictoraxes,
)

blue, orange, green = Makie.wong_colors()[[1, 2, 3]]

trait1 = data.Yprim_reg[:, 1]
q1, q2 = quantile(trait1, [1 / 3, 2 / 3])
trait_bins = ifelse.(trait1 .<= q1, "low",
             ifelse.(trait1 .<= q2, "mid", "high"))

rt_axis, mz_axis = predictoraxes(mf_reg)

nothing # hide
```

## Score Plots

[`scoreplot`](@ref) has two main entry points:

- `scoreplot(mf)` uses the stored sample labels, sample classes, and first two latent
  variables from a fitted model.
- `scoreplot(samples, groups, scores)` is the lower-level form for custom grouping or
  for plotting scores that were assembled outside the fitted-model convenience path.

```@docs
NCPLS.scoreplot
```

The next figure shows both patterns. On the left, `scoreplot(mf_da)` uses the fitted DA
model directly. On the right, the same plotting function is given explicit sample labels,
custom groups, and the first two regression scores so that the samples are colored by
trait bins rather than by class.

```@example visualization_examples
fig_scores = Figure(size=(1200, 500))

scoreplot(
    mf_da;
    backend=:makie,
    figure=fig_scores,
    axis=Axis(fig_scores[1, 1]),
    title="DA scores by class",
    group_order=["minor", "major"],
    default_marker=(; markersize=12),
    group_marker=Dict(
        "minor" => (; color=orange),
        "major" => (; color=blue),
    ),
    show_inspector=false,
)

scoreplot(
    data.samplelabels,
    trait_bins,
    xscores(mf_reg, 1:2);
    backend=:makie,
    figure=fig_scores,
    axis=Axis(fig_scores[1, 2]),
    title="Regression scores grouped by trait bins",
    group_order=["low", "mid", "high"],
    default_marker=(; markersize=12),
    group_marker=Dict(
        "low" => (; color=blue),
        "mid" => (; color=green),
        "high" => (; color=orange),
    ),
    show_inspector=false,
)

save("visualization_scoreplots.svg", fig_scores)
nothing # hide
```

![](visualization_scoreplots.svg)

The Plotly backend uses the same high-level call but produces an interactive score plot.

```@example visualization_examples
scoreplot_da_plotly = scoreplot(
    mf_da;
    backend=:plotly,
    title="NCPLS DA scores",
    group_order=["minor", "major"],
    default_marker=(; size=10),
    group_marker=Dict(
        "minor" => (; color="rgb(230,159,0)"),
        "major" => (; color="rgb(0,114,178)"),
    ),
)

PlotlyJS.savefig(scoreplot_da_plotly, "visualization_scoreplot.html")
nothing # hide
```

[Open the interactive score plot](visualization_scoreplot.html)

## Predictor Surfaces

NCPLS stores fitted objects that still live on the predictor axes themselves. For a
genuinely two-mode predictor, two views are especially useful:

- `weightlandscape(mf; lv=k)` shows which regions of the predictor surface define one
  latent variable.
- `coefficientlandscape(mf; response=j)` or
  `coefficientlandscape(mf; response_contrast=(pos, neg))` shows how the fitted model
  maps the predictor surface to one response or to a signed response contrast.

These are different objects. Weight landscapes are component objects on the predictor
side. Coefficient landscapes are response objects derived from the cumulative fitted
model. For fits with several responses, it is often clearer to request the response or
contrast explicitly rather than relying on defaults.

```@docs
NCPLS.weightlandscape
NCPLS.weightlandscapeplot
NCPLS.coefficientlandscape
NCPLS.coefflandscapeplot
```

The figure below compares two latent-variable weight surfaces with two coefficient
surfaces from the regression fit.

```@example visualization_examples
W_lv1 = weightlandscape(mf_reg; lv=1)
W_lv2 = weightlandscape(mf_reg; lv=2)
B_trait1 = coefficientlandscape(mf_reg; response=1)
B_contrast = coefficientlandscape(mf_reg; response_contrast=(2, 1))

weight_lim = maximum(abs.(vcat(vec(W_lv1), vec(W_lv2))))
coef_lim = maximum(abs.(vcat(vec(B_trait1), vec(B_contrast))))

fig_surfaces = Figure(size=(1300, 850))

ax_w1 = Axis(
    fig_surfaces[1, 1],
    title="LV1 weight landscape",
    xlabel=rt_axis.name,
    ylabel=mz_axis.name,
)
heatmap!(
    ax_w1,
    rt_axis.values,
    mz_axis.values,
    W_lv1;
    colormap=:RdBu,
    colorrange=(-weight_lim, weight_lim),
)

ax_w2 = Axis(
    fig_surfaces[1, 2],
    title="LV2 weight landscape",
    xlabel=rt_axis.name,
    ylabel=mz_axis.name,
)
heatmap!(
    ax_w2,
    rt_axis.values,
    mz_axis.values,
    W_lv2;
    colormap=:RdBu,
    colorrange=(-weight_lim, weight_lim),
)

ax_b1 = Axis(
    fig_surfaces[2, 1],
    title="$(data.responselabels_reg[1]) coefficient landscape",
    xlabel=rt_axis.name,
    ylabel=mz_axis.name,
)
heatmap!(
    ax_b1,
    rt_axis.values,
    mz_axis.values,
    B_trait1;
    colormap=:RdBu,
    colorrange=(-coef_lim, coef_lim),
)

ax_bc = Axis(
    fig_surfaces[2, 2],
    title="$(data.responselabels_reg[2]) - $(data.responselabels_reg[1])",
    xlabel=rt_axis.name,
    ylabel=mz_axis.name,
)
heatmap!(
    ax_bc,
    rt_axis.values,
    mz_axis.values,
    B_contrast;
    colormap=:RdBu,
    colorrange=(-coef_lim, coef_lim),
)

Colorbar(
    fig_surfaces[1, 3],
    limits=(-weight_lim, weight_lim),
    colormap=:RdBu,
    label="Weight",
)

Colorbar(
    fig_surfaces[2, 3],
    limits=(-coef_lim, coef_lim),
    colormap=:RdBu,
    label="Coefficient",
)

save("visualization_surfaces.svg", fig_surfaces)
nothing # hide
```

![](visualization_surfaces.svg)

The interactive plot wrappers build these views directly from the fitted model. For
combined weight summaries across all components, absolute combinations such as
`combine=:meanabs` are often easier to interpret than signed sums because latent-variable
signs are arbitrary.

```@example visualization_examples
weight_plotly = weightlandscapeplot(
    mf_reg;
    lv=:combined,
    combine=:meanabs,
    title="Combined weight landscape (meanabs)",
)

coef_plotly = coefflandscapeplot(
    mf_reg;
    response_contrast=(2, 1),
    title="Coefficient landscape: trait2 - trait1",
)

PlotlyJS.savefig(weight_plotly, "visualization_weight_landscape.html")
PlotlyJS.savefig(coef_plotly, "visualization_coefficient_landscape.html")
nothing # hide
```

[Open the interactive combined weight landscape](visualization_weight_landscape.html)

[Open the interactive coefficient landscape](visualization_coefficient_landscape.html)

## Multilinear Weight Profiles

When `multilinear=true`, NCPLS also stores one loading-weight vector per predictor mode
and per component. [`weightprofiles`](@ref) returns those vectors directly, whereas
[`weightprofilesplot`](@ref) stacks them into one Plotly figure with one subplot per
predictor axis.

This view is often easier to interpret than a 2D surface when the main question is how
each individual mode contributes across its own coordinate system.

```@docs
NCPLS.weightprofiles
NCPLS.weightprofilesplot
```

Here we compare the mode-specific profiles for LV1 and LV2 and add a combined absolute
summary across all stored components.

```@example visualization_examples
profiles_lv1 = weightprofiles(mf_reg; lv=1)
profiles_lv2 = weightprofiles(mf_reg; lv=2)
profiles_all = weightprofiles(mf_reg; lv=:combined, combine=:meanabs)

fig_profiles = Figure(size=(1200, 450))

for (j, axis_meta) in enumerate(data.predictoraxes)
    ax = Axis(
        fig_profiles[1, j],
        title="$(axis_meta.name) weight profiles",
        xlabel=axis_meta.name,
        ylabel="Weight",
    )
    lines!(
        ax,
        axis_meta.values,
        zeros(length(axis_meta.values));
        color=:gray60,
        linestyle=:dash,
    )
    lines!(ax, axis_meta.values, profiles_lv1[j], color=blue, label="LV1")
    lines!(ax, axis_meta.values, profiles_lv2[j], color=orange, label="LV2")
    lines!(
        ax,
        axis_meta.values,
        profiles_all[j];
        color=:black,
        linestyle=:dot,
        label="meanabs all",
    )
    axislegend(ax, position=:rt)
end

save("visualization_weightprofiles.svg", fig_profiles)
nothing # hide
```

![](visualization_weightprofiles.svg)

```@example visualization_examples
profiles_plotly = weightprofilesplot(
    mf_reg;
    lv=:combined,
    combine=:meanabs,
    title="Combined weight profiles (meanabs)",
)

PlotlyJS.savefig(profiles_plotly, "visualization_weightprofiles.html")
nothing # hide
```

[Open the interactive weight profiles](visualization_weightprofiles.html)

## API

- [`scoreplot`](@ref NCPLS.scoreplot)
- [`weightlandscape`](@ref NCPLS.weightlandscape)
- [`weightlandscapeplot`](@ref NCPLS.weightlandscapeplot)
- [`coefficientlandscape`](@ref NCPLS.coefficientlandscape)
- [`coefflandscapeplot`](@ref NCPLS.coefflandscapeplot)
- [`weightprofiles`](@ref NCPLS.weightprofiles)
- [`weightprofilesplot`](@ref NCPLS.weightprofilesplot)
