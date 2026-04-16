# Projection and Prediction

After fitting an NCPLS model with [`fit`](@ref), you can apply it in two main ways:

- Use [`project`](@ref) to map new samples into the latent score space defined by the
  fitted model.
- Use [`predict`](@ref) to generate predicted responses from new predictor arrays.

`project(mf, Xnew)` returns a matrix with one row per new sample and one column per latent
component. In `predict(mf, Xnew, A)`, `A::Integer` is the requested number of latent
components to use, with `1 ≤ A ≤ ncomponents(mf)`. The return value is a numeric array of
size `(n_samples, A, n_responses)`, where slice `[:, a, :]` is the cumulative prediction
based on the first `a` components. This means that even `predict(mf, Xnew, 1)` returns a
three-dimensional array with one component slice rather than a plain matrix.

For classification-capable fits, NCPLS provides two decoding layers on top of `predict`:

- `onehot(mf, Xnew, A)` or `onehot(mf, predictions)` converts the final requested
  component slice to a one-hot class matrix.
- `predictclasses(mf, Xnew, A)` or `predictclasses(mf, predictions)` maps those class
  scores back to class labels.

For mixed response blocks such as `[class indicators | continuous traits]`, `predict`
still returns the full numeric response block, while `onehot` and `predictclasses`
automatically use only the inferred class-response columns.

!!! note
    Once the model has been fitted, prediction-time calls need only new predictor data
    `Xnew`. Optional `Yadd` and `obs_weights` influence component extraction during
    fitting, but they are not required for `project`, `predict`, `onehot`, or
    `predictclasses`.

## Example Data

The examples below reuse the synthetic multilinear dataset introduced on the
[Fit](fit.md) page. We hold out six samples from each class and fit the models on the
remaining observations, so the same training and hold-out split can be reused across the
projection, regression, discriminant, and hybrid examples.

```@example predict_examples
using NCPLS
using Statistics
using CairoMakie

data = synthetic_multilinear_hybrid_data(
    nmajor=48,
    nminor=32,
    mode_dims=(30, 20),
    orthogonal_truth=true,
    integer_counts=false,
    class_component_strength=6.0,
    regression_component_strength=5.5,
    nuisance_component_strength=3.5,
    x_noise_scale_clean=0.02,
    x_noise_scale_noisy=0.10,
    yreg_noise_scale_clean=0.02,
    yreg_noise_scale_noisy=0.08,
    yadd_noise_scale=0.02,
)

major_idx = findall(==("major"), data.sampleclasses_string)
minor_idx = findall(==("minor"), data.sampleclasses_string)
holdout_idx = vcat(major_idx[1:6], minor_idx[1:6])
train_idx = setdiff(collect(axes(data.X, 1)), holdout_idx)

X_train = data.X[train_idx, :, :]
X_holdout = data.X[holdout_idx, :, :]
Yadd_train = data.Yadd[train_idx, :]
obs_weights_train = data.obs_weights[train_idx]

labels_train = data.samplelabels[train_idx]
labels_holdout = data.samplelabels[holdout_idx]
classes_train = categorical(data.sampleclasses_string[train_idx])
classes_holdout = data.sampleclasses_string[holdout_idx]
plot_classes_holdout = "projected " .* classes_holdout

model = NCPLSModel(
    ncomponents=2,
    multilinear=true,
    orthogonalize_mode_weights=false,
    scale_X=false,
)

blue, orange = Makie.wong_colors()[[1, 2]]

nothing # hide
```

## Projection

[`project`](@ref) maps new predictor arrays into the latent score space defined by a
fitted model. Use it when you want latent coordinates for new samples, score plots that
combine training and hold-out observations, or a projection step before downstream
interpretation.

```@docs
NCPLS.project
```

We first fit a discriminant model and then project the held-out samples into the score
space defined by the training data.

```@example predict_examples
mf_da = fit(
    model,
    X_train,
    classes_train;
    Yadd=Yadd_train,
    obs_weights=obs_weights_train,
    samplelabels=labels_train,
    predictoraxes=data.predictoraxes,
)

heldout_scores = project(mf_da, X_holdout)

fig_da = Figure(size=(900, 600))

scoreplot(
    vcat(labels_train, labels_holdout),
    vcat(data.sampleclasses_string[train_idx], plot_classes_holdout),
    vcat(xscores(mf_da), heldout_scores);
    backend=:makie,
    figure=fig_da,
    axis=Axis(fig_da[1, 1]),
    title="Projected hold-out samples",
    group_order=["minor", "projected minor", "major", "projected major"],
    default_marker=(; markersize=10),
    group_marker=Dict(
        "minor" => (; color=orange),
        "projected minor" => (; color=orange, marker=:x, markersize=15, strokecolor=:black, strokewidth=1),
        "major" => (; color=blue),
        "projected major" => (; color=blue, marker=:x, markersize=15, strokecolor=:black, strokewidth=1),
    ),
    show_inspector=false,
)

save("predict_projected_holdout.svg", fig_da)
nothing # hide
```

![](predict_projected_holdout.svg)

The returned `heldout_scores` matrix has one row per held-out sample and one column per
latent component. These coordinates live in the same score space as `xscores(mf_da)`, so
training samples and projected new samples can be plotted together directly.

## Predicting Responses

[`predict`](@ref) returns the cumulative response predictions for the requested number of
latent components. The return value is always numeric and always contains the full
response block, even for discriminant or hybrid fits.

```@docs
NCPLS.predict
```

For regression, the response block is interpreted as continuous variables.

```@example predict_examples
mf_reg = fit(
    model,
    X_train,
    data.Yprim_reg[train_idx, :];
    Yadd=Yadd_train,
    obs_weights=obs_weights_train,
    samplelabels=labels_train,
    responselabels=data.responselabels_reg,
    predictoraxes=data.predictoraxes,
)

Yhat_reg = predict(mf_reg, X_holdout, 2)
holdout_reg = @view Yhat_reg[:, end, :]
trait_correlations = [
    cor(holdout_reg[:, j], data.Yprim_reg[holdout_idx, j]) for j in axes(data.Yprim_reg, 2)
]
tensor_size=size(Yhat_reg)
```

```@example predict_examples
holdout_correlations=collect(zip(data.responselabels_reg, round.(trait_correlations; digits=3)))
```

```@example predict_examples
fig_reg = Figure(size=(1000, 420))

for j in axes(data.Yprim_reg, 2)
    observed = data.Yprim_reg[holdout_idx, j]
    predicted = holdout_reg[:, j]
    lo, hi = extrema(vcat(observed, predicted))
    pad = 0.05 * (hi - lo + eps(Float64))

    ax = Axis(
        fig_reg[1, j],
        title="$(data.responselabels_reg[j]): observed vs predicted",
        xlabel="Observed",
        ylabel="Predicted",
    )
    scatter!(ax, observed, predicted, color=(j == 1 ? blue : orange, 0.85), markersize=12)
    lines!(ax, [lo - pad, hi + pad], [lo - pad, hi + pad], color=:black, linestyle=:dash)
end

save("predict_regression_holdout.svg", fig_reg)
nothing # hide
```

![](predict_regression_holdout.svg)

The final matrix of predicted responses is the last component slice,
`Yhat_reg[:, end, :]`. Earlier slices show the cumulative prediction after fewer latent
components.

## Decoding Class Predictions

For discriminant models, the raw output of `predict` remains numeric: it is a tensor of
class scores, not a vector of labels. Use [`onehot`](@ref) when you want a one-hot class
matrix and [`predictclasses`](@ref) when you want decoded class labels.

```@docs
NCPLS.onehot(::NCPLS.AbstractNCPLSFit, ::AbstractArray{<:Real}, ::Integer)
NCPLS.onehot(::NCPLS.AbstractNCPLSFit, ::AbstractArray{<:Real,3})
NCPLS.predictclasses
```

```@example predict_examples
Yhat_da = predict(mf_da, X_holdout, 2)
predicted_da = predictclasses(mf_da, Yhat_da)
tensor_size=size(Yhat_da)
```

```@example predict_examples
final_class_scores=round.(Yhat_da[1:4, end, :]; digits=3)
```

```@example predict_examples
hcat(labels_holdout, classes_holdout, predicted_da)
```

The final requested component slice is `Yhat_da[:, end, :]`. If you prefer one-hot class
assignments instead of labels, use `onehot(mf_da, Yhat_da)` or the convenience wrapper
`onehot(mf_da, X_holdout, 2)`.

## Hybrid Response Blocks

Mixed response models combine class-indicator columns and continuous targets in a single
response block. The fitted model still uses `predict` for the full numeric response, but
the class helpers automatically isolate the class block.

```@example predict_examples
mf_hybrid = fit(
    model,
    X_train,
    data.Yprim_hybrid[train_idx, :];
    Yadd=Yadd_train,
    obs_weights=obs_weights_train,
    samplelabels=labels_train,
    sampleclasses=data.sampleclasses_string[train_idx],
    responselabels=data.responselabels_hybrid,
    predictoraxes=data.predictoraxes,
)

Yhat_hybrid = predict(mf_hybrid, X_holdout, 2)
predicted_hybrid_classes = predictclasses(mf_hybrid, Yhat_hybrid)
predicted_hybrid_onehot = onehot(mf_hybrid, Yhat_hybrid)
predicted_hybrid_traits = @view Yhat_hybrid[:, end, data.regressioncols]

(;
    tensor_size=size(Yhat_hybrid),
    onehot_size=size(predicted_hybrid_onehot),
    continuous_block_size=size(predicted_hybrid_traits),
    class_accuracy=mean(predicted_hybrid_classes .== classes_holdout),
)
```

```@example predict_examples
hcat(
    labels_holdout,
    classes_holdout,
    predicted_hybrid_classes,
    string.(round.(predicted_hybrid_traits[:, 1]; digits=2)),
    string.(round.(predicted_hybrid_traits[:, 2]; digits=2)),
)
```

This illustrates the main downstream pattern for hybrid responses:

```julia
Yhat = predict(mf_hybrid, Xnew, 2)
class_labels = predictclasses(mf_hybrid, Yhat)
class_onehot = onehot(mf_hybrid, Yhat)
continuous_cols = data.regressioncols
continuous_targets = Yhat[:, end, continuous_cols]
```

`predict` keeps the full response block intact, while `predictclasses` and `onehot`
decode only the inferred class-response columns.

## API

[`project`](@ref NCPLS.project)
[`predict`](@ref StatsAPI.predict)
[`onehot(mf, X, ncomps)`](@ref NCPLS.onehot(::NCPLS.AbstractNCPLSFit, ::AbstractArray{<:Real}, ::Integer))
[`onehot(mf, predictions)`](@ref NCPLS.onehot(::NCPLS.AbstractNCPLSFit, ::AbstractArray{<:Real,3}))
[`predictclasses(mf, X, ncomps)`](@ref NCPLS.predictclasses(::NCPLS.NCPLSFit, ::AbstractArray{<:Real}, ::Integer))
`predictclasses(mf, predictions)`](@ref NCPLS.predictclasses(::NCPLS.NCPLSFit, ::AbstractArray{<:Real,3}))
