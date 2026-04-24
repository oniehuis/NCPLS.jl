"""
    onehot(label_indices::AbstractVector{<:Integer}, n_labels::Integer)

Convert 1-based integer class indices to a dense one-hot matrix with `n_labels`
columns.
"""
function onehot(label_indices::AbstractVector{<:Integer}, n_labels::Integer)
    n_labels ≥ 0 || throw(ArgumentError("n_labels must be nonnegative, got $n_labels"))
    all(≥(1), label_indices) || throw(ArgumentError(
        "label_indices must contain only positive 1-based class indices"))
    isempty(label_indices) || maximum(label_indices) ≤ n_labels || throw(ArgumentError(
        "n_labels must be at least maximum(label_indices) = $(maximum(label_indices)), " *
        "got $n_labels"))

    one_hot = zeros(Int, length(label_indices), n_labels)
    @inbounds for (i, label_idx) in enumerate(label_indices)
        one_hot[i, label_idx] = 1
    end

    one_hot
end

"""
    onehot(labels::AbstractVector)

Encode arbitrary labels into a one-hot matrix and return `(matrix, ordered_labels)`.
"""
function onehot(labels::AbstractVector)
    unique_labels = sort(collect(Set(labels)))
    label_to_index = Dict(label => i for (i, label) in enumerate(unique_labels))

    one_hot = zeros(Int, length(labels), length(unique_labels))
    @inbounds for (i, label) in enumerate(labels)
        one_hot[i, label_to_index[label]] = 1
    end

    one_hot, unique_labels
end

"""
    sampleclasses(one_hot_matrix::AbstractMatrix{<:Real})

Decode one-hot rows back to 1-based class indices.
"""
sampleclasses(one_hot_matrix::AbstractMatrix{<:Real}) = decode_one_hot_indices(one_hot_matrix)

"""
    invfreqweights(samples::AbstractVector)

Return normalized inverse-frequency weights for `samples`.
"""
function invfreqweights(samples::AbstractVector)
    countof = Dict{eltype(samples), Int}()
    for sample in samples
        countof[sample] = get(countof, sample, 0) + 1
    end

    weights = [1 / countof[sample] for sample in samples]
    weights ./ sum(weights)
end

"""
    nmc(Y_true_one_hot, Y_pred_one_hot, weighted)

Compute the normalized misclassification cost between two one-hot label matrices.
"""
function nmc(
    Y_true_one_hot::AbstractMatrix{<:Real},
    Y_pred_one_hot::AbstractMatrix{<:Real},
    weighted::Bool,
)
    size(Y_true_one_hot) == size(Y_pred_one_hot) || throw(DimensionMismatch(
        "Input matrices must have the same dimensions"))

    n_samples = size(Y_true_one_hot, 1)
    n_samples > 0 || throw(ArgumentError(
        "Cannot compute weighted NMC: input has zero samples"))

    !weighted && return mean(Y_true_one_hot .≠ Y_pred_one_hot)

    true_labels = sampleclasses(Y_true_one_hot)
    pred_labels = sampleclasses(Y_pred_one_hot)
    sample_weights = invfreqweights(true_labels)
    errors = true_labels .≠ pred_labels

    clamp(sum(sample_weights[errors]), 0.0, 1.0)
end

"""
    pvalue(null_scores, observed_score; tail=:upper)

Compute a one-sided empirical p-value from permutation or null scores.
"""
function pvalue(
    null_scores::AbstractVector{<:Real},
    observed_score::Real;
    tail::Symbol=:upper,
)
    tail in (:upper, :lower) || throw(ArgumentError(
        "tail must be :upper or :lower, got $tail"))

    count_fn = if tail === :upper
        x -> x ≥ observed_score || x ≈ observed_score
    else
        x -> x ≤ observed_score || x ≈ observed_score
    end

    (count(count_fn, null_scores) + 1) / (length(null_scores) + 1)
end

"""
    onehot(mf::AbstractNCPLSFit, X::AbstractArray{<:Real}, ncomps::Integer=ncomponents(mf))

Generate one-hot predictions from a fitted NCPLS model. Unlike CPPLS, NCPLS stores
cumulative predictions along the component axis, so the last requested component slice is
used directly.
"""
function onehot(
    mf::AbstractNCPLSFit,
    X::AbstractArray{<:Real},
    ncomps::Integer=ncomponents(mf),
)
    onehot(mf, predict(mf, X, ncomps))
end

"""
    onehot(mf::AbstractNCPLSFit, predictions::AbstractArray{<:Real, 3})

Convert NCPLS prediction tensors `(samples, components, responses)` into one-hot labels.
For full `NCPLSFit` objects, NCPLS uses the inferred class-response block only, so mixed
response fits of the form `[class scores | continuous traits]` are supported. The reduced
`NCPLSFitLight` fallback uses the full response block and is intended mainly for internal
cross-validation helpers on pure classification responses.
"""
function onehot(
    ::AbstractNCPLSFit,
    predictions::AbstractArray{<:Real, 3},
)
    size(predictions, 2) > 0 || throw(ArgumentError(
        "predictions must contain at least one component slice"))

    predicted_scores = @views predictions[:, end, :]
    predicted_class_indices = argmax.(eachrow(predicted_scores))
    onehot(predicted_class_indices, size(predictions, 3))
end

function class_response_columns(mf::NCPLSFit)
    cols = class_response_columns(sampleclasses(mf), responselabels(mf))
    if !isnothing(cols)
        return cols
    end

    Ytrain = fitted(mf) + residuals(mf)
    is_one_hot_matrix(Ytrain) && return collect(1:size(Ytrain, 2))

    throw(ArgumentError(
        "This fitted model does not define class-response columns. Pass categorical " *
        "labels to `fit`, or provide `sampleclasses` plus matching `responselabels` " *
        "for the class-indicator part of a custom response matrix."
    ))
end

function onehot(
    mf::NCPLSFit,
    predictions::AbstractArray{<:Real, 3},
)
    size(predictions, 2) > 0 || throw(ArgumentError(
        "predictions must contain at least one component slice"))

    classcols = class_response_columns(mf)
    predicted_scores = @views predictions[:, end, classcols]
    predicted_class_indices = argmax.(eachrow(predicted_scores))
    onehot(predicted_class_indices, length(classcols))
end

"""
    predictclasses(mf::NCPLSFit, X::AbstractArray{<:Real}, ncomps::Integer=ncomponents(mf))
    predictclasses(mf::NCPLSFit, predictions::AbstractArray{<:Real, 3})

Map NCPLS predictions back to class labels using the inferred class-response block.
"""
function predictclasses(
    mf::NCPLSFit,
    X::AbstractArray{<:Real},
    ncomps::Integer=ncomponents(mf),
)
    predictclasses(mf, predict(mf, X, ncomps))
end

function predictclasses(
    mf::NCPLSFit,
    predictions::AbstractArray{<:Real, 3},
)
    classcols = class_response_columns(mf)
    isempty(responselabels(mf)) && throw(ArgumentError(
        "responselabels must be provided to map predictions to class labels"))

    classlabels = responselabels(mf)[classcols]
    classlabels[sampleclasses(onehot(mf, predictions))]
end

"""
    random_batch_indices(strata, num_batches, rng=Random.GLOBAL_RNG)

Construct stratified batches by shuffling the samples within each stratum and then
dealing them round-robin into `num_batches` folds.
"""
function random_batch_indices(
    strata::AbstractVector,
    num_batches::Integer,
    rng::AbstractRNG=Random.GLOBAL_RNG,
)
    n_samples = length(strata)

    num_batches ≥ 1 || throw(ArgumentError("Number of batches must be at least 1."))
    num_batches ≤ n_samples || throw(ArgumentError(
        "Number of batches ($num_batches) exceeds number of samples ($n_samples)."))

    strata_groups = Dict(stratum => findall(==(stratum), strata) for stratum in unique(strata))
    min_stratum_size = minimum(length, values(strata_groups))

    num_batches ≤ fld(min_stratum_size, 2) || throw(ArgumentError(
        "Number of batches ($num_batches) is too large for the smallest stratum " *
        "(size = $min_stratum_size). Each fold must have at least 2 samples per stratum, " *
        "so num_batches must be ≤ $(fld(min_stratum_size, 2))."))

    batches = [Int[] for _ in 1:num_batches]
    for (stratum, indices) in strata_groups
        shuffled = shuffle(rng, indices)
        if !(length(shuffled) % num_batches ≈ 0)
            @info ("Stratum $stratum (size = $(length(shuffled))) not evenly divisible by " *
                "$num_batches batches.")
        end
        for (i, idx) in enumerate(shuffled)
            push!(batches[mod1(i, num_batches)], idx)
        end
    end

    batches
end

"""
    cv_classification(; weighted::Bool=true)

Return the default callback bundle for NCPLS discriminant analysis.
"""
function cv_classification(; weighted::Bool=true)
    score_fn = (Y_true, Y_pred) -> 1 - nmc(Y_true, Y_pred, weighted)
    predict_fn = (model, X, k) -> onehot(model, X, k)
    select_fn = argmax
    flag_fn = (Y_true, Y_pred) -> sampleclasses(Y_pred) .≠ sampleclasses(Y_true)
    (score_fn=score_fn, predict_fn=predict_fn, select_fn=select_fn, flag_fn=flag_fn)
end

"""
    cv_regression(; score_fn=..., select_fn=argmin)

Return the default callback bundle for NCPLS regression.
"""
function cv_regression(;
    score_fn::Function=(Y_true, Y_pred) -> sqrt(mean((Y_true .- Y_pred) .^ 2)),
    select_fn::Function=argmin,
)
    predict_fn = (model, X, k) -> @views predict(model, X, k)[:, end, :]
    (score_fn=score_fn, predict_fn=predict_fn, select_fn=select_fn)
end

"""
    cvreg(X, Y; spec, fit_kwargs=(;), num_outer_folds=8, num_outer_folds_repeats=num_outer_folds,
          num_inner_folds=7, num_inner_folds_repeats=num_inner_folds,
          reshuffle_outer_folds=false, rng=Random.GLOBAL_RNG, verbose=true)
    cvreg(X, y; kwargs...)

Run nested cross-validation for NCPLS regression.

The matrix method expects a numeric response block `Y`; the vector method is a convenience
wrapper for univariate regression and reshapes `y` to a one-column matrix. The return
value is the same tuple as [`nestedcv`](@ref): the outer-fold scores and the selected
number of latent variables per outer fold. Component counts are selected from
`1:spec.ncomponents`.
"""
function cvreg(
    X::AbstractArray{<:Real},
    Y::AbstractMatrix{<:Real};
    spec::NCPLSModel,
    fit_kwargs::NamedTuple=(;),
    num_outer_folds::Integer=8,
    num_outer_folds_repeats::Integer=num_outer_folds,
    num_inner_folds::Integer=7,
    num_inner_folds_repeats::Integer=num_inner_folds,
    reshuffle_outer_folds::Bool=false,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    verbose::Bool=true,
)
    cb = cv_regression()

    nestedcv(
        X,
        Y;
        spec=spec,
        fit_kwargs=fit_kwargs,
        score_fn=cb.score_fn,
        predict_fn=cb.predict_fn,
        select_fn=cb.select_fn,
        num_outer_folds=num_outer_folds,
        num_outer_folds_repeats=num_outer_folds_repeats,
        num_inner_folds=num_inner_folds,
        num_inner_folds_repeats=num_inner_folds_repeats,
        max_components=spec.ncomponents,
        reshuffle_outer_folds=reshuffle_outer_folds,
        rng=rng,
        verbose=verbose,
    )
end

function cvreg(
    X::AbstractArray{<:Real},
    y::AbstractVector{<:Real};
    kwargs...,
)
    cvreg(X, reshape(y, :, 1); kwargs...)
end

"""
    permreg(X, Y; spec, fit_kwargs=(;), num_permutations=999, num_outer_folds=8,
            num_outer_folds_repeats=num_outer_folds, num_inner_folds=7,
            num_inner_folds_repeats=num_inner_folds, reshuffle_outer_folds=false,
            rng=Random.GLOBAL_RNG, verbose=true)
    permreg(X, y; kwargs...)

Run a permutation test around nested NCPLS regression cross-validation.

The matrix method expects a numeric response block `Y`; the vector method is a
convenience wrapper for univariate regression. The returned vector contains one
cross-validation score for each permutation and can be compared to the observed score
from [`cvreg`](@ref), for example via [`pvalue`](@ref). Component counts are selected
from `1:spec.ncomponents`.
"""
function permreg(
    X::AbstractArray{<:Real},
    Y::AbstractMatrix{<:Real};
    spec::NCPLSModel,
    fit_kwargs::NamedTuple=(;),
    num_permutations::Integer=999,
    num_outer_folds::Integer=8,
    num_outer_folds_repeats::Integer=num_outer_folds,
    num_inner_folds::Integer=7,
    num_inner_folds_repeats::Integer=num_inner_folds,
    reshuffle_outer_folds::Bool=false,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    verbose::Bool=true,
)
    cb = cv_regression()

    nestedcvperm(
        X,
        Y;
        spec=spec,
        fit_kwargs=fit_kwargs,
        score_fn=cb.score_fn,
        predict_fn=cb.predict_fn,
        select_fn=cb.select_fn,
        num_permutations=num_permutations,
        num_outer_folds=num_outer_folds,
        num_outer_folds_repeats=num_outer_folds_repeats,
        num_inner_folds=num_inner_folds,
        num_inner_folds_repeats=num_inner_folds_repeats,
        max_components=spec.ncomponents,
        reshuffle_outer_folds=reshuffle_outer_folds,
        rng=rng,
        verbose=verbose,
    )
end

function permreg(
    X::AbstractArray{<:Real},
    y::AbstractVector{<:Real};
    kwargs...,
)
    permreg(X, reshape(y, :, 1); kwargs...)
end

"""
    cvda(X, Y; spec, fit_kwargs=(;), weighted=true, num_outer_folds=8,
         num_outer_folds_repeats=num_outer_folds, num_inner_folds=7,
         num_inner_folds_repeats=num_inner_folds, reshuffle_outer_folds=false,
         rng=Random.GLOBAL_RNG, verbose=true)
    cvda(X, sample_labels::AbstractCategoricalArray; kwargs...)

Run nested cross-validation for NCPLS discriminant analysis.

The matrix method expects a one-hot encoded response matrix `Y`; the categorical-label
method accepts an `AbstractCategoricalArray` of class labels and converts it internally to
one-hot form. Outer and inner folds are stratified by class. The return value is the same
tuple as [`nestedcv`](@ref): the outer-fold classification scores and the selected number
of latent variables per outer fold. Mixed response blocks with additional continuous
columns are not supported by this helper; use [`nestedcv`](@ref) directly with custom
callbacks in that case. Component counts are selected from `1:spec.ncomponents`.
"""
function cvda(
    X::AbstractArray{<:Real},
    Y::AbstractMatrix{<:Real};
    spec::NCPLSModel,
    fit_kwargs::NamedTuple=(;),
    weighted::Bool=true,
    num_outer_folds::Integer=8,
    num_outer_folds_repeats::Integer=num_outer_folds,
    num_inner_folds::Integer=7,
    num_inner_folds_repeats::Integer=num_inner_folds,
    reshuffle_outer_folds::Bool=false,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    verbose::Bool=true,
)
    cb = cv_classification(; weighted=weighted)

    nestedcv(
        X,
        Y;
        spec=spec,
        fit_kwargs=fit_kwargs,
        obs_weight_fn=default_da_obs_weight_fn,
        score_fn=cb.score_fn,
        predict_fn=cb.predict_fn,
        select_fn=cb.select_fn,
        num_outer_folds=num_outer_folds,
        num_outer_folds_repeats=num_outer_folds_repeats,
        num_inner_folds=num_inner_folds,
        num_inner_folds_repeats=num_inner_folds_repeats,
        max_components=spec.ncomponents,
        strata=sampleclasses(Y),
        reshuffle_outer_folds=reshuffle_outer_folds,
        rng=rng,
        verbose=verbose,
    )
end

function cvda(
    X::AbstractArray{<:Real},
    sample_labels::AbstractCategoricalArray{T,1,R,V,C,U};
    spec::NCPLSModel,
    fit_kwargs::NamedTuple=(;),
    weighted::Bool=true,
    num_outer_folds::Integer=8,
    num_outer_folds_repeats::Integer=num_outer_folds,
    num_inner_folds::Integer=7,
    num_inner_folds_repeats::Integer=num_inner_folds,
    reshuffle_outer_folds::Bool=false,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    verbose::Bool=true,
) where {T,R,V,C,U}
    Y, responselabels = onehot(sample_labels)
    fit_kwargs = with_response_labels(fit_kwargs, responselabels)

    cvda(
        X,
        Y;
        spec=spec,
        fit_kwargs=fit_kwargs,
        weighted=weighted,
        num_outer_folds=num_outer_folds,
        num_outer_folds_repeats=num_outer_folds_repeats,
        num_inner_folds=num_inner_folds,
        num_inner_folds_repeats=num_inner_folds_repeats,
        reshuffle_outer_folds=reshuffle_outer_folds,
        rng=rng,
        verbose=verbose,
    )
end

function cvda(
    X::AbstractArray{<:Real},
    sample_labels::AbstractVector;
    kwargs...,
)
    throw(ArgumentError(
        "cvda expects a categorical vector of class labels. Wrap the labels in " *
        "`categorical(...)`, or pass a one-hot response matrix to the matrix method."))
end

"""
    permda(X, Y; spec, fit_kwargs=(;), weighted=true, num_permutations=999,
           num_outer_folds=8, num_outer_folds_repeats=num_outer_folds,
           num_inner_folds=7, num_inner_folds_repeats=num_inner_folds,
           reshuffle_outer_folds=false, rng=Random.GLOBAL_RNG, verbose=true)
    permda(X, sample_labels::AbstractCategoricalArray; kwargs...)

Run a permutation test around nested NCPLS discriminant-analysis cross-validation.

The matrix method expects a one-hot encoded response matrix `Y`; the categorical-label
method accepts an `AbstractCategoricalArray` of class labels and converts it internally to
one-hot form. The returned vector contains one cross-validation score for each
permutation and can be compared to the observed score from [`cvda`](@ref), for example
via [`pvalue`](@ref). Mixed response blocks with additional continuous columns are not
supported by this helper; use [`nestedcvperm`](@ref) directly with custom callbacks in
that case. Component counts are selected from `1:spec.ncomponents`.
"""
function permda(
    X::AbstractArray{<:Real},
    Y::AbstractMatrix{<:Real};
    spec::NCPLSModel,
    fit_kwargs::NamedTuple=(;),
    weighted::Bool=true,
    num_permutations::Integer=999,
    num_outer_folds::Integer=8,
    num_outer_folds_repeats::Integer=num_outer_folds,
    num_inner_folds::Integer=7,
    num_inner_folds_repeats::Integer=num_inner_folds,
    reshuffle_outer_folds::Bool=false,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    verbose::Bool=true,
)
    cb = cv_classification(; weighted=weighted)

    nestedcvperm(
        X,
        Y;
        spec=spec,
        fit_kwargs=fit_kwargs,
        obs_weight_fn=default_da_obs_weight_fn,
        score_fn=cb.score_fn,
        predict_fn=cb.predict_fn,
        select_fn=cb.select_fn,
        num_permutations=num_permutations,
        num_outer_folds=num_outer_folds,
        num_outer_folds_repeats=num_outer_folds_repeats,
        num_inner_folds=num_inner_folds,
        num_inner_folds_repeats=num_inner_folds_repeats,
        max_components=spec.ncomponents,
        strata=sampleclasses(Y),
        reshuffle_outer_folds=reshuffle_outer_folds,
        rng=rng,
        verbose=verbose,
    )
end

function permda(
    X::AbstractArray{<:Real},
    sample_labels::AbstractCategoricalArray{T,1,R,V,C,U};
    spec::NCPLSModel,
    fit_kwargs::NamedTuple=(;),
    weighted::Bool=true,
    num_permutations::Integer=999,
    num_outer_folds::Integer=8,
    num_outer_folds_repeats::Integer=num_outer_folds,
    num_inner_folds::Integer=7,
    num_inner_folds_repeats::Integer=num_inner_folds,
    reshuffle_outer_folds::Bool=false,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    verbose::Bool=true,
) where {T,R,V,C,U}
    Y, responselabels = onehot(sample_labels)
    fit_kwargs = with_response_labels(fit_kwargs, responselabels)

    permda(
        X,
        Y;
        spec=spec,
        fit_kwargs=fit_kwargs,
        weighted=weighted,
        num_permutations=num_permutations,
        num_outer_folds=num_outer_folds,
        num_outer_folds_repeats=num_outer_folds_repeats,
        num_inner_folds=num_inner_folds,
        num_inner_folds_repeats=num_inner_folds_repeats,
        reshuffle_outer_folds=reshuffle_outer_folds,
        rng=rng,
        verbose=verbose,
    )
end

function permda(
    X::AbstractArray{<:Real},
    sample_labels::AbstractVector;
    kwargs...,
)
    throw(ArgumentError(
        "permda expects a categorical vector of class labels. Wrap the labels in " *
        "`categorical(...)`, or pass a one-hot response matrix to the matrix method."))
end

"""
    nestedcv(X, Y; ...)

Run explicit nested cross-validation for NCPLS.
"""
function nestedcv(
    X::AbstractArray{<:Real},
    Y::AbstractMatrix{<:Real};
    spec::NCPLSModel,
    fit_kwargs::NamedTuple=(;),
    obs_weight_fn::Union{Function, Nothing}=nothing,
    score_fn::Function,
    predict_fn::Function,
    select_fn::Function,
    num_outer_folds::Integer=8,
    num_outer_folds_repeats::Integer=num_outer_folds,
    num_inner_folds::Integer=7,
    num_inner_folds_repeats::Integer=num_inner_folds,
    max_components::Integer=spec.ncomponents,
    strata::Union{AbstractVector, Nothing}=nothing,
    reshuffle_outer_folds::Bool=false,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    verbose::Bool=true,
)
    num_outer_folds_repeats > 0 || throw(ArgumentError(
        "The number of outer folds must be greater than zero"))
    num_inner_folds_repeats > 0 || throw(ArgumentError(
        "The number of inner folds must be greater than zero"))
    reshuffle_outer_folds || num_outer_folds_repeats ≤ num_outer_folds || throw(ArgumentError(
        "The number of outer fold repeats cannot exceed the number of outer folds unless " *
        "reshuffle_outer_folds=true"))
    num_inner_folds_repeats ≤ num_inner_folds || throw(ArgumentError(
        "The number of inner fold repeats cannot exceed the number of inner folds"))
    max_components > 0 || throw(ArgumentError(
        "The number of components must be greater than zero"))

    n_samples = size(X, 1)
    size(Y, 1) == n_samples || throw(DimensionMismatch("Row count mismatch between X and Y"))
    isnothing(strata) || length(strata) == n_samples || throw(DimensionMismatch(
        "Length of strata must match the number of samples."))

    outer_fold_scores = Vector{Float64}(undef, num_outer_folds_repeats)
    optimal_num_latent_variables = Vector{Int}(undef, num_outer_folds_repeats)
    fixed_outer_folds = reshuffle_outer_folds ? nothing :
        build_folds(n_samples, num_outer_folds, rng; strata=strata)

    for outer_fold_idx in 1:num_outer_folds_repeats
        folds = reshuffle_outer_folds ?
            build_folds(n_samples, num_outer_folds, rng; strata=strata) :
            fixed_outer_folds
        test_indices = reshuffle_outer_folds ? folds[1] : folds[outer_fold_idx]

        verbose && println("Outer fold: ", outer_fold_idx, " / ", num_outer_folds_repeats)

        X_test = subset_samples(X, test_indices)
        Y_test = subset_samples(Y, test_indices)

        train_indices = setdiff(1:n_samples, test_indices)
        X_train = subset_samples(X, train_indices)
        Y_train = subset_samples(Y, train_indices)

        base_fold_kwargs = subset_fit_kwargs(fit_kwargs, train_indices, n_samples)
        fold_kwargs = resolve_obs_weights(
            base_fold_kwargs,
            obs_weight_fn,
            X_train,
            Y_train,
            train_indices,
            spec,
        )
        isnothing(strata) || (inner_strata = strata[train_indices])
        isnothing(strata) && (inner_strata = nothing)

        fold_kwargs = ensure_response_labels(fold_kwargs, Y_train)

        best_k = optimize_num_latent_variables(
            X_train,
            Y_train,
            max_components,
            num_inner_folds,
            num_inner_folds_repeats,
            spec,
            base_fold_kwargs,
            obs_weight_fn,
            score_fn,
            predict_fn,
            select_fn,
            rng,
            verbose;
            strata=inner_strata,
            sample_indices=train_indices,
        )
        optimal_num_latent_variables[outer_fold_idx] = best_k

        final_model = fit_ncpls_light(
            with_n_components(spec, best_k),
            X_train,
            Y_train;
            fold_kwargs...,
        )
        Y_pred = predict_fn(final_model, X_test, best_k)
        score = score_fn(Y_test, Y_pred)
        score isa Real || throw(ArgumentError("score_fn must return a Real"))
        outer_fold_scores[outer_fold_idx] = score

        verbose && println("Score for outer fold: ", outer_fold_scores[outer_fold_idx], "\n")
    end

    outer_fold_scores, optimal_num_latent_variables
end

"""
    nestedcvperm(X, Y; ...)

Run a permutation test around `nestedcv`.
"""
function nestedcvperm(
    X::AbstractArray{<:Real},
    Y::AbstractMatrix{<:Real};
    spec::NCPLSModel,
    fit_kwargs::NamedTuple=(;),
    obs_weight_fn::Union{Function, Nothing}=nothing,
    score_fn::Function,
    predict_fn::Function,
    select_fn::Function,
    num_permutations::Integer=999,
    num_outer_folds::Integer=8,
    num_outer_folds_repeats::Integer=num_outer_folds,
    num_inner_folds::Integer=7,
    num_inner_folds_repeats::Integer=num_inner_folds,
    max_components::Integer=spec.ncomponents,
    strata::Union{AbstractVector, Nothing}=nothing,
    reshuffle_outer_folds::Bool=false,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    verbose::Bool=true,
)
    num_permutations > 0 || throw(ArgumentError(
        "num_permutations must be greater than zero"))

    n_samples = size(X, 1)
    size(Y, 1) == n_samples || throw(DimensionMismatch("Row count mismatch between X and Y"))
    isnothing(strata) || length(strata) == n_samples || throw(DimensionMismatch(
        "Length of strata must match the number of samples."))

    permutation_scores = Vector{Float64}(undef, num_permutations)
    for perm_idx in 1:num_permutations
        perm = randperm(rng, n_samples)
        Y_perm = subset_samples(Y, perm)
        strata_perm = isnothing(strata) ? nothing : strata[perm]

        verbose && println("Permutation: ", perm_idx, " / ", num_permutations)

        scores, _ = nestedcv(
            X,
            Y_perm;
            spec=spec,
            fit_kwargs=fit_kwargs,
            obs_weight_fn=obs_weight_fn,
            score_fn=score_fn,
            predict_fn=predict_fn,
            select_fn=select_fn,
            num_outer_folds=num_outer_folds,
            num_outer_folds_repeats=num_outer_folds_repeats,
            num_inner_folds=num_inner_folds,
            num_inner_folds_repeats=num_inner_folds_repeats,
            max_components=max_components,
            strata=strata_perm,
            reshuffle_outer_folds=reshuffle_outer_folds,
            rng=rng,
            verbose=verbose,
        )
        permutation_scores[perm_idx] = mean(scores)
    end

    permutation_scores
end

"""
    outlierscan(X, Y; spec, fit_kwargs=(;), obs_weight_fn=default_da_obs_weight_fn,
                weighted=true, num_outer_folds=8,
                num_outer_folds_repeats=10*num_outer_folds, ...)

Run repeated nested discriminant-analysis CV and count how often each sample is flagged.
The response must be an `AbstractCategoricalArray` of class labels or a pure one-hot
class-indicator matrix; mixed response blocks with additional continuous columns are not
supported here. `weighted` controls the class-balanced score used to choose the number
of components in the inner CV; final outlier flags are still raw per-sample
misclassifications. Component counts are selected from `1:spec.ncomponents`.
"""
function outlierscan(
    X::AbstractArray{<:Real},
    Y::AbstractMatrix{<:Real};
    spec::NCPLSModel,
    fit_kwargs::NamedTuple=(;),
    obs_weight_fn::Union{Function, Nothing}=default_da_obs_weight_fn,
    weighted::Bool=true,
    num_outer_folds::Integer=8,
    num_outer_folds_repeats::Integer=10 * num_outer_folds,
    num_inner_folds::Integer=7,
    num_inner_folds_repeats::Integer=num_inner_folds,
    reshuffle_outer_folds::Bool=true,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    verbose::Bool=true,
)
    n_samples = size(X, 1)
    size(Y, 1) == n_samples || throw(DimensionMismatch("Row count mismatch between X and Y"))

    cb = cv_classification(; weighted=weighted)
    strata = sampleclasses(Y)
    n_tested = zeros(Int, n_samples)
    n_flagged = zeros(Int, n_samples)

    reshuffle_outer_folds || num_outer_folds_repeats ≤ num_outer_folds || throw(ArgumentError(
        "The number of outer fold repeats cannot exceed the number of outer folds unless " *
        "reshuffle_outer_folds=true"))

    fixed_folds = reshuffle_outer_folds ? nothing :
        build_folds(n_samples, num_outer_folds, rng; strata=strata)

    for outer_fold_idx in 1:num_outer_folds_repeats
        folds = reshuffle_outer_folds ?
            build_folds(n_samples, num_outer_folds, rng; strata=strata) :
            fixed_folds
        test_indices = reshuffle_outer_folds ? folds[1] : folds[outer_fold_idx]

        verbose && println("Outer fold: ", outer_fold_idx, " / ", num_outer_folds_repeats)

        X_test = subset_samples(X, test_indices)
        Y_test = subset_samples(Y, test_indices)

        train_indices = setdiff(1:n_samples, test_indices)
        X_train = subset_samples(X, train_indices)
        Y_train = subset_samples(Y, train_indices)

        base_fold_kwargs = subset_fit_kwargs(fit_kwargs, train_indices, n_samples)
        fold_kwargs = resolve_obs_weights(
            base_fold_kwargs,
            obs_weight_fn,
            X_train,
            Y_train,
            train_indices,
            spec,
        )
        fold_kwargs = ensure_response_labels(fold_kwargs, Y_train)

        best_k = optimize_num_latent_variables(
            X_train,
            Y_train,
            spec.ncomponents,
            num_inner_folds,
            num_inner_folds_repeats,
            spec,
            base_fold_kwargs,
            obs_weight_fn,
            cb.score_fn,
            cb.predict_fn,
            cb.select_fn,
            rng,
            verbose;
            strata=strata[train_indices],
            sample_indices=train_indices,
        )

        final_model = fit_ncpls_light(
            with_n_components(spec, best_k),
            X_train,
            Y_train;
            fold_kwargs...,
        )
        Y_pred = cb.predict_fn(final_model, X_test, best_k)
        flags = cb.flag_fn(Y_test, Y_pred)
        length(flags) == length(test_indices) || throw(DimensionMismatch(
            "flag_fn must return one flag per test sample"))

        n_tested[test_indices] .+= 1
        n_flagged[test_indices] .+= flags
    end

    rate = n_flagged ./ max.(1, n_tested)
    (n_tested=n_tested, n_flagged=n_flagged, rate=rate)
end

function outlierscan(
    X::AbstractArray{<:Real},
    sample_labels::AbstractCategoricalArray{T,1,R,V,C,U};
    kwargs...,
) where {T,R,V,C,U}
    Y, _ = onehot(sample_labels)
    outlierscan(X, Y; kwargs...)
end

function outlierscan(
    X::AbstractArray{<:Real},
    sample_labels::AbstractVector;
    kwargs...,
)
    throw(ArgumentError(
        "outlierscan expects a categorical vector of class labels. Wrap the labels in " *
        "`categorical(...)`, or pass a one-hot response matrix to the matrix method."))
end

function optimize_num_latent_variables(
    X_train_full::AbstractArray{<:Real},
    Y_train_full::AbstractMatrix{<:Real},
    max_components::Integer,
    num_inner_folds::Integer,
    num_inner_folds_repeats::Integer,
    spec::NCPLSModel,
    fit_kwargs::NamedTuple,
    obs_weight_fn::Union{Function, Nothing},
    score_fn::Function,
    predict_fn::Function,
    select_fn::Function,
    rng::AbstractRNG,
    verbose::Bool;
    strata::Union{AbstractVector, Nothing}=nothing,
    sample_indices::AbstractVector{<:Integer}=collect(1:size(X_train_full, 1)),
)
    max_components > 0 || throw(ArgumentError(
        "The number of components must be greater than zero"))
    num_inner_folds_repeats ≤ num_inner_folds || throw(ArgumentError(
        "The number of inner fold repeats cannot exceed the number of inner folds"))

    n_samples = size(X_train_full, 1)
    size(Y_train_full, 1) == n_samples || throw(DimensionMismatch(
        "Row count mismatch between X_train_full and Y_train_full"))
    length(sample_indices) == n_samples || throw(DimensionMismatch(
        "Length of sample_indices must match the number of training samples."))

    inner_folds = build_folds(n_samples, num_inner_folds, rng; strata=strata)
    best_num_latent_vars_per_fold = Vector{Int}(undef, num_inner_folds_repeats)

    for inner_fold_idx in 1:num_inner_folds_repeats
        test_indices = inner_folds[inner_fold_idx]

        verbose && println("  Inner fold: ", inner_fold_idx, " / ", num_inner_folds_repeats)

        X_validation = subset_samples(X_train_full, test_indices)
        Y_validation = subset_samples(Y_train_full, test_indices)

        train_indices = setdiff(1:n_samples, test_indices)
        X_train = subset_samples(X_train_full, train_indices)
        Y_train = subset_samples(Y_train_full, train_indices)
        fold_sample_indices = sample_indices[train_indices]

        fold_kwargs = subset_fit_kwargs(fit_kwargs, train_indices, n_samples)
        fold_kwargs = resolve_obs_weights(
            fold_kwargs,
            obs_weight_fn,
            X_train,
            Y_train,
            fold_sample_indices,
            spec,
        )
        fold_kwargs = ensure_response_labels(fold_kwargs, Y_train)

        model = fit_ncpls_light(
            with_n_components(spec, max_components),
            X_train,
            Y_train;
            fold_kwargs...,
        )
        scores = Vector{Float64}(undef, max_components)
        for k in 1:max_components
            score = score_fn(Y_validation, predict_fn(model, X_validation, k))
            score isa Real || throw(ArgumentError("score_fn must return a Real"))
            scores[k] = score
        end

        best_k = select_fn(scores)
        1 ≤ best_k ≤ max_components || throw(ArgumentError(
            "select_fn must return an integer between 1 and $max_components"))

        best_num_latent_vars_per_fold[inner_fold_idx] = best_k
    end

    floor(Int, median(best_num_latent_vars_per_fold))
end

function resolve_obs_weights(
    fit_kwargs::NamedTuple,
    obs_weight_fn::Union{Function, Nothing},
    X_train::AbstractArray{<:Real},
    Y_train::AbstractMatrix{<:Real},
    sample_indices::AbstractVector{<:Integer},
    spec::NCPLSModel,
)
    isnothing(obs_weight_fn) && return fit_kwargs

    base_weights = haskey(fit_kwargs, :obs_weights) ? fit_kwargs.obs_weights : nothing
    derived_weights = obs_weight_fn(
        X_train,
        Y_train;
        sample_indices=sample_indices,
        fit_kwargs=fit_kwargs,
        spec=spec,
    )
    isnothing(derived_weights) && return fit_kwargs

    checked_weights = validate_obs_weight_output(derived_weights, size(X_train, 1))
    final_weights = isnothing(base_weights) ? checked_weights :
        combine_obs_weights(base_weights, checked_weights)

    merge(fit_kwargs, (; obs_weights=final_weights))
end

function validate_obs_weight_output(weights, n_samples::Int)
    weights isa AbstractVector{<:Real} || throw(ArgumentError(
        "obs_weight_fn must return an AbstractVector of real numbers or nothing."))
    length(weights) == n_samples || throw(DimensionMismatch(
        "obs_weight_fn returned $(length(weights)) weights for $n_samples training samples."))
    all(isfinite, weights) || throw(ArgumentError(
        "obs_weight_fn returned non-finite observation weights."))
    all(≥(0), weights) || throw(ArgumentError(
        "obs_weight_fn returned negative observation weights."))
    any(>(0), weights) || throw(ArgumentError(
        "obs_weight_fn returned only zero observation weights."))
    weights isa Vector{Float64} ? weights : Float64.(weights)
end

function combine_obs_weights(
    base_weights::AbstractVector{<:Real},
    derived_weights::AbstractVector{<:Real},
)
    length(base_weights) == length(derived_weights) || throw(DimensionMismatch(
        "obs_weights and obs_weight_fn output must have the same length."))
    combined = Float64.(base_weights) .* Float64.(derived_weights)
    any(>(0), combined) || throw(ArgumentError(
        "Combining obs_weights with obs_weight_fn output produced only zero weights."))
    combined
end

default_da_obs_weight_fn(X_train, Y_train; kwargs...) = invfreqweights(sampleclasses(Y_train))

function with_response_labels(
    fit_kwargs::NamedTuple,
    responselabels::AbstractVector,
)
    haskey(fit_kwargs, :responselabels) && return fit_kwargs
    merge(fit_kwargs, (; responselabels=responselabels))
end

function build_folds(
    n_samples::Integer,
    num_folds::Integer,
    rng::AbstractRNG;
    strata::Union{AbstractVector, Nothing}=nothing,
)
    num_folds ≥ 1 || throw(ArgumentError("Number of folds must be at least 1."))
    num_folds ≤ n_samples || throw(ArgumentError(
        "Number of folds ($num_folds) exceeds number of samples ($n_samples)."))

    if isnothing(strata)
        indices = shuffle(rng, collect(1:n_samples))
        base = fld(n_samples, num_folds)
        extra = n_samples % num_folds
        folds = Vector{Vector{Int}}(undef, num_folds)
        start = 1
        for i in 1:num_folds
            n_take = base + (i ≤ extra ? 1 : 0)
            stop = start + n_take - 1
            folds[i] = indices[start:stop]
            start = stop + 1
        end
        return folds
    end

    length(strata) == n_samples || throw(DimensionMismatch(
        "Length of strata must match the number of samples."))
    random_batch_indices(strata, num_folds, rng)
end

function subset_fit_kwargs(
    fit_kwargs::NamedTuple,
    train_indices::AbstractVector{<:Integer},
    n_samples::Integer,
)
    isempty(fit_kwargs) && return fit_kwargs

    out_pairs = Pair{Symbol, Any}[]
    for (key, value) in pairs(fit_kwargs)
        adjusted = if key in (:obs_weights, :samplelabels, :sampleclasses)
            subset_vector_like(value, train_indices, n_samples, key)
        elseif key === :Yadd
            subset_matrix_like(value, train_indices, n_samples, key)
        else
            value
        end
        push!(out_pairs, key => adjusted)
    end

    (; out_pairs...)
end

function ensure_response_labels(
    fit_kwargs::NamedTuple,
    Y::AbstractMatrix{<:Real},
)
    haskey(fit_kwargs, :responselabels) && return fit_kwargs
    merge(fit_kwargs, (responselabels=string.(1:size(Y, 2)),))
end

function subset_vector_like(
    values,
    train_indices::AbstractVector{<:Integer},
    n_samples::Integer,
    name::Symbol,
)
    isnothing(values) && return values
    values isa AbstractVector || return values

    if length(values) == n_samples
        values[train_indices]
    elseif length(values) == length(train_indices)
        values
    else
        throw(DimensionMismatch(
            "Length of $name must match the total sample count or the number of " *
            "training samples."))
    end
end

function subset_matrix_like(
    values,
    train_indices::AbstractVector{<:Integer},
    n_samples::Integer,
    name::Symbol,
)
    isnothing(values) && return values
    values isa AbstractVector && return subset_vector_like(values, train_indices, n_samples, name)
    values isa AbstractMatrix || return values

    if size(values, 1) == n_samples
        values[train_indices, :]
    elseif size(values, 1) == length(train_indices)
        values
    else
        throw(DimensionMismatch(
            "Row count of $name must match the total sample count or the number of " *
            "training samples."))
    end
end

function with_n_components(m::NCPLSModel, ncomponents::Integer)
    NCPLSModel(
        ncomponents=ncomponents,
        center_X=m.center_X,
        scale_X=m.scale_X,
        center_Yprim=m.center_Yprim,
        multilinear=m.multilinear,
        orthogonalize_mode_weights=m.orthogonalize_mode_weights,
        multilinear_maxiter=m.multilinear_maxiter,
        multilinear_tol=m.multilinear_tol,
        multilinear_init=m.multilinear_init,
        multilinear_seed=m.multilinear_seed,
    )
end

subset_samples(X::AbstractArray, sample_indices::AbstractVector{<:Integer}) =
    view(X, sample_indices, ntuple(_ -> Colon(), ndims(X) - 1)...)
