"""
    synthetic_obs_weighted_multilinear_data(;
        nsamples::Integer=100,
        mode_dims::Tuple{Vararg{Int}}=(40, 30),
        ncomponents::Integer=2,
        nresponses::Integer=1,
        predictive_components::Union{Nothing, AbstractVector{<:Integer}}=nothing,
        noisy_fraction::Real=0.30,
        clean_weight::Real=1.0,
        noisy_weight::Real=0.2,
        x_noise_scale_clean::Real=0.05,
        x_noise_scale_noisy::Real=0.40,
        y_noise_scale_clean::Real=0.05,
        y_noise_scale_noisy::Real=0.30,
        rng::AbstractRNG=MersenneTwister(1),
    )

Generate synthetic multilinear regression data with sample-specific noise levels and
observation weights. A subset of samples is marked as noisy and assigned lower weights,
so the data can be used to assess whether weighted NCPLS fitting emphasizes the
higher-quality samples.
"""
function synthetic_obs_weighted_multilinear_data(;
    nsamples::Integer=100,
    mode_dims::Tuple{Vararg{Int}}=(40, 30),
    ncomponents::Integer=2,
    nresponses::Integer=1,
    predictive_components::Union{Nothing, AbstractVector{<:Integer}}=nothing,
    noisy_fraction::Real=0.30,
    clean_weight::Real=1.0,
    noisy_weight::Real=0.2,
    x_noise_scale_clean::Real=0.05,
    x_noise_scale_noisy::Real=0.40,
    y_noise_scale_clean::Real=0.05,
    y_noise_scale_noisy::Real=0.30,
    rng::AbstractRNG=MersenneTwister(1),
)
    0 < noisy_fraction < 1 || throw(ArgumentError(
        "noisy_fraction must lie strictly between 0 and 1"))
    clean_weight > 0 || throw(ArgumentError("clean_weight must be greater than zero"))
    noisy_weight > 0 || throw(ArgumentError("noisy_weight must be greater than zero"))
    x_noise_scale_clean ≥ 0 || throw(ArgumentError(
        "x_noise_scale_clean must be non-negative"))
    x_noise_scale_noisy ≥ 0 || throw(ArgumentError(
        "x_noise_scale_noisy must be non-negative"))
    y_noise_scale_clean ≥ 0 || throw(ArgumentError(
        "y_noise_scale_clean must be non-negative"))
    y_noise_scale_noisy ≥ 0 || throw(ArgumentError(
        "y_noise_scale_noisy must be non-negative"))

    base = synthetic_multilinear_regression_data(
        nsamples = nsamples,
        mode_dims = mode_dims,
        ncomponents = ncomponents,
        nresponses = nresponses,
        predictive_components = predictive_components,
        integer_counts = false,
        x_noise_scale = 0.0,
        y_noise_scale = 0.0,
        rng = rng,
    )

    n_noisy = clamp(round(Int, noisy_fraction * nsamples), 1, nsamples - 1)
    noisy_idx = sort(randperm(rng, nsamples)[1:n_noisy])
    noisy_mask = falses(nsamples)
    noisy_mask[noisy_idx] .= true
    clean_idx = findall(.!noisy_mask)

    obs_weights_raw = ifelse.(noisy_mask, Float64(noisy_weight), Float64(clean_weight))
    obs_weights = obs_weights_raw .* (nsamples / sum(obs_weights_raw))

    X = Array{Float64}(undef, size(base.X))
    Yprim = Matrix{Float64}(undef, size(base.Yclean))
    y_scale_ref = max(std(base.Yclean), eps(Float64))
    tail = ntuple(_ -> Colon(), ndims(base.X) - 1)

    for i in 1:nsamples
        λi = Array(@view base.lambda[i, tail...])
        x_noise_scale_i = noisy_mask[i] ? x_noise_scale_noisy : x_noise_scale_clean
        x_noise_ref = max(mean(λi), eps(Float64))
        Xi = λi .+ x_noise_scale_i * x_noise_ref .* randn(rng, size(λi)...)
        X[i, tail...] = max.(Xi, 0.0)

        y_noise_scale_i = noisy_mask[i] ? y_noise_scale_noisy : y_noise_scale_clean
        Yprim[i, :] = base.Yclean[i, :] .+
            y_noise_scale_i * y_scale_ref .* randn(rng, size(base.Yclean, 2))
    end

    (
        X = X,
        Yprim = Yprim,
        T = base.T,
        Qtrue = base.Qtrue,
        templates = base.templates,
        mode_weights = base.mode_weights,
        component_strengths = base.component_strengths,
        lambda = base.lambda,
        Yclean = base.Yclean,
        mode_dims = base.mode_dims,
        rt_mode = base.rt_mode,
        mz_mode = base.mz_mode,
        rt_peak_centers = base.rt_peak_centers,
        active_mz_channels = base.active_mz_channels,
        predictive_components = base.predictive_components,
        integer_counts = false,
        obs_weights = obs_weights,
        obs_weights_raw = obs_weights_raw,
        noisy_mask = noisy_mask,
        noisy_idx = noisy_idx,
        clean_idx = clean_idx,
        noisy_fraction = noisy_fraction,
        x_noise_scale_clean = x_noise_scale_clean,
        x_noise_scale_noisy = x_noise_scale_noisy,
        y_noise_scale_clean = y_noise_scale_clean,
        y_noise_scale_noisy = y_noise_scale_noisy,
    )
end

"""
    compare_obs_weights_effect(
        data;
        model::Union{NCPLSModel, Nothing}=nothing,
        ncomponents::Integer=1,
        multilinear::Bool=true,
        center_X::Bool=true,
        scale_X::Bool=false,
        center_Yprim::Bool=true,
        test_fraction::Real=0.25,
        rng::AbstractRNG=MersenneTwister(1),
        train_idx::Union{AbstractVector{<:Integer}, Nothing}=nothing,
        test_idx::Union{AbstractVector{<:Integer}, Nothing}=nothing,
        verbose::Bool=false,
    )

Fit NCPLS with and without observation weights on the same synthetic train/test split and
summarize whether weighting improves prediction and recovery on heteroskedastic data.
"""
function compare_obs_weights_effect(
    data;
    model::Union{NCPLSModel, Nothing}=nothing,
    ncomponents::Integer=1,
    multilinear::Bool=true,
    center_X::Bool=true,
    scale_X::Bool=false,
    center_Yprim::Bool=true,
    test_fraction::Real=0.25,
    rng::AbstractRNG=MersenneTwister(1),
    train_idx::Union{AbstractVector{<:Integer}, Nothing}=nothing,
    test_idx::Union{AbstractVector{<:Integer}, Nothing}=nothing,
    verbose::Bool=false,
)
    ncomponents > 0 || throw(ArgumentError("ncomponents must be greater than zero"))
    hasproperty(data, :obs_weights) || throw(ArgumentError(
        "data must provide obs_weights"))

    X = data.X
    Yprim = data.Yprim
    obs_weights = float64(data.obs_weights)

    size(X, 1) == size(Yprim, 1) || throw(DimensionMismatch(
        "data.X and data.Yprim must have the same number of samples"))
    length(obs_weights) == size(X, 1) || throw(DimensionMismatch(
        "Length of data.obs_weights must match the number of samples"))

    train_idx, test_idx = split_synthetic_gcms_samples(
        size(X, 1);
        test_fraction = test_fraction,
        rng = rng,
        train_idx = train_idx,
        test_idx = test_idx,
    )

    tail = ntuple(_ -> Colon(), ndims(X) - 1)
    Xtrain = X[train_idx, tail...]
    Xtest = X[test_idx, tail...]
    Ytrain = Yprim[train_idx, :]
    Ytest = Yprim[test_idx, :]
    obs_weights_train = obs_weights[train_idx]
    obs_weights_test = obs_weights[test_idx]

    ncpls_model = isnothing(model) ? NCPLSModel(
        ncomponents = ncomponents,
        center_X = center_X,
        scale_X = scale_X,
        center_Yprim = center_Yprim,
        multilinear = multilinear,
        orthogonalize_mode_weights = false,
    ) : model

    mf_without = fit(
        ncpls_model,
        Xtrain,
        Ytrain;
        obs_weights = nothing,
        verbose = verbose,
    )

    mf_with = fit(
        ncpls_model,
        Xtrain,
        Ytrain;
        obs_weights = obs_weights_train,
        verbose = verbose,
    )

    without = obs_weight_run_summary(
        mf_without, ncpls_model, data, train_idx, test_idx, Xtrain, Xtest, Ytrain, Ytest,
        nothing, obs_weights_test)
    with = obs_weight_run_summary(
        mf_with, ncpls_model, data, train_idx, test_idx, Xtrain, Xtest, Ytrain, Ytest,
        obs_weights_train, obs_weights_test)

    a_common = min(
        ncomponents,
        size(without.scores_test, 2),
        size(with.scores_test, 2),
        length(data.predictive_components),
    )

    without_true_score_abs_cor_test = Vector{Float64}(undef, a_common)
    with_true_score_abs_cor_test = Vector{Float64}(undef, a_common)
    without_mode_abs_cor = multilinear ? Matrix{Float64}(undef, length(data.mode_weights), a_common) : nothing
    with_mode_abs_cor = multilinear ? Matrix{Float64}(undef, length(data.mode_weights), a_common) : nothing
    without_recovered_top_mz = multilinear ? Vector{Vector{Int}}(undef, a_common) : nothing
    with_recovered_top_mz = multilinear ? Vector{Vector{Int}}(undef, a_common) : nothing
    without_mz_overlap = multilinear ? Vector{Int}(undef, a_common) : nothing
    with_mz_overlap = multilinear ? Vector{Int}(undef, a_common) : nothing

    for a in 1:a_common
        true_a = data.predictive_components[a]
        without_true_score_abs_cor_test[a] = abs(safe_correlation(
            without.scores_test[:, a], data.T[test_idx, true_a]))
        with_true_score_abs_cor_test[a] = abs(safe_correlation(
            with.scores_test[:, a], data.T[test_idx, true_a]))

        if multilinear
            for j in eachindex(data.mode_weights)
                without_mode_abs_cor[j, a] = abs(safe_correlation(
                    mf_without.W_modes[j][:, a], data.mode_weights[j][:, true_a]))
                with_mode_abs_cor[j, a] = abs(safe_correlation(
                    mf_with.W_modes[j][:, a], data.mode_weights[j][:, true_a]))
            end

            k = length(data.active_mz_channels[true_a])
            without_recovered_top_mz[a] = topk_abs_indices(mf_without.W_modes[data.mz_mode][:, a], k)
            with_recovered_top_mz[a] = topk_abs_indices(mf_with.W_modes[data.mz_mode][:, a], k)
            without_mz_overlap[a] = length(intersect(
                without_recovered_top_mz[a], data.active_mz_channels[true_a]))
            with_mz_overlap[a] = length(intersect(
                with_recovered_top_mz[a], data.active_mz_channels[true_a]))
        end
    end

    rmse_test_delta = with.rmse_test_overall[1:a_common] .- without.rmse_test_overall[1:a_common]
    weighted_rmse_test_delta = with.weighted_rmse_test_overall[1:a_common] .-
        without.weighted_rmse_test_overall[1:a_common]
    r2_test_delta = with.r2_test_overall[1:a_common] .- without.r2_test_overall[1:a_common]
    weighted_r2_test_delta = with.weighted_r2_test_overall[1:a_common] .-
        without.weighted_r2_test_overall[1:a_common]

    better_rmse = [
        comparison_winner_weights(
            without.rmse_test_overall[a],
            with.rmse_test_overall[a];
            lower_is_better = true,
        ) for a in 1:a_common
    ]
    better_weighted_rmse = [
        comparison_winner_weights(
            without.weighted_rmse_test_overall[a],
            with.weighted_rmse_test_overall[a];
            lower_is_better = true,
        ) for a in 1:a_common
    ]
    better_r2 = [
        comparison_winner_weights(
            without.r2_test_overall[a],
            with.r2_test_overall[a];
            lower_is_better = false,
        ) for a in 1:a_common
    ]
    better_weighted_r2 = [
        comparison_winner_weights(
            without.weighted_r2_test_overall[a],
            with.weighted_r2_test_overall[a];
            lower_is_better = false,
        ) for a in 1:a_common
    ]

    (
        train_idx = train_idx,
        test_idx = test_idx,
        without_weights = without,
        with_weights = with,
        common_ncomponents = a_common,
        noisy_test_count = count(data.noisy_mask[test_idx]),
        clean_test_count = count(.!data.noisy_mask[test_idx]),
        without_true_score_abs_cor_test = without_true_score_abs_cor_test,
        with_true_score_abs_cor_test = with_true_score_abs_cor_test,
        without_mode_abs_cor = without_mode_abs_cor,
        with_mode_abs_cor = with_mode_abs_cor,
        true_active_mz_channels = multilinear ? data.active_mz_channels[data.predictive_components[1:a_common]] : nothing,
        without_recovered_top_mz = without_recovered_top_mz,
        with_recovered_top_mz = with_recovered_top_mz,
        without_mz_overlap = without_mz_overlap,
        with_mz_overlap = with_mz_overlap,
        rmse_test_delta = rmse_test_delta,
        weighted_rmse_test_delta = weighted_rmse_test_delta,
        r2_test_delta = r2_test_delta,
        weighted_r2_test_delta = weighted_r2_test_delta,
        better_model_test_rmse = better_rmse,
        better_model_weighted_test_rmse = better_weighted_rmse,
        better_model_test_r2 = better_r2,
        better_model_weighted_test_r2 = better_weighted_r2,
    )
end

function obs_weight_run_summary(
    mf::NCPLSFit,
    ncpls_model::NCPLSModel,
    data,
    train_idx::AbstractVector{<:Integer},
    test_idx::AbstractVector{<:Integer},
    Xtrain,
    Xtest,
    Ytrain::AbstractMatrix{<:Real},
    Ytest::AbstractMatrix{<:Real},
    obs_weights_train::Union{Nothing, AbstractVector{<:Real}},
    obs_weights_test::AbstractVector{<:Real},
)
    Yhat_train = components_last_predictions(predict(mf, Xtrain), 2)
    Yhat_test = components_last_predictions(predict(mf, Xtest), 2)
    scores_train = project(mf, Xtrain)
    scores_test = project(mf, Xtest)

    train_metrics = componentwise_regression_metrics(Yhat_train, Ytrain)
    test_metrics = componentwise_regression_metrics(Yhat_test, Ytest)
    weighted_train = weighted_componentwise_regression_metrics(
        Yhat_train, Ytrain, isnothing(obs_weights_train) ? ones(size(Ytrain, 1)) : obs_weights_train)
    weighted_test = weighted_componentwise_regression_metrics(Yhat_test, Ytest, obs_weights_test)

    clean_test_idx_local = findall(.!data.noisy_mask[test_idx])
    noisy_test_idx_local = findall(data.noisy_mask[test_idx])

    (
        ncpls_model = ncpls_model,
        ncplsfit = mf,
        train_idx = train_idx,
        test_idx = test_idx,
        Xtrain = Xtrain,
        Xtest = Xtest,
        Ytrain = Ytrain,
        Ytest = Ytest,
        obs_weights_train = obs_weights_train,
        obs_weights_test = obs_weights_test,
        scores_train = scores_train,
        scores_test = scores_test,
        Yhat_train = Yhat_train,
        Yhat_test = Yhat_test,
        rmse_train = train_metrics.rmse,
        rmse_test = test_metrics.rmse,
        r2_train = train_metrics.r2,
        r2_test = test_metrics.r2,
        rmse_train_overall = train_metrics.rmse_overall,
        rmse_test_overall = test_metrics.rmse_overall,
        r2_train_overall = train_metrics.r2_overall,
        r2_test_overall = test_metrics.r2_overall,
        weighted_rmse_train_overall = weighted_train.rmse_overall,
        weighted_rmse_test_overall = weighted_test.rmse_overall,
        weighted_r2_train_overall = weighted_train.r2_overall,
        weighted_r2_test_overall = weighted_test.r2_overall,
        clean_rmse_test_overall = subset_rmse_overall(Yhat_test, Ytest, clean_test_idx_local),
        noisy_rmse_test_overall = subset_rmse_overall(Yhat_test, Ytest, noisy_test_idx_local),
    )
end

function weighted_componentwise_regression_metrics(
    Yhat_components_last::AbstractArray{<:Real, 3},
    Y::AbstractMatrix{<:Real},
    obs_weights::AbstractVector{<:Real},
)
    n, m, a = size(Yhat_components_last)
    size(Y) == (n, m) || throw(DimensionMismatch(
        "Yhat and Y must agree in sample and response dimensions"))
    length(obs_weights) == n || throw(DimensionMismatch(
        "obs_weights must match the number of samples"))

    w = float64(obs_weights)
    wsum = sum(w)
    wsum > 0 || throw(ArgumentError("obs_weights must sum to a positive value"))

    weighted_mean = vec(sum(Y .* reshape(w, :, 1), dims = 1) / wsum)
    sst_total = sum(reshape(w, :, 1) .* (Y .- reshape(weighted_mean, 1, :)) .^ 2)

    rmse_overall = Vector{Float64}(undef, a)
    r2_overall = Vector{Float64}(undef, a)

    for i in 1:a
        pred = @view Yhat_components_last[:, :, i]
        err = pred .- Y
        sse = sum(reshape(w, :, 1) .* err .^ 2)
        rmse_overall[i] = sqrt(sse / (wsum * m))
        r2_overall[i] = sst_total > 0 ? 1 - sse / sst_total : NaN
    end

    (
        rmse_overall = rmse_overall,
        r2_overall = r2_overall,
    )
end

function subset_rmse_overall(
    Yhat_components_last::AbstractArray{<:Real, 3},
    Y::AbstractMatrix{<:Real},
    idx::AbstractVector{<:Integer},
)
    a = size(Yhat_components_last, 3)
    isempty(idx) && return fill(NaN, a)

    rmse = Vector{Float64}(undef, a)
    for i in 1:a
        pred = @view Yhat_components_last[idx, :, i]
        err = pred .- Y[idx, :]
        rmse[i] = sqrt(mean(err .^ 2))
    end
    rmse
end

function comparison_winner_weights(
    without::Real,
    with::Real;
    atol::Real=1e-10,
    lower_is_better::Bool=true,
)
    if isapprox(without, with; atol = atol, rtol = 0)
        return :tie
    elseif lower_is_better
        return without < with ? :without_weights : :with_weights
    else
        return without > with ? :without_weights : :with_weights
    end
end
