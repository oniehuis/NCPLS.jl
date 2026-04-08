"""
    analyze_synthetic_multilinear_with_ncpls(
        data;
        model::Union{NCPLSModel, Nothing}=nothing,
        ncomponents::Integer=1,
        multilinear::Bool=false,
        center_X::Bool=true,
        scale_X::Bool=false,
        center_Yprim::Bool=true,
        test_fraction::Real=0.3,
        rng::AbstractRNG=MersenneTwister(1),
        train_idx::Union{AbstractVector{<:Integer}, Nothing}=nothing,
        test_idx::Union{AbstractVector{<:Integer}, Nothing}=nothing,
        verbose::Bool=false,
    )

Fit NCPLS to synthetic multilinear regression data and return train/test predictions,
scores, and regression metrics.
"""
function analyze_synthetic_multilinear_with_ncpls(
    data;
    model::Union{NCPLSModel, Nothing}=nothing,
    ncomponents::Integer=1,
    multilinear::Bool=false,
    center_X::Bool=true,
    scale_X::Bool=false,
    center_Yprim::Bool=true,
    test_fraction::Real=0.3,
    rng::AbstractRNG=MersenneTwister(1),
    train_idx::Union{AbstractVector{<:Integer}, Nothing}=nothing,
    test_idx::Union{AbstractVector{<:Integer}, Nothing}=nothing,
    verbose::Bool=false,
)
    ncomponents > 0 || throw(ArgumentError("ncomponents must be greater than zero"))

    X = data.X
    Yprim = data.Yprim

    ndims(X) ≥ 2 || throw(ArgumentError(
        "data.X must have at least 2 dimensions: samples × variables[/modes]"))
    size(X, 1) == size(Yprim, 1) || throw(DimensionMismatch(
        "data.X and data.Yprim must have the same number of samples"))

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

    ncpls_model = isnothing(model) ? NCPLSModel(
        ncomponents = ncomponents,
        center_X = center_X,
        scale_X = scale_X,
        center_Yprim = center_Yprim,
        multilinear = multilinear,
        orthogonalize_mode_weights = false,
    ) : model

    mf = fit(
        ncpls_model,
        Xtrain,
        Ytrain;
        Yadd = nothing,
        obs_weights = nothing,
        verbose = verbose,
    )

    Yhat_train = components_last_predictions(predict(mf, Xtrain), 2)
    Yhat_test = components_last_predictions(predict(mf, Xtest), 2)
    scores_train = project(mf, Xtrain)
    scores_test = project(mf, Xtest)

    train_metrics = componentwise_regression_metrics(Yhat_train, Ytrain)
    test_metrics = componentwise_regression_metrics(Yhat_test, Ytest)

    (
        ncpls_model = ncpls_model,
        ncplsfit = mf,
        train_idx = train_idx,
        test_idx = test_idx,
        Xtrain = Xtrain,
        Xtest = Xtest,
        Ytrain = Ytrain,
        Ytest = Ytest,
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
    )
end

"""
    compare_synthetic_multilinear_models(
        data;
        unfolded_model::Union{NCPLSModel, Nothing}=nothing,
        multilinear_model::Union{NCPLSModel, Nothing}=nothing,
        ncomponents::Integer=1,
        test_fraction::Real=0.3,
        rng::AbstractRNG=MersenneTwister(1),
        verbose::Bool=false,
    )

Fit unfolded and multilinear NCPLS on the same synthetic multilinear train/test split and
summarize prediction and structure recovery.
"""
function compare_synthetic_multilinear_models(
    data;
    unfolded_model::Union{NCPLSModel, Nothing}=nothing,
    multilinear_model::Union{NCPLSModel, Nothing}=nothing,
    ncomponents::Integer=1,
    test_fraction::Real=0.3,
    rng::AbstractRNG=MersenneTwister(1),
    verbose::Bool=false,
)
    train_idx, test_idx = split_synthetic_gcms_samples(
        size(data.X, 1);
        test_fraction = test_fraction,
        rng = rng,
    )

    unfolded_result = analyze_synthetic_multilinear_with_ncpls(
        data;
        model = unfolded_model,
        ncomponents = ncomponents,
        multilinear = false,
        test_fraction = test_fraction,
        rng = rng,
        train_idx = train_idx,
        test_idx = test_idx,
    )

    multilinear_result = analyze_synthetic_multilinear_with_ncpls(
        data;
        model = multilinear_model,
        ncomponents = ncomponents,
        multilinear = true,
        test_fraction = test_fraction,
        rng = rng,
        train_idx = train_idx,
        test_idx = test_idx,
        verbose = verbose,
    )

    a_common = min(
        size(unfolded_result.Yhat_test, 3),
        size(multilinear_result.Yhat_test, 3),
        size(data.T, 2),
    )

    prediction_rmse_train = Vector{Float64}(undef, a_common)
    prediction_rmse_test = Vector{Float64}(undef, a_common)
    prediction_cor_train = Vector{Float64}(undef, a_common)
    prediction_cor_test = Vector{Float64}(undef, a_common)
    unfolded_true_score_abs_cor_train = Vector{Float64}(undef, a_common)
    unfolded_true_score_abs_cor_test = Vector{Float64}(undef, a_common)
    multilinear_true_score_abs_cor_train = Vector{Float64}(undef, a_common)
    multilinear_true_score_abs_cor_test = Vector{Float64}(undef, a_common)
    multilinear_mode_abs_cor = Matrix{Float64}(undef, length(data.mode_weights), a_common)
    multilinear_recovered_top_mz = Vector{Vector{Int}}(undef, a_common)
    multilinear_mz_overlap = Vector{Int}(undef, a_common)

    for a in 1:a_common
        unfolded_train = vec(@view unfolded_result.Yhat_train[:, :, a])
        unfolded_test = vec(@view unfolded_result.Yhat_test[:, :, a])
        multilinear_train = vec(@view multilinear_result.Yhat_train[:, :, a])
        multilinear_test = vec(@view multilinear_result.Yhat_test[:, :, a])

        prediction_rmse_train[a] = sqrt(mean((unfolded_train .- multilinear_train) .^ 2))
        prediction_rmse_test[a] = sqrt(mean((unfolded_test .- multilinear_test) .^ 2))
        prediction_cor_train[a] = safe_correlation(unfolded_train, multilinear_train)
        prediction_cor_test[a] = safe_correlation(unfolded_test, multilinear_test)

        unfolded_true_score_abs_cor_train[a] = abs(safe_correlation(
            unfolded_result.scores_train[:, a], data.T[train_idx, a]))
        unfolded_true_score_abs_cor_test[a] = abs(safe_correlation(
            unfolded_result.scores_test[:, a], data.T[test_idx, a]))
        multilinear_true_score_abs_cor_train[a] = abs(safe_correlation(
            multilinear_result.scores_train[:, a], data.T[train_idx, a]))
        multilinear_true_score_abs_cor_test[a] = abs(safe_correlation(
            multilinear_result.scores_test[:, a], data.T[test_idx, a]))

        for j in eachindex(data.mode_weights)
            multilinear_mode_abs_cor[j, a] = abs(safe_correlation(
                multilinear_result.ncplsfit.W_modes[j][:, a],
                data.mode_weights[j][:, a],
            ))
        end

        k = length(data.active_mz_channels[a])
        recovered = topk_abs_indices(
            multilinear_result.ncplsfit.W_modes[data.mz_mode][:, a],
            k,
        )
        multilinear_recovered_top_mz[a] = recovered
        multilinear_mz_overlap[a] = length(intersect(recovered, data.active_mz_channels[a]))
    end

    rmse_test_delta = multilinear_result.rmse_test_overall[1:a_common] .-
        unfolded_result.rmse_test_overall[1:a_common]
    r2_test_delta = multilinear_result.r2_test_overall[1:a_common] .-
        unfolded_result.r2_test_overall[1:a_common]

    rmse_winner = [
        comparison_winner_multilinear(
            unfolded_result.rmse_test_overall[a],
            multilinear_result.rmse_test_overall[a];
            lower_is_better = true,
        ) for a in 1:a_common
    ]
    r2_winner = [
        comparison_winner_multilinear(
            unfolded_result.r2_test_overall[a],
            multilinear_result.r2_test_overall[a];
            lower_is_better = false,
        ) for a in 1:a_common
    ]

    (
        train_idx = train_idx,
        test_idx = test_idx,
        unfolded = unfolded_result,
        multilinear = multilinear_result,
        common_ncomponents = a_common,
        prediction_rmse_train = prediction_rmse_train,
        prediction_rmse_test = prediction_rmse_test,
        prediction_cor_train = prediction_cor_train,
        prediction_cor_test = prediction_cor_test,
        unfolded_true_score_abs_cor_train = unfolded_true_score_abs_cor_train,
        unfolded_true_score_abs_cor_test = unfolded_true_score_abs_cor_test,
        multilinear_true_score_abs_cor_train = multilinear_true_score_abs_cor_train,
        multilinear_true_score_abs_cor_test = multilinear_true_score_abs_cor_test,
        multilinear_mode_abs_cor = multilinear_mode_abs_cor,
        true_active_mz_channels = data.active_mz_channels[1:a_common],
        multilinear_recovered_top_mz = multilinear_recovered_top_mz,
        multilinear_mz_overlap = multilinear_mz_overlap,
        predictive_components = data.predictive_components,
        true_q = data.Qtrue,
        rmse_test_delta = rmse_test_delta,
        r2_test_delta = r2_test_delta,
        better_model_test_rmse = rmse_winner,
        better_model_test_r2 = r2_winner,
    )
end

function topk_abs_indices(v::AbstractVector{<:Real}, k::Integer)
    k > 0 || throw(ArgumentError("k must be greater than zero"))
    sortperm(abs.(v); rev = true)[1:min(k, length(v))]
end

function comparison_winner_multilinear(
    unfolded::Real,
    multilinear::Real;
    atol::Real=1e-10,
    lower_is_better::Bool=true,
)
    if isapprox(unfolded, multilinear; atol = atol, rtol = 0)
        return :tie
    elseif lower_is_better
        return unfolded < multilinear ? :unfolded : :multilinear
    else
        return unfolded > multilinear ? :unfolded : :multilinear
    end
end
