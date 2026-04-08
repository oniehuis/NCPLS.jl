"""
    compare_synthetic_gcms_models(
        data;
        cppls_model::Union{CPPLS.CPPLSModel, Nothing}=nothing,
        ncpls_model::Union{NCPLSModel, Nothing}=nothing,
        ncomponents::Integer=2,
        ncpls_multilinear::Bool=false,
        test_fraction::Real=0.3,
        rng::AbstractRNG=MersenneTwister(1),
        verbose::Bool=false,
    )

Fit unfolded CPPLS and NCPLS on the same synthetic GC-MS train/test split and summarize
their agreement.

The returned object contains the individual analysis results together with overall RMSE
and R² differences, per-component prediction agreement, and absolute score
correlations. When `ncpls_model` is not supplied, a comparison model is constructed with
the CPPLS-compatible response preprocessing convention `center_Yprim=false`.
"""
function compare_synthetic_gcms_models(
    data;
    cppls_model::Union{CPPLS.CPPLSModel, Nothing}=nothing,
    ncpls_model::Union{NCPLSModel, Nothing}=nothing,
    ncomponents::Integer=2,
    ncpls_multilinear::Bool=false,
    test_fraction::Real=0.3,
    rng::AbstractRNG=MersenneTwister(1),
    verbose::Bool=false,
)
    train_idx, test_idx = split_synthetic_gcms_samples(
        size(data.X, 1);
        test_fraction = test_fraction,
        rng = rng,
    )

    cppls_result = analyze_synthetic_gcms_with_cppls(
        data;
        model = cppls_model,
        ncomponents = ncomponents,
        test_fraction = test_fraction,
        rng = rng,
        train_idx = train_idx,
        test_idx = test_idx,
    )

    comparison_ncpls_model = isnothing(ncpls_model) ? NCPLSModel(
        ncomponents = cppls_result.cppls_model.ncomponents,
        center_X = cppls_result.cppls_model.center_X,
        scale_X = cppls_result.cppls_model.scale_X,
        center_Yprim = false,
        multilinear = ncpls_multilinear,
        orthogonalize_mode_weights = false,
    ) : ncpls_model

    ncpls_result = analyze_synthetic_gcms_with_ncpls(
        data;
        model = comparison_ncpls_model,
        ncomponents = ncomponents,
        multilinear = ncpls_multilinear,
        test_fraction = test_fraction,
        rng = rng,
        train_idx = train_idx,
        test_idx = test_idx,
        verbose = verbose,
    )

    a_cppls = size(cppls_result.Yhat_test, 3)
    a_ncpls = size(ncpls_result.Yhat_test, 3)
    a_common = min(a_cppls, a_ncpls)

    prediction_rmse_train = Vector{Float64}(undef, a_common)
    prediction_rmse_test = Vector{Float64}(undef, a_common)
    prediction_cor_train = Vector{Float64}(undef, a_common)
    prediction_cor_test = Vector{Float64}(undef, a_common)
    score_abs_cor_train = Vector{Float64}(undef, a_common)
    score_abs_cor_test = Vector{Float64}(undef, a_common)

    for a in 1:a_common
        cp_train = vec(@view cppls_result.Yhat_train[:, :, a])
        cp_test = vec(@view cppls_result.Yhat_test[:, :, a])
        nc_train = vec(@view ncpls_result.Yhat_train[:, :, a])
        nc_test = vec(@view ncpls_result.Yhat_test[:, :, a])

        prediction_rmse_train[a] = sqrt(mean((cp_train .- nc_train) .^ 2))
        prediction_rmse_test[a] = sqrt(mean((cp_test .- nc_test) .^ 2))
        prediction_cor_train[a] = safe_correlation(cp_train, nc_train)
        prediction_cor_test[a] = safe_correlation(cp_test, nc_test)

        score_abs_cor_train[a] = abs(safe_correlation(
            cppls_result.scores_train[:, a], ncpls_result.scores_train[:, a]))
        score_abs_cor_test[a] = abs(safe_correlation(
            cppls_result.scores_test[:, a], ncpls_result.scores_test[:, a]))
    end

    rmse_test_delta = ncpls_result.rmse_test_overall[1:a_common] .-
        cppls_result.rmse_test_overall[1:a_common]
    r2_test_delta = ncpls_result.r2_test_overall[1:a_common] .-
        cppls_result.r2_test_overall[1:a_common]

    rmse_winner = [
        comparison_winner(
            cppls_result.rmse_test_overall[a],
            ncpls_result.rmse_test_overall[a];
            lower_is_better = true,
        ) for a in 1:a_common
    ]
    r2_winner = [
        comparison_winner(
            cppls_result.r2_test_overall[a],
            ncpls_result.r2_test_overall[a];
            lower_is_better = false,
        ) for a in 1:a_common
    ]

    (
        train_idx = train_idx,
        test_idx = test_idx,
        cppls = cppls_result,
        ncpls = ncpls_result,
        common_ncomponents = a_common,
        prediction_rmse_train = prediction_rmse_train,
        prediction_rmse_test = prediction_rmse_test,
        prediction_cor_train = prediction_cor_train,
        prediction_cor_test = prediction_cor_test,
        score_abs_cor_train = score_abs_cor_train,
        score_abs_cor_test = score_abs_cor_test,
        rmse_test_delta = rmse_test_delta,
        r2_test_delta = r2_test_delta,
        better_model_test_rmse = rmse_winner,
        better_model_test_r2 = r2_winner,
    )
end
