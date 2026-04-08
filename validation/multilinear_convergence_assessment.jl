"""
    compare_multilinear_convergence(
        data;
        ncomponents::Integer=1,
        settings::AbstractVector{<:NamedTuple}=[
            (maxiter = 5, tol = 1e-2),
            (maxiter = 20, tol = 1e-4),
            (maxiter = 100, tol = 1e-8),
        ],
        reference_setting::NamedTuple=(maxiter = 500, tol = 1e-10),
        multilinear_init::Symbol=:random,
        multilinear_seed::Int=1,
        test_fraction::Real=0.3,
        rng::AbstractRNG=MersenneTwister(1),
        verbose::Bool=false,
    )

Compare PARAFAC convergence settings on the same synthetic multilinear train/test split.

The returned object contains one strict reference fit and one summary per candidate
`(maxiter, tol)` setting. All runs share the same initialization choice and, for
`multilinear_init = :random`, the same random seed.
"""
function compare_multilinear_convergence(
    data;
    ncomponents::Integer=1,
    settings::AbstractVector{<:NamedTuple}=[
        (maxiter = 5, tol = 1e-2),
        (maxiter = 20, tol = 1e-4),
        (maxiter = 100, tol = 1e-8),
    ],
    reference_setting::NamedTuple=(maxiter = 500, tol = 1e-10),
    multilinear_init::Symbol=:random,
    multilinear_seed::Int=1,
    test_fraction::Real=0.3,
    rng::AbstractRNG=MersenneTwister(1),
    verbose::Bool=false,
)
    ndims(data.X) ≥ 4 || throw(ArgumentError(
        "compare_multilinear_convergence is intended for d ≥ 3 multilinear predictor modes"))
    ncomponents > 0 || throw(ArgumentError("ncomponents must be greater than zero"))
    !isempty(settings) || throw(ArgumentError("settings must not be empty"))
    multilinear_init in (:hosvd, :random) || throw(ArgumentError(
        "multilinear_init must be :hosvd or :random"))

    validate_convergence_setting(reference_setting)
    foreach(validate_convergence_setting, settings)

    train_idx, test_idx = split_synthetic_gcms_samples(
        size(data.X, 1);
        test_fraction = test_fraction,
        rng = rng,
    )

    reference = analyze_synthetic_multilinear_with_ncpls(
        data;
        model = NCPLSModel(
            ncomponents = ncomponents,
            center_X = true,
            scale_X = false,
            center_Yprim = true,
            multilinear = true,
            orthogonalize_mode_weights = false,
            multilinear_maxiter = reference_setting.maxiter,
            multilinear_tol = reference_setting.tol,
            multilinear_init = multilinear_init,
            multilinear_seed = multilinear_seed,
        ),
        ncomponents = ncomponents,
        multilinear = true,
        test_fraction = test_fraction,
        rng = rng,
        train_idx = train_idx,
        test_idx = test_idx,
        verbose = verbose,
    )

    a_common = min(ncomponents, size(data.T, 2), size(reference.scores_test, 2))
    reference_order = best_component_alignment(reference, data, train_idx, a_common)
    reference_summary = merge(
        (
            multilinear_init = multilinear_init,
            multilinear_seed = multilinear_seed,
            maxiter = Int(reference_setting.maxiter),
            tol = Float64(reference_setting.tol),
        ),
        convergence_run_summary(
            reference,
            data,
            reference_order,
            train_idx,
            test_idx,
            a_common,
        ),
    )

    setting_results = Vector{NamedTuple}(undef, length(settings))
    for i in eachindex(settings)
        setting = settings[i]
        result = analyze_synthetic_multilinear_with_ncpls(
            data;
            model = NCPLSModel(
                ncomponents = ncomponents,
                center_X = true,
                scale_X = false,
                center_Yprim = true,
                multilinear = true,
                orthogonalize_mode_weights = false,
                multilinear_maxiter = setting.maxiter,
                multilinear_tol = setting.tol,
                multilinear_init = multilinear_init,
                multilinear_seed = multilinear_seed,
            ),
            ncomponents = ncomponents,
            multilinear = true,
            test_fraction = test_fraction,
            rng = rng,
            train_idx = train_idx,
            test_idx = test_idx,
            verbose = verbose,
        )

        order = best_component_alignment(result, data, train_idx, a_common)

        setting_results[i] = merge(
            (
                multilinear_init = multilinear_init,
                multilinear_seed = multilinear_seed,
                maxiter = Int(setting.maxiter),
                tol = Float64(setting.tol),
            ),
            convergence_run_summary(result, data, order, train_idx, test_idx, a_common),
            (
                pred_rmse_vs_reference = componentwise_prediction_rmse(
                    reference.Yhat_test, result.Yhat_test, a_common),
                score_abs_cor_vs_reference = aligned_cross_score_correlations(
                    reference.scores_test,
                    result.scores_test,
                    reference_order,
                    order,
                    a_common,
                ),
            ),
        )
    end

    rmse_mat = reduce(hcat, [s.rmse_test_overall for s in setting_results])
    r2_mat = reduce(hcat, [s.r2_test_overall for s in setting_results])
    score_mat = reduce(hcat, [s.true_score_abs_cor_test for s in setting_results])
    mode_cat = cat([s.mode_abs_cor for s in setting_results]...; dims = 3)

    (
        train_idx = train_idx,
        test_idx = test_idx,
        reference = reference_summary,
        settings = setting_results,
        rmse_test_mean = vec(mean(rmse_mat; dims = 2)),
        rmse_test_std = vec(std(rmse_mat; dims = 2)),
        r2_test_mean = vec(mean(r2_mat; dims = 2)),
        r2_test_std = vec(std(r2_mat; dims = 2)),
        score_cor_test_mean = vec(mean(score_mat; dims = 2)),
        score_cor_test_std = vec(std(score_mat; dims = 2)),
        mode_cor_mean = mean(mode_cat; dims = 3),
        mode_cor_std = std(mode_cat; dims = 3),
    )
end

function validate_convergence_setting(setting::NamedTuple)
    hasproperty(setting, :maxiter) || throw(ArgumentError(
        "Each convergence setting must define maxiter"))
    hasproperty(setting, :tol) || throw(ArgumentError(
        "Each convergence setting must define tol"))
    setting.maxiter > 0 || throw(ArgumentError("maxiter must be greater than zero"))
    setting.tol ≥ 0 || throw(ArgumentError("tol must be non-negative"))
    nothing
end

function convergence_run_summary(result, data, order, train_idx, test_idx, a_common)
    (
        rmse_test_overall = result.rmse_test_overall[1:a_common],
        r2_test_overall = result.r2_test_overall[1:a_common],
        true_score_abs_cor_train = aligned_score_correlations(
            result.scores_train, data.T[train_idx, :], order),
        true_score_abs_cor_test = aligned_score_correlations(
            result.scores_test, data.T[test_idx, :], order),
        mode_abs_cor = aligned_mode_correlations(result, data, order),
        recovered_top_mz = recovered_top_mz_summary(result, data, order),
        mz_overlap = recovered_mz_overlap_summary(result, data, order),
        parafac_relerr = result.ncplsfit.W_multilinear_relerr[1:a_common],
        component_order = order,
    )
end
