"""
    compare_multilinear_init(
        data;
        ncomponents::Integer=1,
        random_seeds::AbstractVector{<:Integer}=[1, 2, 3, 4, 5],
        test_fraction::Real=0.3,
        rng::AbstractRNG=MersenneTwister(1),
        multilinear_maxiter::Int=500,
        multilinear_tol::Float64=1e-10,
        verbose::Bool=false,
    )

Compare `multilinear_init = :hosvd` against repeated `:random` starts on the same
synthetic multilinear train/test split.

This utility is intended primarily for the `d ≥ 3` PARAFAC branch, where initialization
may affect convergence to local optima.
"""
function compare_multilinear_init(
    data;
    ncomponents::Integer=1,
    random_seeds::AbstractVector{<:Integer}=[1, 2, 3, 4, 5],
    test_fraction::Real=0.3,
    rng::AbstractRNG=MersenneTwister(1),
    multilinear_maxiter::Int=500,
    multilinear_tol::Float64=1e-10,
    verbose::Bool=false,
)
    ndims(data.X) ≥ 4 || throw(ArgumentError(
        "compare_multilinear_init is intended for d ≥ 3 multilinear predictor modes"))
    ncomponents > 0 || throw(ArgumentError("ncomponents must be greater than zero"))
    !isempty(random_seeds) || throw(ArgumentError("random_seeds must not be empty"))

    train_idx, test_idx = split_synthetic_gcms_samples(
        size(data.X, 1);
        test_fraction = test_fraction,
        rng = rng,
    )

    hosvd_result = analyze_synthetic_multilinear_with_ncpls(
        data;
        model = NCPLSModel(
            ncomponents = ncomponents,
            center_X = true,
            scale_X = false,
            center_Yprim = true,
            multilinear = true,
            orthogonalize_mode_weights = false,
            multilinear_maxiter = multilinear_maxiter,
            multilinear_tol = multilinear_tol,
            multilinear_init = :hosvd,
            multilinear_seed = 1,
        ),
        ncomponents = ncomponents,
        multilinear = true,
        test_fraction = test_fraction,
        rng = rng,
        train_idx = train_idx,
        test_idx = test_idx,
        verbose = verbose,
    )

    random_results = [
        analyze_synthetic_multilinear_with_ncpls(
            data;
            model = NCPLSModel(
                ncomponents = ncomponents,
                center_X = true,
                scale_X = false,
                center_Yprim = true,
                multilinear = true,
                orthogonalize_mode_weights = false,
                multilinear_maxiter = multilinear_maxiter,
                multilinear_tol = multilinear_tol,
                multilinear_init = :random,
                multilinear_seed = seed,
            ),
            ncomponents = ncomponents,
            multilinear = true,
            test_fraction = test_fraction,
            rng = rng,
            train_idx = train_idx,
            test_idx = test_idx,
            verbose = verbose,
        ) for seed in Int.(collect(random_seeds))
    ]

    a_common = min(
        ncomponents,
        size(data.T, 2),
        size(hosvd_result.scores_test, 2),
        minimum(size(rr.scores_test, 2) for rr in random_results),
    )

    hosvd_order = best_component_alignment(hosvd_result, data, train_idx, a_common)
    random_orders = [
        best_component_alignment(rr, data, train_idx, a_common) for rr in random_results
    ]

    hosvd_summary = init_run_summary(hosvd_result, data, hosvd_order, train_idx, test_idx, a_common)
    random_summaries = [
        merge(
            (seed = Int(random_seeds[i]),),
            init_run_summary(random_results[i], data, random_orders[i], train_idx, test_idx, a_common),
        ) for i in eachindex(random_results)
    ]

    random_test_rmse = [summary.rmse_test_overall for summary in random_summaries]
    random_test_r2 = [summary.r2_test_overall for summary in random_summaries]
    random_score_cor_test = [summary.true_score_abs_cor_test for summary in random_summaries]
    random_mode_cor = [summary.mode_abs_cor for summary in random_summaries]
    random_pred_rmse_vs_hosvd = [
        componentwise_prediction_rmse(hosvd_result.Yhat_test, random_results[i].Yhat_test, a_common)
        for i in eachindex(random_results)
    ]
    random_score_abs_cor_vs_hosvd = [
        aligned_cross_score_correlations(
            hosvd_result.scores_test,
            random_results[i].scores_test,
            hosvd_order,
            random_orders[i],
            a_common,
        ) for i in eachindex(random_results)
    ]

    (
        train_idx = train_idx,
        test_idx = test_idx,
        hosvd = merge((seed = 1,), hosvd_summary),
        random = random_summaries,
        random_seeds = Int.(collect(random_seeds)),
        random_test_rmse_mean = vec(mean(reduce(hcat, random_test_rmse); dims = 2)),
        random_test_rmse_std = vec(std(reduce(hcat, random_test_rmse); dims = 2)),
        random_test_r2_mean = vec(mean(reduce(hcat, random_test_r2); dims = 2)),
        random_test_r2_std = vec(std(reduce(hcat, random_test_r2); dims = 2)),
        random_score_cor_test_mean = vec(mean(reduce(hcat, random_score_cor_test); dims = 2)),
        random_score_cor_test_std = vec(std(reduce(hcat, random_score_cor_test); dims = 2)),
        random_mode_cor_mean = mean(cat(random_mode_cor...; dims = 3); dims = 3),
        random_mode_cor_std = std(cat(random_mode_cor...; dims = 3); dims = 3),
        random_pred_rmse_vs_hosvd = random_pred_rmse_vs_hosvd,
        random_score_abs_cor_vs_hosvd = random_score_abs_cor_vs_hosvd,
    )
end

function init_run_summary(result, data, order, train_idx, test_idx, a_common)
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
        component_order = order,
    )
end

function best_component_alignment(result, data, train_idx, a_common)
    perms = component_orders(size(data.T, 2), a_common)
    best_order = collect(1:a_common)
    best_score = -Inf

    for order in perms
        score = 0.0
        for a in 1:a_common
            score += abs(safe_correlation(result.scores_train[:, a], data.T[train_idx, order[a]]))
            for j in eachindex(data.mode_weights)
                score += abs(safe_correlation(
                    result.ncplsfit.W_modes[j][:, a],
                    data.mode_weights[j][:, order[a]],
                ))
            end
        end

        if score > best_score
            best_score = score
            best_order = collect(order)
        end
    end

    best_order
end

function component_orders(n::Integer, k::Integer)
    0 ≤ k ≤ n || throw(ArgumentError("k must satisfy 0 ≤ k ≤ n"))
    out = Vector{Vector{Int}}()
    current = Int[]
    used = falses(n)

    function backtrack!()
        if length(current) == k
            push!(out, copy(current))
            return
        end

        for item in 1:n
            if !used[item]
                push!(current, item)
                used[item] = true
                backtrack!()
                used[item] = false
                pop!(current)
            end
        end
    end

    backtrack!()
    out
end

function componentwise_prediction_rmse(
    Yhat1::AbstractArray{<:Real, 3},
    Yhat2::AbstractArray{<:Real, 3},
    a_common::Integer,
)
    [sqrt(mean((vec(@view Yhat1[:, :, a]) .- vec(@view Yhat2[:, :, a])) .^ 2))
     for a in 1:a_common]
end

function aligned_cross_score_correlations(
    scores1::AbstractMatrix{<:Real},
    scores2::AbstractMatrix{<:Real},
    order1::AbstractVector{<:Integer},
    order2::AbstractVector{<:Integer},
    a_common::Integer,
)
    corrs = Vector{Float64}(undef, a_common)
    for a in 1:a_common
        truth_idx = order1[a]
        b = findfirst(==(truth_idx), order2)
        corrs[a] = isnothing(b) ? NaN : abs(safe_correlation(scores1[:, a], scores2[:, b]))
    end
    corrs
end

function recovered_top_mz_summary(result, data, order::AbstractVector{<:Integer})
    [
        topk_abs_indices(
            result.ncplsfit.W_modes[data.mz_mode][:, a],
            length(data.active_mz_channels[order[a]]),
        ) for a in eachindex(order)
    ]
end

function recovered_mz_overlap_summary(result, data, order::AbstractVector{<:Integer})
    [
        length(intersect(
            topk_abs_indices(
                result.ncplsfit.W_modes[data.mz_mode][:, a],
                length(data.active_mz_channels[order[a]]),
            ),
            data.active_mz_channels[order[a]],
        )) for a in eachindex(order)
    ]
end
