"""
    synthetic_mode_orthogonality_data(;
        nsamples::Integer=100,
        mode_dims::Tuple{Vararg{Int}}=(40, 30),
        orthogonal_truth::Bool=true,
        rt_mode::Union{Nothing, Integer}=length(mode_dims) >= 2 ? 1 : nothing,
        mz_mode::Integer=length(mode_dims),
        baseline::Real=0.0,
        integer_counts::Bool=false,
        x_noise_scale::Real=0.05,
        y_noise_scale::Real=0.05,
        active_mz_per_component::Integer=4,
        rng::AbstractRNG=MersenneTwister(1),
    )

Generate two-component multilinear regression data for assessing
`orthogonalize_mode_weights`.

When `orthogonal_truth=true`, the true mode vectors are orthogonal within each mode.
When `orthogonal_truth=false`, the true mode vectors overlap within each mode so that
mode-wise orthogonalization becomes an explicit modeling restriction.
"""
function synthetic_mode_orthogonality_data(;
    nsamples::Integer=100,
    mode_dims::Tuple{Vararg{Int}}=(40, 30),
    orthogonal_truth::Bool=true,
    rt_mode::Union{Nothing, Integer}=length(mode_dims) >= 2 ? 1 : nothing,
    mz_mode::Integer=length(mode_dims),
    baseline::Real=0.0,
    integer_counts::Bool=false,
    x_noise_scale::Real=0.05,
    y_noise_scale::Real=0.05,
    active_mz_per_component::Integer=4,
    rng::AbstractRNG=MersenneTwister(1),
)
    nsamples > 1 || throw(ArgumentError("nsamples must be greater than one"))
    length(mode_dims) ≥ 2 || throw(ArgumentError(
        "mode_dims must contain at least two predictor modes"))
    all(d -> d > 3, mode_dims) || throw(ArgumentError(
        "All predictor mode dimensions must be greater than three"))
    1 ≤ mz_mode ≤ length(mode_dims) || throw(ArgumentError(
        "mz_mode must index an existing predictor mode"))
    if !isnothing(rt_mode)
        1 ≤ rt_mode ≤ length(mode_dims) || throw(ArgumentError(
            "rt_mode must index an existing predictor mode"))
        rt_mode == mz_mode && throw(ArgumentError(
            "rt_mode and mz_mode must differ"))
    end
    baseline ≥ 0 || throw(ArgumentError("baseline must be non-negative"))
    x_noise_scale ≥ 0 || throw(ArgumentError("x_noise_scale must be non-negative"))
    y_noise_scale ≥ 0 || throw(ArgumentError("y_noise_scale must be non-negative"))
    active_mz_per_component > 1 || throw(ArgumentError(
        "active_mz_per_component must be greater than one"))

    ncomponents = 2
    d = length(mode_dims)
    mode_weights = [Matrix{Float64}(undef, mode_dims[j], ncomponents) for j in 1:d]
    active_mz_channels = Vector{Vector{Int}}(undef, ncomponents)
    templates = Array{Float64}(undef, mode_dims..., ncomponents)

    for j in 1:d
        if !isnothing(rt_mode) && j == rt_mode
            pair = orthogonal_truth ?
                orthogonal_rt_pair(mode_dims[j], rng) :
                overlapping_rt_pair(mode_dims[j], rng)
            mode_weights[j][:, 1] = pair[1]
            mode_weights[j][:, 2] = pair[2]
        elseif j == mz_mode
            if orthogonal_truth
                pair, channels = orthogonal_mz_pair(
                    mode_dims[j],
                    min(active_mz_per_component, max(2, fld(mode_dims[j], 2))),
                    rng,
                )
            else
                pair, channels = overlapping_mz_pair(
                    mode_dims[j],
                    min(active_mz_per_component, mode_dims[j] - 1),
                    rng,
                )
            end
            mode_weights[j][:, 1] = pair[1]
            mode_weights[j][:, 2] = pair[2]
            active_mz_channels[1] = channels[1]
            active_mz_channels[2] = channels[2]
        else
            pair = orthogonal_truth ?
                orthogonal_auxiliary_pair(mode_dims[j], rng) :
                overlapping_auxiliary_pair(mode_dims[j], rng)
            mode_weights[j][:, 1] = pair[1]
            mode_weights[j][:, 2] = pair[2]
        end
    end

    for a in 1:ncomponents
        factors = [mode_weights[j][:, a] for j in 1:d]
        templates[ntuple(_ -> Colon(), d)..., a] .= outer_tensor(factors)
    end

    T = Matrix{Float64}(undef, nsamples, ncomponents)
    T[:, 1] = exp.(0.35 .* randn(rng, nsamples)) .* (1 .+ 2 .* rand(rng, nsamples))
    T[:, 2] = exp.(0.35 .* randn(rng, nsamples)) .* (1 .+ 1.5 .* rand(rng, nsamples))

    component_strengths = [7.0, 5.5]
    lambda = Array{Float64}(undef, nsamples, mode_dims...)
    X = integer_counts ? Array{Int}(undef, nsamples, mode_dims...) :
        Array{Float64}(undef, nsamples, mode_dims...)

    for i in 1:nsamples
        λi = fill(Float64(baseline), mode_dims...)
        for a in 1:ncomponents
            λi .+= T[i, a] .* component_strengths[a] .* view(
                templates, ntuple(_ -> Colon(), d)..., a)
        end
        if x_noise_scale > 0
            λi .*= exp.(x_noise_scale .* randn(rng, mode_dims...))
        end
        lambda[i, ntuple(_ -> Colon(), d)...] = λi

        if integer_counts
            for I in CartesianIndices(λi)
                X[(i, Tuple(I)...)...] = sample_poisson_count(rng, λi[I])
            end
        else
            Xi = λi .+ x_noise_scale .* mean(λi) .* randn(rng, mode_dims...)
            X[i, ntuple(_ -> Colon(), d)...] = Xi
        end
    end

    Qtrue = reshape([1.0, 0.8], ncomponents, 1)
    Yclean = T * Qtrue
    Yprim = copy(Yclean)
    if y_noise_scale > 0
        Yprim .+= y_noise_scale .* std(Yclean) .* randn(rng, size(Yclean)...)
    end

    (
        X = X,
        Yprim = Yprim,
        T = T,
        Qtrue = Qtrue,
        templates = templates,
        mode_weights = mode_weights,
        lambda = lambda,
        orthogonal_truth = orthogonal_truth,
        rt_mode = rt_mode,
        mz_mode = mz_mode,
        active_mz_channels = active_mz_channels,
        true_mode_abs_inner_products = [
            abs(dot(mode_weights[j][:, 1], mode_weights[j][:, 2]))
            for j in 1:d
        ],
    )
end

"""
    compare_orthogonalize_mode_weights(
        data;
        test_fraction::Real=0.3,
        rng::AbstractRNG=MersenneTwister(1),
        verbose::Bool=false,
    )

Fit multilinear NCPLS with and without `orthogonalize_mode_weights` on the same
synthetic train/test split and summarize prediction, component recovery, and the actual
orthogonality of the stored mode weights.
"""
function compare_orthogonalize_mode_weights(
    data;
    test_fraction::Real=0.3,
    rng::AbstractRNG=MersenneTwister(1),
    verbose::Bool=false,
)
    size(data.T, 2) == 2 || throw(ArgumentError(
        "compare_orthogonalize_mode_weights currently expects two true components"))

    train_idx, test_idx = split_synthetic_gcms_samples(
        size(data.X, 1);
        test_fraction = test_fraction,
        rng = rng,
    )

    without_result = analyze_synthetic_multilinear_with_ncpls(
        data;
        model = NCPLSModel(
            ncomponents = 2,
            center_X = true,
            scale_X = false,
            center_Yprim = true,
            multilinear = true,
            orthogonalize_mode_weights = false,
        ),
        ncomponents = 2,
        multilinear = true,
        test_fraction = test_fraction,
        rng = rng,
        train_idx = train_idx,
        test_idx = test_idx,
        verbose = verbose,
    )

    with_result = analyze_synthetic_multilinear_with_ncpls(
        data;
        model = NCPLSModel(
            ncomponents = 2,
            center_X = true,
            scale_X = false,
            center_Yprim = true,
            multilinear = true,
            orthogonalize_mode_weights = true,
        ),
        ncomponents = 2,
        multilinear = true,
        test_fraction = test_fraction,
        rng = rng,
        train_idx = train_idx,
        test_idx = test_idx,
        verbose = verbose,
    )

    without_order = best_two_component_alignment(without_result, data, train_idx)
    with_order = best_two_component_alignment(with_result, data, train_idx)

    without_mode_abs_cor = aligned_mode_correlations(without_result, data, without_order)
    with_mode_abs_cor = aligned_mode_correlations(with_result, data, with_order)

    without_true_score_abs_cor_train = aligned_score_correlations(
        without_result.scores_train, data.T[train_idx, :], without_order)
    without_true_score_abs_cor_test = aligned_score_correlations(
        without_result.scores_test, data.T[test_idx, :], without_order)
    with_true_score_abs_cor_train = aligned_score_correlations(
        with_result.scores_train, data.T[train_idx, :], with_order)
    with_true_score_abs_cor_test = aligned_score_correlations(
        with_result.scores_test, data.T[test_idx, :], with_order)

    without_recovered_top_mz, without_mz_overlap = recovered_mz_summary(
        without_result, data, without_order)
    with_recovered_top_mz, with_mz_overlap = recovered_mz_summary(
        with_result, data, with_order)

    rmse_test_delta = with_result.rmse_test_overall .- without_result.rmse_test_overall
    r2_test_delta = with_result.r2_test_overall .- without_result.r2_test_overall

    rmse_winner = [
        comparison_winner_orthogonalization(
            without_result.rmse_test_overall[a],
            with_result.rmse_test_overall[a];
            lower_is_better = true,
        ) for a in eachindex(rmse_test_delta)
    ]
    r2_winner = [
        comparison_winner_orthogonalization(
            without_result.r2_test_overall[a],
            with_result.r2_test_overall[a];
            lower_is_better = false,
        ) for a in eachindex(r2_test_delta)
    ]

    (
        train_idx = train_idx,
        test_idx = test_idx,
        without_orthogonalization = without_result,
        with_orthogonalization = with_result,
        true_mode_abs_inner_products = data.true_mode_abs_inner_products,
        without_mode_abs_inner_products = [
            abs(dot(without_result.ncplsfit.W_modes[j][:, 1], without_result.ncplsfit.W_modes[j][:, 2]))
            for j in eachindex(without_result.ncplsfit.W_modes)
        ],
        with_mode_abs_inner_products = [
            abs(dot(with_result.ncplsfit.W_modes[j][:, 1], with_result.ncplsfit.W_modes[j][:, 2]))
            for j in eachindex(with_result.ncplsfit.W_modes)
        ],
        without_component_order = without_order,
        with_component_order = with_order,
        without_true_score_abs_cor_train = without_true_score_abs_cor_train,
        without_true_score_abs_cor_test = without_true_score_abs_cor_test,
        with_true_score_abs_cor_train = with_true_score_abs_cor_train,
        with_true_score_abs_cor_test = with_true_score_abs_cor_test,
        without_mode_abs_cor = without_mode_abs_cor,
        with_mode_abs_cor = with_mode_abs_cor,
        true_active_mz_channels = data.active_mz_channels,
        without_recovered_top_mz = without_recovered_top_mz,
        with_recovered_top_mz = with_recovered_top_mz,
        without_mz_overlap = without_mz_overlap,
        with_mz_overlap = with_mz_overlap,
        rmse_test_delta = rmse_test_delta,
        r2_test_delta = r2_test_delta,
        better_model_test_rmse = rmse_winner,
        better_model_test_r2 = r2_winner,
    )
end

function orthogonal_rt_pair(n::Integer, rng::AbstractRNG)
    width = max(2, fld(n, 5))
    gap = max(1, fld(n, 8))
    max_start1 = max(1, n - 2 * width - gap + 1)
    start1 = rand(rng, 1:max_start1)
    start2 = min(n - width + 1, start1 + width + gap)
    v1 = zeros(Float64, n)
    v2 = zeros(Float64, n)
    v1[start1:(start1 + width - 1)] = 0.5 .+ rand(rng, width)
    v2[start2:(start2 + width - 1)] = 0.5 .+ rand(rng, width)
    v1 ./ norm(v1), v2 ./ norm(v2)
end

function overlapping_rt_pair(n::Integer, rng::AbstractRNG)
    axis = collect(1:n)
    center1 = 0.35 * n + 0.10 * n * rand(rng)
    center2 = center1 + 0.12 * n + 0.05 * n * rand(rng)
    width1 = 0.10 * n + 0.04 * n * rand(rng)
    width2 = 0.10 * n + 0.04 * n * rand(rng)
    v1 = exp.(-0.5 .* ((axis .- center1) ./ width1) .^ 2)
    v2 = exp.(-0.5 .* ((axis .- center2) ./ width2) .^ 2)
    v1 ./ norm(v1), v2 ./ norm(v2)
end

function orthogonal_mz_pair(n::Integer, nactive::Integer, rng::AbstractRNG)
    2 * nactive ≤ n || throw(ArgumentError(
        "nactive must allow two disjoint channel sets"))
    perm = randperm(rng, n)
    active1 = sort(perm[1:nactive])
    active2 = sort(perm[(nactive + 1):(2 * nactive)])
    v1 = zeros(Float64, n)
    v2 = zeros(Float64, n)
    v1[active1] .= 0.5 .+ rand(rng, nactive)
    v2[active2] .= 0.5 .+ rand(rng, nactive)
    (v1 ./ norm(v1), v2 ./ norm(v2)), (active1, active2)
end

function overlapping_mz_pair(n::Integer, nactive::Integer, rng::AbstractRNG)
    nactive < n || throw(ArgumentError(
        "nactive must be smaller than the number of channels"))
    nshared = max(1, fld(nactive, 2))
    nunique = nactive - nshared
    perm = randperm(rng, n)
    shared = sort(perm[1:nshared])
    unique1 = sort(perm[(nshared + 1):(nshared + nunique)])
    unique2 = sort(perm[(nshared + nunique + 1):(nshared + 2 * nunique)])
    active1 = sort(vcat(shared, unique1))
    active2 = sort(vcat(shared, unique2))
    v1 = zeros(Float64, n)
    v2 = zeros(Float64, n)
    v1[active1] .= 0.5 .+ rand(rng, length(active1))
    v2[active2] .= 0.5 .+ rand(rng, length(active2))
    (v1 ./ norm(v1), v2 ./ norm(v2)), (active1, active2)
end

function orthogonal_auxiliary_pair(n::Integer, rng::AbstractRNG)
    Q, _ = qr(randn(rng, n, 2))
    Matrix(Q)[:, 1], Matrix(Q)[:, 2]
end

function overlapping_auxiliary_pair(n::Integer, rng::AbstractRNG)
    u = randn(rng, n)
    u ./= norm(u)
    z = randn(rng, n)
    z .-= dot(z, u) .* u
    z ./= norm(z)
    v = 0.65 .* u .+ sqrt(1 - 0.65^2) .* z
    u, v ./ norm(v)
end

function best_two_component_alignment(result, data, train_idx)
    orders = ([1, 2], [2, 1])
    best_order = orders[1]
    best_score = -Inf

    for order in orders
        score = 0.0
        for a in 1:2
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
            best_order = order
        end
    end

    collect(best_order)
end

function aligned_mode_correlations(result, data, order::AbstractVector{<:Integer})
    d = length(data.mode_weights)
    A = length(order)
    corrs = Matrix{Float64}(undef, d, A)

    for a in 1:A
        true_idx = order[a]
        for j in 1:d
            corrs[j, a] = abs(safe_correlation(
                result.ncplsfit.W_modes[j][:, a],
                data.mode_weights[j][:, true_idx],
            ))
        end
    end

    corrs
end

function aligned_score_correlations(
    scores::AbstractMatrix{<:Real},
    true_scores::AbstractMatrix{<:Real},
    order::AbstractVector{<:Integer},
)
    [abs(safe_correlation(scores[:, a], true_scores[:, order[a]])) for a in eachindex(order)]
end

function recovered_mz_summary(result, data, order::AbstractVector{<:Integer})
    recovered = Vector{Vector{Int}}(undef, length(order))
    overlaps = Vector{Int}(undef, length(order))

    for a in eachindex(order)
        true_idx = order[a]
        k = length(data.active_mz_channels[true_idx])
        recovered[a] = topk_abs_indices(result.ncplsfit.W_modes[data.mz_mode][:, a], k)
        overlaps[a] = length(intersect(recovered[a], data.active_mz_channels[true_idx]))
    end

    recovered, overlaps
end

function comparison_winner_orthogonalization(
    x::Real,
    y::Real;
    atol::Real=1e-10,
    lower_is_better::Bool=true,
)
    if isapprox(x, y; atol = atol, rtol = 0)
        return :tie
    elseif lower_is_better
        return x < y ? :without_orthogonalization : :with_orthogonalization
    else
        return x > y ? :without_orthogonalization : :with_orthogonalization
    end
end
