"""
    synthetic_multilinear_yadd_data(;
        nsamples::Integer=100,
        mode_dims::Tuple{Vararg{Int}}=(40, 30),
        ncomponents::Integer=2,
        predictive_component::Integer=1,
        nadditional::Integer=2,
        rt_mode::Union{Nothing, Integer}=length(mode_dims) >= 2 ? 1 : nothing,
        mz_mode::Integer=length(mode_dims),
        predictive_strength::Real=5.0,
        nuisance_strength::Real=10.0,
        baseline::Real=1.0,
        integer_counts::Bool=true,
        x_noise_scale::Real=0.0,
        yprim_noise_scale::Real=0.35,
        yadd_noise_scale::Real=0.05,
        active_mz_per_component::Integer=4,
        rng::AbstractRNG=MersenneTwister(1),
    )

Generate synthetic multilinear regression data for assessing the effect of `Yadd`.

`Yprim` is a noisy measurement of one predictive latent component, while `Yadd` provides
additional lower-noise response channels aligned with the same predictive component. The
predictor array also contains nuisance components that do not directly determine `Yprim`.
"""
function synthetic_multilinear_yadd_data(;
    nsamples::Integer=100,
    mode_dims::Tuple{Vararg{Int}}=(40, 30),
    ncomponents::Integer=2,
    predictive_component::Integer=1,
    nadditional::Integer=2,
    rt_mode::Union{Nothing, Integer}=length(mode_dims) >= 2 ? 1 : nothing,
    mz_mode::Integer=length(mode_dims),
    predictive_strength::Real=5.0,
    nuisance_strength::Real=10.0,
    baseline::Real=1.0,
    integer_counts::Bool=true,
    x_noise_scale::Real=0.0,
    yprim_noise_scale::Real=0.35,
    yadd_noise_scale::Real=0.05,
    active_mz_per_component::Integer=4,
    rng::AbstractRNG=MersenneTwister(1),
)
    nsamples > 1 || throw(ArgumentError("nsamples must be greater than one"))
    !isempty(mode_dims) || throw(ArgumentError("mode_dims must not be empty"))
    all(d -> d > 1, mode_dims) || throw(ArgumentError(
        "All predictor mode dimensions must be greater than one"))
    ncomponents > 1 || throw(ArgumentError(
        "ncomponents must be greater than one to create predictive and nuisance structure"))
    1 ≤ predictive_component ≤ ncomponents || throw(ArgumentError(
        "predictive_component must index an existing component"))
    nadditional > 0 || throw(ArgumentError("nadditional must be greater than zero"))
    predictive_strength > 0 || throw(ArgumentError("predictive_strength must be positive"))
    nuisance_strength > 0 || throw(ArgumentError("nuisance_strength must be positive"))
    baseline >= 0 || throw(ArgumentError("baseline must be non-negative"))
    x_noise_scale >= 0 || throw(ArgumentError("x_noise_scale must be non-negative"))
    yprim_noise_scale >= 0 || throw(ArgumentError("yprim_noise_scale must be non-negative"))
    yadd_noise_scale >= 0 || throw(ArgumentError("yadd_noise_scale must be non-negative"))
    active_mz_per_component > 0 || throw(ArgumentError(
        "active_mz_per_component must be greater than zero"))

    d = length(mode_dims)
    1 ≤ mz_mode ≤ d || throw(ArgumentError("mz_mode must index an existing predictor mode"))
    if !isnothing(rt_mode)
        1 ≤ rt_mode ≤ d || throw(ArgumentError("rt_mode must index an existing predictor mode"))
        rt_mode == mz_mode && throw(ArgumentError("rt_mode and mz_mode must differ"))
    end

    mode_weights = [Matrix{Float64}(undef, mode_dims[j], ncomponents) for j in 1:d]
    rt_peak_centers = isnothing(rt_mode) ? nothing : Matrix{Float64}(undef, 2, ncomponents)
    active_mz_channels = Vector{Vector{Int}}(undef, ncomponents)
    templates = Array{Float64}(undef, mode_dims..., ncomponents)

    for a in 1:ncomponents
        factors = Vector{Vector{Float64}}(undef, d)
        for j in 1:d
            if !isnothing(rt_mode) && j == rt_mode
                vec, centers = synthetic_rt_mode(mode_dims[j], rng)
                factors[j] = vec
                rt_peak_centers[:, a] = centers
            elseif j == mz_mode
                vec, channels = synthetic_mz_mode(
                    mode_dims[j],
                    min(active_mz_per_component, mode_dims[j]),
                    rng,
                )
                factors[j] = vec
                active_mz_channels[a] = channels
            else
                factors[j] = synthetic_auxiliary_mode(mode_dims[j], rng)
            end
            mode_weights[j][:, a] = factors[j]
        end
        templates[ntuple(_ -> Colon(), d)..., a] .= outer_tensor(factors)
    end

    T = Array{Float64}(undef, nsamples, ncomponents)
    T[:, predictive_component] = exp.(0.55 .* randn(rng, nsamples)) .* (1 .+ 3 .* rand(rng, nsamples))
    for a in 1:ncomponents
        a == predictive_component && continue
        T[:, a] = exp.(0.55 .* randn(rng, nsamples)) .* (1 .+ 4 .* rand(rng, nsamples))
    end

    component_strengths = fill(Float64(nuisance_strength), ncomponents)
    component_strengths[predictive_component] = Float64(predictive_strength)

    lambda = Array{Float64}(undef, nsamples, mode_dims...)
    if integer_counts
        X = Array{Int}(undef, nsamples, mode_dims...)
    else
        X = Array{Float64}(undef, nsamples, mode_dims...)
    end

    for i in 1:nsamples
        λi = fill(Float64(baseline), mode_dims...)
        for a in 1:ncomponents
            λi .+= T[i, a] .* component_strengths[a] .* view(templates, ntuple(_ -> Colon(), d)..., a)
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
            Xi = max.(Xi, 0.0)
            X[i, ntuple(_ -> Colon(), d)...] = Xi
        end
    end

    predictive_score = T[:, predictive_component]
    Yclean = reshape(predictive_score, :, 1)
    Yprim = copy(Yclean)
    if yprim_noise_scale > 0
        Yprim .+= yprim_noise_scale .* std(Yclean) .* randn(rng, size(Yclean)...)
    end

    Yadd = Matrix{Float64}(undef, nsamples, nadditional)
    for j in 1:nadditional
        scale = 0.7 + 0.4 * rand(rng)
        offset = 0.05 * std(Yclean) * randn(rng)
        Yadd[:, j] = scale .* predictive_score .+ offset
        if yadd_noise_scale > 0
            Yadd[:, j] .+= yadd_noise_scale .* std(Yclean) .* randn(rng, nsamples)
        end
    end

    (
        X = X,
        Yprim = Yprim,
        Yadd = Yadd,
        T = T,
        templates = templates,
        mode_weights = mode_weights,
        component_strengths = component_strengths,
        lambda = lambda,
        Yclean = Yclean,
        mode_dims = mode_dims,
        rt_mode = rt_mode,
        mz_mode = mz_mode,
        rt_peak_centers = rt_peak_centers,
        active_mz_channels = active_mz_channels,
        predictive_component = predictive_component,
        integer_counts = integer_counts,
    )
end

"""
    analyze_synthetic_yadd_with_ncpls(
        data;
        model::Union{NCPLSModel, Nothing}=nothing,
        ncomponents::Integer=1,
        multilinear::Bool=true,
        use_yadd::Bool=true,
        test_fraction::Real=0.3,
        rng::AbstractRNG=MersenneTwister(1),
        train_idx::Union{AbstractVector{<:Integer}, Nothing}=nothing,
        test_idx::Union{AbstractVector{<:Integer}, Nothing}=nothing,
        verbose::Bool=false,
    )

Fit NCPLS to synthetic `Yadd` validation data and return train/test predictions, scores,
and regression metrics.
"""
function analyze_synthetic_yadd_with_ncpls(
    data;
    model::Union{NCPLSModel, Nothing}=nothing,
    ncomponents::Integer=1,
    multilinear::Bool=true,
    use_yadd::Bool=true,
    test_fraction::Real=0.3,
    rng::AbstractRNG=MersenneTwister(1),
    train_idx::Union{AbstractVector{<:Integer}, Nothing}=nothing,
    test_idx::Union{AbstractVector{<:Integer}, Nothing}=nothing,
    verbose::Bool=false,
)
    ncomponents == 1 || throw(ArgumentError(
        "The synthetic Yadd assessment currently supports ncomponents = 1"))

    X = data.X
    Yprim = data.Yprim
    Yadd = use_yadd ? data.Yadd : nothing

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
    Yadd_train = isnothing(Yadd) ? nothing : Yadd[train_idx, :]
    Yadd_test = isnothing(Yadd) ? nothing : Yadd[test_idx, :]

    ncpls_model = isnothing(model) ? NCPLSModel(
        ncomponents = 1,
        center_X = true,
        scale_X = false,
        center_Yprim = true,
        multilinear = multilinear,
        orthogonalize_mode_weights = false,
    ) : model

    mf = fit(
        ncpls_model,
        Xtrain,
        Ytrain;
        Yadd = Yadd_train,
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
        use_yadd = use_yadd,
        ncpls_model = ncpls_model,
        ncplsfit = mf,
        train_idx = train_idx,
        test_idx = test_idx,
        Xtrain = Xtrain,
        Xtest = Xtest,
        Ytrain = Ytrain,
        Ytest = Ytest,
        Yadd_train = Yadd_train,
        Yadd_test = Yadd_test,
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
    compare_synthetic_yadd_effect(
        data;
        model_without_yadd::Union{NCPLSModel, Nothing}=nothing,
        model_with_yadd::Union{NCPLSModel, Nothing}=nothing,
        multilinear::Bool=true,
        test_fraction::Real=0.3,
        rng::AbstractRNG=MersenneTwister(1),
        verbose::Bool=false,
    )

Compare NCPLS fits with and without `Yadd` on the same synthetic train/test split and
summarize the effect on prediction and structure recovery.
"""
function compare_synthetic_yadd_effect(
    data;
    model_without_yadd::Union{NCPLSModel, Nothing}=nothing,
    model_with_yadd::Union{NCPLSModel, Nothing}=nothing,
    multilinear::Bool=true,
    test_fraction::Real=0.3,
    rng::AbstractRNG=MersenneTwister(1),
    verbose::Bool=false,
)
    train_idx, test_idx = split_synthetic_gcms_samples(
        size(data.X, 1);
        test_fraction = test_fraction,
        rng = rng,
    )

    without_yadd = analyze_synthetic_yadd_with_ncpls(
        data;
        model = model_without_yadd,
        multilinear = multilinear,
        use_yadd = false,
        test_fraction = test_fraction,
        rng = rng,
        train_idx = train_idx,
        test_idx = test_idx,
        verbose = verbose,
    )

    with_yadd = analyze_synthetic_yadd_with_ncpls(
        data;
        model = model_with_yadd,
        multilinear = multilinear,
        use_yadd = true,
        test_fraction = test_fraction,
        rng = rng,
        train_idx = train_idx,
        test_idx = test_idx,
        verbose = verbose,
    )

    predictive = data.predictive_component
    without_true_score_abs_cor_train = abs(safe_correlation(
        without_yadd.scores_train[:, 1], data.T[train_idx, predictive]))
    without_true_score_abs_cor_test = abs(safe_correlation(
        without_yadd.scores_test[:, 1], data.T[test_idx, predictive]))
    with_true_score_abs_cor_train = abs(safe_correlation(
        with_yadd.scores_train[:, 1], data.T[train_idx, predictive]))
    with_true_score_abs_cor_test = abs(safe_correlation(
        with_yadd.scores_test[:, 1], data.T[test_idx, predictive]))

    if multilinear
        without_mode_abs_cor = [
            abs(safe_correlation(
                without_yadd.ncplsfit.W_modes[j][:, 1],
                data.mode_weights[j][:, predictive],
            )) for j in eachindex(data.mode_weights)
        ]
        with_mode_abs_cor = [
            abs(safe_correlation(
                with_yadd.ncplsfit.W_modes[j][:, 1],
                data.mode_weights[j][:, predictive],
            )) for j in eachindex(data.mode_weights)
        ]

        k = length(data.active_mz_channels[predictive])
        without_recovered_top_mz = sortperm(
            abs.(without_yadd.ncplsfit.W_modes[data.mz_mode][:, 1]); rev = true)[1:k]
        with_recovered_top_mz = sortperm(
            abs.(with_yadd.ncplsfit.W_modes[data.mz_mode][:, 1]); rev = true)[1:k]
        without_mz_overlap = length(intersect(without_recovered_top_mz, data.active_mz_channels[predictive]))
        with_mz_overlap = length(intersect(with_recovered_top_mz, data.active_mz_channels[predictive]))
    else
        without_mode_abs_cor = nothing
        with_mode_abs_cor = nothing
        without_recovered_top_mz = nothing
        with_recovered_top_mz = nothing
        without_mz_overlap = nothing
        with_mz_overlap = nothing
    end

    rmse_test_delta = with_yadd.rmse_test_overall[1] - without_yadd.rmse_test_overall[1]
    r2_test_delta = with_yadd.r2_test_overall[1] - without_yadd.r2_test_overall[1]
    better_model_test_rmse = comparison_winner_yadd(
        without_yadd.rmse_test_overall[1],
        with_yadd.rmse_test_overall[1];
        lower_is_better = true,
    )
    better_model_test_r2 = comparison_winner_yadd(
        without_yadd.r2_test_overall[1],
        with_yadd.r2_test_overall[1];
        lower_is_better = false,
    )

    (
        train_idx = train_idx,
        test_idx = test_idx,
        without_yadd = without_yadd,
        with_yadd = with_yadd,
        predictive_component = predictive,
        true_active_mz_channels = data.active_mz_channels[predictive],
        true_q = data.Yclean,
        without_true_score_abs_cor_train = without_true_score_abs_cor_train,
        without_true_score_abs_cor_test = without_true_score_abs_cor_test,
        with_true_score_abs_cor_train = with_true_score_abs_cor_train,
        with_true_score_abs_cor_test = with_true_score_abs_cor_test,
        without_mode_abs_cor = without_mode_abs_cor,
        with_mode_abs_cor = with_mode_abs_cor,
        without_recovered_top_mz = without_recovered_top_mz,
        with_recovered_top_mz = with_recovered_top_mz,
        without_mz_overlap = without_mz_overlap,
        with_mz_overlap = with_mz_overlap,
        rmse_test_delta = rmse_test_delta,
        r2_test_delta = r2_test_delta,
        better_model_test_rmse = better_model_test_rmse,
        better_model_test_r2 = better_model_test_r2,
    )
end

function comparison_winner_yadd(
    without_yadd::Real,
    with_yadd::Real;
    atol::Real=1e-10,
    lower_is_better::Bool=true,
)
    if isapprox(without_yadd, with_yadd; atol = atol, rtol = 0)
        return :tie
    elseif lower_is_better
        return without_yadd < with_yadd ? :without_yadd : :with_yadd
    else
        return without_yadd > with_yadd ? :without_yadd : :with_yadd
    end
end
