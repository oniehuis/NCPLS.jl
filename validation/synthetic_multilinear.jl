"""
    synthetic_multilinear_regression_data(;
        nsamples::Integer=80,
        mode_dims::Tuple{Vararg{Int}}=(40, 30),
        ncomponents::Integer=3,
        nresponses::Integer=1,
        predictive_components::Union{Nothing, AbstractVector{<:Integer}}=nothing,
        rt_mode::Union{Nothing, Integer}=length(mode_dims) >= 2 ? 1 : nothing,
        mz_mode::Integer=length(mode_dims),
        baseline::Real=1.0,
        component_strength_scale::Real=1.0,
        integer_counts::Bool=true,
        x_noise_scale::Real=0.0,
        y_noise_scale::Real=0.05,
        active_mz_per_component::Integer=4,
        rng::AbstractRNG=MersenneTwister(1),
    )

Generate synthetic multilinear regression data with known component scores and mode
weights.

The first dimension of the returned predictor array `X` stores samples. The remaining
dimensions are generated as rank-1 mode combinations so that the data can be used to
assess the vector (`d = 1`), matrix (`d = 2`), and tensor (`d ≥ 3`) branches of the
multilinear NCPLS algorithm. When `integer_counts=true`, `X` contains non-negative
integer counts sampled from a Poisson model. The response matrix `Yprim` is constructed
from the true sample scores `T` and response loading matrix `Qtrue`.
`component_strength_scale` scales the predictor signal strength relative to the fixed
response definition.
"""
function synthetic_multilinear_regression_data(;
    nsamples::Integer=80,
    mode_dims::Tuple{Vararg{Int}}=(40, 30),
    ncomponents::Integer=3,
    nresponses::Integer=1,
    predictive_components::Union{Nothing, AbstractVector{<:Integer}}=nothing,
    rt_mode::Union{Nothing, Integer}=length(mode_dims) >= 2 ? 1 : nothing,
    mz_mode::Integer=length(mode_dims),
    baseline::Real=1.0,
    component_strength_scale::Real=1.0,
    integer_counts::Bool=true,
    x_noise_scale::Real=0.0,
    y_noise_scale::Real=0.05,
    active_mz_per_component::Integer=4,
    rng::AbstractRNG=MersenneTwister(1),
)
    nsamples > 1 || throw(ArgumentError("nsamples must be greater than one"))
    !isempty(mode_dims) || throw(ArgumentError("mode_dims must not be empty"))
    all(d -> d > 1, mode_dims) || throw(ArgumentError(
        "All predictor mode dimensions must be greater than one"))
    ncomponents > 0 || throw(ArgumentError("ncomponents must be greater than zero"))
    nresponses > 0 || throw(ArgumentError("nresponses must be greater than zero"))
    baseline >= 0 || throw(ArgumentError("baseline must be non-negative"))
    component_strength_scale > 0 || throw(ArgumentError(
        "component_strength_scale must be greater than zero"))
    x_noise_scale >= 0 || throw(ArgumentError("x_noise_scale must be non-negative"))
    y_noise_scale >= 0 || throw(ArgumentError("y_noise_scale must be non-negative"))
    active_mz_per_component > 0 || throw(ArgumentError(
        "active_mz_per_component must be greater than zero"))

    d = length(mode_dims)
    1 ≤ mz_mode ≤ d || throw(ArgumentError("mz_mode must index an existing predictor mode"))
    if !isnothing(rt_mode)
        1 ≤ rt_mode ≤ d || throw(ArgumentError("rt_mode must index an existing predictor mode"))
        rt_mode == mz_mode && throw(ArgumentError("rt_mode and mz_mode must differ"))
    end

    predictive = isnothing(predictive_components) ?
        collect(1:min(ncomponents, nresponses)) :
        Int.(collect(predictive_components))
    isempty(predictive) && throw(ArgumentError(
        "predictive_components must not be empty"))
    all(1 ≤ idx ≤ ncomponents for idx in predictive) || throw(ArgumentError(
        "predictive_components must index existing components"))
    allunique(predictive) || throw(ArgumentError(
        "predictive_components must not contain duplicates"))

    mode_weights = [Matrix{Float64}(undef, mode_dims[j], ncomponents) for j in 1:d]
    rt_peak_centers = isnothing(rt_mode) ? nothing : Matrix{Float64}(undef, 2, ncomponents)
    active_mz_channels = Vector{Vector{Int}}(undef, ncomponents)
    templates = Array{Float64}(undef, mode_dims..., ncomponents)
    component_strengths = component_strength_scale .* (6 .+ 6 .* rand(rng, ncomponents))

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
    base = exp.(0.35 .* randn(rng, nsamples))
    for a in 1:ncomponents
        modulation = exp.(0.35 .* randn(rng, nsamples)) .* (1 .+ 2 .* rand(rng, nsamples))
        T[:, a] = base .* modulation
    end

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

    Qtrue = zeros(Float64, ncomponents, nresponses)
    for m in 1:nresponses
        a = predictive[mod1(m, length(predictive))]
        Qtrue[a, m] = 1.0
    end
    for a in setdiff(collect(1:ncomponents), predictive)
        Qtrue[a, :] .= 0.10 .* randn(rng, nresponses)
    end

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
        component_strengths = component_strengths,
        lambda = lambda,
        Yclean = Yclean,
        mode_dims = mode_dims,
        rt_mode = rt_mode,
        mz_mode = mz_mode,
        rt_peak_centers = rt_peak_centers,
        active_mz_channels = active_mz_channels,
        predictive_components = predictive,
        integer_counts = integer_counts,
        component_strength_scale = component_strength_scale,
    )
end

function synthetic_rt_mode(n::Integer, rng::AbstractRNG)
    axis = collect(1:n)
    center1 = 0.20 * (n + 1) + 0.55 * n * rand(rng)
    width1 = 0.04 * n + 0.06 * n * rand(rng)
    mode = exp.(-0.5 .* ((axis .- center1) ./ width1) .^ 2)

    if rand(rng) < 0.5
        center2 = clamp(center1 + rand(rng, (-0.25 * n):(0.25 * n)), 1, n)
        width2 = 0.03 * n + 0.04 * n * rand(rng)
        amp2 = 0.35 + 0.40 * rand(rng)
        mode .+= amp2 .* exp.(-0.5 .* ((axis .- center2) ./ width2) .^ 2)
    else
        center2 = center1
    end

    mode ./= norm(mode)
    mode, [center1, center2]
end

function synthetic_mz_mode(
    n::Integer,
    nactive::Integer,
    rng::AbstractRNG
)
    active = sort(randperm(rng, n)[1:nactive])
    mode = zeros(Float64, n)
    mode[active] .= 0.5 .+ rand(rng, nactive)
    mode ./= norm(mode)
    mode, active
end

function synthetic_auxiliary_mode(n::Integer, rng::AbstractRNG)
    axis = collect(1:n)
    nbump = rand(rng, 1:3)
    mode = zeros(Float64, n)

    for _ in 1:nbump
        center = 0.15 * (n + 1) + 0.70 * n * rand(rng)
        width = 0.05 * n + 0.10 * n * rand(rng)
        amp = 0.5 + 0.7 * rand(rng)
        mode .+= amp .* exp.(-0.5 .* ((axis .- center) ./ width) .^ 2)
    end

    mode ./= norm(mode)
    mode
end
