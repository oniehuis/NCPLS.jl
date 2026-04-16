"""
    synthetic_multilinear_hybrid_data(;
        nmajor::Integer=70,
        nminor::Integer=30,
        mode_dims::Tuple{Vararg{Int}}=(40, 30),
        orthogonal_truth::Bool=true,
        rt_mode::Union{Nothing, Integer}=length(mode_dims) >= 2 ? 1 : nothing,
        mz_mode::Integer=length(mode_dims),
        baseline::Real=0.5,
        integer_counts::Bool=false,
        class_component_strength::Real=4.5,
        regression_component_strength::Real=5.0,
        nuisance_component_strength::Real=7.0,
        noisy_fraction::Real=0.25,
        clean_weight::Real=1.0,
        noisy_weight::Real=0.25,
        x_noise_scale_clean::Real=0.04,
        x_noise_scale_noisy::Real=0.20,
        yreg_noise_scale_clean::Real=0.05,
        yreg_noise_scale_noisy::Real=0.20,
        yadd_noise_scale::Real=0.03,
        active_mz_per_component::Integer=4,
        rng::AbstractRNG=MersenneTwister(1),
    )

Generate synthetic multilinear data that supports pure discriminant analysis, pure
regression, and hybrid response modelling in a single dataset.

The returned object contains:

- `sampleclasses`: a categorical two-class response for DA,
- `Yprim_da`: the corresponding one-hot class-indicator matrix,
- `Yprim_reg`: continuous regression targets,
- `Yprim_hybrid`: the combined response block `[Yprim_da Yprim_reg]`,
- `Yadd`: lower-noise auxiliary responses aligned with the main predictive components,
- `obs_weights`: sample weights that downweight a deliberately noisier subset of samples.

The predictor array is generated from three latent multilinear components:

1. a class-related component,
2. a continuous regression component,
3. a nuisance component that affects `X` but is not itself a prediction target.

When `orthogonal_truth=true`, the true mode vectors are close to orthogonal within each
mode, which makes the dataset useful for illustrating the effect of
`orthogonalize_mode_weights=true`. When `orthogonal_truth=false`, the true mode vectors
overlap within each mode so the orthogonalization option becomes a stricter modelling
assumption.
"""
function synthetic_multilinear_hybrid_data(;
    nmajor::Integer=70,
    nminor::Integer=30,
    mode_dims::Tuple{Vararg{Int}}=(40, 30),
    orthogonal_truth::Bool=true,
    rt_mode::Union{Nothing, Integer}=length(mode_dims) >= 2 ? 1 : nothing,
    mz_mode::Integer=length(mode_dims),
    baseline::Real=0.5,
    integer_counts::Bool=false,
    class_component_strength::Real=4.5,
    regression_component_strength::Real=5.0,
    nuisance_component_strength::Real=7.0,
    noisy_fraction::Real=0.25,
    clean_weight::Real=1.0,
    noisy_weight::Real=0.25,
    x_noise_scale_clean::Real=0.04,
    x_noise_scale_noisy::Real=0.20,
    yreg_noise_scale_clean::Real=0.05,
    yreg_noise_scale_noisy::Real=0.20,
    yadd_noise_scale::Real=0.03,
    active_mz_per_component::Integer=4,
    rng::AbstractRNG=MersenneTwister(1),
)
    nmajor > 1 || throw(ArgumentError("nmajor must be greater than one"))
    nminor > 1 || throw(ArgumentError("nminor must be greater than one"))
    !isempty(mode_dims) || throw(ArgumentError("mode_dims must not be empty"))
    all(d -> d > 3, mode_dims) || throw(ArgumentError(
        "All predictor mode dimensions must be greater than three"))
    0 < noisy_fraction < 1 || throw(ArgumentError(
        "noisy_fraction must lie strictly between 0 and 1"))
    baseline >= 0 || throw(ArgumentError("baseline must be non-negative"))
    clean_weight > 0 || throw(ArgumentError("clean_weight must be greater than zero"))
    noisy_weight > 0 || throw(ArgumentError("noisy_weight must be greater than zero"))
    x_noise_scale_clean >= 0 || throw(ArgumentError(
        "x_noise_scale_clean must be non-negative"))
    x_noise_scale_noisy >= 0 || throw(ArgumentError(
        "x_noise_scale_noisy must be non-negative"))
    yreg_noise_scale_clean >= 0 || throw(ArgumentError(
        "yreg_noise_scale_clean must be non-negative"))
    yreg_noise_scale_noisy >= 0 || throw(ArgumentError(
        "yreg_noise_scale_noisy must be non-negative"))
    yadd_noise_scale >= 0 || throw(ArgumentError(
        "yadd_noise_scale must be non-negative"))
    active_mz_per_component > 0 || throw(ArgumentError(
        "active_mz_per_component must be greater than zero"))
    class_component_strength > 0 || throw(ArgumentError(
        "class_component_strength must be positive"))
    regression_component_strength > 0 || throw(ArgumentError(
        "regression_component_strength must be positive"))
    nuisance_component_strength > 0 || throw(ArgumentError(
        "nuisance_component_strength must be positive"))

    d = length(mode_dims)
    1 ≤ mz_mode ≤ d || throw(ArgumentError("mz_mode must index an existing predictor mode"))
    if !isnothing(rt_mode)
        1 ≤ rt_mode ≤ d || throw(ArgumentError("rt_mode must index an existing predictor mode"))
        rt_mode == mz_mode && throw(ArgumentError("rt_mode and mz_mode must differ"))
    end

    nsamples = nmajor + nminor
    ncomponents = 3
    component_names = ["class", "regression", "nuisance"]
    component_strengths = [
        Float64(class_component_strength),
        Float64(regression_component_strength),
        Float64(nuisance_component_strength),
    ]

    class_strings = vcat(fill("major", nmajor), fill("minor", nminor))
    perm = randperm(rng, nsamples)
    class_strings = class_strings[perm]
    sampleclasses = categorical(class_strings)
    samplelabels = string.(1:nsamples)

    noisy_mask = synthetic_hybrid_noisy_mask(class_strings, noisy_fraction, rng)
    noisy_idx = findall(noisy_mask)
    clean_idx = findall(.!noisy_mask)

    obs_weights_raw = ifelse.(noisy_mask, Float64(noisy_weight), Float64(clean_weight))
    obs_weights = obs_weights_raw .* (nsamples / sum(obs_weights_raw))

    mode_weights = [Matrix{Float64}(undef, mode_dims[j], ncomponents) for j in 1:d]
    rt_peak_centers = isnothing(rt_mode) ? nothing : Vector{Float64}(undef, ncomponents)
    active_mz_channels = Vector{Vector{Int}}(undef, ncomponents)

    for j in 1:d
        if !isnothing(rt_mode) && j == rt_mode
            weights_j, centers_j = synthetic_hybrid_rt_modes(
                mode_dims[j], orthogonal_truth)
            mode_weights[j] .= weights_j
            rt_peak_centers .= centers_j
        elseif j == mz_mode
            weights_j, active_j = synthetic_hybrid_mz_modes(
                mode_dims[j],
                orthogonal_truth,
                active_mz_per_component,
                rng,
            )
            mode_weights[j] .= weights_j
            active_mz_channels .= active_j
        else
            mode_weights[j] .= synthetic_hybrid_auxiliary_modes(
                mode_dims[j], orthogonal_truth, j)
        end
    end

    templates = Array{Float64}(undef, mode_dims..., ncomponents)
    for a in 1:ncomponents
        templates[ntuple(_ -> Colon(), d)..., a] .=
            outer_tensor([mode_weights[j][:, a] for j in 1:d])
    end

    T = Matrix{Float64}(undef, nsamples, ncomponents)
    for i in 1:nsamples
        is_minor = class_strings[i] == "minor"
        class_mean = is_minor ? 1.85 : 1.00
        class_sd = noisy_mask[i] ? 0.30 : 0.12
        T[i, 1] = max(0.10, class_mean + class_sd * randn(rng))
        T[i, 2] = 0.80 + exp(0.30 * randn(rng)) * (1 + 1.2 * rand(rng))
        T[i, 3] = 0.90 + exp(0.40 * randn(rng)) * (1 + 1.5 * rand(rng))
    end

    lambda = Array{Float64}(undef, nsamples, mode_dims...)
    X = integer_counts ? Array{Int}(undef, nsamples, mode_dims...) :
        Array{Float64}(undef, nsamples, mode_dims...)
    tail = ntuple(_ -> Colon(), d)

    for i in 1:nsamples
        λi = fill(Float64(baseline), mode_dims...)
        for a in 1:ncomponents
            λi .+= T[i, a] * component_strengths[a] .* view(templates, tail..., a)
        end
        lambda[i, tail...] = λi

        x_noise_scale_i = noisy_mask[i] ? x_noise_scale_noisy : x_noise_scale_clean
        if integer_counts
            λobs = x_noise_scale_i > 0 ? λi .* exp.(x_noise_scale_i .* randn(rng, mode_dims...)) : λi
            for I in CartesianIndices(λobs)
                X[(i, Tuple(I)...)...] = synthetic_hybrid_poisson_count(rng, λobs[I])
            end
        else
            Xi = λi .+ x_noise_scale_i * max(mean(λi), eps(Float64)) .* randn(rng, mode_dims...)
            X[i, tail...] = max.(Xi, 0.0)
        end
    end

    z1 = synthetic_zscore(T[:, 1])
    z2 = synthetic_zscore(T[:, 2])
    z3 = synthetic_zscore(T[:, 3])

    Yreg_clean = hcat(
        0.35 .* z1 .+ 0.95 .* z2,
        -0.20 .* z1 .+ 0.75 .* z2 .+ 0.15 .* z3,
    )
    Yreg_scale = vec(std(Yreg_clean; dims=1))
    Yprim_reg = copy(Yreg_clean)
    for i in 1:nsamples
        y_noise_scale_i = noisy_mask[i] ? yreg_noise_scale_noisy : yreg_noise_scale_clean
        Yprim_reg[i, :] .+= y_noise_scale_i .* Yreg_scale .* randn(rng, size(Yreg_clean, 2))
    end

    Yprim_da, class_levels = onehot(class_strings)
    responselabels_da = String.(class_levels)
    responselabels_reg = ["trait1", "trait2"]
    responselabels_hybrid = vcat(responselabels_da, responselabels_reg)
    Yprim_hybrid = hcat(Yprim_da, Yprim_reg)

    Yadd_clean = hcat(
        z1,
        z2,
        0.65 .* z1 .+ 0.55 .* z2,
    )
    yadd_scale = vec(std(Yadd_clean; dims=1))
    Yadd = copy(Yadd_clean)
    if yadd_noise_scale > 0
        Yadd .+= yadd_noise_scale .* reshape(yadd_scale, 1, :) .* randn(rng, size(Yadd)...)
    end
    yaddlabels = ["class_proxy", "regression_proxy", "blend_proxy"]

    predictoraxes = synthetic_hybrid_predictoraxes(mode_dims, rt_mode, mz_mode)
    true_mode_inner_products = [mode_weights[j]' * mode_weights[j] for j in 1:d]

    (
        X = X,
        Yprim = Yprim_hybrid,
        Yprim_da = Yprim_da,
        Yprim_reg = Yprim_reg,
        Yprim_hybrid = Yprim_hybrid,
        Yreg_clean = Yreg_clean,
        Yadd = Yadd,
        Yadd_clean = Yadd_clean,
        yaddlabels = yaddlabels,
        sampleclasses = sampleclasses,
        sampleclasses_string = class_strings,
        samplelabels = samplelabels,
        responselabels = responselabels_hybrid,
        responselabels_da = responselabels_da,
        responselabels_reg = responselabels_reg,
        responselabels_hybrid = responselabels_hybrid,
        classcols = collect(1:size(Yprim_da, 2)),
        regressioncols = collect(size(Yprim_da, 2) + 1:size(Yprim_hybrid, 2)),
        obs_weights = obs_weights,
        obs_weights_raw = obs_weights_raw,
        noisy_mask = noisy_mask,
        noisy_idx = noisy_idx,
        clean_idx = clean_idx,
        T = T,
        templates = templates,
        mode_weights = mode_weights,
        lambda = lambda,
        component_names = component_names,
        component_strengths = component_strengths,
        class_component = 1,
        regression_component = 2,
        nuisance_component = 3,
        orthogonal_truth = orthogonal_truth,
        true_mode_inner_products = true_mode_inner_products,
        predictoraxes = predictoraxes,
        mode_dims = mode_dims,
        rt_mode = rt_mode,
        mz_mode = mz_mode,
        rt_peak_centers = rt_peak_centers,
        active_mz_channels = active_mz_channels,
        integer_counts = integer_counts,
    )
end

function synthetic_hybrid_noisy_mask(
    class_strings::AbstractVector{<:AbstractString},
    noisy_fraction::Real,
    rng::AbstractRNG,
)
    mask = falses(length(class_strings))
    for label in unique(class_strings)
        idx = findall(==(label), class_strings)
        n_noisy = clamp(round(Int, noisy_fraction * length(idx)), 1, length(idx) - 1)
        mask[shuffle(rng, idx)[1:n_noisy]] .= true
    end
    mask
end

function synthetic_hybrid_rt_modes(n::Integer, orthogonal_truth::Bool)
    axis = collect(1:n)
    centers = orthogonal_truth ?
        ((0.18, 0.50, 0.82) .* (n + 1)) :
        ((0.36, 0.52, 0.66) .* (n + 1))
    widths = orthogonal_truth ?
        (0.07, 0.08, 0.07) .* n :
        (0.14, 0.12, 0.14) .* n

    W = Matrix{Float64}(undef, n, 3)
    for a in 1:3
        vec = exp.(-0.5 .* ((axis .- centers[a]) ./ widths[a]) .^ 2)
        W[:, a] = vec ./ norm(vec)
    end

    W, collect(centers)
end

function synthetic_hybrid_mz_modes(
    n::Integer,
    orthogonal_truth::Bool,
    active_mz_per_component::Integer,
    rng::AbstractRNG,
)
    nactive = min(active_mz_per_component, max(1, fld(n, orthogonal_truth ? 3 : 2)))
    W = zeros(Float64, n, 3)
    active = Vector{Vector{Int}}(undef, 3)

    if orthogonal_truth
        segment_edges = round.(Int, range(1, n + 1; length=4))
        for a in 1:3
            seg = collect(segment_edges[a]:(segment_edges[a + 1] - 1))
            chosen = sort(shuffle(rng, seg)[1:min(nactive, length(seg))])
            W[chosen, a] .= 0.8 .+ rand(rng, length(chosen))
            active[a] = chosen
        end
    else
        centers = clamp.(round.(Int, [0.40, 0.52, 0.64] .* n), 1, n)
        halfwidth = max(0, fld(nactive - 1, 2))
        for a in 1:3
            chosen = collect(max(1, centers[a] - halfwidth):min(n, centers[a] + halfwidth))
            if length(chosen) < nactive
                extra = setdiff(collect(max(1, centers[a] - nactive):min(n, centers[a] + nactive)), chosen)
                append!(chosen, shuffle(rng, extra)[1:min(nactive - length(chosen), length(extra))])
                sort!(chosen)
            end
            chosen = chosen[1:min(nactive, length(chosen))]
            W[chosen, a] .= 0.8 .+ rand(rng, length(chosen))
            active[a] = chosen
        end
    end

    for a in 1:3
        W[:, a] ./= norm(W[:, a])
    end

    W, active
end

function synthetic_hybrid_auxiliary_modes(
    n::Integer,
    orthogonal_truth::Bool,
    mode_index::Integer,
)
    axis = collect(1:n)
    shift = 0.03 * (mode_index - 1)
    centers = orthogonal_truth ?
        ((0.24 + shift, 0.56 + 0.5 * shift, 0.84 - 0.5 * shift) .* (n + 1)) :
        ((0.38 + 0.2 * shift, 0.52, 0.66 - 0.2 * shift) .* (n + 1))
    widths = orthogonal_truth ?
        (0.10, 0.09, 0.08) .* n :
        (0.18, 0.15, 0.18) .* n

    W = Matrix{Float64}(undef, n, 3)
    for a in 1:3
        primary = exp.(-0.5 .* ((axis .- centers[a]) ./ widths[a]) .^ 2)
        shoulder_center = clamp(centers[a] + (-1)^a * 0.12 * n, 1, n)
        shoulder = 0.30 .* exp.(-0.5 .* ((axis .- shoulder_center) ./ (1.4 * widths[a])) .^ 2)
        vec = primary .+ shoulder
        W[:, a] = vec ./ norm(vec)
    end

    W
end

function synthetic_hybrid_predictoraxes(
    mode_dims::Tuple{Vararg{Int}},
    rt_mode::Union{Nothing, Integer},
    mz_mode::Integer,
)
    axes = PredictorAxis[]
    for (j, dim_j) in enumerate(mode_dims)
        if !isnothing(rt_mode) && j == rt_mode
            push!(axes, PredictorAxis("RT", collect(1:dim_j); unit="a.u."))
        elseif j == mz_mode
            push!(axes, PredictorAxis("m/z", collect(1:dim_j); unit="a.u."))
        else
            push!(axes, PredictorAxis("mode$(j)", collect(1:dim_j)))
        end
    end
    axes
end

function synthetic_zscore(x::AbstractVector{<:Real})
    μ = mean(x)
    σ = std(x)
    σ > 0 || return zeros(Float64, length(x))
    (float64(x) .- μ) ./ σ
end

function synthetic_hybrid_poisson_count(rng::AbstractRNG, λ::Real)
    λ >= 0 || throw(ArgumentError("Poisson mean must be non-negative"))
    λ == 0 && return 0

    limit = exp(-Float64(λ))
    product = 1.0
    count = 0

    while product > limit
        count += 1
        product *= rand(rng)
    end

    count - 1
end
