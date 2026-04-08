"""
    synthetic_gcms_regression_data(;
        nsamples::Integer=60,
        n_rt::Integer=40,
        n_mz::Integer=30,
        n_compounds::Integer=5,
        target_compounds::AbstractVector{<:Integer}=[1],
        additional_compounds::AbstractVector{<:Integer}=Int[],
        baseline::Real=1.0,
        rng::AbstractRNG=MersenneTwister(1),
    )

Generate synthetic GC-MS regression data with samples along the first dimension, retention
time along the second, and m/z along the third.

Each compound is built as a rank-1 outer product of an RT profile and an m/z pattern, so
the resulting tensor is suitable for testing both unfolded and multilinear NCPLS fits.
The returned predictor array `X` contains non-negative integer counts. `Yprim` contains
the concentrations of the selected `target_compounds`, and `Yadd` contains the
concentrations of `additional_compounds` or `nothing` when no additional targets are
requested.
"""
function synthetic_gcms_regression_data(;
    nsamples::Integer=60,
    n_rt::Integer=40,
    n_mz::Integer=30,
    n_compounds::Integer=5,
    target_compounds::AbstractVector{<:Integer}=[1],
    additional_compounds::AbstractVector{<:Integer}=Int[],
    baseline::Real=1.0,
    rng::AbstractRNG=MersenneTwister(1),
)
    nsamples > 0 || throw(ArgumentError("nsamples must be greater than zero"))
    n_rt > 1 || throw(ArgumentError("n_rt must be greater than one"))
    n_mz > 1 || throw(ArgumentError("n_mz must be greater than one"))
    n_compounds > 0 || throw(ArgumentError("n_compounds must be greater than zero"))
    baseline >= 0 || throw(ArgumentError("baseline must be non-negative"))

    target = Int.(collect(target_compounds))
    additional = Int.(collect(additional_compounds))

    isempty(target) && throw(ArgumentError("target_compounds must not be empty"))
    allunique(vcat(target, additional)) || throw(ArgumentError(
        "target_compounds and additional_compounds must be disjoint"))
    all(1 ≤ idx ≤ n_compounds for idx in target) || throw(ArgumentError(
        "target_compounds must index existing compounds"))
    all(1 ≤ idx ≤ n_compounds for idx in additional) || throw(ArgumentError(
        "additional_compounds must index existing compounds"))

    rt_axis = collect(1:n_rt)
    mz_axis = collect(1:n_mz)

    rt_profiles = Matrix{Float64}(undef, n_rt, n_compounds)
    mz_patterns = Matrix{Float64}(undef, n_mz, n_compounds)
    templates = Array{Float64}(undef, n_rt, n_mz, n_compounds)

    for k in 1:n_compounds
        rt_center = 0.15 * (n_rt + 1) + 0.70 * n_rt * rand(rng)
        rt_width = 0.04 * n_rt + 0.08 * n_rt * rand(rng)
        rt_profile = exp.(-0.5 .* ((rt_axis .- rt_center) ./ rt_width) .^ 2)
        rt_profile ./= maximum(rt_profile)

        mz_pattern = zeros(Float64, n_mz)
        n_fragments = min(n_mz, rand(rng, 3:6))
        fragment_positions = sort(randperm(rng, n_mz)[1:n_fragments])
        for pos in fragment_positions
            amplitude = 0.4 + 0.6 * rand(rng)
            width = 0.35 + 0.85 * rand(rng)
            mz_pattern .+= amplitude .* exp.(
                -0.5 .* ((mz_axis .- pos) ./ width) .^ 2
            )
        end
        mz_pattern ./= maximum(mz_pattern)

        rt_profiles[:, k] = rt_profile
        mz_patterns[:, k] = mz_pattern
        templates[:, :, k] = rt_profile * mz_pattern'
    end

    concentration_base = exp.(1.2 .+ 0.35 .* randn(rng, nsamples))
    concentration_modulation = exp.(0.55 .* randn(rng, nsamples, n_compounds))
    concentrations = reshape(concentration_base, :, 1) .*
        (2 .+ 6 .* rand(rng, nsamples, n_compounds)) .* concentration_modulation

    λ = Array{Float64}(undef, nsamples, n_rt, n_mz)
    X = Array{Int}(undef, nsamples, n_rt, n_mz)

    for i in 1:nsamples
        λi = fill(Float64(baseline), n_rt, n_mz)
        for k in 1:n_compounds
            λi .+= concentrations[i, k] .* templates[:, :, k]
        end
        λ[i, :, :] = λi

        for rt in 1:n_rt, mz in 1:n_mz
            X[i, rt, mz] = sample_poisson_count(rng, λi[rt, mz])
        end
    end

    Yprim = concentrations[:, target]
    Yadd = isempty(additional) ? nothing : concentrations[:, additional]

    (
        X=X,
        Yprim=Yprim,
        Yadd=Yadd,
        concentrations=concentrations,
        rt_profiles=rt_profiles,
        mz_patterns=mz_patterns,
        templates=templates,
        lambda=λ,
        rt_axis=rt_axis,
        mz_axis=mz_axis,
        target_compounds=target,
        additional_compounds=additional,
    )
end

function sample_poisson_count(rng::AbstractRNG, λ::Real)
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
