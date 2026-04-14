
"""
    fit_ncpls_light(
        m::NCPLSModel,
        X::AbstractArray{<:Real},
        Yprim::AbstractMatrix{<:Real};
        Yadd::Union{AbstractMatrix{<:Real}, Nothing}=nothing,
        obs_weights::Union{AbstractVector{<:Real}, Nothing}=nothing,
        samplelabels::AbstractVector=String[],
        responselabels::AbstractVector=String[],
        sampleclasses::Union{AbstractVector, Nothing}=nothing,
        predictoraxes=(),
        verbose::Bool=false
    ) -> NCPLSFitLight

Low-level NCPLS fitting routine used by internal cross-validation helpers that returns an
`NCPLSFitLight`. The keyword interface mirrors [`fit`](@ref) for compatibility with
fold-local cross-validation kwargs, but the reduced fit stores only prediction-critical
state.
"""
function fit_ncpls_light(
    m::NCPLSModel,
    X::AbstractArray{<:Real},
    Yprim::AbstractMatrix{<:Real};
    kwargs...
)
    fit_ncpls_light_core(m, X, Yprim; kwargs...)
end


"""
    fit_ncpls_light_core(
        m::NCPLSModel,
        X::AbstractArray{<:Real},
        Yprim::AbstractMatrix{<:Real};
        Yadd::Union{AbstractMatrix{<:Real}, Nothing}=nothing,
        obs_weights::Union{AbstractVector{<:Real}, Nothing}=nothing,
        samplelabels::AbstractVector=String[],
        responselabels::AbstractVector=String[],
        sampleclasses::Union{AbstractVector, Nothing}=nothing,
        predictoraxes=(),
        verbose::Bool=false
    ) -> NCPLSFitLight

Fit an NCPLS model and return a reduced prediction-only fit object.

This is the low-level fitting routine used by internal cross-validation helpers. The
keyword interface mirrors [`fit_ncpls_core`](@ref) so fold-local fit kwargs continue to
work, even though metadata fields are not stored on the reduced fit.
"""
function fit_ncpls_light_core(
    m::NCPLSModel,
    X::AbstractArray{<:Real},
    Yprim::AbstractMatrix{<:Real};
    Yadd::T1=nothing,
    obs_weights::T2=nothing,
    samplelabels::T3=String[],
    responselabels::T4=String[],
    sampleclasses::T5=nothing,
    predictoraxes=(),
    verbose::Bool=false
) where {
    T1<:Union{AbstractMatrix{<:Real}, Nothing},
    T2<:Union{AbstractVector{<:Real}, Nothing},
    T3<:AbstractVector,
    T4<:AbstractVector,
    T5<:Union{AbstractVector, Nothing},
}

    d = preprocess(m, X, Yprim, Yadd, obs_weights)

    T = zeros(Float64, size(d.X, 1), m.ncomponents)
    P = Array{Float64}(undef, size(d.X)[2:end]..., m.ncomponents)
    Q = Matrix{Float64}(undef, size(d.Yprim, 2), m.ncomponents)
    W_A = Array{Float64}(undef, size(d.X)[2:end]..., m.ncomponents)
    W_modes = if m.multilinear
        [
            Array{Float64}(undef, size(d.X, j + 1), m.ncomponents)
            for j in 1:(ndims(d.X) - 1)
        ]
    else
        nothing
    end
    rng = m.multilinear ? MersenneTwister(m.multilinear_seed) : nothing

    cca_obs_weights = isnothing(obs_weights) ? nothing : sqrt.(obs_weights)

    Y = copy(d.Yprim)
    for i = 1:m.ncomponents
        Ycomb = isnothing(d.Yadd) ? Y : hcat(Y, d.Yadd)
        W₀ = candidate_loading_weights(d.X, Ycomb, obs_weights)

        Z₀_raw = candidate_scores(d.X, W₀)
        Z₀ = Z₀_raw
        if !isnothing(d.Yadd)
            Z₀ = orthogonalize_on_accumulated_scores(Z₀, T[:, 1:i-1])
            if norm(Z₀) ≤ sqrt(eps(Float64)) * max(norm(Z₀_raw), 1.0)
                Z₀ = Z₀_raw
            end
        end

        C, _, _ = cca_coeffs_and_corr(Z₀, Y, cca_obs_weights)
        W = loading_weights(W₀, C[:, 1])

        if m.multilinear
            W_modes_prev = [W_modes[j][:, 1:i-1] for j in eachindex(W_modes)]
            ml = multilinear_loading_weight_tensor(W, W_modes_prev, m, rng; verbose=verbose)

            for j in eachindex(W_modes)
                W_modes[j][:, i] = ml.factors[j]
            end

            Wᵒ = ml.Wᵒ
        else
            Wᵒ = W
        end

        selectdim(W_A, ndims(W_A), i) .= Wᵒ

        t_raw = score_vector(d.X, Wᵒ)
        t = orthogonalize_on_accumulated_scores(t_raw, T[:, 1:i-1])
        if norm(t) ≤ sqrt(eps(Float64)) * max(norm(t_raw), 1.0)
            t = t_raw
        end
        t = normalize_vector(t)
        T[:, i] = t

        Pᵢ = loading_tensor(d.X, t)
        selectdim(P, ndims(P), i) .= Pᵢ

        q = response_loading_vector(Y, t)
        Q[:, i] = q

        deflate_responses!(Y, t, q)
    end

    R = score_projection_tensors(W_A, P)
    B = regression_coefficients(R, Q)

    NCPLSFitLight(B, d.X_mean, d.X_std, d.Yprim_mean)
end