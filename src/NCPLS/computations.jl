"""
    candidate_loading_weights(
        X::AbstractArray{<:Real},
        Y::AbstractMatrix{<:Real},
        obs_weights::Union{AbstractVector{<:Real}, Nothing}
    )

Form the candidate loading weights `W₀` for NCPLS by contracting the predictor array `X`
with the response matrix `Y` over the sample dimension. Observation weights are applied
along the same dimension when provided.
"""
function candidate_loading_weights(
    X::AbstractArray{<:Real},
    Y::AbstractMatrix{<:Real},
    obs_weights::Union{AbstractVector{<:Real}, Nothing}
)
    n = size(X, 1)
    Xₘ = reshape(X, n, :)

    W₀ₘ = if isnothing(obs_weights)
        Xₘ' * Y
    else
        w = reshape(float64(obs_weights), n, 1)
        (Xₘ .* w)' * Y
    end

    reshape(W₀ₘ, size(X)[2:end]..., size(Y, 2))
end

"""
    candidate_scores(
        X::AbstractArray{<:Real},
        W₀::AbstractArray{<:Real}
    )

Form the candidate score matrix `Z₀ = X ⓓ W₀`. The returned matrix has one row per sample
and one column per response direction in the last dimension of `W₀`.
"""
function candidate_scores(
    X::AbstractArray{<:Real},
    W₀::AbstractArray{<:Real},
)
    ndims(X) ≥ 2 || throw(ArgumentError(
        "X must have at least 2 dimensions: samples × variables[/modes]"))

    ndims(W₀) == ndims(X) || throw(DimensionMismatch(
        "W₀ must have the same number of dimensions as X"))

    size(W₀)[1:end-1] == size(X)[2:end] || throw(DimensionMismatch(
        "Predictor dimensions of W₀ must match the non-sample dimensions of X"))

    n = size(X, 1)
    q = size(W₀, ndims(W₀))

    Xₘ = reshape(X, n, :)
    W₀ₘ = reshape(W₀, :, q)

    Xₘ * W₀ₘ
end

"""
    orthogonalize_on_accumulated_scores(
        X::AbstractVecOrMat{Float64},
        T_A::AbstractMatrix{Float64}
    )

Orthogonalize the vector or matrix `X` on the accumulated score vectors stored in `T_A`
using the manuscript formula `X - T_A * (T_A' * X)`.
"""
function orthogonalize_on_accumulated_scores(
    X::AbstractVecOrMat{Float64},
    T_A::AbstractMatrix{Float64},
)
    size(T_A, 1) == size(X, 1) || throw(DimensionMismatch(
        "T_A and X must have the same number of rows"))

    size(T_A, 2) == 0 && return X

    X - T_A * (T_A' * X)
end

"""
    loading_weights(
        W₀::AbstractArray{<:Real},
        c::AbstractVector{<:Real}
    )

Collapse the candidate loading weights `W₀` with the dominant left canonical weight
vector `c` to form `W = W₀ ⓐ₁ c`.
"""
function loading_weights(
    W₀::AbstractArray{<:Real},
    c::AbstractVector{<:Real},
)
    q = size(W₀, ndims(W₀))
    q == length(c) || throw(DimensionMismatch(
        "Length of c must match the last dimension of W₀"))

    predictor_dims = size(W₀)[1:end-1]
    W₀ₘ = reshape(W₀, :, q)
    Wₘ = W₀ₘ * c
    reshape(Wₘ, predictor_dims...)
end

"""
    score_vector(
        X::AbstractArray{<:Real},
        Wᵒ::AbstractArray{<:Real}
    )

Form the score vector `t = X ⓓ Wᵒ` by contracting `X` with the loading-weight object
`Wᵒ` over all non-sample dimensions.
"""
function score_vector(
    X::AbstractArray{<:Real},
    Wᵒ::AbstractArray{<:Real},
)
    size(Wᵒ) == size(X)[2:end] || throw(DimensionMismatch(
        "Dimensions of Wᵒ must match the non-sample dimensions of X"))

    reshape(X, size(X, 1), :) * vec(Wᵒ)
end

"""
    normalize_vector(t::AbstractVector{Float64})

Return `t / norm(t)`. An `ArgumentError` is thrown if `t` has zero norm.
"""
function normalize_vector(
    t::AbstractVector{Float64},
)
    t_norm = norm(t)
    t_norm > 0 || throw(ArgumentError("Score vector has zero norm"))

    t / t_norm
end

"""
    loading_tensor(
        X::AbstractArray{Float64},
        t::AbstractVector{Float64}
    )

Form the loading tensor `P = Xᵗ_d ⓐ₁ t` by contracting `X` with the score vector `t`
over the sample dimension.
"""
function loading_tensor(
    X::AbstractArray{Float64},
    t::AbstractVector{Float64},
)
    size(X, 1) == length(t) || throw(DimensionMismatch(
        "Length of t must match size(X, 1)"))

    Pₘ = reshape(X, size(X, 1), :)' * t
    reshape(Pₘ, size(X)[2:end]...)
end

"""
    response_loading_vector(
        Yprim::AbstractMatrix{Float64},
        t::AbstractVector{Float64}
    )

Form the response loading vector `q = Y' * t` from the current primary-response matrix
and score vector.
"""
function response_loading_vector(
    Yprim::AbstractMatrix{Float64},
    t::AbstractVector{Float64},
)
    size(Yprim, 1) == length(t) || throw(DimensionMismatch(
        "Length of t must match size(Yprim, 1)"))

    Yprim' * t
end

"""
    deflate_responses!(
        Yprim::AbstractMatrix{Float64},
        t::AbstractVector{Float64},
        q::AbstractVector{Float64}
    )

Deflate the response matrix in place using the rank-1 update `Y := Y - t q'` and return
the mutated matrix.
"""
function deflate_responses!(
    Yprim::AbstractMatrix{Float64},
    t::AbstractVector{Float64},
    q::AbstractVector{Float64},
)
    size(Yprim, 1) == length(t) || throw(DimensionMismatch(
        "Length of t must match size(Yprim, 1)"))
    size(Yprim, 2) == length(q) || throw(DimensionMismatch(
        "Length of q must match size(Yprim, 2)"))

    Yprim .-= t * q'
    Yprim
end

"""
    score_projection_tensors(
        W_A::AbstractArray{Float64},
        P_A::AbstractArray{Float64}
    )

Form the score projection tensor `R = W_A ⓐ₁ (P_Aᵗ¹ ⓓ W_A)⁻¹` used for score
prediction after the component loop.
"""
function score_projection_tensors(
    W_A::AbstractArray{Float64},
    P_A::AbstractArray{Float64},
)
    size(W_A) == size(P_A) || throw(DimensionMismatch(
        "W_A and P_A must have the same dimensions"))

    A = size(W_A, ndims(W_A))
    W_Am = reshape(W_A, :, A)
    P_Am = reshape(P_A, :, A)

    M = P_Am' * W_Am
    Rm = W_Am * inv(M)

    reshape(Rm, size(W_A)...)
end

"""
    regression_coefficients(
        R::AbstractArray{Float64},
        Q_A::AbstractMatrix{Float64}
    )

Form the cumulative regression coefficients `B = cumsum(R ⊙₁ Q_Aᵗ)` along the component
dimension.
"""
function regression_coefficients(
    R::AbstractArray{Float64},
    Q_A::AbstractMatrix{Float64},
)
    A = size(R, ndims(R))
    size(Q_A, 2) == A || throw(DimensionMismatch(
        "The number of columns in Q_A must match the component dimension of R"))

    predictor_dims = size(R)[1:end-1]
    M = size(Q_A, 1)

    R_exp = reshape(R, predictor_dims..., A, 1)
    Q_exp = reshape(
        permutedims(Q_A),                      # A × M
        ntuple(_ -> 1, length(predictor_dims))...,
        A,
        M,
    )

    cumsum(R_exp .* Q_exp; dims = length(predictor_dims) + 1)
end

"""
    multilinear_loading_weight_tensor(W, W_modes_prev, m, rng; verbose=false)

Apply the multilinear loading-weight branch of the NCPLS algorithm to `W`. The factors
are extracted according to the dimensionality of `W`, optionally orthogonalized on the
previous mode weights stored in `W_modes_prev`, normalized, and finally recombined into
`Wᵒ` as an outer product.
"""
function multilinear_loading_weight_tensor(
    W::AbstractArray{Float64},
    W_modes_prev::AbstractVector{<:AbstractMatrix{Float64}},
    m::NCPLSModel,
    rng::MersenneTwister;
    verbose::Bool=false)

    ml = multilinear_weights(W, m, rng; verbose=verbose)

    factors = [copy(f) for f in ml.factors]

    for j in eachindex(factors)
        if m.orthogonalize_mode_weights
            factors[j] = orthogonalize_on_accumulated_scores(factors[j], W_modes_prev[j])
        end
        factors[j] = normalize_vector(factors[j])
    end

    (
        factors = factors,
        Wᵒ = outer_tensor(factors),
        lambda = ml.lambda,
        relerr = ml.relerr,
        method = ml.method,
    )
end

"""
    multilinear_weights(W, m, rng; verbose=false)

Extract multilinear mode weights from `W` using the control settings stored in `m`. A
vector is normalized directly, a matrix is reduced by its dominant rank-1 SVD pair, and
an array with three or more dimensions is approximated by a rank-1 PARAFAC model. The
returned named tuple contains `factors`, `W_rank1`, `lambda`, `relerr`, and `method`.
"""
function multilinear_weights(
    W::AbstractArray, 
    m::NCPLSModel, 
    rng::MersenneTwister;
    verbose::Bool=false
)
    d = ndims(W)

    if d == 1
        w = collect(Float64, W)
        nw = norm(w)
        nw == 0 && error("Zero vector W cannot be normalized.")
        w ./= nw

        return (
            factors = [w],
            W_rank1 = w,
            lambda = 1.0,
            relerr = 0.0,
            method = :vector,
        )
    elseif d == 2
        U, S, V = svd(Matrix{Float64}(W))
        λ = S[1]
        w1 = U[:, 1]
        w2 = V[:, 1]
        What = λ .* (w1 * w2')

        return (
            factors = [w1, w2],
            W_rank1 = What,
            lambda = λ,
            relerr = norm(W .- What) / norm(W),
            method = :svd,
        )
    else
        pf = parafac_rank1(Array{Float64}(W), m, rng; verbose=verbose)


        return (
            factors = pf.factors,
            W_rank1 = pf.Xhat,
            lambda = pf.lambda,
            relerr = pf.relerr,
            method = :parafac,
        )
    end
end

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

"""
    outer_tensor(factors)

Construct the rank-1 outer product tensor `a¹ ∘ a² ∘ ... ∘ aᵈ` from the vectors in
`factors`.
"""
function outer_tensor(factors::AbstractVector{<:AbstractVector})
    dims = map(length, factors)
    T = Array{Float64}(undef, dims...)
    for I in CartesianIndices(T)
        v = 1.0
        @inbounds for m in eachindex(factors)
            v *= factors[m][I[m]]
        end
        T[I] = v
    end
    T
end

"""
    contract_except(X, factors, mode)

Contract `X` with all vectors in `factors` except the one indexed by `mode`. This is the
alternating least-squares update used by `parafac_rank1`.
"""
function contract_except(
    X::AbstractArray, 
    factors::AbstractVector{<:AbstractVector}, 
    mode::Int
)
    nd = ndims(X)
    @assert length(factors) == nd
    out = zeros(Float64, size(X, mode))

    for I in CartesianIndices(X)
        prod = 1.0
        @inbounds for m in 1:nd
            if m ≠ mode
                prod *= factors[m][I[m]]
            end
        end
        @inbounds out[I[mode]] += X[I] * prod
    end

    out
end

"""
    parafac_rank1(X, m, rng; verbose=false)

Fit the rank-1 PARAFAC model `X ≈ λ * a¹ ∘ a² ∘ ... ∘ aᵈ` using the initialization,
iteration limit, and convergence tolerance stored in `m`. The returned named tuple
contains `lambda`, `factors`, `Xhat`, `fit`, `relerr`, `niter`, and `converged`.
"""
function parafac_rank1(
    X::AbstractArray, 
    m::NCPLSModel, 
    rng::MersenneTwister; 
    verbose::Bool=false
)
    nd = ndims(X)
    dims = size(X)
    Xnorm = norm(X)
    Xnorm == 0 && error("Zero tensor not supported.")

    factors = Vector{Vector{Float64}}(undef, nd)

    if m.multilinear_init ≡ :random
        for m in 1:nd
            v = randn(rng, dims[m])
            factors[m] = v / norm(v)
        end
    elseif m.multilinear_init ≡ :hosvd
        for m in 1:nd
            perm = (m, filter(i -> i ≠ m, 1:nd)...)
            Xm = permutedims(X, perm)
            Xmat = reshape(Xm, dims[m], :)
            U, _, _ = svd(Xmat)
            factors[m] = U[:, 1]
        end
    else
        error("Unknown init = $(m.multilinear_init). Use :random or :hosvd.")
    end

    relerr_old = Inf
    converged = false

    for iter in 1:m.multilinear_maxiter
        for m in 1:nd
            v = contract_except(X, factors, m)
            nv = norm(v)
            nv == 0 && error("Degenerate update encountered in mode $m.")
            factors[m] = v / nv
        end

        rank1_unit = outer_tensor(factors)
        λ = sum(X .* rank1_unit)
        Xhat = λ .* rank1_unit
        relerr = norm(X .- Xhat) / Xnorm

        if verbose
            @info "iter=$iter relerr=$relerr λ=$λ"
        end

        if abs(relerr_old - relerr) < m.multilinear_tol
            converged = true
            return (
                lambda = λ,
                factors = factors,
                Xhat = Xhat,
                fit = norm(Xhat)^2 / Xnorm^2,
                relerr = relerr,
                niter = iter,
                converged = converged,
            )
        end

        relerr_old = relerr
    end

    rank1_unit = outer_tensor(factors)
    λ = sum(X .* rank1_unit)
    Xhat = λ .* rank1_unit

    (
        lambda = λ,
        factors = factors,
        Xhat = Xhat,
        fit = norm(Xhat)^2 / Xnorm^2,
        relerr = norm(X .- Xhat) / Xnorm,
        niter = m.multilinear_maxiter,
        converged = converged
    )
end
