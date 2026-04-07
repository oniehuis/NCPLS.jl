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

function orthogonalize_on_accumulated_scores(
    X::AbstractVecOrMat{Float64},
    T_A::AbstractMatrix{Float64},
)
    size(T_A, 1) == size(X, 1) || throw(DimensionMismatch(
        "T_A and X must have the same number of rows"))

    size(T_A, 2) == 0 && return X

    X - T_A * (T_A' * X)
end


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
    W = reshape(Wₘ, predictor_dims...)

    nw = norm(W)
    nw > 0 || throw(ArgumentError("Loading weights have zero norm"))

    W ./ nw
end

function score_vector(
    X::AbstractArray{<:Real},
    Wᵒ::AbstractArray{<:Real},
)
    size(Wᵒ) == size(X)[2:end] || throw(DimensionMismatch(
        "Dimensions of Wᵒ must match the non-sample dimensions of X"))

    reshape(X, size(X, 1), :) * vec(Wᵒ)
end

function normalize_score_vector(
    t::AbstractVector{Float64},
)
    t_norm = norm(t)
    t_norm > 0 || throw(ArgumentError("Score vector has zero norm"))

    t / t_norm
end

function loading_tensor(
    X::AbstractArray{Float64},
    t::AbstractVector{Float64},
)
    size(X, 1) == length(t) || throw(DimensionMismatch(
        "Length of t must match size(X, 1)"))

    Pₘ = reshape(X, size(X, 1), :)' * t
    reshape(Pₘ, size(X)[2:end]...)
end

function response_loading_vector(
    Yprim::AbstractMatrix{Float64},
    t::AbstractVector{Float64},
)
    size(Yprim, 1) == length(t) || throw(DimensionMismatch(
        "Length of t must match size(Yprim, 1)"))

    Yprim' * t
end

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


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

"""
    outer_tensor(factors)

Construct the rank-1 outer product tensor:
    a¹ ∘ a² ∘ ... ∘ aᵈ
from a vector of factor vectors.
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
    return T
end

"""
    contract_except(X, factors, mode)

Contract tensor X with all factor vectors except `mode`.

If X has size (K1, K2, ..., Kd), the result is a vector of length K_mode.
"""
function contract_except(X::AbstractArray, factors::AbstractVector{<:AbstractVector}, mode::Int)
    nd = ndims(X)
    @assert length(factors) == nd
    out = zeros(Float64, size(X, mode))

    for I in CartesianIndices(X)
        prod = 1.0
        @inbounds for m in 1:nd
            if m != mode
                prod *= factors[m][I[m]]
            end
        end
        @inbounds out[I[mode]] += X[I] * prod
    end

    return out
end

"""
    parafac_rank1(X; maxiter=500, tol=1e-10, init=:hosvd, rng=MersenneTwister(1), verbose=false)

Fit a rank-1 PARAFAC / CP model:
    X ≈ λ * a¹ ∘ a² ∘ ... ∘ aᵈ

Returns:
    (lambda, factors, Xhat, fit, relerr, niter, converged)
"""
function parafac_rank1(
    X::AbstractArray;
    maxiter::Int = 500,
    tol::Float64 = 1e-10,
    init::Symbol = :hosvd,
    rng = MersenneTwister(1),
    verbose::Bool = false,
)
    nd = ndims(X)
    dims = size(X)
    Xnorm = norm(X)
    Xnorm == 0 && error("Zero tensor not supported.")

    factors = Vector{Vector{Float64}}(undef, nd)

    if init == :random
        for m in 1:nd
            v = randn(rng, dims[m])
            factors[m] = v / norm(v)
        end
    elseif init == :hosvd
        for m in 1:nd
            perm = (m, filter(i -> i != m, 1:nd)...)
            Xm = permutedims(X, perm)
            Xmat = reshape(Xm, dims[m], :)
            U, _, _ = svd(Xmat)
            factors[m] = U[:, 1]
        end
    else
        error("Unknown init = $init. Use :random or :hosvd.")
    end

    relerr_old = Inf
    converged = false

    for iter in 1:maxiter
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

        if abs(relerr_old - relerr) < tol
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

    return (
        lambda = λ,
        factors = factors,
        Xhat = Xhat,
        fit = norm(Xhat)^2 / Xnorm^2,
        relerr = norm(X .- Xhat) / Xnorm,
        niter = maxiter,
        converged = converged,
    )
end

# ------------------------------------------------------------
# N-CPLS helper
# ------------------------------------------------------------

"""
    multilinear_weights(W; maxiter=500, tol=1e-10, init=:hosvd, rng=MersenneTwister(1))

Convert a full weight object W into multilinear mode-wise weights.

Behavior:
- if W is a vector: normalize it
- if W is a matrix: use rank-1 SVD
- if W is 3D or higher: use rank-1 PARAFAC

Returns a named tuple:
    (
        factors,     # Vector of mode-wise weight vectors
        W_rank1,     # rank-1 approximation tensor/matrix/vector
        lambda,      # scalar magnitude
        relerr,      # relative reconstruction error
        method       # :vector, :svd, or :parafac
    )

This is the helper you can drop into the multilinear branch of N-CPLS.
"""
function multilinear_weights(
    W::AbstractArray;
    maxiter::Int = 500,
    tol::Float64 = 1e-10,
    init::Symbol = :hosvd,
    rng = MersenneTwister(1),
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
        fit = parafac_rank1(
            Array{Float64}(W);
            maxiter=maxiter,
            tol=tol,
            init=init,
            rng=rng,
        )

        return (
            factors = fit.factors,
            W_rank1 = fit.Xhat,
            lambda = fit.lambda,
            relerr = fit.relerr,
            method = :parafac,
        )
    end
end
