using LinearAlgebra
using Random

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
