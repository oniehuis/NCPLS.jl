"""
    cca_coeffs_and_corr(
        X::AbstractMatrix{<:Real},
        Y::AbstractMatrix{<:Real},
        obs_weights::Union{AbstractVector{<:Real}, Nothing},
    )

Compute canonical coefficient matrices for `X` and `Y` together with the leading
canonical correlation. Observation weights are applied by scaling rows before the CCA
decomposition, matching the weighting convention used in NCPLS fitting.
"""
function cca_coeffs_and_corr(
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    obs_weights::Union{AbstractVector{<:Real}, Nothing},
)
    n_rows, n_cols, qx, qy, dx, dy, left_singular_vectors, right_singular_vectors, rho =
        cca_decomposition(X, Y, obs_weights)

    k = min(dx, dy)

    a = qx.R[1:dx, 1:dx] \ left_singular_vectors[:, 1:k]
    a *= sqrt(n_rows - 1)
    remaining_rows = n_cols - size(a, 1)
    if remaining_rows > 0
        a = vcat(a, zeros(remaining_rows, k))
    end
    a = a[invperm(qx.p), :]

    b = qy.R[1:dy, 1:dy] \ right_singular_vectors[:, 1:k]
    b *= sqrt(n_rows - 1)
    remaining_rows = size(Y, 2) - size(b, 1)
    if remaining_rows > 0
        b = vcat(b, zeros(remaining_rows, k))
    end
    b = b[invperm(qy.p), :]

    a, b, rho
end

@inline function cca_decomposition(
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    ::Nothing,
)
    cca_decomposition(X, Y)
end

@inline function cca_decomposition(
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    obs_weights::AbstractVector{<:Real},
)
    cca_decomposition(X .* obs_weights, Y .* obs_weights)
end

function cca_decomposition(
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)
    n_rows, n_cols = size(X)

    qx = qr(X, ColumnNorm())
    qy = qr(Y, ColumnNorm())

    dx = rank(qx.R)
    dy = rank(qy.R)

    @inbounds if dx == 0
        throw(ErrorException("X has rank 0"))
    end
    @inbounds if dy == 0
        throw(ErrorException("Y has rank 0"))
    end

    A = ((qx.Q' * qy.Q) * rectangular_identity(n_rows, dy))[1:dx, :]
    left_singular_vecs, singular_vals, right_singular_vecs_t = svd(A; full = true)
    right_singular_vecs = right_singular_vecs_t'
    rho = clamp(first(singular_vals), 0.0, 1.0)

    n_rows, n_cols, qx, qy, dx, dy, left_singular_vecs, right_singular_vecs, rho
end

@inline function rectangular_identity(rowcount::Integer, columncount::Integer)
    M = zeros(rowcount, columncount)
    @inbounds for i = 1:min(rowcount, columncount)
        M[i, i] = 1
    end
    M
end
