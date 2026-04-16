"""
    predict(
        mf::AbstractNCPLSFit,
        X::AbstractArray{<:Real},
        ncomps::Integer=ncomponents(mf)
    ) -> Array{Float64, 3}

Predict the response matrix for each sample in `X` using the fitted NCPLS model. The
result has size `(n_samples, ncomponents, n_responses)`, where slice `[:, a, :]`
contains the predictions formed from the first `a` components. The result is always
numeric and always contains the full response block. Even for discriminant or mixed
response fits, `predict` does not apply an `argmax`; use [`onehot`](@ref),
[`sampleclasses`](@ref), or [`predictclasses`](@ref) when class labels should be decoded
from a class-score sub-block.
"""
function predict(
    mf::AbstractNCPLSFit,
    X::AbstractArray{<:Real},
    ncomps::Integer=ncomponents(mf)
)
    ncomps = validate_ncomponents(mf, ncomps)

    ndims(X) ≥ 2 || throw(ArgumentError(
        "X must have at least 2 dimensions: samples × variables[/modes]"))

    predictor_dims = size(mf.B)[1:end-2]
    size(X)[2:end] == predictor_dims || throw(DimensionMismatch(
        "Predictor dimensions of X must match the fitted model"))

    n = size(X, 1)
    A = ncomponents(mf)
    M = size(mf.B, ndims(mf.B))

    Xnorm = normalize_predictors(X, mf)
    Xmat = reshape(Xnorm, n, :)
    Bmat = reshape(mf.B, :, A * M)

    Yhat = reshape(Xmat * Bmat, n, A, M)
    Yhat = @views Yhat[:, 1:ncomps, :]

    restore_response_scale(Yhat, mf; add_mean=true)
end

"""
    project(mf::AbstractNCPLSFit, X::AbstractArray{<:Real}) -> Matrix{Float64}

Compute latent component X scores by projecting new predictors `X` with a fitted NCPLS
model. The predictors are centered and scaled using the stored preprocessing statistics
and then multiplied by the unfolded score projection tensor `R`.
"""
function project(
    mf::AbstractNCPLSFit,
    X::AbstractArray{<:Real},
)
    ndims(X) ≥ 2 || throw(ArgumentError(
        "X must have at least 2 dimensions: samples × variables[/modes]"))

    predictor_dims = size(mf.B)[1:end-2]
    size(X)[2:end] == predictor_dims || throw(DimensionMismatch(
        "Predictor dimensions of X must match the fitted model"))

    n = size(X, 1)
    A = ncomponents(mf)

    Xnorm = normalize_predictors(X, mf)
    Xmat = reshape(Xnorm, n, :)
    Rmat = reshape(mf.R, :, A)

    Xmat * Rmat
end

"""
    normalize_predictors(X::AbstractArray{<:Real}, mf::AbstractNCPLSFit)

Center and scale predictor data using the preprocessing statistics stored in the fitted
NCPLS model.
"""
function normalize_predictors(
    X::AbstractArray{<:Real},
    mf::AbstractNCPLSFit
)
    predictor_dims = size(X)[2:end]
    Xmean = reshape(mf.X_mean, 1, predictor_dims...)
    Xstd = reshape(mf.X_std, 1, predictor_dims...)

    (float64(X) .- Xmean) ./ Xstd
end
