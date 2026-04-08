function split_synthetic_gcms_samples(
    nsamples::Integer;
    test_fraction::Real=0.3,
    rng::AbstractRNG=MersenneTwister(1),
    train_idx::Union{AbstractVector{<:Integer}, Nothing}=nothing,
    test_idx::Union{AbstractVector{<:Integer}, Nothing}=nothing,
)
    0 < test_fraction < 1 || throw(ArgumentError(
        "test_fraction must lie strictly between 0 and 1"))
    nsamples > 1 || throw(ArgumentError("At least two samples are required"))

    if isnothing(train_idx) && isnothing(test_idx)
        n_test = clamp(round(Int, test_fraction * nsamples), 1, nsamples - 1)
        perm = randperm(rng, nsamples)
        test = sort(collect(perm[1:n_test]))
        train = sort(collect(perm[(n_test + 1):end]))
        return train, test
    end

    if isnothing(train_idx)
        test = sort(Int.(collect(test_idx)))
        train = sort(setdiff(collect(1:nsamples), test))
    elseif isnothing(test_idx)
        train = sort(Int.(collect(train_idx)))
        test = sort(setdiff(collect(1:nsamples), train))
    else
        train = sort(Int.(collect(train_idx)))
        test = sort(Int.(collect(test_idx)))
    end

    all(1 ≤ idx ≤ nsamples for idx in train) || throw(ArgumentError(
        "train_idx contains indices outside 1:$nsamples"))
    all(1 ≤ idx ≤ nsamples for idx in test) || throw(ArgumentError(
        "test_idx contains indices outside 1:$nsamples"))
    allunique(train) || throw(ArgumentError("train_idx must not contain duplicates"))
    allunique(test) || throw(ArgumentError("test_idx must not contain duplicates"))
    isempty(intersect(train, test)) || throw(ArgumentError(
        "train_idx and test_idx must be disjoint"))
    !isempty(train) || throw(ArgumentError("train_idx must not be empty"))
    !isempty(test) || throw(ArgumentError("test_idx must not be empty"))
    sort(vcat(train, test)) == collect(1:nsamples) || throw(ArgumentError(
        "train_idx and test_idx must partition all samples"))

    train, test
end

function synthetic_gcms_labels(data)
    X = data.X
    ndims(X) == 3 || throw(ArgumentError(
        "Synthetic GC-MS utilities currently expect a 3-dimensional predictor tensor"))

    predictorlabels = vec([string("rt", rt, "_mz", mz)
        for rt in 1:size(X, 2), mz in 1:size(X, 3)])

    responselabels = if hasproperty(data, :target_compounds)
        [string("compound_", idx) for idx in data.target_compounds]
    else
        [string("response_", j) for j in 1:size(data.Yprim, 2)]
    end

    predictorlabels, responselabels
end

function components_last_predictions(
    Yhat::AbstractArray{<:Real, 3},
    components_dim::Integer,
)
    components_dim == 3 && return Array{Float64, 3}(Yhat)
    components_dim == 2 && return permutedims(Array{Float64, 3}(Yhat), (1, 3, 2))
    throw(ArgumentError("components_dim must be 2 or 3"))
end

function componentwise_regression_metrics(
    Yhat_components_last::AbstractArray{<:Real, 3},
    Y::AbstractMatrix{<:Real},
)
    n, m, a = size(Yhat_components_last)
    size(Y) == (n, m) || throw(DimensionMismatch(
        "Yhat and Y must agree in sample and response dimensions"))

    rmse = Matrix{Float64}(undef, m, a)
    r2 = Matrix{Float64}(undef, m, a)
    rmse_overall = Vector{Float64}(undef, a)
    r2_overall = Vector{Float64}(undef, a)

    y_mean = vec(mean(Y; dims = 1))
    sst = vec(sum((Y .- reshape(y_mean, 1, :)) .^ 2; dims = 1))
    sst_total = sum((Y .- mean(Y)) .^ 2)

    for i in 1:a
        pred = @view Yhat_components_last[:, :, i]
        err = pred .- Y

        rmse[:, i] = vec(sqrt.(mean(err .^ 2; dims = 1)))
        sse = vec(sum(err .^ 2; dims = 1))
        r2[:, i] = [sstj > 0 ? 1 - ssej / sstj : NaN for (ssej, sstj) in zip(sse, sst)]

        rmse_overall[i] = sqrt(mean(err .^ 2))
        r2_overall[i] = sst_total > 0 ? 1 - sum(err .^ 2) / sst_total : NaN
    end

    (
        rmse = rmse,
        r2 = r2,
        rmse_overall = rmse_overall,
        r2_overall = r2_overall,
    )
end

function safe_correlation(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    length(x) == length(y) || throw(DimensionMismatch(
        "x and y must have the same length"))
    length(x) > 1 || return NaN

    sx = std(x)
    sy = std(y)
    (sx > 0 && sy > 0) || return NaN

    cor(x, y)
end

function comparison_winner(x::Real, y::Real; atol::Real=1e-10, lower_is_better::Bool=true)
    if isapprox(x, y; atol = atol, rtol = 0)
        return :tie
    elseif lower_is_better
        return x < y ? :cppls : :ncpls
    else
        return x > y ? :cppls : :ncpls
    end
end
