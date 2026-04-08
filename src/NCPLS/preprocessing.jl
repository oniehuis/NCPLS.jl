"""
    preprocess(
        m::NCPLSModel,
        X::AbstractArray{<:Real},
        Yprim::AbstractMatrix{<:Real},
        Yadd::Union{AbstractMatrix{<:Real}, Nothing},
        obs_weights::Union{AbstractVector{<:Real}, Nothing}
    )

Prepare predictors and responses for NCPLS by converting them to `Float64`, validating
their dimensions, and applying the centering and scaling options stored in `m`. `Yprim`
is centered but not scaled. `Yadd` is only converted to `Float64` and validated against
`Yprim`.
"""
function preprocess(
    m::NCPLSModel,
    X::AbstractArray{<:Real},
    Yprim::AbstractMatrix{<:Real},
    Yadd::Union{AbstractMatrix{<:Real}, Nothing},
    obs_weights::Union{AbstractVector{<:Real}, Nothing}
)

    ndims(X) ≥ 2 || throw(ArgumentError(
        "X must have at least 2 dimensions: samples × variables[/modes]"))

    nrow_X = size(X, 1)
    nrow_Yprim, _ = size(Yprim)

    nrow_X ≠ nrow_Yprim && throw(DimensionMismatch(
        "Number of rows in X and Yprim must be equal"))

    X, X_mean, X_std = centerscale(float64(X), m.center_X, m.scale_X, obs_weights)

    Yprim, Yprim_mean, Yprim_std = 
        centerscale(float64(Yprim), m.center_Yprim, false, obs_weights)

    if !isnothing(Yadd)
        nrow_Yadd, _ = size(Yadd)
        nrow_Yprim == nrow_Yadd || throw(DimensionMismatch(
            "Yprim and Yadd must have the same number of rows"))
        Yadd = float64(Yadd)
    end
    
    (   # Preprocessed predictors
        X=X,
        X_mean=X_mean, 
        X_std=X_std, 

        # Preprocessed primary responses
        Yprim=Yprim,
        Yprim_mean=Yprim_mean,
        Yprim_std=Yprim_std, 

        # Preprocessed additional responses
        Yadd=Yadd,
    )
end

"""
    centerscale(
        X::AbstractArray{<:Real},
        center::Bool,
        scale::Bool,
        obs_weights::Union{AbstractVector{<:Real}, Nothing}
    )

Center and scale `X` along the sample dimension. If `obs_weights` is provided, weighted
means and weighted standard deviations are used.
"""
function centerscale(
    X::AbstractArray{<:Real},
    center::Bool,
    scale::Bool,
    obs_weights::Union{AbstractVector{<:Real}, Nothing}
)

    ndims(X) ≥ 2 || throw(ArgumentError(
        "X must have at least 2 dimensions: samples × variables[/modes]"))
    
    validate_obs_weights(X, obs_weights)

    Xf = float64(X)
    n = size(Xf, 1)
    statshape = ntuple(d -> d == 1 ? 1 : size(Xf, d), ndims(Xf))

    if isnothing(obs_weights)
        if center
            μ = mean(Xf; dims=1)
            Xwork = Xf .- μ
        else
            μ = zeros(Float64, statshape)
            Xwork = Xf
        end

        if scale
            σ = if center
                sqrt.(sum(Xwork .^ 2; dims=1) / n)
            else
                μ0 = mean(Xf; dims=1)
                sqrt.(sum((Xf .- μ0) .^ 2; dims=1) / n)
            end
            σ = map(s -> isfinite(s) && s ≠ 0.0 ? s : 1.0, σ)
            Xwork = Xwork ./ σ
        else
            σ = ones(Float64, statshape)
        end
    else
        wsum = sum(obs_weights)
        w = reshape(float64(obs_weights), size(Xf, 1), ntuple(_ -> 1, ndims(Xf) - 1)...)

        if center
            μ = sum(Xf .* w; dims=1) / wsum
            Xwork = Xf .- μ
        else
            μ = zeros(Float64, statshape)
            Xwork = Xf
        end

        if scale
            σ = if center
                sqrt.(sum(w .* Xwork .^ 2; dims=1) / wsum)
            else
                μ0 = sum(Xf .* w; dims=1) / wsum
                sqrt.(sum(w .* (Xf .- μ0) .^ 2; dims=1) / wsum)
            end
            σ = map(s -> isfinite(s) && s ≠ 0.0 ? s : 1.0, σ)
            Xwork = Xwork ./ σ
        else
            σ = ones(Float64, statshape)
        end
    end

    Xwork, dropdims(μ; dims=1), dropdims(σ; dims=1)
end

"""
    validate_obs_weights(
        X::AbstractArray{<:Real},
        obs_weights::Union{AbstractVector{<:Real}, Nothing}
    )

Validate observation weights against the sample dimension of `X`. Returns `nothing` if
the weights are valid or absent.
"""
function validate_obs_weights(
    X::AbstractArray{<:Real},
    obs_weights::Union{AbstractVector{<:Real}, Nothing}
)
    isnothing(obs_weights) && return nothing

    length(obs_weights) == size(X, 1) || throw(DimensionMismatch(
        "Length of obs_weights must match size(X, 1)"))

    all(isfinite, obs_weights) || throw(ArgumentError(
        "obs_weights must contain only finite values"))

    any(w -> w < 0, obs_weights) && throw(ArgumentError(
        "obs_weights must be non-negative"))

    sum(obs_weights) > 0 || throw(ArgumentError(
        "obs_weights must sum to a positive value"))

    nothing
end

"""
    float64(X::AbstractArray{<:Real})

Convert `X` to `Float64` unless it already has element type `Float64`.
"""
float64(X::AbstractArray{T}) where {T<:Real} = T ≡ Float64 ? X : Float64.(X)
