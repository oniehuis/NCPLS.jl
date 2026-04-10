"""
    fit(
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
    ) -> NCPLSFit

Fit an NCPLS model to predictors `X` and primary responses `Yprim`. The model object `m`
controls the number of components, centering and scaling of `X`, centering of `Yprim`,
whether the multilinear loading-weight branch is used, whether mode weights are
orthogonalized on previous components, and the PARAFAC control settings
`multilinear_maxiter`, `multilinear_tol`, `multilinear_init`, and
`multilinear_seed`.

`Yadd` may be used to supply additional responses that influence the loading-weight
calculation but are not predicted. `obs_weights` applies sample weights during
preprocessing and the candidate-weight and CCA steps. Optional `samplelabels`,
`responselabels`, `sampleclasses`, and `predictoraxes` metadata are stored on the fitted
object for downstream plotting and interpretation. If `verbose=true`, iteration progress
from the PARAFAC step is printed when `m.multilinear` is enabled.
"""
function fit(
    m::NCPLSModel,
    X::AbstractArray{<:Real},
    Yprim::AbstractMatrix{<:Real};
    kwargs...
)
    fit_ncpls_core(m, X, Yprim; kwargs...)
end

"""
    fit_ncpls_core(
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
    ) -> NCPLSFit

Fit an NCPLS model and return an `NCPLSFit`.

This is the low-level fitting routine used by [`fit`](@ref). `Yadd` supplies
additional responses for the loading-weight calculation, `obs_weights` gives
optional observation weights, `samplelabels`, `responselabels`, `sampleclasses`, and
`predictoraxes` are stored as fit metadata, and `verbose` controls PARAFAC iteration
logging in multilinear fits.
"""
function fit_ncpls_core(
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

    # Preprocess data: center/scale, optionally with weights.
    d = preprocess(m, X, Yprim, Yadd, obs_weights)

    samplelabels = default_sample_labels(
        validate_label_length(samplelabels, size(d.X, 1), "samplelabels"),
        size(d.X, 1),
    )
    responselabels = normalize_string_labels(
        responselabels,
        size(d.Yprim, 2),
        "responselabels",
    )
    sampleclasses = normalize_sampleclasses(sampleclasses, size(d.X, 1))
    predictoraxes = normalize_predictoraxes_metadata(
        predictoraxes,
        size(d.X)[2:end],
    )

    # Preallocate arrays for scores, loadings, regression coefficients, and diagnostics.
    T = zeros(Float64, size(d.X, 1), m.ncomponents)
    P = Array{Float64}(undef, size(d.X)[2:end]..., m.ncomponents)
    Q = Matrix{Float64}(undef, size(d.Yprim, 2), m.ncomponents)
    W_A = Array{Float64}(undef, size(d.X)[2:end]..., m.ncomponents)
    q_comb = size(d.Yprim, 2) + (isnothing(d.Yadd) ? 0 : size(d.Yadd, 2))
    W0 = Array{Float64}(undef, size(d.X)[2:end]..., q_comb, m.ncomponents)
    c = Matrix{Float64}(undef, q_comb, m.ncomponents)
    rho = Vector{Float64}(undef, m.ncomponents)
    if m.multilinear
        W_modes = [
            Array{Float64}(undef, size(d.X, j + 1), m.ncomponents)
            for j in 1:(ndims(d.X) - 1)
        ]
        W_multilinear_relerr = Vector{Float64}(undef, m.ncomponents)
        W_multilinear_method = Vector{Symbol}(undef, m.ncomponents)
        W_multilinear_lambda = Vector{Float64}(undef, m.ncomponents)
        W_multilinear_niter = Vector{Int}(undef, m.ncomponents)
        W_multilinear_converged = Vector{Bool}(undef, m.ncomponents)
    else
        W_modes = nothing
        W_multilinear_relerr = nothing
        W_multilinear_method = nothing
        W_multilinear_lambda = nothing
        W_multilinear_niter = nothing
        W_multilinear_converged = nothing
    end
    rng = m.multilinear ? MersenneTwister(m.multilinear_seed) : nothing

    # Apply observation weights consistently with covariance weighting (sqrt for covariance).
    cca_obs_weights = isnothing(obs_weights) ? nothing : sqrt.(obs_weights)

    # Main loop over components: compute weights, scores, loadings, deflate, and 
    # store results.
    Y = copy(d.Yprim)
    for i = 1:m.ncomponents
        # W₀ = Xᵗ_d ⓐ₁ [Y Yadditional]
        Ycomb = isnothing(d.Yadd) ? Y : hcat(Y, d.Yadd)
        W₀ = candidate_loading_weights(d.X, Ycomb, obs_weights)
        selectdim(W0, ndims(W0), i) .= W₀

        #Z₀ = X ⓓ W₀
        Z₀ = candidate_scores(d.X, W₀)
        # Z₀ := Z₀ - T_A T_Aᵗ Z₀ only when Yadditional is used
        if !isnothing(d.Yadd)
            Z₀ = orthogonalize_on_accumulated_scores(Z₀, T[:, 1:i-1])
        end

        # C ⇐ canoncorr(Z₀, Y)
        C, _, rho[i] = cca_coeffs_and_corr(Z₀, Y, cca_obs_weights)
        c[:, i] = C[:, 1]

        # W = W₀ ⓐ₁ C
        W = loading_weights(W₀, c[:, i])

        if m.multilinear
            W_modes_prev = [W_modes[j][:, 1:i-1] for j in eachindex(W_modes)]

            ml = multilinear_loading_weight_tensor(W, W_modes_prev, m, rng; verbose=verbose)

            for j in eachindex(W_modes)
                W_modes[j][:, i] = ml.factors[j]
            end
            W_multilinear_relerr[i] = ml.relerr
            W_multilinear_method[i] = ml.method
            W_multilinear_lambda[i] = ml.lambda
            W_multilinear_niter[i] = ml.niter
            W_multilinear_converged[i] = ml.converged

            Wᵒ = ml.Wᵒ
        else # unfolded branch
            # For unfolded analyses, the manuscript bypasses the switching part and sets Wᵒ := W.
            Wᵒ = W
        end

        selectdim(W_A, ndims(W_A), i) .= Wᵒ

        # t = X ⓓ Wᵒ
        t = score_vector(d.X, Wᵒ)
        # t = X ⓓ Wᵒ
        t  = orthogonalize_on_accumulated_scores(t,  T[:, 1:i-1])
        # t := t / ||t||
        t = normalize_vector(t)
        T[:, i] = t

        # P = Xᵗ_d ⓐ₁ t
        Pᵢ = loading_tensor(d.X, t)
        selectdim(P, ndims(P), i) .= Pᵢ

        # q = Yᵗ t
        q = response_loading_vector(Y, t)
        Q[:, i] = q

        # Y := Y - t qᵗ
        deflate_responses!(Y, t, q)
    end

    #R = W_A ⓐ₁ (P_Aᵗ¹ ⓓ W_A)⁻¹
    R = score_projection_tensors(W_A, P)
    
    # B = cumsum(R ⊙₁ Q_Aᵗ)
    B = regression_coefficients(R, Q)
    
    NCPLSFit(
        m,
        B,
        R,
        T,
        P,
        Q,
        W_A,
        W_modes,
        c,
        W0,
        rho,
        Y,
        W_multilinear_relerr,
        W_multilinear_method,
        W_multilinear_lambda,
        W_multilinear_niter,
        W_multilinear_converged,
        d.X_mean,
        d.X_std,
        d.Yprim_mean,
        samplelabels = samplelabels,
        responselabels = responselabels,
        sampleclasses = sampleclasses,
        predictoraxes = predictoraxes,
    )
end

function validate_label_length(
    labels::AbstractVector,
    expected::Integer,
    name::AbstractString,
)
    isempty(labels) || length(labels) == expected || throw(ArgumentError(
        "`$name` must have length $expected, got $(length(labels))"))
    labels
end

function default_sample_labels(labels::AbstractVector, n_samples::Integer)
    isempty(labels) ? string.(1:n_samples) : string.(labels)
end

function normalize_string_labels(
    labels::AbstractVector,
    expected::Integer,
    name::AbstractString,
)
    validate_label_length(labels, expected, name)
    isempty(labels) ? String[] : string.(labels)
end

function normalize_sampleclasses(
    sampleclasses::Union{AbstractVector, Nothing},
    n_samples::Integer,
)
    isnothing(sampleclasses) && return nothing
    length(sampleclasses) == n_samples || throw(ArgumentError(
        "`sampleclasses` must have length $n_samples, got $(length(sampleclasses))"))
    collect(sampleclasses)
end

function normalize_predictoraxes_metadata(
    predictoraxes,
    predictor_dims::NTuple{N, Int},
) where {N}
    (isnothing(predictoraxes) || isempty(predictoraxes)) && return PredictorAxis[]

    length(predictoraxes) == N || throw(ArgumentError(
        "`predictoraxes` must contain $N axis descriptions, got $(length(predictoraxes))"))

    axes = PredictorAxis[]
    for (j, axis_meta) in enumerate(predictoraxes)
        axis = normalize_predictoraxis(axis_meta)
        length(axis.values) == predictor_dims[j] || throw(ArgumentError(
            "`predictoraxes[$j].values` must have length $(predictor_dims[j]), " *
            "got $(length(axis.values))"))
        push!(axes, axis)
    end

    axes
end

normalize_predictoraxis(axis::PredictorAxis) = axis

function normalize_predictoraxis(axis)
    props = propertynames(axis)
    (:name in props && :values in props) || throw(ArgumentError(
        "Each predictor axis must provide `name` and `values` fields"))
    unit = :unit in props ? getproperty(axis, :unit) : nothing
    PredictorAxis(
        string(getproperty(axis, :name)),
        getproperty(axis, :values);
        unit = isnothing(unit) ? nothing : string(unit),
    )
end
