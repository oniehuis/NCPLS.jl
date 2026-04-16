"""
    fit(
        m::NCPLSModel,
        X::AbstractArray{<:Real},
        Yprim;
        Yadd::Union{AbstractMatrix{<:Real}, Nothing}=nothing,
        obs_weights::Union{AbstractVector{<:Real}, Nothing}=nothing,
        samplelabels::AbstractVector=String[],
        responselabels::AbstractVector=String[],
        sampleclasses::Union{AbstractVector, Nothing}=nothing,
        predictoraxes=(),
        verbose::Bool=false
    ) -> NCPLSFit

Fit a NCPLS model using the StatsAPI entry point and an explicit NCPLSModel. The model
specification supplies the number of components, centering and scaling, whether the 
multilinear loading-weight branch is used, whether mode weights are orthogonalized on 
previous components, and the PARAFAC control settings `multilinear_maxiter`,
`multilinear_tol`, `multilinear_init`, and `multilinear_seed`, while 
the call to `fit` supplies data, optional weights, additional responses, and label metadata.

The interpretation of the third argument `Yprim` depends on its type. When `Yprim` is an
`AbstractMatrix{<:Real}`, it is treated as a user-supplied response block and used as-is;
it may contain arbitrary combinations of continuous responses, one-hot encoded class
indicators, or other custom encodings. When `Yprim` is an `AbstractVector{<:Real}`, it is
interpreted as a univariate numeric response and internally reshaped to a one-column
matrix, corresponding to regression-style fitting. When `Yprim` is an
`AbstractCategoricalArray`, it is interpreted as a vector of class labels; the labels are
converted internally to a one-hot encoded response matrix and class names are inferred as
response labels.

Keyword arguments accepted by `fit` include `obs_weights` for per-sample weighting and
`Yadd` for additional response columns. `Yadd` must have the same number of rows as `X`
and is concatenated internally to `Yprim` to build the supervised projection, while
prediction targets always remain the primary responses.

Optional `samplelabels`, `responselabels`, `sampleclasses`, and `predictoraxes` are stored
on the fitted object for downstream plotting and interpretation. When `Yprim` is a numeric
matrix or vector, `sampleclasses` can be used as grouping metadata. When
`sampleclasses` and `responselabels` match a one-hot class block in `Yprim`, the fitted
model also enables class-prediction helpers on that block. When class labels are passed
positionally as a categorical array, they define the supervised response and are also
stored as metadata. For custom response matrices this matching is validated as follows:
if none of the unique `sampleclasses` occur in `responselabels`, no class block is
inferred and the labels remain metadata only; if only some occur, `fit` throws an error;
if all occur, each must appear exactly once and the matched response columns must form a
row-wise consistent one-hot block. If `verbose=true`,
iteration progress from the PARAFAC step is printed when `m.multilinear` is enabled.

The return value is a `NCPLSFit` containing scores, loadings, regression coefficients,
and the metadata needed for downstream prediction and diagnostics. Use `NCPLS.fit` or
`StatsAPI.fit` when disambiguation is required in your namespace.
"""
function fit(
    m::NCPLSModel,
    X::AbstractArray{<:Real},
    Yprim::AbstractMatrix{<:Real};
    kwargs...
)
    fit_ncpls(m, X, Yprim; kwargs...)
end

function fit(
    m::NCPLSModel,
    X::AbstractArray{<:Real},
    Yprim::AbstractVector{<:Real};
    kwargs...
)
    fit_ncpls(m, X, Yprim; kwargs...)
end

function fit(
    m::NCPLSModel,
    X::AbstractArray{<:Real},
    Yprim::AbstractCategoricalArray{T,1,R,V,C,U};
    kwargs...
) where {T,R,V,C,U}

    fit_ncpls(m, X, Yprim; kwargs...)
end


function fit_ncpls(
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
    T1<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing},
    T2<:Union{AbstractVector{<:Real}, Nothing},
    T3<:AbstractVector,
    T4<:AbstractVector,
    T5<:Union{AbstractVector, Nothing}
}

    fit_ncpls_core(m, X, Yprim;
        Yadd=Yadd, 
        obs_weights=obs_weights, 
        samplelabels=samplelabels,
        responselabels=responselabels,
        sampleclasses=sampleclasses,
        predictoraxes=predictoraxes, 
        verbose=verbose
    )
end

function fit_ncpls(
    m::NCPLSModel,
    X::AbstractArray{<:Real},
    Yprim::AbstractVector{<:Real};
    Yadd::T1=nothing,
    obs_weights::T2=nothing,
    samplelabels::T3=String[],
    responselabels::T4=String[],
    sampleclasses::T5=nothing,
    predictoraxes=(),
    verbose::Bool=false
) where {
    T1<:Union{LinearAlgebra.AbstractVecOrMat, Nothing},
    T2<:Union{AbstractVector{<:Real}, Nothing},
    T3<:AbstractVector,
    T4<:AbstractVector,
    T5<:Union{AbstractVector, Nothing}
}

    Yprim_matrix = reshape(Yprim, :, 1)

    fit_ncpls_core(m, X, Yprim_matrix;
        Yadd=Yadd, 
        obs_weights=obs_weights, 
        samplelabels=samplelabels,
        responselabels=responselabels,
        sampleclasses=sampleclasses,
        predictoraxes=predictoraxes, 
        verbose=verbose
    )
end

function fit_ncpls(
    m::NCPLSModel,
    X::AbstractArray{<:Real},
    sampleclasses::AbstractCategoricalArray{T,1,R,V,C,U};
    Yadd::T1=nothing,
    obs_weights::T2=nothing,
    samplelabels::T3=String[],
    responselabels::T4=String[],
    predictoraxes=(),
    verbose::Bool=false
) where {
    T, R, V, C, U,
    T1<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing},
    T2<:Union{AbstractVector{<:Real}, Nothing},
    T3<:AbstractVector,
    T4<:AbstractVector
}
    isempty(responselabels) || throw(ArgumentError("`responselabels` cannot be provided" *
        " when passing sample classes; response labels are inferred automatically."))
    
    Yprim, classes = onehot(sampleclasses)

    fit_ncpls_core(m, X, Yprim;
        Yadd=Yadd, 
        obs_weights=obs_weights, 
        samplelabels=samplelabels,
        responselabels=classes,
        sampleclasses=copy(sampleclasses),
        predictoraxes=predictoraxes, 
        verbose=verbose
    )
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

    samplelabels = default_sample_labels(
        validate_label_length(samplelabels, size(X, 1), "samplelabels"),
        size(X, 1),
    )
    responselabels = normalize_string_labels(
        responselabels,
        size(Yprim, 2),
        "responselabels",
    )
    sampleclasses = normalize_sampleclasses(sampleclasses, size(X, 1))
    predictoraxes = normalize_predictoraxes_metadata(
        predictoraxes,
        size(X)[2:end],
    )
    validate_class_response_metadata(Yprim, sampleclasses, responselabels)

    # Preprocess data: center/scale, optionally with weights.
    d = preprocess(m, X, Yprim, Yadd, obs_weights)

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

        # Form candidate scores for the supervised CCA step. When Yadd is present,
        # the manuscript orthogonalizes these scores on previous components. For
        # rank-deficient toy cases that orthogonalization can collapse to numerical
        # zero on some BLAS/LAPACK combinations, so fall back to the raw candidate
        # scores when that happens.
        Z₀_raw = candidate_scores(d.X, W₀)
        Z₀ = Z₀_raw
        if !isnothing(d.Yadd)
            Z₀ = orthogonalize_on_accumulated_scores(Z₀, T[:, 1:i-1])
            if norm(Z₀) ≤ sqrt(eps(Float64)) * max(norm(Z₀_raw), 1.0)
                Z₀ = Z₀_raw
            end
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

        # Form the score vector and orthogonalize it on previous components.
        # In rank-deficient toy cases the orthogonalized vector can collapse to
        # numerical zero; fall back to the raw score direction in that case so
        # component extraction remains well-defined across Julia versions.
        t_raw = score_vector(d.X, Wᵒ)
        t = orthogonalize_on_accumulated_scores(t_raw, T[:, 1:i-1])
        if norm(t) ≤ sqrt(eps(Float64)) * max(norm(t_raw), 1.0)
            t = t_raw
        end
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

function decode_one_hot_indices(one_hot_matrix::AbstractMatrix{<:Real})
    all(value -> (value == 0) || (value == 1), one_hot_matrix) || throw(ArgumentError(
        "one_hot_matrix must contain only 0/1 entries"))

    row_sums = vec(sum(one_hot_matrix; dims=2))
    all(==(1), row_sums) || throw(ArgumentError(
        "each row of one_hot_matrix must contain exactly one 1"))

    [argmax(row) for row in eachrow(one_hot_matrix)]
end

function is_one_hot_matrix(Y::AbstractMatrix{<:Real})
    all(value -> isapprox(value, 0; atol=1e-12) || isapprox(value, 1; atol=1e-12), Y) ||
        return false
    row_sums = vec(sum(Y; dims=2))
    all(sum_i -> isapprox(sum_i, 1; atol=1e-12), row_sums)
end

function class_response_columns(
    sampleclasses::Union{AbstractVector, Nothing},
    responselabels::AbstractVector{<:AbstractString},
)
    isnothing(sampleclasses) && return nothing
    isempty(responselabels) && return nothing

    classlabels = unique(string.(sampleclasses))
    matched = [label for label in classlabels if label in responselabels]
    isempty(matched) && return nothing

    length(matched) == length(classlabels) || throw(ArgumentError(
        "All unique `sampleclasses` must be represented in `responselabels`, or none of " *
        "them. Missing labels: $(join(repr.(setdiff(classlabels, responselabels)), ", "))."))

    cols = Int[]
    for label in classlabels
        positions = findall(==(label), responselabels)
        length(positions) == 1 || throw(ArgumentError(
            "Each class label inferred from `sampleclasses` must occur exactly once in " *
            "`responselabels`. Problematic label: $(repr(label))."))
        push!(cols, only(positions))
    end

    cols
end

function validate_class_response_metadata(
    Yprim::AbstractMatrix{<:Real},
    sampleclasses::Union{AbstractVector, Nothing},
    responselabels::AbstractVector{<:AbstractString},
)
    classcols = class_response_columns(sampleclasses, responselabels)
    isnothing(classcols) && return nothing

    classlabels = unique(string.(sampleclasses))
    classblock = Yprim[:, classcols]
    predicted_indices = try
        decode_one_hot_indices(classblock)
    catch err
        if err isa ArgumentError
            throw(ArgumentError(
                "The response columns matched by `sampleclasses` in `responselabels` " *
                "must form a one-hot block. " * sprint(showerror, err)))
        end
        rethrow()
    end

    expected_labels = string.(sampleclasses)
    decoded_labels = classlabels[predicted_indices]
    decoded_labels == expected_labels || throw(ArgumentError(
        "The response columns matched by `sampleclasses` in `responselabels` must agree " *
        "row-wise with `sampleclasses`."))

    nothing
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
