"""
    NCPLSModel

Model specification passed to `fit`. An `NCPLSModel` stores the centering and scaling
options for `X`, the centering option for `Yprim`, the number of extracted components,
and the settings that control the multilinear loading-weight branch.
"""
struct NCPLSModel
    ncomponents::Int
    center_X::Bool
    scale_X::Bool
    center_Yprim::Bool
    multilinear::Bool
    orthogonalize_mode_weights::Bool
    multilinear_maxiter::Int
    multilinear_tol::Float64
    multilinear_init::Symbol
    multilinear_seed::Int
end

"""
    NCPLSModel(; 
        ncomponents::Integer=2,
        center_X::Bool=true,
        scale_X::Bool=false,
        center_Yprim::Bool=true,
        multilinear::Bool=true,
        orthogonalize_mode_weights::Bool=false,
        multilinear_maxiter::Int=500,
        multilinear_tol::Float64=1e-10,
        multilinear_init::Symbol=:hosvd,
        multilinear_seed::Int=1
    )

Construct an `NCPLSModel` with the given fitting options. By default, the multilinear
loading-weight branch is enabled. The multilinear control fields govern
initialization, iteration limits, and convergence in the PARAFAC step.

`multilinear_init` selects the starting values for rank-1 PARAFAC fits. The supported
options are `:hosvd`, which initializes each mode with the leading left singular vector
of the corresponding unfolding, and `:random`, which initializes each mode with a
random unit vector drawn using `multilinear_seed`.
"""
function NCPLSModel(;
    ncomponents::T1=2,
    center_X::Bool=true,
    scale_X::Bool=false,
    center_Yprim::Bool=true,
    multilinear::Bool=true,
    orthogonalize_mode_weights::Bool=false,
    multilinear_maxiter::Int=500,
    multilinear_tol::Float64=1e-10,
    multilinear_init::Symbol=:hosvd,
    multilinear_seed::Int=1
) where {
        T1<:Integer
    }

    ncomponents > 0 || throw(ArgumentError("ncomponents must be greater than zero"))
    multilinear_maxiter > 0 || throw(ArgumentError("multilinear_maxiter must be greater than zero"))
    multilinear_tol >= 0 || throw(ArgumentError("multilinear_tol must be non-negative"))
    multilinear_init in (:hosvd, :random) || throw(ArgumentError(
        "multilinear_init must be :hosvd or :random"))

    NCPLSModel(
        Int(ncomponents),
        center_X,
        scale_X,
        center_Yprim,
        multilinear,
        orthogonalize_mode_weights,
        multilinear_maxiter,
        multilinear_tol,
        multilinear_init,
        multilinear_seed,
    )
end

function Base.show(io::IO, m::NCPLSModel)
    print(io, "NCPLSModel(",
        "ncomponents=", m.ncomponents,
        ", center_X=", m.center_X,
        ", scale_X=", m.scale_X,
        ", center_Yprim=", m.center_Yprim,
        ", multilinear=", m.multilinear,
        ", orthogonalize_mode_weights=", m.orthogonalize_mode_weights,
        ", multilinear_maxiter=", m.multilinear_maxiter,
        ", multilinear_tol=", m.multilinear_tol,
        ", multilinear_init=", repr(m.multilinear_init),
        ", multilinear_seed=", m.multilinear_seed,
        ")")
end

function Base.show(io::IO, ::MIME"text/plain", m::NCPLSModel)
    println(io, "NCPLSModel")
    println(io, "  ncomponents: ", m.ncomponents)
    println(io, "  center_X: ", m.center_X)
    println(io, "  scale_X: ", m.scale_X)
    println(io, "  center_Yprim: ", m.center_Yprim)
    println(io, "  multilinear: ", m.multilinear)
    println(io, "  orthogonalize_mode_weights: ", m.orthogonalize_mode_weights)
    println(io, "  multilinear_maxiter: ", m.multilinear_maxiter)
    println(io, "  multilinear_tol: ", m.multilinear_tol)
    println(io, "  multilinear_init: ", m.multilinear_init)
    println(io, "  multilinear_seed: ", m.multilinear_seed)
end

"""
    PredictorAxis(name, values; unit=nothing)
    PredictorAxis(; name, values, unit=nothing)

Store metadata for one non-sample predictor axis of a multiway NCPLS fit. `name`
identifies the axis (for example `"RT"` or `"m/z"`), `values` stores the axis positions,
and `unit` optionally stores the physical unit separately from the numeric values.
"""
struct PredictorAxis{TValues<:AbstractVector}
    name::String
    values::TValues
    unit::Union{String, Nothing}
end

function PredictorAxis(
    name::AbstractString,
    values::AbstractVector;
    unit::Union{AbstractString, Nothing}=nothing,
)
    isempty(name) && throw(ArgumentError("PredictorAxis name must be non-empty"))
    PredictorAxis(
        String(name),
        collect(values),
        isnothing(unit) || isempty(unit) ? nothing : String(unit),
    )
end

PredictorAxis(;
    name::AbstractString,
    values::AbstractVector,
    unit::Union{AbstractString, Nothing}=nothing,
) = PredictorAxis(name, values; unit=unit)

function Base.show(io::IO, axis::PredictorAxis)
    print(io, "PredictorAxis(",
        "name=", repr(axis.name),
        ", length=", length(axis.values),
        ", unit=", isnothing(axis.unit) ? "nothing" : repr(axis.unit),
        ")")
end

"""
    AbstractNCPLSFit

Abstract supertype for fitted NCPLS models.
"""
abstract type AbstractNCPLSFit end

"""
    NCPLSFitLight{TB, TXStat, TYStat}

Reduced fitted NCPLS model that retains only the information needed for prediction. This
type is used mainly for efficient internal prediction during cross-validation.
"""
struct NCPLSFitLight{TB, TXStat, TYStat} <: AbstractNCPLSFit
    B::TB
    X_mean::TXStat
    X_std::TXStat
    Yprim_mean::TYStat
end

function Base.show(io::IO, mf::NCPLSFitLight)
    print(io, "NCPLSFitLight(",
        "predictor_dims=", repr(size(mf.B)[1:end-2]),
        ", responses=", size(mf.B, ndims(mf.B)),
        ", components=", size(mf.B, ndims(mf.B) - 1),
        ")")
end

function Base.show(io::IO, ::MIME"text/plain", mf::NCPLSFitLight)
    println(io, "NCPLSFitLight")
    println(io, "  predictor_dims: ", repr(size(mf.B)[1:end-2]))
    println(io, "  responses: ", size(mf.B, ndims(mf.B)))
    print(io, "  components: ", size(mf.B, ndims(mf.B) - 1))
end

"""
    coef(mf::AbstractNCPLSFit)
    coef(mf::AbstractNCPLSFit, ncomps::Integer)

Return the regression coefficients for the final or requested number of components.
The returned tensor acts on preprocessed predictors, i.e. on `X` after the centering and
optional scaling stored in the fitted model have been applied.
"""
coef(mf::AbstractNCPLSFit) = coef(mf, ncomponents(mf))
coef(mf::AbstractNCPLSFit, ncomps::Integer) = @views selectdim(
    mf.B, ndims(mf.B) - 1, validate_ncomponents(mf, ncomps))

"""
    xmean(mf::AbstractNCPLSFit)

Return the predictor mean array for the fitted model.
"""
xmean(mf::AbstractNCPLSFit) = mf.X_mean

"""
    xstd(mf::AbstractNCPLSFit)

Return the predictor standard deviation array for the fitted model.
"""
xstd(mf::AbstractNCPLSFit) = mf.X_std

"""
    ymean(mf::AbstractNCPLSFit)

Return the primary-response mean vector for the fitted model.
"""
ymean(mf::AbstractNCPLSFit) = mf.Yprim_mean

"""
    NCPLSFit

Fitted NCPLS model returned by `fit`. An `NCPLSFit` stores the regression and projection
objects, component-wise scores and loadings, multilinear diagnostics, and the
preprocessing statistics needed for prediction and inspection. For multilinear fits, the
stored diagnostics include the PARAFAC approximation error, method, scaling, iteration
count, and convergence flag for each extracted component.
"""
struct NCPLSFit{
    TModel<:NCPLSModel,
    TB,
    TR,
    TT,
    TP,
    TQ,
    TW,
    TWModes,
    Tc,
    TW0,
    Trho,
    TYres,
    TWMLRelerr,
    TWMLMethod,
    TWMLLambda,
    TWMLNiter,
    TWMLConverged,
    TXStat,
    TYStat,
    TSampleLabels,
    TResponseLabels,
    TSampleClasses,
    TPredictorAxes,
} <: AbstractNCPLSFit

    model::TModel
    B::TB
    R::TR
    T::TT
    P::TP
    Q::TQ
    W::TW
    W_modes::TWModes
    c::Tc
    W0::TW0
    rho::Trho
    Yres::TYres
    W_multilinear_relerr::TWMLRelerr
    W_multilinear_method::TWMLMethod
    W_multilinear_lambda::TWMLLambda
    W_multilinear_niter::TWMLNiter
    W_multilinear_converged::TWMLConverged
    X_mean::TXStat
    X_std::TXStat
    Yprim_mean::TYStat
    samplelabels::TSampleLabels
    responselabels::TResponseLabels
    sampleclasses::TSampleClasses
    predictoraxes::TPredictorAxes
end

function NCPLSFit(
    model,
    B,
    R,
    T,
    P,
    Q,
    W,
    W_modes,
    c,
    W0,
    rho,
    Yres,
    W_multilinear_relerr,
    W_multilinear_method,
    W_multilinear_lambda,
    W_multilinear_niter,
    W_multilinear_converged,
    X_mean,
    X_std,
    Yprim_mean;
    samplelabels::AbstractVector=String[],
    responselabels::AbstractVector=String[],
    sampleclasses::Union{AbstractVector, Nothing}=nothing,
    predictoraxes=(),
)
    nsamples = size(T, 1)
    nresponses = size(Q, 1)
    predictor_dims = size(B)[1:end-2]

    samplelabels = default_sample_labels(
        validate_label_length(samplelabels, nsamples, "samplelabels"),
        nsamples,
    )
    responselabels = normalize_string_labels(
        responselabels,
        nresponses,
        "responselabels",
    )
    sampleclasses = normalize_sampleclasses(sampleclasses, nsamples)
    predictoraxes = normalize_predictoraxes_metadata(predictoraxes, predictor_dims)

    NCPLSFit(
        model,
        B,
        R,
        T,
        P,
        Q,
        W,
        W_modes,
        c,
        W0,
        rho,
        Yres,
        W_multilinear_relerr,
        W_multilinear_method,
        W_multilinear_lambda,
        W_multilinear_niter,
        W_multilinear_converged,
        X_mean,
        X_std,
        Yprim_mean,
        samplelabels,
        responselabels,
        sampleclasses,
        predictoraxes,
    )
end

function Base.show(io::IO, mf::NCPLSFit)
    print(io, "NCPLSFit(",
        "samples=", size(mf.T, 1),
        ", predictor_dims=", repr(size(mf.B)[1:end-2]),
        ", responses=", size(mf.B, ndims(mf.B)),
        ", components=", size(mf.B, ndims(mf.B) - 1),
        ", multilinear=", mf.model.multilinear,
        ")")
end

function Base.show(io::IO, ::MIME"text/plain", mf::NCPLSFit)
    println(io, "NCPLSFit")
    println(io, "  samples: ", size(mf.T, 1))
    println(io, "  predictor_dims: ", repr(size(mf.B)[1:end-2]))
    println(io, "  responses: ", size(mf.B, ndims(mf.B)))
    println(io, "  components: ", size(mf.B, ndims(mf.B) - 1))
    print(io, "  multilinear: ", mf.model.multilinear)
end

"""
    predictoraxes(mf::NCPLSFit)

Return the stored metadata for the non-sample predictor axes of the fitted model.
"""
predictoraxes(mf::NCPLSFit) = mf.predictoraxes

"""
    responselabels(mf::NCPLSFit)

Return the stored response labels for the fitted model.
"""
responselabels(mf::NCPLSFit) = mf.responselabels

"""
    sampleclasses(mf::NCPLSFit)

Return the stored per-sample class labels, or `nothing` when no classes were supplied at
fit time.
"""
sampleclasses(mf::NCPLSFit) = mf.sampleclasses

"""
    samplelabels(mf::NCPLSFit)

Return the stored sample labels for the fitted model.
"""
samplelabels(mf::NCPLSFit) = mf.samplelabels

"""
    xscores(mf::NCPLSFit)
    xscores(mf::NCPLSFit, comp::Integer)
    xscores(mf::NCPLSFit, comps::AbstractUnitRange{<:Integer})
    xscores(mf::NCPLSFit, comps::AbstractVector{<:Integer})

Return the predictor score matrix for the fitted model, or a subset of its columns.
"""
xscores(mf::NCPLSFit) = mf.T

function xscores(mf::NCPLSFit, comp::Integer)
    ncomp = size(mf.T, 2)
    1 ≤ comp ≤ ncomp || throw(
        ArgumentError("Component index $comp out of bounds (1:$ncomp)"))
    view(mf.T, :, comp)
end

function xscores(mf::NCPLSFit, comps::AbstractUnitRange{<:Integer})
    ncomp = size(mf.T, 2)
    (1 ≤ first(comps) ≤ ncomp && 1 ≤ last(comps) ≤ ncomp) || throw(
        ArgumentError("Component range $(comps) out of bounds (1:$ncomp)"))
    view(mf.T, :, comps)
end

function xscores(mf::NCPLSFit, comps::AbstractVector{<:Integer})
    ncomp = size(mf.T, 2)
    all(1 .≤ comps .≤ ncomp) || throw(
        ArgumentError("Component indices $(comps) out of bounds (1:$ncomp)"))
    view(mf.T, :, comps)
end

"""
    fitted(mf::NCPLSFit)
    fitted(mf::NCPLSFit, ncomps::Integer)

Return the fitted response matrix for the final or requested number of components.
"""
fitted(mf::NCPLSFit) = fitted(mf, ncomponents(mf))
function fitted(mf::NCPLSFit, ncomps::Integer)
    ncomps = validate_ncomponents(mf, ncomps)
    Yhat = @views mf.T[:, 1:ncomps] * mf.Q[:, 1:ncomps]'
    restore_response_scale(Yhat, mf; add_mean=true)
end

"""
    residuals(mf::NCPLSFit)
    residuals(mf::NCPLSFit, ncomps::Integer)

Return the response residual matrix for the final or requested number of components.
"""
residuals(mf::NCPLSFit) = residuals(mf, ncomponents(mf))
function residuals(mf::NCPLSFit, ncomps::Integer)
    ncomps = validate_ncomponents(mf, ncomps)

    Yres = if ncomps == ncomponents(mf)
        mf.Yres
    else
        @views mf.Yres + mf.T[:, ncomps+1:end] * mf.Q[:, ncomps+1:end]'
    end

    restore_response_scale(Yres, mf; add_mean=false)
end

"""
    ncomponents(mf::AbstractNCPLSFit)

Return the number of latent components stored in the fitted NCPLS model.
"""
ncomponents(mf::AbstractNCPLSFit) = size(mf.B, ndims(mf.B) - 1)

function validate_ncomponents(mf::AbstractNCPLSFit, ncomps::Integer)
    1 ≤ ncomps ≤ ncomponents(mf) || throw(DimensionMismatch(
        "ncomps exceeds the number of components in the model"))
    ncomps
end

"""
    restore_response_scale(Y::AbstractArray{<:Real}, mf::AbstractNCPLSFit; add_mean::Bool)

Restore response values from the fitted model's centered response space, optionally
adding the stored response mean.
"""
function restore_response_scale(
    Y::AbstractArray{<:Real},
    mf::AbstractNCPLSFit;
    add_mean::Bool
)
    Y_restored = float64(Y)
    add_mean || return Y_restored

    M = length(mf.Yprim_mean)
    lead_dims = ntuple(_ -> 1, ndims(Y) - 1)
    Ymean = reshape(mf.Yprim_mean, lead_dims..., M)
    Y_restored .+ Ymean
end
