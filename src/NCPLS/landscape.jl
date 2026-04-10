"""
    coefficientlandscape(
        mf::NCPLSFit;
        lv::Union{Symbol, Integer}=:final,
        response::Union{Nothing, Integer}=nothing,
        response_contrast::Union{Nothing, Tuple{<:Integer, <:Integer}}=nothing,
    )

Return a 2D coefficient landscape from a fitted NCPLS model. The returned matrix is the
requested predictor-surface view of the regression coefficients:

- `lv = :final` returns the cumulative final model coefficients.
- `lv = k::Integer` returns the *isolated* contribution of latent variable `k`, i.e.
  `coef(mf, k) - coef(mf, k - 1)` for `k > 1`.

When the fit has one response, that response is used automatically. When the fit has two
responses and neither `response` nor `response_contrast` is provided, the default
contrast is `response 2 - response 1`. For more than two responses, `response` or
`response_contrast = (positive, negative)` must be supplied explicitly.
"""
function coefficientlandscape(
    mf::NCPLSFit;
    lv::Union{Symbol, Integer}=:final,
    response::Union{Nothing, Integer}=nothing,
    response_contrast::Union{Nothing, Tuple{<:Integer, <:Integer}}=nothing,
)
    isnothing(response) || isnothing(response_contrast) || throw(ArgumentError(
        "Specify either `response` or `response_contrast`, not both"))

    B = coefficient_tensor_for_lv(mf, lv)
    response_dim = ndims(B)
    nresponses = size(B, response_dim)

    landscape = if !isnothing(response)
        1 ≤ response ≤ nresponses || throw(ArgumentError(
            "`response` must be between 1 and $nresponses, got $response"))
        @views selectdim(B, response_dim, response)
    elseif !isnothing(response_contrast)
        pos, neg = response_contrast
        1 ≤ pos ≤ nresponses || throw(ArgumentError(
            "Positive response index must be between 1 and $nresponses, got $pos"))
        1 ≤ neg ≤ nresponses || throw(ArgumentError(
            "Negative response index must be between 1 and $nresponses, got $neg"))
        pos == neg && throw(ArgumentError(
            "`response_contrast` must reference two different response indices"))
        @views selectdim(B, response_dim, pos) .- selectdim(B, response_dim, neg)
    elseif nresponses == 1
        @views selectdim(B, response_dim, 1)
    elseif nresponses == 2
        @views selectdim(B, response_dim, 2) .- selectdim(B, response_dim, 1)
    else
        throw(ArgumentError(
            "Fits with $nresponses responses require `response` or `response_contrast`"))
    end

    ndims(landscape) == 2 || throw(ArgumentError(
        "coefficientlandscape currently expects exactly 2 predictor axes; got $(ndims(landscape))"))

    landscape
end

"""
    coefflandscape(mf::NCPLSFit; kwargs...)

Compatibility alias for [`coefficientlandscape`](@ref). This shorter name mirrors
`weightlandscape` and `weightlandscapeplot`.
"""
coefflandscape(mf::NCPLSFit; kwargs...) = coefficientlandscape(mf; kwargs...)

"""
    weightlandscape(
        mf::NCPLSFit;
        lv::Union{Symbol, Integer}=1,
        combine::Symbol=:sum,
    )

Return a 2D loading-weight landscape from a fitted NCPLS model.

- `lv = k::Integer` returns the component-specific loading-weight surface `W[:, :, k]`.
- `lv = :combined` or `:all` combines all component surfaces according to `combine`.

Supported combination rules are:
- `:sum` for a signed sum of the component surfaces.
- `:mean` for a signed mean of the component surfaces.
- `:sumabs` for the sum of absolute component weights.
- `:meanabs` for the mean absolute component weight.

Because latent-variable signs are arbitrary, `:sumabs` or `:meanabs` are often more
stable than signed combinations when inspecting all components together.
"""
function weightlandscape(
    mf::NCPLSFit;
    lv::Union{Symbol, Integer}=1,
    combine::Symbol=:sum,
)
    W = if lv isa Integer
        validate_ncomponents(mf, lv)
        @views selectdim(mf.W, ndims(mf.W), lv)
    elseif lv === :combined || lv === :all
        combine_weightlandscapes(mf.W, combine)
    else
        throw(ArgumentError(
            "`lv` must be a component index or one of `:combined` / `:all`"))
    end

    ndims(W) == 2 || throw(ArgumentError(
        "weightlandscape currently expects exactly 2 predictor axes; got $(ndims(W))"))

    W
end

"""
    weightprofiles(
        mf::NCPLSFit;
        lv::Union{Symbol, Integer}=1,
        combine::Symbol=:sum,
    )

Return the multilinear loading-weight vectors stored in `W_modes`.

- `lv = k::Integer` returns the per-axis vectors for latent variable `k`.
- `lv = :combined` or `:all` combines the per-axis vectors across all components using
  `combine`.

Supported combination rules are `:sum`, `:mean`, `:sumabs`, and `:meanabs`.
"""
function weightprofiles(
    mf::NCPLSFit;
    lv::Union{Symbol, Integer}=1,
    combine::Symbol=:sum,
)
    isnothing(mf.W_modes) && throw(ArgumentError(
        "This fit has no multilinear mode weights. Refit with `multilinear=true`."))

    if lv isa Integer
        validate_ncomponents(mf, lv)
        return [copy(mf.W_modes[j][:, lv]) for j in eachindex(mf.W_modes)]
    elseif lv === :combined || lv === :all
        return [combine_weightprofiles(mf.W_modes[j], combine) for j in eachindex(mf.W_modes)]
    else
        throw(ArgumentError(
            "`lv` must be a component index or one of `:combined` / `:all`"))
    end
end

function coefficient_tensor_for_lv(
    mf::NCPLSFit,
    lv::Union{Symbol, Integer},
)
    if lv === :final || lv === :cumulative
        coef(mf)
    elseif lv isa Integer
        validate_ncomponents(mf, lv)
        lv == 1 ? coef(mf, 1) : coef(mf, lv) .- coef(mf, lv - 1)
    else
        throw(ArgumentError("`lv` must be `:final`, `:cumulative`, or a component index"))
    end
end

function combine_weightlandscapes(
    W::AbstractArray{<:Real},
    combine::Symbol,
)
    component_dim = ndims(W)
    combined = if combine === :sum
        sum(W; dims = component_dim)
    elseif combine === :mean
        mean(W; dims = component_dim)
    elseif combine === :sumabs
        sum(abs.(W); dims = component_dim)
    elseif combine === :meanabs
        mean(abs.(W); dims = component_dim)
    else
        throw(ArgumentError(
            "`combine` must be one of :sum, :mean, :sumabs, or :meanabs"))
    end

    dropdims(combined; dims = component_dim)
end

function combine_weightprofiles(
    profiles::AbstractMatrix{<:Real},
    combine::Symbol,
)
    combined = if combine === :sum
        sum(profiles; dims = 2)
    elseif combine === :mean
        mean(profiles; dims = 2)
    elseif combine === :sumabs
        sum(abs.(profiles); dims = 2)
    elseif combine === :meanabs
        mean(abs.(profiles); dims = 2)
    else
        throw(ArgumentError(
            "`combine` must be one of :sum, :mean, :sumabs, or :meanabs"))
    end

    vec(combined)
end

function landscape_predictoraxes(mf::NCPLSFit, landscape::AbstractMatrix)
    axes = predictoraxes(mf)

    if isempty(axes)
        return (
            PredictorAxis("Axis 1", collect(1:size(landscape, 1))),
            PredictorAxis("Axis 2", collect(1:size(landscape, 2))),
        )
    end

    length(axes) == 2 || throw(ArgumentError(
        "landscape plotting currently expects 2 predictor axes, got $(length(axes))"))

    ax1, ax2 = axes
    length(ax1.values) == size(landscape, 1) || throw(DimensionMismatch(
        "First predictor axis has length $(length(ax1.values)), expected $(size(landscape, 1))"))
    length(ax2.values) == size(landscape, 2) || throw(DimensionMismatch(
        "Second predictor axis has length $(length(ax2.values)), expected $(size(landscape, 2))"))

    ax1, ax2
end

landscape_axis_label(axis::PredictorAxis) = isnothing(axis.unit) ? axis.name :
    string(axis.name, " (", axis.unit, ")")

function default_landscape_title(
    mf::NCPLSFit,
    lv::Union{Symbol, Integer},
    response::Union{Nothing, Integer},
    response_contrast::Union{Nothing, Tuple{<:Integer, <:Integer}},
)
    lv_label = lv isa Integer ? "LV$(lv)" : ""
    response_label = landscape_response_label(mf, response, response_contrast)
    parts = String["Coefficient Landscape"]
    !isempty(lv_label) && push!(parts, lv_label)
    !isempty(response_label) && push!(parts, "($response_label)")
    join(parts, " ")
end

function landscape_response_label(
    mf::NCPLSFit,
    response::Union{Nothing, Integer},
    response_contrast::Union{Nothing, Tuple{<:Integer, <:Integer}},
)
    labels = responselabels(mf)
    nresponses = size(mf.Q, 1)

    getlabel(i::Integer) = 1 ≤ i ≤ length(labels) ? labels[i] : string("response_", i)

    if !isnothing(response_contrast)
        pos, neg = response_contrast
        return string(getlabel(pos), " – ", getlabel(neg))
    elseif !isnothing(response)
        return getlabel(response)
    elseif nresponses == 1
        return isempty(labels) ? "" : getlabel(1)
    elseif nresponses == 2
        return string(getlabel(2), " – ", getlabel(1))
    else
        return ""
    end
end

function default_weight_title(
    mf::NCPLSFit,
    lv::Union{Symbol, Integer},
    combine::Symbol,
)
    if lv isa Integer
        return "NCPLS LV$(lv) Weight Landscape"
    else
        return "NCPLS Combined Weight Landscape ($(combine))"
    end
end

function default_weightprofiles_title(
    lv::Union{Symbol, Integer},
    combine::Symbol,
)
    lv isa Integer ? "NCPLS LV$(lv) Weight Profiles" :
        "NCPLS Combined Weight Profiles ($(combine))"
end

const LANDSCAPEPLOT_DOC = """
    coefflandscapeplot(mf; backend=:plotly, kwargs...)
    landscapeplot(mf; backend=:plotly, kwargs...)

Backend dispatcher for NCPLS coefficient-landscape plots. The fit must describe exactly
two predictor axes, either through stored `predictoraxes` metadata or implicitly through
the matrix returned by [`coefficientlandscape`](@ref).

General keywords
- `backend::Symbol = :plotly`
  Select the plotting backend. Supported values: `:plotly`.
- `lv::Union{Symbol,Integer} = :final`
  Use `:final` for the cumulative fitted model or an integer to plot the isolated
  contribution of one latent variable.
- `response::Union{Nothing,Integer} = nothing`
  Plot one response surface directly.
- `response_contrast::Union{Nothing,Tuple{Int,Int}} = nothing`
  Plot a signed contrast `positive - negative` between two responses.

PlotlyJS backend keywords
- `sample_size::Integer = 50_000`
  Maximum number of coefficient values used when estimating the robust color range.
- `iqr_multiplier::Real = 8.0`
  Multiplier for the IQR fence used in the robust color range.
- `clip_quantile::Real = 0.995`
  High quantile of `abs.(coef)` used as a lower bound on the clipping limit.
- `colorscale::AbstractString = "RdBu"`
- `colorbar_title::AbstractString = "Coefficient"`
- `hovertemplate::Union{Nothing,AbstractString} = nothing`
- `title::Union{Nothing,AbstractString} = nothing`
- `layout = nothing`
- `plot_kwargs = (;)`
"""
function coefflandscapeplot end
Base.@doc LANDSCAPEPLOT_DOC coefflandscapeplot

function landscapeplot end
Base.@doc LANDSCAPEPLOT_DOC landscapeplot

function landscapeplot_plotly end

const WEIGHTLANDSCAPEPLOT_DOC = """
    weightlandscapeplot(mf; backend=:plotly, kwargs...)

Backend dispatcher for NCPLS loading-weight landscape plots. The fit must describe
exactly two predictor axes, either through stored `predictoraxes` metadata or implicitly
through the returned weight surface.

General keywords
- `backend::Symbol = :plotly`
  Select the plotting backend. Supported values: `:plotly`.
- `lv::Union{Symbol,Integer} = 1`
  Use an integer for one component or `:combined` / `:all` for an aggregate across all
  stored components.
- `combine::Symbol = :sum`
  Combination rule for `lv = :combined`. Supported values: `:sum`, `:mean`, `:sumabs`,
  and `:meanabs`.

PlotlyJS backend keywords
- `sample_size::Integer = 50_000`
- `iqr_multiplier::Real = 8.0`
- `clip_quantile::Real = 0.995`
- `colorscale::Union{Nothing,AbstractString} = nothing`
- `colorbar_title::AbstractString = "Weight"`
- `hovertemplate::Union{Nothing,AbstractString} = nothing`
- `title::Union{Nothing,AbstractString} = nothing`
- `layout = nothing`
- `plot_kwargs = (;)`
"""
function weightlandscapeplot end
Base.@doc WEIGHTLANDSCAPEPLOT_DOC weightlandscapeplot

function weightlandscapeplot_plotly end

const WEIGHTPROFILESPLOT_DOC = """
    weightprofilesplot(mf; backend=:plotly, kwargs...)

Backend dispatcher for NCPLS multilinear weight-profile plots. These plots require a
multilinear fit with stored `W_modes`.

General keywords
- `backend::Symbol = :plotly`
  Select the plotting backend. Supported values: `:plotly`.
- `lv::Union{Symbol,Integer} = 1`
  Use an integer for one component or `:combined` / `:all` for an aggregate across all
  stored components.
- `combine::Symbol = :sum`
  Combination rule for `lv = :combined`. Supported values: `:sum`, `:mean`, `:sumabs`,
  and `:meanabs`.

PlotlyJS backend keywords
- `title::Union{Nothing,AbstractString} = nothing`
- `layout = nothing`
- `line_kwargs = (;)`
- `zero_line::Bool = true`
- `plot_kwargs = (;)`
"""
function weightprofilesplot end
Base.@doc WEIGHTPROFILESPLOT_DOC weightprofilesplot

function weightprofilesplot_plotly end

function _require_landscape_extension(extsym::Symbol, pkg::AbstractString)
    Base.get_extension(@__MODULE__, extsym) === nothing &&
        error("Backend $(pkg) not loaded. Run `using $(pkg)` first.")
    return nothing
end

const _require_landscape_extension_ref = Ref{Function}(_require_landscape_extension)
const _landscapeplot_plotly_ref = Ref{Function}(
    (mf; kwargs...) -> landscapeplot_plotly(mf; kwargs...)
)
const _weightlandscapeplot_plotly_ref = Ref{Function}(
    (mf; kwargs...) -> weightlandscapeplot_plotly(mf; kwargs...)
)
const _weightprofilesplot_plotly_ref = Ref{Function}(
    (mf; kwargs...) -> weightprofilesplot_plotly(mf; kwargs...)
)

function landscapeplot(
    mf::NCPLSFit;
    backend::Symbol=:plotly,
    kwargs...,
)
    if backend === :plotly
        _require_landscape_extension_ref[](:PlotlyJSExtension, "PlotlyJS")
        return _landscapeplot_plotly_ref[](mf; kwargs...)
    else
        error("Unknown backend")
    end
end

function coefflandscapeplot(
    mf::NCPLSFit;
    backend::Symbol=:plotly,
    kwargs...,
)
    landscapeplot(mf; backend = backend, kwargs...)
end

function weightprofilesplot(
    mf::NCPLSFit;
    backend::Symbol=:plotly,
    kwargs...,
)
    if backend === :plotly
        _require_landscape_extension_ref[](:PlotlyJSExtension, "PlotlyJS")
        return _weightprofilesplot_plotly_ref[](mf; kwargs...)
    else
        error("Unknown backend")
    end
end

function weightlandscapeplot(
    mf::NCPLSFit;
    backend::Symbol=:plotly,
    kwargs...,
)
    if backend === :plotly
        _require_landscape_extension_ref[](:PlotlyJSExtension, "PlotlyJS")
        return _weightlandscapeplot_plotly_ref[](mf; kwargs...)
    else
        error("Unknown backend")
    end
end
