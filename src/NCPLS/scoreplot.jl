"""
    scoreplot(samples, groups, scores; backend=:plotly, kwargs...)
    scoreplot(mf::NCPLSFit; backend=:plotly, kwargs...)

Backend dispatcher for NCPLS score plots.

The raw method accepts sample labels, grouping labels, and a score matrix with at least
two columns. The fit method reads `samplelabels`, `sampleclasses`, and the first two
predictor scores from an `NCPLSFit`.

Requirements
- `scores` must have at least two columns.
- `sampleclasses(mf)` must be present.
- The fitted model must contain at least two latent variables.

General keywords
- `backend::Symbol = :plotly`
  Select the plotting backend. Supported values: `:plotly` and `:makie`.
"""
function scoreplot end

function scoreplot_makie end
function scoreplot_makie(args...; kwargs...)
    _require_scoreplot_extension(:MakieExtension, "Makie or CairoMakie")
    error("Unreachable")
end

function scoreplot_plotly end
function scoreplot_plotly(args...; kwargs...)
    _require_scoreplot_extension(:PlotlyJSExtension, "PlotlyJS")
    error("Unreachable")
end

function _require_scoreplot_extension(extsym::Symbol, pkg::AbstractString)
    Base.get_extension(@__MODULE__, extsym) === nothing &&
        error("Backend $(pkg) not loaded. Run `using $(pkg)` first.")
    return nothing
end

const _require_scoreplot_extension_ref = Ref{Function}(_require_scoreplot_extension)
const _scoreplot_makie_ref = Ref{Function}(
    (samples, groups, scores; kwargs...) -> scoreplot_makie(samples, groups, scores; kwargs...)
)
const _scoreplot_plotly_ref = Ref{Function}(
    (samples, groups, scores; kwargs...) -> scoreplot_plotly(samples, groups, scores; kwargs...)
)

function scoreplot(
    samples::AbstractVector{<:AbstractString},
    groups,
    scores::AbstractMatrix{<:Real};
    backend::Symbol = :plotly,
    kwargs...,
)
    if backend === :plotly
        _require_scoreplot_extension_ref[](:PlotlyJSExtension, "PlotlyJS")
        return _scoreplot_plotly_ref[](samples, groups, scores; kwargs...)
    elseif backend === :makie
        _require_scoreplot_extension_ref[](:MakieExtension, "Makie or CairoMakie")
        return _scoreplot_makie_ref[](samples, groups, scores; kwargs...)
    else
        error("Unknown backend")
    end
end

function scoreplot(
    mf::NCPLSFit;
    backend::Symbol=:plotly,
    kwargs...,
)
    groups = sampleclasses(mf)
    isnothing(groups) && throw(ArgumentError(
        "scoreplot(mf) requires sampleclasses in the fitted model. " *
        "Pass `sampleclasses=` to `fit` or call scoreplot(samples, groups, scores) directly."
    ))

    size(mf.T, 2) >= 2 || throw(ArgumentError(
        "scoreplot(mf) requires at least 2 components; got $(size(mf.T, 2)). " *
        "Refit with ncomponents >= 2 or call scoreplot(samples, groups, scores) with a custom 2-column score matrix."
    ))

    scoreplot(samplelabels(mf), groups, xscores(mf, 1:2); backend = backend, kwargs...)
end
