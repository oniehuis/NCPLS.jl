"""
    scoreplot(mf::NCPLSFit; backend=:plotly, kwargs...)

Equivalent of `CPPLS.scoreplot(cppls)` for fitted NCPLS models. The method forwards to
`CPPLS.scoreplot(samples, groups, scores; ...)` after reading `samplelabels`,
`sampleclasses`, and the first two predictor scores from the NCPLS fit.

Requirements
- `sampleclasses(mf)` must be present.
- The fitted model must contain at least two latent variables.

All backend-specific keyword arguments accepted by `CPPLS.scoreplot` are forwarded
unchanged.
"""
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

    CPPLS.scoreplot(samplelabels(mf), groups, xscores(mf, 1:2); backend = backend, kwargs...)
end
