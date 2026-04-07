function fit(
    m::NCPLSModel,
    X::AbstractArray{<:Real},
    Yprim::AbstractMatrix{<:Real};
    kwargs...
)
    fit_ncpls_core(m, X, Yprim; kwargs...)
end

function fit_ncpls_core(
    m::NCPLSModel,
    X::AbstractArray{<:Real},
    Yprim::AbstractMatrix{<:Real};
    Yadd::T1=nothing,
    obs_weights::T2=nothing
) where {
    T1<:Union{AbstractMatrix{<:Real}, Nothing},
    T2<:Union{AbstractVector{<:Real}, Nothing}
}

    # Preprocess data: center/scale, optionally with weights.
    d = preprocess(m, X, Yprim, Yadd, obs_weights)

    # Preallocate arrays for scores, loadings, regression coefficients, and diagnostics.

    # Main loop over components: compute weights, scores, loadings, deflate, and 
    # store results.
    for i = 1:m.ncomponents
        # compute_ncpls_weights

    end

    NCPLSFit(d.X_mean, d.X_std, d.Yprim_mean, d.Yprim_std, d.Yadd_mean, d.Yadd_std)
end
