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
    T = zeros(Float64, size(d.X, 1), m.ncomponents)
    P = Array{Float64}(undef, size(d.X)[2:end]..., m.ncomponents)
    Q = Matrix{Float64}(undef, size(d.Yprim, 2), m.ncomponents)
    W_A = Array{Float64}(undef, size(d.X)[2:end]..., m.ncomponents)
    q_comb = size(d.Yprim, 2) + (isnothing(d.Yadd) ? 0 : size(d.Yadd, 2))
    W0 = Array{Float64}(undef, size(d.X)[2:end]..., q_comb, m.ncomponents)
    c = Matrix{Float64}(undef, q_comb, m.ncomponents)
    rho = Vector{Float64}(undef, m.ncomponents)

    # Apply observation weights consistently with covariance weighting (sqrt for covariance).
    cca_obs_weights = isnothing(obs_weights) ? nothing : sqrt.(obs_weights)

    # Main loop over components: compute weights, scores, loadings, deflate, and 
    # store results.
    Y = copy(d.Yprim)
    for i = 1:m.ncomponents
        # W₀ = Xᵗ_d ⓐ₁ [Y Yadditional]
        Ycomb = isnothing(Yadd) ? Y : hcat(Y, d.Yadd)
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

        if m.multilinear  # multilinear branch
            throw(ArgumentError("Multilinear loading weights option is not yet implemented."))
        else # unfolded branch
            # Wᵒ := W
            W⁰ = W
        end

        selectdim(W_A, ndims(W_A), i) .= W⁰

        # t = X ⓓ Wᵒ
        t = score_vector(d.X, W⁰)
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
        c,
        W0,
        rho,
        Y,
        d.X_mean,
        d.X_std,
        d.Yprim_mean,
        d.Yprim_std,
        d.Yadd_mean,
        d.Yadd_std,
    )
end
