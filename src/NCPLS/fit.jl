function fit_ncpls_core(
    m::NCPLSModel,
    X::AbstractArray{<:Real},
    Yprim::AbstractMatrix{<:Real};
    obs_weights::T1=nothing,
    Yaux::T2=nothing
) where {
    T1<:Union{AbstractVector{<:Real}, Nothing},
    T2<:Union{LinearAlgebra.AbstractVecOrMat{<:Real}, Nothing}
}

end