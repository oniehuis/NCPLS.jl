"""
    NCPLSModel

Model specification passed to `fit`. A `NCPLSModel` stores the user-controlled settings
for n-CPLS fitting, most importantly `ncomponents` and centering and scaling of predictor 
and response variables.
"""
struct NCPLSModel
    ncomponents::Int
    center_X::Bool
    scale_X::Bool
    center_Yprim::Bool
    scale_Yprim::Bool
    center_Yadd::Bool
    scale_Yadd::Bool
    multilinear::Bool
end

"""
    NCPLSModel(; 
        ncomponents::Integer=2,
        center_X::Bool=true,
        scale_X::Bool=false,
        center_Yprim::Bool=true,
        scale_Yprim::Bool=false,
        center_Yadd::Bool=true,
        scale_Yadd::Bool=false,
        multilinear::Bool=false
    )

Construct a model specification for `fit`. The most commonly adjusted setting is
`ncomponents`.
"""
function NCPLSModel(;
    ncomponents::T1=2,
    center_X::Bool=true,
    scale_X::Bool=false,
    center_Yprim::Bool=true,
    scale_Yprim::Bool=false,
    center_Yadd::Bool=true,
    scale_Yadd::Bool=false,
    multilinear::Bool=false
) where {
        T1<:Integer
    }

    ncomponents > 0 || throw(ArgumentError("ncomponents must be greater than zero"))

    NCPLSModel(
        Int(ncomponents),
        center_X,
        scale_X,
        center_Yprim,
        scale_Yprim,
        center_Yadd,
        scale_Yadd,
        multilinear
    )
end

function Base.show(io::IO, m::NCPLSModel)
    print(io, "NCPLSModel(",
        "ncomponents=", m.ncomponents,
        ", center_X=", m.center_X,
        ", scale_X=", m.scale_X,
        ", center_Yprim=", m.center_Yprim,
        ", scale_Yprim=", m.scale_Yprim,
        ", center_Yadd=", m.center_Yadd,
        ", scale_Yadd=", m.scale_Yadd,
        ", multilinear=", m.multilinear,
        ")")
end

function Base.show(io::IO, ::MIME"text/plain", m::NCPLSModel)
    println(io, "NCPLSModel")
    println(io, "  ncomponents: ", m.ncomponents)
    println(io, "  center_X: ", m.center_X)
    println(io, "  scale_X: ", m.scale_X)
    println(io, "  center_Yprim: ", m.center_Yprim)
    println(io, "  scale_Yprim: ", m.scale_Yprim)
    println(io, "  center_Yadd: ", m.center_Yadd)
    println(io, "  scale_Yadd: ", m.scale_Yadd)
    println(io, "  multilinear: ", m.multilinear)
end

"""
    AbstractNCPLSFit

Common supertype for fitted NCPLS models that share the fields ... . 
"""
abstract type AbstractNCPLSFit end

"""
    NCPLSFit

Full fitted NCPLS model returned by `fit`. This stores the fitted projection and
regression objects together with component-wise scores/loadings and preprocessing
statistics needed for prediction and diagnostics.
"""
struct NCPLSFit{
    TModel<:NCPLSModel,
    TB,
    TR,
    TT,
    TP,
    TQ,
    TW,
    Tc,
    TW0,
    Trho,
    TYres,
    TXStat,
    TYStat,
    TYAddStat,
} <: AbstractNCPLSFit

    model::TModel
    B::TB
    R::TR
    T::TT
    P::TP
    Q::TQ
    W::TW
    c::Tc
    W0::TW0
    rho::Trho
    Yres::TYres
    X_mean::TXStat
    X_std::TXStat
    Yprim_mean::TYStat
    Yprim_std::TYStat
    Yadd_mean::TYAddStat
    Yadd_std::TYAddStat
end

function Base.show(io::IO, mf::NCPLSFit)
    print(io, "NCPLSFit")
end

function Base.show(io::IO, ::MIME"text/plain", mf::NCPLSFit)
    println(io, "NCPLSFit")
end
