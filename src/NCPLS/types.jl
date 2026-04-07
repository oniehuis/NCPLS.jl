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
end

"""
    NCPLSModel(; 
        ncomponents::Integer=2,
        center_X::Bool=true,
        scale_X::Bool=false,
        center_Yprim::Bool=true,
        scale_Yprim::Bool=false
    )

Construct a model specification for `fit`. The most commonly adjusted setting is
`ncomponents`.
"""
function NCPLSModel(;
    ncomponents::T1=2,
    center_X::Bool=true,
    scale_X::Bool=false,
    center_Yprim::Bool=true,
    scale_Yprim::Bool=false
) where {
        T1<:Integer
    }

    ncomponents > 0 || throw(ArgumentError("ncomponents must be greater than zero"))

    NCPLSModel(
        Int(ncomponents),
        center_X,
        scale_X,
        center_Yprim,
        scale_Yprim
    )
end

function Base.show(io::IO, spec::NCPLSModel)
    print(io, "NCPLSModel(",
        "ncomponents=", spec.ncomponents,
        ", center_X=", spec.center_X,
        ", scale_X=", spec.scale_X,
        ", center_Yprim=", spec.center_Yprim,
        ", scale_Yprim=", spec.scale_Yprim,
        ")")
end

function Base.show(io::IO, ::MIME"text/plain", spec::NCPLSModel)
    println(io, "NCPLSModel")
    println(io, "  ncomponents: ", spec.ncomponents)
    println(io, "  center_X: ", spec.center_X)
    println(io, "  scale_X: ", spec.scale_X)
    println(io, "  center_Yprim: ", spec.center_Yprim)
    println(io, "  scale_Yprim: ", spec.scale_Yprim)
end

"""
    AbstractNCPLSFit

Common supertype for fitted NCPLS models that share the fields ... . 
"""
abstract type AbstractNCPLSFit end

"""
    NCPLSFit{T1, T2}

Full fitted NCPLS model returned by `fit`. This type stores ...
together with the intermediate quantities needed for diagnostics, projections, and
plotting.

Most users will work with a `NCPLSFit` through ... .
"""
struct NCPLSFit{
    T1<:Real,
    T2<:AbstractArray{T1},
    T3<:AbstractVector{T1}
} <: AbstractNCPLSFit

    X_mean::T2
    X_std::T2
    Yprim_mean::T3
    Yprim_std::T3
end

# function NCPLSFit(
    # X_mean::T2
    # X_std::T2
    # Yprim_mean::T3
    # Yprim_std::T3
# ) where {
#    T1<:Real,
#    T2<:AbstractArray{T1},
#    T3<:AbstractVector{T1}
# }
#     NCPLSFit{T1, T2, T3}(X_mean, X_std, Yprim_mean, Yprim_std)
# end

function Base.show(io::IO, mf::NCPLSFit)
    print(io, "NCPLSFit")
end

function Base.show(io::IO, ::MIME"text/plain", mf::NCPLSFit)
    println(io, "NCPLSFit")
end
