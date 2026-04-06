"""
    NCPLSModel

Model specification passed to `fit`. A `NCPLSModel` stores the user-controlled settings
for n-CPLS fitting, most importantly `ncomponents`, centering and scaling of predictor 
and response variables, and `mode`.
"""
struct NCPLSModel
    ncomponents::Int
    center_X::Bool
    scale_X::Bool
    scale_Yprim::Bool
    X_tolerance::Float64
    X_loading_weight_tolerance::Float64
    t_squared_norm_tolerance::Float64
    mode::Symbol
end

"""
    NCPLSModel(; 
        ncomponents::Integer=2,
        center_X::Bool=true,
        scale_X::Bool=false,
        scale_Yprim::Bool=false,
        X_tolerance::Real=1e-12, 
        X_loading_weight_tolerance::Real=eps(Float64), 
        t_squared_norm_tolerance::Real=1e-10,
        mode::Symbol=:regression
    )

Construct a model specification for `fit`. The most commonly adjusted settings are
`ncomponents`, and `mode`.
"""
function NCPLSModel(;
    ncomponents::T1=2,
    center_X::Bool=true,
    scale_X::Bool=false,
    scale_Yprim::Bool=false,
    X_tolerance::T2=1e-12,
    X_loading_weight_tolerance::T3=eps(Float64),
    t_squared_norm_tolerance::T4=1e-10,
    mode::Symbol=:regression
) where {
        T1<:Integer, 
        T2<:Real, 
        T3<:Real, 
        T4<:Real
    }

    ncomponents > 0 || throw(ArgumentError("ncomponents must be greater than zero"))
    mode in (:regression, :discriminant) || throw(ArgumentError(
            "mode must be :regression or :discriminant, got $mode"))

    NCPLSModel(
        Int(ncomponents),
        center_X,
        scale_X,
        scale_Yprim,
        Float64(X_tolerance),
        Float64(X_loading_weight_tolerance),
        Float64(t_squared_norm_tolerance),
        mode
    )
end

function Base.show(io::IO, spec::NCPLSModel)
    print(io, "NCPLSModel(",
        "ncomponents=", spec.ncomponents,
        ", center_X=", spec.center_X,
        ", scale_X=", spec.scale_X,
        ", scale_Yprim=", spec.scale_Yprim,
        ", mode=", spec.mode,
        ")")
end

function Base.show(io::IO, ::MIME"text/plain", spec::NCPLSModel)
    println(io, "NCPLSModel")
    println(io, "  ncomponents: ", spec.ncomponents)
    println(io, "  center_X: ", spec.center_X)
    println(io, "  scale_X: ", spec.scale_X)
    println(io, "  scale_Yprim: ", spec.scale_Yprim)
    print(io, "  mode: ", spec.mode)
end
