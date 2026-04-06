module NCPLS

using LinearAlgebra

include("NCPLS/types.jl")
include("NCPLS/fit.jl")

export NCPLSModel
export fit_ncpls_core

end # module NCPLS
