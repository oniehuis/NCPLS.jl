module NCPLS

using LinearAlgebra
using Random
using Statistics
import StatsAPI: fit

include("NCPLS/types.jl")
include("NCPLS/cca.jl")
include("NCPLS/fit.jl")
include("NCPLS/preprocessing.jl")

export NCPLSModel
export fit_ncpls_core

end # module NCPLS
