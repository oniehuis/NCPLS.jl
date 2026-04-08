module NCPLS

using LinearAlgebra
using Random
using Statistics
import CPPLS
import StatsAPI: fit, predict, fitted, coef, residuals
import CPPLS: cca_coeffs_and_corr

include("NCPLS/types.jl")
include("NCPLS/computations.jl")
include("NCPLS/fit.jl")
include("NCPLS/preprocessing.jl")
include("NCPLS/predict.jl")

export AbstractNCPLSFit
export NCPLSFit
export NCPLSModel
export coef
export fit
export fitted
export ncomponents
export predict
export project
export residuals
export xmean
export xscores
export xstd
export ymean
export ystd

end # module NCPLS
