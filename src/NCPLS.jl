module NCPLS

using LinearAlgebra
using Random
using Statistics
import StatsAPI: fit, predict, fitted, coef, residuals

include("NCPLS/types.jl")
include("NCPLS/computations.jl")
include("NCPLS/cca.jl")
include("NCPLS/fit.jl")
include("NCPLS/landscape.jl")
include("NCPLS/scoreplot.jl")
include("NCPLS/preprocessing.jl")
include("NCPLS/predict.jl")

export AbstractNCPLSFit
export NCPLSFit
export NCPLSModel
export PredictorAxis
export coef
export coefflandscape
export coefflandscapeplot
export coefficientlandscape
export fit
export fitted
export landscapeplot
export ncomponents
export predict
export predictoraxes
export project
export residuals
export responselabels
export sampleclasses
export samplelabels
export scoreplot
export weightlandscape
export weightlandscapeplot
export weightprofiles
export weightprofilesplot
export xmean
export xscores
export xstd
export ymean

end # module NCPLS
