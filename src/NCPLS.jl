module NCPLS

using LinearAlgebra
using Random
using Statistics
import StatsAPI: fit, predict, fitted, coef, residuals
using CategoricalArrays

using Reexport: @reexport
@reexport using CategoricalArrays

include("NCPLS/types.jl")
include("NCPLS/computations.jl")
include("NCPLS/cca.jl")
include("NCPLS/fit.jl")
include("NCPLS/fit_light.jl")
include("NCPLS/landscape.jl")
include("NCPLS/scoreplot.jl")
include("NCPLS/preprocessing.jl")
include("NCPLS/predict.jl")
include("NCPLS/crossvalidation.jl")

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
export invfreqweights
export landscapeplot
export nestedcv
export nestedcvperm
export nmc
export ncomponents
export onehot
export outlierscan
export permda
export permreg
export pvalue
export predict
export predictclasses
export predictoraxes
export project
export random_batch_indices
export residuals
export responselabels
export sampleclasses
export samplelabels
export scoreplot
export cv_classification
export cv_regression
export cvda
export cvreg
export weightlandscape
export weightlandscapeplot
export weightprofiles
export weightprofilesplot
export xmean
export xscores
export xstd
export ymean

end # module NCPLS
