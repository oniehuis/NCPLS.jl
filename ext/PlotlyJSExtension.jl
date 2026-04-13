module PlotlyJSExtension

using PlotlyJS
using Statistics
import NCPLS

const ROOT = joinpath(@__DIR__, "..")
include(joinpath(ROOT, "ext", "plotly_extensions", "scoreplot.jl"))
include(joinpath(ROOT, "ext", "plotly_extensions", "landscapeplot.jl"))

end
