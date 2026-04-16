module MakieExtension

using Makie
import NCPLS

const ROOT = joinpath(@__DIR__, "..")
include(joinpath(ROOT, "ext", "makie_extensions", "scoreplot.jl"))

end
