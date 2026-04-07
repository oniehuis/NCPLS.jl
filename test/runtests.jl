using NCPLS
using Test

@testset "NCPLS/types.jl" begin
    include(joinpath("NCPLS", "types.jl"))
end

# @testset "NCPLS/cca.jl" begin
#     include(joinpath("NCPLS", "cca.jl"))
# end

@testset "NCPLS/fit.jl" begin
    include(joinpath("NCPLS", "fit.jl"))
end

@testset "NCPLS/preprocessing.jl" begin
    include(joinpath("NCPLS", "preprocessing.jl"))
end
