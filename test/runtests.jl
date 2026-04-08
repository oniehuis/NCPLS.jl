using NCPLS
using Test

@testset "NCPLS/types.jl" begin
    include(joinpath("NCPLS", "types.jl"))
end

@testset "NCPLS/computations.jl" begin
    include(joinpath("NCPLS", "computations.jl"))
end

@testset "NCPLS/fit.jl" begin
    include(joinpath("NCPLS", "fit.jl"))
end

@testset "NCPLS/predict.jl" begin
    include(joinpath("NCPLS", "predict.jl"))
end

@testset "NCPLS/preprocessing.jl" begin
    include(joinpath("NCPLS", "preprocessing.jl"))
end

@testset "NCPLS/validation.jl" begin
    include(joinpath("NCPLS", "validation.jl"))
end
