using NCPLS
using Test

@testset "NCPLS/types.jl" begin
    include(joinpath("NCPLS", "types.jl"))
end

@testset "NCPLS/computations.jl" begin
    include(joinpath("NCPLS", "computations.jl"))
end

@testset "NCPLS/cca.jl" begin
    include(joinpath("NCPLS", "cca.jl"))
end

@testset "NCPLS/fit.jl" begin
    include(joinpath("NCPLS", "fit.jl"))
end

@testset "NCPLS/landscape.jl" begin
    include(joinpath("NCPLS", "landscape.jl"))
end

@testset "NCPLS/predict.jl" begin
    include(joinpath("NCPLS", "predict.jl"))
end

@testset "NCPLS/scoreplot.jl" begin
    include(joinpath("NCPLS", "scoreplot.jl"))
end

@testset "NCPLS/preprocessing.jl" begin
    include(joinpath("NCPLS", "preprocessing.jl"))
end

@testset "NCPLS/crossvalidation.jl" begin
    include(joinpath("NCPLS", "crossvalidation.jl"))
end

@testset "NCPLS/validation.jl" begin
    include(joinpath("NCPLS", "validation.jl"))
end
