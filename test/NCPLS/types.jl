@testset "NCPLSModel constructor and fields" begin
    model = NCPLS.NCPLSModel()
    @test model.ncomponents == 2
    @test model.center_X === true
    @test model.scale_X === false
    @test model.center_Yprim === true
    @test model.scale_Yprim === false
    @test model.center_Yadd === true
    @test model.scale_Yadd === false

    custom = NCPLS.NCPLSModel(
        ncomponents = 4,
        center_X = false,
        scale_X = true,
        center_Yprim = false,
        scale_Yprim = true,
        center_Yadd = false,
        scale_Yadd = true,
    )
    @test custom.ncomponents == 4
    @test custom.center_X === false
    @test custom.scale_X === true
    @test custom.center_Yprim === false
    @test custom.scale_Yprim === true
    @test custom.center_Yadd === false
    @test custom.scale_Yadd === true

    @test_throws ArgumentError NCPLS.NCPLSModel(ncomponents = 0)
    @test_throws ArgumentError NCPLS.NCPLSModel(ncomponents = -1)
end

@testset "NCPLSModel show methods" begin
    model = NCPLS.NCPLSModel(
        ncomponents = 3,
        center_X = false,
        scale_X = true,
        center_Yprim = true,
        scale_Yprim = true,
        center_Yadd = false,
        scale_Yadd = true,
    )

    compact = sprint(show, model)
    @test occursin("NCPLSModel(", compact)
    @test occursin("ncomponents=3", compact)
    @test occursin("center_X=false", compact)
    @test occursin("scale_X=true", compact)
    @test occursin("center_Yprim=true", compact)
    @test occursin("scale_Yprim=true", compact)
    @test occursin("center_Yadd=false", compact)
    @test occursin("scale_Yadd=true", compact)

    plain = sprint(show, MIME"text/plain"(), model)
    @test occursin("NCPLSModel", plain)
    @test occursin("ncomponents: 3", plain)
    @test occursin("center_X: false", plain)
    @test occursin("scale_X: true", plain)
    @test occursin("center_Yprim: true", plain)
    @test occursin("scale_Yprim: true", plain)
    @test occursin("center_Yadd: false", plain)
    @test occursin("scale_Yadd: true", plain)
end

@testset "NCPLSFit stores matrix and tensor preprocessing metadata" begin
    matrix_fit = NCPLS.NCPLSFit(
        [1.0, 2.0],
        [0.5, 1.5],
        [3.0],
        [2.0],
        [4.0, 5.0],
        [1.0, 1.5],
    )
    @test matrix_fit isa NCPLS.AbstractNCPLSFit
    @test matrix_fit isa NCPLS.NCPLSFit{Float64, Vector{Float64}, Vector{Float64}, Vector{Float64}}
    @test matrix_fit.X_mean == [1.0, 2.0]
    @test matrix_fit.X_std == [0.5, 1.5]
    @test matrix_fit.Yprim_mean == [3.0]
    @test matrix_fit.Yprim_std == [2.0]
    @test matrix_fit.Yadd_mean == [4.0, 5.0]
    @test matrix_fit.Yadd_std == [1.0, 1.5]

    tensor_fit = NCPLS.NCPLSFit(
        [1.0 2.0; 3.0 4.0],
        [0.5 1.5; 2.5 3.5],
        [2.0, 5.0],
        [1.0, 4.0],
        [3.0],
        [2.0],
    )
    @test tensor_fit isa NCPLS.AbstractNCPLSFit
    @test tensor_fit isa NCPLS.NCPLSFit{Float64, Matrix{Float64}, Vector{Float64}, Vector{Float64}}
    @test tensor_fit.X_mean == [1.0 2.0; 3.0 4.0]
    @test tensor_fit.X_std == [0.5 1.5; 2.5 3.5]
    @test tensor_fit.Yprim_mean == [2.0, 5.0]
    @test tensor_fit.Yprim_std == [1.0, 4.0]
    @test tensor_fit.Yadd_mean == [3.0]
    @test tensor_fit.Yadd_std == [2.0]

    no_yadd_fit = NCPLS.NCPLSFit(
        [1.0, 2.0],
        [0.5, 1.5],
        [3.0],
        [2.0],
        nothing,
        nothing,
    )
    @test no_yadd_fit isa NCPLS.AbstractNCPLSFit
    @test no_yadd_fit isa NCPLS.NCPLSFit{Float64, Vector{Float64}, Vector{Float64}, Nothing}
    @test no_yadd_fit.Yadd_mean === nothing
    @test no_yadd_fit.Yadd_std === nothing
end

@testset "NCPLSFit show methods" begin
    fit = NCPLS.NCPLSFit(
        [1.0, 2.0],
        [1.0, 1.0],
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    )

    compact = sprint(show, fit)
    @test compact == "NCPLSFit"

    plain = sprint(show, MIME"text/plain"(), fit)
    @test plain == "NCPLSFit\n"
end
