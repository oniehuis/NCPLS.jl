@testset "NCPLSModel constructor and fields" begin
    model = NCPLS.NCPLSModel()
    @test model.ncomponents == 2
    @test model.center_X === true
    @test model.scale_X === false
    @test model.center_Yprim === true
    @test model.scale_Yprim === false
    @test model.center_Yadd === true
    @test model.scale_Yadd === false
    @test model.multilinear === false

    custom = NCPLS.NCPLSModel(
        ncomponents = 4,
        center_X = false,
        scale_X = true,
        center_Yprim = false,
        scale_Yprim = true,
        center_Yadd = false,
        scale_Yadd = true,
        multilinear = true,
    )
    @test custom.ncomponents == 4
    @test custom.center_X === false
    @test custom.scale_X === true
    @test custom.center_Yprim === false
    @test custom.scale_Yprim === true
    @test custom.center_Yadd === false
    @test custom.scale_Yadd === true
    @test custom.multilinear === true

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
        multilinear = true,
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
    @test occursin("multilinear=true", compact)

    plain = sprint(show, MIME"text/plain"(), model)
    @test occursin("NCPLSModel", plain)
    @test occursin("ncomponents: 3", plain)
    @test occursin("center_X: false", plain)
    @test occursin("scale_X: true", plain)
    @test occursin("center_Yprim: true", plain)
    @test occursin("scale_Yprim: true", plain)
    @test occursin("center_Yadd: false", plain)
    @test occursin("scale_Yadd: true", plain)
    @test occursin("multilinear: true", plain)
end

@testset "NCPLSFit stores fitted arrays and preprocessing metadata" begin
    model = NCPLS.NCPLSModel(ncomponents = 2)
    matrix_fit = NCPLS.NCPLSFit(
        model,
        reshape([1.0, 2.0, 3.0, 4.0], 2, 2, 1),
        [1.0 2.0; 3.0 4.0],
        [1.0 0.0; 0.0 1.0; 1.0 1.0],
        [2.0 1.0; 0.0 1.0],
        reshape([5.0, 6.0], 1, 2),
        [1.0 0.0; 0.0 1.0],
        reshape([1.0, 2.0], 1, 2),
        reshape(collect(1.0:4.0), 2, 1, 2),
        [0.5, 0.7],
        reshape([1.0, 2.0, 3.0], :, 1),
        [1.0, 2.0],
        [0.5, 1.5],
        [3.0],
        [2.0],
        [4.0, 5.0],
        [1.0, 1.5],
    )
    @test matrix_fit isa NCPLS.AbstractNCPLSFit
    @test matrix_fit isa NCPLS.NCPLSFit
    @test matrix_fit.model == model
    @test size(matrix_fit.B) == (2, 2, 1)
    @test size(matrix_fit.R) == (2, 2)
    @test size(matrix_fit.T) == (3, 2)
    @test size(matrix_fit.P) == (2, 2)
    @test size(matrix_fit.Q) == (1, 2)
    @test size(matrix_fit.W) == (2, 2)
    @test size(matrix_fit.c) == (1, 2)
    @test size(matrix_fit.W0) == (2, 1, 2)
    @test matrix_fit.rho == [0.5, 0.7]
    @test size(matrix_fit.Yres) == (3, 1)
    @test matrix_fit.X_mean == [1.0, 2.0]
    @test matrix_fit.X_std == [0.5, 1.5]
    @test matrix_fit.Yprim_mean == [3.0]
    @test matrix_fit.Yprim_std == [2.0]
    @test matrix_fit.Yadd_mean == [4.0, 5.0]
    @test matrix_fit.Yadd_std == [1.0, 1.5]

    tensor_fit = NCPLS.NCPLSFit(
        model,
        reshape(collect(1.0:16.0), 2, 2, 2, 2),
        reshape(collect(1.0:8.0), 2, 2, 2),
        [1.0 0.0; 0.0 1.0; 1.0 1.0; 0.5 0.5],
        reshape(collect(1.0:8.0), 2, 2, 2),
        [1.0 2.0; 3.0 4.0],
        reshape(collect(1.0:8.0), 2, 2, 2),
        [0.1 0.2; 0.3 0.4],
        reshape(collect(1.0:16.0), 2, 2, 2, 2),
        [0.6, 0.8],
        [1.0 0.0; 0.0 1.0; 1.0 1.0; 0.5 0.5],
        [1.0 2.0; 3.0 4.0],
        [0.5 1.5; 2.5 3.5],
        [2.0, 5.0],
        [1.0, 4.0],
        [3.0],
        [2.0],
    )
    @test tensor_fit isa NCPLS.AbstractNCPLSFit
    @test tensor_fit isa NCPLS.NCPLSFit
    @test tensor_fit.model == model
    @test size(tensor_fit.B) == (2, 2, 2, 2)
    @test size(tensor_fit.R) == (2, 2, 2)
    @test size(tensor_fit.T) == (4, 2)
    @test size(tensor_fit.P) == (2, 2, 2)
    @test size(tensor_fit.Q) == (2, 2)
    @test size(tensor_fit.W) == (2, 2, 2)
    @test size(tensor_fit.c) == (2, 2)
    @test size(tensor_fit.W0) == (2, 2, 2, 2)
    @test tensor_fit.rho == [0.6, 0.8]
    @test size(tensor_fit.Yres) == (4, 2)
    @test tensor_fit.X_mean == [1.0 2.0; 3.0 4.0]
    @test tensor_fit.X_std == [0.5 1.5; 2.5 3.5]
    @test tensor_fit.Yprim_mean == [2.0, 5.0]
    @test tensor_fit.Yprim_std == [1.0, 4.0]
    @test tensor_fit.Yadd_mean == [3.0]
    @test tensor_fit.Yadd_std == [2.0]

    no_yadd_fit = NCPLS.NCPLSFit(
        model,
        reshape([1.0, 2.0, 3.0, 4.0], 2, 2, 1),
        [1.0 2.0; 3.0 4.0],
        [1.0 0.0; 0.0 1.0; 1.0 1.0],
        [2.0 1.0; 0.0 1.0],
        reshape([5.0, 6.0], 1, 2),
        [1.0 0.0; 0.0 1.0],
        reshape([1.0, 2.0], 1, 2),
        reshape(collect(1.0:4.0), 2, 1, 2),
        [0.5, 0.7],
        reshape([1.0, 2.0, 3.0], :, 1),
        [1.0, 2.0],
        [0.5, 1.5],
        [3.0],
        [2.0],
        nothing,
        nothing,
    )
    @test no_yadd_fit isa NCPLS.AbstractNCPLSFit
    @test no_yadd_fit isa NCPLS.NCPLSFit
    @test no_yadd_fit.Yadd_mean === nothing
    @test no_yadd_fit.Yadd_std === nothing
end

@testset "NCPLSFit show methods" begin
    model = NCPLS.NCPLSModel()
    fit = NCPLS.NCPLSFit(
        model,
        reshape([1.0, 2.0, 3.0, 4.0], 2, 2, 1),
        [1.0 2.0; 3.0 4.0],
        [1.0 0.0; 0.0 1.0; 1.0 1.0],
        [2.0 1.0; 0.0 1.0],
        reshape([5.0, 6.0], 1, 2),
        [1.0 0.0; 0.0 1.0],
        reshape([1.0, 2.0], 1, 2),
        reshape(collect(1.0:4.0), 2, 1, 2),
        [0.5, 0.7],
        reshape([1.0, 2.0, 3.0], :, 1),
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
