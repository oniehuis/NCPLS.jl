function mock_matrix_fit()
    model = NCPLS.NCPLSModel(ncomponents = 2)
    NCPLS.NCPLSFit(
        model,
        reshape([1.0, 2.0, 3.0, 4.0], 2, 2, 1),
        [1.0 2.0; 3.0 4.0],
        [1.0 0.0; 0.0 1.0; 1.0 1.0],
        [2.0 1.0; 0.0 1.0],
        reshape([5.0, 6.0], 1, 2),
        [1.0 0.0; 0.0 1.0],
        nothing,
        reshape([1.0, 2.0], 1, 2),
        reshape(collect(1.0:4.0), 2, 1, 2),
        [0.5, 0.7],
        reshape([1.0, 2.0, 3.0], :, 1),
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        [1.0, 2.0],
        [2.0, 4.0],
        [10.0],
        [5.0],
    )
end

function mock_tensor_fit()
    model = NCPLS.NCPLSModel(ncomponents = 2)
    NCPLS.NCPLSFit(
        model,
        reshape(collect(1.0:16.0), 2, 2, 2, 2),
        reshape(collect(1.0:8.0), 2, 2, 2),
        [1.0 0.0; 0.0 1.0; 1.0 1.0; 0.5 0.5],
        reshape(collect(1.0:8.0), 2, 2, 2),
        [1.0 2.0; 3.0 4.0],
        reshape(collect(1.0:8.0), 2, 2, 2),
        nothing,
        [0.1 0.2; 0.3 0.4],
        reshape(collect(1.0:16.0), 2, 2, 2, 2),
        [0.6, 0.8],
        [1.0 0.0; 0.0 1.0; 1.0 1.0; 0.5 0.5],
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        [1.0 2.0; 3.0 4.0],
        [0.5 1.5; 2.5 3.5],
        [2.0, 5.0],
        [1.0, 4.0],
    )
end

@testset "NCPLSModel constructor and fields" begin
    model = NCPLS.NCPLSModel()
    @test model.ncomponents == 2
    @test model.center_X === true
    @test model.scale_X === false
    @test model.center_Yprim === true
    @test model.multilinear === false
    @test model.orthogonalize_mode_weights === false
    @test model.multilinear_maxiter == 500
    @test model.multilinear_tol == 1e-10
    @test model.multilinear_init == :hosvd
    @test model.multilinear_seed == 1

    custom = NCPLS.NCPLSModel(
        ncomponents = 4,
        center_X = false,
        scale_X = true,
        center_Yprim = false,
        multilinear = true,
        orthogonalize_mode_weights = true,
        multilinear_maxiter = 50,
        multilinear_tol = 1e-8,
        multilinear_init = :random,
        multilinear_seed = 7,
    )
    @test custom.ncomponents == 4
    @test custom.center_X === false
    @test custom.scale_X === true
    @test custom.center_Yprim === false
    @test custom.multilinear === true
    @test custom.orthogonalize_mode_weights === true
    @test custom.multilinear_maxiter == 50
    @test custom.multilinear_tol == 1e-8
    @test custom.multilinear_init == :random
    @test custom.multilinear_seed == 7

    @test_throws ArgumentError NCPLS.NCPLSModel(ncomponents = 0)
    @test_throws ArgumentError NCPLS.NCPLSModel(ncomponents = -1)
    @test_throws ArgumentError NCPLS.NCPLSModel(multilinear_maxiter = 0)
    @test_throws ArgumentError NCPLS.NCPLSModel(multilinear_tol = -1.0)
    @test_throws ArgumentError NCPLS.NCPLSModel(multilinear_init = :bad)
end

@testset "NCPLSModel show methods" begin
    model = NCPLS.NCPLSModel(
        ncomponents = 3,
        center_X = false,
        scale_X = true,
        center_Yprim = true,
        multilinear = true,
        orthogonalize_mode_weights = true,
        multilinear_maxiter = 75,
        multilinear_tol = 1e-8,
        multilinear_init = :random,
        multilinear_seed = 9,
    )

    compact = sprint(show, model)
    @test occursin("NCPLSModel(", compact)
    @test occursin("ncomponents=3", compact)
    @test occursin("multilinear=true", compact)
    @test occursin("multilinear_init=:random", compact)

    plain = sprint(show, MIME"text/plain"(), model)
    @test occursin("NCPLSModel", plain)
    @test occursin("ncomponents: 3", plain)
    @test occursin("multilinear: true", plain)
    @test occursin("multilinear_init: random", plain)
end

@testset "NCPLSFit stores fitted state and optional metadata" begin
    matrix_mf = mock_matrix_fit()
    @test matrix_mf isa NCPLS.NCPLSFit
    @test size(matrix_mf.B) == (2, 2, 1)
    @test size(matrix_mf.R) == (2, 2)
    @test size(matrix_mf.T) == (3, 2)
    @test size(matrix_mf.P) == (2, 2)
    @test size(matrix_mf.Q) == (1, 2)
    @test size(matrix_mf.W) == (2, 2)
    @test size(matrix_mf.c) == (1, 2)
    @test size(matrix_mf.W0) == (2, 1, 2)
    @test matrix_mf.W_modes === nothing
    @test matrix_mf.W_multilinear_relerr === nothing
    @test matrix_mf.W_multilinear_method === nothing
    @test matrix_mf.W_multilinear_lambda === nothing
    @test matrix_mf.W_multilinear_niter === nothing
    @test matrix_mf.W_multilinear_converged === nothing

    tensor_mf = mock_tensor_fit()
    @test tensor_mf isa NCPLS.NCPLSFit
    @test size(tensor_mf.B) == (2, 2, 2, 2)
    @test size(tensor_mf.R) == (2, 2, 2)
    @test size(tensor_mf.T) == (4, 2)
    @test size(tensor_mf.P) == (2, 2, 2)
    @test size(tensor_mf.Q) == (2, 2)
    @test size(tensor_mf.W) == (2, 2, 2)
    @test size(tensor_mf.c) == (2, 2)
    @test size(tensor_mf.W0) == (2, 2, 2, 2)
    @test size(tensor_mf.X_mean) == (2, 2)
    @test size(tensor_mf.X_std) == (2, 2)
end

@testset "NCPLSFit show methods" begin
    mf = mock_matrix_fit()

    compact = sprint(show, mf)
    @test compact == "NCPLSFit(samples=3, predictor_dims=(2,), responses=1, components=2, multilinear=false)"

    plain = sprint(show, MIME"text/plain"(), mf)
    @test plain == """
NCPLSFit
  samples: 3
  predictor_dims: (2,)
  responses: 1
  components: 2
  multilinear: false"""
end

@testset "NCPLSFit helper methods" begin
    mf = mock_matrix_fit()

    @test NCPLS.ncomponents(mf) == 2
    @test NCPLS.validate_ncomponents(mf, 1) == 1
    @test NCPLS.validate_ncomponents(mf, 2) == 2
    @test_throws DimensionMismatch NCPLS.validate_ncomponents(mf, 0)
    @test_throws DimensionMismatch NCPLS.validate_ncomponents(mf, 3)

    Y = [1.0 2.0; 3.0 4.0]
    @test NCPLS.restore_response_scale(Y, mf; add_mean = true) ≈
        Y .* reshape(mf.Yprim_std, 1, :) .+ reshape(mf.Yprim_mean, 1, :)
    @test NCPLS.restore_response_scale(Y, mf; add_mean = false) ≈
        Y .* reshape(mf.Yprim_std, 1, :)
end

@testset "NCPLSFit getters" begin
    mf = mock_matrix_fit()

    @test NCPLS.xmean(mf) === mf.X_mean
    @test NCPLS.xstd(mf) === mf.X_std
    @test NCPLS.ymean(mf) === mf.Yprim_mean
    @test NCPLS.ystd(mf) === mf.Yprim_std

    @test NCPLS.xscores(mf) === mf.T
    @test NCPLS.xscores(mf, 1) == mf.T[:, 1]
    @test NCPLS.xscores(mf, 1:2) == mf.T[:, 1:2]
    @test NCPLS.xscores(mf, [2, 1]) == mf.T[:, [2, 1]]

    @test_throws ArgumentError NCPLS.xscores(mf, 0)
    @test_throws ArgumentError NCPLS.xscores(mf, 3)
    @test_throws ArgumentError NCPLS.xscores(mf, 0:1)
    @test_throws ArgumentError NCPLS.xscores(mf, [1, 3])
end
