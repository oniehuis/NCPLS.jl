import Random
import Logging

@testset "fit_ncpls_core returns fitted arrays for matrices" begin
    model = NCPLS.NCPLSModel(
        ncomponents = 2,
        center_X = true,
        scale_X = true,
        center_Yprim = true,
        multilinear = false,
    )
    X = Float64[
        1 2
        3 4
        5 6
        7 8
    ]
    Y = Float64[
        1 0
        0 1
        1 0
        0 1
    ]
    Yadd = reshape([2.0, 4.0, 6.0, 8.0], :, 1)
    weights = [1.0, 2.0, 1.0, 2.0]

    d = NCPLS.preprocess(model, X, Y, Yadd, weights)
    mf = NCPLS.fit_ncpls_core(model, X, Y; Yadd = Yadd, obs_weights = weights)

    @test mf isa NCPLS.NCPLSFit
    @test mf.model == model
    @test size(mf.B) == (2, 2, 2)
    @test size(mf.R) == (2, 2)
    @test size(mf.T) == (4, 2)
    @test size(mf.P) == (2, 2)
    @test size(mf.Q) == (2, 2)
    @test size(mf.W) == (2, 2)
    @test size(mf.c) == (3, 2)
    @test size(mf.W0) == (2, 3, 2)
    @test size(mf.Yres) == size(Y)
    @test mf.W_modes === nothing
    @test mf.W_multilinear_relerr === nothing
    @test mf.W_multilinear_method === nothing
    @test mf.W_multilinear_lambda === nothing
    @test mf.W_multilinear_niter === nothing
    @test mf.W_multilinear_converged === nothing
    @test vec(sum(mf.T .^ 2; dims = 1)) ≈ ones(2) atol = 1e-12
    @test mf.R ≈ NCPLS.score_projection_tensors(mf.W, mf.P)
    @test mf.B ≈ NCPLS.regression_coefficients(mf.R, mf.Q)
    @test mf.X_mean ≈ d.X_mean
    @test mf.X_std ≈ d.X_std
    @test mf.Yprim_mean ≈ d.Yprim_mean
    @test NCPLS.samplelabels(mf) == ["1", "2", "3", "4"]
    @test NCPLS.responselabels(mf) == String[]
    @test NCPLS.sampleclasses(mf) === nothing
    @test NCPLS.predictoraxes(mf) == NCPLS.PredictorAxis[]
end

@testset "fit_ncpls_core returns fitted arrays for tensors" begin
    model = NCPLS.NCPLSModel(
        ncomponents = 2,
        center_X = true,
        scale_X = true,
        center_Yprim = true,
        multilinear = false,
    )
    X = reshape(collect(1.0:24.0), 4, 3, 2)
    Y = Float64[
        1 0
        0 1
        1 0
        0 1
    ]
    Yadd = reshape([10.0, 20.0, 10.0, 20.0], :, 1)

    d = NCPLS.preprocess(model, X, Y, Yadd, nothing)
    mf = NCPLS.fit_ncpls_core(model, X, Y; Yadd = Yadd, obs_weights = nothing)

    @test mf isa NCPLS.NCPLSFit
    @test mf.model == model
    @test size(mf.B) == (3, 2, 2, 2)
    @test size(mf.R) == (3, 2, 2)
    @test size(mf.T) == (4, 2)
    @test size(mf.P) == (3, 2, 2)
    @test size(mf.Q) == (2, 2)
    @test size(mf.W) == (3, 2, 2)
    @test size(mf.c) == (3, 2)
    @test size(mf.W0) == (3, 2, 3, 2)
    @test size(mf.Yres) == size(Y)
    @test mf.W_modes === nothing
    @test mf.W_multilinear_relerr === nothing
    @test mf.W_multilinear_method === nothing
    @test mf.W_multilinear_lambda === nothing
    @test mf.W_multilinear_niter === nothing
    @test mf.W_multilinear_converged === nothing
    @test vec(sum(mf.T .^ 2; dims = 1)) ≈ ones(2) atol = 1e-12
    @test mf.R ≈ NCPLS.score_projection_tensors(mf.W, mf.P)
    @test mf.B ≈ NCPLS.regression_coefficients(mf.R, mf.Q)
    @test size(mf.X_mean) == (3, 2)
    @test mf.X_mean ≈ d.X_mean
    @test mf.X_std ≈ d.X_std
    @test mf.Yprim_mean ≈ d.Yprim_mean
    @test NCPLS.samplelabels(mf) == ["1", "2", "3", "4"]
    @test NCPLS.responselabels(mf) == String[]
    @test NCPLS.sampleclasses(mf) === nothing
    @test NCPLS.predictoraxes(mf) == NCPLS.PredictorAxis[]
end

@testset "fit wrapper delegates to fit_ncpls_core" begin
    model = NCPLS.NCPLSModel(
        ncomponents = 1,
        center_X = true,
        scale_X = false,
        center_Yprim = true,
        multilinear = false,
    )
    X = Float64[
        2 1 0
        0 3 1
        4 5 2
        1 4 3
    ]
    Y = Float64[
        1 0
        0 1
        1 1
        0 1
    ]
    Yadd = Float64[
        1 3
        2 4
        3 5
        4 6
    ]
    weights = [1.0, 2.0, 1.0, 0.5]

    via_wrapper = NCPLS.fit(model, X, Y; Yadd = Yadd, obs_weights = weights)
    via_core = NCPLS.fit_ncpls_core(model, X, Y; Yadd = Yadd, obs_weights = weights)

    @test via_wrapper.B ≈ via_core.B
    @test via_wrapper.R ≈ via_core.R
    @test via_wrapper.T ≈ via_core.T
    @test via_wrapper.P ≈ via_core.P
    @test via_wrapper.Q ≈ via_core.Q
    @test via_wrapper.W ≈ via_core.W
    @test via_wrapper.rho ≈ via_core.rho
    @test via_wrapper.Yres ≈ via_core.Yres
    @test via_wrapper.X_mean ≈ via_core.X_mean
    @test via_wrapper.X_std ≈ via_core.X_std
    @test via_wrapper.Yprim_mean ≈ via_core.Yprim_mean
    @test via_wrapper.samplelabels == via_core.samplelabels
    @test via_wrapper.responselabels == via_core.responselabels
    @test via_wrapper.sampleclasses === via_core.sampleclasses
    @test via_wrapper.predictoraxes == via_core.predictoraxes
end

@testset "fit paths handle optional Yadd and surface preprocessing validation errors" begin
    model = NCPLS.NCPLSModel(multilinear = false)
    X_matrix = rand(4, 2)
    X_tensor = rand(4, 2, 2)
    Y = rand(4, 2)
    Yadd = rand(4, 1)

    mf_no_yadd = NCPLS.fit_ncpls_core(
        model,
        X_matrix,
        Y;
        obs_weights = nothing,
    )
    @test mf_no_yadd isa NCPLS.NCPLSFit
    @test size(mf_no_yadd.W0) == (2, 2, 2)
    @test mf_no_yadd.W_modes === nothing
    @test mf_no_yadd.W_multilinear_relerr === nothing
    @test mf_no_yadd.W_multilinear_method === nothing
    @test mf_no_yadd.W_multilinear_lambda === nothing
    @test mf_no_yadd.W_multilinear_niter === nothing
    @test mf_no_yadd.W_multilinear_converged === nothing

    err_x_yprim = try
        NCPLS.fit_ncpls_core(
            model,
            X_matrix,
            Y[1:3, :];
            Yadd = Yadd,
            obs_weights = nothing,
        )
        nothing
    catch err
        err
    end
    @test err_x_yprim isa DimensionMismatch
    @test occursin("Number of rows in X and Yprim must be equal", sprint(showerror, err_x_yprim))

    @test_throws ArgumentError NCPLS.fit_ncpls_core(
        model,
        X_tensor,
        Y;
        Yadd = Yadd,
        obs_weights = [1.0, -1.0, 1.0, 1.0],
    )

    err_yprim_yadd = try
        NCPLS.fit(
            model,
            X_tensor,
            Y;
            Yadd = Yadd[1:3, :],
            obs_weights = nothing,
        )
        nothing
    catch err
        err
    end
    @test err_yprim_yadd isa DimensionMismatch
    @test occursin("Yprim and Yadd must have the same number of rows", sprint(showerror, err_yprim_yadd))
end

@testset "fit stores and validates metadata" begin
    model = NCPLS.NCPLSModel(ncomponents = 1, multilinear = false)
    X = reshape(collect(1.0:24.0), 4, 3, 2)
    Y = Float64[
        1 0
        0 1
        1 0
        0 1
    ]

    axes = (
        (name = "RT", values = [5.0, 5.5, 6.0], unit = "min"),
        NCPLS.PredictorAxis("m/z", [91, 105]; unit = "Da"),
    )

    mf = NCPLS.fit_ncpls_core(
        model,
        X,
        Y;
        samplelabels = [:s1, :s2, :s3, :s4],
        responselabels = [:species_A, :species_B],
        sampleclasses = [:A, :B, :A, :B],
        predictoraxes = axes,
    )

    @test NCPLS.samplelabels(mf) == ["s1", "s2", "s3", "s4"]
    @test NCPLS.responselabels(mf) == ["species_A", "species_B"]
    @test NCPLS.sampleclasses(mf) == [:A, :B, :A, :B]
    @test length(NCPLS.predictoraxes(mf)) == 2
    @test NCPLS.predictoraxes(mf)[1].name == "RT"
    @test NCPLS.predictoraxes(mf)[1].values == [5.0, 5.5, 6.0]
    @test NCPLS.predictoraxes(mf)[1].unit == "min"
    @test NCPLS.predictoraxes(mf)[2].name == "m/z"
    @test NCPLS.predictoraxes(mf)[2].values == [91, 105]
    @test NCPLS.predictoraxes(mf)[2].unit == "Da"

    @test_throws ArgumentError NCPLS.fit_ncpls_core(
        model,
        X,
        Y;
        samplelabels = ["s1", "s2", "s3"],
    )
    @test_throws ArgumentError NCPLS.fit_ncpls_core(
        model,
        X,
        Y;
        responselabels = ["species_A"],
    )
    @test_throws ArgumentError NCPLS.fit_ncpls_core(
        model,
        X,
        Y;
        sampleclasses = [:A, :B, :A],
    )
    @test_throws ArgumentError NCPLS.fit_ncpls_core(
        model,
        X,
        Y;
        predictoraxes = [(name = "RT", values = [1.0, 2.0, 3.0])],
    )
    @test_throws ArgumentError NCPLS.fit_ncpls_core(
        model,
        X,
        Y;
        predictoraxes = (
            (name = "RT", values = [1.0, 2.0], unit = "min"),
            (name = "m/z", values = [91, 105], unit = "Da"),
        ),
    )
end

@testset "fit stores multilinear mode weights and diagnostics" begin
    model = NCPLS.NCPLSModel(
        ncomponents = 2,
        center_X = true,
        scale_X = false,
        center_Yprim = true,
        multilinear = true,
        orthogonalize_mode_weights = false,
    )
    X = reshape(collect(1.0:24.0), 4, 3, 2)
    Y = Float64[
        1 0
        0 1
        1 0
        0 1
    ]
    Yadd = reshape([1.0, 2.0, 3.0, 4.0], :, 1)

    mf = NCPLS.fit_ncpls_core(model, X, Y; Yadd = Yadd, obs_weights = nothing)

    @test mf isa NCPLS.NCPLSFit
    @test mf.W_modes isa AbstractVector
    @test length(mf.W_modes) == ndims(X) - 1
    @test size(mf.W_modes[1]) == (size(X, 2), model.ncomponents)
    @test size(mf.W_modes[2]) == (size(X, 3), model.ncomponents)
    @test all(i -> isapprox(sqrt(sum(abs2, mf.W_modes[1][:, i])), 1.0; atol = 1e-12), 1:model.ncomponents)
    @test all(i -> isapprox(sqrt(sum(abs2, mf.W_modes[2][:, i])), 1.0; atol = 1e-12), 1:model.ncomponents)
    @test mf.W_multilinear_relerr isa AbstractVector
    @test mf.W_multilinear_method isa AbstractVector
    @test mf.W_multilinear_lambda isa AbstractVector
    @test mf.W_multilinear_niter isa AbstractVector
    @test mf.W_multilinear_converged isa AbstractVector
    @test length(mf.W_multilinear_relerr) == model.ncomponents
    @test length(mf.W_multilinear_method) == model.ncomponents
    @test length(mf.W_multilinear_lambda) == model.ncomponents
    @test length(mf.W_multilinear_niter) == model.ncomponents
    @test length(mf.W_multilinear_converged) == model.ncomponents
    @test all(m -> m == :svd, mf.W_multilinear_method)
    @test all(==(0), mf.W_multilinear_niter)
    @test all(identity, mf.W_multilinear_converged)

    for i in 1:model.ncomponents
        Wi = selectdim(mf.W, ndims(mf.W), i)
        expected = NCPLS.outer_tensor([mf.W_modes[1][:, i], mf.W_modes[2][:, i]])
        @test Wi ≈ expected atol = 1e-10
    end

    @test mf.R ≈ NCPLS.score_projection_tensors(mf.W, mf.P)
    @test mf.B ≈ NCPLS.regression_coefficients(mf.R, mf.Q)
end

@testset "fit forwards multilinear control settings from the model" begin
    model = NCPLS.NCPLSModel(
        ncomponents = 1,
        multilinear = true,
        orthogonalize_mode_weights = false,
        multilinear_maxiter = 1,
        multilinear_tol = 0.0,
        multilinear_init = :random,
        multilinear_seed = 7,
    )
    X = reshape(collect(1.0:48.0), 4, 2, 3, 2)
    Y = Float64[
        1 0
        0 1
        1 0
        0 1
    ]

    mf = NCPLS.fit_ncpls_core(model, X, Y; obs_weights = nothing)

    W0_1 = selectdim(mf.W0, ndims(mf.W0), 1)
    W_1 = NCPLS.loading_weights(W0_1, mf.c[:, 1])
    W_modes_prev = [zeros(size(X, j + 1), 0) for j in 1:(ndims(X) - 1)]
    direct = NCPLS.multilinear_loading_weight_tensor(
        W_1,
        W_modes_prev,
        model,
        Random.MersenneTwister(model.multilinear_seed),
    )

    @test mf.W_multilinear_method[1] == direct.method
    @test mf.W_multilinear_relerr[1] ≈ direct.relerr
    @test mf.W_multilinear_lambda[1] ≈ direct.lambda
    @test mf.W_multilinear_niter[1] == direct.niter
    @test mf.W_multilinear_converged[1] == direct.converged
    @test selectdim(mf.W, ndims(mf.W), 1) ≈ direct.Wᵒ
    @test mf.W_modes[1][:, 1] ≈ direct.factors[1]
    @test mf.W_modes[2][:, 1] ≈ direct.factors[2]
    @test mf.W_modes[3][:, 1] ≈ direct.factors[3]
end

@testset "fit accepts verbose multilinear fitting runtime control" begin
    model = NCPLS.NCPLSModel(
        ncomponents = 1,
        multilinear = true,
        multilinear_init = :random,
        multilinear_maxiter = 1,
        multilinear_tol = 0.0,
        multilinear_seed = 3,
    )
    X = reshape(collect(1.0:48.0), 4, 2, 3, 2)
    Y = Float64[
        1 0
        0 1
        1 0
        0 1
    ]

    mf = Logging.with_logger(Logging.NullLogger()) do
        NCPLS.fit_ncpls_core(model, X, Y; obs_weights = nothing, verbose = true)
    end

    @test mf isa NCPLS.NCPLSFit
    @test mf.W_modes isa AbstractVector
    @test mf.W_multilinear_method[1] == :parafac
    @test mf.W_multilinear_niter[1] == model.multilinear_maxiter
    @test mf.W_multilinear_converged[1] === false
end
