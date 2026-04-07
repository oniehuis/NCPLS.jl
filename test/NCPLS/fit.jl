@testset "fit_ncpls_core returns preprocessing-backed fit for matrices" begin
    model = NCPLS.NCPLSModel(
        ncomponents = 2,
        center_X = true,
        scale_X = true,
        center_Yprim = true,
        scale_Yprim = true,
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
    weights = [1.0, 2.0, 1.0, 2.0]

    d = NCPLS.preprocess(model, X, Y, weights)
    fit = NCPLS.fit_ncpls_core(model, X, Y; obs_weights = weights)

    @test fit isa NCPLS.AbstractNCPLSFit
    @test fit isa NCPLS.NCPLSFit{Float64, Vector{Float64}, Vector{Float64}}
    @test fit.X_mean ≈ d.X_mean
    @test fit.X_std ≈ d.X_std
    @test fit.Yprim_mean ≈ d.Yprim_mean
    @test fit.Yprim_std ≈ d.Yprim_std
end

@testset "fit_ncpls_core returns preprocessing-backed fit for tensors" begin
    model = NCPLS.NCPLSModel(
        ncomponents = 3,
        center_X = true,
        scale_X = true,
        center_Yprim = true,
        scale_Yprim = false,
    )
    X = reshape(collect(1.0:24.0), 4, 3, 2)
    Y = Float64[
        1 0
        0 1
        1 0
        0 1
    ]

    d = NCPLS.preprocess(model, X, Y, nothing)
    fit = NCPLS.fit_ncpls_core(model, X, Y; obs_weights = nothing)

    @test fit isa NCPLS.AbstractNCPLSFit
    @test fit isa NCPLS.NCPLSFit{Float64, Matrix{Float64}, Vector{Float64}}
    @test size(fit.X_mean) == (3, 2)
    @test fit.X_mean ≈ d.X_mean
    @test fit.X_std ≈ d.X_std
    @test fit.Yprim_mean ≈ d.Yprim_mean
    @test fit.Yprim_std ≈ d.Yprim_std
end

@testset "fit wrapper delegates to fit_ncpls_core" begin
    model = NCPLS.NCPLSModel(
        ncomponents = 1,
        center_X = true,
        scale_X = false,
        center_Yprim = true,
        scale_Yprim = true,
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
    weights = [1.0, 2.0, 1.0, 0.5]

    via_wrapper = NCPLS.fit(model, X, Y; obs_weights = weights)
    via_core = NCPLS.fit_ncpls_core(model, X, Y; obs_weights = weights)

    @test via_wrapper.X_mean ≈ via_core.X_mean
    @test via_wrapper.X_std ≈ via_core.X_std
    @test via_wrapper.Yprim_mean ≈ via_core.Yprim_mean
    @test via_wrapper.Yprim_std ≈ via_core.Yprim_std
end

@testset "fit paths surface preprocessing validation errors" begin
    model = NCPLS.NCPLSModel()
    X_matrix = rand(4, 2)
    X_tensor = rand(4, 2, 2)
    Y = rand(4, 2)

    @test_throws DimensionMismatch NCPLS.fit_ncpls_core(
        model,
        X_matrix,
        Y[1:3, :];
        obs_weights = nothing,
    )

    @test_throws ArgumentError NCPLS.fit_ncpls_core(
        model,
        X_tensor,
        Y;
        obs_weights = [1.0, -1.0, 1.0, 1.0],
    )

    @test_throws DimensionMismatch NCPLS.fit(
        model,
        X_tensor,
        Y;
        obs_weights = [1.0, 2.0, 1.0],
    )
end
