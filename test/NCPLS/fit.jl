@testset "fit_ncpls_core returns preprocessing-backed fit for matrices" begin
    model = NCPLS.NCPLSModel(
        ncomponents = 2,
        center_X = true,
        scale_X = true,
        center_Yprim = true,
        scale_Yprim = true,
        center_Yadd = true,
        scale_Yadd = true,
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
    fit = NCPLS.fit_ncpls_core(model, X, Y; Yadd = Yadd, obs_weights = weights)

    @test fit isa NCPLS.AbstractNCPLSFit
    @test fit isa NCPLS.NCPLSFit{Float64, Vector{Float64}, Vector{Float64}, Vector{Float64}}
    @test fit.X_mean ≈ d.X_mean
    @test fit.X_std ≈ d.X_std
    @test fit.Yprim_mean ≈ d.Yprim_mean
    @test fit.Yprim_std ≈ d.Yprim_std
    @test fit.Yadd_mean ≈ d.Yadd_mean
    @test fit.Yadd_std ≈ d.Yadd_std
end

@testset "fit_ncpls_core returns preprocessing-backed fit for tensors" begin
    model = NCPLS.NCPLSModel(
        ncomponents = 3,
        center_X = true,
        scale_X = true,
        center_Yprim = true,
        scale_Yprim = false,
        center_Yadd = true,
        scale_Yadd = false,
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
    fit = NCPLS.fit_ncpls_core(model, X, Y; Yadd = Yadd, obs_weights = nothing)

    @test fit isa NCPLS.AbstractNCPLSFit
    @test fit isa NCPLS.NCPLSFit{Float64, Matrix{Float64}, Vector{Float64}, Vector{Float64}}
    @test size(fit.X_mean) == (3, 2)
    @test fit.X_mean ≈ d.X_mean
    @test fit.X_std ≈ d.X_std
    @test fit.Yprim_mean ≈ d.Yprim_mean
    @test fit.Yprim_std ≈ d.Yprim_std
    @test fit.Yadd_mean ≈ d.Yadd_mean
    @test fit.Yadd_std ≈ d.Yadd_std
end

@testset "fit wrapper delegates to fit_ncpls_core" begin
    model = NCPLS.NCPLSModel(
        ncomponents = 1,
        center_X = true,
        scale_X = false,
        center_Yprim = true,
        scale_Yprim = true,
        center_Yadd = false,
        scale_Yadd = true,
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

    @test via_wrapper.X_mean ≈ via_core.X_mean
    @test via_wrapper.X_std ≈ via_core.X_std
    @test via_wrapper.Yprim_mean ≈ via_core.Yprim_mean
    @test via_wrapper.Yprim_std ≈ via_core.Yprim_std
    @test via_wrapper.Yadd_mean ≈ via_core.Yadd_mean
    @test via_wrapper.Yadd_std ≈ via_core.Yadd_std
end

@testset "fit paths handle optional Yadd and surface preprocessing validation errors" begin
    model = NCPLS.NCPLSModel()
    X_matrix = rand(4, 2)
    X_tensor = rand(4, 2, 2)
    Y = rand(4, 2)
    Yadd = rand(4, 1)

    fit_no_yadd = NCPLS.fit_ncpls_core(
        model,
        X_matrix,
        Y;
        obs_weights = nothing,
    )
    @test fit_no_yadd isa NCPLS.AbstractNCPLSFit
    @test fit_no_yadd isa NCPLS.NCPLSFit{Float64, Vector{Float64}, Vector{Float64}, Nothing}
    @test fit_no_yadd.Yadd_mean === nothing
    @test fit_no_yadd.Yadd_std === nothing

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
