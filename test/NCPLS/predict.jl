import Random

function mock_matrix_fit()
    NCPLS.NCPLSFit(
        NCPLS.NCPLSModel(multilinear = false),
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
    )
end

function mock_tensor_fit()
    NCPLS.NCPLSFit(
        NCPLS.NCPLSModel(multilinear = false),
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
        [2.0 4.0; 5.0 10.0],
        [2.0, 5.0],
    )
end

@testset "normalize_predictors helper" begin
    matrix_mf = mock_matrix_fit()
    Xmat = [3.0 10.0; 5.0 14.0]
    @test NCPLS.normalize_predictors(Xmat, matrix_mf) ≈ [1.0 2.0; 2.0 3.0]

    tensor_mf = mock_tensor_fit()
    Xtensor = reshape([3.0, 5.0, 13.0, 18.0, 14.0, 18.0, 44.0, 54.0], 2, 2, 2)
    expected = reshape([1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0], 2, 2, 2)
    @test NCPLS.normalize_predictors(Xtensor, tensor_mf) ≈ expected
end

@testset "coef, fitted, residuals, and predict for matrix fits" begin
    rng = Random.MersenneTwister(1)
    model = NCPLS.NCPLSModel(
        ncomponents = 2,
        center_X = true,
        scale_X = true,
        center_Yprim = true,
    )

    X = randn(rng, 6, 4)
    Y = randn(rng, 6, 2)

    mf = NCPLS.fit_ncpls_core(model, X, Y; obs_weights = nothing)

    @test size(NCPLS.coef(mf)) == (4, 2)
    @test size(NCPLS.coef(mf, 1)) == (4, 2)
    @test NCPLS.coef(mf) ≈ selectdim(mf.B, ndims(mf.B) - 1, 2)
    @test NCPLS.coef(mf, 1) ≈ selectdim(mf.B, ndims(mf.B) - 1, 1)

    Yhat1 = NCPLS.fitted(mf, 1)
    Yhat2 = NCPLS.fitted(mf)
    Fres1 = NCPLS.residuals(mf, 1)
    Fres2 = NCPLS.residuals(mf)

    @test size(Yhat1) == size(Y)
    @test size(Yhat2) == size(Y)
    @test size(Fres1) == size(Y)
    @test size(Fres2) == size(Y)
    @test Yhat1 + Fres1 ≈ Y
    @test Yhat2 + Fres2 ≈ Y

    Ypred1 = NCPLS.predict(mf, X, 1)
    Ypred2 = NCPLS.predict(mf, X)

    @test size(Ypred1) == (size(X, 1), 1, size(Y, 2))
    @test size(Ypred2) == (size(X, 1), 2, size(Y, 2))
    @test dropdims(Ypred1; dims = 2) ≈ Yhat1
    @test @view(Ypred2[:, 2, :]) ≈ Yhat2

    Xnorm = (NCPLS.float64(X) .- reshape(mf.X_mean, 1, :)) ./ reshape(mf.X_std, 1, :)
    expected1 = Xnorm * NCPLS.coef(mf, 1)
    expected1 = expected1 .+ reshape(mf.Yprim_mean, 1, :)
    @test dropdims(Ypred1; dims = 2) ≈ expected1

    Tproj = NCPLS.project(mf, X)
    @test size(Tproj) == (size(X, 1), NCPLS.ncomponents(mf))
    @test Tproj ≈ reshape(Xnorm, size(X, 1), :) * reshape(mf.R, :, NCPLS.ncomponents(mf))

    @test_throws DimensionMismatch NCPLS.predict(mf, rand(4, 3))
    @test_throws DimensionMismatch NCPLS.predict(mf, X, 3)
    @test_throws DimensionMismatch NCPLS.project(mf, rand(4, 3))
    @test_throws ArgumentError NCPLS.project(mf, rand(4))
end

@testset "coef and predict for tensor fits" begin
    rng = Random.MersenneTwister(2)
    model = NCPLS.NCPLSModel(
        ncomponents = 2,
        center_X = true,
        scale_X = true,
        center_Yprim = true,
    )

    X = randn(rng, 6, 3, 2)
    Y = randn(rng, 6, 2)

    mf = NCPLS.fit_ncpls_core(model, X, Y; obs_weights = nothing)

    @test size(NCPLS.coef(mf)) == (3, 2, 2)
    @test size(NCPLS.coef(mf, 1)) == (3, 2, 2)

    Ypred = NCPLS.predict(mf, X)
    @test size(Ypred) == (size(X, 1), 2, size(Y, 2))
    @test @view(Ypred[:, 2, :]) ≈ NCPLS.fitted(mf)

    Xnorm = (NCPLS.float64(X) .- reshape(mf.X_mean, 1, size(mf.X_mean)...)) ./
        reshape(mf.X_std, 1, size(mf.X_std)...)
    coef1 = NCPLS.coef(mf, 1)
    expected1 = reshape(reshape(Xnorm, size(X, 1), :) * reshape(coef1, :, size(Y, 2)),
        size(X, 1), size(Y, 2))
    expected1 = expected1 .+ reshape(mf.Yprim_mean, 1, :)

    @test dropdims(NCPLS.predict(mf, X, 1); dims = 2) ≈ expected1
    @test_throws DimensionMismatch NCPLS.predict(mf, rand(4, 3, 3))

    Tproj = NCPLS.project(mf, X)
    @test size(Tproj) == (size(X, 1), NCPLS.ncomponents(mf))
    @test Tproj ≈ reshape(Xnorm, size(X, 1), :) * reshape(mf.R, :, NCPLS.ncomponents(mf))
    @test_throws DimensionMismatch NCPLS.project(mf, rand(4, 3, 3))
    @test_throws ArgumentError NCPLS.project(mf, rand(4))
end
