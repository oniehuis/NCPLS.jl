@testset "float64 handles arrays and preserves Float64 storage" begin
    X_float = rand(3, 2)
    @test NCPLS.float64(X_float) === X_float

    X_int = reshape(1:12, 3, 2, 2)
    X_conv = NCPLS.float64(X_int)
    @test X_conv isa Array{Float64, 3}
    @test X_conv == Float64.(X_int)

    weights_int = [1, 2, 3]
    weights_conv = NCPLS.float64(weights_int)
    @test weights_conv isa Vector{Float64}
    @test weights_conv == [1.0, 2.0, 3.0]
end

@testset "validate_obs_weights enforces length and value constraints" begin
    X = rand(4, 3, 2)

    @test NCPLS.validate_obs_weights(X, nothing) === nothing
    @test NCPLS.validate_obs_weights(X, [1.0, 2.0, 1.0, 0.5]) === nothing

    @test_throws DimensionMismatch NCPLS.validate_obs_weights(X, [1.0, 2.0, 1.0])
    @test_throws ArgumentError NCPLS.validate_obs_weights(X, [1.0, -1.0, 1.0, 1.0])
    @test_throws ArgumentError NCPLS.validate_obs_weights(X, [1.0, NaN, 1.0, 1.0])
    @test_throws ArgumentError NCPLS.validate_obs_weights(X, zeros(4))
end

@testset "centerscale handles unweighted matrices and rejects vectors" begin
    X = Float64[
        1 2
        3 4
        5 6
    ]

    X_cs, μ, σ = NCPLS.centerscale(X, true, true, nothing)

    expected_μ = [3.0, 4.0]
    expected_σ = [sqrt(8 / 3), sqrt(8 / 3)]

    @test size(X_cs) == size(X)
    @test μ ≈ expected_μ
    @test σ ≈ expected_σ
    @test vec(sum(X_cs, dims = 1)) ≈ [0.0, 0.0] atol=1e-12
    @test vec(sqrt.(sum(X_cs .^ 2, dims = 1) / size(X, 1))) ≈ [1.0, 1.0]

    X_degenerate = Float64[
        1 5
        1 6
        1 7
    ]
    X_deg_cs, μ_deg, σ_deg = NCPLS.centerscale(X_degenerate, true, true, nothing)
    @test μ_deg ≈ [1.0, 6.0]
    @test σ_deg[1] == 1.0
    @test all(iszero, X_deg_cs[:, 1])

    @test_throws ArgumentError NCPLS.centerscale([1.0, 2.0, 3.0], true, true, nothing)
end

@testset "centerscale handles weighted tensors along sample dimension" begin
    X = Array{Float64}(undef, 4, 2, 2)
    X[:, 1, 1] = [1.0, 2.0, 3.0, 4.0]
    X[:, 2, 1] = [2.0, 4.0, 6.0, 8.0]
    X[:, 1, 2] = [1.0, 3.0, 5.0, 7.0]
    X[:, 2, 2] = [4.0, 6.0, 8.0, 10.0]
    weights = [1.0, 2.0, 1.0, 2.0]

    X_cs, μ, σ = NCPLS.centerscale(X, true, true, weights)

    wsum = sum(weights)
    w = reshape(weights, :, 1, 1)
    expected_μ = dropdims(sum(X .* w, dims = 1) / wsum; dims = 1)
    centered = X .- reshape(expected_μ, 1, size(X, 2), size(X, 3))
    expected_σ = dropdims(sqrt.(sum(w .* centered .^ 2, dims = 1) / wsum); dims = 1)

    @test size(X_cs) == size(X)
    @test size(μ) == (2, 2)
    @test size(σ) == (2, 2)
    @test μ ≈ expected_μ
    @test σ ≈ expected_σ
    @test sum(X_cs .* w, dims = 1) ./ wsum ≈ zeros(1, 2, 2) atol=1e-12
    @test sqrt.(sum(w .* X_cs .^ 2, dims = 1) / wsum) ≈ ones(1, 2, 2)
end

@testset "preprocess returns aligned arrays for matrices and tensors" begin
    model = NCPLS.NCPLSModel(
        ncomponents = 2,
        center_X = true,
        scale_X = true,
        center_Yprim = true,
    )

    X_matrix = Float64[
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

    d_matrix = NCPLS.preprocess(model, X_matrix, Y, Yadd, weights)
    Ycomb_matrix = hcat(d_matrix.Yprim, d_matrix.Yadd)
    @test size(d_matrix.X) == size(X_matrix)
    @test size(d_matrix.Yprim) == size(Y)
    @test size(d_matrix.Yadd) == size(Yadd)
    @test size(Ycomb_matrix) == (4, 3)
    @test size(d_matrix.X_mean) == (2,)
    @test size(d_matrix.X_std) == (2,)
    @test size(d_matrix.Yprim_mean) == (2,)
    @test Ycomb_matrix == hcat(d_matrix.Yprim, d_matrix.Yadd)
    @test d_matrix.Yadd == Float64.(Yadd)

    w_matrix = reshape(weights, :, 1)
    @test sum(d_matrix.X .* w_matrix, dims = 1) ./ sum(weights) ≈ zeros(1, 2) atol=1e-12
    @test sum(d_matrix.Yprim .* w_matrix, dims = 1) ./ sum(weights) ≈ zeros(1, 2) atol=1e-12

    X_tensor = reshape(collect(1.0:24.0), 4, 3, 2)
    d_tensor = NCPLS.preprocess(model, X_tensor, Y, Yadd, nothing)
    Ycomb_tensor = hcat(d_tensor.Yprim, d_tensor.Yadd)
    @test size(d_tensor.X) == size(X_tensor)
    @test size(d_tensor.X_mean) == (3, 2)
    @test size(d_tensor.X_std) == (3, 2)
    @test size(d_tensor.Yprim_mean) == (2,)
    @test size(Ycomb_tensor) == (4, 3)
    @test Ycomb_tensor == hcat(d_tensor.Yprim, d_tensor.Yadd)
    @test sum(d_tensor.X, dims = 1) ≈ zeros(1, 3, 2) atol=1e-12
    @test vec(sum(d_tensor.Yprim, dims = 1)) ≈ [0.0, 0.0] atol=1e-12
    @test d_tensor.Yadd == Float64.(Yadd)

    d_no_yadd = NCPLS.preprocess(model, X_tensor, Y, nothing, nothing)
    @test size(d_no_yadd.Yprim) == size(Y)
    @test d_no_yadd.Yadd === nothing

    err_x_yprim = try
        NCPLS.preprocess(model, X_tensor, Y[1:3, :], Yadd, nothing)
        nothing
    catch err
        err
    end
    @test err_x_yprim isa DimensionMismatch
    @test occursin("Number of rows in X and Yprim must be equal", sprint(showerror, err_x_yprim))

    err_yprim_yadd = try
        NCPLS.preprocess(model, X_tensor, Y, Yadd[1:3, :], nothing)
        nothing
    catch err
        err
    end
    @test err_yprim_yadd isa DimensionMismatch
    @test occursin("Yprim and Yadd must have the same number of rows", sprint(showerror, err_yprim_yadd))

    @test_throws ArgumentError NCPLS.preprocess(
        model,
        [1.0, 2.0, 3.0, 4.0],
        reshape([1.0, 2.0, 3.0, 4.0], :, 1),
        reshape([5.0, 6.0, 7.0, 8.0], :, 1),
        nothing,
    )
end
