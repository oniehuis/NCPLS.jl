@testset "candidate_loading_weights returns matrix-shaped weights for matrix X" begin
    X = Float64[
        1 2
        3 4
        5 6
    ]
    Y = Float64[
        1 0
        0 1
        1 1
    ]

    W0 = NCPLS.candidate_loading_weights(X, Y, nothing)
    expected = X' * Y

    @test size(W0) == (2, 2)
    @test W0 ≈ expected
end

@testset "candidate_loading_weights respects observation weights for matrix X" begin
    X = Float64[
        1 2 3
        4 5 6
        7 8 9
        2 4 6
    ]
    Y = Float64[
        1 0
        0 1
        1 1
        0 2
    ]
    weights = [1.0, 2.0, 1.0, 0.5]

    W0 = NCPLS.candidate_loading_weights(X, Y, weights)
    expected = (X .* reshape(weights, :, 1))' * Y

    @test size(W0) == (3, 2)
    @test W0 ≈ expected
end

@testset "candidate_loading_weights returns tensor-shaped weights for tensor X" begin
    X = reshape(collect(1.0:24.0), 4, 3, 2)
    Y = Float64[
        1 0
        0 1
        1 1
        0 0
    ]

    W0 = NCPLS.candidate_loading_weights(X, Y, nothing)
    Xmat = reshape(X, size(X, 1), :)
    expected_mat = Xmat' * Y
    expected = reshape(expected_mat, 3, 2, 2)

    @test size(W0) == (3, 2, 2)
    @test W0 ≈ expected
    @test reshape(W0, :, size(Y, 2)) ≈ expected_mat
end

@testset "candidate_loading_weights respects observation weights for tensor X" begin
    X = Array{Float64}(undef, 4, 2, 2)
    X[:, 1, 1] = [1.0, 2.0, 3.0, 4.0]
    X[:, 2, 1] = [2.0, 3.0, 5.0, 7.0]
    X[:, 1, 2] = [0.0, 1.0, 1.0, 2.0]
    X[:, 2, 2] = [4.0, 6.0, 8.0, 10.0]
    Y = Float64[
        1 0 1
        0 1 0
        1 1 1
        0 0 1
    ]
    weights = [1.0, 2.0, 1.0, 0.5]

    W0 = NCPLS.candidate_loading_weights(X, Y, weights)
    Xmat = reshape(X, size(X, 1), :)
    expected_mat = (Xmat .* reshape(weights, :, 1))' * Y
    expected = reshape(expected_mat, 2, 2, 3)

    @test size(W0) == (2, 2, 3)
    @test W0 ≈ expected
    @test reshape(W0, :, size(Y, 2)) ≈ expected_mat
end

@testset "candidate_loading_weights aligns with preprocessed combined responses" begin
    model = NCPLS.NCPLSModel(
        center_X = true,
        scale_X = false,
        center_Yprim = true,
        scale_Yprim = false,
        center_Yadd = true,
        scale_Yadd = false,
    )
    X = reshape(collect(1.0:24.0), 4, 3, 2)
    Yprim = Float64[
        1 0
        0 1
        1 0
        0 1
    ]
    Yadd = reshape([2.0, 4.0, 6.0, 8.0], :, 1)
    weights = [1.0, 2.0, 1.0, 2.0]

    d = NCPLS.preprocess(model, X, Yprim, Yadd, weights)
    Ycomb = hcat(d.Yprim, d.Yadd)
    W0 = NCPLS.candidate_loading_weights(d.X, Ycomb, weights)
    expected = reshape(
        (reshape(d.X, size(d.X, 1), :) .* reshape(weights, :, 1))' * Ycomb,
        size(d.X)[2:end]...,
        size(Ycomb, 2),
    )

    @test size(W0) == (3, 2, 3)
    @test W0 ≈ expected
end

@testset "candidate_scores returns matrix-shaped scores for matrix X" begin
    X = Float64[
        1 2
        3 4
        5 6
    ]
    Y = Float64[
        1 0
        0 1
        1 1
    ]

    W0 = NCPLS.candidate_loading_weights(X, Y, nothing)
    Z0 = NCPLS.candidate_scores(X, W0)
    expected = X * W0

    @test size(Z0) == size(Y)
    @test Z0 ≈ expected
end

@testset "candidate_scores returns matrix-shaped scores for tensor X" begin
    X = reshape(collect(1.0:24.0), 4, 3, 2)
    Y = Float64[
        1 0
        0 1
        1 1
        0 0
    ]

    W0 = NCPLS.candidate_loading_weights(X, Y, nothing)
    Z0 = NCPLS.candidate_scores(X, W0)
    expected = reshape(X, size(X, 1), :) * reshape(W0, :, size(Y, 2))

    @test size(Z0) == size(Y)
    @test Z0 ≈ expected
end

@testset "candidate_scores aligns with preprocessed combined responses" begin
    model = NCPLS.NCPLSModel(
        center_X = true,
        scale_X = false,
        center_Yprim = true,
        scale_Yprim = false,
        center_Yadd = true,
        scale_Yadd = false,
    )
    X = reshape(collect(1.0:24.0), 4, 3, 2)
    Yprim = Float64[
        1 0
        0 1
        1 0
        0 1
    ]
    Yadd = reshape([2.0, 4.0, 6.0, 8.0], :, 1)
    weights = [1.0, 2.0, 1.0, 2.0]

    d = NCPLS.preprocess(model, X, Yprim, Yadd, weights)
    Ycomb = hcat(d.Yprim, d.Yadd)
    W0 = NCPLS.candidate_loading_weights(d.X, Ycomb, weights)
    Z0 = NCPLS.candidate_scores(d.X, W0)
    expected = reshape(d.X, size(d.X, 1), :) * reshape(W0, :, size(Ycomb, 2))

    @test size(Z0) == size(Ycomb)
    @test Z0 ≈ expected
end

@testset "candidate_scores validates W0 shape against X" begin
    X_matrix = rand(4, 3)
    bad_W0_matrix = rand(4, 2)
    err_matrix = try
        NCPLS.candidate_scores(X_matrix, bad_W0_matrix)
        nothing
    catch err
        err
    end
    @test err_matrix isa DimensionMismatch
    @test occursin(
        "Predictor dimensions of W₀ must match the non-sample dimensions of X",
        sprint(showerror, err_matrix),
    )

    X_tensor = rand(4, 2, 2)
    bad_W0_tensor = rand(2, 2)
    err_tensor = try
        NCPLS.candidate_scores(X_tensor, bad_W0_tensor)
        nothing
    catch err
        err
    end
    @test err_tensor isa DimensionMismatch
    @test occursin(
        "W₀ must have the same number of dimensions as X",
        sprint(showerror, err_tensor),
    )
end

@testset "orthogonalize_on_accumulated_scores handles empty previous scores" begin
    Z0 = Float64[
        1 2
        3 4
        5 6
    ]
    t = [1.0, 3.0, 5.0]

    @test NCPLS.orthogonalize_on_accumulated_scores(Z0, zeros(3, 0)) ≈ Z0
    @test NCPLS.orthogonalize_on_accumulated_scores(t, zeros(3, 0)) ≈ t
end

@testset "orthogonalize_on_accumulated_scores applies the manuscript formula for matrices" begin
    t1 = [1.0, 0.0, 0.0, 0.0]
    t2 = [0.0, 1.0, 0.0, 0.0]
    T_A = hcat(t1, t2)

    Z0 = Float64[
        3 1
        2 4
        3 5
        2 8
    ]

    Z = NCPLS.orthogonalize_on_accumulated_scores(Z0, T_A)
    expected = Z0 - T_A * (T_A' * Z0)

    @test size(Z) == size(Z0)
    @test Z ≈ expected
    @test T_A' * Z ≈ zeros(size(T_A, 2), size(Z0, 2)) atol = 1e-12
end

@testset "orthogonalize_on_accumulated_scores applies the manuscript formula for vectors" begin
    T_A = Float64[
        1 0
        0 1
        0 0
        0 0
    ]
    t = [3.0, 4.0, 5.0, 6.0]

    t_orth = NCPLS.orthogonalize_on_accumulated_scores(t, T_A)
    expected = t - T_A * (T_A' * t)

    @test length(t_orth) == length(t)
    @test t_orth ≈ expected
    @test T_A' * t_orth ≈ zeros(size(T_A, 2)) atol = 1e-12
end

@testset "orthogonalize_on_accumulated_scores validates row alignment" begin
    Z0 = rand(4, 2)
    T_A = rand(3, 1)

    err = try
        NCPLS.orthogonalize_on_accumulated_scores(Z0, T_A)
        nothing
    catch err
        err
    end

    @test err isa DimensionMismatch
    @test occursin(
        "T_A and X must have the same number of rows",
        sprint(showerror, err),
    )
end

@testset "loading_weights contracts response dimension for matrix W0" begin
    W0 = Float64[
        1 2
        3 4
        5 6
    ]
    c = [2.0, -1.0]

    w = NCPLS.loading_weights(W0, c)
    expected = W0 * c

    @test size(w) == (3,)
    @test w ≈ expected
end

@testset "loading_weights contracts response dimension for tensor W0" begin
    W0 = Array{Float64}(undef, 2, 2, 3)
    W0[:, :, 1] = [1.0 2.0; 3.0 4.0]
    W0[:, :, 2] = [0.0 1.0; 1.0 0.0]
    W0[:, :, 3] = [2.0 0.0; 0.0 2.0]
    c = [1.0, -2.0, 0.5]

    w = NCPLS.loading_weights(W0, c)
    expected = reshape(reshape(W0, :, length(c)) * c, 2, 2)

    @test size(w) == (2, 2)
    @test w ≈ expected
end

@testset "loading_weights validates c length and allows zero output norm" begin
    W0 = Float64[
        1 2
        3 4
    ]

    err_dim = try
        NCPLS.loading_weights(W0, [1.0, 2.0, 3.0])
        nothing
    catch err
        err
    end
    @test err_dim isa DimensionMismatch
    @test occursin("Length of c must match the last dimension of W₀", sprint(showerror, err_dim))

    W0_zero = Float64[
        1 2
        2 4
    ]
    w_zero = NCPLS.loading_weights(W0_zero, [2.0, -1.0])
    @test w_zero == zeros(2)
end

@testset "score_vector forms score vectors for matrix and tensor inputs" begin
    X_matrix = Float64[
        1 2
        3 4
        5 6
    ]
    W⁰_matrix = [0.6, 0.8]
    t_matrix = NCPLS.score_vector(X_matrix, W⁰_matrix)
    @test t_matrix ≈ X_matrix * W⁰_matrix

    X_tensor = reshape(collect(1.0:24.0), 4, 3, 2)
    W⁰_tensor = reshape(collect(1.0:6.0), 3, 2)
    t_tensor = NCPLS.score_vector(X_tensor, W⁰_tensor)
    expected_tensor = reshape(X_tensor, size(X_tensor, 1), :) * vec(W⁰_tensor)

    @test length(t_tensor) == size(X_tensor, 1)
    @test t_tensor ≈ expected_tensor
end

@testset "score_vector validates loading-weight dimensions" begin
    X = reshape(collect(1.0:24.0), 4, 3, 2)
    bad_W⁰ = [1.0, 2.0, 3.0]

    err = try
        NCPLS.score_vector(X, bad_W⁰)
        nothing
    catch err
        err
    end

    @test err isa DimensionMismatch
    @test occursin(
        "Dimensions of Wᵒ must match the non-sample dimensions of X",
        sprint(showerror, err),
    )
end

@testset "normalize_vector scales vectors to unit norm" begin
    t = [3.0, 4.0]
    t_normed = NCPLS.normalize_vector(t)

    @test t_normed ≈ [0.6, 0.8]
    @test sum(t_normed .^ 2) ≈ 1.0 atol = 1e-12
end

@testset "normalize_vector rejects zero vectors" begin
    err = try
        NCPLS.normalize_vector(zeros(3))
        nothing
    catch err
        err
    end

    @test err isa ArgumentError
    @test occursin("Score vector has zero norm", sprint(showerror, err))
end

@testset "loading_tensor forms predictor-shaped loadings for matrix and tensor inputs" begin
    X_matrix = Float64[
        1 2
        3 4
        5 6
    ]
    t_matrix = [0.2, 0.3, 0.5]
    P_matrix = NCPLS.loading_tensor(X_matrix, t_matrix)
    expected_matrix = X_matrix' * t_matrix

    @test size(P_matrix) == (2,)
    @test P_matrix ≈ expected_matrix

    X_tensor = reshape(collect(1.0:24.0), 4, 3, 2)
    t_tensor = [0.1, 0.2, 0.3, 0.4]
    P_tensor = NCPLS.loading_tensor(X_tensor, t_tensor)
    expected_tensor = reshape(reshape(X_tensor, size(X_tensor, 1), :)' * t_tensor, 3, 2)

    @test size(P_tensor) == (3, 2)
    @test P_tensor ≈ expected_tensor
end

@testset "loading_tensor validates score length against sample dimension" begin
    X = reshape(collect(1.0:24.0), 4, 3, 2)
    err = try
        NCPLS.loading_tensor(X, [1.0, 2.0, 3.0])
        nothing
    catch err
        err
    end

    @test err isa DimensionMismatch
    @test occursin("Length of t must match size(X, 1)", sprint(showerror, err))
end

@testset "response_loading_vector forms response loadings in primary-response space" begin
    Yprim = Float64[
        1 2
        3 4
        5 6
    ]
    t = [0.2, 0.3, 0.5]

    q = NCPLS.response_loading_vector(Yprim, t)
    expected = Yprim' * t

    @test size(q) == (2,)
    @test q ≈ expected
end

@testset "response_loading_vector validates score length against responses" begin
    Yprim = rand(4, 2)

    err = try
        NCPLS.response_loading_vector(Yprim, [1.0, 2.0, 3.0])
        nothing
    catch err
        err
    end

    @test err isa DimensionMismatch
    @test occursin("Length of t must match size(Yprim, 1)", sprint(showerror, err))
end

@testset "deflate_responses! subtracts the rank-1 response fit in place" begin
    Yprim = Float64[
        1 2
        3 4
        5 6
    ]
    Ycopy = copy(Yprim)
    t = [0.2, 0.3, 0.5]
    q = [7.0, 11.0]

    returned = NCPLS.deflate_responses!(Yprim, t, q)
    expected = Ycopy .- t * q'

    @test returned === Yprim
    @test Yprim ≈ expected
end

@testset "deflate_responses! validates t and q dimensions" begin
    Yprim = rand(4, 2)

    err_t = try
        NCPLS.deflate_responses!(copy(Yprim), [1.0, 2.0, 3.0], [1.0, 2.0])
        nothing
    catch err
        err
    end
    @test err_t isa DimensionMismatch
    @test occursin("Length of t must match size(Yprim, 1)", sprint(showerror, err_t))

    err_q = try
        NCPLS.deflate_responses!(copy(Yprim), [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0])
        nothing
    catch err
        err
    end
    @test err_q isa DimensionMismatch
    @test occursin("Length of q must match size(Yprim, 2)", sprint(showerror, err_q))
end

@testset "score_projection_tensors matches the unfolded projection formula for matrices" begin
    W_A = Float64[
        1 2
        3 4
        5 6
    ]
    P_A = Float64[
        2 1
        1 1
        0 1
    ]

    R = NCPLS.score_projection_tensors(W_A, P_A)
    expected = W_A * inv(P_A' * W_A)

    @test size(R) == size(W_A)
    @test R ≈ expected
end

@testset "score_projection_tensors matches the unfolded projection formula for tensors" begin
    W_A = Array{Float64}(undef, 2, 2, 2)
    W_A[:, :, 1] = [1.0 2.0; 3.0 4.0]
    W_A[:, :, 2] = [0.0 1.0; 1.0 0.0]

    P_A = Array{Float64}(undef, 2, 2, 2)
    P_A[:, :, 1] = [2.0 0.0; 1.0 1.0]
    P_A[:, :, 2] = [1.0 1.0; 0.0 2.0]

    R = NCPLS.score_projection_tensors(W_A, P_A)

    W_Am = reshape(W_A, :, size(W_A, 3))
    P_Am = reshape(P_A, :, size(P_A, 3))
    expected = reshape(W_Am * inv(P_Am' * W_Am), size(W_A)...)

    @test size(R) == size(W_A)
    @test R ≈ expected
end

@testset "score_projection_tensors validates matching tensor dimensions" begin
    W_A = rand(2, 2, 2)
    P_A = rand(2, 2, 3)

    err = try
        NCPLS.score_projection_tensors(W_A, P_A)
        nothing
    catch err
        err
    end

    @test err isa DimensionMismatch
    @test occursin("W_A and P_A must have the same dimensions", sprint(showerror, err))
end

@testset "regression_coefficients matches cumulative component contributions for matrix predictors" begin
    R = Float64[
        1 2
        3 4
        5 6
    ]
    Q_A = Float64[
        10 20
        30 40
    ]

    B = NCPLS.regression_coefficients(R, Q_A)

    @test size(B) == (3, 2, 2)
    @test B[:, 1, :] ≈ R[:, 1] * Q_A[:, 1]'
    @test B[:, 2, :] ≈ R[:, 1] * Q_A[:, 1]' .+ R[:, 2] * Q_A[:, 2]'
end

@testset "regression_coefficients matches cumulative component contributions for tensor predictors" begin
    R = Array{Float64}(undef, 2, 2, 2)
    R[:, :, 1] = [1.0 2.0; 3.0 4.0]
    R[:, :, 2] = [0.0 1.0; 1.0 0.0]
    Q_A = Float64[
        2 5
        3 7
    ]

    B = NCPLS.regression_coefficients(R, Q_A)

    @test size(B) == (2, 2, 2, 2)
    @test B[:, :, 1, :] ≈ reshape(vec(R[:, :, 1]) * Q_A[:, 1]', 2, 2, 2)
    expected2 = reshape(vec(R[:, :, 1]) * Q_A[:, 1]' .+ vec(R[:, :, 2]) * Q_A[:, 2]', 2, 2, 2)
    @test B[:, :, 2, :] ≈ expected2
end

@testset "regression_coefficients validates component dimension alignment" begin
    R = rand(2, 2, 2)
    Q_A = rand(3, 3)

    err = try
        NCPLS.regression_coefficients(R, Q_A)
        nothing
    catch err
        err
    end

    @test err isa DimensionMismatch
    @test occursin(
        "The number of columns in Q_A must match the component dimension of R",
        sprint(showerror, err),
    )
end

@testset "outer_tensor forms the expected rank-1 tensor" begin
    factors = [
        [1.0, 2.0],
        [3.0, 4.0, 5.0],
        [2.0, -1.0],
    ]

    T = NCPLS.outer_tensor(factors)

    @test size(T) == (2, 3, 2)
    @test T[1, 1, 1] == 1.0 * 3.0 * 2.0
    @test T[2, 3, 2] == 2.0 * 5.0 * -1.0
    @test T[2, 2, 1] == 2.0 * 4.0 * 2.0
end

@testset "contract_except contracts all but one mode" begin
    X = Array{Float64}(undef, 2, 3, 2)
    X[:, :, 1] = [1.0 2.0 3.0; 4.0 5.0 6.0]
    X[:, :, 2] = [0.0 1.0 0.0; 1.0 0.0 1.0]
    factors = [
        [1.0, 2.0],
        [0.5, -1.0, 2.0],
        [3.0, -2.0],
    ]

    out_mode_2 = NCPLS.contract_except(X, factors, 2)
    expected_mode_2 = [
        sum(X[i, 1, k] * factors[1][i] * factors[3][k] for i in 1:2, k in 1:2),
        sum(X[i, 2, k] * factors[1][i] * factors[3][k] for i in 1:2, k in 1:2),
        sum(X[i, 3, k] * factors[1][i] * factors[3][k] for i in 1:2, k in 1:2),
    ]

    @test out_mode_2 ≈ expected_mode_2

    err = try
        NCPLS.contract_except(X, factors[1:2], 1)
        nothing
    catch err
        err
    end
    @test err isa AssertionError
end

@testset "parafac_rank1 recovers an exact rank-1 tensor" begin
    factors = [
        [1.0, 2.0],
        [3.0, -1.0, 2.0],
        [2.0, 1.0],
    ]
    λ = 4.5
    X = λ .* NCPLS.outer_tensor(factors)

    fit = NCPLS.parafac_rank1(X; maxiter = 200, tol = 1e-12, init = :hosvd)

    @test fit.converged === true
    @test size(fit.Xhat) == size(X)
    @test fit.Xhat ≈ X atol = 1e-8
    @test fit.relerr ≤ 1e-8
    @test fit.fit ≈ 1.0 atol = 1e-8
    @test length(fit.factors) == 3
end

@testset "parafac_rank1 rejects zero tensors" begin
    err = try
        NCPLS.parafac_rank1(zeros(2, 2, 2))
        nothing
    catch err
        err
    end
    @test err isa ErrorException
    @test occursin("Zero tensor not supported.", sprint(showerror, err))
end

@testset "multilinear_weights handles vector, matrix, and tensor inputs" begin
    v = [3.0, 4.0]
    mv = NCPLS.multilinear_weights(v)
    @test mv.method == :vector
    @test mv.W_rank1 ≈ v / 5.0
    @test mv.relerr == 0.0

    u = [1.0, 2.0]
    s = 3.0
    r = [2.0, -1.0, 1.0]
    M = s .* (u * r')
    mm = NCPLS.multilinear_weights(M)
    @test mm.method == :svd
    @test size(mm.W_rank1) == size(M)
    @test mm.W_rank1 ≈ M atol = 1e-8
    @test mm.relerr ≤ 1e-8
    @test length(mm.factors) == 2

    factors = [
        [1.0, -2.0],
        [0.5, 1.5, -1.0],
        [2.0, 1.0],
    ]
    T = 2.5 .* NCPLS.outer_tensor(factors)
    mt = NCPLS.multilinear_weights(T; maxiter = 200, tol = 1e-12)
    @test mt.method == :parafac
    @test size(mt.W_rank1) == size(T)
    @test mt.W_rank1 ≈ T atol = 1e-8
    @test mt.relerr ≤ 1e-8
    @test length(mt.factors) == 3
end
