@testset "cca_coeffs_and_corr returns stable dimensions and weighted rho" begin
    X = [
        1.0 2.0 0.5
        2.0 1.0 0.3
        3.0 4.0 1.1
        4.0 3.0 0.9
        5.0 6.0 1.8
    ]
    Y = [
        1.0 0.2
        1.5 0.4
        2.9 0.8
        3.1 0.7
        4.8 1.2
    ]
    weights = [1.0, 0.5, 1.0, 0.75, 1.25]

    a, b, rho = NCPLS.cca_coeffs_and_corr(X, Y, nothing)
    aw, bw, rhow = NCPLS.cca_coeffs_and_corr(X, Y, weights)

    @test size(a) == (size(X, 2), min(size(X, 2), size(Y, 2)))
    @test size(b) == (size(Y, 2), min(size(X, 2), size(Y, 2)))
    @test 0.0 ≤ rho ≤ 1.0
    @test size(aw) == size(a)
    @test size(bw) == size(b)
    @test 0.0 ≤ rhow ≤ 1.0
    @test isfinite(rho)
    @test isfinite(rhow)
end
