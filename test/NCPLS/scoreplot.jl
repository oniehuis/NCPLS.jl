function mock_scoreplot_fit(;
    ncomponents=2,
    samplelabels=["s1", "s2", "s3", "s4"],
    sampleclasses=["A", "A", "B", "B"],
)
    model = NCPLS.NCPLSModel(ncomponents = ncomponents, multilinear = false)
    B = zeros(2, 3, ncomponents, 2)
    T = reshape(collect(1.0:(4 * ncomponents)), 4, ncomponents)

    NCPLS.NCPLSFit(
        model,
        B,
        zeros(2, 3, ncomponents),
        T,
        zeros(2, 3, ncomponents),
        zeros(2, ncomponents),
        zeros(2, 3, ncomponents),
        nothing,
        zeros(2, ncomponents),
        zeros(2, 3, ncomponents, 2),
        zeros(ncomponents),
        zeros(4, 2),
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        zeros(2, 3),
        ones(2, 3),
        zeros(2);
        samplelabels = samplelabels,
        sampleclasses = sampleclasses,
    )
end

@testset "scoreplot fit wrapper validates metadata and component count" begin
    mf = mock_scoreplot_fit()

    @test_throws ErrorException NCPLS.scoreplot(
        ["s1", "s2"],
        ["A", "B"],
        [1.0 2.0; 3.0 4.0];
        backend = :unknown,
    )
    @test_throws ArgumentError NCPLS.scoreplot(
        mock_scoreplot_fit(sampleclasses = nothing);
        backend = :plotly,
    )
    @test_throws ArgumentError NCPLS.scoreplot(
        mock_scoreplot_fit(ncomponents = 1);
        backend = :plotly,
    )
    @test size(NCPLS.xscores(mf, 1:2)) == (4, 2)
end

@testset "scoreplot plotly smoke test" begin
    if !isnothing(Base.find_package("PlotlyJS"))
        @eval using PlotlyJS

        mf = mock_scoreplot_fit()
        plt = NCPLS.scoreplot(mf)

        @test nameof(typeof(plt)) == :SyncPlot
        @test length(plt.plot.data) == 2
        @test all(length(trace[:x]) == 2 for trace in plt.plot.data)
    end
end
