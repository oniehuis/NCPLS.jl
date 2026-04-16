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

@testset "scoreplot dispatch guards" begin
    samples = ["s1", "s2"]
    groups = ["g1", "g2"]
    scores = [1.0 2.0; 3.0 4.0]

    @test_throws ErrorException NCPLS._require_extension(:MissingExtension, "Missing")
    @test_throws ErrorException NCPLS.scoreplot(samples, groups, scores; backend = :unknown)
end

@testset "scoreplot dispatch success" begin
    old_require = NCPLS._require_extension_ref[]
    old_plotly = NCPLS._scoreplot_plotly_ref[]
    old_makie = NCPLS._scoreplot_makie_ref[]

    NCPLS._require_extension_ref[] = (extsym::Symbol, pkg::AbstractString) -> nothing
    NCPLS._scoreplot_plotly_ref[] = (samples, groups, scores; kwargs...) ->
        (:plotly, samples, groups, scores, kwargs)
    NCPLS._scoreplot_makie_ref[] = (samples, groups, scores; kwargs...) ->
        (:makie, samples, groups, scores, kwargs)

    try
        samples = [SubString("s1", 1:2), SubString("s2", 1:2)]
        groups = ["g1", "g2"]
        scores = [1.0 2.0; 3.0 4.0]

        res = NCPLS.scoreplot(samples, groups, scores; backend = :plotly)
        @test res[1] == :plotly
        @test res[2] == samples
        @test res[3] == groups
        @test res[4] == scores

        res = NCPLS.scoreplot(samples, groups, scores; backend = :makie)
        @test res[1] == :makie
        @test res[2] == samples
        @test res[3] == groups
        @test res[4] == scores

        mf = mock_scoreplot_fit(
            samplelabels = [SubString("a", 1:1), SubString("b", 1:1), SubString("c", 1:1), SubString("d", 1:1)],
            sampleclasses = NCPLS.categorical(["A", "A", "B", "B"]),
        )

        res = NCPLS.scoreplot(mf; backend = :plotly)
        @test res[1] == :plotly
        @test res[2] == NCPLS.samplelabels(mf)
        @test res[3] == NCPLS.sampleclasses(mf)
        @test res[4] == mf.T[:, 1:2]

        mf_no_groups = mock_scoreplot_fit(sampleclasses = nothing)
        @test_throws ArgumentError NCPLS.scoreplot(mf_no_groups; backend = :plotly)
        @test_throws ErrorException NCPLS.scoreplot(mf; backend = :unknown)
    finally
        NCPLS._require_extension_ref[] = old_require
        NCPLS._scoreplot_plotly_ref[] = old_plotly
        NCPLS._scoreplot_makie_ref[] = old_makie
    end
end

@testset "scoreplot(mf) rejects one-component fits" begin
    @test_throws ArgumentError NCPLS.scoreplot(
        mock_scoreplot_fit(ncomponents = 1);
        backend = :makie,
        show_legend = false,
        show_inspector = false,
    )
end

@testset "scoreplot plotly smoke test" begin
    if !isnothing(Base.find_package("PlotlyJS"))
        @eval using PlotlyJS

        mf = mock_scoreplot_fit(sampleclasses = NCPLS.categorical(["A", "A", "B", "B"]))
        plt = NCPLS.scoreplot(mf; backend = :plotly)

        @test nameof(typeof(plt)) == :SyncPlot
        @test length(plt.plot.data) == 2
        @test all(length(trace[:x]) == 2 for trace in plt.plot.data)

        samples = ["s1", "s2", "s3"]
        groups = [:a, :b, :a]
        scores = [1.0 2.0; 3.0 4.0; 5.0 6.0]

        default_trace = Dict("marker" => PlotlyJS.attr(color = "red"))
        group_trace = Dict("b" => Dict("name" => "Bee", "hovertemplate" => "B", "text" => ["tb"]))
        default_marker = Dict("size" => 9)
        group_marker = Dict("a" => Dict("color" => "blue"))

        res = NCPLS.scoreplot_plotly(
            samples,
            groups,
            scores;
            group_order = [:b, :missing, :a],
            default_trace = default_trace,
            group_trace = group_trace,
            default_marker = default_marker,
            group_marker = group_marker,
            show_legend = false,
            plot_kwargs = Dict("config" => PlotlyJS.PlotConfig(displayModeBar = false)),
        )
        @test res isa Union{PlotlyJS.Plot, PlotlyJS.SyncPlot}
        plot = res isa PlotlyJS.SyncPlot ? res.plot : res
        @test length(plot.data) == 2

        groups_cat = NCPLS.categorical(["x", "y", "x"]; levels = ["x", "y", "z"])
        layout = PlotlyJS.Layout(title = "Custom")
        res2 = NCPLS.scoreplot_plotly(
            samples,
            groups_cat,
            scores;
            default_trace = nothing,
            default_marker = nothing,
            layout = layout,
            group_marker = Dict("x" => Dict("color" => "green")),
        )
        @test res2 isa Union{PlotlyJS.Plot, PlotlyJS.SyncPlot}
        plot2 = res2 isa PlotlyJS.SyncPlot ? res2.plot : res2
        @test plot2.layout.title == "Custom"
    end
end

@testset "scoreplot makie smoke test" begin
    if !isnothing(Base.find_package("CairoMakie"))
        @eval using CairoMakie

        MakieExt = Base.get_extension(NCPLS, :MakieExtension)
        @test MakieExt !== nothing
        @test NCPLS._require_extension(:MakieExtension, "Makie") === nothing

        function set_backend(mod)
            MakieExt._current_backend_ref[] = () -> mod
            nothing
        end

        set_backend(missing)

        samples = ["s1", "s2", "s3"]
        groups_cat = NCPLS.categorical(["a", "b", "a"]; levels = ["a", "b", "c"])
        scores = [1.0 2.0; 3.0 4.0; 5.0 6.0]

        fig1 = Figure(size = (200, 200))
        ax1 = Axis(fig1[1, 1])
        res1 = NCPLS.scoreplot_makie(
            samples,
            groups_cat,
            scores;
            figure = fig1,
            axis = ax1,
            axis_kwargs = nothing,
            legend_kwargs = nothing,
            show_legend = false,
            show_inspector = true,
            default_marker = Dict("markersize" => 5),
            group_trace = Dict("a" => (; alpha = 0.8)),
        )
        @test res1 === fig1

        groups = [:a, :b, :a]
        fig2 = Figure(size = (200, 200))
        ax2 = Axis(fig2[1, 1])
        res2 = NCPLS.scoreplot_makie(
            samples,
            groups,
            scores;
            figure = fig2,
            axis = ax2,
            group_order = [:b, :a, :missing],
            show_legend = false,
            show_inspector = false,
            default_scatter = (; color = :red),
            default_trace = Dict(:alpha => 0.5),
            group_marker = Dict(
                "a" => (; label = "custom", color = :blue),
                "b" => (; color = :green),
            ),
        )
        @test res2 === fig2

        res3 = NCPLS.scoreplot(
            samples,
            groups,
            scores;
            backend = :makie,
            figure_kwargs = (; size = (300, 200)),
            axis_kwargs = (; xlabel = "LV1"),
            show_legend = false,
            show_inspector = false,
        )
        @test res3 isa Figure

        @eval Main module GLMakie end
        set_backend(Main.GLMakie)

        res4 = NCPLS.scoreplot_makie(
            ["s1", "s2"],
            [:a, :b],
            [1.0 2.0; 3.0 4.0];
            show_legend = true,
            show_inspector = true,
            legend_kwargs = Dict(:position => :rt),
            inspector_kwargs = Dict(:enabled => true),
        )
        @test res4 isa Figure

        set_backend(missing)
    end
end
