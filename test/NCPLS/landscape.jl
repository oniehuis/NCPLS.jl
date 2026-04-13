import Statistics: mean

function mock_landscape_fit(;
    predictoraxes=NCPLS.PredictorAxis[],
    responselabels=String[],
)
    model = NCPLS.NCPLSModel(ncomponents = 2, multilinear = false)

    comp1_r1 = [1.0 2.0 3.0; 4.0 5.0 6.0]
    comp1_r2 = [10.0 20.0 30.0; 40.0 50.0 60.0]
    comp2_r1 = [0.5 1.0 1.5; 2.0 2.5 3.0]
    comp2_r2 = [5.0 10.0 15.0; 20.0 25.0 30.0]

    B = Array{Float64}(undef, 2, 3, 2, 2)
    B[:, :, 1, 1] = comp1_r1
    B[:, :, 1, 2] = comp1_r2
    B[:, :, 2, 1] = comp1_r1 .+ comp2_r1
    B[:, :, 2, 2] = comp1_r2 .+ comp2_r2

    NCPLS.NCPLSFit(
        model,
        B,
        zeros(2, 3, 2),
        zeros(4, 2),
        zeros(2, 3, 2),
        zeros(2, 2),
        zeros(2, 3, 2),
        nothing,
        zeros(2, 2),
        zeros(2, 3, 2, 2),
        [0.0, 0.0],
        zeros(4, 2),
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        zeros(2, 3),
        ones(2, 3),
        zeros(2);
        responselabels = responselabels,
        predictoraxes = predictoraxes,
    )
end

function mock_multilinear_landscape_fit(; predictoraxes=NCPLS.PredictorAxis[])
    model = NCPLS.NCPLSModel(ncomponents = 2, multilinear = true)

    B = zeros(2, 3, 2, 2)
    W = Array{Float64}(undef, 2, 3, 2)
    W[:, :, 1] = [1.0 2.0 3.0; 4.0 5.0 6.0]
    W[:, :, 2] = [10.0 20.0 30.0; 40.0 50.0 60.0]
    W_modes = [
        [1.0 10.0; 2.0 20.0],
        [3.0 30.0; 4.0 40.0; 5.0 50.0],
    ]

    NCPLS.NCPLSFit(
        model,
        B,
        zeros(2, 3, 2),
        zeros(4, 2),
        zeros(2, 3, 2),
        zeros(2, 2),
        W,
        W_modes,
        zeros(2, 2),
        zeros(2, 3, 2, 2),
        [0.0, 0.0],
        zeros(4, 2),
        [0.0, 0.0],
        [:svd, :svd],
        [1.0, 1.0],
        [0, 0],
        [true, true],
        zeros(2, 3),
        ones(2, 3),
        zeros(2);
        predictoraxes = predictoraxes,
    )
end

@testset "coefficientlandscape extracts final and LV-specific surfaces" begin
    mf = mock_landscape_fit(responselabels = ["species_A", "species_B"])

    final_expected = @views selectdim(coef(mf), 3, 2) .- selectdim(coef(mf), 3, 1)
    lv1_expected = @views selectdim(coef(mf, 1), 3, 2) .- selectdim(coef(mf, 1), 3, 1)
    lv2_tensor = coef(mf, 2) .- coef(mf, 1)
    lv2_expected = @views selectdim(lv2_tensor, 3, 2) .- selectdim(lv2_tensor, 3, 1)

    @test NCPLS.coefficientlandscape(mf) == final_expected
    @test NCPLS.coefflandscape(mf) == final_expected
    @test NCPLS.coefficientlandscape(mf; lv = :final) == final_expected
    @test NCPLS.coefficientlandscape(mf; lv = 1) == lv1_expected
    @test NCPLS.coefficientlandscape(mf; lv = 2) == lv2_expected

    @test NCPLS.coefficientlandscape(mf; response = 1) == @views selectdim(coef(mf), 3, 1)
    @test NCPLS.coefficientlandscape(mf; response_contrast = (2, 1)) == final_expected

    @test_throws ArgumentError NCPLS.coefficientlandscape(
        mf;
        response = 1,
        response_contrast = (2, 1),
    )
    @test_throws DimensionMismatch NCPLS.coefficientlandscape(mf; lv = 3)
end

@testset "weightlandscape extracts LV-specific and combined weight surfaces" begin
    mf = mock_landscape_fit()

    @test NCPLS.weightlandscape(mf; lv = 1) == @views selectdim(mf.W, 3, 1)
    @test NCPLS.weightlandscape(mf; lv = 2) == @views selectdim(mf.W, 3, 2)
    @test NCPLS.weightlandscape(mf; lv = :combined, combine = :sum) ==
        dropdims(sum(mf.W; dims = 3); dims = 3)
    @test NCPLS.weightlandscape(mf; lv = :all, combine = :mean) ==
        dropdims(mean(mf.W; dims = 3); dims = 3)
    @test NCPLS.weightlandscape(mf; lv = :combined, combine = :sumabs) ==
        dropdims(sum(abs.(mf.W); dims = 3); dims = 3)
    @test NCPLS.weightlandscape(mf; lv = :combined, combine = :meanabs) ==
        dropdims(mean(abs.(mf.W); dims = 3); dims = 3)

    @test_throws ArgumentError NCPLS.weightlandscape(mf; lv = :final)
    @test_throws ArgumentError NCPLS.weightlandscape(mf; lv = :combined, combine = :bad)
    @test_throws DimensionMismatch NCPLS.weightlandscape(mf; lv = 3)
end

@testset "weightprofiles extract LV-specific and combined per-axis vectors" begin
    mf = mock_multilinear_landscape_fit()

    profiles1 = NCPLS.weightprofiles(mf; lv = 1)
    @test profiles1[1] == [1.0, 2.0]
    @test profiles1[2] == [3.0, 4.0, 5.0]

    profiles2 = NCPLS.weightprofiles(mf; lv = 2)
    @test profiles2[1] == [10.0, 20.0]
    @test profiles2[2] == [30.0, 40.0, 50.0]

    combined_sum = NCPLS.weightprofiles(mf; lv = :combined, combine = :sum)
    @test combined_sum[1] == [11.0, 22.0]
    @test combined_sum[2] == [33.0, 44.0, 55.0]

    combined_meanabs = NCPLS.weightprofiles(mf; lv = :all, combine = :meanabs)
    @test combined_meanabs[1] == [5.5, 11.0]
    @test combined_meanabs[2] == [16.5, 22.0, 27.5]

    @test_throws ArgumentError NCPLS.weightprofiles(mock_landscape_fit(); lv = 1)
    @test_throws ArgumentError NCPLS.weightprofiles(mf; lv = :combined, combine = :bad)
    @test_throws DimensionMismatch NCPLS.weightprofiles(mf; lv = 3)
end

@testset "coefficientlandscape and landscape axes honor predictor axis metadata" begin
    axes = (
        NCPLS.PredictorAxis("RT", [5.0, 5.5]; unit = "min"),
        NCPLS.PredictorAxis("m/z", [91, 105, 121]; unit = "Da"),
    )
    mf = mock_landscape_fit(predictoraxes = axes)
    landscape = NCPLS.coefficientlandscape(mf)
    ax1, ax2 = NCPLS.landscape_predictoraxes(mf, landscape)

    @test ax1.name == "RT"
    @test ax1.values == [5.0, 5.5]
    @test ax1.unit == "min"
    @test ax2.name == "m/z"
    @test ax2.values == [91, 105, 121]
    @test ax2.unit == "Da"
    @test NCPLS.landscape_axis_label(ax1) == "RT (min)"
    @test NCPLS.landscape_axis_label(ax2) == "m/z (Da)"
end

@testset "landscape axes fall back to index positions when metadata are absent" begin
    mf = mock_landscape_fit()
    landscape = NCPLS.coefficientlandscape(mf)
    ax1, ax2 = NCPLS.landscape_predictoraxes(mf, landscape)

    @test ax1.name == "Axis 1"
    @test ax1.values == [1, 2]
    @test isnothing(ax1.unit)
    @test ax2.name == "Axis 2"
    @test ax2.values == [1, 2, 3]
    @test isnothing(ax2.unit)
end

@testset "weightprofilesplot plotly smoke test" begin
    if !isnothing(Base.find_package("PlotlyJS"))
        @eval using PlotlyJS

        axes = (
            NCPLS.PredictorAxis("RT", [5.0, 5.5]; unit = "min"),
            NCPLS.PredictorAxis("m/z", [91, 105, 121]; unit = "Da"),
        )
        mf = mock_multilinear_landscape_fit(predictoraxes = axes)
        plt = NCPLS.weightprofilesplot(mf; lv = 1)

        @test nameof(typeof(plt)) == :SyncPlot
        @test length(plt.plot.data) == 4
    end
end

@testset "coefflandscapeplot aliases landscapeplot" begin
    if !isnothing(Base.find_package("PlotlyJS"))
        @eval using PlotlyJS

        axes = (
            NCPLS.PredictorAxis("RT", [5.0, 5.5]; unit = "min"),
            NCPLS.PredictorAxis("m/z", [91, 105, 121]; unit = "Da"),
        )
        mf = mock_landscape_fit(
            predictoraxes = axes,
            responselabels = ["species_A", "species_B"],
        )
        plt = NCPLS.coefflandscapeplot(mf)

        @test nameof(typeof(plt)) == :SyncPlot
        @test length(plt.plot.data) == 1
    end
end
