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

function mock_response_landscape_fit(nresponses::Integer; responselabels=String[])
    model = NCPLS.NCPLSModel(ncomponents = 2, multilinear = false)

    B = Array{Float64}(undef, 2, 3, 2, nresponses)
    base = reshape(collect(1.0:6.0), 2, 3)
    for a in 1:2, r in 1:nresponses
        B[:, :, a, r] = base .* (10a + r)
    end

    NCPLS.NCPLSFit(
        model,
        B,
        zeros(2, 3, 2),
        zeros(4, 2),
        zeros(2, 3, 2),
        zeros(nresponses, 2),
        zeros(2, 3, 2),
        nothing,
        zeros(nresponses, 2),
        zeros(2, 3, nresponses, 2),
        [0.0, 0.0],
        zeros(4, nresponses),
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        zeros(2, 3),
        ones(2, 3),
        zeros(nresponses);
        responselabels = responselabels,
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
    @test_throws ArgumentError NCPLS.coefficientlandscape(mf; lv = :all)
    @test_throws DimensionMismatch NCPLS.coefficientlandscape(mf; lv = 3)
end

@testset "coefficientlandscape handles response-count defaults" begin
    one_response = mock_response_landscape_fit(1; responselabels = ["intensity"])
    one_expected = @views selectdim(coef(one_response), 3, 1)
    @test NCPLS.coefficientlandscape(one_response) == one_expected

    three_response = mock_response_landscape_fit(3)
    response2_expected = @views selectdim(coef(three_response), 3, 2)
    contrast_expected = @views selectdim(coef(three_response), 3, 3) .-
        selectdim(coef(three_response), 3, 1)

    @test NCPLS.coefficientlandscape(three_response; response = 2) == response2_expected
    @test NCPLS.coefficientlandscape(three_response; response_contrast = (3, 1)) ==
        contrast_expected
    @test_throws ArgumentError NCPLS.coefficientlandscape(three_response)
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

    combined_mean = NCPLS.weightprofiles(mf; lv = :combined, combine = :mean)
    @test combined_mean[1] == [5.5, 11.0]
    @test combined_mean[2] == [16.5, 22.0, 27.5]

    combined_sumabs = NCPLS.weightprofiles(mf; lv = :combined, combine = :sumabs)
    @test combined_sumabs[1] == [11.0, 22.0]
    @test combined_sumabs[2] == [33.0, 44.0, 55.0]

    combined_meanabs = NCPLS.weightprofiles(mf; lv = :all, combine = :meanabs)
    @test combined_meanabs[1] == [5.5, 11.0]
    @test combined_meanabs[2] == [16.5, 22.0, 27.5]

    @test_throws ArgumentError NCPLS.weightprofiles(mock_landscape_fit(); lv = 1)
    @test_throws ArgumentError NCPLS.weightprofiles(mf; lv = :final)
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

@testset "landscape default titles reflect component and response selections" begin
    mf = mock_landscape_fit(responselabels = ["species_A", "species_B"])
    one_labeled = mock_response_landscape_fit(1; responselabels = ["intensity"])
    one_unlabeled = mock_response_landscape_fit(1)
    three_unlabeled = mock_response_landscape_fit(3)

    @test NCPLS.landscape_response_label(mf, nothing, (2, 1)) == "species_B – species_A"
    @test NCPLS.landscape_response_label(mf, 1, nothing) == "species_A"
    @test NCPLS.landscape_response_label(one_labeled, nothing, nothing) == "intensity"
    @test NCPLS.landscape_response_label(one_unlabeled, nothing, nothing) == ""
    @test NCPLS.landscape_response_label(mf, nothing, nothing) == "species_B – species_A"
    @test NCPLS.landscape_response_label(three_unlabeled, nothing, nothing) == ""
    @test NCPLS.landscape_response_label(three_unlabeled, 3, nothing) == "response_3"

    @test NCPLS.default_landscape_title(mf, 1, nothing, nothing) ==
        "Coefficient Landscape LV1 (species_B – species_A)"
    @test NCPLS.default_landscape_title(mf, :final, 1, nothing) ==
        "Coefficient Landscape (species_A)"
    @test NCPLS.default_landscape_title(one_unlabeled, :final, nothing, nothing) ==
        "Coefficient Landscape"
    @test NCPLS.default_weight_title(mf, 2, :sum) == "NCPLS LV2 Weight Landscape"
    @test NCPLS.default_weight_title(mf, :combined, :sumabs) ==
        "NCPLS Combined Weight Landscape (sumabs)"
    @test NCPLS.default_weightprofiles_title(1, :sum) == "NCPLS LV1 Weight Profiles"
    @test NCPLS.default_weightprofiles_title(:combined, :meanabs) ==
        "NCPLS Combined Weight Profiles (meanabs)"
end

@testset "landscape plot dispatch guards and delegates through plotly refs" begin
    mf = mock_multilinear_landscape_fit()

    @test_throws ErrorException NCPLS._require_landscape_extension(:MissingExtension, "Missing")
    @test_throws ErrorException NCPLS.landscapeplot_plotly(mf)
    @test_throws ErrorException NCPLS.weightlandscapeplot_plotly(mf)
    @test_throws ErrorException NCPLS.weightprofilesplot_plotly(mf)
    @test_throws ErrorException NCPLS.landscapeplot(mf; backend = :unknown)
    @test_throws ErrorException NCPLS.coefflandscapeplot(mf; backend = :unknown)
    @test_throws ErrorException NCPLS.weightlandscapeplot(mf; backend = :unknown)
    @test_throws ErrorException NCPLS.weightprofilesplot(mf; backend = :unknown)

    old_require = NCPLS._require_landscape_extension_ref[]
    old_landscape = NCPLS._landscapeplot_plotly_ref[]
    old_weightlandscape = NCPLS._weightlandscapeplot_plotly_ref[]
    old_weightprofiles = NCPLS._weightprofilesplot_plotly_ref[]

    NCPLS._require_landscape_extension_ref[] =
        (extsym::Symbol, pkg::AbstractString) -> nothing
    NCPLS._landscapeplot_plotly_ref[] = (mf; kwargs...) -> (:landscape, mf, kwargs)
    NCPLS._weightlandscapeplot_plotly_ref[] =
        (mf; kwargs...) -> (:weightlandscape, mf, kwargs)
    NCPLS._weightprofilesplot_plotly_ref[] =
        (mf; kwargs...) -> (:weightprofiles, mf, kwargs)

    try
        @test_throws ErrorException NCPLS.landscapeplot_plotly(mf)
        @test_throws ErrorException NCPLS.weightlandscapeplot_plotly(mf)
        @test_throws ErrorException NCPLS.weightprofilesplot_plotly(mf)

        landscape_res = NCPLS.landscapeplot(mf; backend = :plotly, lv = 1)
        @test landscape_res[1] == :landscape
        @test landscape_res[2] === mf

        alias_res = NCPLS.coefflandscapeplot(mf; backend = :plotly, lv = 2)
        @test alias_res[1] == :landscape
        @test alias_res[2] === mf

        weightlandscape_res = NCPLS.weightlandscapeplot(mf; backend = :plotly, lv = 1)
        @test weightlandscape_res[1] == :weightlandscape
        @test weightlandscape_res[2] === mf

        weightprofiles_res = NCPLS.weightprofilesplot(mf; backend = :plotly, lv = 1)
        @test weightprofiles_res[1] == :weightprofiles
        @test weightprofiles_res[2] === mf
    finally
        NCPLS._require_landscape_extension_ref[] = old_require
        NCPLS._landscapeplot_plotly_ref[] = old_landscape
        NCPLS._weightlandscapeplot_plotly_ref[] = old_weightlandscape
        NCPLS._weightprofilesplot_plotly_ref[] = old_weightprofiles
    end
end

@testset "landscape extension guard accepts loaded extensions" begin
    if !isnothing(Base.find_package("Makie"))
        try
            @eval using Makie
            @test NCPLS._require_landscape_extension(:MakieExtension, "Makie") === nothing
        catch
        end
    end
end

const _LANDSCAPE_PLOTLYJS_AVAILABLE = Ref{Union{Nothing, Bool}}(nothing)

function landscape_plotlyjs_available()
    cached = _LANDSCAPE_PLOTLYJS_AVAILABLE[]
    cached isa Bool && return cached

    available = if isnothing(Base.find_package("PlotlyJS"))
        false
    else
        try
            @eval using PlotlyJS
            true
        catch
            false
        end
    end

    _LANDSCAPE_PLOTLYJS_AVAILABLE[] = available
    available
end

@testset "weightprofilesplot plotly smoke test" begin
    if landscape_plotlyjs_available()
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
    if landscape_plotlyjs_available()
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

module FakeLandscapePlotlyExtension

using Statistics
import NCPLS

module PlotlyJS

abstract type AbstractTrace end

struct Trace <: AbstractTrace
    kind::Symbol
    kwargs::NamedTuple
end

Base.getindex(trace::Trace, key::Symbol) = getproperty(trace, key)
function Base.getproperty(trace::Trace, key::Symbol)
    key in (:kind, :kwargs) && return getfield(trace, key)
    get(trace.kwargs, key, nothing)
end

struct Layout
    kwargs::NamedTuple
end
Layout(; kwargs...) = Layout((; kwargs...))

mutable struct Plot
    data::Vector{AbstractTrace}
    layout
    kwargs::NamedTuple
end

mutable struct Subplot
    plot::Plot
    kwargs::NamedTuple
end

attr(; kwargs...) = (; kwargs...)
scatter(; kwargs...) = Trace(:scatter, (; kwargs...))
heatmap(; kwargs...) = Trace(:heatmap, (; kwargs...))
plot(trace::AbstractTrace, layout; kwargs...) =
    Plot(AbstractTrace[trace], layout, (; kwargs...))
plot(plot::Plot; kwargs...) = Plot(plot.data, plot.layout, (; kwargs...))
function make_subplots(; kwargs...)
    Subplot(Plot(AbstractTrace[], Layout(), (;)), (; kwargs...))
end
function add_trace!(fig::Subplot, trace::AbstractTrace; row, col)
    push!(fig.plot.data, trace)
    trace
end
function relayout!(fig::Subplot; kwargs...)
    fig.plot.layout = (; kwargs...)
    fig
end
function relayout!(fig::Subplot, layout::Layout)
    fig.plot.layout = layout
    fig
end

end

include(joinpath(@__DIR__, "..", "..", "ext", "plotly_extensions", "landscapeplot.jl"))

end

function mock_mismatched_weightprofile_fit()
    model = NCPLS.NCPLSModel(ncomponents = 1, multilinear = true)
    axes = (
        NCPLS.PredictorAxis("Axis 1", [1.0, 2.0]),
        NCPLS.PredictorAxis("Axis 2", [1.0, 2.0, 3.0]),
    )

    NCPLS.NCPLSFit(
        model,
        zeros(2, 3, 1, 1),
        zeros(2, 3, 1),
        zeros(4, 1),
        zeros(2, 3, 1),
        zeros(1, 1),
        zeros(2, 3, 1),
        [ones(2, 1)],
        zeros(1, 1),
        zeros(2, 3, 1, 1),
        [0.0],
        zeros(4, 1),
        [0.0],
        [:svd],
        [1.0],
        [0],
        [true],
        zeros(2, 3),
        ones(2, 3),
        zeros(1);
        predictoraxes = axes,
    )
end

@testset "landscape plotly extension logic with fake backend" begin
    FakePlotlyJS = FakeLandscapePlotlyExtension.PlotlyJS

    @test FakeLandscapePlotlyExtension.plotly_axis_values([1, 2, 3]) == [1, 2, 3]
    @test FakeLandscapePlotlyExtension.default_plotly_colorscale([-1.0 2.0]) == "RdBu"
    @test FakeLandscapePlotlyExtension.default_plotly_colorscale([1.0 2.0]) == "Viridis"
    @test FakeLandscapePlotlyExtension.robust_symmetric_limit(zeros(0, 0)) == 1.0
    @test FakeLandscapePlotlyExtension.robust_symmetric_limit(zeros(2, 2)) == 1.0
    @test FakeLandscapePlotlyExtension.robust_symmetric_limit([1.0 2.0; 3.0 4.0]) > 0
    @test_throws ArgumentError FakeLandscapePlotlyExtension.robust_symmetric_limit(
        [1.0 2.0];
        sample_size = 0,
    )
    @test_throws ArgumentError FakeLandscapePlotlyExtension.robust_symmetric_limit(
        [1.0 2.0];
        clip_quantile = 0.0,
    )

    axes = (
        NCPLS.PredictorAxis("RT", [5.0, 5.5]; unit = "min"),
        NCPLS.PredictorAxis("m/z", [91, 105, 121]; unit = "Da"),
    )
    mf = mock_landscape_fit(
        predictoraxes = axes,
        responselabels = ["species_A", "species_B"],
    )

    coeff_plot = NCPLS.landscapeplot_plotly(mf; plot_kwargs = Dict("config" => :cfg))
    @test length(coeff_plot.data) == 1
    @test coeff_plot.data[1].kind == :heatmap
    @test coeff_plot.data[1][:x] == [5.0, 5.5]
    @test coeff_plot.layout.kwargs.title == "Coefficient Landscape (species_B – species_A)"
    @test coeff_plot.kwargs == (config = :cfg,)

    custom_layout = FakePlotlyJS.Layout(title = "Custom")
    custom_coeff_plot = NCPLS.landscapeplot_plotly(
        mf;
        lv = 1,
        response = 1,
        hovertemplate = "H",
        title = "T",
        layout = custom_layout,
        plot_kwargs = nothing,
    )
    @test custom_coeff_plot.layout === custom_layout
    @test custom_coeff_plot.data[1][:hovertemplate] == "H"
    @test custom_coeff_plot.kwargs == (;)

    positive_weight_plot = NCPLS.weightlandscapeplot_plotly(
        mf;
        lv = :combined,
        colorscale = "Cividis",
        hovertemplate = "W",
        title = "Weights",
        layout = custom_layout,
        plot_kwargs = Dict("config" => :weights),
    )
    @test positive_weight_plot.data[1][:colorscale] == "Cividis"
    @test positive_weight_plot.data[1][:zmin] == 0.0
    @test positive_weight_plot.layout === custom_layout
    @test positive_weight_plot.kwargs == (config = :weights,)

    positive_weight_no_kwargs = NCPLS.weightlandscapeplot_plotly(
        mf;
        lv = :combined,
        layout = custom_layout,
        plot_kwargs = nothing,
    )
    @test positive_weight_no_kwargs.kwargs == (;)

    signed_mf = mock_landscape_fit(predictoraxes = axes)
    signed_mf.W[1, 1, 1] = -1.0
    signed_mf.W[1, 2, 1] = 2.0
    signed_weight_plot = NCPLS.weightlandscapeplot_plotly(signed_mf; lv = 1)
    @test signed_weight_plot.data[1][:colorscale] == "RdBu"
    @test signed_weight_plot.data[1][:zmin] < 0
    @test signed_weight_plot.data[1][:zmid] == 0

    profile_mf = mock_multilinear_landscape_fit()
    profile_plot = NCPLS.weightprofilesplot_plotly(
        profile_mf;
        line_kwargs = Dict("color" => "red"),
        plot_kwargs = Dict("config" => :profiles),
    )
    @test length(profile_plot.data) == 4
    @test profile_plot.kwargs == (config = :profiles,)
    @test profile_plot.layout.title_text == "NCPLS LV1 Weight Profiles"

    axes_profile_mf = mock_multilinear_landscape_fit(predictoraxes = axes)
    profile_custom_plot = NCPLS.weightprofilesplot_plotly(
        axes_profile_mf;
        lv = :combined,
        zero_line = false,
        layout = custom_layout,
        line_kwargs = nothing,
        plot_kwargs = nothing,
    )
    @test length(profile_custom_plot.data) == 2
    @test profile_custom_plot.layout === custom_layout
    @test profile_custom_plot.kwargs == (;)

    @test length(NCPLS._landscapeplot_plotly_ref[](mf).data) == 1
    @test length(NCPLS._weightlandscapeplot_plotly_ref[](mf).data) == 1
    @test length(NCPLS._weightprofilesplot_plotly_ref[](profile_mf).data) == 4

    @test_throws DimensionMismatch NCPLS.weightprofilesplot_plotly(
        mock_mismatched_weightprofile_fit(),
    )
end
