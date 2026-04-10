function plotly_axis_values(values::AbstractVector)
    collect(values)
end

function default_plotly_colorscale(landscape::AbstractMatrix{<:Real})
    any(<(0), landscape) && any(>(0), landscape) ? "RdBu" : "Viridis"
end

function robust_symmetric_limit(
    landscape::AbstractMatrix{<:Real};
    sample_size::Integer=50_000,
    iqr_multiplier::Real=8.0,
    clip_quantile::Real=0.995,
)
    sample_size > 0 || throw(ArgumentError("`sample_size` must be positive"))
    0 < clip_quantile ≤ 1 || throw(ArgumentError("`clip_quantile` must be in (0, 1]"))

    abs_vals = abs.(vec(Float64.(landscape)))
    isempty(abs_vals) && return 1.0

    sample_step = max(1, cld(length(abs_vals), sample_size))
    abs_sample = @view abs_vals[1:sample_step:end]
    q1, q3, qhi = quantile(abs_sample, [0.25, 0.75, clip_quantile])
    iqr = q3 - q1
    lim = max(q3 + iqr_multiplier * iqr, qhi)

    if !(isfinite(lim) && lim > 0)
        lim = maximum(abs_vals)
    end

    lim > 0 ? lim : 1.0
end

function NCPLS.landscapeplot_plotly(
    mf::NCPLS.NCPLSFit;
    lv::Union{Symbol, Integer}=:final,
    response::Union{Nothing, Integer}=nothing,
    response_contrast::Union{Nothing, Tuple{<:Integer, <:Integer}}=nothing,
    sample_size::Integer=50_000,
    iqr_multiplier::Real=8.0,
    clip_quantile::Real=0.995,
    colorscale::AbstractString="RdBu",
    colorbar_title::AbstractString="Coefficient",
    hovertemplate::Union{Nothing, AbstractString}=nothing,
    title::Union{Nothing, AbstractString}=nothing,
    layout::Union{Nothing, PlotlyJS.Layout}=nothing,
    plot_kwargs=(;),
)
    landscape = NCPLS.coefficientlandscape(
        mf;
        lv = lv,
        response = response,
        response_contrast = response_contrast,
    )
    axis_x, axis_y = NCPLS.landscape_predictoraxes(mf, landscape)
    xvals = plotly_axis_values(axis_x.values)
    yvals = plotly_axis_values(axis_y.values)

    lim = robust_symmetric_limit(
        landscape;
        sample_size = sample_size,
        iqr_multiplier = iqr_multiplier,
        clip_quantile = clip_quantile,
    )

    title = isnothing(title) ?
        NCPLS.default_landscape_title(mf, lv, response, response_contrast) : title
    hovertemplate = isnothing(hovertemplate) ?
        string(
            axis_x.name, ": %{x}<br>",
            axis_y.name, ": %{y}<br>",
            colorbar_title, ": %{z:.6g}<extra></extra>",
        ) : hovertemplate

    trace = PlotlyJS.heatmap(
        x = xvals,
        y = yvals,
        z = permutedims(landscape),
        colorscale = colorscale,
        zmin = -lim,
        zmax = lim,
        zmid = 0,
        colorbar = PlotlyJS.attr(title = colorbar_title),
        hovertemplate = hovertemplate,
    )

    layout = isnothing(layout) ? PlotlyJS.Layout(
        title = title,
        xaxis = PlotlyJS.attr(title = NCPLS.landscape_axis_label(axis_x)),
        yaxis = PlotlyJS.attr(title = NCPLS.landscape_axis_label(axis_y)),
    ) : layout

    to_nt(x::NamedTuple) = x
    to_nt(x::AbstractDict) = (; (Symbol(k) => v for (k, v) in x)...)
    to_nt(::Nothing) = (;)

    PlotlyJS.plot(trace, layout; to_nt(plot_kwargs)...)
end

function NCPLS.weightlandscapeplot_plotly(
    mf::NCPLS.NCPLSFit;
    lv::Union{Symbol, Integer}=1,
    combine::Symbol=:sum,
    sample_size::Integer=50_000,
    iqr_multiplier::Real=8.0,
    clip_quantile::Real=0.995,
    colorscale::Union{Nothing, AbstractString}=nothing,
    colorbar_title::AbstractString="Weight",
    hovertemplate::Union{Nothing, AbstractString}=nothing,
    title::Union{Nothing, AbstractString}=nothing,
    layout::Union{Nothing, PlotlyJS.Layout}=nothing,
    plot_kwargs=(;),
)
    landscape = NCPLS.weightlandscape(mf; lv = lv, combine = combine)
    axis_x, axis_y = NCPLS.landscape_predictoraxes(mf, landscape)
    xvals = plotly_axis_values(axis_x.values)
    yvals = plotly_axis_values(axis_y.values)

    colorscale = isnothing(colorscale) ? default_plotly_colorscale(landscape) : colorscale
    lim = robust_symmetric_limit(
        landscape;
        sample_size = sample_size,
        iqr_multiplier = iqr_multiplier,
        clip_quantile = clip_quantile,
    )

    signed = any(<(0), landscape) && any(>(0), landscape)
    title = isnothing(title) ? NCPLS.default_weight_title(mf, lv, combine) : title
    hovertemplate = isnothing(hovertemplate) ?
        string(
            axis_x.name, ": %{x}<br>",
            axis_y.name, ": %{y}<br>",
            colorbar_title, ": %{z:.6g}<extra></extra>",
        ) : hovertemplate

    heatmap_kwargs = if signed
        (zmin = -lim, zmax = lim, zmid = 0)
    else
        (zmin = 0.0, zmax = lim)
    end

    trace = PlotlyJS.heatmap(;
        x = xvals,
        y = yvals,
        z = permutedims(landscape),
        colorscale = colorscale,
        colorbar = PlotlyJS.attr(title = colorbar_title),
        hovertemplate = hovertemplate,
        heatmap_kwargs...,
    )

    layout = isnothing(layout) ? PlotlyJS.Layout(
        title = title,
        xaxis = PlotlyJS.attr(title = NCPLS.landscape_axis_label(axis_x)),
        yaxis = PlotlyJS.attr(title = NCPLS.landscape_axis_label(axis_y)),
    ) : layout

    to_nt(x::NamedTuple) = x
    to_nt(x::AbstractDict) = (; (Symbol(k) => v for (k, v) in x)...)
    to_nt(::Nothing) = (;)

    PlotlyJS.plot(trace, layout; to_nt(plot_kwargs)...)
end

function NCPLS.weightprofilesplot_plotly(
    mf::NCPLS.NCPLSFit;
    lv::Union{Symbol, Integer}=1,
    combine::Symbol=:sum,
    title::Union{Nothing, AbstractString}=nothing,
    layout::Union{Nothing, PlotlyJS.Layout}=nothing,
    line_kwargs=(;),
    zero_line::Bool=true,
    plot_kwargs=(;),
)
    profiles = NCPLS.weightprofiles(mf; lv = lv, combine = combine)
    axes = NCPLS.predictoraxes(mf)
    if isempty(axes)
        axes = [NCPLS.PredictorAxis("Axis $j", collect(1:length(profiles[j]))) for j in eachindex(profiles)]
    end
    length(axes) == length(profiles) || throw(DimensionMismatch(
        "Stored predictor axis metadata do not match the number of weight profiles"))
    subplot_titles = reshape(
        Union{Missing, String}[NCPLS.landscape_axis_label(ax) for ax in axes],
        length(profiles),
        1,
    )

    fig = PlotlyJS.make_subplots(
        rows = length(profiles),
        cols = 1,
        shared_xaxes = false,
        vertical_spacing = 0.08,
        subplot_titles = subplot_titles,
    )

    to_nt(x::NamedTuple) = x
    to_nt(x::AbstractDict) = (; (Symbol(k) => v for (k, v) in x)...)
    to_nt(::Nothing) = (;)
    line_nt = to_nt(line_kwargs)

    for (j, profile) in enumerate(profiles)
        ax = axes[j]
        xvals = plotly_axis_values(ax.values)

        if zero_line
            PlotlyJS.add_trace!(
                fig,
                PlotlyJS.scatter(
                    x = xvals,
                    y = zeros(length(profile)),
                    mode = "lines",
                    line = PlotlyJS.attr(color = "rgba(120,120,120,0.7)", dash = "dash"),
                    hoverinfo = "skip",
                    showlegend = false,
                );
                row = j,
                col = 1,
            )
        end

        PlotlyJS.add_trace!(
            fig,
            PlotlyJS.scatter(
                x = xvals,
                y = profile,
                mode = "lines",
                line = PlotlyJS.attr(; line_nt...),
                hovertemplate = string(
                    ax.name, ": %{x}<br>",
                    "Weight: %{y:.6g}<extra></extra>",
                ),
                showlegend = false,
            );
            row = j,
            col = 1,
        )
    end

    title = isnothing(title) ? NCPLS.default_weightprofiles_title(lv, combine) : title
    if isnothing(layout)
        PlotlyJS.relayout!(
            fig,
            title_text = title,
            height = max(320, 260 * length(profiles)),
        )
    else
        PlotlyJS.relayout!(fig, layout)
    end

    PlotlyJS.plot(fig.plot; to_nt(plot_kwargs)...)
end
