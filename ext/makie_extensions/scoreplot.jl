function NCPLS.scoreplot_makie(
    samples::AbstractVector{<:AbstractString},
    groups,
    scores::AbstractMatrix{<:Real};
    group_order::Union{Nothing, AbstractVector} = nothing,
    default_scatter = (;),
    group_scatter::AbstractDict = Dict(),
    default_color = nothing,
    group_color::AbstractDict = Dict(),
    palette = Makie.wong_colors(),
    axis = (;),
    figure = (;),
    legend = (;),
    show_legend::Bool = true,
    title::AbstractString = "Scores",
    xlabel::AbstractString = "Latent Variable 1",
    ylabel::AbstractString = "Latent Variable 2",
)
    size(scores, 2) >= 2 || error("scores must have at least two columns.")

    to_nt(x::NamedTuple) = x
    to_nt(x::AbstractDict) = (; (Symbol(k) => v for (k, v) in x)...)
    to_nt(::Nothing) = (;)

    groups_vec = collect(groups)
    order = isnothing(group_order) ? unique(groups_vec) : collect(group_order)

    getkw(d, g) = haskey(d, g) ? d[g] :
                  haskey(d, string(g)) ? d[string(g)] : nothing

    fig = Makie.Figure(; to_nt(figure)...)
    ax = Makie.Axis(
        fig[1, 1];
        xlabel = xlabel,
        ylabel = ylabel,
        title = title,
        to_nt(axis)...,
    )

    fallback_colors = collect(palette)
    isempty(fallback_colors) && (fallback_colors = collect(Makie.wong_colors()))

    for (k, g) in enumerate(order)
        idx = findall(==(g), groups_vec)
        isempty(idx) && continue

        scatter_kw = merge(to_nt(default_scatter), to_nt(getkw(group_scatter, g)))
        color_kw = getkw(group_color, g)
        color = isnothing(color_kw) ? default_color : color_kw
        isnothing(color) && (color = fallback_colors[mod1(k, length(fallback_colors))])

        Makie.scatter!(
            ax,
            scores[idx, 1],
            scores[idx, 2];
            label = string(g),
            color = color,
            scatter_kw...,
        )
    end

    show_legend && Makie.axislegend(ax; to_nt(legend)...)

    fig
end
