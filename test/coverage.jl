using Pkg
Pkg.instantiate()
using Coverage

coverage = vcat(
    process_folder(joinpath("src", "NCPLS")),
    process_folder(joinpath("ext", "makie_extensions")),
    process_folder(joinpath("ext", "plotly_extensions")),
)

LCOV.writefile("lcov.info", coverage)
