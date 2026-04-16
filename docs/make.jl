import Pkg

if Base.active_project() ≠ joinpath(@__DIR__, "Project.toml")
    Pkg.activate(@__DIR__)
end

strict_docs_env = get(ENV, "NCPLS_DOCS_STRICT_ENV", "false") == "true"

if strict_docs_env
    empty!(LOAD_PATH)
    append!(LOAD_PATH, ["@", "@stdlib"])
end

if get(ENV, "CI", "false") == "true"
    Pkg.instantiate()
end

using Documenter
using Markdown
using NCPLS
using StatsAPI

const REPO = Documenter.Remotes.GitHub("oniehuis", "NCPLS.jl")

DocMeta.setdocmeta!(
    NCPLS,
    :DocTestSetup,
    :(using NCPLS; using StatsAPI);
    recursive = true,
)

makedocs(
    sitename = "NCPLS",
    format = Documenter.HTML(
        mathengine = Documenter.KaTeX(),
        edit_link = "main",
    ),
    modules = [NCPLS],
    repo = REPO,
    checkdocs = :none,
    authors = "Oliver Niehuis",
    pages = [
        "Home" => "index.md",
        "NCPLS" => Any[
            "NCPLS/theory.md",
            "NCPLS/types.md",
            "NCPLS/fit.md",
            "NCPLS/predict.md",
            "NCPLS/crossvalidation.md"
        ],
        "Register" => "register.md",
    ],
)

deploydocs(
    repo = "github.com/oniehuis/NCPLS.jl",
    devbranch = "main", 
    push_preview = false,
    versions = [
		"stable" => "v^",
		"v#.#.#",
		"dev" => "dev",
	],
)
