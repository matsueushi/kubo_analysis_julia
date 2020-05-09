root = joinpath(@__DIR__, "..")
using Pkg; Pkg.activate(root)
using Literate
# using Plots

src = joinpath(root, "src")
out_markdown = joinpath(root, "markdown")
out = joinpath(root, "notebook")

function preprocess(s)
    s = "using Pkg; Pkg.activate(\".\"); Pkg.instantiate()\n#-\n" * s
end

for f in ["Project.toml", "Manifest.toml"]
    cp(joinpath(root, f), joinpath(out, f), force = true)
end

# for x in filter(x -> x != "build.jl", readdir("src"))
for x in [
    "section2.jl"
    ] 
    Literate.markdown(joinpath(src, x), out_markdown; documenter = false)
    Literate.notebook(joinpath(src, x), out; preprocess = preprocess, execute = false, documenter = true)
end