# ## 9 GLMのベイズモデル化と事後分布の推定
# GraphViz はJulia 1.0をサポートしていないため、ForneyLabを使う

## #src
#-
using CSV
using DataFrames
using ForneyLab 
using GLM
using LaTeXStrings
using LinearAlgebra
using Mamba
using Plots
using Random
Random.seed!(0)

## #src
#-
df = CSV.read(joinpath("..", "data/section9.csv"))
disallowmissing!(df)

## #src
# ## 9.1 例題 : 種子数のポアソン回帰 (個体差なし)
#-
glm_model = glm(@formula(y ~ x), df, GLM.Poisson())

## #src
#-
scatter(df.x, df.y, label="", xlabel="Size", ylabel="Number of seeds")
xs = 3:0.1:7
xx = DataFrame(x=xs)
plot!(xs, exp.(1.5 .+ 0.1 .* xs), label="Actual", linewidth=2, 
    linestyle=:dash, linecolor=:red)
plot!(xs, predict(glm_model, xx), label="Prediction", linewidth=2, 
    linecolor=:black)

## #src
# ## 9.4 ベイズ統計モデルの事後分布の推定
# ### 9.4.1 ベイズ統計モデルのコーディング
# $$\begin{align} y_i \sim \text{Poisson}(\lambda_i)\\
# \log(\lambda_i) = \beta_1 + \beta_2 \cdot x_i\\
# \beta_1 \sim \text{Normal}(0, 100^2)\\
# \beta_2 \sim \text{Normal}(0, 100^2) \end{align}$$
#
# Mamba.jl ( https://mambajl.readthedocs.io/ )を使う
# ### モデルの作成
#-
model = Model(
    y = Stochastic(1,
        (N, lambda) ->
        (UnivariateDistribution[Mamba.Poisson(lambda[i]) for i in 1:N]),
        false
    ),

    lambda = Logical(1,
        (x, beta) -> exp.(x * beta)
    ),

    beta = Stochastic(1,
        () -> MvNormal(2, 100)
    ),
)


## #src
# ### モデルをプロットする
#-
display("image/svg+xml", ForneyLab.dot2svg(graph2dot(model)))

## #src
# ### データの設定
#-
data = Dict{Symbol, Any}(
    :x => [ones(length(df.x)) df.x],
    :y => df.y[:],
    :N => length(df.x)
)

## #src
# ### 初期値の設定
#-
inits = [
    Dict{Symbol, Any}(
        :y => df.y[:],
        :beta => zeros(2),
    ) for _ in 1:3
]

## #src
# ### サンプラーの設定
#-
scheme = [NUTS([:beta])]

## #src
# ### MCMCシミュレーション
#-
setsamplers!(model, scheme)
sim = mcmc(model, data, inits, 1600, burnin=100, thin=3, chains=3)

## #src
# ### 9.4.3 どれだけ長くMCMCサンプリングすればいいのか？
#-
p = Mamba.plot(sim[:, :beta, :], [:autocor, :mean], legend=true)
Mamba.draw(p, nrow=2, ncol=2)

## #src
# ### Gelman-Rubin diagnostic ($\hat{R}$)の推定値
#-
gelmandiag(sim[:, :beta, :], mpsrf=true, transform=true)

## #src
# ## 9.5 MCMCサンプルから事後分布を推定¶
# ### シミュレーション結果の表示と図示
#-
p = Mamba.plot(sim[:, :beta, :], legend=true)
Mamba.draw(p, nrow=2, ncol=2)

## #src
# ### 9.5.1 事後分布の統計量
#-
describe(sim[:, :beta, :])

## #src
#-
xs = collect(3:0.1:7)
xs_mat = [ones(length(xs)) xs]

sim_beta = sim[:, [:beta], :].value
sim_beta = reshape(permutedims(sim_beta, (2, 1, 3)), size(sim_beta, 2), :)

## #src
#-
beta_median = vec(mapslices(median, sim[:, [:beta], :].value, dims=(1,3)))

## #src
#-
Plots.plot(xs, exp.(xs_mat * sim_beta), label="", linecolor=:red, linealpha=0.1)
Plots.plot!(xs, exp.(xs_mat * beta_median), label="", 
            linewidth=2, linecolor=:black)
scatter!(df.x, df.y, 
    label="", 
    markercolor=:orange, 
    xlabel="Size", ylabel="Number of seeds", 
    title="Estimation of lambda")

## #src
#-
p = Mamba.plot(sim[:, :beta, :], :contour)
Mamba.draw(p, nrow=1, ncol=1)
