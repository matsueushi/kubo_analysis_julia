# # 11 空間構造のある階層ベイズモデル

## #src
#-
using Distributed
addprocs(3)
@everywhere begin
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
end

## #src
#-
using CSV
using DataFrames
using Distributions
using ForneyLab
using LinearAlgebra
@everywhere using Mamba
using Plots
@everywhere using Random
using SparseArrays
using Statistics
using StatsBase
@everywhere Random.seed!(0)

## #src
# ## 11.1 例題 : 一次元空間上の個体数分布
# ### 例題の一次元空間上の架空データ
#-
df = CSV.read(joinpath("..", "data/section11.csv"))
disallowmissing!(df)

## #src
# 横軸は調査区間の位置, 縦軸は観測された個体数, 破線はデータをポアソン乱数で生成するときに使った平均値
#-
scatter(1:50, df.y, xlabel="Position", ylabel="Population", markercolor=:white, label="y")
plot!(1:50, df.m, linewidth=2, linecolor=:black, linestyle=:dash, label="")

## #src
# ## 11.2 階層ベイズモデルに空間構造を組み込む
# まず, 個体数$y_i$は, すべでの区画で共通する平均$\lambda$のポアソン分布に従うとしてみる.
# $$\begin{align} y_i \sim \text{Poisson}(\lambda),\ p(y_j \mid \lambda) = \frac{\lambda^{y_i}\exp(-\lambda)}{y_j!} \end{align}$$
# このように仮定してすると, 平均$\lambda$と標本平均と等しいとしてみる.
# ### 標本平均
#-
mean(df.y)

## #src
# ところが標本分散を求めてみると,
#-
var(df.y)

## #src
# となり, 標本平均の3倍近くなっている. そのため個体数が全ての区画で共通する平均$\lambda$のPoisson分布に従うと言う仮定は成立していなさそう.
#
# そのため, 区間$j$ごとに平均$\lambda_i$が異なっているとし, 平均個体数を切片$\beta$と場所差$r_j$を用いて
# $$\begin{align} \log \lambda_i = \beta + r_j \end{align}$$
# と表す.
#
# ### 11.2.1. 空間構造のない階層事前分布
# 場所差$r_j$の事前分布を第10章で使ったような階層事前分布
# $$\begin{align} r_j \sim \text{Normal}(0, s^2),\ p(r_j \mid s) = \frac{1}{\sqrt{2\pi s^2}}\exp \left( -\frac{r_j^2}{2s^2} \right) \end{align}$$
# であるとしてモデリングしてみる.
#-
model1 = Model(

    y = Stochastic(1,
        (lambda, N) ->
            UnivariateDistribution[
            (Mamba.Poisson(lambda[i])) for i in 1:N
        ],
        false
    ),
    
    lambda = Logical(1,
        (beta, r) -> exp.(beta .+ r)),
    
    beta = Stochastic(() -> Normal(0, 100)),
    
    r = Stochastic(1, s -> Normal(0, s), false),
    
    s = Stochastic(() -> Uniform(0, 10000))
)

## #src
#-
data1 = let y=df.y[:]
    Dict{Symbol, Any}(
        :y => y,
        :N => length(y),
    )
end

## #src
#-
inits1 = [
    Dict{Symbol, Any}(
        :y => data1[:y],
        :beta => 0.0,
        :r => rand(Normal(0, 0.1), length(data1[:y])),
        :s => 1.0,
    ) for i in 1:3
]

## #src
#-
scheme1 = [
    AMWG(:r, 1),
    Slice(:s, 1.0),
    Slice(:beta, 1.0),
]

## #src
#-
setsamplers!(model1, scheme1)
sim1 = mcmc(model1, data1, inits1, 60000, burnin=10000, thin=10, chains=3)

## #src
#-
describe(sim1)

## #src
#-
p1 = Mamba.plot(sim1, legend=true)
Mamba.draw(p1[:, 1:2], nrow=2, ncol=2)

## #src
#-
p2 = Mamba.plot(sim1, [:autocor, :mean], legend=true)
Mamba.draw(p2[:, 1:2], nrow=2, ncol=2)

## #src
# $\lambda$の中央値、95%信頼区間をプロットしてみる
#-
mre = quantile(sim1).value[3:end, :]

## #src
#-
Plots.plot(1:50, mre[:, 1], lw=0, 
    fillrange=mre[:, 5],
    fillcolor=:skyblue,
    fillalpha=0.6, 
    label="95% Confidence interval")
scatter!(1:50, df.y, xlabel="Position", ylabel="Population", markercolor=:white, label="y")
plot!(1:50, df.m, linewidth=2, linecolor=:black, linestyle=:dash, label="")
plot!(1:50, mre[:, 3], linewidth=2, linecolor=:red, label="Median")

## #src
# 11.2.2 空間構造のある階層事前分布
# ある区間はそれと隣接する区間とだけ相互作用すると仮定する.
# 近傍数は$n_j = 2 \ (j \neq 1, 50), 1 \ (j = 1, 50)$となる.
# $$\begin{align} \mu_j = \frac{r_{j-1} + r_{j+1}}{2}\ (j \neq 1, 50), \mu_1 = r_2, \mu_{50} = r_{49},\\
# r_j \mid \mu_j, s \sim \text{Normal}\left(\mu_j, \frac{s}{\sqrt{n_j}}\right) \end{align}$$
# と言うモデルを考える.
# ### 11.3 空間統計モデルをデータに当てはめる
# CARモデルを実装する。 参考 : https://github.com/matsueushi/lip_stick_mamba
#-
model2 = Model(

    y = Stochastic(1,
        (lambda, N) ->
            UnivariateDistribution[
            (Mamba.Poisson(lambda[i])) for i in 1:N
        ],
        false
    ),
    
    lambda = Logical(1,
        (beta, r) -> exp.(beta .+ r)),
    
    beta = Stochastic(() -> Normal(0, 100)),
    
    r = Stochastic(1,
        (s, alpha, N, D, adj) ->
            MvNormalCanon(zeros(N), 1 / (s * s) * (D - alpha * adj)),
        false
    ),
    
    alpha = Stochastic(() -> Uniform()),
    
    s = Stochastic(() -> Uniform(0, 10000))
)

## #src
#-
adj = zeros(50, 50)
for i in 1:50-1
    adj[i, i+1] = 1
    adj[i+1, i] = 1
end
adj

## #src
#-
D = Diagonal(vec(sum(adj, dims=2)))

## #src
#-
data2 = let y=df.y[:]
    Dict{Symbol, Any}(
        :y => y,
        :N => length(y),
        :adj => adj,
        :D => D,
    )
end

## #src
#-
inits2 = [
    Dict{Symbol, Any}(
        :y => data2[:y],
        :alpha => 0.9,
        :beta => 0.0,
        :r => rand(Normal(0, 0.1), data2[:N]),
        :s => 1.0,
    ) for i in 1:3
]

## #src
#-
scheme2 = [
    AMWG(:r, 1),
    Slice(:s, 1.0),
    Slice([:alpha, :beta], 1.0),
]

## #src
#-
setsamplers!(model2, scheme2)

## #src
#-
sim2 = mcmc(model2, data2, inits2, 60000, burnin=10000, thin=10, chains=3)

## #src
#-
describe(sim2)

## #src
#-
p3 = Mamba.plot(sim2, legend=true)
Mamba.draw(p3[:, 1:3], nrow=3, ncol=2)

## #src
#-
p4 = Mamba.plot(sim2, [:autocor, :mean], legend=true)
Mamba.draw(p4[:, 1:3], nrow=3, ncol=2)

## #src
#-
mre2 = quantile(sim2).value[4:end, :]

## #src
# ### Plot
#-
Plots.plot(1:50, mre2[:, 1], lw=0, 
    fillrange=mre2[:, 5],
    fillcolor=:skyblue,
    fillalpha=0.6, 
    label="95% Confidence interval")
scatter!(1:50, df.y, xlabel="Position", ylabel="Population", 
        markercolor=:white, label="y")
plot!(1:50, df.m, linewidth=2, linecolor=:black, linestyle=:dash, label="")
plot!(1:50, mre2[:, 3], linewidth=2, linecolor=:red, label= "Median")

## #src
# ## 11.5 空間相関モデルと欠測のある観測データ
# Missing Values Sampler
# 
# https://mambajl.readthedocs.io/en/latest/samplers/miss.html
#
# を使って、欠測のある観測データを使った予測を行う。
# ### まずは、空間相関のないモデル
#-
y_missing = convert(Vector{Union{Missing, Float64}}, df.y)
missing_place = [6, 9, 12, 13, 26, 27, 28, 29, 30]
y_missing[missing_place] .= NaN
y_missing

## #src
#-
data1_missing = Dict{Symbol, Any}(
    :y => y_missing,
    :N => length(df.y[:]),
)

## #src
#-
inits1_missing = [
    Dict{Symbol, Any}(
        :y => y_missing,
        :beta => 0.0,
        :r => rand(Normal(0, 0.1), data1_missing[:N]),
        :s => 1.0,
    ) for i in 1:3
]

## #src
#-
scheme1_missing = [
    MISS(:y),
    AMWG(:r, 1),
    Slice(:s, 1.0),
    Slice(:beta, 1.0),
]

## #src
#-
setsamplers!(model1, scheme1_missing)
sim1_missing = mcmc(model1, data1_missing, inits1_missing, 60000, 
                    burnin=10000, thin=10, chains=3)

## #src
#-
describe(sim1_missing)

## #src
#-
p5 = Mamba.plot(sim1_missing, legend=true)
Mamba.draw(p5[:, 1:2], nrow=2, ncol=2)

## #src
#-
p6 = Mamba.plot(sim1_missing, [:autocor, :mean], legend=true)
Mamba.draw(p6[:, 1:2], nrow=2, ncol=2)

## #src
# ### 次に、空間相関のあるモデル
#-
data2_missing = Dict{Symbol, Any}(
    :y => y_missing,
    :N => length(df.y[:]),
    :adj => adj,
    :D => D,
)

## #src
#-
inits2_missing = [
    Dict{Symbol, Any}(
        :y => y_missing,
        :alpha => 0.9,
        :beta => 0.0,
        :r => rand(Normal(0, 0.1), data2_missing[:N]),
        :s => 1.0,
    ) for i in 1:3
]

## #src
#-
scheme2_missing = [
    MISS(:y),
    AMWG(:r, 1),
    Slice(:s, 1.0),
    Slice([:alpha, :beta], 1.0),
]

## #src
#-
setsamplers!(model2, scheme2_missing)
sim2_missing = mcmc(model2, data2_missing, inits2_missing, 60000, 
                    burnin=10000, thin=10, chains=3)

## #src
#-
describe(sim2_missing)

## #src
#-
p7 = Mamba.plot(sim2_missing, legend=true)
Mamba.draw(p7[:, 1:3], nrow=3, ncol=2)

## #src
#-
p8 = Mamba.plot(sim2_missing, [:autocor, :mean], legend=true)
Mamba.draw(p8[:, 1:3], nrow=3, ncol=2)

## #src
# ### モデルの比較
# 空間相関のあるモデル・ないモデルを比較する。
#-
scatter_color = fill(:white, 50)
scatter_color[missing_place] .= :black

## #src
#-
vsspan_x = collect(Iterators.flatten(zip(missing_place .- 0.5, missing_place .+ 0.5)))

## #src
# 空間相関を考慮していないモデル
#-
mre3 = quantile(sim1_missing).value[3:end, :]
vspan(vsspan_x, fillcolor=:black, linecolor=:transparent, fillalpha=0.15, label="")
plot!(1:50, mre3[:, 1], lw=0, 
    fillrange=mre3[:, 5],
    fillcolor=:skyblue,
    fillalpha=0.6, 
    label="95% Confidence interval")
scatter!(1:50, df.y, xlabel="Position", ylabel="Population", 
        markercolor=scatter_color, label="y")
plot!(1:50, df.m, linewidth=2, linecolor=:black, linestyle=:dash, label="")
plot!(1:50, mre3[:, 3], linewidth=2, linecolor=:red, label="Median")

## #src
# 空間相関を考慮しているモデル
#-
mre4 = quantile(sim2_missing).value[4:end, :]
vspan(vsspan_x, fillcolor=:black, linecolor=:transparent, fillalpha=0.15, label="")
plot!(1:50, mre4[:, 1], lw=0, 
    fillrange=mre4[:, 5],
    fillcolor=:skyblue,
    fillalpha=0.6, 
    label="95% Confidence interval")
scatter!(1:50, df.y, xlabel="Position", ylabel="Population", 
        markercolor=scatter_color, label="y")
plot!(1:50, df.m, linewidth=2, linecolor=:black, linestyle=:dash, label="")
plot!(1:50, mre4[:, 3], linewidth=2, linecolor=:red, label="Median")

## #src
# 空間相関を考慮すると、欠測データに対し隣同士の相互作用を用いた予測ができるため、相関を考慮しないものに比べて予測区間の幅が小さくなる
