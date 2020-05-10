# 10 階層ベイズモデル -GLMMのベイズモデル化-

```julia
using Distributed
addprocs(3)
@everywhere begin
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
end
```

```julia
using CSV
using DataFrames
using DataFramesMeta
using Distributions
using ForneyLab
using LaTeXStrings
@everywhere using Mamba
using Plots
using QuadGK
@everywhere using Random
using StatsBase
using StatsFuns
using StatsPlots
@everywhere Random.seed!(0)
```

## 10.1 例題 : 個体差と生存種子数 (個体差あり)

```julia
df = CSV.read(joinpath("..", "data/section10a.csv"))
disallowmissing!(df)
```

青丸は観測データ, 白丸は生存確率0.504の二項分布. 二項分布では観測データのばらつきが説明できていない

```julia
scatter(sort(unique(df.y)), counts(df.y), label="")
ys = pdf.(Binomial(8, 0.504), 0:8) .* 100
plot!(0:8, ys, linecolor=:red, linewidth=2,
    marker=4, label="",
    xlabel="Survived", ylabel="Observed")
```

## 10.3 階層ベイズモデルの推定・予測
### 10.3.1 階層ベイズモデルのMCMCサンプリング
### モデルの作成
$$\begin{align} y_i \sim \text{Binomial}(q_i, 8) \\
\text{logit}(q_i) = \beta + r_i \\
\beta \sim \text{Normal}(0, 100^2) \\
r_i \sim \text{Normal}(0, \tau^2) \\
\tau \sim \text{Uniform}(0, 10000) \end{align}$$

```julia
model = Model(
    y = Stochastic(1,
        (beta, r, N) ->
            UnivariateDistribution[
                (q = invlogit(beta + r[i]);
                Binomial(8, q)) for i in 1:N
            ],
        false
    ),

    beta = Stochastic(() -> Normal(0, 100)),

    r = Stochastic(1, s -> Normal(0, s), false),

    s = Stochastic(() -> Uniform(0, 10000)),
)
```

### モデルの図示

```julia
display("image/svg+xml", ForneyLab.dot2svg(graph2dot(model)))
```

### データの設定

```julia
data = let y=df.y[:]
    Dict{Symbol, Any}(
        :y => y,
        :N => length(y),
    )
end
```

### 初期値の設定

```julia
inits = let y=df.y[:]
    [
        Dict{Symbol, Any}(
            :y => y,
            :beta => 0.0,
            :r => rand(Normal(0, 0.1), length(y)),
            :s => 1.0,
        ) for i in 1:3
    ]
end
```

### MCMCサンプル方法の設定
NUTSを使うと遅くなってしまったので、AMWGを使用

```julia
scheme = [
    AMWG(:r, 1),
    Slice(:s, 1.0, Mamba.Univariate),
    Slice(:beta, 1.0, Mamba.Univariate),
]
```

### MCMCサンプリング
サンプリングは21000回実施し, burn-inの数は1000, 10飛ばしの結果の記録を3チェイン行うので、
サンプルの数は(21000-1000)/10*3=6000個

```julia
setsamplers!(model, scheme)
sim = mcmc(model, data, inits, 21000, burnin=1000, thin=10, chains=3)
```

### サンプリング結果を表示

```julia
describe(sim)
```

```julia
p = Mamba.plot(sim, legend=true)
Mamba.draw(p, nrow=2, ncol=2)
```

```julia
p = Mamba.plot(sim, [:autocor, :mean], legend=true)
Mamba.draw(p, nrow=2, ncol=2)
```

```julia
p = Mamba.plot(sim, :contour)
Mamba.draw(p, nrow=1, ncol=1)
```

### 10.3.2 階層ベイズモデルの事後分布推定と予測
$p(y \mid \beta, r)p(r \mid s)$を求める関数

```julia
function f_gaussian_binorm(alpha, x, size, fixed, sd)
    pdf(Binomial(size, logistic(fixed + alpha)), x) * pdf(Normal(0, sd), alpha)
end
```

$p(y \mid \beta, s) = \int_{-\infty}^\infty p(y \mid \beta, r)p(r \mid s)dr$を求める関数.
実際は$\int_{-10s}^{10s} p(y \mid \beta, r)p(r \mid s)dr$を求めている

```julia
function d_gaussian_binorm(x, fixed, sd)
    quadgk(y -> f_gaussian_binorm(y, x, 8, fixed, sd), -sd * 10, sd * 10)[1]
end
```

各パスに対し, $y = 0, \ldots, 8$ に対して $p(y \mid \beta, s)$ を評価する.

```julia
posterior = map((x, y) -> d_gaussian_binorm.(0:8, x, y), sim[:, [:beta], :].value[:], sim[:, [:s], :].value[:])
```

各パスに対して、$P(X=y)=p(y \mid \beta, s)\ \text{for}\ y = 0, \ldots, 8$というCategorical distributionから100個体のサンプリングを行い,
$y$ごとに出現回数を数える. 横方向は各サンプリング, 縦方向は$y = 0, \ldots, 8$の出現回数に該当

```julia
population_samples = hcat(map(x -> fit(Histogram, rand(Distributions.Categorical(x), 100) .- 1, 0:9).weights, posterior)...)
```

$y$毎に出現回数の2.5%, 50%, 97.5%点を計算

```julia
function quantile_sample(r)
    map(x -> quantile(population_samples[x, :], r), 1:9)
end
quantile_sample_0025 = quantile_sample(0.025)
quantile_sample_0975 = quantile_sample(0.975)
quantile_sample_median = quantile_sample(0.5)
```

生存種子数の予測分布
各 $y$ における中央値, 及び95%区間の領域を表示する

```julia
scatter(sort(unique(df.y)), counts(df.y), label="")
Plots.plot!(0:8, quantile_sample_0025, lw = 0,
    fillrange=quantile_sample_0975,
    fillalpha=0.3, fillcolor=:orange,
    label="95% Confidence interval")
Plots.plot!(0:8, quantile_sample_median, linewidth=2, marker=4, label="",
            xlabel="Survived", ylabel="Observed")
```

## 10.5 個体差 + 場所差の階層ベイズモデル

```julia
df2 = CSV.read(joinpath("..", "data/section10b.csv"))
disallowmissing!(df2)
```

```julia
marker_dict = Dict(
    "A" => :circle,
    "B" => :ltriangle,
    "C" => :star5,
    "D" => :diamond,
    "E" => :dtriangle,
    "F" => :xcross,
    "G" => :star4,
    "H" => :utriangle,
    "I" => :rect,
    "J" => :rtriangle
)
```

### 個体ごとの表示
赤線は無処理、青線は堆肥処理した個体の平均

```julia
plt = Plots.plot()
for k in marker_dict |> keys |> collect |> sort
    @linq df_k = df2 |> where(:pot .== k)
    scatter!(df_k.id, df_k.y, label=k, markershape=marker_dict[k],
            legend=:topleft, legendfontsize=6, xlabel=L"i", ylabel=L"y_i")
end
plot!(1:50, fill(mean(df2[1:50, :].y), 50),
    linestyle=:dash, linewidth=2, linecolor=:red, label="")
plot!(51:100, fill(mean(df2[51:100, :].y), 50),
    linestyle=:dash, linewidth=2, linecolor=:blue, label="")
plt
```

### 植木鉢毎に箱ひげ図として図示

```julia
boxplot(df2[1:50, :].pot, df2[1:50, :].y, label="")
boxplot!(df2[51:100, :].pot, df2[51:100, :].y, label="", xlabel="pot", ylabel=L"y_i")
```

### GLMM化したポアソン回帰
個体$i$の種子数$y_i$を平均$\lambda_i$のポアソン回帰

$$\begin{align} p(y_i \mid \lambda_i) = \frac{\lambda_i^{y_i}\exp(-\lambda_i)}{y_i!} \end{align}$$

で表現し, 平均種子数は切片$\beta_1$, 堆肥処理の有無を表す因子型の説明変数$f_i$の係数$\beta_2$, 個体$i$の効果$r_i$と植木鉢$j$の効果$t_{j(i)}$で

$$\begin{align} \log \lambda_i = \beta_1 + \beta_2 f_i + r_i + t_{j(i)} \end{align}$$

で表現.
$$\begin{align} y_i \sim \text{Poisson}(\lambda_i), \ i = 1, \ldots, 100 \\
\log \lambda_i = \beta_1 + \beta_2 f_i + r_i + t_{j(i)} \\
\beta_1, \beta_2 \sim \text{Normal}(0, 100) \\
r_i \sim \text{Normal}(0, s_r^2) \\
t_j \sim \text{Normal}(0, s_t^2), j = 1, \ldots, 10 \\
s_r, s_t \sim \text{Uniform}(0, 10000) \end{align}$$

```julia
model2 = Model(

    y = Stochastic(1,
        (beta1, beta2, f, r, t, pot, N_r) ->
            UnivariateDistribution[
                (lambda=exp(beta1 + beta2 * f[i] + r[i] + t[pot[i]]);
                Mamba.Poisson(lambda)) for i in 1:N_r
            ],
        false
    ),

    r = Stochastic(1, s_r -> Normal(0, s_r), false),

    t = Stochastic(1, s_t -> Normal(0, s_t), false),

    beta1 = Stochastic(() -> Normal(0, 100)),
    beta2 = Stochastic(() -> Normal(0, 100)),

    s_r = Stochastic(() -> Uniform(0, 10000)),
    s_t = Stochastic(() -> Uniform(0, 10000)),
)
```

### モデルの図示

```julia
display("image/svg+xml", ForneyLab.dot2svg(graph2dot(model2)))
```

### 入力データの設定

```julia
pot_dict = Dict(string(y) => x for (x, y) in enumerate("ABCDEFGHIJ"))
f_dict = Dict("C" => 0, "T" => 1)
data2 = let y=df2.y[:], pot=df2.pot[:], f=df2.f[:]
    Dict{Symbol, Any}(
        :y => y,
        :N_r => length(y),
        :N_t => length(unique(pot)),
        :pot => [pot_dict[x] for x in pot],
        :f => [f_dict[x] for x in f],
    )
end
```

### 初期値の設定

```julia
inits2 = [
    Dict{Symbol, Any}(
        :y => data2[:y],
        :beta1 => 0.0,
        :beta2 => 0.0,
        :r => rand(Normal(0, 0.1), data2[:N_r]),
        :t => rand(Normal(0, 0.1), data2[:N_t]),
        :s_r => 1.0,
        :s_t => 1.0,
    ) for i in 1:3
]
```

```julia
scheme2 = [
    AMWG([:r], 0.1),
    AMWG([:t], 0.1),
    Slice([:s_r, :s_t], 0.1, Mamba.Univariate),
    Slice([:beta1, :beta2], 1.0, Mamba.Univariate)
]
```

```julia
setsamplers!(model2, scheme2)
sim2 = mcmc(model2, data2, inits2, 22000, burnin=2000, thin=10, chains=3)
```

### 事後分布を確認する
β2の95%区間を見ると、堆肥処理の効果はなさそう

```julia
describe(sim2)
```

```julia
p = Mamba.plot(sim2, legend=true)
Mamba.draw(p[:, 1:2], nrow=2, ncol=2)
```

```julia
Mamba.draw(p[:, 3:4], nrow=2, ncol=2)
```

```julia
p = Mamba.plot(sim2, [:autocor, :mean], legend=true)
Mamba.draw(p[:, 1:2], nrow=2, ncol=2)
```

```julia
Mamba.draw(p[:, 3:4], nrow=2, ncol=2)
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

