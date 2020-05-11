# 5 GLMの尤度比検定と検定の非対称性

```julia
using CSV
using Distributions
using GLM
using Plots
using StatsBase
using Statistics
```

```julia
df = CSV.read(joinpath("..", "data/section3.csv"))
```

### 5.4.1 方法(1) 汎用性のあるパラメトリックブートストラップ法
一定モデル

```julia
fit1 = glm(@formula(y ~ 1), df, GLM.Poisson())
```

xモデル

```julia
fit2 = glm(@formula(y ~ x), df, GLM.Poisson())
```

### 残差逸脱度の差

```julia
deviance(fit1) - deviance(fit2)
```

### 真のモデルから100個体分のデータを新しく生成

```julia
df.y_rnd = rand(Poisson(mean(df.y)), 100)
```

### 一定モデルとxモデルをこの真データに当てはめる

```julia
fit1 = glm(@formula(y_rnd ~ 1), df, GLM.Poisson())
fit2 = glm(@formula(y_rnd ~ x), df, GLM.Poisson())
deviance(fit1) - deviance(fit2)
```

### PB法を実行する関数
データの生成と逸脱度差の評価

```julia
function get_dd(df)
    n_samples = size(df, 1)
    y_mean = mean(df.y)
    df.y_rnd = rand(Poisson(y_mean), n_samples)
    fit1 = glm(@formula(y_rnd ~ 1), df, GLM.Poisson())
    fit2 = glm(@formula(y_rnd ~ x), df, GLM.Poisson())
    deviance(fit1) - deviance(fit2)
end
```

```julia
function pb(df, n_bootstrap)
    [get_dd(df) for _ in 1:n_bootstrap]
end
```

### 逸脱度の差のサンプルを1000個を作成

```julia
dd12 = pb(df, 1000)
```

```julia
describe(dd12)
```

```julia
histogram(dd12, bins=100, label="")
plot!([4.5], seriestype=:vline, linestyle=:dash, label="")
```

合計1000個の$\Delta D_{1,2}$のうちいくつが4.5より右にあるか

```julia
sum(dd12 .>= 4.5)
```

$P=0.05$となる逸脱度の差

```julia
quantile(dd12, 0.95)
```

### 方法(2) $\chi^2$分布を使った近似計算法

```julia
ccdf(Chisq(1), 4.513)
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

