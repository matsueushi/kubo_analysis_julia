# 3 一般化線形モデル(GLM) -ポアソン回帰-

```julia
using CSV
using DataFrames
using GLM
using LaTeXStrings
using Plots
using StatsBase
using StatsPlots
```

## 3.2 観測されたデータの概要を調べる

```julia
df = CSV.read(joinpath("..", "data/section3.csv"))
```

### 列ごとにデータを表示

```julia
df.x
```

```julia
df.y
```

```julia
df.f
```

### データオブジェクトの型を調べる

```julia
typeof(df)
```

```julia
typeof(df.y)
```

```julia
typeof(df.x)
```

```julia
typeof(df.f)

### データの要約
```

```julia
describe(df)
```

## 3.3 統計モデリングの前にデータを図示する
### データの図示
散布図

```julia
scatter(df.x, df.y, group=df.f)
```

箱ひげ図

```julia
boxplot(df.f, df.y, label="")
```

## 3.4 ポアソン回帰の統計モデル
### 3.4.2 当てはめと当てはまりの良さ
### GLMのフィッティング
http://juliastats.github.io/GLM.jl/latest/examples/

```julia
names(df)
```

```julia
result = glm(@formula(y ~ x), df, GLM.Poisson())
```

```julia
loglikelihood(result)
```

### 3.4.3 ポアソン回帰モデルによる予測

```julia
plot(df.x, df.y, group=df.f, seriestype=:scatter)
xx = range(minimum(df.x), maximum(df.x), length=100)
plot!(xx, exp.(1.29 .+ 0.0757 .* xx), linewidth=2, label=L"\lambda")
```

```julia
plot(df.x, df.y, group = df.f, seriestype=:scatter)
xx = DataFrame(x=range(minimum(df.x), maximum(df.x), length=100))
yy = predict(result, xx)
plot!(xx.x, yy, linewidth=2, label=L"\lambda")
```

## 3.5 説明変数が因子型の統計モデル

```julia
result_f = glm(@formula(y ~ f), df, GLM.Poisson())
```

```julia
loglikelihood(result_f)
```

## 3.6 説明変数が数量型 + 因子型の統計モデル

```julia
result_all = glm(@formula(y ~ x + f), df, GLM.Poisson())
```

```julia
loglikelihood(result_all)
```

### 対数リンク関数のわかりやすさ : 掛け算される効果
恒等リンク関数

```julia
result_identity = glm(@formula(y ~ x + f), df, GLM.Poisson(), IdentityLink())
```

```julia
loglikelihood(result_identity)
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

