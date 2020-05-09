# Chapter 1. Computer Arithmetic

```julia
using DataStructures
using Distributions
using Plots
using StatsBase
using Statistics
```

## 2.1 例題 : 種子数の統計モデリング
### 種子数データ

```julia
data = [2, 2, 4, 6, 4, 5, 2, 3, 1, 2, 0, 4, 3, 3, 3, 3,4, 2, 7, 2, 4, 3, 3, 3, 4,
3, 7, 5, 3, 1, 7, 6, 4, 6, 5, 2, 4, 7, 2, 2, 6, 2, 4, 5, 4, 5, 1, 3, 2, 3]
```

### データ数

```julia
length(data)
```

### データの要約

```julia
describe(data)
```

### 度数分布

```julia
SortedDict(countmap(data))
```

### ヒストグラム

```julia
histogram(data, bins=10, label="Data")
```

### 標本分散

```julia
var(data)
```

### 標本標準偏差

```julia
std(data)
```

## 2.2 データと確率分布の対応関係をながめる
### ポアソン分布
https://juliastats.github.io/Distributions.jl/latest/univariate.html#Distributions.Poisson

```julia
y = 0:9
prob = pdf.(Poisson(3.56), y)
```

```julia
plot(prob, linewidth = 2, linestyle = :dash, marker = 4)
```

### 観測データと確率分布の対応

```julia
histogram(data, bins = 10, label = "Data")
plot!(prob * 50, linewidth = 2, linestyle = :dash, marker = 4, label = "")
```

## 2.4 ポアソン分布のパラメーターの最尤推定
### 対数尤度 $\log L(\lambda)$と$\lambda$の関係

```julia
logL(m) = sum(log.(pdf.(Poisson(m), data)))
```

```julia
lambda = 2:0.1:5
plot(lambda, logL.(lambda), linewidth = 2, title = "log likelihood", label = "")
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

