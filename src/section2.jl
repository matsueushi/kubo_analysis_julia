# # 2 確率分布と統計モデルの最尤推定

## #src
#-
using DataStructures
using Distributions
using Plots
using StatsBase
using Statistics

## #src
# ## 2.1 例題 : 種子数の統計モデリング
# ### 種子数データ
#-
data = [2, 2, 4, 6, 4, 5, 2, 3, 1, 2, 0, 4, 3, 3, 3, 3,4, 2, 7, 2, 4, 3, 3, 3, 4,
3, 7, 5, 3, 1, 7, 6, 4, 6, 5, 2, 4, 7, 2, 2, 6, 2, 4, 5, 4, 5, 1, 3, 2, 3]

## #src
# ### データ数
#-
length(data)

## #src
# ### データの要約
#-
describe(data)

## #src
# ### 度数分布
#-
SortedDict(countmap(data))

## #src
# ### ヒストグラム
#-
histogram(data, bins=10, label="Data")

## #src
# ### 標本分散
#-
var(data)

## #src
# ### 標本標準偏差
#-
std(data)

## #src
# ## 2.2 データと確率分布の対応関係をながめる
# ### ポアソン分布
# https://juliastats.github.io/Distributions.jl/latest/univariate.html#Distributions.Poisson
#-
y = 0:9
prob = pdf.(Poisson(3.56), y)

## #src
#-
plot(prob, linewidth=2, linestyle=:dash, marker=4)

## #src
# ### 観測データと確率分布の対応
#-
histogram(data, bins=10, label="Data")
plot!(prob * 50, linewidth=2, linestyle=:dash, marker=4, label="")

## #src
# ## 2.4 ポアソン分布のパラメーターの最尤推定
# ### 対数尤度 $\log L(\lambda)$と$\lambda$の関係
#-
logL(m) = sum(log.(pdf.(Poisson(m), data)))

#-
lambda = 2:0.1:5
plot(lambda, logL.(lambda), linewidth=2, title="log likelihood", label="")
