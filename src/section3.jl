# # 3 一般化線形モデル(GLM) -ポアソン回帰-

## #src
#-
using CSV
using DataFrames
using GLM
using LaTeXStrings
using Plots
using StatsBase
using StatsPlots

## #src
# ## 3.2 観測されたデータの概要を調べる
#-
df = CSV.read(joinpath("..", "data/section3.csv"))

## #src
# ### 列ごとにデータを表示
#-
df.x

## #src
#-
df.y

## #src
#-
df.f

## #src
# ### データオブジェクトの型を調べる
#-
typeof(df)

## #src
#-
typeof(df.y)

## #src
#-
typeof(df.x)

## #src
#-
typeof(df.f)

## #src
### データの要約
#-
describe(df)

## #src
# ## 3.3 統計モデリングの前にデータを図示する
# ### データの図示
# 散布図
#-
scatter(df.x, df.y, group=df.f)

## #src
# 箱ひげ図
#-
boxplot(df.f, df.y, label="")

## #src
# ## 3.4 ポアソン回帰の統計モデル
# ### 3.4.2 当てはめと当てはまりの良さ
# ### GLMのフィッティング
# http://juliastats.github.io/GLM.jl/latest/examples/
#-
names(df)

## #src
#-
result = glm(@formula(y ~ x), df, GLM.Poisson())

## #src
#-
loglikelihood(result)

## #src
# ### 3.4.3 ポアソン回帰モデルによる予測
#-
plot(df.x, df.y, group=df.f, seriestype=:scatter)
xx = range(minimum(df.x), maximum(df.x), length=100)
plot!(xx, exp.(1.29 .+ 0.0757 .* xx), linewidth=2, label=L"\lambda")

## #src
#-
plot(df.x, df.y, group = df.f, seriestype=:scatter)
xx = DataFrame(x=range(minimum(df.x), maximum(df.x), length=100))
yy = predict(result, xx)
plot!(xx.x, yy, linewidth=2, label=L"\lambda")

## #src
# ## 3.5 説明変数が因子型の統計モデル
#-
result_f = glm(@formula(y ~ f), df, GLM.Poisson())

## #src
#-
loglikelihood(result_f)

## #src
# ## 3.6 説明変数が数量型 + 因子型の統計モデル
#-
result_all = glm(@formula(y ~ x + f), df, GLM.Poisson())

## #src
#-
loglikelihood(result_all)

## #src
# ### 対数リンク関数のわかりやすさ : 掛け算される効果
# 恒等リンク関数
#-
result_identity = glm(@formula(y ~ x + f), df, GLM.Poisson(), IdentityLink())

## #src
#-
loglikelihood(result_identity)
