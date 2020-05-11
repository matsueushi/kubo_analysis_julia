# # 5 GLMの尤度比検定と検定の非対称性

## #src
#-
using CSV
using Distributions
using GLM
using Plots
using StatsBase
using Statistics

## #src
#-
df = CSV.read(joinpath("..", "data/section3.csv"))

## #src
# ### 5.4.1 方法(1) 汎用性のあるパラメトリックブートストラップ法
# 一定モデル
#-
fit1 = glm(@formula(y ~ 1), df, GLM.Poisson())

## #src
# xモデル
#-
fit2 = glm(@formula(y ~ x), df, GLM.Poisson())

## #src
# ### 残差逸脱度の差
#-
deviance(fit1) - deviance(fit2)

## #src
# ### 真のモデルから100個体分のデータを新しく生成
#-
df.y_rnd = rand(Poisson(mean(df.y)), 100)

## #src
# ### 一定モデルとxモデルをこの真データに当てはめる
#-
fit1 = glm(@formula(y_rnd ~ 1), df, GLM.Poisson())
fit2 = glm(@formula(y_rnd ~ x), df, GLM.Poisson())
deviance(fit1) - deviance(fit2)

## #src
# ### PB法を実行する関数
# データの生成と逸脱度差の評価
function get_dd(df)
    n_samples = size(df, 1)
    y_mean = mean(df.y)
    df.y_rnd = rand(Poisson(y_mean), n_samples)
    fit1 = glm(@formula(y_rnd ~ 1), df, GLM.Poisson())
    fit2 = glm(@formula(y_rnd ~ x), df, GLM.Poisson())
    deviance(fit1) - deviance(fit2)
end

## #src
#-
function pb(df, n_bootstrap)
    [get_dd(df) for _ in 1:n_bootstrap]
end

## #src
# ### 逸脱度の差のサンプルを1000個を作成
#-
dd12 = pb(df, 1000)

## #src
#-
describe(dd12)

## #src
#-
histogram(dd12, bins=100, label="")
plot!([4.5], seriestype=:vline, linestyle=:dash, label="")

## #src
# 合計1000個の$\Delta D_{1,2}$のうちいくつが4.5より右にあるか
#-
sum(dd12 .>= 4.5)

## #src
# $P=0.05$となる逸脱度の差
#-
quantile(dd12, 0.95)

## #src
# ### 方法(2) $\chi^2$分布を使った近似計算法
#-
ccdf(Chisq(1), 4.513)
