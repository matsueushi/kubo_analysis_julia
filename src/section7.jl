
# # 7 一般化線形混合モデル (GLMM) -個体差のモデリング-

## #src
#-
using CSV
using DataFrames
using DataStructures
using Distributions
using GLM
using LaTeXStrings
using LinearAlgebra
using Plots
using QuadGK
using StatsBase
using Statistics
using MixedModels

## #src
# ## 7.1 例題 : GLMでは説明できないカウントデータ
#-
df = CSV.read(joinpath("..", "data/section7.csv"))

## #src
#-
df.N = convert(Array{Float64}, df.N)
df.yy = df.y ./ df.N
disallowmissing!(df)
categorical!(df, :id)

## #src
# ### データの図示
# 個体数をマーカーの大きさに反映させている
#
# 破線は「真の」生存確率の一例
#-
logistic(z) = 1 / (1 + exp(-z))

xs = 2:0.1:6
cols = [:x, :y]
df_plot = combine(groupby(df, cols), names(df, Not(cols)) .=> length)

scatter(df_plot.x, df_plot.y, markersize=df_plot.id_length .* 2, label="")
plot!(xs, mean.(Binomial.(8, logistic.(-4 .+ 1 .* xs))), label="",
    xlabel=L"x_i", ylabel=L"y_i",
    linestyle=:dash, linewidth=2, linecolor=:black)

## #src
# ### GLMを使ってデータから種子の生存確率を推定
#-
glm_model = glm(@formula(yy ~ x), df, Binomial(), wts=df.N)

## #src
#-
scatter(df_plot.x, df_plot.y, markersize=df_plot.id_length .* 2, label="",
    xlabel=L"x_i", ylabel=L"y_i")

plot!(xs, mean.(Binomial.(8, logistic.(-4 .+ 1 .* xs))), label="",
    linestyle=:dash, linewidth=2, linecolor=:red)

xx = DataFrame(x = xs)

plot!(xs, predict(glm_model, xx) .* 8,
    label="Prediction", linewidth=2, linecolor=:black)

## #src
# ### 葉数が4となるデータのサブセットを作る
#-
d4 = df[df.x .== 4, :]

## #src
#-
cols = [:x, :y]
d4_plot = combine(groupby(d4, cols), names(d4, Not(cols)) .=> length)

## #src
#-
xs4 = 0:8
ys4 = pdf.(Binomial(8, 0.47), xs4) * 20
scatter(d4_plot.y, d4_plot.id_length, label="",
    xlabel=L"y_i", ylabel="count")
plot!(xs4, ys4, linewidth=2, label="", marker=4)

## #src
# ## 7.2 過分散と個体差¶
# ### 7.2.1 過分散 : ばらつきが大きすぎる
# 生存数ごとにカウントする
#-
SortedDict(countmap(d4.y))

## #src
# データの平均と分散を調べる
mean(d4.y), var(d4.y)

## #src
# ### 7.4.1 Juliaを使ってGLMMのパラメーターを推定
# MixedModels.jlを使う
# https://juliastats.org/MixedModels.jl/latest/index.html
#
# フィッティング結果は本文のものとは少し異なる。
#
# $\hat{\beta}_1 = -4.19, \hat{\beta}_2 = 1.00, \hat{s}=2.41,$ residual deviance 
#-
glmm_model = fit(MixedModel, @formula(yy ~ x + (1 | id)), df, Binomial(); wts=df.N)

## #src
#-
scatter(df_plot.x, df_plot.y, markersize=df_plot.id_length .* 2, label="",
    xlabel=L"x_i", ylabel=L"y_i", legend=:topleft)

plot!(xs, mean.(Binomial.(8, logistic.(-4 .+ 1 .* xs))), 
    label="Actual",
    linestyle=:dash, linewidth=2, linecolor=:red)

plot!(xs, mean.(Binomial.(8, logistic.([fill(1, length(xs)) xs] * coef(glmm_model)))), 
    label="Prediction", linewidth=2, linecolor=:blue)

## #src
#-
Binomial(8, logistic(dot([1 4], coef(glmm_model))))

## #src
# ### 分布を混ぜる
#-
function f_gaussian_binorm(alpha, x, size, fixed, sd)
    pdf(Binomial(size, logistic(fixed + alpha)), x) * pdf(Normal(0, sd), alpha)
end

function d_gaussian_binorm(x, fixed, sd)
    quadgk(y -> f_gaussian_binorm(y, x, 8, fixed, sd), -sd * 10, sd * 10)[1]
end

## #src
# ### GLMMから予測された混合二項分布をプロットする
#-

## #src
#-
coef(glmm_model)

## #src
#-
glmm_model.σs

## #src
#-
pdf_gaussian_binorm = d_gaussian_binorm.(0:8, dot([1 4], coef(glmm_model)), glmm_model.σs[1][1])
scatter(d4_plot.y, d4_plot.id_length, label="",
    xlabel=L"y_i", ylabel="count")
plot!(0:8, pdf_gaussian_binorm * 20, linewidth=2, marker=4, label="")
