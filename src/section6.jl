# # 6 GLMの応用範囲をひろげる -ロジスティック回帰など-

## #src
#-
using CSV
using DataFrames
using Distributions
using GLM
using LaTeXStrings
using LinearAlgebra
using Plots
using StatsBase

## #src
# ## 6.2 例題 : 上限のあるカウントデータ
#-
df = CSV.read(joinpath("..", "data/section6a.csv"))

## #src
#-
disallowmissing!(df)

## #src
#-
scatter(df.x, df.y, group=df.f, 
        xlabel="Plant size", 
        ylabel="Number of survival seeds")

## #src
#-
describe(df)

## #src
# ## 6.3 二項分布で表現する「あり・なし」カウントデータ
# https://juliastats.github.io/Distributions.jl/latest/univariate.html#Distributions.Binomial
#-
xx = 0:8
plt = plot(ylabel="Probability")
for x = [0.1, 0.3, 0.8]
    yy = pdf.(Binomial(8, x), xx)
    plot!(xx, yy, lw=2, label=x, markershape=:auto,
        legendtitle="q", legend=:bottomright)
end
display(plt)

## #src
# ## 6.4 ロジスティック回帰とロジットリンク関数
#-
logistic(z) = 1 / (1 + exp(-z))

## #src
#-
z = -6:0.1:6
plot(z, logistic.(z), lw=2, title="Logistic function", 
    xlabel="Linear predictor", ylabel="Probability", label="")

## #src
# ### 6.4.2 パラメータ推定¶
# Juliaでは、Binomial分布の場合、応答変数はfractionにしなければならない
# https://github.com/JuliaStats/GLM.jl/issues/228#issuecomment-387340111
#-
df.yy = df.y ./ df.N
df.N = convert(Array{Float64}, df.N)
result = glm(@formula(yy ~ x + f), df, Binomial(), wts=df.N)

## #src
#-
dfc = df[df.f .== "C", :];
scatter(dfc.x, dfc.y, label="C")
xx = DataFrame(x=range(minimum(dfc.x), maximum(dfc.x), length=100), f="C")
yy = predict(result, xx) * 8.0
plot!(xx.x, yy, lw=2,label="",
    xlabel="Plant size", ylabel="Number of survival seeds")

## #src
#-
dft = df[df.f .== "T", :];
scatter(dft.x, dft.y, label="T", color=:red)
xx = DataFrame(x = range(minimum(dft.x), maximum(dft.x), length=100), f="T")
yy = predict(result, xx) * 8.0
plot!(xx.x, yy, lw=2, label="",
    xlabel="Plant size", ylabel="Number of survival seeds")

## #src
#-
typeof(result)

## #src
# ## 6.4.4 ロジスティック回帰のモデル選択
# $k, \log L^*, $deriance$ - 2\log L^*$, residual deviance, AIC
#-
function model_selection_table(result)
    dof(result), loglikelihood(result), -2loglikelihood(result), deviance(result), aic(result)
end

## #src
#-
const_model = glm(@formula(yy ~ 1), df, Binomial(), wts=df.N)
model_selection_table(const_model)

## #src
#-
f_model = glm(@formula(yy ~ 1 + f), df, Binomial(), wts=df.N)
model_selection_table(f_model)

## #src
#-
x_model = glm(@formula(yy ~ 1 + x), df, Binomial(), wts=df.N)
model_selection_table(x_model)

## #src
#-
xf_model = glm(@formula(yy ~ 1 + x + f), df, Binomial(), wts=df.N)
model_selection_table(xf_model)

## #src
# ## 6.5 交互作用項の入った線形予測子
#-
interaction_model = glm(@formula(yy ~ x + f + x * f), df, Binomial(), wts=df.N)

## #src
#-
model_selection_table(interaction_model)

## #src
# ## 6.6 割算値の統計モデリングはやめよう
# ### 6.6.1 割算値いらずのオフセット項わざ
#-
df_population = CSV.read(joinpath("..", "data/section6b.csv"))

## #src
#-
scatter(df_population.A, df_population.y, 
    markeralpha=df_population.x, label="",
    xlabel="Area", ylabel="Plant population")

## #src
#-
population_reseult = glm(@formula(y ~ x), df_population, GLM.Poisson(), offset=log.(df_population.A))

## #src
# ### 明るさごとの平均個体数の予測
#-
plt = scatter(df_population.A, df_population.y, 
    markeralpha=df_population.x, label="", 
    title="Prediction",
    xlabel="Area", ylabel="Plant population",
    legendtitle="Brightness", legend=:topleft)

for j = 0.1:0.2:0.9
    xx = DataFrame(A=range(minimum(df_population.A), maximum(df_population.A), length=100), x=j)
    yy = predict(population_reseult, xx, offset=log.(xx.A))
    plot!(xx.A, yy, lw=2, color=:red, linealpha=j, label=j)
end

display(plt)

## #src
# ## 6.7 正規分布とその尤度¶
# https://juliastats.github.io/Distributions.jl/latest/univariate.html#Distributions.Normal
# ### 確率密度関数のプロット
# オレンジの領域の面積は$1.2 \le y \le 1.8$となる確率を表す
#-
y = -5:0.1:5
fill_y = 1.2:0.1:1.8

plot(y, pdf.(Normal(), y), lw=2, label="", ylims=(0, 0.4),
    title=L"\mu=0, \sigma=1",
    ylabel="Probability density")

plot!(fill_y, pdf.(Normal(), fill_y),
    label="", fillrange=0, fillalpha=0.5, 
    linecolor=:transparent, fillcolor=:orange)

## #src
#-
plot(y, pdf.(Normal(0, 3), y), lw=2, label="", ylims=(0, 0.4),
    title=L"\mu=0, \sigma=3",
    ylabel="Probability density")

plot!(fill_y, pdf.(Normal(0, 3), fill_y), label="", fillrange=0, fillalpha=0.5, 
    linecolor=:transparent, fillcolor=:orange)

## #src
#-
plot(y, pdf.(Normal(2, 1), y), lw=2, label="", ylims = (0, 0.4),
    title = L"\mu=2, \sigma=1",
    ylabel = "Probability density")

plot!(fill_y, pdf.(Normal(2, 1), fill_y), label="", fillrange=0, fillalpha=0.5, 
    linecolor=:transparent, fillcolor=:orange)

## #src
# $p(1.2 \le y \le 1.8 | \mu, \sigma)$を評価する
#-
cdf(Normal(), 1.8) - cdf(Normal(), 1.2)

## #src
# 近似
#-
pdf.(Normal(), 1.5) * 0.6

## #src
# ## 6.8 ガンマ分布のGLM
# https://juliastats.github.io/Distributions.jl/latest/univariate.html#Distributions.Gamma
#
# Distribution.jlのパラメトライズ $$ f(x; \alpha, \theta) = \frac{x^{\alpha-1} e^{-x/\theta}}{\Gamma(\alpha) \theta^\alpha}, \quad x &gt; 0 $$ 
#-
d = CSV.read(joinpath("..", "data/section6c.csv"))

## #src
#-
d.logx = log.(d.x)
d

## #src
#-
scatter(d.x, d.y, label="",
    xlabel="Weight of leaf", ylabel="Weight of flower")

## #src
# ## 確率密度関数のプロット
# オレンジの領域の面積は$1.2 \le y \le 1.8$となる確率を表す
#-
y = 0:0.01:5
fill_y = 1.2:0.05:1.8

plot(y, pdf.(Gamma(1, 1), y), lw=2, label="", ylims=(0, 1.0),
    title=L"r=s=1",
    ylabel="Probability density")

plot!(fill_y, pdf.(Gamma(1, 1), fill_y),
    label="", fillrange=0, fillalpha=0.5, 
    linecolor=:transparent, fillcolor=:orange)

## #src
#-
plot(y, pdf.(Gamma(5, 1 / 5), y), lw=2, label="", ylims=(0, 1.0),
    title=L"r=s=5",
    ylabel="Probability density")

plot!(fill_y, pdf.(Gamma(5, 1 / 5), fill_y),
    label="", fillrange=0, fillalpha=0.5, 
    linecolor=:transparent, fillcolor=:orange)

## #src
#-
plot(y, pdf.(Gamma(0.1, 1 / 0.1), y), lw=2, label="", ylims=(0, 1.0),
    title=L"r=s=0.1",
    ylabel="Probability density")

plot!(fill_y, pdf.(Gamma(0.1, 1 / 0.1), fill_y),
    label="", fillrange=0, fillalpha=0.5, 
    linecolor=:transparent, fillcolor=:orange)

## #src
# ### GLM
#-
gamma_result = glm(@formula(y ~ logx), d, Gamma(), LogLink())

## #src
#-
mm = gamma_result.model

## #src
# 推定された結果、ガンマ分布を使って評価された50%と90%区間の予測を示す
#-
xx = DataFrame(x=0.01:0.01:0.8)
xx.logx = log.(xx.x)
mm_mean = predict(gamma_result, xx)

## #src
# Dispersion parameter
# https://juliastats.github.io/GLM.jl/stable/api/#GLM.dispersion
#-
mm_phi = GLM.dispersion(mm, true)

## #src
#-
mm_alpha = 1 / mm_phi
mm_theta = mm_mean .* mm_phi

## #src
#-
xs_gamma = Gamma.(mm_alpha, mm_theta)

## #src
#-
scatter(d.x, d.y, label="", legend=:topleft,
    xlabel="Weight of leaf", ylabel="Weight of flower")

g_quantile(r) = quantile.(xs_gamma, r)

plot!(xx.x, g_quantile(0.95), lw=0, 
    fillrange=g_quantile(0.05), fillalpha=0.3, fillcolor=:orange,
    label="90% Confidence interval")
plot!(xx.x, g_quantile(0.75), lw=0, fillcolor=:red,
    fillrange=g_quantile(0.25), fillalpha=0.3,
    label="50% Confidence interval")
plot!(xx.x, mm_mean, lw=2, label="Prediction", linecolor=:red)
plot!(xx.x, mean.(Gamma.(1 / 0.3, exp.([ones(80) xx.logx] * [-1, 0.7]) .* 0.3)), 
    lw=2, label="True mean", linestyle=:dash, linecolor=:blue)
plot!(xx.x, g_quantile(0.5), lw=2, label="50% Percentile", linecolor=:green)

