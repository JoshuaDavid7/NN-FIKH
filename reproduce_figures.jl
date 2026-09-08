# =============================================================================
#  reproduce_figures.jl — regenerate the model figures of
#  "Scientific machine learning for modeling complex fluids with thixotropy
#   and plasticity"  (Rathinaraj, Lennon & McKinley, PNAS Nexus).
#
#  Usage:  julia reproduce_figures.jl          (from this folder; Julia 1.8)
#
#  Produces, in figures/reproduced/:
#    fig1a_saos_moduli.png            Fig. 1(a)  SAOS moduli + fractional Maxwell gel fit
#    fig1b_flow_curve_HB.png          Fig. 1(b)  steady flow curve + Herschel–Bulkley fit
#    fig3a_pipkin_FIKH.png            Fig. 3(a)  FIKH fits, 18-protocol Pipkin grid
#    fig3b_test_FIKH.png              Fig. 3(b)  FIKH predictions, ω = 3 rad/s test set
#    fig3c4c_flow_curve_models.png    Figs. 3(c)/4(c)  flow curve: data vs FIKH vs NN-FIKH
#    fig3d4d_error_bars.png           Figs. 3(d)/4(d)  first/ninth-cycle error residuals
#    fig4a_pipkin_NNFIKH.png          Fig. 4(a)  NN-FIKH fits, Pipkin grid
#    fig4b_test_NNFIKH.png            Fig. 4(b)  NN-FIKH predictions, test set
#    fig4e_timeseries.png             Fig. 4(e)  σ(t) at γ0=500%, ω=5 rad/s
#    fig5_swan_timeseries.png         Fig. 5(d,f,h)  SWAN stress responses
#    fig5_swan_lissajous.png          Fig. 5(e,g,i)  SWAN Lissajous curves
#    fig6_internal_variables.png      Fig. 6  λ(t) and A(t), FIKH vs NN-FIKH
#    figS4_swan_FIKH_vs_NNFIKH.png    Fig. S4  SWAN: FIKH vs NN-FIKH vs data
#
#  Loads Example trained models/twocycle_5input_run52.jld2 by default; the
#  same network is used for every figure ("one network, no retraining").
#  See Example trained models/MODEL_CARD.md to swap in the other network.
#  Runtime: ~5–10 min (mostly Julia compilation; solves take under a minute).
# =============================================================================
cd(@__DIR__)
include("src/nnfikh_model.jl")
using Printf

const OUT = "figures/reproduced/"
mkpath(OUT)
θ = ComponentArray(load("Example trained models/twocycle_5input_run52.jld2", "θi"))
load_data!()
t0_all = time()
stamp() = @sprintf("[%6.0f s]", time() - t0_all)

# ---------------------------------------------------------------- Fig. 1(a,b)
println("$(stamp()) Fig 1(a): SAOS moduli + FMG fit"); flush(stdout)
sa = readdlm("data/saos_moduli.csv", ',', skipstart=1)
# FMG model curve drawn with the paper's global parameter set (G = 380 Pa,
# 𝕍 = 4616 Pa·s^α, α = 0.33) — the same values used in every simulation in
# this repository. With this parameter set the single-mode FMG underestimates
# G″ at these frequencies; G′ (which dominates for this gel-like material,
# tan δ < 0.12) is captured well. See the note on 𝕍 in README.md.
G0 = 380.0; α = 0.33; τc = (VV[]/G0)^(1/α)
ωf = 10 .^ range(-1.2, 1.2, length=200)
Gstar = [G0*(im*w*τc)^α/(1+(im*w*τc)^α) for w in ωf]
plt = plot(size=(700,470), xaxis=:log10, yaxis=:log10, xlabel="ω [rad/s]",
           ylabel="G′, G″ [Pa]", legend=:bottomright, title="cf Fig 1(a)")
scatter!(plt, sa[:,1], sa[:,2], mc=:red, ms=6, msc=:red, label="G′ data")
scatter!(plt, sa[:,1], sa[:,3], mc=:white, ms=6, msc=:red, label="G″ data")
plot!(plt, ωf, real.(Gstar), lc=:black, lw=2, label="G′ FMG (G=380, 𝕍=4616, α=0.33)")
plot!(plt, ωf, imag.(Gstar), lc=:black, ls=:dash, lw=2, label="G″ FMG")
savefig(plt, OUT*"fig1a_saos_moduli.png")

println("$(stamp()) Fig 1(b): flow curve + Herschel–Bulkley fit"); flush(stdout)
fcd = readdlm("data/flow_curve_steady.csv", ',', skipstart=1)
γ̇f = 10 .^ range(-1.1, 3.1, length=200)
σHB = 38.0 .+ 0.36 .* γ̇f .^ 0.67          # σ_y = 38 Pa, K_HB = 0.36 Pa·s^n, n = 0.67 (Fig 1b)
plt = plot(size=(650,450), xaxis=:log10, xlabel="γ̇ [1/s]", ylabel="σ [Pa]",
           legend=:topleft, title="cf Fig 1(b)")
scatter!(plt, fcd[:,1], fcd[:,2], mc=:red, ms=6, msc=:red, label="steady flow data")
plot!(plt, γ̇f, σHB, lc=:crimson, lw=2, label="Herschel–Bulkley fit")
savefig(plt, OUT*"fig1b_flow_curve_HB.png")

# ------------------------------------------------- solve all protocols once/model
# training protocols 1–18 (files data/laos/1..18.csv), test set 19,21,22,24 (ω = 3)
const TEST_IDS = [19, 21, 22, 24]
test_data = Dict{Int,Any}()
for i in TEST_IDS
    test_data[i] = load_protocol(i)     # (t, γ(t), σ(t)/gm, γ0, ω)
end

sols = Dict{Bool,Dict{String,Any}}()
for nn_on in (false, true)
    NN_ON[] = nn_on
    tag = nn_on ? "NN-FIKH" : "FIKH"
    println("$(stamp()) solving 18 training + 4 test protocols  [$tag]"); flush(stdout)
    d = Dict{String,Any}()
    d["train"] = [solve_one(θ, i) for i in 1:N_train]
    d["test"]  = Dict(i => solve_protocol(θ, test_data[i][4], test_data[i][5]) for i in TEST_IDS)
    sols[nn_on] = d
end
NN_ON[] = true

# ------------------------------------------------------------- Fig. 3(a)/4(a)
println("$(stamp()) Figs 3(a)/4(a): Pipkin grids"); flush(stdout)
for (nn_on, fname, col) in ((false,"fig3a_pipkin_FIKH.png",:green), (true,"fig4a_pipkin_NNFIKH.png",:black))
    local plt = plot(layout=(6,3), size=(1000,1800), legend=false, grid=false, axis=false)
    for i in 1:N_train
        sol = sols[nn_on]["train"][i]
        pred = [sol.u[j][3] for j in 1:length(sol.t)]*gm
        scatter!(plt, strain[i,:], stress[i,:]*gm, subplot=i, mc=:red, ms=2, markerstrokecolor=:red)
        plot!(plt, strain[i,1:length(pred)], pred, subplot=i, lw=4, lc=col)
    end
    savefig(plt, OUT*fname); println("  saved ", fname)
end

# ------------------------------------------------------------- Fig. 3(b)/4(b)
println("$(stamp()) Figs 3(b)/4(b): ω = 3 rad/s test predictions"); flush(stdout)
for (nn_on, fname, col) in ((false,"fig3b_test_FIKH.png",:green), (true,"fig4b_test_NNFIKH.png",:black))
    local plt = plot(layout=(1,4), size=(1500,400), legend=false, grid=false, axis=false)
    for (k,i) in enumerate(TEST_IDS)
        t, γ, σd, γ0, ω = test_data[i]
        sol = sols[nn_on]["test"][i]
        pred = [sol.u[j][3] for j in 1:length(sol.t)]*gm
        tγ = γ0 .* sin.(round(ω) .* sol.t)
        scatter!(plt, γ, σd*gm, subplot=k, mc=:blue, ms=2, markerstrokecolor=:blue)
        plot!(plt, tγ[1:length(pred)], pred, subplot=k, lw=3.5, lc=col)
    end
    savefig(plt, OUT*fname); println("  saved ", fname)
end

# ------------------------------------------------------------- Fig. 3(d)/4(d)
println("$(stamp()) Figs 3(d)/4(d): per-cycle error residuals"); flush(stdout)
# ε_c: mean-square stress residual of cycle c, normalized by (C/q)² = 24².
# (This normalization reproduces the magnitudes plotted in the published bars;
#  the three trivial SAOS training records are excluded from the training average.)
norm2 = 24.0^2
function cyc_eps(sol, σdat)
    pred = [sol.u[j][3] for j in 1:length(sol.t)]*gm
    dat  = σdat*gm
    pts  = div(N, 9)
    ε(c) = sum(abs2, dat[(c-1)*pts+1:(c==9 ? N : c*pts)] .- pred[(c-1)*pts+1:(c==9 ? N : c*pts)]) /
           (c==9 ? N-8*pts : pts) / norm2
    (ε(1), ε(9))
end
eps = Dict{Bool,Dict{String,Tuple{Float64,Float64}}}()
for nn_on in (false, true)
    tr1 = Float64[]; tr9 = Float64[]; te1 = Float64[]; te9 = Float64[]
    for i in 1:N_train
        strain_amp[i] < 0.05 && continue      # exclude the three trivial SAOS records
        e1, e9 = cyc_eps(sols[nn_on]["train"][i], stress[i,:]); push!(tr1,e1); push!(tr9,e9)
    end
    for i in TEST_IDS
        e1, e9 = cyc_eps(sols[nn_on]["test"][i], test_data[i][3]); push!(te1,e1); push!(te9,e9)
    end
    eps[nn_on] = Dict("train"=>(sum(tr1)/length(tr1), sum(tr9)/length(tr9)),
                      "test" =>(sum(te1)/length(te1), sum(te9)/length(te9)))
end
@printf("  FIKH    ε1/ε9 train = %.4f / %.4f   test = %.4f / %.4f\n",
        eps[false]["train"]..., eps[false]["test"]...)
@printf("  NN-FIKH ε1/ε9 train = %.4f / %.4f   test = %.4f / %.4f\n",
        eps[true]["train"]...,  eps[true]["test"]...)
groups = ["Training ε₁","Training ε₉","Testing ε₁","Testing ε₉"]
fik = [eps[false]["train"][1], eps[false]["train"][2], eps[false]["test"][1], eps[false]["test"][2]]
nnf = [eps[true]["train"][1],  eps[true]["train"][2],  eps[true]["test"][1],  eps[true]["test"][2]]
x = collect(1:4)
plt = plot(size=(750,450), xticks=(x, groups), ylabel="ε (per cycle, ÷(C/q)²)",
           legend=:topright, title="cf Figs 3(d)/4(d)")
bar!(plt, x .- 0.17, fik, bar_width=0.3, fillcolor=:seagreen, label="FIKH")
bar!(plt, x .+ 0.17, nnf, bar_width=0.3, fillcolor=:black, label="NN-FIKH")
savefig(plt, OUT*"fig3d4d_error_bars.png")

# ------------------------------------------------------------------ Fig. 4(e)
println("$(stamp()) Fig 4(e): time series γ0=500%, ω=5"); flush(stdout)
i4e = findfirst(k -> abs(strain_amp[k]-5.0) < 0.2 && abs(protocols[k,2]-5.0) < 0.2, 1:N_train)
let s1 = sols[true]["train"][i4e], s2 = sols[false]["train"][i4e]
    σnn = [s1.u[j][3] for j in 1:length(s1.t)]*gm
    σf  = [s2.u[j][3] for j in 1:length(s2.t)]*gm
    tend = 2*pi/round(protocols[i4e,2])*N_CYCLES[]
    plt = plot(size=(950,420), xlab="t [s]", ylab="σ [Pa]",
               title="γ0=500%, ω=5 rad/s (cf Fig 4e)", legend=:topright)
    scatter!(plt, LinRange(0,tend,N), stress[i4e,:]*gm, mc=:red, ms=3, msc=:red, label="data", ma=0.6)
    plot!(plt, s2.t, σf, lw=2, lc=:green, label="FIKH")
    plot!(plt, s1.t, σnn, lw=2.5, lc=:black, label="NN-FIKH")
    savefig(plt, OUT*"fig4e_timeseries.png")
    @printf("  first-cycle peak: data %.1f | NN-FIKH %.1f | FIKH %.1f Pa\n",
            maximum(stress[i4e,:]*gm), maximum(σnn), maximum(σf))
end

# --------------------------------------------------------------------- Fig. 6
println("$(stamp()) Fig 6: internal variables λ(t), A(t)"); flush(stdout)
i5   = findfirst(k -> abs(strain_amp[k]-5.0)  < 0.2 && abs(protocols[k,2]-5.0) < 0.2, 1:N_train)
i003 = findfirst(k -> strain_amp[k] < 0.05        && abs(protocols[k,2]-5.0) < 0.2, 1:N_train)
plt = plot(layout=(1,2), size=(1100,420))
for (nn_on, col) in ((false,:green), (true,:black))
    for (ip, ls) in ((i5,:solid), (i003,:dot))
        sol = sols[nn_on]["train"][ip]
        λt = [sol.u[j][5] for j in 1:length(sol.t)]
        At = [sol.u[j][4] for j in 1:length(sol.t)]
        lbl = (nn_on ? "NN-FIKH" : "FIKH")*" (γ0=$(round(strain_amp[ip],digits=2)))"
        plot!(plt, sol.t, λt, subplot=1, lc=col, ls=ls, lw=2, label=lbl)
        plot!(plt, sol.t, At, subplot=2, lc=col, ls=ls, lw=2, label="")
    end
end
plot!(plt, subplot=1, xlabel="t [s]", ylabel="λ", title="cf Fig 6(a)", legend=:topright)
plot!(plt, subplot=2, xlabel="t [s]", ylabel="A", title="cf Fig 6(b)")
savefig(plt, OUT*"fig6_internal_variables.png")

# ------------------------------------------------------- Fig. 3(c)/4(c) flow curve
println("$(stamp()) Figs 3(c)/4(c): steady flow curve (42 fractional solves — slow)"); flush(stdout)
rates = fcd[:,1]
σ_fikh = zeros(length(rates)); σ_nn = zeros(length(rates))
for (k, r) in enumerate(rates)
    NN_ON[] = false; σ_fikh[k] = solve_steady_stress(θ, r)
    NN_ON[] = true;  σ_nn[k]   = solve_steady_stress(θ, r)
    @printf("  rate %8.3g 1/s:  data %.1f | FIKH %.1f | NN-FIKH %.1f Pa\n",
            r, fcd[k,2], σ_fikh[k], σ_nn[k]); flush(stdout)
end
plt = plot(size=(700,470), xaxis=:log10, xlabel="γ̇ [1/s]", ylabel="σ [Pa]",
           legend=:topleft, title="cf Figs 3(c)/4(c)")
scatter!(plt, rates, fcd[:,2], mc=:red, ms=6, msc=:red, label="data")
plot!(plt, rates, σ_fikh, lc=:green, lw=2, marker=:circle, ms=3, label="FIKH")
scatter!(plt, rates, σ_nn, mc=:black, ms=5, msc=:black, marker=:square, label="NN-FIKH")
savefig(plt, OUT*"fig3c4c_flow_curve_models.png")

# ------------------------------------------------------------- Fig. 5 and S4
println("$(stamp()) Figs 5 & S4: SWAN protocol (6 fractional solves — slow)"); flush(stdout)
swan_files = [("data/swan/swan_gdot_0.38.csv","0.38"), ("data/swan/swan_gdot_38.csv","38"),
              ("data/swan/swan_gdot_380.csv","380")]
swan = Dict{String,Any}()
for (f, lab) in swan_files
    NN_ON[] = true;  td, sd, tm_, snn = solve_swan_data(θ, f)
    NN_ON[] = false; _,  _,  tmf, sf  = solve_swan_data(θ, f)
    NN_ON[] = true
    d = readdlm(f, ','); if d[1,1] isa AbstractString
        m = match(r"[-+0-9.eE].*", d[1,1]); d[1,1] = parse(Float64, m===nothing ? d[1,1] : m.match) end
    swan[lab] = (td=td, sd=sd, γd=Float64.(d[:,2])./100, tm=tm_, snn=snn, tmf=tmf, sf=sf)
    println("  γ̇0 = ±$lab 1/s done"); flush(stdout)
end
# Fig 5(d,f,h): NN-FIKH vs data time series
plt = plot(layout=(3,1), size=(950,950))
for (j,(f,lab)) in enumerate(swan_files)
    s = swan[lab]
    scatter!(plt, s.td, s.sd, subplot=j, mc=:blue, ms=1.5, ma=0.4, msc=:blue, label="data")
    plot!(plt, s.tm, s.snn, subplot=j, lw=2, lc=:black, label="NN-FIKH",
          title="SWAN  γ̇0 = ±$lab s⁻¹  (cf Fig 5)", titlefontsize=9, xlab="t [s]", ylab="σ [Pa]",
          legend=(j==1 ? :topright : false))
end
savefig(plt, OUT*"fig5_swan_timeseries.png")
# Fig 5(e,g,i): Lissajous σ vs γ
plt = plot(layout=(1,3), size=(1400,430))
for (j,(f,lab)) in enumerate(swan_files)
    s = swan[lab]
    iσ = linear_interpolation(s.tm, s.snn, extrapolation_bc=Line())
    scatter!(plt, s.γd, s.sd, subplot=j, mc=:blue, ms=1.5, ma=0.4, msc=:blue, label="data")
    plot!(plt, s.γd, iσ.(s.td), subplot=j, lw=1.5, lc=:black, label="NN-FIKH",
          title="γ̇0 = ±$lab s⁻¹", xlab="γ", ylab="σ [Pa]", legend=(j==1 ? :topright : false))
end
savefig(plt, OUT*"fig5_swan_lissajous.png")
# Fig S4: FIKH vs NN-FIKH vs data (full record + startup zoom)
plt = plot(layout=(3,2), size=(1150,1050))
for (j,(f,lab)) in enumerate(swan_files)
    s = swan[lab]; lc_ = 2j-1; rc = 2j
    scatter!(plt, s.td, s.sd, subplot=lc_, mc=:dodgerblue, ms=1.5, ma=0.25, msc=:dodgerblue, label="experiment")
    plot!(plt, s.tmf, s.sf, subplot=lc_, lc=RGB(0.85,0.15,0.15), lw=1.3, ls=:dash, label="FIKH")
    plot!(plt, s.tm, s.snn, subplot=lc_, lc=:black, lw=1.3, label="NN-FIKH",
          title="γ̇0 = ±$lab s⁻¹ — full record", titlefontsize=9, ylab="σ [Pa]",
          legend=(j==1 ? :topright : false))
    j==3 && plot!(plt, subplot=lc_, xlab="t [s]")
    zz = s.td .<= 14; zm = s.tm .<= 14; zf = s.tmf .<= 14
    scatter!(plt, s.td[zz], s.sd[zz], subplot=rc, mc=:dodgerblue, ms=3, ma=0.5, msc=:dodgerblue, label="")
    plot!(plt, s.tmf[zf], s.sf[zf], subplot=rc, lc=RGB(0.85,0.15,0.15), lw=2, ls=:dash, label="")
    plot!(plt, s.tm[zm], s.snn[zm], subplot=rc, lc=:black, lw=2, label="",
          title="startup (first ~2 cycles)", titlefontsize=9)
    j==3 && plot!(plt, subplot=rc, xlab="t [s]")
end
savefig(plt, OUT*"figS4_swan_FIKH_vs_NNFIKH.png")

println("$(stamp()) ALL FIGURES DONE — outputs in $(OUT)")
