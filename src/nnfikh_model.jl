# nnfikh_model.jl — the NN-FIKH constitutive model, data loading, and solvers.
#
# Self-contained model library for the paper
#   "Scientific machine learning for modeling complex fluids with thixotropy
#    and plasticity" (Rathinaraj, Lennon & McKinley, PNAS Nexus).
# It LOADS trained weights (no training) and provides the ODE right-hand sides,
# data loaders, and solve routines used by reproduce_figures.jl.
#
# States: u = [γ_p, γ_ve, σ, A, λ]  (plastic strain, viscoelastic strain,
#          stress, back strain, structure parameter); stresses internally
#          normalized by gm = G = 380 Pa.
using OrdinaryDiffEq
using LinearAlgebra, Statistics, DelimitedFiles
using FractionalCalculus
using JLD2
using ComponentArrays, Lux, StableRNGs, NNlib
using Interpolations
using Plots; gr()

const tm = 1
const gm = 380
rng = StableRNG(1111)
u0 = [0.0, 0.0, 0.0, 0.0, 1.0]
const NN_ON = Ref(true)   # false => analytic FIKH (neural terms off) = Fig. 3 baseline
const VV = Ref(4616.0)    # 𝕍 quasiproperty [Pa·s^α] (τ_c = (𝕍/G)^(1/α) = 1933 s) — used for ALL
                          # simulations in the paper; see "Note on the viscoelastic quasiproperty 𝕍" in README.md
const DATADIR = Ref("data/laos/")  # folder holding the LAOS csv files 1.csv … 24.csv
const N_CYCLES = Ref(9.0) # number of oscillation cycles used for data window + model solve (training/eval)
const K1 = Ref(0.08)      # 1/τ_thix [s⁻¹]; k_- = 0.59*K1 = 0.047 (as in the paper)

nn_lambda = Lux.Chain(Lux.Dense(5, 14, tanh), Lux.Dense(14, 14, tanh),
                      Lux.Dense(14, 14, tanh), Lux.Dense(14, 3))
_p0, st1 = Lux.setup(rng, nn_lambda)

function ude_mikh!(du, u, p, t, ampfreq)
    gp, gv, σ, A, λ = u
    G = 380/gm; V = VV[]/(gm*tm); k = 4.66/(gm*tm); n = 0.34
    k1 = K1[]*tm; k2 = 0.59*k1; C = 88/gm; q = C/24*gm; m = 0.49; k3 = 15/gm
    if abs(σ - C*A) < k3*λ
        du[1] = 0.0
    else
        du[1] = sign(σ - C*A)*(((abs(σ - C*A) - k3*λ)/k))^(1/n)
    end
    gdot(t) = ampfreq[1]*round(ampfreq[2])*cos(round(ampfreq[2])*t)
    inputs_l = [abs(A), abs(du[1]), abs(σ), abs(λ), abs(gdot(t))]
    nn_l = NN_ON[] ? nn_lambda(inputs_l, p, st1)[1] : (0.0, 0.0, 0.0)
    du[2] = gdot(t) - du[1]
    du[3] = -G/V*fracdiff(σ, 0.67, t, 0.000001, Caputo_Piecewise()) + G*du[2]
    du[4] = du[1] - q^m*(A^2)^(m/2)*abs(du[1])*sign(A) + nn_l[1]*abs(du[1])*sign(A)
    du[5] = (k1 + nn_l[3])*(1 - λ) - k2*λ*abs(du[1]) - nn_l[2]*abs(du[1])*λ
end

const N = 1000
const N_train = 18
strain = zeros(N_train, N); stress = zeros(N_train, N); rate = zeros(N_train, N)
freq = zeros(N_train); strain_amp = zeros(N_train); rate_amp = zeros(N_train)
protocols = zeros(N_train, 2)

function load_data!()
    for i in 1:N_train
        data = readdlm(joinpath(DATADIR[], string(i)*".csv"), ',')
        s1 = data[1,1]
        if s1 isa AbstractString   # strip a possible UTF-8 BOM on the first cell
            mtc = match(r"[-+0-9.eE].*", s1)
            data[1,1] = parse(Float64, mtc === nothing ? s1 : mtc.match)
        end
        t = LinRange(0.0, 60.0003, N)/tm
        il = linear_interpolation(data[:,1]/tm, data[:,2]/100, extrapolation_bc=Line()); strain[i,:] = il(t)
        il = linear_interpolation(data[:,1]/tm, data[:,3]/gm, extrapolation_bc=Line()); stress[i,:] = il(t)
        il = linear_interpolation(data[:,1]/tm, data[:,5], extrapolation_bc=Line()); rate[i,:] = il(t)
        freq[i] = maximum(rate[i,:])/maximum(strain[i,:]); strain_amp[i] = maximum(strain[i,:]); rate_amp[i] = strain_amp[i]*freq[i]
        protocols[i,:] = [strain_amp[i], freq[i]]
        t = LinRange(0.0, 2*pi/freq[i]*N_CYCLES[], N)/tm
        il = linear_interpolation(data[:,1]/tm, data[:,2]/100, extrapolation_bc=Line()); strain[i,:] = il(t)
        il = linear_interpolation(data[:,1]/tm, data[:,3]/gm, extrapolation_bc=Line()); stress[i,:] = il(t)
        il = linear_interpolation(data[:,1]/tm, data[:,5], extrapolation_bc=Line()); rate[i,:] = il(t)
        freq[i] = maximum(rate[i,:])/maximum(strain[i,:]); strain_amp[i] = maximum(strain[i,:]); rate_amp[i] = strain_amp[i]*freq[i]
        protocols[i,:] = [strain_amp[i], freq[i]]
    end
end

function solve_one(p, i)
    tend = 2*pi/round(protocols[i,2])*N_CYCLES[]
    f!(du,u,p,t) = ude_mikh!(du,u,p,t,protocols[i,:])
    prob = ODEProblem(f!, u0, [0.0, tend], p)
    solve(prob, TRBDF2(), saveat=LinRange(0.0, tend, N))
end

function total_loss(p)
    L = 0.0; ok = 0
    for i in 1:N_train
        try
            sol = solve_one(p, i)
            if length(sol.t) == N
                pred = [sol.u[j][3] for j in 1:N]
                L += sum(abs2, stress[i,:] .- pred); ok += 1
            else
                L += Inf
            end
        catch
            L += Inf
        end
    end
    L, ok
end

function plot_grid(p, fname)
    plt = plot(layout=(6,3), size=(1000,1800), legend=false, grid=false, axis=false)
    for i in 1:N_train
        sol = solve_one(p, i)
        pred = [sol.u[j][3] for j in 1:length(sol.t)]*gm
        scatter!(plt, strain[i,:], stress[i,:]*gm, subplot=i, mc=:red, ms=2, markerstrokecolor=:red)
        plot!(plt, strain[i,1:length(pred)], pred, subplot=i, lw=4, lc=:black)
    end
    savefig(plt, fname); println("saved figure: ", fname)
end

# ---- SWAN (sawtooth) reproduction: rate-driven model (teady form from nnikh.jl) ----
swan_rate(t) = (mod(t, 2*pi) < pi ? 1.0 : -1.0)   # ω0=1 rad/s square-wave rate (triangle strain)
function teady_mikh!(du, u, p, t, p_true)
    gp, gv, σ, A, λ = u
    G = 380/gm; V = VV[]/(gm*tm); k = 4.66/(gm*tm); n = 0.34
    k1 = K1[]*tm; k2 = 0.59*k1; C = 88/gm; q = C/24*gm; m = 0.49; k3 = 15/gm
    if abs(σ - C*A) < k3*λ
        du[1] = 0.0
    else
        du[1] = sign(σ - C*A)*(((abs(σ - C*A) - k3*λ)/k))^(1/n)
    end
    gdot(t) = p_true*swan_rate(t)
    inputs_l = [abs(A), abs(du[1]), abs(σ), abs(λ), abs(gdot(t))]
    nn_l = NN_ON[] ? nn_lambda(inputs_l, p, st1)[1] : (0.0, 0.0, 0.0)
    du[2] = gdot(t) - du[1]
    du[3] = -G/V*fracdiff(σ, 0.67, t, 0.000001, Caputo_Piecewise()) + G*du[2]
    du[4] = du[1] - q^m*(A^2)^(m/2)*abs(du[1])*sign(A) + nn_l[1]*abs(du[1])*sign(A)
    du[5] = (k1 + nn_l[3])*(1 - λ) - k2*λ*abs(du[1]) - nn_l[2]*abs(du[1])*λ
end
function solve_swan(p, gdot0; tend=60.0)
    f!(du,u,p,t) = teady_mikh!(du,u,p,t,gdot0)
    prob = ODEProblem(f!, u0, [0.0, tend], p)
    solve(prob, TRBDF2(), saveat=LinRange(0.0, tend, 2000), tstops=collect(pi:pi:tend))
end

# ---- Fig. 4(e) time series + Fig. 3 (FIKH=green) helpers ----
function plot_grid2(p, fname, col)
    plt = plot(layout=(6,3), size=(1000,1800), legend=false, grid=false, axis=false)
    for i in 1:N_train
        sol = solve_one(p, i)
        pred = [sol.u[j][3] for j in 1:length(sol.t)]*gm
        scatter!(plt, strain[i,:], stress[i,:]*gm, subplot=i, mc=:red, ms=2, markerstrokecolor=:red)
        plot!(plt, strain[i,1:length(pred)], pred, subplot=i, lw=4, lc=col)
    end
    savefig(plt, fname); println("saved figure: ", fname)
end
function fig4e(p, i, fname)
    tend = 2*pi/round(protocols[i,2])*N_CYCLES[]
    NN_ON[] = true;  s1 = solve_one(p, i); σ_nn = [s1.u[j][3] for j in 1:length(s1.t)]*gm
    NN_ON[] = false; s2 = solve_one(p, i); σ_f  = [s2.u[j][3] for j in 1:length(s2.t)]*gm
    NN_ON[] = true
    plt = plot(size=(950,420), xlab="t [s]", ylab="σ [Pa]",
               title="γ0=$(round(Int,strain_amp[i]*100))%, ω=$(round(Int,protocols[i,2])) rad/s  (cf Fig 4e)", legend=:topright)
    scatter!(plt, LinRange(0,tend,N), stress[i,:]*gm, mc=:red, ms=3, markerstrokecolor=:red, label="data", ma=0.6)
    plot!(plt, s2.t, σ_f, lw=2, lc=:green, label="FIKH (NN off) — Fig 3")
    plot!(plt, s1.t, σ_nn, lw=2.5, lc=:black, label="NN-FIKH — Fig 4")
    savefig(plt, fname); println("saved figure: ", fname)
    println("  data peak/trough = ", round(maximum(stress[i,:]*gm),digits=1), " / ", round(minimum(stress[i,:]*gm),digits=1), " Pa")
    println("  NN-FIKH peak/trough = ", round(maximum(σ_nn),digits=1), " / ", round(minimum(σ_nn),digits=1), " Pa")
    println("  FIKH    peak/trough = ", round(maximum(σ_f),digits=1), " / ", round(minimum(σ_f),digits=1), " Pa")
end

# ---- Fig. 5 robust: drive the model with the REAL measured rheometer rate signal ----
gdot_data = Ref{Any}(nothing)
function teady_data!(du, u, p, t, p_true)
    gp, gv, σ, A, λ = u
    G = 380/gm; V = VV[]/(gm*tm); k = 4.66/(gm*tm); n = 0.34
    k1 = K1[]*tm; k2 = 0.59*k1; C = 88/gm; q = C/24*gm; m = 0.49; k3 = 15/gm
    if abs(σ - C*A) < k3*λ
        du[1] = 0.0
    else
        du[1] = sign(σ - C*A)*(((abs(σ - C*A) - k3*λ)/k))^(1/n)
    end
    gdot(t) = p_true*gdot_data[](t)
    inputs_l = [abs(A), abs(du[1]), abs(σ), abs(λ), abs(gdot(t))]
    nn_l = NN_ON[] ? nn_lambda(inputs_l, p, st1)[1] : (0.0, 0.0, 0.0)
    du[2] = gdot(t) - du[1]
    du[3] = -G/V*fracdiff(σ, 0.67, t, 0.000001, Caputo_Piecewise()) + G*du[2]
    du[4] = du[1] - q^m*(A^2)^(m/2)*abs(du[1])*sign(A) + nn_l[1]*abs(du[1])*sign(A)
    du[5] = (k1 + nn_l[3])*(1 - λ) - k2*λ*abs(du[1]) - nn_l[2]*abs(du[1])*λ
end
function solve_swan_data(p, csvfile)
    d = readdlm(csvfile, ',')
    if d[1,1] isa AbstractString
        mtc = match(r"[-+0-9.eE].*", d[1,1]); d[1,1] = parse(Float64, mtc===nothing ? d[1,1] : mtc.match)
    end
    tt = Float64.(d[:,1]); rr = Float64.(d[:,5]); ss = Float64.(d[:,3])
    gdot_data[] = linear_interpolation(tt, rr, extrapolation_bc=Line())
    sgn = [rr[i]*rr[i-1] < 0 ? tt[i] : NaN for i in 2:length(tt)]; tstops = filter(!isnan, sgn)
    f!(du,u,p,t) = teady_data!(du,u,p,t,1.0)
    prob = ODEProblem(f!, u0, [tt[1], tt[end]], p)
    sol = solve(prob, TRBDF2(), saveat=LinRange(tt[1], tt[end], 3000), tstops=tstops)
    tt, ss, sol.t, [sol.u[j][3] for j in 1:length(sol.t)]*gm
end

# ---- per-cycle error decomposition (how well each cycle is fit) ----
function cycle_errors(p, i; ncyc::Int=9)
    sol = solve_one(p, i)
    pred = [sol.u[j][3] for j in 1:length(sol.t)]
    per = zeros(ncyc); pts = div(N, ncyc)
    for c in 1:ncyc
        a = (c-1)*pts + 1; b = c==ncyc ? N : c*pts
        per[c] = sum(abs2, stress[i, a:b] .- pred[a:b])
    end
    per
end

# ---- generic protocol loader (any csv by number, e.g. the ω=3 test set 19,21,22,24) ----
function load_protocol(i)
    data = readdlm(joinpath(DATADIR[], string(i)*".csv"), ',')
    s1 = data[1,1]
    if s1 isa AbstractString
        mtc = match(r"[-+0-9.eE].*", s1)
        data[1,1] = parse(Float64, mtc === nothing ? s1 : mtc.match)
    end
    t0 = LinRange(0.0, 60.0003, N)/tm
    il = linear_interpolation(data[:,1]/tm, data[:,5], extrapolation_bc=Line())
    r0 = il(t0)
    il = linear_interpolation(data[:,1]/tm, data[:,2]/100, extrapolation_bc=Line())
    s0 = il(t0)
    ω  = maximum(abs.(r0))/maximum(abs.(s0)); γ0 = maximum(abs.(s0))
    t  = LinRange(0.0, 2*pi/ω*N_CYCLES[], N)/tm
    il = linear_interpolation(data[:,1]/tm, data[:,2]/100, extrapolation_bc=Line()); γ = il(t)
    il = linear_interpolation(data[:,1]/tm, data[:,3]/gm,  extrapolation_bc=Line()); σ = il(t)
    (collect(t), γ, σ, γ0, ω)
end

function solve_protocol(p, γ0, ω)
    tend = 2*pi/round(ω)*N_CYCLES[]
    f!(du,u,p,t) = ude_mikh!(du,u,p,t,[γ0,ω])
    prob = ODEProblem(f!, u0, [0.0, tend], p)
    solve(prob, TRBDF2(), saveat=LinRange(0.0, tend, N))
end

# ---- constant-rate startup → steady state (flow curve, Figs 1(b), 3(c), 4(c)) ----
function steady_mikh!(du, u, p, t, rate)
    gp, gv, σ, A, λ = u
    G = 380/gm; V = VV[]/(gm*tm); k = 4.66/(gm*tm); n = 0.34
    k1 = K1[]*tm; k2 = 0.59*k1; C = 88/gm; q = C/24*gm; m = 0.49; k3 = 15/gm
    if abs(σ - C*A) < k3*λ
        du[1] = 0.0
    else
        du[1] = sign(σ - C*A)*(((abs(σ - C*A) - k3*λ)/k))^(1/n)
    end
    inputs_l = [abs(A), abs(du[1]), abs(σ), abs(λ), abs(rate)]
    nn_l = NN_ON[] ? nn_lambda(inputs_l, p, st1)[1] : (0.0, 0.0, 0.0)
    du[2] = rate - du[1]
    du[3] = -G/V*fracdiff(σ, 0.67, t, 0.000001, Caputo_Piecewise()) + G*du[2]
    du[4] = du[1] - q^m*(A^2)^(m/2)*abs(du[1])*sign(A) + nn_l[1]*abs(du[1])*sign(A)
    du[5] = (k1 + nn_l[3])*(1 - λ) - k2*λ*abs(du[1]) - nn_l[2]*abs(du[1])*λ
end
function solve_steady_stress(p, rate; tend=60.0)
    f!(du,u,p,t) = steady_mikh!(du,u,p,t,rate)
    prob = ODEProblem(f!, u0, [0.0, tend], p)
    sol = solve(prob, TRBDF2(), saveat=[0.0, tend])
    sol.u[end][3]*gm
end
