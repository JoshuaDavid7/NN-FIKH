# =============================================================================
#  train_nnfikh.jl — train the NN-FIKH neural network on oscillatory data.
#
#  Trains the 549-parameter correction network from random initialization
#  using the recipe of the paper: training is performed in the ELASTIC LIMIT
#  of the framework (σ ≈ G·γ_ve, no fractional memory term — cheap, and
#  differentiable end-to-end with a reverse-mode adjoint), and the result is
#  then EVALUATED in the full fractional model. The network starts at ≈0
#  output, i.e. at the analytic FIKH model, and ADAM discovers corrections.
#
#  Usage (from this folder; see also "Using NN-FIKH on your own data" in the
#  README — in particular, set the material parameters in src/nnfikh_model.jl
#  to values fitted for YOUR material before training on your own data):
#
#    julia --project=. train_nnfikh.jl cal
#        Time one forward solve + one gradient, and estimate iterations/hour.
#        Run this first — it verifies the whole pipeline in a few minutes.
#
#    julia --project=. train_nnfikh.jl run <hours>
#        Train for the given wall-clock budget (e.g. 2.0). Checkpoints the
#        best weights continuously to "Example trained models/my_model.jld2"
#        (key "θi" — directly loadable by reproduce_figures.jl).
#
#    julia --project=. train_nnfikh.jl eval
#        Evaluate my_model.jld2 against the analytic FIKH baseline in the
#        full fractional model and save a fit-grid figure.
#
#  Configuration (edit below): TRAIN_PROTS, N_CYC, LR, SEED, OUTFILE.
# =============================================================================
cd(@__DIR__)
include("src/nnfikh_model.jl")
using SciMLSensitivity, Zygote
using Printf

# ------------------------------- configuration ------------------------------
# Protocol indices (files DATADIR/<i>.csv) to train on. Prefer a representative
# subset spanning your amplitude-frequency grid; protocols that never yield
# (SAOS) carry no signal for the network and slow training down.
const TRAIN_PROTS = [15, 4, 8, 3, 1, 10, 13, 16]
const N_CYC   = 9.0        # number of cycles of each record to fit
const LR      = 0.01       # ADAM learning rate
const SEED    = 2024       # random seed for the initial weights
const OUTFILE = joinpath("Example trained models", "my_model.jld2")
const TR_TOL  = (1e-5, 1e-4)   # abstol/reltol for training solves (looser = faster)

# ------------------- elastic-limit model (adjoint-friendly) ------------------
# Out-of-place right-hand side over the reduced state [γ_p, A, λ]; the stress
# is algebraic, σ = G(γ − γ_p). Reads the same material parameters (matpars)
# and the same network as the full model in src/nnfikh_model.jl.
const u0_el  = [0.0, 0.0, 1.0]
function elastic_oop(u, p, t, ampfreq)
    γp = u[1]; A = u[2]; λ = u[3]
    G, V, k, n, k1, k2, C, q, m, k3 = matpars()
    γ0 = ampfreq[1]; w = round(ampfreq[2])
    γ = γ0*sin(w*t); gd = γ0*w*cos(w*t)
    σ = G*(γ - γp)
    yld = abs(σ - C*A) - k3*λ
    dγp = yld < 0 ? zero(σ) : sign(σ - C*A)*((yld/k))^(1/n)
    inl = [abs(A), abs(dγp), abs(σ), abs(λ), abs(gd)]
    nn  = NN_ON[] ? nn_lambda(inl, p, st1)[1] : (zero(σ), zero(σ), zero(σ))
    d1 = dγp
    d2 = dγp - q^m*(A^2)^(m/2)*abs(dγp)*sign(A) + nn[1]*abs(dγp)*sign(A)
    d3 = (k1 + nn[3])*(1 - λ) - k2*λ*abs(dγp) - nn[2]*abs(dγp)*λ
    [d1, d2, d3]
end

# data interpolants for the loss integrand
const DFUN = Vector{Any}(undef, N_train)
function build_dfun!()
    for i in 1:N_train
        tend = 2*pi/round(protocols[i,2])*N_CYCLES[]
        DFUN[i] = linear_interpolation(collect(LinRange(0.0, tend, N)), stress[i,:],
                                       extrapolation_bc=Line())
    end
end

# augmented state: 4th component accumulates ∫(σ−data)² dt ⇒ loss = final state
const u0_aug = [0.0, 0.0, 1.0, 0.0]
function aug_oop(u, p, t, amp, df)
    d3 = elastic_oop(u[1:3], p, t, amp)
    G = matpars()[1]; γ0 = amp[1]; w = round(amp[2])
    σ = G*(γ0*sin(w*t) - u[1])
    err = σ - Zygote.dropgrad(df(t))
    vcat(d3, err*err)
end

function loss_one(θ, i)
    tend = 2*pi/round(protocols[i,2])*N_CYCLES[]
    df = DFUN[i]
    f(u,p,t) = aug_oop(u, p, t, protocols[i,:], df)
    prob = ODEProblem(f, u0_aug, (0.0, tend), θ)
    sol = solve(prob, TRBDF2(autodiff=false),
                sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false)),
                abstol=TR_TOL[1], reltol=TR_TOL[2])
    sol.u[end][4]
end
loss_rev(θ) = sum(loss_one(θ, i) for i in TRAIN_PROTS)

function init_params(seed)
    ps, _ = Lux.setup(StableRNG(seed), nn_lambda)
    θ = ComponentArray(ps)
    θ.layer_4.weight .*= 0.01f0   # output ≈ 0 ⇒ training starts at analytic FIKH
    θ.layer_4.bias   .*= 0.0f0
    Float64.(θ)
end

# ------------------------- ADAM with checkpointing ---------------------------
function train(θ0, budget_s)
    ax = getaxes(θ0); θv = Float64.(getdata(θ0))
    lossv(x) = loss_rev(ComponentArray(x, ax))
    m = zero(θv); v = zero(θv); b1 = 0.9; b2 = 0.999; ϵ = 1e-8
    t0 = time(); it = 0; best = Inf; bestv = copy(θv); stale = 0; patience = 40
    while time() - t0 < budget_s && stale < patience
        it += 1
        L, back = Zygote.pullback(lossv, θv)
        g = back(1.0)[1]
        @. m = b1*m + (1-b1)*g
        @. v = b2*v + (1-b2)*g*g
        @. θv = θv - LR*(m/(1-b1^it))/(sqrt(v/(1-b2^it)) + ϵ)
        if L < best - 1e-7
            best = L; bestv = copy(θv); stale = 0
            save(OUTFILE, "θi", ComponentArray(bestv, ax))
        else
            stale += 1
        end
        (it % 5 == 0 || it == 1) &&
            @printf("it=%d  loss=%.5f  best=%.5f  t=%ds\n", it, L, best, round(Int, time()-t0))
        flush(stdout)
    end
    @printf("finished: %d iterations, best elastic-limit loss = %.5f\n", it, best)
    println("best weights saved to ", OUTFILE)
    ComponentArray(bestv, ax)
end

# ----------------------- evaluation in the full model ------------------------
function eval_full(θ)
    N_CYCLES[] = N_CYC; load_data!()
    NN_ON[] = true;  Lnn, _ = total_loss(θ)
    NN_ON[] = false; Lfk, _ = total_loss(θ); NN_ON[] = true
    @printf("full-model oscillatory SSE:  NN-FIKH = %.4f   FIKH = %.4f   improvement = %.1f%%\n",
            Lnn, Lfk, 100*(Lfk - Lnn)/Lfk)
    mkpath("figures")
    plot_grid(θ, "figures/training_fit_grid.png")
end

# --------------------------------- modes -------------------------------------
mode = length(ARGS) >= 1 ? ARGS[1] : "cal"

if mode == "cal"
    N_CYCLES[] = N_CYC; load_data!(); build_dfun!()
    θ = init_params(SEED); ax = getaxes(θ); θv = Float64.(getdata(θ))
    lossv(x) = loss_rev(ComponentArray(x, ax))
    t1 = time(); L0 = lossv(θv)
    @printf("forward loss = %.4f   (%.1f s)\n", L0, time()-t1)
    t2 = time(); L, back = Zygote.pullback(lossv, θv); g = back(1.0)[1]; dg = time()-t2
    @printf("one gradient  = %.1f s   |g| = %.4f   => ~%d iterations/hour\n",
            dg, sqrt(sum(abs2, g)), round(Int, 3600/dg))
    println("CAL DONE — pipeline works; now run:  julia --project=. train_nnfikh.jl run <hours>")

elseif mode == "run"
    hours = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 2.0
    N_CYCLES[] = N_CYC; load_data!(); build_dfun!()
    println("training for $(hours) h on protocols $(TRAIN_PROTS) ...")
    θ = train(init_params(SEED), hours*3600)
    eval_full(θ)

elseif mode == "eval"
    θ = ComponentArray(load(OUTFILE, "θi"))
    eval_full(θ)
else
    error("unknown mode '$mode' — use cal | run <hours> | eval")
end
