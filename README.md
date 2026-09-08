# NN-FIKH — code and data for *"Scientific machine learning for modeling complex fluids with thixotropy and plasticity"*

J. D. J. Rathinaraj, K. R. Lennon & G. H. McKinley (PNAS Nexus).

This repository contains the NN-FIKH constitutive model (**N**eural
**N**etwork **F**ractional **I**sotropic **K**inematic **H**ardening),
example trained networks, and the experimental rheological data for the
3.5 wt.% Laponite dispersion studied in the paper, together with a script
that regenerates the model figures of the paper and a training script for
applying the framework to your own data.

## Quick start

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # first time only
julia --project=. reproduce_figures.jl
```

This regenerates the model figures of the paper (Figs. 1, 3–6 and S4) into
`figures/reproduced/` in about 5–10 minutes; the header of
`reproduce_figures.jl` maps each output file to its figure in the paper.

## Using NN-FIKH on your own data — step by step

The neural network in this framework does not replace the physics — it
learns targeted corrections *inside* a classical constitutive model. So the
workflow has two parts: first establish the analytic FIKH description of
your material (classical rheology), then let the provided script train the
neural corrections. In detail:

**1. Characterize your material with three standard experiments.**

   * a small-amplitude (SAOS) frequency sweep — G′ and G″ vs ω;
   * a steady flow curve — stress vs shear rate;
   * a set of large-amplitude oscillatory (LAOS) tests spanning a grid of
     strain amplitudes and frequencies — this is what the network learns from.

**2. Put the oscillatory data in the repository's format.** One CSV per
   test, columns `time [s], strain [%], stress [Pa], (unused), shear
   rate [1/s]`, files named `1.csv … M.csv` (see `data/DATA_DICTIONARY.md`;
   the Laponite files in `data/laos/` are a working example). Then point the
   code at your folder in `src/nnfikh_model.jl`:

   ```julia
   const DATADIR = Ref("data/my_material/")
   const N_train = 12          # your number of files
   ```

**3. Fit the eleven FIKH parameters — the genuinely scientific step.**
   Estimate them hierarchically, each from the experiment that constrains
   it, as described in the paper:

   | from | you obtain |
   |---|---|
   | SAOS sweep | G, 𝕍, α (fractional viscoelastic backbone) |
   | steady flow curve | plastic-flow and yield parameters (𝕂, n, σ_p⁰, C/q, k₋τ_thix) |
   | oscillatory data | hardening and thixotropy parameters (m, q, C, k₋, τ_thix) |

   Enter your values in the labeled `Ref` slots at the top of
   `src/nnfikh_model.jl` (the defaults are the paper's Laponite values):

   ```julia
   const GG = Ref(380.0)     # ← your elastic modulus G [Pa]
   const VV = Ref(4616.0)    # ← your quasiproperty 𝕍 [Pa·sᵅ]
   ...
   ```

   This step matters: the network only corrects the evolution of the two
   internal variables, so it refines a good base model — it cannot rescue a
   poor one.

**4. Choose the training protocols.** Set `TRAIN_PROTS` at the top of
   `train_nnfikh.jl` to a representative subset of your file indices,
   spanning the amplitude–frequency grid and favoring tests that actually
   yield (purely linear SAOS records carry no signal for the network).

**5. Verify the pipeline (minutes).**

   ```bash
   julia --project=. train_nnfikh.jl cal
   ```

   This runs one forward solve and one gradient and reports the timing. If
   it succeeds, your data format, your parameters, and the training
   machinery all work — any problem surfaces here, not hours into a run.

**6. Train (typically overnight).**

   ```bash
   julia --project=. train_nnfikh.jl run 8.0     # wall-clock budget in hours
   ```

   The network starts from ≈zero output — exactly your analytic FIKH model —
   and ADAM learns corrections to the back-strain and structure evolution
   equations, in the elastic limit of the framework with a reverse-mode
   adjoint. The best weights are checkpointed continuously to
   `Example trained models/my_model.jld2`, so nothing is lost if the run is
   interrupted. Expect roughly 15–20 iterations per hour on a laptop.

**7. Evaluate.**

   ```bash
   julia --project=. train_nnfikh.jl eval
   ```

   This reports the improvement of your NN-FIKH model over your FIKH
   baseline in the full fractional model and saves a model-vs-data grid
   figure. The saved weights use the same key (`"θi"`) the plotting driver
   loads, so `reproduce_figures.jl` can plot your model by changing one
   filename.

Two notes on scope: training assumes sinusoidal strain input
(γ = γ₀ sin ωt, inferred from each record); evaluating the trained model
against non-sinusoidal, measured-rate protocols is supported
(`solve_swan_data`), but training on such histories would require adapting
`elastic_oop` in `train_nnfikh.jl`. And the material should be one the FIKH
structure suits — a thixotropic elasto-visco-plastic fluid reasonably
described by fractional viscoelasticity, a yield process, kinematic
hardening, and a single structure parameter.

## Repository structure

```
├── reproduce_figures.jl      driver — regenerates the model figures
├── train_nnfikh.jl           train the correction network (paper data or your own)
├── src/
│   └── nnfikh_model.jl       the NN-FIKH model: equations, data loaders, solvers
├── Example trained models/
│   ├── example_model_1.jld2  trained network (default in the driver)
│   ├── example_model_2.jld2  independently trained network
│   └── MODEL_CARD.md         architecture, inputs/outputs, how to load
├── data/
│   ├── laos/                 22 oscillatory records: 18 training + 4 test (ω = 3 rad/s)
│   ├── swan/                 3 SWAN sawtooth records (γ̇₀ = ±0.38, ±38, ±380 s⁻¹)
│   ├── saos_moduli.csv       SAOS frequency sweep (G′, G″) — Fig. 1(a)
│   ├── flow_curve_steady.csv steady flow curve — Figs. 1(b), 3(c), 4(c)
│   └── DATA_DICTIONARY.md    per-file protocol table and column definitions
└── Project.toml              Julia dependencies
```

## Trained models

`Example trained models/` provides two independently trained networks with
identical architecture. Both closely reproduce the NN-FIKH results of the
paper, and their predictions nearly coincide — comparing them illustrates
how tightly the embedded physics constrains the learned correction. The
driver loads `example_model_1.jld2` by default; to use the other, change the filename on
the `load(...)` line of `reproduce_figures.jl`. Details in
`Example trained models/MODEL_CARD.md`.

## The model in brief

The FIKH framework (fractional Maxwell viscoelasticity + isotropic/kinematic
hardening + thixotropic structure kinetics) is augmented with a small
fully-connected network (5→14→14→14→3, tanh, 549 parameters) whose outputs
enter only the evolution equations of the back strain A(t) and the structure
parameter λ(t) as bounded corrective terms. The 11 physical parameters are
held fixed; the network is trained on oscillatory data only, and the same
network, with no retraining, predicts the sawtooth SWAN protocol. Setting
`NN_ON[] = false` in `src/nnfikh_model.jl` switches the neural terms off,
recovering the analytic FIKH baseline — every figure comparison uses the
same fixed parameter set for both models. The stiff fractional ODE system
(Caputo derivative of order 1−α = 0.67) is integrated with TRBDF2.

## Note on the viscoelastic quasiproperty $`\mathbb{V}`$

All simulations in the paper and in this repository — the FIKH baseline and
the NN-FIKH model alike — use one fixed parameter set:

```math
\{\mathbb{G},\ \mathbb{V},\ \alpha,\ \mathbb{K},\ n,\ 1/\tau_{\mathrm{thix}},\ k_-,\ \sigma_p^0,\ C,\ q,\ m\}
= \{380\ \mathrm{Pa},\ 4616\ \mathrm{Pa\,s}^{\alpha},\ 0.33,\ 4.66\ \mathrm{Pa\,s}^{n},\ 0.34,\ 0.08\ \mathrm{s}^{-1},\ 0.047,\ 15\ \mathrm{Pa},\ 88\ \mathrm{Pa},\ 3.67,\ 0.49\}.
```

Because every improvement reported in the paper is a relative comparison
between the two models at this shared, fixed parameter set, the results and
conclusions do not depend on this choice.

For completeness: a transcription error occurred during the sequential
parameter estimation. The value of $`\mathbb{V}`$ that best fits the
small-amplitude (SAOS) moduli of Fig. 1(a) — in particular the loss modulus
$`G''`$ — is $`\mathbb{V} \approx 1616\ \mathrm{Pa\,s}^{\alpha}`$
($`\tau_c \approx 80`$ s), while $`\mathbb{V} = 4616\ \mathrm{Pa\,s}^{\alpha}`$
($`\tau_c \approx 1933`$ s) was carried into all subsequent dynamic
simulations. This is why the $`G''`$ curve produced by
`reproduce_figures.jl` for Fig. 1(a) lies below the data. Users who prefer
the SAOS-optimal value can set `VV[] = 1616.0` in `src/nnfikh_model.jl` and
retrain the network (the trained networks provided here correspond to
$`\mathbb{V} = 4616\ \mathrm{Pa\,s}^{\alpha}`$). Comparable relative
improvements of NN-FIKH over FIKH are expected, since the learned
corrections adapt to whichever base parameter set is held fixed.

## Citation

If you use this code or data, please cite:

> Rathinaraj, J. D. J., Lennon, K. R. & McKinley, G. H. Scientific machine
> learning for modeling complex fluids with thixotropy and plasticity.
> *PNAS Nexus* (in press).
