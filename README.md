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

Total runtime is roughly 5–10 minutes (mostly package
loading and compilation). The script writes the panels of Figs. 1, 3, 4, 5,
6 and S4 to `figures/reproduced/` — the header of `reproduce_figures.jl`
lists the output-file-to-figure mapping — and prints key checkpoints as it
runs, e.g. the first-cycle stress peak at γ₀ = 500%, ω = 5 rad/s
(data ≈ 72 Pa, NN-FIKH ≈ 71 Pa, FIKH ≈ 51 Pa) and the SWAN overshoot at
γ̇₀ = ±38 s⁻¹ (data ≈ 77 Pa, NN-FIKH ≈ 74 Pa, FIKH ≈ 53 Pa).

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

## Using NN-FIKH on your own data

The repository is set up so the framework can be applied to other materials,
not only to reproduce the paper:

1. **Format your data.** Export your oscillatory tests as CSV files with the
   column layout in `data/DATA_DICTIONARY.md`
   (`time [s], strain [%], stress [Pa], (unused), shear rate [1/s]`), name
   them `1.csv … M.csv`, point `DATADIR[]` in `src/nnfikh_model.jl` at their
   folder, and set `N_train = M`.

2. **Fit your material's parameters.** The eleven FIKH parameters live at the
   top of `src/nnfikh_model.jl` as documented `Ref` values (`GG`, `VV`,
   `ALFA`, `KK`, `NEXP`, `K1`, `KRAT`, `SP0`, `CC`, `CQ`, `MM`), currently
   set to the Laponite values of the paper. Estimate them for your material
   the way the paper does: {G, 𝕍, α} from a SAOS frequency sweep, the
   plastic-flow and yield parameters from a steady flow curve, and the
   hardening/thixotropy parameters from the oscillatory data.

3. **Train the network:**
   ```bash
   julia --project=. train_nnfikh.jl cal          # verify the pipeline, time one gradient
   julia --project=. train_nnfikh.jl run 2.0      # train for 2 h (checkpoints continuously)
   julia --project=. train_nnfikh.jl eval         # compare against the FIKH baseline
   ```
   Training follows the paper's recipe: the network starts at zero output
   (the analytic FIKH model) and ADAM learns corrections in the elastic
   limit of the framework, with a reverse-mode adjoint through the stiff
   solver; the result is evaluated in the full fractional model. Best
   weights are saved to `Example trained models/my_model.jld2` with the
   same key (`"θi"`) the driver loads, so pointing `reproduce_figures.jl`
   at your file plots your model with no other changes. Pick the training
   protocols (`TRAIN_PROTS` at the top of `train_nnfikh.jl`) to span your
   amplitude-frequency grid, favoring records that actually yield.

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
