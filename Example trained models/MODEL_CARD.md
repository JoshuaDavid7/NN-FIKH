# Example trained models

Two independently trained NN-FIKH networks with identical architecture. Both
closely reproduce the NN-FIKH results of the paper (Figs. 4, 5, 6 and S4),
and their predictions nearly coincide (stress differences of order 1 Pa) —
comparing them illustrates how tightly the embedded physics constrains the
learned correction. Use whichever you prefer; `reproduce_figures.jl` loads
`example_model_1.jld2` by default.

| property | value |
|---|---|
| architecture | `Dense(5→14, tanh) → Dense(14→14, tanh) → Dense(14→14, tanh) → Dense(14→3)` |
| parameters | 549 (Float32) |
| inputs `U(t)` | `[ \|A\|, \|γ̇_p\|, \|σ\|, λ, \|γ̇\| ]` (absolute values; directionality is carried by the analytic sign terms) |
| outputs | `NN_A`, `NN_λ⁻`, `NN_λ⁺` — corrective terms in the evolution equations (9a)/(9b) for the back strain A and structure parameter λ |
| file key | `"θi"` (a `ComponentArray`-compatible named tuple) |
| training | ADAM, elastic-limit approximation (σ ≈ G·γ_ve) of the FIKH framework, oscillatory training data only; evaluated in the full fractional model |

Load with:

```julia
using JLD2, ComponentArrays
θ = ComponentArray(load("Example trained models/example_model_1.jld2", "θi"))
```

The 11 physical FIKH parameters are **not** stored in these files — they are
configurable `Ref` values at the top of `src/nnfikh_model.jl`, defaulting to
the paper's Laponite set:
{G, 𝕍, α, 𝕂, n, 1/τ_thix, k₋, σ_p⁰, C, q, m} =
{380 Pa, 4616 Pa·sᵅ, 0.33, 4.66 Pa·sⁿ, 0.34, 0.08 s⁻¹, 0.047, 15 Pa, 88 Pa,
3.67, 0.49}, as reported in the paper.
