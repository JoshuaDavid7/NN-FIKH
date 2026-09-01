# Model card — trained NN-FIKH networks

## `twocycle_5input_run52.jld2` — **the network of the paper**

This is the single trained network used for **every** NN-FIKH result in the
manuscript (Figs. 4, 5, 6 and S4): one network, no retraining between the
oscillatory and SWAN protocols.

| property | value |
|---|---|
| architecture | `Dense(5→14, tanh) → Dense(14→14, tanh) → Dense(14→14, tanh) → Dense(14→3)` |
| parameters | 549 (Float32) |
| inputs `U(t)` | `[ \|A\|, \|γ̇_p\|, \|σ\|, λ, \|γ̇\| ]` (absolute values; directionality is carried by the analytic sign terms) |
| outputs | `NN_A`, `NN_λ⁻`, `NN_λ⁺` — corrective terms in the evolution equations (9a)/(9b) for the back strain A and structure parameter λ |
| file key | `"θi"` (a `ComponentArray`-compatible named tuple) |
| training | ADAM, elastic-limit approximation (σ ≈ G·γ_ve) of the FIKH framework, oscillatory training data of Figs. 3(a)/3(c); evaluated in the full fractional model |

Load with:
```julia
using JLD2, ComponentArrays
θ = ComponentArray(load("Example trained models/twocycle_5input_run52.jld2", "θi"))
```

The 11 physical FIKH parameters are **not** stored in this file — they are
fixed constants in `src/nnfikh_model.jl`:
{G, 𝕍, α, 𝕂, n, 1/τ_thix, k₋, σ_p⁰, C, q, m} =
{380 Pa, 4616 Pa·sᵅ, 0.33, 4.66 Pa·sⁿ, 0.34, 0.08 s⁻¹, 0.047, 15 Pa, 88 Pa,
3.67, 0.49}, exactly as reported in the paper.

## `twocycle_5input_run36.jld2` — independently trained comparison network

A second network trained independently with the same architecture and recipe.
Its stress predictions are indistinguishable from run52 (differences < 1 Pa).
It is **not** used in any figure of the paper; it is included solely because
the response to the reviewers (Reviewer 1, Comment 6) uses it to demonstrate
how tightly the inferred internal variables A(t), λ(t) are constrained across
independently trained networks. For all reproduction purposes, use run52.
