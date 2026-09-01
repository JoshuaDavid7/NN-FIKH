# Data dictionary

All rheological measurements were performed on a 3.5 wt.% aqueous Laponite RD
dispersion with an ARES-G2 strain-controlled rheometer (TA Instruments), DIN
concentric-cylinder geometry (27.7 mm inner rotor). Before every test the
sample was presheared at 500 s⁻¹ for 300 s followed by a 180 s rest, erasing
prior microstructural orientation and stress (see *Materials and Methods* in
the paper).

## `laos/` — oscillatory shear records (γ(t) = γ₀ sin(ωt), 9+ cycles)

CSV columns (no header):
`time [s], strain [%], shear stress [Pa], (unused), shear rate [1/s]`

| file | γ₀ [%] | ω [rad/s] | role |
|------|-------:|----------:|------|
| 1.csv  |   70 | 5 | training |
| 2.csv  |  150 | 5 | training |
| 3.csv  | 1000 | 1 | training |
| 4.csv  |  300 | 1 | training |
| 5.csv  |    3 | 9 | training |
| 6.csv  |  300 | 5 | training |
| 7.csv  |  300 | 9 | training |
| 8.csv  |  500 | 1 | training |
| 9.csv  |  150 | 1 | training |
| 10.csv |  500 | 5 | training |
| 11.csv |  500 | 9 | training |
| 12.csv |    3 | 5 | training |
| 13.csv | 1000 | 5 | training |
| 14.csv |    3 | 1 | training |
| 15.csv |   70 | 1 | training |
| 16.csv | 1000 | 9 | training |
| 17.csv |  150 | 9 | training |
| 18.csv |   70 | 9 | training |
| 19.csv |    3 | 3 | **test** (Figs. 3(b)/4(b)) |
| 21.csv |  150 | 3 | **test** (Figs. 3(b)/4(b)) |
| 22.csv |  300 | 3 | **test** (Figs. 3(b)/4(b)) |
| 24.csv | 1000 | 3 | **test** (Figs. 3(b)/4(b)) |

Files 1–18 form the 6 × 3 Pipkin training grid
(γ₀ ∈ {3, 70, 150, 300, 500, 1000} %, ω ∈ {1, 5, 9} rad/s) shown in
Figs. 3(a)/4(a). Files 19–24 are the held-out test set at the interleaved
frequency ω = 3 rad/s. (File numbers 20 and 23 were repeat measurements of
training conditions not used in the paper and are not included.)

## `swan/` — SWAN sawtooth (triangular-strain) protocol, ω₀ = 1 rad/s

Same column layout as `laos/`. The measured shear-rate signal (column 5) is
used to drive the model; the measured stress (column 3) is the comparison
target (Figs. 5 and S4).

| file | rate amplitude γ̇₀ [1/s] |
|------|------------------------:|
| swan_gdot_0.38.csv | ±0.38 |
| swan_gdot_38.csv   | ±38   |
| swan_gdot_380.csv  | ±380  |

## Characterization data (Fig. 1, and Figs. 3(c)/4(c))

| file | contents |
|------|----------|
| `saos_moduli.csv` | small-amplitude oscillatory frequency sweep: `omega_rad_per_s, G_prime_Pa, G_doubleprime_Pa` (11 points, ω = 0.1–10 rad/s). Used to fit the fractional Maxwell gel parameters {G, 𝕍, α}. |
| `flow_curve_steady.csv` | steady-shear flow curve: `shear_rate_1_per_s, stress_Pa` (21 points, γ̇ = 0.1–1000 1/s). Used to fit {n, 𝕂, σ_p⁰}, C/q and k₋τ_thix, and compared against both models in Figs. 3(c)/4(c). |

These two tables were recovered point-for-point from the original MATLAB
figure sources for the published Figs. 1(a) and 1(b).
