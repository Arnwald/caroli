# caroli — architecture and roadmap

## Crates

```
crates/
  caroli-lattice      Bravais lattices with a basis (1D/2D/3D), finite regions, slices
  caroli-model        tight-binding models: on-site and hopping matrices, site-dependent terms,
                      dense / Bloch Hamiltonians, transport blocks; BHZ Dirac matrices and presets
  caroli-green        steady state: lead modes and self-energies, recursive Green's function,
                      Caroli transmission
  caroli-disorder     reproducible random streams (xoshiro256**, gfortran-compatible seeding),
                      Anderson disorder, classical magnetic impurities
  caroli-keldysh      non-equilibrium steady state: lesser Green's function by lead injection,
                      bond currents, interface currents, occupations
  caroli-born         disorder-averaged Green's functions: self-consistent Born approximation
                      (mean field + cumulant vertices; Anderson and magnetic impurities)
  caroli-torus        Hamiltonians on periodic 2D supercells: k-space part through FFTs (rustfft),
                      local terms in real space
  caroli-kpm          kernel polynomial method: Chebyshev moments, Jackson kernel, densities
  caroli-dynamics     time evolution: split-operator propagation on tori
  caroli              facade: re-exports and prelude
  caroli-validation   (not published) comparisons with the thesis golden outputs
```

The dependencies only go downward: lattice ← model ← green ← caroli.

## Design choices

- **Linear algebra: faer** (pure Rust, no system LAPACK). Complex scalars are `faer::c64`
  (= `num_complex::Complex<f64>`), matrices `faer::Mat<c64>` (`CMat`).
- **Orbital layout** is fixed by the `Region`: site `i` occupies orbitals
  `offset(i) .. offset(i) + orbitals(i)`. `Region::cuboid` orders the sites with the first axis
  varying slowest, so the slices along x are contiguous.
- **Model** = on-site matrix per sublattice + hoppings `⟨(R, a)| H |(R + offset, b)⟩` (their
  conjugates are implied) + site-dependent terms (`OnSite` trait: disorder, potentials).
  The Bloch convention is `H(k) = Σ_n H_{0n} e^{i k·n}`.
- **BHZ conventions** follow the thesis codes (2014+): basis `(E1↑, H1↑, E1↓, H1↓)`, Γ matrices
  of `numbers.f90`. Exchange coupling: `GFactors::Same` (g_E = g_H, `n·s`) and `Opposite`
  (g_E = −g_H, `n·s τ_z`). The thesis calls these "ferro" and "antiferro".
- **Lead self-energies by mode matching** (`caroli_green::modes`). The Bloch factors λ come from
  the linearised quadratic eigenproblem. The modes are split into right- and left-going ones
  (decaying, or propagating with the sign of the velocity, which is diagonalised within each
  degenerate group). Then `Σ_R = H1 F₊` and `Σ_L = H1† F₋⁻¹`.
  - The calculation is exact at real energy; no `η` is needed.
  - Each mode is stored as `(ψ_j, ψ_{j+1})`, so singular couplings (λ = 0 or ∞) are handled.
  - The generalised eigenproblem is solved through a Möbius shift `(A − σB)⁻¹B`, because faer's
    QZ is about 150× slower than its standard eigensolver at these sizes.
  - Decimation (Lopez Sancho – Rubio) is kept as a cross-check only. It loses all accuracy when
    E is close to an eigenvalue of the lead cell: T of a clean chain at E = 1e-9 is off by 0.5
    with η = 1e-12.
- **Dev builds** optimise dependencies (`[profile.dev.package."*"] opt-level = 3`); faer is
  unusable unoptimised.

## Validation

`reference/baseline/` compiles the thesis Fortran and records 26 golden cases. The
`caroli-validation` tests compare against them.

| test | golden case | status |
|---|---|---|
| 6×6 flake spectrum | `ti-diag` | equal to 1e-10 |
| ribbon bands with exchange J_s | `ti-ribbon-bands` | equal to 1e-10 |
| clean ribbon conductance | `ti-cond2t-clean` | caroli gives exact integers; the thesis is off by ≤ 1 % (Re Γ truncation) |
| conductance with magnetic impurities (same-g along z, isotropic, opposite-g) | `ti-cond2t-same-g`, `-isotropic`, `-opposite-g` | thesis formula (Re Γ, Σ at E + iη) reproduced to 1e-13 with the same impurity configurations; the exact transmission differs by up to 0.22 e²/h |
| g2g1 conductance maps (400×64) | thesis data, 50 realisations | caroli's averages agree within statistical error; Re Γ truncation negligible at this width (`examples/gamma_impact.rs`) |
| Keldysh bond currents and occupied densities (same-g, opposite-g, Anderson) | `ti-lcur-same-g`, `-opposite-g`, `-anderson` | impurity positions identical; thesis currents reproduced to 1e-17 and densities to 1e-14 (thesis conventions: element-wise products, slots) |
| SCBA: Anderson, magnetic matrix (same-g, opposite-g), magnetic reduced (same-g, opposite-g) | `ti-scba-*` | reproduced to 1e-15 with the thesis iteration schedules |
| four-terminal cross with Anderson disorder (spin-up BHZ block) | `ti-cond4t-anderson` | thesis formula reproduced to 3e-5 (the thesis puts E + iη on one slice) with a general multi-terminal solver |
| KPM densities of states (TI same-g, opposite-g; graphene AF) | `ti-dos-kpm-*`, `graphene-dos-kpm` | reproduced to 1e-15 (same random vectors and impurities) |
| graphene under a field: current and pair creation (clean, AF, para) | `graphene-current-*` | reproduced to 1e-16 (replica keeps gfortran's −0 imaginary parts, which fix the gauge of the initial band state) |
| graphene wave packet: width w(t), sublattice density maps | `graphene-width`, `graphene-density` | reproduced to 1e-16 (width) and 1e-19 (densities) |
| 2013 ribbon bands, Fourier in y (λ = −0.1) | `ti-ribbon-bands-fourier-y` | equal to 2e-14; the 2013 Γ14 is anti-Hermitian and `heev` reads its upper triangle (edge states move by ≤ 1.5e-3) |
| 2013 recursive "spin DOS" | `ti-dos-spin-lesser` | replayed to 7e-11; the program never forms G^< and its up/down sums are a split by position |
| 2014 manuscript code: G, LDOS, bond currents with Anderson disorder | `ti-cond-ldos-current` | reproduced to 1e-14 (its leads sit at −ev while the boundary slices carry +ev) |
| QSH plateau, modes vs decimation, T_LR = T_RL, current conservation = Landauer | — | internal consistency |

Golden cases with disorder use the same impurity configurations as the thesis.
`caroli_disorder::rng::Xoshiro256::gfortran` reproduces gfortran's `random_seed(put=…)` and
`random_number` bit for bit, and the disorder generators follow the thesis draw order.
`caroli_validation::thesis` rebuilds each thesis setup.

**Thesis bug found this way:** all conductance codes use `real(IM*(Σ − Σ†))` for the broadening
(see `reference/thesis/README.md`).

## Roadmap

1. ✅ Baseline: thesis codes built, 26 golden cases.
2. ✅ Lattice, model (BHZ), Green's functions (modes, RGF, Caroli transmission), validated on
   spectra and clean conductance.
3. ✅ Disorder (`caroli-disorder`): Anderson; classical magnetic impurities (isotropic, cone θ0,
   polarisation M_z) with same-g / opposite-g coupling; reproducible random streams. Validated on
   `ti-cond2t-*` (exact).
4. ✅ Multi-terminal transport (`caroli_green::MultiTerminal`: any geometry, any number of leads,
   direct solve), validated on `ti-cond4t-anderson`. A ring recursion for large multi-terminal
   devices is still to do.
5. ✅ Keldysh (`caroli-keldysh`): G^< by lead injection, bond currents, occupied LDOS, bias
   windows. Validated on `ti-lcur-*` (exact).
6. ✅ Self-consistent Born approximation (`caroli-born`), validated on `ti-scba-*` (exact).
   The reduced (analytic) thesis SCBA differs from the general one by the sign of the mass
   self-energy (see `reference/thesis/ti/born/scba-magnetic-analytic/README.md`).
7. ✅ Spectral tools: KPM (Chebyshev moments, Jackson kernel, stochastic trace) on tori
   (`caroli-torus`); validated on the TI and graphene KPM codes (exact).
8. ✅ Time evolution: split-operator on tori (`caroli-dynamics`); graphene current, pair
   creation, wave-packet width and density validated (exact).
9. Interface: declarative system and simulation files with a CLI; Python bindings to consider.
10. Acceleration:
    - CPU parallelism (rayon) over energies and samples first.
    - Then GPU kernels for sparse mat-vec (KPM, time evolution) and batched dense solves.
    - Apple GPUs have no f64, so Metal kernels must be f32 (or emulated double-single), which
      suits KPM and time evolution better than transmissions.
    - CUDA allows f64.
11. Examples: the two thesis studies (TI with magnetic impurities: G maps, local currents, Born
    phase diagram; graphene under a field: current, pair creation, DOS).
