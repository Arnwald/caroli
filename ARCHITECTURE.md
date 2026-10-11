# caroli — architecture and roadmap

## Crates

```
crates/
  caroli-lattice      Bravais lattices with a basis (1D/2D/3D), finite regions, slices
  caroli-model        tight-binding models: on-site and hopping matrices, site-dependent terms,
                      dense / Bloch Hamiltonians, transport blocks; BHZ Dirac matrices and presets
  caroli-green        steady state: lead modes and self-energies, recursive Green's function,
                      Caroli transmission
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
| clean ribbon conductance | `ti-cond2t-clean` | caroli gives exact integers; within 0.025 of the thesis, whose clean-wire G deviates by up to ~1 % |
| QSH plateau, modes vs decimation | — | internal consistency |

Golden cases with disorder need the same impurity configuration. Two routes are possible:
- use the configurations written by the thesis programs (`imp_XX.dat`, local-current codes);
- reproduce gfortran's generator (xoshiro256\*\* with `random_seed(put=…)` scrambling) in a test
  helper.

## Roadmap

1. ✅ Baseline: thesis codes built, 26 golden cases.
2. ✅ Lattice, model (BHZ), Green's functions (modes, RGF, Caroli transmission), validated on
   spectra and clean conductance.
3. Disorder (`caroli-disorder`): Anderson; classical magnetic impurities (isotropic, cone θ0,
   polarisation M_z) with same-g / opposite-g coupling; reproducible random streams. Validate on
   `ti-cond2t-*`.
4. Multi-terminal transport (4 leads, transmissions T_ij), validated on `ti-cond4t-anderson`.
5. Keldysh (`caroli-keldysh`): G^< by lead injection, bond currents, occupied LDOS, bias
   windows. Validate on `ti-lcur-*` with the impurity files `imp_00.dat`.
6. Self-consistent Born approximation, validated on `ti-scba-*` (deterministic).
7. Spectral tools: KPM DOS/LDOS (Chebyshev, Jackson kernel, stochastic trace); graphene and TI
   presets.
8. Time evolution (finite differences): split-operator, Chebyshev propagator; graphene current and
   pair creation (`graphene-current-clean` is deterministic).
9. Interface: declarative system and simulation files with a CLI; Python bindings to consider.
10. Acceleration:
    - CPU parallelism (rayon) over energies and samples first.
    - Then GPU kernels for sparse mat-vec (KPM, time evolution) and batched dense solves.
    - Apple GPUs have no f64, so Metal kernels must be f32 (or emulated double-single), which
      suits KPM and time evolution better than transmissions.
    - CUDA allows f64.
11. Examples: the two thesis studies (TI with magnetic impurities: G maps, local currents, Born
    phase diagram; graphene under a field: current, pair creation, DOS).
