# caroli — architecture and roadmap

## Crates

```
crates/
  caroli-lattice      Bravais lattices with a basis (1D/2D/3D), finite regions, slices
  caroli-model        tight-binding models: on-site and hopping matrices, site-dependent terms,
                      dense / sparse (CSR) / Bloch Hamiltonians, transport blocks; BHZ Dirac
                      matrices and presets
  caroli-dense        dense complex kernels: Schur decomposition and reordering, LU, products
                      (Apple Accelerate on macOS, faer elsewhere)
  caroli-green        steady state: lead modes and self-energies, recursive Green's function,
                      nested dissection (two and many terminals), Caroli transmission
  caroli-disorder     reproducible random streams (xoshiro256**, gfortran-compatible seeding),
                      Anderson disorder, classical magnetic impurities
  caroli-keldysh      non-equilibrium steady state: lesser Green's function by lead injection,
                      bond currents, interface currents, occupations
  caroli-born         disorder-averaged Green's functions: self-consistent Born approximation
                      (mean field + cumulant vertices; Anderson and magnetic impurities)
  caroli-torus        Hamiltonians on periodic 2D supercells: k-space part through FFTs (rustfft),
                      local terms in real space
  caroli-kpm          kernel polynomial method: Chebyshev moments, Jackson kernel, densities,
                      Kubo–Bastin conductivities
  caroli-dynamics     time evolution: split-operator on tori, Chebyshev propagation, fourth-order
                      commutator-free Magnus integrator
  caroli-gpu          Chebyshev kernels (KPM, propagation) on Metal (f32, double-single) and
                      CUDA (f64), the sweeps of caroli-walk on Metal; the device, options and
                      report shared by the runs checked against the CPU
  caroli-walk         discrete-time quantum walks: a walker on a graph coupled to spins on its
                      edges (Sellapillay & Verga), on the CPU (f64, f32) and the GPU
  caroli              facade: re-exports and prelude
  caroli-validation   (not published) comparisons with the thesis golden outputs
  caroli-bench        (not published) benchmarks against other codes (`benchmarks/`)
```

The dependencies only go downward: lattice ← model ← green ← caroli, and dense ← green.

## Design choices

- **Linear algebra: faer.** Complex scalars are `faer::c64` (= `num_complex::Complex<f64>`),
  matrices `faer::Mat<c64>` (`CMat`).
  - The kernels that dominate transport go through `caroli-dense`: `schur` with `partition`
    and `reorder`, `Lu` (solves with `a` and `a†`, pivots, a condition estimate), `svd`,
    `hermitian_eigen`, `orthonormal_columns`, `matmul` and `product`. Products of lead-sized
    matrices in `caroli-green` use them too (`mul`, `sandwich`, `caroli_trace`), never faer's
    `*`.
  - On macOS they call Apple's Accelerate framework (feature `accelerate`, on by default):
    LAPACK 3.9+ (`$NEWLAPACK` symbols) and BLAS, which run on the matrix units of Apple silicon.
    It is part of the system, so nothing extra is installed, but it needs macOS 13.3 or later.
  - Elsewhere, or with the feature off, they use faer and the in-house QR algorithm.
  - Measured on an M5, one thread, n = 512: Accelerate's products are 5× faster than faer's.
    Its Schur decomposition (zhseqr) takes 0.35 s on a lead pencil. faer's multishift QR is
    1.8× faster for the eigenvalues alone, but only 14 % faster with the Schur vectors (0.31 s),
    and it is private (`pub(crate)`): not worth vendoring for now.
- **Orbital layout** is fixed by the `Region`: site `i` occupies orbitals
  `offset(i) .. offset(i) + orbitals(i)`. `Region::cuboid` orders the sites with the first axis
  varying slowest, so the slices along x are contiguous.
- **Model** = on-site matrix per sublattice + hoppings `⟨(R, a)| H |(R + offset, b)⟩` (their
  conjugates are implied) + site-dependent terms (`OnSite` trait: disorder, potentials) +
  site-dependent hopping terms (`HoppingTerm`, Kwant's value functions for hoppings: bond
  disorder, strain; the reverse hop is the conjugate, so `H` stays Hermitian).
  The Bloch convention is `H(k) = Σ_n H_{0n} e^{i k·n}`.
- **Continuum models** (`caroli_model::continuum`): `Continuum::parse` reads a k·p Hamiltonian
  (numbers, parameters, `k_x k_y k_z`, Pauli matrices, `kron`, `+ − * / ^`) into a polynomial in
  `k` with matrix coefficients; `discretize(a)` replaces `k → −i∂` by central differences on a
  chain, square or cubic lattice, as Kwant's `continuum.discretize`. The continuum BHZ
  Hamiltonian gives the thesis preset exactly. Coefficients are constant for now.
- **Model inputs**: on-site and hopping terms take a `CMat` or, for one orbital, a number
  (`IntoMatrix`). `pauli::{s0, sx, sy, sz, kron}` and `combine(&[(c, &M), ...])` (real or complex
  coefficients) build spin and orbital matrices. `lead_voltages` solves the Landauer–Büttiker
  equations of a multi-terminal device (Hall and longitudinal resistances).
- **Magnetic field**: Peierls phases, `Model::with_peierls(φ)` multiplies every hopping
  `⟨i| H |j⟩` by `exp(i φ(r_i, r_j))`. `caroli_model::gauge` gives uniform fields in the Landau
  (x or y) and symmetric gauges; `b` is the phase per unit area, so the flux quantum is 2π.
  - `Model::with_field(b)` sets a uniform field. The model works in the Landau gauge
    invariant along x; each lead along a direction `d` is taken to the Landau gauge invariant
    along `d` by `ψ → exp(iχ) ψ`, with `χ` quadratic. The device keeps its gauge and the
    device-lead couplings are mixed, so leads can point in any direction, as with Kwant's
    `magnetic_gauge`. With `with_peierls` (any field), each lead must already see a gauge
    invariant along it.
  - Every transport constructor checks that the lead Hamiltonian is the same from cell to cell
    (in the lead's gauge), and panics otherwise.
  - Tests:
    - gauge invariance of flake spectra and of transmissions;
    - quantum Hall plateaus (T = 1, 2, 3 to 1e-6 in disordered ribbons, along x and along y);
    - a Hall cross: symmetry under a quarter turn of a clean cross (x and y leads in different
      gauges), Onsager reciprocity `T_pq(b) = T_qp(−b)`, and the chiral transmissions of the
      quantum Hall regime (ν into the next lead, 0 elsewhere).
- **BHZ conventions** follow the thesis codes (2014+): basis `(E1↑, H1↑, E1↓, H1↓)`, Γ matrices
  of `numbers.f90`. Exchange coupling: `GFactors::Same` (g_E = g_H, `n·s`) and `Opposite`
  (g_E = −g_H, `n·s τ_z`). The thesis calls these "ferro" and "antiferro".
- **Lead self-energies by mode matching** (`caroli_green::modes`), after Wimmer and Kwant.
  - The coupling is compressed by its SVD, `H1 = u v†` (rank r). The Bloch modes then solve a
    regular 2r pencil in `x = (u†ψ_{j−1}, v†ψ_j)`, stabilised by `i(uu† + vv†)` so that it holds
    even when `H0 − E` is singular. Low-rank couplings (graphene cut along a slanted direction)
    give small, well-conditioned problems.
  - Invariant subspaces come from a complex Schur decomposition with reordering
    (`caroli-dense`). Eigenvectors are not used for the evanescent modes: for strongly decaying
    modes they are nearly dependent (inverse condition 1e-15 in a 100-wide ribbon).
    - The decomposition is Accelerate's on macOS. Elsewhere, faer's Hessenberg reduction is
      followed by our own single-shift QR, because faer's QZ is slow near n = 80 and its
      standard Schur is private.
    - The reordering moves eigenvalues by adjacent swaps (as LAPACK ztrexc), in windows of 64:
      the rotations of a window reach the rest of the matrix as one product (Kressner's block
      algorithm). Moving half the spectrum of a 512 matrix to the front takes 17 ms, against
      145 ms swap by swap and 181 ms with Accelerate's ztrsen.
    - The propagating eigenvectors are back-substituted in the Schur form, then transformed,
      projected and multiplied by `H1` as matrix products over all the modes at once.
    - **Real problems** (real `H0`, `H1` and energy: graphene, square lattices with
      non-definite couplings, Anderson disorder) keep the pencil real: the stabilisation is
      `s(uu† + vv†)` with a real s (0.618…) instead of i, the SVD of `H1` is real (dgesdd), and
      the Schur decomposition goes through the real Schur form (dhseqr), whose 2 × 2 blocks are
      then split by complex rotations (`rsf2csf`, O(n) per block). Everything after it is
      unchanged. The real form costs a third of the complex one at n = 512 to 1024; graphene
      ribbons, bound by their leads, are 1.5–1.6× faster (W = 128 and 256). If the real
      stabilisation is ill-conditioned at some energy, the complex one takes over.
  - Propagating modes are split by velocity sign. The current is diagonalised within each
    degenerate group.
  - `Σ_R = u X_γ X_α⁻¹ u†` and `Σ_L = v Y_α Y_γ⁻¹ v†`, from orthonormal bases of the right- and
    left-going subspaces.
  - The result is exact at real energy, with no `η`. The exceptions are subband edges (Jordan
    blocks, zero velocity) and bound states of the lead surface (singular `X_α`, as for graphene
    at E = 0): the modes are then computed at `E + δ`, with δ from 1e-9 to 1e-6 of the
    bandwidth, which gives the limit from above.
  - **Fast path for Hermitian definite couplings**, `H1 = ±P` with `P > 0`: square lattices,
    and multi-orbital leads without spin-orbit coupling or magnetic field along the axis. At a
    real energy, `μ = λ + 1/λ` turns the Bloch equation into the Hermitian problem
    `(H0 − E) ψ = ∓μ P ψ`, of size n instead of a 2n Schur decomposition.
    - Each transverse mode gives two roots of `λ² − μλ + 1 = 0`, and the factors `u = ±L`,
      `v = L` of `P = L L†` make the mode bases orthonormal as they stand.
    - Subband edges are detected as in the general algorithm: a real `|λ|` within 1e-6 of 1, or
      two roots closer than 1e-7 (a Jordan block).
    - At W = 256 the lead modes take 16 ms instead of 0.47 s.
    - `LeadModes::at_general` keeps the general algorithm for cross-checks, and a test compares
      the two on strips (also on subband edges), degenerate channels and random leads.
  - Identical leads share one set of modes. Independent blocks (spin-conserving disorder) are
    solved separately (`TwoTerminal::blocks`).
  - Decimation (Lopez Sancho – Rubio) is kept as a cross-check only. It loses all accuracy when
    E is close to an eigenvalue of the lead cell: T of a clean chain at E = 1e-9 is off by 0.5
    with η = 1e-12.
- **Two-terminal solvers** (`caroli_green`):
  - `TwoTerminal`: the recursive Green's function, slice by slice. One dense inversion of the
    slice size per slice: O(L W³) for a length × width strip, O(n⁴) for an n × n sample.
  - `NestedTwoTerminal`: geometric nested dissection (the multifrontal method). Separator layers
    cut the device recursively. Each node eliminates its separator from a dense front and passes
    the Schur complement on its boundary to its parent. The two lead slices are never
    eliminated: the root holds the Schur complement on them and gives `G_LR`. The cost is O(n³)
    for an n × n sample and O(L W²) for a long strip. Only pending Schur complements are kept.
  - In `NestedTwoTerminal`, a separator block that cannot be eliminated stably is delayed to the
    parent. This happens for a piece of a clean lattice at one of its eigenvalues. Independent
    branches run in parallel.
  - `NestedMultiTerminal`: any number of leads attached anywhere (the interfaces of
    `MultiTerminal`). The sites the leads couple to stay at the root, whose inverse gives
    `G_pq` and `T_pq` for every pair of leads. `MultiTerminal` inverts the whole device densely;
    it stays as the reference in the tests (crosses of square and BHZ lattices, graphene with a
    narrow lead).
  - The two nested solvers share the elimination (`SiteTree`, `Tree`).
  - `NestedTwoTerminal::scattering_states`: the elimination keeps `X = F11⁻¹ F12` of every
    front, and a back-substitution from the lead slices gives `ψ_p = G W_p` (`W_p W_p† = Γ_p`)
    on the whole device. From them come the occupied densities and the bond currents of the
    Keldysh conventions (`G^< = i Σ_p f_p ψ_p ψ_p†`), which match `caroli-keldysh`'s recursive
    `Lesser` to 1e-10 (square, BHZ, graphene, field). `NestedMultiTerminal::scattering_states`
    does the same for any number of leads: in a cross, the current out of each arm is `T_pq`
    and Kirchhoff's law holds at every inner site, with or without a field. With an on-site
    operator `O` (`site_densities`, `operator_currents`: `Re Tr[(O H_ij + H_ij O) G^<_ji]`) they
    give spin densities and spin currents: in a BHZ ribbon with a Zeeman term, the `s_z`
    current through every cut is `T↑ − T↓`.
  - `NestedMultiTerminal::sector_transmissions`: transmissions between the sectors of a
    conservation law in the leads (spin, electrons and holes), with `Γ_{p,a} = V P_a γ_p P_a V†`;
    within one lead they are reflection probabilities between sectors, such as Andreev
    reflection. Tests: no spin flip in BHZ, the sectors add up to `T_pq`, and an NS junction
    reflects electrons as holes below the gap.
  - `smatrix` (both nested solvers): the scattering matrix in the basis of the propagating
    lead modes, as `kwant.smatrix`. `LeadModes` keeps the propagating modes normalised to unit
    current. An incoming mode `φ λ^n` acts on the device as the source `V (λ − g H1†) φ`; the
    outgoing wave on the first lead cell is `g V† ψ_D` (minus `g H1† φ` in the incoming lead),
    and its amplitude on each outgoing mode is its current overlap with it. Tests: unitarity to
    1e-9 (crosses of square and BHZ lattices, graphene, a field), `Σ |S_pq|² = T_pq`, a clean
    ribbon sends each mode straight through, and the two solvers give the same matrix.
  - `NestedTwoTerminal::transmission_eigenvalues` (squared singular values of
    `W_L† G_LR W_R`) and `fano_factor` (shot noise); `Lead::bands` for lead band structures.
  - Both solvers split independent blocks of orbitals, such as the spin blocks of BHZ.
  - `NestedTwoTerminal::self_energies` and `transmission_with` reuse the lead self-energies of
    one energy for every disorder configuration with the same leads. For BHZ at W = 128 the
    leads take 85 % of the time per energy, so a disorder average gets about 6× faster.
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
| Kwant's tutorials (`caroli/tests/tutorials.rs`): quantum wire steps, Rashba wire, Aharonov–Bohm ring, zigzag graphene ribbon, Hall bar resistances | — | the physics each one shows |

Golden cases with disorder use the same impurity configurations as the thesis.
`caroli_disorder::rng::Xoshiro256::gfortran` reproduces gfortran's `random_seed(put=…)` and
`random_number` bit for bit, and the disorder generators follow the thesis draw order.
`caroli_validation::thesis` rebuilds each thesis setup.

**Thesis bug found this way:** all conductance codes use `real(IM*(Σ − Σ†))` for the broadening
(recorded in the README of the thesis reference, which is kept out of this repository).

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
   self-energy (recorded in the thesis reference).
7. ✅ Spectral tools: KPM (Chebyshev moments, Jackson kernel, stochastic trace) on tori
   (`caroli-torus`); validated on the TI and graphene KPM codes (exact). Kubo–Bastin
   conductivities (`caroli_kpm::kubo`, García, Covaci and Rappoport 2015, as Kwant's
   `kpm.conductivity` and KITE) with `Model::sparse_velocity`: the Qi–Wu–Zhang Chern insulator
   gives `σ_xy = ∓1.004 e²/h` and `σ_xx ≈ 1e-6` in its gap (32 × 32 torus, 256 moments).
8. ✅ Time evolution: split-operator on tori (`caroli-dynamics`); graphene current, pair
   creation, wave-packet width and density validated (exact).
9. Interface: declarative system and simulation files with a CLI; Python bindings to consider.
10. ✅ Acceleration:
    - CPU: rayon over energies (`TwoTerminal::transmissions`), k sums, random vectors, FFT rows,
      sparse products; transport split into independent blocks (`TwoTerminal::blocks`).
    - Faster methods: KPM doubling relations and block products, Anderson-accelerated SCBA,
      Chebyshev propagation, fourth-order commutator-free Magnus integrator.
    - GPU (`caroli-gpu`): Chebyshev kernels (KPM moments, propagation) on CSR matrices.
      - Metal: f32, and double-single (~1e-14) for f64-like accuracy. Tested on an M5: KPM on
        a 1M-orbital torus is 4× (f32) or 2× (double-single) faster than 10 CPU cores.
      - CUDA: f64 through NVRTC, feature `cuda`. Not yet run on NVIDIA hardware.
    - Checked runs in reduced precision (`caroli_gpu::{Device, Options, Report}`): the user
      chooses the device and the precision; the CPU checks a sample of the work in f64 and takes
      over when the error would exceed the tolerance or the GPU fails, and every run reports
      where it ran and the error measured.
      - `kpm::accelerated::moments`: one start vector in fifty recomputed on the CPU.
      - `dynamics::accelerated::propagate`: 2 % of the steps on both (calibration, error scaled
        to the whole run), the norms checked at the end.
      - `walk::evolve`: the same calibration, then the norm checked at every checkpoint, f32
        giving way to double-single and then to the CPU, which restarts from the last good
        checkpoint.
    - What the GPU of Apple silicon brings, measured:
      - Memory-bound sweeps (the quantum walk): the CPU and the GPU share the same memory and
        reach the same bandwidth (about 230 GB/s on the M4 Pro). What halves the time is f32,
        on either: at n = 19 the CPU on 14 threads and the GPU take 2.5 ms per step in f32,
        5.3 ms in f64 or double-single. The GPU wins on states that fit its cache (n ≤ 15).
      - Dense algebra (`benchmarks`-sized fronts), M5: f32 products on the GPU (Metal
        Performance Shaders, real embedding of the complex matrices) are 4× faster than
        Accelerate's f64 complex products from n = 1024 up, the LU 1.6× at n = 1024 and 3–5×
        from n = 2048; below n = 512 there is no gain. Only the top fronts of the largest
        samples are that large: about 2× on those in f32, with the transport errors measured in
        item 12. Not implemented; a CUDA card with fast f64 is the better target for transport.
    - Not on the GPU: dense transport solves, and FFT-based torus operators.
12. Transport precision, chosen by the user (planned). The RGF sweep would take a `Precision`:
    - `Double`: the default, on the CPU.
    - `Single`: GPU in f32, for previews.
    - `Mixed`: each slice inversion in f32, refined once (`X ← X + X(1 − A X)`) with the
      residual in double-single.

    The loss was measured with `crates/caroli-validation/examples/rgf_precision.rs`:
    - systems: the spin blocks of the g2g1 maps (disordered length 256), at 6 energies per
      disorder strength;
    - quantity: |ΔT| per block against `Double`;
    - samples: 2 configurations at W = 64, 1 at W ≥ 128.

    | W | `Single`: median / 99 % / max | `Single` on plateaus, max | `Mixed`: median / 99 % / max |
    |---|---|---|---|
    | 64 | 5.6e-6 / 1.6e-3 / 1.0e-2 | 2.4e-4 | 7.5e-12 / 4.0e-7 / 8.7e-7 |
    | 128 | 1.3e-5 / 5.2e-3 / 2.5e-2 | 1.3e-3 | 1.9e-11 / 8.9e-7 / 8.5e-5 |
    | 256 (*) | 1.7e-5 / 2.0e-2 / 2.0e-2 | 3.2e-3 | 2.8e-10 / 2.4e-5 / 1.0e-4 |

    (*) Three disorder strengths per case, near the transitions.

    - **Where the loss comes from.** The single-precision inversions. Rounding the lead
      self-energies costs less than 1e-7, and a double-precision final trace does not help.
    - **How it grows.** On the plateaus, about ×2.5–5 per doubling of W: at W ≥ 128, `Single`
      exceeds the 1e-3 used to call a point quantized. It is largest in diffusive regions.
      Transmissions below 1e-38 underflow to zero, so localisation lengths need `Double` or
      `Mixed`.
    - **`Mixed` also grows with W**, up to 1e-4 at W ≥ 128. A second refinement step, or
      refining until the residual falls below a threshold, should bring it back near double
      precision.
    - **Reporting the loss.** Every reduced-precision run should report its own error:
      - the residual norms ‖1 − A X‖ of the inversions, which `Mixed` computes anyway;
      - a random fraction of the points (say 2 %) recomputed in `Double`, with the median and
        maximum |ΔT| returned with the results.
    - **CPU timing.** On the M5 CPU, the f32 sweep takes half the time of the f64 one (4.4 s
      against 8.7 s at W = 128, 25.7 s against 48.9 s at W = 256).
    - **Faster double path, to do first.** Two changes:
      - the coupling between slices is diagonal site by site, so two of the three products per
        slice become O(n²);
      - dense algebra through Accelerate (AMX). Done for the inversions (`caroli-dense`): the
        square lattice at W = 256 went from 4.9 s to 3.5 s per energy on the M5.
11. Examples: the two thesis studies (TI with magnetic impurities: G maps, local currents, Born
    phase diagram; graphene under a field: current, pair creation, DOS).
13. Speed: the fastest code of its kind, measured against the reference codes: Kwant, TBtrans and
    Quantica.jl so far (`benchmarks/results/report-2026-10-12-53d13c9.md`); KITE (KPM) later.
    - `benchmarks/` and `caroli-bench`: every code solves the same exported Hamiltonians, the
      results must agree before times count, and each run gets its own process with its peak
      memory. The machine is a Mac mini M4 Pro.
    - First baseline: two-terminal transmission per energy, on 1 thread. The RGF is slower than
      Kwant (MUMPS) beyond W = 64 on the square lattice: 4.9 s against 1.1 s at W = 256.
    - `NestedTwoTerminal` closes the gap on large samples. First measurement, on one thread with
      a CI job running on the machine: 1.6 s at W = 256 (square; Kwant 1.1 s) and 2.8 s for BHZ
      at W = 128 (Kwant 5.2 s), with far less memory. At that size the lead modes take most of
      the time (1.25 of 1.43 s).
    - Dense kernels (`caroli-dense`), measured on the M5, one thread, square lattice, mean per
      energy: `NestedTwoTerminal` went from 1.35 s to 0.47 s at W = 256, and from 0.17 s to
      0.08 s at W = 128.
      - Lead modes at W = 256 (pencil of 512): 1.25 s → 0.47 s. The propagating modes went
        from 228 ms to 19 ms with batched products.
      - What remains: the Schur decomposition (0.35 s), the orthonormal bases with their
        condition checks (47 ms), and the elimination (0.11 s).
    - The other stages of the general algorithm also use Accelerate: SVD of the coupling
      (zgesdd), one product for the four blocks of `[u v]† G̃ [u v]`, QR of the mode bases, and
      a 1-norm condition estimate from an LU (as zgecon) instead of an SVD of `X_α` and `Y_γ`.
      BHZ at W = 128 went from 0.72 s to 0.62 s per energy (two spin blocks with a 512 pencil
      each; the Schur decomposition is now 60 % of the time).
    - The Hermitian fast path removes the Schur decomposition for square-lattice leads:
      `NestedTwoTerminal` takes 0.127 s per energy at W = 256 and 0.026 s at W = 128 (M5, one
      thread). The elimination is now the main cost there. BHZ leads keep the general
      algorithm.
    - Benchmarks of 2026-10-12 on the idle Mac mini (`benchmarks/results/`), one thread, per
      energy: caroli is 9–13× faster than Kwant on the square lattice (W up to 512), the long
      strips and the four-terminal cross, 3–11× on BHZ (W up to 256) and 2–4× on graphene,
      with 2–8× less memory; transmissions agree to 1e-10 (7e-8 at worst). On 14 threads the
      eight energies of a problem take 5–6× less time than on one: the matrix units are shared
      by each cluster of cores.
    - Leaves of 16 orbitals instead of 64: a dense leaf of L sites costs L³, so the leaves
      cost N L² in all and outweighed the top separators. Square lattice at W = 256: 0.106 s
      per energy (was 0.134 s).
    - Multi-core throughput (Mac mini, 56 energies, one BLAS thread per worker): Accelerate on
      every core beats faer and any mix of the two (faer is 2–3× slower per call, so letting
      some cores run it while others hold the matrix units loses). mimalloc and Accelerate's
      own threads change nothing. 14 threads (10 performance and 4 efficiency cores) give
      7–8× the throughput of one; 8–32 orbitals per leaf are equivalent.
    - Profiling showed 40 % of the time in faer's `*` (vector units) and in a per-energy
      eigendecomposition: self-energies, Caroli traces, S matrix and RGF products now go
      through Accelerate, the Caroli trace takes two products instead of three, and a lead with
      `H1 = c·1` computes the spectrum of its cell once (`ModeSolver`, kept by the solvers).
      Square lattice at W = 256: 9.1 → 12.9 energies/s on one thread, 65 → 77 on 14
      (graphene at W = 128 +12 % and +8 %, BHZ +6 % and +4 %).
    - Fronts below 64 variables cost as much as the large ones (square lattice at W = 256): the
      time went to calls, copies and index searches, not arithmetic. Each front now has an
      assembly plan computed with the tree (its variables, the positions of its `H` entries and
      of its children's Schur complements), used whenever nothing below it was delayed, and
      small fronts are eliminated by plain Gaussian elimination without the dense library.
      +12 % on the square lattice at W = 256, +9 % at W = 512 (M5, one thread); graphene and
      BHZ, dominated by their leads, do not change.
    - Second campaign (2026-10-12, commit 53d13c9), against Kwant (MUMPS on Accelerate), TBtrans
      and Quantica.jl: Kwant is the fastest of them from W = 64 up, except on the long strips up
      to W = 64 (Quantica.jl). caroli is 11–17× faster than the fastest on the square lattice,
      14–20× on the cross, 6–12× on the strips, 4.7–9× on BHZ and 4.6–7.3× on graphene, on one
      thread; on all the cores, 10× faster than a pool of Kwant processes on the largest square
      samples, strips, cross and BHZ, 4.7–5.9× on graphene. Large samples scale 4–6× on 14
      threads (shared matrix units and memory bandwidth); running fewer energies at once with
      more threads each is slower.
    - Next: BHZ is bound by the complex Schur decomposition of its leads (the real pencil does
      not apply: a spin block breaks time reversal).
14. ✅ Quantum walks (`caroli-walk`): the walk of K. Sellapillay and A. D. Verga (Phys. Rev. E
    103, 032123 (2021)), a walker on any graph with one spin per edge, `U = V(J) M C`.
    - Layout `ψ[h · 2^|E| + s]` (half-edge, spin configuration fastest): the coin pairs the rows
      of a node, `V M` the two rows of an edge and, in them, the configurations with that
      edge's spin up and down, like a one-qubit gate of a state-vector simulator. A step is two
      sweeps over memory, without a matrix; low spin bits are processed in whole blocks.
    - Validated against the explicit sparse `U` (and its unitarity), the free walk, and the
      curves of the paper's Fig. 3 (`examples/paper.rs`); the benchmark checks every method's
      `p(x)` against caroli's f64.
    - Kernels generic over f64 and f32; `step_serial` runs one walk per thread for scans of
      parameters, `step_serial_accumulate` adds the half-edge probabilities without another
      sweep; Metal kernels in f32 and double-single, run through `walk::evolve`.
    - Benchmark (`benchmarks/walk`, Mac mini M4 Pro), against the method of the paper written
      in Python (their code is not public): at n = 19 (dimension 2·10⁷, their largest),
      51 ms per step with U as a SciPy sparse matrix, 5.3 ms for caroli in f64 on 14 threads
      (10×), 2.5 ms in f32 on the CPU or the GPU (20×); caroli runs n = 25 (1.7·10⁹
      amplitudes), 0.22 s per step on the GPU in f32.
