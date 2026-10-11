# reference/baseline — golden outputs of the thesis codes

This folder compiles the thesis programs in `reference/thesis/` and runs small, fast cases. The
outputs it records are the regression baseline for caroli. The thesis sources are compiled as
they are; a few minimal fixes are applied at build time to a copy (see [Build-time patches](#build-time-patches)).

```bash
python3 reference/baseline/baseline.py deps            # shims + LAPACK95 -> target/reference/
python3 reference/baseline/baseline.py build           # 18 programs -> target/reference/bin/
python3 reference/baseline/baseline.py run             # run every case, compare with golden/
python3 reference/baseline/baseline.py run --update    # re-record golden/ (after a deliberate change)
python3 reference/baseline/baseline.py build --check   # -O0 + bounds checking -> bin-check/
python3 reference/baseline/baseline.py run --check     # same cases with the checking build
python3 reference/baseline/baseline.py show FILE       # records of a Fortran output file
python3 reference/baseline/baseline.py list            # programs and cases
```

Status (2026-10-11):
- 26/26 cases pass.
- Results are deterministic from run to run.
- The `-O2` build and the `-O0 -fcheck=bounds` build agree within the case tolerances (rtol 1e-8).
- Every case runs in under a second.

## Requirements

| dependency | where it comes from |
|---|---|
| gfortran (tested: Homebrew GCC 16.1), a C compiler | `FC`, `CC` |
| FFTW 3 (double, threads) | `brew install fftw`; `FFTW_PREFIX` to override |
| LAPACK + BLAS | Accelerate on macOS, `-llapack -lblas` elsewhere; `LAPACK_LIBS` to override |
| LAPACK95 sources (netlib) | not in the repository: `LAPACK95_SRC`, default `~/Projects/electronic-transport/TI/Fortran/librairies/LAPACK95/SRC` (on another machine: netlib `lapack95.tgz`). Only the double and double-complex routines are built. |
| MPI | not needed: `shims/` provides a serial stand-in |
| Python ≥ 3.11 | standard library only |

Everything is built into `target/reference/` (git-ignored).

## Shims

- `shims/mpif.h`, `shims/mpistub.c` — **serial MPI** (one rank).
  - It implements the calls the codes use: `Init`, `Comm_rank`, `Comm_size`, `Get_processor_name`, `Bcast`, `Reduce`, `Allreduce`, `Gather(v)`, `Scatter(v)`, `Barrier`, `Wtime`, `Finalize`.
  - Collectives become copies. `Send`/`Recv` abort, so `conductance-2t-laurent` is not built.
  - The results are those of a 1-process run. Programs that spread their work over ranks (energies, samples) still compute everything. The exception is the KPM DOS codes, whose ranks have fixed roles: rank 0 does spin up, rank 1 spin down, ranks 2–5 the LDOS. On one rank only the spin-up DOS is filled, so their cases use `choice = 1` (no LDOS).
- `shims/fftw_threads_f77.c` — Fortran wrappers `dfftw_init_threads`, `dfftw_plan_with_nthreads` and `dfftw_cleanup_threads`. Homebrew's FFTW leaves them out.

## Build-time patches

`PATCHES` in `baseline.py` applies these regular-expression fixes to a copy of the sources in
`target/reference/src/<program>/`. `target/reference/build/<program>/patches.txt` lists what was
applied to each program.

| patch | where | why | effect on results |
|---|---|---|---|
| `bumpchar-local-index` | `filename.f90` (every program) | `bumpchar` modifies the module variable `i`, the DO index of its caller `filen`. gfortran ≥ 14 rejects it. | none: the DO trip count is fixed on entry |
| `y-current-loop-in-bounds` | `ti/local-currents` `green.f90` | the 2015 y-current loop `iy = 2:Ny` reads `h_0` beyond its last column at `iy = Ny`, which gives garbage that changes from run to run | the non-existent bond Ny→Ny+1 gets 0 (zero hopping); every other value is unchanged; bond 1→2 is still missing, as in the thesis |
| `time-tables-ntdiag-plus-one` | `graphene/wavepacket-width` `g_s.f90` | the time tables have NTDIAG values but receive NTDIAG+1, which corrupts the heap and causes random crashes | none on in-bounds values; the last value becomes well defined |
| `density-time-tables-ntdiag-plus-one` | `graphene/wavepacket-density` `g_s.f90` | same | none: only NTDIAG values are written |
| `xy-mesh-in-bounds` | graphene `utilities.f90`, `utilitieslight.f90` | `xy_mesh` loops over N+1 points into N×N arrays | none on in-bounds values; no more writes past the arrays (which may have altered w(t) in the thesis runs) |
| `fixed-seed-no-clock` | `graphene/current-pair-creation` `g_s.f90` | the disorder seed adds the wall-clock date and time | the realisation becomes reproducible (seed `1237 + rank`) |

## Cases

Each case lives in `cases/<name>/`: a `case.toml` with the program, description and optional
`rtol`/`atol`/`timeout`, plus the input file the program reads. Inputs keep the production
parameters and shrink the sizes. The golden outputs (`golden/<name>/`, ~650 KiB in total) are
the files exactly as the program wrote them.

| case | program | what it covers |
|---|---|---|
| `ti-cond2t-clean` | conductance-2t | clean BHZ ribbon 16×10, G(E), E ∈ [−0.3, 0.3]: G ≈ 2 in the gap, 8 in the bands |
| `ti-cond2t-same-g` | conductance-2t | magnetic impurities along z, same-g (g_E = g_H), J_s ∈ [0.5, 1.5] |
| `ti-cond2t-isotropic` | conductance-2t | isotropic random impurity spins |
| `ti-cond2t-opposite-g` | conductance-2t-opposite-g | as `same-g` with g_E = −g_H |
| `ti-cond4t-anderson` | conductance-4t | 8×8 square, four leads, Anderson disorder: T13, T24 |
| `ti-cond-ldos-current` | conductance-ldos-current | manuscript code: G, LDOS, bond currents with Anderson disorder |
| `ti-lcur-same-g`, `-opposite-g`, `-anderson` | local-currents-2t | Keldysh bond currents and occupied LDOS for the three couplings |
| `ti-scba-anderson` | scba-anderson | SCBA, Anderson disorder (TAI) |
| `ti-scba-matrix-same-g`, `-opposite-g` | scba-magnetic-matrix | 4×4 SCBA, magnetic impurities |
| `ti-scba-analytic-same-g`, `-opposite-g` | scba-magnetic-analytic | reduced SCBA (g2g1 paper) |
| `ti-dos-kpm-same-g`, `-opposite-g` | dos-kpm-magnetic(-opposite-g) | KPM DOS, spin-up block only (one rank) |
| `ti-ribbon-bands` | ribbon-bands | E(k_y), uniform exchange J_s |
| `ti-ribbon-bands-fourier-y` | ribbon-bands-fourier-y | 2013 code; needs Nx = Ny |
| `ti-diag` | ti-diag | exact diagonalisation of a 6×6 flake |
| `ti-dos-spin-lesser` | dos-spin-lesser | 2013 recursive G^< DOS |
| `graphene-current-clean`, `-af`, `-para` | graphene-current | split-operator j(t), n(t) under E, clean / AF / para impurities |
| `graphene-width` | graphene-width | wave-packet width w(t) |
| `graphene-density` | graphene-density | sublattice electron/hole densities |
| `graphene-dos-kpm` | graphene-dos-kpm | KPM DOS (spin-up block only) |

## Comparing outputs

- **Format.** Outputs are Fortran sequential unformatted files: each record is an int32 length, the payload, then the same int32. The payloads are almost always float64 or complex128. `param*.dat` files are namelist text.
- **Comparison rules.** `compare_file` compares byte by byte first. Otherwise it compares record by record as float64 with tolerance, or token by token for text. Text lines with timings (`CPU …`) are skipped.
- **Reading the files elsewhere.** The Rust side can parse the same files: split on the int32 markers, then read little-endian f64.
- **Random disorder.** Realisations come from gfortran's `random_number` (xoshiro256\*\* since GCC 8) seeded with `random_seed(put = [seed, seed, …])`. They cannot be reproduced bit for bit outside gfortran. caroli should therefore check disordered cases in one of three ways:
  - feed in the impurity configuration the program writes (`imp_XX.dat` in the local-current codes);
  - reimplement this seeding;
  - compare statistics over realisations.

  Clean cases and the Born approximation (no randomness) can be compared directly.

## Findings

- Bounds checking confirmed or uncovered the out-of-bounds accesses listed above. They are now
  recorded in the thesis READMEs.
- The clean-wire conductance is not an exact integer: 1.988–2.012 in `ti-cond2t-clean`, and ≈ 1.99
  in the thesis-era G(E) data. A perfectly matched clean wire should give an integer, so this is
  the first thing to check with the caroli Green's function solver.
