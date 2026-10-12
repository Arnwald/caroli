# Benchmarks

caroli against the other codes of the field, on problems that every code solves identically.
Python and Julia are allowed here only to drive the other libraries. Orchestration, data
preparation and reports are in Rust (`crates/caroli-bench`).

## Rules

- **Same Hamiltonian.** `caroli-bench prepare` writes, for each problem, the model (on-site and
  hopping matrices) and the disorder of each width to `data/`. The other codes build their
  systems from these files.
- **Same results first.** The report gives, for each run, the largest difference of its
  transmissions from caroli's. Times count only when the results agree.
- **One process per run**, under `/usr/bin/time -l`, for the peak memory.
- **Idle machine.** The Mac mini also hosts a GitHub runner (huygens). A run during which a
  runner job was active is flagged `(runner busy)` in the report.
- **Threads.**
  - The other codes run single-threaded: `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
    `MKL_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` are all set to 1, and Kwant's MUMPS is
    sequential.
  - caroli runs with `RAYON_NUM_THREADS` = `--threads`. `times_s` holds the single-energy
    latencies, computed one after the other, and `parallel_s` the time for all the energies
    together.
  - Throughput (`--count N`): N energies on all the cores, builds included. caroli runs them on
    its thread pool, Kwant on a pool of as many processes (how Kwant users run sweeps in
    parallel; MUMPS is sequential).
- **Each code at its best.**
  - Kwant uses MUMPS, computes only the needed block of the scattering matrix, and skips the
    Hermiticity check. With the `kwant-accelerate` environment its BLAS is Apple's Accelerate
    (5–10 % faster than OpenBLAS). MUMPS's other orderings and `nrhs` gave nothing.
  - TBtrans takes the electrodes at η = 1e-6 eV (agreement to about 1e-5, for 20 % more time
    than its default 1 meV) and writes double precision. Its conda-forge build with
    Accelerate is not faster, and the new-LAPACK variant crashes.
  - Quantica.jl (GS.SparseLU for the sample, GS.Schur for the leads) first runs a small
    instance, so that compilation is not timed.
  - caroli splits independent blocks, such as the spin blocks of BHZ.

## Problems (`transport.toml`)

Two-terminal transmission of a length × width sample (length = width by default) between clean
leads. caroli's recursive Green's function (`caroli-rgf`) runs up to W = 256 only: it costs
O(L W³). TBtrans and Quantica.jl run the two-terminal problems up to W = 256 (their runners
have no four-terminal set-up); TBtrans needs a real Hamiltonian, so it skips BHZ.

| name | model | widths | disorder |
|---|---|---|---|
| `square-clean` | square lattice, 1 orbital | 32–512 | none |
| `square-anderson` | square lattice, 1 orbital | 32–512 | uniform, w = 2 |
| `bhz-anderson` | thesis BHZ preset, 4 orbitals | 32–256 | uniform, w = 0.5 |
| `square-strip` | square lattice, 1 orbital, length 2048 | 32–128 | uniform, w = 1 |
| `graphene-anderson` | honeycomb, t = 1, zigzag axis (width in cells of 2 sites) | 32–256 | uniform, w = 1 |
| `square-cross` | square lattice, W × W square with four arms of W/4 and four leads (every T_pq) | 32–256 | uniform, w = 1 |

## Setup (Mac mini, once)

Everything lives in `~/Projects/caroli/benchmarks` (ignored by git):

```bash
# micromamba (official mamba-org binary), then Kwant with MUMPS from conda-forge
mkdir -p .tools && curl -fsSL https://micro.mamba.pm/api/micromamba/osx-arm64/latest | tar -xj -C .tools bin/micromamba
export MAMBA_ROOT_PREFIX=$PWD/.mamba
./.tools/bin/micromamba create -y -p $PWD/envs/kwant -c conda-forge python=3.12 kwant python-mumps numpy scipy
```

TBtrans (SIESTA 5.4.2, with sisl) and Kwant on Accelerate, from conda-forge, and Quantica.jl:

```bash
./.tools/bin/micromamba create -y -p $PWD/envs/tbtrans -c conda-forge python=3.12 "siesta=5.4.2=nompi*" sisl netcdf4 numpy scipy
./.tools/bin/micromamba create -y -p $PWD/envs/kwant-accelerate -c conda-forge python=3.12 kwant python-mumps numpy scipy "libblas=*=*newaccelerate" "liblapack=*=*newaccelerate" "libcblas=*=*newaccelerate"
# Julia 1.13.1 (official build) in .tools, packages in .julia (quantica/Project.toml, Manifest.toml)
JULIA_DEPOT_PATH=$PWD/.julia ./.tools/julia-1.13.1/bin/julia --project=quantica -e 'using Pkg; Pkg.instantiate()'
```

## Running

```bash
cargo build --release -p caroli-bench
./target/release/caroli-bench prepare
./target/release/caroli-bench run                                  # caroli, its RGF and Kwant, 1 thread
./target/release/caroli-bench run --code competitors               # TBtrans and Quantica.jl, 1 thread
./target/release/caroli-bench run --code caroli --threads 14 --count 56   # throughput, all cores
./target/release/caroli-bench run --code kwant --threads 14 --count 56    # Kwant, 14 processes
./target/release/caroli-bench report > results/report.md
```

Results are appended to `results/<machine>.jsonl`, one JSON line per run. The report keeps
the latest run of each case.

## Latest results

[results/report-2026-10-12-53d13c9.md](results/report-2026-10-12-53d13c9.md): caroli against
Kwant, TBtrans and Quantica.jl on a Mac mini M4 Pro.

- Kwant is the fastest of the other codes from W = 64 up, except on the long strips up to
  W = 64 (Quantica.jl).
- One thread, against the fastest other code of each case: 11–17× on the square lattice,
  14–20× on the four-terminal cross, 6–12× on the strips, 4.7–9× on BHZ, 4.6–7.3× on graphene.
- All the cores (caroli on 14 threads, Kwant on 14 processes): 10× on the largest square
  samples, strips, cross and BHZ, 4.7–5.9× on graphene.

The first campaign, against Kwant only: [results/report-2026-10-12-08246f7.md](results/report-2026-10-12-08246f7.md).
