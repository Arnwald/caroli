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

Interactive page, with the figures and the Hamiltonians:
[caroli benchmarks](https://claude.ai/artifact/Mvd6XU9f71A38wuHoynoWi).

[results/report-2026-10-12-53d13c9.md](results/report-2026-10-12-53d13c9.md): caroli against
Kwant, TBtrans and Quantica.jl on a Mac mini M4 Pro.

- Kwant is the fastest of the other codes from W = 64 up, except on the long strips up to
  W = 64 (Quantica.jl).
- One thread, against the fastest other code of each case: 11–17× on the square lattice,
  14–20× on the four-terminal cross, 6–12× on the strips, 4.7–9× on BHZ, 4.6–7.3× on graphene.
- All the cores (caroli on 14 threads, Kwant on 14 processes): 10× on the largest square
  samples, strips, cross and BHZ, 4.7–5.9× on graphene.

The first campaign, against Kwant only: [results/report-2026-10-12-08246f7.md](results/report-2026-10-12-08246f7.md).

Quantum walk: [results/walk-report-2026-10-12-c660c96.md](results/walk-report-2026-10-12-c660c96.md),
Mac mini M4 Pro, against the paper's method in Python (the authors' code is not public).

- n = 19 (2·10⁷ amplitudes, the paper's largest): 51 ms per step for U as a SciPy sparse
  matrix, 5.3 ms for caroli in f64 on 14 threads (10×), 2.5 ms in f32 on the CPU or the GPU
  (20×). caroli runs n = 25 (1.7·10⁹ amplitudes).
- The scan of Fig. 21 (8192 walks of 2000 steps): 4.4 h in Python on one process, 6.7 min for
  caroli in f64 (39×), 3.7 min in f32 with the GPU joining the 14 threads (70×). The memory
  bandwidth is shared: the GPU brings as much as the CPU, and f32 is what halves the traffic.

## Quantum walk on a graph of spins (`walk/`)

The model of K. Sellapillay and A. D. Verga (Phys. Rev. E 103, 032123 (2021)): a walker on a
ring of n nodes with one spin per edge, coin angle θ, coupling J; the dimension is
n · 2^(n+1), 2·10⁷ at n = 19, their largest system. Their code is not public: the competitor is
the method their paper describes, written in Python by `walk/reference.py`, the way a Python
code would:

- `sparse`: the one-step operator U built as a SciPy sparse matrix, then `psi = U @ psi` at
  every step;
- `numpy`: the same step as vectorised NumPy operations on `psi[h, s]`, without a matrix.

caroli applies the step without a matrix (`crates/caroli-walk`): on the CPU in f64 or f32, on
all the cores or one, as the explicit sparse U for comparison, and on the GPU in f32 or
double-single, raw or checked against the CPU (`caroli_walk::evolve`).

- **Same results first.** Every method's position distribution p(x) after the same steps is
  compared with caroli's in f64: 10⁻¹⁵ for the f64 methods, 10⁻⁷ for f32.
- **Python on one thread** (one process, BLAS threads at 1); caroli on all the cores and on one.
- **Steps**: about a second of the SciPy reference per size, after a warm-up.
- **Two workloads.** The time per step on rings of 13 to 25 nodes (`walk run`), and the scan of
  the (θ, J) plane of the paper's Fig. 21 (`walk phase`): for each point and for the initial
  spins along z and along x, 2000 steps from the middle node, the time-averaged p(x) and its
  Kolmogorov–Smirnov distance to the uniform distribution. caroli runs one walk per thread
  (`step_serial_accumulate`), in f64 and in f32; Python runs a sample of the points, and its
  time is extrapolated to the whole scan.

```bash
cargo build --release -p caroli-bench --bin walk
./target/release/walk run --sizes 13,15,17,19,21 --python-max-n 21
./target/release/walk run --sizes 23,25 --python-max-n 0
./target/release/walk phase --n 13 --grid 64 --steps 2000 --sample 8
./target/release/walk report > results/walk-report.md
```

Results go to `results/walk-<machine>.jsonl`, the maps of the scan to
`results/walk-phase-<machine>.json`.
