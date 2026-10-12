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
- **Each code at its best.** Kwant uses MUMPS, computes only the needed block of the scattering
  matrix, and skips the Hermiticity check. caroli splits independent blocks, such as the spin
  blocks of BHZ.

## Problems (`transport.toml`)

Two-terminal transmission of a length × width sample (length = width by default) between clean
leads. caroli's recursive Green's function (`caroli-rgf`) runs up to W = 256 only: it costs
O(L W³).

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

## Running

```bash
cargo build --release -p caroli-bench
./target/release/caroli-bench prepare
./target/release/caroli-bench run                                  # caroli and Kwant, 1 thread
./target/release/caroli-bench run --code caroli --threads 14       # caroli on all cores
./target/release/caroli-bench report > results/report.md
```

Results are appended to `results/<machine>.jsonl`, one JSON line per run. The report keeps
the latest run of each case.
