# CLAUDE.md — caroli

Rust toolbox for electronic phenomena in materials (tight-binding systems, geometry, disorder;
finite-difference, steady-state Green's function and Keldysh solvers; CUDA/Metal acceleration).
Read `README.md` for the scope.

## Status

- Early development: `crates/` holds the library crates (lattice, model, dense kernels, Green's
  functions, disorder, Keldysh, Born, torus, KPM, dynamics, GPU, quantum walks), the `caroli`
  facade, the `caroli` command (caroli-cli), caroli-validation (tests against the thesis golden
  outputs) and caroli-bench (the benchmarks); the table of `ARCHITECTURE.md` lists them. `ARCHITECTURE.md` has the design
  choices and the roadmap; keep it up to date. `cargo test --workspace` must stay green (on Linux,
  `cargo test --workspace --profile ci`: the dense kernels need optimised code without
  Accelerate), and so must `cargo clippy`. CI runs the tests with `--profile ci` on Linux and
  macOS.
- The author's PhD thesis codes (Fortran 90 + MPI, Python 2), deduplicated and documented, are
  NOT in this public repository. They live in a private repository on the author's machine,
  `~/Projects/caroli-thesis/thesis` (start with its `README.md`; each program has its own README:
  physics, inputs, outputs, versions, known bugs, modernisation notes). Never copy their code
  into this repository; the models and the maths are written from their documentation.
- The thesis reference is a **frozen reference**: do not edit the sources there. Record bugs and
  findings in its READMEs. Its outputs (`reference/baseline/golden`) are the regression baseline
  for the Rust rewrite.
- `reference/baseline/` compiles the thesis programs (gfortran, serial MPI shim, LAPACK95, FFTW)
  from `THESIS_SRC` and checks 26 small cases against `golden/`. Run `cargo run --release -p caroli-validation --bin baseline -- run`
  after rebuilding (`deps`, then `build`). Never re-record goldens (`--update`) without saying why.
- The thesis data (5.3 GB of results) are NOT in the repo. They live on the author's machine in
  `~/Projects/electronic-transport/`, and `cargo run --release -p caroli-validation --bin baseline -- coverage` can only check
  coverage there.

## Conventions

- Everything in English: code, docs, comments, commit messages.
- No Python: tooling is Rust. Python is only allowed for example scripts that plot paper figures.
- Commit and push each validated step directly to `main`.
- Magnetic exchange coupling: use **same-g** / **opposite-g** (g_E = ±g_H). The thesis
  "ferro"/"antiferro" labels stay only where they name original files or folders.
- The repository is public. Neither the thesis codes nor the code of collaborators (Laurent's
  contact modules, A. Verga's FFTW wrapper and scripts; listed in the thesis reference's
  `THIRD_PARTY.tsv`) may enter it or its history.
