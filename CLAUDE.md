# CLAUDE.md — caroli

Rust toolbox for electronic phenomena in materials (tight-binding systems, geometry, disorder;
finite-difference, steady-state Green's function and Keldysh solvers; CUDA/Metal acceleration).
Read `README.md` for the scope.

## Status

- Design phase: no Rust code yet.
- `reference/thesis/` holds the author's PhD thesis codes (Fortran 90 + MPI, Python 2), deduplicated
  and documented. Start with `reference/thesis/README.md`; each program has its own README
  (physics, inputs, outputs, versions, known bugs, modernisation notes).
- `reference/thesis/` is a **frozen reference**: do not edit the sources there. Record bugs and
  findings in its READMEs. It is the regression baseline for the Rust rewrite.
- The thesis data (5.3 GB of results) are NOT in the repo. They live on the author's machine in
  `~/Projects/electronic-transport/`, and `reference/thesis/tools/coverage.py` can only check
  coverage there.

## Conventions

- Everything in English: code, docs, comments, commit messages.
- Magnetic exchange coupling: use **same-g** / **opposite-g** (g_E = ±g_H). The thesis
  "ferro"/"antiferro" labels stay only where they name original files or folders.
- The repository is private until the toolbox is ready and the rights to the reference code are
  settled. It includes code by collaborators: Laurent's contact modules and A. Verga's FFTW wrapper.
