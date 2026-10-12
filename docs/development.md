# How caroli is made

## The story

I did my PhD on electronic transport in topological insulators and graphene, with codes written
in Fortran 90 and MPI. Years later, for my professional work, I had Claude, Anthropic's AI model,
build an optimised boundary element method (BEM) solver. The gains over the codes available were
huge, and I was curious to do the same with quantum transport codes, starting again from the
codes of my thesis.

caroli is the result. From the start, the aim was a library worth publishing: as complete as
Kwant, easier to use, faster than every code it is compared with, and with every result checked
against my thesis.

## Who does what

Claude writes the code under my supervision. I say so from the first line of the project, and
every commit says it too (`Co-Authored-By: Claude`).

- **I** set the scientific scope (the physics of my thesis and what a transport toolbox needs),
  the goals and the constraints. I provide the thesis codes and data, decide between the options
  Claude offers, review the results and the physics, and have the last word on what is kept.
- **Claude** (Claude Opus 5.5, in Claude Code) proposes the architecture and the algorithms,
  writes the code, the tests, the documentation and the benchmark harness, runs the validation
  and the benchmarks on my machines, and reports what it finds, failures included.

Nothing is accepted on the AI's word. Every result is checked:

- **Against my thesis.** `reference/baseline` compiles the original Fortran programs and records
  the outputs of 26 cases; caroli must reproduce all of them (`cargo test -p caroli-validation`).
- **Against itself.** Each solver is tested against an independent one: nested dissection
  against the recursive Green's function, mode matching against decimation, the fast lead path
  against the general one, the real pencil against the complex one.
- **Against physics.** Conductance steps, quantum Hall plateaus, unitarity of the scattering
  matrix, current conservation, Onsager relations and gauge invariance.
- **Against the other codes.** Before any time is compared, the benchmarks check that Kwant,
  TBtrans and Quantica.jl give the same transmissions as caroli on the same Hamiltonians.

## Why Rust

Rust has become a mainstream language. Its learning curve is steep because its compiler is very
strict, but it quickly becomes intuitive, and that is probably a big part of its popularity. Its
other strengths, for a scientific library:

- **Speed.** Native code through LLVM, no garbage collector, no runtime and abstractions that
  cost nothing at run time, so performance is predictable: no just-in-time warm-up and no
  collection pauses.
- **Memory safety.** Safe Rust has no out-of-bounds writes, use-after-free or data races. My
  thesis codes had several such bugs (`reference/thesis/README.md`); here they are compile errors
  or immediate panics.
- **Parallelism without fear.** Energies, disorder realisations and the branches of a
  dissection tree run in parallel (rayon), and the compiler rejects any data race.
- **Tooling.** One tool, Cargo, builds, tests, benchmarks, documents and fetches dependencies;
  rustfmt, clippy and rust-analyzer come with it, and documentation examples are tests.
- **Ecosystem.** Unlike Fortran or Julia, which remain niche languages, Rust has an extremely
  rich ecosystem (faer, rayon, serde...), and an answer to almost any problem is easy to find
  online. It also calls C and Fortran libraries directly (here Apple's Accelerate LAPACK), and can
  be called from Python (PyO3) or run in a browser (WebAssembly).
- **Distribution.** A library or a single binary, with nothing to install on a cluster: no
  interpreter, no environment.
- **Stability.** Editions and backward compatibility: code written today keeps building.
- **A good partner for an AI.** I believe its compiler is a huge advantage for agentic
  development: it explains every error, exhaustively and in plain language, often with the fix.
  With the type system, most mistakes surface before the code runs.

All in all, a popular language is not a problem, quite the opposite. And since companies ask for
Rust more and more, learning it is also an asset for a PhD student.
