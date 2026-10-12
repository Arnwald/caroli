# How caroli is made

## The story

Arnaud Demion did his PhD on electronic transport in topological insulators and graphene, with
codes written in Fortran 90 and MPI. Years later, in his professional work, he had Claude, an AI
model by Anthropic, write an optimised boundary element method (BEM) solver. The gains over the
codes he had been using were very large. He then wondered whether the same approach would work
for quantum transport, and started again from the codes of his thesis.

caroli is the result. From the start, the aim was a library worth publishing: as complete as
Kwant, easier to use, faster than every code it is compared with, and with every result checked
against the thesis.

## Who does what

The code is written by Claude, under the supervision of Arnaud Demion. This is stated from the
first line of the project, and every commit says so (`Co-Authored-By: Claude`).

- **Arnaud Demion** sets the scientific scope (the physics of his thesis and what a transport
  toolbox needs), the goals and the constraints, provides the thesis codes and data, decides
  between the options he is offered, reviews the results and the physics, and has the last word
  on what is kept.
- **Claude** (Claude Opus 5.5, in Claude Code) proposes the architecture and the algorithms,
  writes the code, the tests, the documentation and the benchmark harness, runs the validation
  and the benchmarks on the author's machines, and reports what it finds, failures included.

Nothing is accepted on the AI's word. Every result is checked:

- **Against the thesis.** `reference/baseline` compiles the original Fortran programs and
  records the outputs of 26 cases; caroli must reproduce all of them
  (`cargo test -p caroli-validation`).
- **Against itself.** Each solver is tested against an independent one: nested dissection
  against the recursive Green's function, mode matching against decimation, the fast lead path
  against the general one, the real pencil against the complex one.
- **Against physics.** Conductance steps, quantum Hall plateaus, unitarity of the scattering
  matrix, current conservation, Onsager relations and gauge invariance.
- **Against the other codes.** Before any time is compared, the benchmarks check that Kwant,
  TBtrans and Quantica.jl give the same transmissions as caroli on the same Hamiltonians.

## Why Rust

Rust has become a mainstream language. Its learning curve is steep because the compiler is
strict, but the rules soon become intuitive, and that is probably part of why it spread so fast.
For a scientific library it brings:

- **Speed.** Native code through LLVM, no garbage collector, no runtime and abstractions that
  cost nothing at run time, so performance is predictable: no just-in-time warm-up and no
  collection pauses.
- **Memory safety.** Safe Rust has no out-of-bounds writes, use-after-free or data races. The
  thesis codes had several such bugs (`reference/thesis/README.md`); here they are compile errors
  or immediate panics.
- **Parallelism without fear.** Energies, disorder realisations and the branches of a
  dissection tree run in parallel (rayon), and the compiler rejects any data race.
- **Tooling.** One tool, Cargo, builds, tests, benchmarks, documents and fetches dependencies;
  rustfmt, clippy and rust-analyzer come with it, and documentation examples are tests.
- **Ecosystem.** crates.io is large and active (faer, rayon, serde...). Rust calls C and Fortran
  libraries directly (here Apple's Accelerate LAPACK), and can be called from Python (PyO3) or run
  in a browser (WebAssembly).
- **Distribution.** A library or a single binary, with nothing to install on a cluster: no
  interpreter, no environment.
- **Stability.** Editions and backward compatibility: code written today keeps building.
- **A popular language.** Questions have answers online, and companies increasingly ask for
  Rust. Fortran and Julia serve scientific computing well, but in narrower circles; for a PhD
  student, learning Rust is also an investment.
- **A good partner for an AI.** The compiler explains each error in plain language, often with
  the fix. With the type system, most mistakes surface before the code runs, which suits code
  written by a model and reviewed by a person.
