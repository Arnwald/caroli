# Contributing to caroli

Thank you for your interest. caroli is a young project: bug reports, questions, physics checks and
comparisons with other codes are as welcome as code.

## Asking for help, reporting a bug

Open an issue on the [issue tracker](https://github.com/Arnwald/caroli/issues). For a bug, give:

- what you ran (a minimal program or input file), the caroli commit and your platform;
- what you expected, and what you got (the full error message or the wrong numbers);
- if the result disagrees with another code (Kwant, TBtrans, Quantica.jl, KITE…), the same
  system in that code.

## Building and testing

caroli needs a stable Rust toolchain (1.85 or later, edition 2024).

```bash
cargo build --workspace
cargo test --workspace          # must pass before any change is merged
cargo clippy --workspace --all-targets
cargo fmt --all
```

On macOS the dense kernels call Apple's Accelerate framework (macOS 13.3 or later); elsewhere they
use faer and in-house kernels, which are too slow unoptimised: there, run the tests with
`cargo test --workspace --profile ci` (optimised, with the debug assertions). Both backends are
tested in CI. The CUDA backend of `caroli-gpu` is behind the `cuda` feature and needs an NVIDIA
driver at run time.

## What a change needs

- **Tests.** A new solver or method is checked against an independent one (another solver, an
  analytic result, a conservation law, or another code on the same Hamiltonian), as described in
  [How caroli is made](docs/development.md). A test that only checks that the code runs is not
  enough.
- **The validation stays green.** `caroli-validation` compares caroli with the outputs of the
  author's thesis codes (`reference/baseline/golden`). These outputs are a frozen regression
  baseline: they are never re-recorded to make a change pass.
- **Documentation.** Public items have doc comments, with an example when it helps.
  `ARCHITECTURE.md` records the design choices and the roadmap: update it with the change.
- **Style.** `cargo fmt` and `cargo clippy` clean. Code, comments, documentation and commit
  messages are in English.
- **Conventions.** For the exchange coupling of magnetic impurities in the BHZ model, write
  **same-g** and **opposite-g** (g_E = ±g_H), not "ferro"/"antiferro" (these are not magnetic
  orders).
- **Code you have the right to publish.** caroli is MIT-licensed. Do not copy code from other
  projects unless its license allows it and you say where it comes from.

Open a pull request against `main` with a description of what changes and how it was checked.

## How the project is run

caroli is maintained by Arnaud Demion, who sets its scope and has the last word on what is merged.
Most of its code is written by Claude (Anthropic's AI model, in Claude Code) under his supervision;
[How caroli is made](docs/development.md) describes who does what and how every result is
checked. Contributions written with AI tools are welcome under the same rule: you have reviewed
them, you can explain them, and they come with their checks.
