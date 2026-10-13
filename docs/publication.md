# Publication plan

caroli is meant to be published in the [Journal of Open Source Software](https://joss.theoj.org)
(JOSS). This page records what JOSS asks and where the project stands (October 2026).

## AI-written code

JOSS accepts software written with generative AI, under conditions
([policies](https://joss.readthedocs.io/en/latest/policies.html)):

- The paper needs an **AI usage disclosure** section: the tools and their versions, where they
  were used (code, documentation, paper), what they did (code generation, refactoring, tests,
  writing), and a statement that people reviewed and validated everything and took the main
  decisions.
- The submission must show the human contributions that could not be replaced: framing the
  problem, the key decisions.
- An inaccurate disclosure is an ethics violation (from desk rejection to institutional
  notification).

caroli's disclosure will say what [How caroli is made](development.md) says: the code, tests,
documentation and benchmarks are written by Claude (Claude Opus 5.5, in Claude Code) under my
supervision; I set the scope and the goals, decide between the options, validate the results
against my thesis and have the last word.

## Requirements

From the [submission guide](https://joss.readthedocs.io/en/latest/submitting.html):

| requirement | status |
|---|---|
| OSI license | MIT |
| public repository, browsable, with an open issue tracker | public since October 2026 |
| at least six months of public history before submission | from October 2026: April 2027 at the earliest |
| no code without the right to publish it | the thesis codes and the code of collaborators are out of the tree and the history; only the thesis outputs stay, as regression data |
| feature-complete ("no half-baked solutions") | transport, KPM, Born and time evolution from a model; the `caroli` command (TOML inputs); Python bindings to consider |
| documentation, automated tests | API docs, tutorials, `cargo test --workspace` |
| packaging per the language's standards | crates.io release to prepare |
| CONTRIBUTING file, support and governance | `CONTRIBUTING.md`; `CITATION.cff` |
| evidence of research use | the two thesis studies, rebuilt as examples (`ti_magnetic`; `graphene_field` to match the paper's figures) |
| a short `paper.md`: summary, statement of need, references, AI disclosure | to write |

## Benchmarks are not the paper

JOSS papers "must not focus on new research results": the paper can cite the benchmarks
(`benchmarks/`), not be about them. A paper on the methods and the measurements belongs in a
journal such as Computer Physics Communications, where TBtrans and TranSIESTA were published,
or SciPost Physics Codebases; their policies on AI-written code are to be checked.

## Before going public

- The history no longer contains the third-party files, but GitHub may keep the old commits for
  a while. For the public release, push the cleaned history to a new repository (or ask GitHub
  support to purge the old objects).
- CONTRIBUTING, the citation file (`CITATION.cff`) and the public CI (`.github/workflows/ci.yml`:
  fmt and clippy, tests on Linux with the faer backend and on macOS with Accelerate) are in place.
