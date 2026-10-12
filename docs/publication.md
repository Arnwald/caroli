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
| public repository, browsable, with an open issue tracker | private until the toolbox is ready |
| at least six months of public history before submission | starts when the repository goes public |
| no third-party code without a license | removed from the tree and from the history (`reference/thesis/THIRD_PARTY.tsv`) |
| feature-complete ("no half-baked solutions") | transport done; time evolution, KPM, Born and the interface to finish |
| documentation, automated tests | API docs, tutorials, `cargo test --workspace` |
| packaging per the language's standards | crates.io release to prepare |
| CONTRIBUTING file, support and governance | to write |
| evidence of research use | the two thesis studies, rebuilt as examples |
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
- Add CONTRIBUTING, a citation file (`CITATION.cff`) and public CI (Linux, with the faer
  backend).
