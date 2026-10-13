# caroli

A Rust toolbox for studying electronic phenomena in materials: build a tight-binding
Hamiltonian, define the geometry and the disorder, and pick a simulation method — from a library
API or from an interface.

Named after the Caroli formula for the transmission, `T = Tr[Γ_L G Γ_R G†]`, and the
Caroli–Combescot–Nozières–Saint-James paper (1971), the first non-equilibrium (Keldysh)
calculation of a tunnelling current.

> Status: early development. The crates build tight-binding systems and compute transport with
> Green's functions: two- and multi-terminal transmissions by nested dissection, magnetic
> fields, scattering states, local densities and currents, Keldysh observables. They are
> validated against the outputs of the Fortran codes of the author's PhD thesis
> (`reference/baseline`), whose two studies the toolbox will reproduce as worked examples, and benchmarked against Kwant, TBtrans and
> Quantica.jl (`benchmarks/`). See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and the
> roadmap, and [docs/kwant-to-caroli.md](docs/kwant-to-caroli.md) if you come from Kwant.

**Benchmarks and figures:** [caroli benchmarks](https://claude.ai/artifact/Mvd6XU9f71A38wuHoynoWi),
caroli against Kwant, TBtrans and Quantica.jl, with local spin currents, quantum Hall edges and
the Hamiltonians.

**How it is made.** caroli is written by Claude (Anthropic's AI model, in Claude Code) under my
supervision: I set the scope from my PhD thesis, decide between the options and validate every
result against my thesis codes. The story, the roles and the choice of Rust are in
[docs/development.md](docs/development.md). — Arnaud Demion

```rust
use caroli::prelude::*;

// Clean BHZ ribbon (HgTe quantum well), 16 × 10 sites: two helical edge channels in the gap.
let model = bhz(BhzParams::default());
let region = Region::cuboid(model.lattice(), [16, 10, 0]);
let device = TwoTerminal::from_model(&model, &model, &region, 0);
assert!((device.transmission(0.0)? - 2.0).abs() < 1e-8);
```

## Scope

- **System**: tight-binding Hamiltonian (orbitals, spin, hoppings, Peierls phases).
- **Geometry**: crystal shape, lattice size, 2D or 3D, leads/contacts.
- **Potential**: Anderson disorder, magnetic disorder (classical spins, cone/polarisation), other
  on-site or hopping perturbations.
- **Simulation**:
  - finite differences / time evolution (split-operator),
  - steady state with Green's functions (recursive Green's function, lead self-energies,
    Landauer–Büttiker/Caroli conductance, multi-terminal),
  - out of equilibrium with Keldysh (G^<, local currents, bias windows),
  - spectral tools (exact diagonalisation, ribbon bands, KPM density of states),
  - disorder averaging (self-consistent Born approximation),
  - discrete-time quantum walks: a walker coupled to spins on the edges of a graph
    (Sellapillay & Verga, `caroli-walk`).
- **Acceleration**: all the cores, and the GPU for the Chebyshev kernels (Metal, and CUDA in
  f64) and the quantum walks (Metal), in reduced precision if the user chooses it: the CPU
  checks a sample of the work and takes over when the error would exceed the tolerance. The
  transport solvers can run in single precision too, as an experimental preview mode (1.1–1.9×
  faster, with errors up to 10⁻² and more in wide samples).
- **Delivery**: a workspace of reusable crates plus a user-facing interface.

## Examples (from the thesis)

Self-contained programs in `crates/caroli/examples`, with the scripts that plot their figures:

1. **2D topological insulator** (`ti_magnetic`) — BHZ model of HgTe quantum wells with magnetic
   impurities: spin-resolved conductance maps, local currents and densities of states, Born
   phase diagrams (anomalous quantum Hall effect induced by disorder).
2. **Graphene with magnetic impurities** (`graphene_field`) — non-linear transport and pair
   creation under a strong electric field, wave-packet spreading, densities of states and
   conductivity.

```bash
cargo run --release -p caroli --example ti_magnetic -- all out/ti
python3 crates/caroli/examples/plot_ti_magnetic.py out/ti
```

## Conventions

- Magnetic exchange coupling in the BHZ model: **same-g** (g_E = g_H, spin matrix s⊗1) and
  **opposite-g** (g_E = −g_H, s⊗σ_z). The thesis codes call these "ferro" and "antiferro"
  (`ferro_ge=gh`, `antiferro_ge=-gh`). These are not magnetic orders.

## Layout

```
crates/              Rust workspace (cargo test --workspace); see ARCHITECTURE.md
reference/baseline/  golden outputs of 26 small cases run by the thesis codes, the regression
                     baseline for caroli; see reference/baseline/README.md
```
