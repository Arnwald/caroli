# caroli

A Rust toolbox for studying electronic phenomena in materials: build a tight-binding
Hamiltonian, define the geometry and the disorder, and pick a simulation method — from a library
API or from an interface.

Named after the Caroli formula for the transmission, `T = Tr[Γ_L G Γ_R G†]`, and the
Caroli–Combescot–Nozières–Saint-James paper (1971), the first non-equilibrium (Keldysh)
calculation of a tunnelling current.

> Status: early development. The first crates build tight-binding systems and compute
> two-terminal transmissions with Green's functions. They are validated against the Fortran
> codes of the author's PhD thesis (`reference/`), which the toolbox will reproduce as its two
> worked examples. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and the roadmap.

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
  - disorder averaging (self-consistent Born approximation).
- **Acceleration**: CUDA and Metal backends where possible.
- **Delivery**: a workspace of reusable crates plus a user-facing interface.

## Examples (from the thesis)

1. **2D topological insulator** — BHZ model of HgTe quantum wells with magnetic impurities:
   two- and four-terminal conductance, local currents, Born phase diagrams (anomalous quantum Hall
   effect induced by disorder).
2. **Graphene with magnetic impurities** — non-linear transport and pair creation under a strong
   electric field, wave-packet spreading, KPM density of states.

## Conventions

- Magnetic exchange coupling in the BHZ model: **same-g** (g_E = g_H, spin matrix s⊗1) and
  **opposite-g** (g_E = −g_H, s⊗σ_z). The thesis codes call these "ferro" and "antiferro"
  (`ferro_ge=gh`, `antiferro_ge=-gh`). These are not magnetic orders.

## Layout

```
crates/              Rust workspace (cargo test --workspace); see ARCHITECTURE.md
reference/thesis/    frozen copy of the thesis codes (Fortran 90 + MPI, Python 2), documented;
                     see reference/thesis/README.md
reference/baseline/  builds the thesis codes and records golden outputs of 26 small cases,
                     the regression baseline for caroli; see reference/baseline/README.md
```
