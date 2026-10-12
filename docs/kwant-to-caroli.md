# caroli for Kwant users

How the usual Kwant workflow translates into caroli. Every snippet below comes from code that
runs in the test suite: `crates/caroli/tests/tutorials.rs` (Kwant's tutorials) and the tests of
`caroli-green`.

```rust
use caroli::prelude::*;
```

## Systems

| Kwant | caroli |
|---|---|
| `kwant.lattice.square(a, norbs=n)` | `Lattice::square(a, n)` |
| `kwant.lattice.honeycomb(a, norbs=n)` | `Lattice::honeycomb(a, n)` |
| `kwant.lattice.general(vectors, basis, norbs)` | `Lattice::new(&vectors).with_sublattice(name, offset, orbitals)` |
| `syst[lat.shape(f, start)] = onsite` | `Region::shaped(&lattice, extent, f)`, `Region::cuboid`, `Region::from_sites` |
| `syst[...] = value` (uniform) | `Model::new(lattice).onsite(sublattice, value)` |
| `syst[kwant.builder.HoppingKind(δ, a, b)] = t` | `.hopping(a, b, -δ, t)` |
| value functions of the site | `.with_onsite_term(\|site\| Some(matrix))` |
| value functions of a hopping | `.with_hopping_term(\|from, to, t\| Some(matrix))` |
| Peierls phases, `kwant.physics.magnetic_gauge` | `.with_peierls(φ)`, or `.with_field(b)` for a uniform field |
| `kwant.continuum.discretize("k_x**2 + ...")` | `Continuum::parse("k_x^2 + ...", &params)?.discretize(a)` |

The model holds what Kwant's builder holds, and the region holds the sites. A hopping
`⟨(R, from)| H |(R + offset, to)⟩ = t` is Kwant's `HoppingKind(−offset, from, to)`; its
conjugate is implied.

Values are a `CMat`, or a number for one orbital. Spin and orbital matrices come from
`s0(), sx(), sy(), sz()`, `kron` and `combine`:

```rust
let (t, alpha, e_z) = (1.0, 0.5, 0.08);
let i = c64::new(0.0, 1.0);
let wire = Model::new(Lattice::square(1.0, 2))
    .onsite(0, combine(&[(4.0 * t, &s0()), (e_z, &sz())]))
    .hopping(0, 0, [1, 0, 0], combine(&[(c64::from(-t), &s0()), (i * alpha / 2.0, &sy())]))
    .hopping(0, 0, [0, 1, 0], combine(&[(c64::from(-t), &s0()), (-i * alpha / 2.0, &sx())]));
```

## Leads

- **Two leads along an axis** (the usual wire or ribbon). The region is cut into slices along
  primitive vector `axis`. Both leads continue the first and last slices with the clean `lead`
  model:

  ```rust
  let region = Region::cuboid(wire.lattice(), [30, 10, 0]);
  let sys = NestedTwoTerminal::from_model(&device, &clean, &region, 0);
  ```

- **Leads anywhere** (Hall bars, crosses, rings). Give one `Interface` per lead: the device
  sites where it attaches and the lattice step that points away from the device:

  ```rust
  let interfaces = [
      Interface { slice: column(-arm), outward: [-1, 0, 0] },
      Interface { slice: row(w + arm - 1), outward: [0, 1, 0] },
      // ...
  ];
  let sys = NestedMultiTerminal::from_model(&device, &clean, &region, &interfaces);
  ```

A lead must be the same from cell to cell. The constructors check it, and in a uniform field
(`with_field`) they take each lead to a gauge that is invariant along it, whatever its direction.

## Results

| Kwant | caroli |
|---|---|
| `kwant.smatrix(fsys, E).transmission(1, 0)` | `sys.transmission(E)` (two terminals) |
| `kwant.smatrix(fsys, E)` (mode basis) | `sys.smatrix(E)`: blocks `S_pq`, Bloch factors of the modes, `full()` |
| all `transmission(p, q)` | `sys.transmissions(E)[p][q]` (from `q` into `p`) |
| conservation law, `transmission((p, a), (q, b))` | `sys.sector_transmissions(E, &sectors)` |
| conductance matrix, then resistances | `lead_voltages(&t, &currents, ground)` |
| `smatrix.submatrix(...)`: transmission eigenvalues | `sys.transmission_eigenvalues(E)`, `sys.fano_factor(E)` |
| `kwant.wave_function(fsys, E)` | `sys.scattering_states(E)`: `ψ_p = G W_p`, `W_p W_p† = Γ_p` |
| `kwant.ldos(fsys, E)` | `states.occupation(&[1.0; leads])` |
| `kwant.operator.Current(fsys)(ψ)` | `sys.bond_currents(&states, &occupations)` |
| `lead.modes(E)` | `Lead::modes(E)` (channels, velocities) |
| `kwant.physics.Bands(lead)` | `Lead::bands(k)` |
| `kwant.kpm.conductivity` | `caroli::kpm::kubo::{kubo_moments, kubo_bastin}` with `Model::sparse_velocity` |

`ScatteringStates` holds one matrix per lead (rows: device orbitals; columns: channels). The
scattering states mix the channels of a lead by a unitary, which leaves densities, currents and
transmissions unchanged. Mode-resolved quantities (which incoming mode goes to which outgoing
one) come from `smatrix`, whose modes carry unit current.

The Hall bar of the tutorials, with a current from the left lead to the grounded right lead and
the bottom and top leads as voltage probes:

```rust
let t = sys.transmissions(e)?;
let v = lead_voltages(&t, &[1.0, -1.0, 0.0, 0.0], 1);
let hall = v[3] - v[2]; // ±h/(ν e²) on the plateaus
```

## Speed

- `NestedTwoTerminal` and `NestedMultiTerminal` eliminate the device by nested dissection, as
  MUMPS does for Kwant, on Apple's Accelerate. `TwoTerminal` (recursive Green's function) stays
  for slice-by-slice quantities and as a reference.
- Lead modes take a Hermitian fast path when the coupling between lead cells is Hermitian and
  definite (square lattices, no spin-orbit coupling or field along the lead).
- Disorder averages: compute the lead self-energies once per energy and reuse them for every
  configuration:

  ```rust
  let sigma = reference.self_energies(e)?;
  let t = configuration.transmission_with(&sigma)?;
  ```

- `transmissions(&energies)` runs the energies in parallel.

## Not yet in caroli

Symmetry analysis (qsymm), position-dependent coefficients inside continuum expressions (add
potentials with `with_onsite_term` and varying hoppings with `with_hopping_term` on the
discretised model), and plotting.
