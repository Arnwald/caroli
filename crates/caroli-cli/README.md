# caroli — the command

`caroli` reads a system and its calculations from a TOML file and writes one CSV table per
calculation.

```bash
cargo install --path crates/caroli-cli        # or: cargo run --release -p caroli-cli --
caroli check input.toml                       # builds the model and the sample, describes them
caroli run input.toml                         # runs every [[run]], tables in ./input/
caroli run input.toml --out results --set J=0.3 --only 0,2
```

`--set NAME=VALUE` overrides a parameter, `--only` picks calculations (0 is the first). Commented
examples are in [`inputs/`](inputs): a clean BHZ ribbon, magnetic impurities in a BHZ ribbon
(spin-resolved maps), the Qi–Wu–Zhang Chern insulator, graphene with magnetic impurities, a
continuum model.

## The file

```toml
title = "BHZ ribbon with magnetic impurities"

[parameters]          # named numbers, usable in every expression below
J = 0.4
M = -0.137

[model]               # the clean model
preset = "bhz"
mass = "M"

[[disorder]]          # any number of disorder terms
type = "magnetic"
concentration = 0.2
exchange = "J"
moments = "polarized"
spin = "opposite-g"
cells = { x = [10, 50] }

[system]              # the sample
size = [60, 32]
leads = ["-x", "+x"]

[[run]]               # any number of calculations
type = "transmission"
energies = { from = -0.2, to = 0.2, count = 41 }
scan = { J = { from = 0, to = 1, count = 11 } }
configurations = 10
```

Unknown keys are errors, so that a misspelt option does not go unnoticed.

### Numbers and expressions

Wherever a number is expected, a string is an expression of the parameters: `"J/2"`,
`"2*pi/3"`, `"1/sqrt(3)"`. Matrices are expressions too, with the Pauli matrices `s0 sx sy sz`,
`kron(a, b)` (the index of `b` fastest), the imaginary unit `i`, `+ − * /`, integer powers `^`,
`pi` and `sqrt exp ln sin cos tan abs`. A number stands for that multiple of the identity.

Lists of values (`energies`, scans, `k`): one number, a list, `{ from, to, count }` (both ends
included) or `{ from, to, step }`.

### `[parameters]`

Named numbers, or expressions of the other parameters (in any order, without cycles). A scanned
parameter must be declared here; the parameters that depend on it follow its values.

### `[model]`

Exactly one of:

- `preset`:
  - `bhz`: the BHZ model of HgTe quantum wells on a square lattice, basis `(E1↑, H1↑, E1↓, H1↓)`;
    options `a b c d mass lambda` (defaults: the HgTe values of the thesis, `A = 1`,
    `B = −0.376406`, `C = 0`, `D = −0.280933`, `M = −0.137`, `λ = 0`).
  - `bhz-up`, `bhz-down`: one spin block (two orbitals `(E1, H1)`).
  - `graphene`: honeycomb lattice, nearest-neighbour distance `spacing` (default 1), hopping
    `−t` (default `t = 1`); `spin = true` for two orbitals per site.
  - `chain`, `square`, `cubic`: hopping `−t` between nearest neighbours, lattice constant
    `spacing`; `spin = true` for two orbitals per site.
  - `qwz`: the Qi–Wu–Zhang Chern insulator, mass `m` (default 1).
- `continuum`: a k·p Hamiltonian in `k_x k_y k_z` discretised on a chain, square or cubic lattice
  of constant `spacing` (as Kwant's `continuum.discretize`).
- `lattice`: a lattice of one's own,
  ```toml
  [model.lattice]
  vectors = [[1, 0], [0.5, 0.866]]
  sublattices = [{ name = "a", position = [0, 0], orbitals = 2 }]
  ```
  with its terms below.

Terms (for any model):

- `[[model.onsite]]` `sublattice`, `matrix`: the on-site matrix of a sublattice (set for a
  lattice of one's own, added to a preset).
- `[[model.hopping]]` `from`, `to` (sublattices), `offset` (cells, one integer per lattice
  direction), `matrix`: `⟨(R, from)| H |(R + offset, to)⟩`; the reverse hop is added.
- `field`: a uniform magnetic field, as a phase per unit area (the flux quantum is 2π).

### `[[disorder]]`

- `type = "anderson"`, `strength = W`: on-site energies uniform in `[−W/2, W/2]` on every orbital
  of a site.
- `type = "magnetic"`: classical moments on a fraction `concentration` of the sites, coupled by
  `exchange · n·S`.
  - `moments`: `isotropic` (default), `polarized` (mean `mz` along z, `n_z` uniform in
    `[2 mz − 1, 1]`) or `cone` (half-angle `theta0` in degrees around z).
  - `spin`: the matrices `S`, `same-g` or `opposite-g` (BHZ: `g_E = ±g_H`), `pauli` (the default
    for two orbitals), or three matrix expressions.
- `cells = { x = [first, end], y = [...] }`: where it lies (half-open cell ranges, the whole
  sample by default); `sublattices = [...]`: which sublattices carry it.
- `seed`: configuration `c` of a calculation draws with `seed + c`.

The Born approximation (`scba`) takes the disorder as its averages over every site (the `cells`
windows do not apply).

### `[system]`

- `size`: cells along each lattice direction.
- `periodic = true`: a torus (spectra, densities of states, conductivities).
- `leads`: faces of the sample with a semi-infinite clean lead, among `-x +x -y +y -z +z`
  (default `["-x", "+x"]`). The leads continue the clean model; `-x` and `+x` alone use the
  two-terminal solver, any other set the multi-terminal one. Leads are numbered in this order.

### `[[run]]`

Every calculation takes `output` (the file name, default `NN-type.csv`), and most take `scan`
(parameters to scan, every combination) and `configurations` (disorder draws, one row each). The
tables start with the scanned parameters and the configuration.

| `type` | what | options | columns |
|---|---|---|---|
| `transmission` | transmissions between the leads | `energies`, `sectors` (labels of the orbitals of a lead site, e.g. `[0, 0, 1, 1]` for the BHZ spins) | `energy`, `T` (two leads) or `T_p_q` from lead `q` into lead `p`, or `T_p.a_q.b` between sectors |
| `local` | scattering states, averaged over `energies` | `occupations` (per lead, default `[1, 0, ...]`), `operators` (name → on-site matrix) | per site: `x y z`, current `jx jy jz`, occupied `density`, local density of states `ldos`, and per operator its density and current |
| `bands` | bands of the ribbon (the cross-section repeated along the first lead axis) | `k` (Bloch phases, default 201 in [−π, π]); or `path` (corners in Bloch phases, one per lattice direction) and `points` per segment for the bulk bands | `k` or `s` (length along the path), `band`, `energy` |
| `spectrum` | eigenvalues of the sample (dense, up to 8000 orbitals) | | `index`, `energy` |
| `dos` | density of states by the kernel polynomial method | `energies`, `moments` (512), `vectors` (10), `seed`, `bounds`, `projections` (name → on-site matrix) | `energy`, `dos` (per sample), projected densities |
| `conductivity` | Kubo–Bastin conductivities of a periodic sample, e²/h | `energies` (chemical potentials), `components` (`["xx", "xy"]`), `temperature`, `moments` (256), `vectors`, `seed`, `bounds` | `energy`, `sigma_xx`, ... |
| `scba` | self-consistent Born approximation of the disorder | `energies`, `k_grid` (64 per direction), `eta` (1e-3) | `energy`, `dos` (per cell), `sigma_re_k sigma_im_k` (diagonal of the self-energy), `residual` |
| `evolution` | time evolution on an n × n torus under a uniform force | `force` (`k → k + F t`), `dt`, `steps`, `every`, `fermi`, `initial` | `t`, `norm`, current density `jx jy`, `excited` (carriers above `fermi` in the instantaneous clean bands, per unit area), `width` |

`initial` of `evolution`: `{ type = "fermi-sea", fermi, vectors, seed, moments }` (random-phase
superpositions of the occupied states; with disorder, the ground state of the disordered
Hamiltonian by a Chebyshev projection), or `{ type = "localized", cell = [x, y], orbital,
window = [lo, hi] }` (one orbital, projected on the clean band states in the window).
