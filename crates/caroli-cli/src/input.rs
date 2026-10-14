//! The input file: TOML tables read with serde. Unknown keys are errors, so that a misspelt
//! option does not pass unnoticed.

use serde::Deserialize;
use std::collections::BTreeMap;

/// A number, or an expression of the parameters (`"J/2"`, `"2*pi/3"`; see
/// [`caroli::model::continuum`] for the syntax).
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Num {
    Value(f64),
    Expr(String),
}

/// Values of a scanned quantity: one number, a list, or a range.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Values {
    One(Num),
    List(Vec<Num>),
    Range(Range),
}

/// `{ from, to, count }` (both ends included) or `{ from, to, step }`.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Range {
    pub from: Num,
    pub to: Num,
    pub count: Option<usize>,
    pub step: Option<Num>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub title: Option<String>,
    /// Named numbers (or expressions of the others) used by every expression of the file.
    #[serde(default)]
    pub parameters: BTreeMap<String, Num>,
    pub model: ModelInput,
    #[serde(default)]
    pub disorder: Vec<DisorderInput>,
    #[serde(default)]
    pub system: SystemInput,
    #[serde(default, rename = "run")]
    pub runs: Vec<RunInput>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelInput {
    /// `bhz`, `bhz-up`, `bhz-down`, `graphene`, `chain`, `square`, `cubic`, `qwz`.
    pub preset: Option<String>,
    /// A k·p Hamiltonian discretised on a chain, square or cubic lattice of constant `spacing`.
    pub continuum: Option<String>,
    /// A lattice of one's own, with `onsite` and `hopping` entries.
    pub lattice: Option<LatticeInput>,
    /// Lattice constant (continuum, chain, square, cubic) or nearest-neighbour distance
    /// (graphene). Default 1.
    pub spacing: Option<Num>,
    /// BHZ parameters (defaults: the HgTe values of the thesis).
    pub a: Option<Num>,
    pub b: Option<Num>,
    pub c: Option<Num>,
    pub d: Option<Num>,
    pub mass: Option<Num>,
    pub lambda: Option<Num>,
    /// Hopping energy of the graphene, chain, square and cubic presets (`−t` between neighbours,
    /// default 1); mass of the Qi–Wu–Zhang model.
    pub t: Option<Num>,
    pub m: Option<Num>,
    /// Spin degeneracy for the graphene, chain, square and cubic presets (two orbitals per site,
    /// spin fastest).
    #[serde(default)]
    pub spin: bool,
    /// On-site matrices: set for a lattice of one's own, added to a preset.
    #[serde(default)]
    pub onsite: Vec<OnsiteInput>,
    /// Hoppings added to the model.
    #[serde(default, rename = "hopping")]
    pub hoppings: Vec<HoppingInput>,
    /// Uniform magnetic field: phase per unit area (the flux quantum is 2π).
    pub field: Option<Num>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatticeInput {
    /// Primitive vectors (1 to 3, each with 1 to 3 components).
    pub vectors: Vec<Vec<f64>>,
    pub sublattices: Vec<SublatticeInput>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SublatticeInput {
    pub name: Option<String>,
    #[serde(default)]
    pub position: Vec<f64>,
    #[serde(default = "one")]
    pub orbitals: usize,
}

fn one() -> usize {
    1
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OnsiteInput {
    #[serde(default)]
    pub sublattice: usize,
    pub matrix: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HoppingInput {
    #[serde(default)]
    pub from: usize,
    #[serde(default)]
    pub to: usize,
    /// Cell offset of the `to` site (one integer per lattice direction).
    pub offset: Vec<i64>,
    pub matrix: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DisorderInput {
    /// `anderson` or `magnetic`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Anderson: on-site energies uniform in [−W/2, W/2].
    pub strength: Option<Num>,
    /// Magnetic: fraction of the sites carrying a moment, exchange `J` (`J n·S`).
    pub concentration: Option<Num>,
    pub exchange: Option<Num>,
    /// `isotropic`, `polarized` (mean `mz` along z) or `cone` (half-angle `theta0`, degrees).
    pub moments: Option<String>,
    pub mz: Option<Num>,
    pub theta0: Option<Num>,
    /// Spin matrices `S`: `same-g` or `opposite-g` (BHZ), `pauli` (two orbitals), or three
    /// matrix expressions.
    pub spin: Option<SpinInput>,
    /// Cells where the disorder lies: `[first, end)` per lattice direction (`x`, `y`, `z`).
    pub cells: Option<CellsInput>,
    /// Sublattices that carry it (all by default).
    pub sublattices: Option<Vec<usize>>,
    /// Seed of the first configuration; configuration `c` uses `seed + c`.
    pub seed: Option<u64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum SpinInput {
    Name(String),
    Matrices(Vec<String>),
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CellsInput {
    pub x: Option<[i64; 2]>,
    pub y: Option<[i64; 2]>,
    pub z: Option<[i64; 2]>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SystemInput {
    /// Cells along each lattice direction.
    pub size: Option<Vec<usize>>,
    /// Periodic boundaries in every direction (a torus): spectra, densities of states,
    /// conductivities, time evolution.
    #[serde(default)]
    pub periodic: bool,
    /// Leads on the faces of the sample, `-x`, `+x`, `-y`, `+y`, `-z`, `+z` (default `-x`, `+x`).
    pub leads: Option<Vec<String>>,
}

/// One calculation.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum RunInput {
    Transmission(TransmissionRun),
    Local(LocalRun),
    Bands(BandsRun),
    Spectrum(SpectrumRun),
    Dos(DosRun),
    Conductivity(ConductivityRun),
    Scba(ScbaRun),
    Evolution(EvolutionRun),
}

/// Transmissions between the leads, averaged over nothing: one row per configuration.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransmissionRun {
    pub output: Option<String>,
    pub energies: Values,
    #[serde(default)]
    pub scan: BTreeMap<String, Values>,
    pub configurations: Option<usize>,
    /// Labels of the orbitals of a lead site (spin sectors: `[0, 0, 1, 1]` for BHZ): the
    /// transmissions between sectors.
    pub sectors: Option<Vec<usize>>,
}

/// Scattering states: bond currents, densities and local densities of states on every site.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalRun {
    pub output: Option<String>,
    /// Energies of the window (the quantities are averaged over them).
    pub energies: Values,
    #[serde(default)]
    pub scan: BTreeMap<String, Values>,
    pub configurations: Option<usize>,
    /// Occupation of each lead (default: the first lead injects, the others are empty).
    pub occupations: Option<Vec<f64>>,
    /// On-site operators (name → matrix expression): their densities and currents.
    #[serde(default)]
    pub operators: BTreeMap<String, String>,
}

/// Bands of the ribbon (the sample's cross-section repeated along the first lead axis), or of
/// the bulk along a path.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BandsRun {
    pub output: Option<String>,
    #[serde(default)]
    pub scan: BTreeMap<String, Values>,
    /// Bloch phases `k·a` of the ribbon (default 201 values in [−π, π]).
    pub k: Option<Values>,
    /// Bulk bands: corners of a path in Bloch phases (one per lattice direction).
    pub path: Option<Vec<Vec<Num>>>,
    /// Points per segment of the path.
    pub points: Option<usize>,
}

/// Eigenvalues of the finite (or periodic) sample.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpectrumRun {
    pub output: Option<String>,
    #[serde(default)]
    pub scan: BTreeMap<String, Values>,
    pub configurations: Option<usize>,
}

/// Densities of states by the kernel polynomial method.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DosRun {
    pub output: Option<String>,
    pub energies: Values,
    #[serde(default)]
    pub scan: BTreeMap<String, Values>,
    pub configurations: Option<usize>,
    pub moments: Option<usize>,
    pub vectors: Option<usize>,
    pub seed: Option<u64>,
    pub bounds: Option<[f64; 2]>,
    /// On-site operators (name → matrix expression): projected densities of states.
    #[serde(default)]
    pub projections: BTreeMap<String, String>,
}

/// Kubo–Bastin conductivities of a periodic sample.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConductivityRun {
    pub output: Option<String>,
    /// Chemical potentials.
    pub energies: Values,
    #[serde(default)]
    pub scan: BTreeMap<String, Values>,
    pub configurations: Option<usize>,
    /// Components, `xx`, `xy`, ... (default `xx`, `xy`).
    pub components: Option<Vec<String>>,
    pub temperature: Option<Num>,
    pub moments: Option<usize>,
    pub vectors: Option<usize>,
    pub seed: Option<u64>,
    pub bounds: Option<[f64; 2]>,
}

/// The self-consistent Born approximation of the model with its disorder.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScbaRun {
    pub output: Option<String>,
    pub energies: Values,
    #[serde(default)]
    pub scan: BTreeMap<String, Values>,
    /// k points per lattice direction (default 64).
    pub k_grid: Option<Vec<usize>>,
    /// Imaginary part of the energy (default 1e-3).
    pub eta: Option<Num>,
}

/// Time evolution on a torus under a uniform force.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvolutionRun {
    pub output: Option<String>,
    #[serde(default)]
    pub scan: BTreeMap<String, Values>,
    pub configurations: Option<usize>,
    /// Force on the electrons (an electric field, the charge absorbed): `k → k + F t`.
    pub force: Option<Vec<Num>>,
    pub dt: Num,
    pub steps: usize,
    /// Observables every `every` steps (default 1).
    pub every: Option<usize>,
    /// Fermi energy of the excited-carrier count (default 0).
    pub fermi: Option<Num>,
    pub initial: InitialInput,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum InitialInput {
    /// The Fermi sea below `fermi`, as random-phase superpositions of the occupied states: the
    /// clean bands, or with disorder the ground state of the disordered Hamiltonian (a Chebyshev
    /// projection with `moments` terms, default 1024).
    FermiSea { fermi: Option<Num>, vectors: Option<usize>, seed: Option<u64>, moments: Option<usize> },
    /// The clean bands below `fermi` (only those on `orbitals` of the cell if given, e.g. one
    /// spin), as random-phase superpositions, also with disorder (which sets them in motion); the
    /// excited carriers are counted in the same bands.
    Band { fermi: Option<Num>, vectors: Option<usize>, seed: Option<u64>, orbitals: Option<Vec<usize>> },
    /// One orbital of one cell, projected on the band states in `window` if given.
    Localized { cell: Vec<usize>, orbital: Option<usize>, window: Option<[f64; 2]> },
}
