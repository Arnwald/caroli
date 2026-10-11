//! Ready-made models.

use crate::gamma::*;
use crate::{CMat, Model, c64, combine};
use caroli_lattice::Lattice;

/// Parameters of the BHZ model of a HgTe/CdTe quantum well, discretised on a square lattice.
///
/// Energies are in units of `A/a`; the defaults are the normalised HgTe values of the thesis
/// (`A/a` = 72.9 meV with A = 364.5 meV·nm, a = 5 nm): inverted regime, `mass < 0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BhzParams {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub mass: f64,
    /// Inter-spin-block spin-orbit coupling.
    pub lambda: f64,
}

impl Default for BhzParams {
    fn default() -> Self {
        Self { a: 1.0, b: -0.376406, c: 0.0, d: -0.280933, mass: -0.137, lambda: 0.0 }
    }
}

/// BHZ model on a square lattice (constant 1), basis `(E1↑, H1↑, E1↓, H1↓)`, as in the thesis
/// codes (`ti.f90`, 2014 and later):
///
/// - on-site: `(C − 4D) Γ0 + (M − 4B) Γ55`
/// - hopping `+x`: `D Γ0 + B Γ55 − (i/2)(A Γ33 − λ Γ13)`
/// - hopping `+y`: `D Γ0 + B Γ55 + (i/2)(A Γ44 − λ Γ14)`
pub fn bhz(p: BhzParams) -> Model {
    let r = |x: f64| c64::new(x, 0.0);
    let i = |x: f64| c64::new(0.0, x);
    let onsite = combine(&[(r(p.c - 4.0 * p.d), &gamma0()), (r(p.mass - 4.0 * p.b), &gamma55())]);
    let tx = combine(&[
        (r(p.d), &gamma0()),
        (r(p.b), &gamma55()),
        (i(-0.5 * p.a), &gamma33()),
        (i(0.5 * p.lambda), &gamma13()),
    ]);
    let ty = combine(&[
        (r(p.d), &gamma0()),
        (r(p.b), &gamma55()),
        (i(0.5 * p.a), &gamma44()),
        (i(-0.5 * p.lambda), &gamma14()),
    ]);
    Model::new(Lattice::square(1.0, 4)).onsite(0, onsite).hopping(0, 0, [1, 0, 0], tx).hopping(
        0,
        0,
        [0, 1, 0],
        ty,
    )
}

/// Exchange coupling of a classical moment to the BHZ electrons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GFactors {
    /// `g_E = g_H`: `n·s` (the thesis "ferro", `ge=gh`).
    Same,
    /// `g_E = −g_H`: `n·s τ_z` (the thesis "antiferro", `ge=-gh`).
    Opposite,
}

/// Exchange matrix `J (n_x S_x + n_y S_y + n_z S_z)` in the BHZ basis for a moment `n`.
pub fn bhz_exchange(coupling: GFactors, j: f64, n: [f64; 3]) -> CMat {
    let (sx, sy, sz) = match coupling {
        GFactors::Same => (gamma23(), gamma31(), gamma12()),
        GFactors::Opposite => (gamma23p(), gamma31p(), gamma12p()),
    };
    combine(&[(c64::new(j * n[0], 0.0), &sx), (c64::new(j * n[1], 0.0), &sy), (c64::new(j * n[2], 0.0), &sz)])
}
