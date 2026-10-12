//! Steady-state transport with Green's functions.
//!
//! - [`modes`]: lead modes and mode-matching self-energies (default, exact, no `η` needed).
//! - [`decimation`]: Lopez Sancho – Rubio surface Green's function (needs `η`; cross-check).
//! - [`TwoTerminal`]: a device cut into slices between two leads; the recursive Green's
//!   function gives the Caroli transmission `T = Tr[Γ_L G_1N Γ_R G_1N†]`.
//! - [`MultiTerminal`]: any geometry with any number of leads, solved directly.
//! - [`NestedTwoTerminal`]: the same two-terminal problem by geometric nested dissection,
//!   O(n³) for an n × n sample instead of O(n⁴).
//! - [`NestedMultiTerminal`]: any number of leads attached anywhere, by nested dissection.

pub mod decimation;
pub mod modes;
pub mod multi;
pub mod nested;
pub mod nested_multi;
mod two_terminal;

pub use multi::{AttachedLead, Interface, MultiSolution, MultiTerminal};
pub use nested::{LeadSelfEnergies, NestedTwoTerminal};
pub use nested_multi::NestedMultiTerminal;
pub use two_terminal::{EndToEnd, Lead, SelfEnergyMethod, Sweeps, TwoTerminal};

use caroli_dense::Lu;
use caroli_model::{CMat, c64};

/// Error of a Green's function computation.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum GreenError {
    #[error("surface Green's function did not converge in {0} decimation steps")]
    NoConvergence(usize),
    #[error("generalized eigenvalue problem of the lead failed")]
    Eigen,
    #[error(
        "lead modes: {right} right-going and {left} left-going, expected {expected} each (energy at a band edge?)"
    )]
    ModeCount { right: usize, left: usize, expected: usize },
    #[error("lead self-energy undefined near E = {0} (subband edge or bound state of the lead surface)")]
    Singular(f64),
}

/// Panics unless the `lead` model is the same on the lead cell `cell` and on the next cells
/// along `step`: a vector potential or a potential that changes along the lead would make its
/// modes wrong.
pub(crate) fn assert_lead_invariant(
    lead: &caroli_model::Model,
    cell: &caroli_lattice::Region,
    step: caroli_lattice::Cell,
) {
    let next = cell.translated(step);
    let after = next.translated(step);
    let (h0, h0_next) = (lead.hamiltonian(cell), lead.hamiltonian(&next));
    let (h1, h1_next) = (lead.coupling(cell, &next), lead.coupling(&next, &after));
    let tol = 1e-12 * h0.norm_max().max(h1.norm_max());
    assert!(
        (&h0 - &h0_next).norm_max() <= tol && (&h1 - &h1_next).norm_max() <= tol,
        "the lead model changes along the lead: use a gauge invariant along it (gauge::landau_x for \
         leads along x) and no potential that varies along it"
    );
}

/// Inverse of a square matrix (partial-pivoting LU).
pub(crate) fn inverse(m: &CMat) -> CMat {
    let n = m.nrows();
    Lu::new(m.as_ref()).solve(CMat::identity(n, n).as_ref())
}

/// `x` solving `x · a = b`.
pub(crate) fn solve_right(b: &CMat, a: &CMat) -> CMat {
    let at = a.transpose().to_owned();
    Lu::new(at.as_ref()).solve(b.transpose()).transpose().to_owned()
}

/// `z·1 − h`.
pub(crate) fn resolvent_matrix(z: c64, h: &CMat) -> CMat {
    let n = h.nrows();
    CMat::from_fn(n, n, |i, j| if i == j { z - h[(i, j)] } else { -h[(i, j)] })
}

/// Broadening `Γ = i (Σ − Σ†)`.
pub fn broadening(sigma: &CMat) -> CMat {
    let n = sigma.nrows();
    CMat::from_fn(n, n, |i, j| c64::new(0.0, 1.0) * (sigma[(i, j)] - sigma[(j, i)].conj()))
}

/// Trace of a square matrix.
pub fn trace(m: &CMat) -> c64 {
    (0..m.nrows()).map(|i| m[(i, i)]).sum()
}
