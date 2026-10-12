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
mod gauge;
pub mod modes;
pub mod multi;
pub mod nested;
pub mod nested_multi;
mod smatrix;
mod two_terminal;

pub use multi::{AttachedLead, Interface, MultiSolution, MultiTerminal, lead_voltages};
pub use nested::{BondCurrent, LeadSelfEnergies, NestedTwoTerminal, ScatteringStates};
pub use nested_multi::{NestedMultiTerminal, SectorTransmissions};
pub use smatrix::SMatrix;
pub use two_terminal::{EndToEnd, Lead, SelfEnergyMethod, Sweeps, TwoTerminal};

use caroli_dense::{Lu, Op, product};
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

// Products of dense matrices go through caroli-dense (Accelerate on macOS, on the matrix units):
// faer's `*` runs on the vector units only, several times slower at the sizes of lead slices.

/// `a b`.
pub(crate) fn mul(a: &CMat, b: &CMat) -> CMat {
    product(a.as_ref(), Op::Plain, b.as_ref(), Op::Plain)
}

/// `a† b`.
pub(crate) fn adj_mul(a: &CMat, b: &CMat) -> CMat {
    product(a.as_ref(), Op::Adjoint, b.as_ref(), Op::Plain)
}

/// `a b†`.
pub(crate) fn mul_adj(a: &CMat, b: &CMat) -> CMat {
    product(a.as_ref(), Op::Plain, b.as_ref(), Op::Adjoint)
}

/// `v m v†`.
pub(crate) fn sandwich(v: &CMat, m: &CMat) -> CMat {
    mul_adj(&mul(v, m), v)
}

/// `v† m v`.
pub(crate) fn sandwich_adj(v: &CMat, m: &CMat) -> CMat {
    mul(&adj_mul(v, m), v)
}

/// Caroli trace `Tr[Γ_p G Γ_q G†]` for the block `G = G_pq`: with `A = Γ_p G` and `B = G Γ_q`,
/// it is `Σ_ij A_ij conj(B_ij)` (two products instead of three).
pub(crate) fn caroli_trace(gamma_p: &CMat, g: &CMat, gamma_q: &CMat) -> f64 {
    let (a, b) = (mul(gamma_p, g), mul(g, gamma_q));
    (0..a.ncols())
        .map(|j| a.col_as_slice(j).iter().zip(b.col_as_slice(j)).map(|(x, y)| (x * y.conj()).re).sum::<f64>())
        .sum()
}
