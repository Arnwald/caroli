//! Steady-state transport with Green's functions.
//!
//! - [`modes`]: lead modes and mode-matching self-energies (default, exact, no `η` needed).
//! - [`decimation`]: Lopez Sancho – Rubio surface Green's function (needs `η`; cross-check).
//! - [`TwoTerminal`]: a device cut into slices between two leads; the recursive Green's
//!   function gives the Caroli transmission `T = Tr[Γ_L G_1N Γ_R G_1N†]`.
//! - [`MultiTerminal`]: any geometry with any number of leads, solved directly.

pub mod decimation;
pub mod modes;
pub mod multi;
mod two_terminal;

pub use multi::{AttachedLead, Interface, MultiSolution, MultiTerminal};
pub use two_terminal::{EndToEnd, Lead, SelfEnergyMethod, Sweeps, TwoTerminal};

use caroli_model::{CMat, c64};
use faer::linalg::solvers::{DenseSolveCore, Solve};

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
}

/// Inverse of a square matrix (partial-pivoting LU).
pub(crate) fn inverse(m: &CMat) -> CMat {
    m.partial_piv_lu().inverse()
}

/// `x` solving `x · a = b`.
pub(crate) fn solve_right(b: &CMat, a: &CMat) -> CMat {
    let xt = a.transpose().partial_piv_lu().solve(b.transpose());
    xt.transpose().to_owned()
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
