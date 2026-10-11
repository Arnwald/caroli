//! Surface Green's function by decimation (Lopez Sancho, Lopez Sancho & Rubio, J. Phys. F 15,
//! 851 (1985)).
//!
//! Needs a finite imaginary part of the energy and loses accuracy when the energy is close to an
//! eigenvalue of the cell Hamiltonian (with `η = 1e-12`, `T` of a clean chain is wrong by 0.5 at
//! `E = 1e-9`). Kept as an independent cross-check of [`crate::modes`].

use crate::{GreenError, inverse, resolvent_matrix};
use caroli_model::{CMat, c64};

/// Surface Green's function of a semi-infinite lead at complex energy `z`. `h00` is the cell
/// Hamiltonian and `away` the coupling `⟨n| H |n+1⟩` from a cell to the next one further away
/// from the surface.
pub fn surface_green(h00: &CMat, away: &CMat, z: c64) -> Result<CMat, GreenError> {
    const MAX_STEPS: usize = 200;
    const TOL: f64 = 1e-14;
    let mut alpha = away.clone();
    let mut beta = away.adjoint().to_owned();
    let mut eps_s = h00.clone();
    let mut eps = h00.clone();
    for _ in 0..MAX_STEPS {
        let g = inverse(&resolvent_matrix(z, &eps));
        let ag = &alpha * &g;
        let bg = &beta * &g;
        let agb = &ag * &beta;
        eps_s += &agb;
        eps += &agb + &bg * &alpha;
        alpha = &ag * &alpha;
        beta = &bg * &beta;
        let scale = 1.0 + eps.norm_max();
        if alpha.norm_max() < TOL * scale && beta.norm_max() < TOL * scale {
            return Ok(inverse(&resolvent_matrix(z, &eps_s)));
        }
    }
    Err(GreenError::NoConvergence(MAX_STEPS))
}
