//! Scattering matrix in the basis of the propagating lead modes, as `kwant.smatrix`.
//!
//! A lead `p` has cells `n = 1, 2, …` going away from the device, the hopping
//! `H1 = ⟨n| H |n+1⟩`, and the coupling `V = ⟨device| H |1⟩`. With `F` the Bloch matrix of its
//! outgoing and decaying modes, `H1 F` is the self-energy of the cells beyond the first and
//! `g = (E − H0 − H1 F)⁻¹` the surface Green's function of the first cell.
//!
//! - An incoming mode `φ λ^n` of lead `q` acts on the device as the source `V_q s` with
//!   `s = (λ − g H1†) φ`, so the device wave is `ψ_D = G V_q s`.
//! - The outgoing wave on the first cell of lead `p` is `ψ_1 = g V_p† ψ_D`, minus `g H1† φ` in
//!   lead `q`; on the second cell, `H1 ψ_2 = H1 F ψ_1`.
//! - Its amplitude on an outgoing mode `a` is the current overlap
//!   `J(a, ψ) = i(a_1† H1 ψ_2 − a_2† H1† ψ_1)`: evanescent modes and the other propagating modes
//!   carry no cross current, and the modes carry unit current.

use crate::modes::{LeadModes, sigma_right};
use crate::{GreenError, inverse};
use caroli_model::{CMat, c64};
use std::rc::Rc;

/// Scattering matrix at one energy in the basis of the propagating lead modes.
#[derive(Clone, Debug)]
pub struct SMatrix {
    pub energy: f64,
    /// `blocks[p][q]`: amplitudes into the outgoing modes of lead `p` (rows) from the incoming
    /// modes of lead `q` (columns), all modes normalised to unit current.
    pub blocks: Vec<Vec<CMat>>,
    /// Bloch factors `λ` of the incoming and outgoing modes of each lead, along the lead away
    /// from the device (`ψ_n = φ λ^n`, `n` the cell).
    pub incoming: Vec<Vec<c64>>,
    pub outgoing: Vec<Vec<c64>>,
}

impl SMatrix {
    /// `Σ |S_pq|²`: the transmission from lead `q` into lead `p` (a reflection for `p = q`).
    pub fn transmission(&self, p: usize, q: usize) -> f64 {
        let b = &self.blocks[p][q];
        (0..b.ncols())
            .flat_map(|c| (0..b.nrows()).map(move |r| (r, c)))
            .map(|(r, c)| b[(r, c)].norm_sqr())
            .sum()
    }

    /// The whole matrix: rows are the outgoing modes of every lead in turn, columns the incoming
    /// ones.
    pub fn full(&self) -> CMat {
        let rows: Vec<usize> = self.outgoing.iter().map(Vec::len).collect();
        let cols: Vec<usize> = self.incoming.iter().map(Vec::len).collect();
        let mut out = CMat::zeros(rows.iter().sum(), cols.iter().sum());
        let mut r0 = 0;
        for (p, &nr) in rows.iter().enumerate() {
            let mut c0 = 0;
            for (q, &nc) in cols.iter().enumerate() {
                out.as_mut().submatrix_mut(r0, c0, nr, nc).copy_from(&self.blocks[p][q]);
                c0 += nc;
            }
            r0 += nr;
        }
        out
    }

    /// The scattering matrix of a device made of independent blocks: the modes of a lead are
    /// those of every block in turn, and the blocks do not mix.
    pub(crate) fn assemble(energy: f64, parts: Vec<BlockScattering>) -> Self {
        let n = parts.first().map_or(0, |p| p.incoming.len());
        let (mut incoming, mut outgoing) = (vec![Vec::new(); n], vec![Vec::new(); n]);
        for part in &parts {
            for p in 0..n {
                incoming[p].extend_from_slice(&part.incoming[p]);
                outgoing[p].extend_from_slice(&part.outgoing[p]);
            }
        }
        let blocks = (0..n)
            .map(|p| {
                (0..n)
                    .map(|q| {
                        let mut out = CMat::zeros(outgoing[p].len(), incoming[q].len());
                        let (mut r0, mut c0) = (0, 0);
                        for part in &parts {
                            let b = &part.blocks[p][q];
                            out.as_mut().submatrix_mut(r0, c0, b.nrows(), b.ncols()).copy_from(b);
                            r0 += part.outgoing[p].len();
                            c0 += part.incoming[q].len();
                        }
                        out
                    })
                    .collect()
            })
            .collect();
        Self { energy, blocks, incoming, outgoing }
    }
}

/// A lead solved at one energy: its modes, `H1 F` (the self-energy of the cells beyond the
/// first) and the surface Green's function `g` of its first cell.
pub(crate) struct LeadSolution {
    pub(crate) modes: LeadModes,
    pub(crate) rest: CMat,
    pub(crate) g: CMat,
}

impl LeadSolution {
    /// The lead `(h00, h01)`, `h01` going away from the device.
    pub(crate) fn new(h00: &CMat, h01: &CMat, z: c64) -> Result<Self, GreenError> {
        let modes = LeadModes::at(h00, h01, z)?;
        let rest = sigma_right(h01, &modes);
        let k = h00.nrows();
        let g = inverse(&CMat::from_fn(k, k, |i, j| {
            (if i == j { z } else { c64::new(0.0, 0.0) }) - h00[(i, j)] - rest[(i, j)]
        }));
        Ok(Self { modes, rest, g })
    }

    /// `Σ = V g V†` on the device orbitals the lead couples to.
    pub(crate) fn self_energy(&self, coupling: &CMat) -> CMat {
        coupling * &self.g * coupling.adjoint()
    }
}

/// A lead of a block for [`scattering_blocks`]: where its device orbitals sit in the root, its
/// coupling `V` and outward hopping `H1`, and its solution.
pub(crate) struct ScatteringLead {
    pub(crate) at: Vec<usize>,
    pub(crate) coupling: CMat,
    pub(crate) h01: CMat,
    pub(crate) solution: Rc<LeadSolution>,
}

/// The scattering matrix of one block: `(incoming λ, outgoing λ, S_pq)` per lead.
pub(crate) struct BlockScattering {
    pub(crate) incoming: Vec<Vec<c64>>,
    pub(crate) outgoing: Vec<Vec<c64>>,
    pub(crate) blocks: Vec<Vec<CMat>>,
}

/// `S_pq` for every pair of leads of a block, from the Green's function `green` of the root
/// (the device orbitals the leads couple to) with the self-energies of these leads included.
pub(crate) fn scattering_blocks(green: &CMat, leads: &[Option<ScatteringLead>]) -> BlockScattering {
    let n = leads.len();
    let i = c64::new(0.0, 1.0);
    let lambdas = |f: fn(&LeadModes) -> &Vec<c64>| -> Vec<Vec<c64>> {
        leads.iter().map(|l| l.as_ref().map_or(Vec::new(), |l| f(&l.solution.modes).clone())).collect()
    };
    let incoming = lambdas(|m| &m.left_lambdas);
    let outgoing = lambdas(|m| &m.right_lambdas);
    let mut blocks = vec![vec![CMat::zeros(0, 0); n]; n];
    for q in 0..n {
        for p in 0..n {
            blocks[p][q] = CMat::zeros(outgoing[p].len(), incoming[q].len());
        }
        let Some(lq) = &leads[q] else { continue };
        let modes_q = &lq.solution.modes;
        let (phi_in, lam_in) = (&modes_q.left_modes, &modes_q.left_lambdas);
        // g H1† Φ_in, and the source s = Φ_in λ − g H1† Φ_in on the first cell.
        let back = &lq.solution.g * (lq.h01.adjoint() * phi_in);
        let source =
            CMat::from_fn(phi_in.nrows(), phi_in.ncols(), |a, c| phi_in[(a, c)] * lam_in[c] - back[(a, c)]);
        let device_source = &lq.coupling * &source;
        for p in 0..n {
            let Some(lp) = &leads[p] else { continue };
            let gpq = CMat::from_fn(lp.at.len(), lq.at.len(), |a, b| green[(lp.at[a], lq.at[b])]);
            let psi_device = &gpq * &device_source;
            let mut psi1 = &lp.solution.g * (lp.coupling.adjoint() * &psi_device);
            if p == q {
                psi1 -= &back;
            }
            // S = i diag(λ̄) [Φ_out† (H1 F) ψ_1 − diag(λ̄) Φ_out† H1† ψ_1].
            let modes_p = &lp.solution.modes;
            let (phi_out, lam_out) = (&modes_p.right_modes, &modes_p.right_lambdas);
            let a = phi_out.adjoint() * (&lp.solution.rest * &psi1);
            let b = phi_out.adjoint() * (lp.h01.adjoint() * &psi1);
            blocks[p][q] = CMat::from_fn(phi_out.ncols(), psi1.ncols(), |m, c| {
                let conj = lam_out[m].conj();
                i * conj * (a[(m, c)] - conj * b[(m, c)])
            });
        }
    }
    BlockScattering { incoming, outgoing, blocks }
}
