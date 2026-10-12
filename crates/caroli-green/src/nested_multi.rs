//! Multi-terminal transport by nested dissection: the leads and interfaces of
//! [`crate::MultiTerminal`], with the device eliminated as in [`crate::NestedTwoTerminal`].
//!
//! The sites that a lead couples to stay at the root of the elimination tree. The root then
//! holds the Schur complement `S` of `z − H − Σ` on their orbitals, and `S⁻¹` gives the Green's
//! function `G_pq` between every pair of leads, hence `T_pq = Tr[Γ_p G_pq Γ_q G_pq†]`. The cost
//! is that of the two-terminal elimination plus one dense inverse on the lead interfaces, where
//! [`crate::MultiTerminal`] inverts the whole device.
//!
//! Independent blocks of orbitals (e.g. the spin blocks of BHZ) are solved separately, each with
//! the part of every lead it reaches.

use crate::multi::{AttachedLead, Interface, attach_leads};
use crate::nested::{Components, SiteTree, Tree, schur};
use crate::{GreenError, SelfEnergyMethod, broadening, trace};
use caroli_dense::Lu;
use caroli_lattice::Region;
use caroli_model::{CMat, Model, SparseMatrix, c64};

/// One independent block of orbitals: its elimination tree and the part of each lead it reaches.
#[derive(Clone, Debug)]
struct MultiBlock {
    tree: Tree,
    leads: Vec<Option<AttachedLead>>,
}

/// A device with any number of attached leads, solved by nested dissection (see the module
/// documentation).
#[derive(Clone, Debug)]
pub struct NestedMultiTerminal {
    h: SparseMatrix,
    lead_count: usize,
    blocks: Vec<MultiBlock>,
}

impl NestedMultiTerminal {
    /// Device from `device` on `region`, with leads of the clean `lead` model continuing each
    /// interface outward, as [`crate::MultiTerminal::from_model`].
    pub fn from_model(device: &Model, lead: &Model, region: &Region, interfaces: &[Interface]) -> Self {
        let h = device.sparse_hamiltonian(region);
        let leads = attach_leads(lead, region, interfaces);
        // Sites with an orbital coupled to a lead stay at the root.
        let mut coupled = vec![false; h.n];
        for l in &leads {
            for &o in &l.orbitals {
                coupled[o] = true;
            }
        }
        let interior: Vec<usize> = (0..region.len())
            .filter(|&i| {
                let o = region.offset(i);
                !(o..o + region.site_orbitals(i)).any(|k| coupled[k])
            })
            .collect();
        let sites = SiteTree::new(device, region, interior);

        // Independent blocks over the device orbitals, then the orbitals of each lead cell.
        let mut offsets = Vec::with_capacity(leads.len());
        let mut total = h.n;
        for l in &leads {
            offsets.push(total);
            total += l.h00.nrows();
        }
        let mut components = Components::new(total);
        for r in 0..h.n {
            for (c, v) in h.row(r) {
                if v != c64::new(0.0, 0.0) {
                    components.join(r, c);
                }
            }
        }
        for (l, &off) in leads.iter().zip(&offsets) {
            let cell: Vec<usize> = (off..off + l.h00.nrows()).collect();
            components.join_matrix(&l.h00, &cell, &cell);
            components.join_matrix(&l.h01, &cell, &cell);
            components.join_matrix(&l.coupling, &l.orbitals, &cell);
        }
        let (roots, labels) = components.labels();
        let blocks = labels
            .iter()
            .filter_map(|&label| {
                let mine = |o: usize| roots[o] == label;
                let block_leads: Vec<Option<AttachedLead>> = leads
                    .iter()
                    .zip(&offsets)
                    .map(|(l, &off)| {
                        let kc: Vec<usize> = (0..l.h00.nrows()).filter(|&k| mine(off + k)).collect();
                        let kd: Vec<usize> = (0..l.orbitals.len()).filter(|&k| mine(l.orbitals[k])).collect();
                        if kc.is_empty() || kd.is_empty() {
                            return None;
                        }
                        let cell = |m: &CMat| CMat::from_fn(kc.len(), kc.len(), |a, b| m[(kc[a], kc[b])]);
                        Some(AttachedLead {
                            h00: cell(&l.h00),
                            h01: cell(&l.h01),
                            orbitals: kd.iter().map(|&k| l.orbitals[k]).collect(),
                            coupling: CMat::from_fn(kd.len(), kc.len(), |a, b| l.coupling[(kd[a], kc[b])]),
                        })
                    })
                    .collect();
                if block_leads.iter().all(Option::is_none) {
                    return None;
                }
                Some(MultiBlock { tree: sites.tree(region, &h, mine), leads: block_leads })
            })
            .collect();
        Self { h, lead_count: leads.len(), blocks }
    }

    /// Number of leads.
    pub fn lead_count(&self) -> usize {
        self.lead_count
    }

    /// Number of independent blocks solved separately.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Transmissions `T[p][q] = Tr[Γ_p G_pq Γ_q G_pq†]` from lead `q` into lead `p` at `energy`,
    /// with mode-matching self-energies, as [`crate::MultiSolution::transmission`].
    pub fn transmissions(&self, energy: f64) -> Result<Vec<Vec<f64>>, GreenError> {
        use rayon::prelude::*;
        let parts: Vec<Result<Vec<Vec<f64>>, GreenError>> =
            self.blocks.par_iter().map(|b| self.block_transmissions(b, energy)).collect();
        let n = self.lead_count;
        let mut t = vec![vec![0.0; n]; n];
        for part in parts {
            for (row, add) in t.iter_mut().zip(part?) {
                row.iter_mut().zip(add).for_each(|(x, y)| *x += y);
            }
        }
        Ok(t)
    }

    fn block_transmissions(&self, block: &MultiBlock, energy: f64) -> Result<Vec<Vec<f64>>, GreenError> {
        let sigmas: Vec<Option<CMat>> = block
            .leads
            .iter()
            .map(|l| l.as_ref().map(|l| l.self_energy(energy, SelfEnergyMethod::Modes)).transpose())
            .collect::<Result<_, _>>()?;
        // The root keeps every orbital a lead couples to.
        let mut kept: Vec<usize> =
            block.leads.iter().flatten().flat_map(|l| l.orbitals.iter().copied()).collect();
        kept.sort_unstable();
        kept.dedup();
        let index = |l: &AttachedLead| -> Vec<usize> {
            l.orbitals.iter().map(|o| kept.binary_search(o).expect("lead orbital kept")).collect()
        };
        let (nd, mut f) = block.tree.root(&self.h, c64::new(energy, 0.0), &kept)?;
        for (l, s) in block.leads.iter().zip(&sigmas) {
            if let (Some(l), Some(s)) = (l, s) {
                let at = index(l);
                for (a, &i) in at.iter().enumerate() {
                    for (b, &j) in at.iter().enumerate() {
                        f[(nd + i, nd + j)] -= s[(a, b)];
                    }
                }
            }
        }
        let s = if nd > 0 { schur(&f, nd).ok_or(GreenError::Singular(energy))? } else { f };
        let k = kept.len();
        let g = Lu::new(s.as_ref()).solve(CMat::identity(k, k).as_ref());
        if !g.norm_max().is_finite() {
            return Err(GreenError::Singular(energy));
        }
        let gammas: Vec<Option<CMat>> = sigmas.iter().map(|s| s.as_ref().map(broadening)).collect();
        let n = self.lead_count;
        let mut t = vec![vec![0.0; n]; n];
        for (p, row) in t.iter_mut().enumerate() {
            for (q, tpq) in row.iter_mut().enumerate() {
                let (Some(lp), Some(lq), Some(gp), Some(gq)) =
                    (&block.leads[p], &block.leads[q], &gammas[p], &gammas[q])
                else {
                    continue;
                };
                let (ip, iq) = (index(lp), index(lq));
                let gpq = CMat::from_fn(ip.len(), iq.len(), |a, b| g[(ip[a], iq[b])]);
                *tpq = trace(&(gp * &gpq * gq * gpq.adjoint())).re;
            }
        }
        Ok(t)
    }
}
