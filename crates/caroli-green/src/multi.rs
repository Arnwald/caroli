//! Multi-terminal devices of any geometry: leads attached to interface slices, Green's function
//! of the whole device by a direct solve, transmissions `T_pq = Tr[Γ_p G_pq Γ_q G_pq†]`.
//!
//! The direct solve costs `O(n³)` in the number of device orbitals `n`: it suits devices up to a
//! few thousand orbitals (Hall bars, crosses, small flakes). Long quasi-one-dimensional devices
//! are better served by [`crate::TwoTerminal`].

use crate::gauge::Gauge;
use crate::modes::{LeadModes, sigma_right};
use crate::{GreenError, SelfEnergyMethod, broadening, decimation, inverse, trace};
use caroli_lattice::{Cell, Region};
use caroli_model::{CMat, Model, c64};

/// A semi-infinite lead attached to a device.
#[derive(Clone, Debug)]
pub struct AttachedLead {
    /// Hamiltonian of one lead cell.
    pub h00: CMat,
    /// Coupling `⟨cell n| H |cell n+1⟩` from a cell to the next one further from the device.
    pub h01: CMat,
    /// Device orbitals coupled to the first lead cell.
    pub orbitals: Vec<usize>,
    /// Coupling from these orbitals to the first lead cell (`orbitals.len() × cell orbitals`).
    pub coupling: CMat,
}

impl AttachedLead {
    /// Surface Green's function of the lead (its first cell and everything beyond).
    pub fn surface_green(&self, energy: f64, method: SelfEnergyMethod) -> Result<CMat, GreenError> {
        let z = match method {
            SelfEnergyMethod::Modes => c64::new(energy, 0.0),
            SelfEnergyMethod::ModesBroadened { eta } | SelfEnergyMethod::Decimation { eta } => {
                c64::new(energy, eta)
            }
        };
        if let SelfEnergyMethod::Decimation { .. } = method {
            return decimation::surface_green(&self.h00, &self.h01, z);
        }
        let rest = sigma_right(&self.h01, &LeadModes::at(&self.h00, &self.h01, z)?);
        let n = self.h00.nrows();
        Ok(inverse(&CMat::from_fn(n, n, |i, j| {
            (if i == j { z } else { c64::new(0.0, 0.0) }) - self.h00[(i, j)] - rest[(i, j)]
        })))
    }

    /// Self-energy on the coupled device orbitals, `Σ = V g V†`.
    pub fn self_energy(&self, energy: f64, method: SelfEnergyMethod) -> Result<CMat, GreenError> {
        let g = self.surface_green(energy, method)?;
        Ok(&self.coupling * g * self.coupling.adjoint())
    }
}

/// A device Hamiltonian with attached leads.
#[derive(Clone, Debug)]
pub struct MultiTerminal {
    pub h: CMat,
    pub leads: Vec<AttachedLead>,
    /// Sites of the device (orbital layout of `h`), when built from a model.
    pub region: Option<Region>,
}

/// One interface of a lead for [`MultiTerminal::from_model`]: the device sites of the interface
/// slice and the lattice step pointing away from the device.
#[derive(Clone, Debug)]
pub struct Interface {
    pub slice: Region,
    pub outward: Cell,
}

impl MultiTerminal {
    /// Device from `device` on `region`; each lead continues an interface slice outward with the
    /// clean `lead` model. The lead couples to every device orbital its first cell reaches (not
    /// only the interface slice: slanted bonds, several sublattices).
    pub fn from_model(device: &Model, lead: &Model, region: &Region, interfaces: &[Interface]) -> Self {
        crate::gauge::assert_same_field(device, lead);
        Self {
            h: device.hamiltonian(region),
            leads: attach_leads(lead, region, interfaces),
            region: Some(region.clone()),
        }
    }

    /// Self-energies of the leads and the retarded Green's function of the whole device.
    pub fn solve(&self, energy: f64, method: SelfEnergyMethod) -> Result<MultiSolution, GreenError> {
        let n = self.h.nrows();
        let eta = match method {
            SelfEnergyMethod::Decimation { eta } => eta,
            _ => 0.0,
        };
        let mut a = CMat::from_fn(n, n, |i, j| {
            (if i == j { c64::new(energy, eta) } else { c64::new(0.0, 0.0) }) - self.h[(i, j)]
        });
        let mut sigmas = Vec::with_capacity(self.leads.len());
        for lead in &self.leads {
            let s = lead.self_energy(energy, method)?;
            for (p, &i) in lead.orbitals.iter().enumerate() {
                for (q, &j) in lead.orbitals.iter().enumerate() {
                    a[(i, j)] -= s[(p, q)];
                }
            }
            sigmas.push(s);
        }
        Ok(MultiSolution { g: inverse(&a), sigmas })
    }
}

/// Leads of `lead` attached to `region` at each interface (see [`MultiTerminal::from_model`]).
pub(crate) fn attach_leads(lead: &Model, region: &Region, interfaces: &[Interface]) -> Vec<AttachedLead> {
    interfaces
        .iter()
        .map(|itf| {
            let shift = |k: i64| -> Cell { [k * itf.outward[0], k * itf.outward[1], k * itf.outward[2]] };
            let first = itf.slice.translated(shift(1));
            assert!(
                first.sites().iter().all(|s| region.index_of(s).is_none()),
                "the first lead cell overlaps the device"
            );
            let skip = lead.coupling(&itf.slice, &itf.slice.translated(shift(2)));
            assert!(skip.norm_max() == 0.0, "a lead hopping reaches beyond the next cell");
            // In a uniform field, the lead in the gauge invariant along it; the device keeps its
            // gauge, and the coupling is mixed.
            let gauge = Gauge::along(lead, itf.outward);
            let (h00, h01) = gauge.lead(&itf.slice, itf.outward, itf.outward);
            let v = gauge.transform_columns(lead.coupling(region, &first), &first);
            let orbitals: Vec<usize> =
                (0..v.nrows()).filter(|&i| (0..v.ncols()).any(|j| v[(i, j)] != c64::new(0.0, 0.0))).collect();
            AttachedLead {
                h00,
                h01,
                coupling: CMat::from_fn(orbitals.len(), v.ncols(), |a, j| v[(orbitals[a], j)]),
                orbitals,
            }
        })
        .collect()
}

/// Result of [`MultiTerminal::solve`].
#[derive(Clone, Debug)]
pub struct MultiSolution {
    /// Retarded Green's function of the device.
    pub g: CMat,
    /// Self-energy of each lead on its interface orbitals.
    pub sigmas: Vec<CMat>,
}

impl MultiSolution {
    /// Block `G_pq` between the interfaces of leads `p` and `q`.
    pub fn block(&self, sys: &MultiTerminal, p: usize, q: usize) -> CMat {
        let (rp, rq) = (&sys.leads[p].orbitals, &sys.leads[q].orbitals);
        CMat::from_fn(rp.len(), rq.len(), |i, j| self.g[(rp[i], rq[j])])
    }

    /// Transmission `T_pq = Tr[Γ_p G_pq Γ_q G_pq†]` from lead `q` to lead `p`.
    pub fn transmission(&self, sys: &MultiTerminal, p: usize, q: usize) -> f64 {
        let g = self.block(sys, p, q);
        trace(&(broadening(&self.sigmas[p]) * &g * broadening(&self.sigmas[q]) * g.adjoint())).re
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TwoTerminal;
    use caroli_lattice::Lattice;

    /// A narrow lead (two rows) on a graphene device: device sites outside the interface slice
    /// bond to the first lead cell. Moving one lead cell into the device must not change T.
    #[test]
    fn narrow_lead_on_graphene() {
        let lat = Lattice::honeycomb(1.0, 1);
        let one = |x: f64| CMat::from_fn(1, 1, |_, _| c64::new(x, 0.0));
        let g = Model::new(lat.clone())
            .hopping(0, 1, [0, 0, 0], one(-1.0))
            .hopping(0, 1, [0, -1, 0], one(-1.0))
            .hopping(0, 1, [1, -1, 0], one(-1.0));
        let site = |x: i64, y: i64, s: usize| caroli_lattice::Site { cell: [x, y, 0], sub: s };
        let column = |x: i64, ys: std::ops::Range<i64>| {
            Region::from_sites(&lat, ys.flat_map(|y| (0..2).map(move |s| site(x, y, s))).collect())
        };
        let device = |x0: i64| {
            let mut sites: Vec<caroli_lattice::Site> =
                (0..6).flat_map(|x| (0..6).flat_map(move |y| (0..2).map(move |s| site(x, y, s)))).collect();
            for x in x0..0 {
                sites.extend((2..4).flat_map(|y| (0..2).map(move |s| site(x, y, s))));
            }
            Region::from_sites(&lat, sites)
        };
        let t = |x0: i64, e: f64| {
            let region = device(x0);
            let sys = MultiTerminal::from_model(
                &g,
                &g,
                &region,
                &[
                    Interface { slice: column(x0, 2..4), outward: [-1, 0, 0] },
                    Interface { slice: column(5, 0..6), outward: [1, 0, 0] },
                ],
            );
            sys.solve(e, SelfEnergyMethod::Modes).unwrap().transmission(&sys, 1, 0)
        };
        for e in [-0.7, 0.45, 1.3] {
            let (a, b) = (t(0, e), t(-1, e));
            assert!((a - b).abs() < 1e-9, "E {e}: {a} vs {b}");
        }
    }

    /// A ribbon with two attached leads gives the same transmission as the two-terminal solver,
    /// in both directions.
    #[test]
    fn agrees_with_two_terminal_and_unitarity() {
        // Square lattice, spinless, hopping −1, on-site disorder on a few sites.
        let lat = Lattice::square(1.0, 1);
        let one = |x: f64| CMat::from_fn(1, 1, |_, _| c64::new(x, 0.0));
        let clean =
            Model::new(lat.clone()).hopping(0, 0, [1, 0, 0], one(-1.0)).hopping(0, 0, [0, 1, 0], one(-1.0));
        let pot = move |s: &caroli_lattice::Site| match s.cell {
            [2, 1, _] => Some(one(0.7)),
            [3, 2, _] => Some(one(-0.4)),
            _ => None,
        };
        let device = Model::new(lat.clone())
            .hopping(0, 0, [1, 0, 0], one(-1.0))
            .hopping(0, 0, [0, 1, 0], one(-1.0))
            .with_onsite_term(pot);
        let region = Region::cuboid(&lat, [6, 4, 0]);
        let slices = region.slices(0);
        let two = TwoTerminal::from_model(&device, &clean, &region, 0);
        let multi = MultiTerminal::from_model(
            &device,
            &clean,
            &region,
            &[
                Interface { slice: slices[0].clone(), outward: [-1, 0, 0] },
                Interface { slice: slices[5].clone(), outward: [1, 0, 0] },
            ],
        );
        for e in [-1.3, 0.2, 0.9] {
            let t2 = two.transmission(e).unwrap();
            let sol = multi.solve(e, SelfEnergyMethod::Modes).unwrap();
            let tm = sol.transmission(&multi, 0, 1);
            assert!((t2 - tm).abs() < 1e-10, "E {e}: {t2} vs {tm}");
            assert!((sol.transmission(&multi, 1, 0) - tm).abs() < 1e-10);
        }
    }
}
