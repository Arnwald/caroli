//! Multi-terminal devices of any geometry: leads attached to interface slices, Green's function
//! of the whole device by a direct solve, transmissions `T_pq = Tr[Γ_p G_pq Γ_q G_pq†]`.
//!
//! The direct solve costs `O(n³)` in the number of device orbitals `n`: it suits devices up to a
//! few thousand orbitals (Hall bars, crosses, small flakes). Long quasi-one-dimensional devices
//! are better served by [`crate::TwoTerminal`].

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
    /// Device orbitals of the interface slice, in the order of the lead-cell orbitals. The
    /// interface couples to the first lead cell through `h01` (the lead continues the slice).
    pub orbitals: Vec<usize>,
}

impl AttachedLead {
    /// Self-energy on the interface orbitals.
    pub fn self_energy(&self, energy: f64, method: SelfEnergyMethod) -> Result<CMat, GreenError> {
        match method {
            SelfEnergyMethod::Modes => {
                Ok(sigma_right(&self.h01, &LeadModes::new(&self.h00, &self.h01, energy)?))
            }
            SelfEnergyMethod::ModesBroadened { eta } => {
                Ok(sigma_right(&self.h01, &LeadModes::at(&self.h00, &self.h01, c64::new(energy, eta))?))
            }
            SelfEnergyMethod::Decimation { eta } => {
                let g = decimation::surface_green(&self.h00, &self.h01, c64::new(energy, eta))?;
                Ok(&self.h01 * &g * self.h01.adjoint())
            }
        }
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
    /// clean `lead` model.
    pub fn from_model(device: &Model, lead: &Model, region: &Region, interfaces: &[Interface]) -> Self {
        let h = device.hamiltonian(region);
        let leads = interfaces
            .iter()
            .map(|itf| {
                let orbitals = itf
                    .slice
                    .sites()
                    .iter()
                    .enumerate()
                    .flat_map(|(k, s)| {
                        let i = region.index_of(s).expect("interface site outside the device");
                        let (o, n) = (region.offset(i), region.site_orbitals(i));
                        debug_assert_eq!(n, itf.slice.site_orbitals(k));
                        o..o + n
                    })
                    .collect();
                AttachedLead {
                    h00: lead.hamiltonian(&itf.slice),
                    h01: lead.coupling(&itf.slice, &itf.slice.translated(itf.outward)),
                    orbitals,
                }
            })
            .collect();
        Self { h, leads, region: Some(region.clone()) }
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
