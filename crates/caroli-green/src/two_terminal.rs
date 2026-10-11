//! Two-terminal devices: recursive Green's function and Caroli transmission.

use crate::modes::{LeadModes, sigma_left, sigma_right};
use crate::{GreenError, broadening, decimation, inverse, resolvent_matrix, solve_right, trace};
use caroli_lattice::Region;
use caroli_model::{CMat, Model, c64};

/// A semi-infinite lead: `h00` is the cell Hamiltonian, `h01 = ⟨n| H |n+1⟩` the coupling to
/// the next cell along the transport axis (towards +∞).
#[derive(Clone, Debug)]
pub struct Lead {
    pub h00: CMat,
    pub h01: CMat,
}

impl Lead {
    /// Modes of the lead at `energy`.
    pub fn modes(&self, energy: f64) -> Result<LeadModes, GreenError> {
        LeadModes::new(&self.h00, &self.h01, energy)
    }
}

/// How lead self-energies are computed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SelfEnergyMethod {
    /// Mode matching (exact at real energy).
    Modes,
    /// Decimation at energy `E + iη`.
    Decimation { eta: f64 },
}

/// A device of `N` slices between a left lead (towards −∞) and a right lead (towards +∞).
///
/// `couplings[n] = ⟨n| H |n+1⟩`. The left lead couples to slice 0 through `left.h01`
/// (`⟨lead| H |0⟩`) and the right lead to slice `N−1` through `right.h01` (`⟨N−1| H |lead⟩`):
/// leads and device share the cross-section at the interfaces.
#[derive(Clone, Debug)]
pub struct TwoTerminal {
    pub slices: Vec<CMat>,
    pub couplings: Vec<CMat>,
    pub left: Lead,
    pub right: Lead,
}

impl TwoTerminal {
    /// Two-terminal system built from a model on a region cut along primitive vector `axis`;
    /// both leads continue the first and last slices of the region with the clean `lead`
    /// model (the region's own model may carry disorder).
    pub fn from_model(device: &Model, lead: &Model, region: &Region, axis: usize) -> Self {
        let (slices, diag, off) = device.slice_blocks(region, axis);
        let mut step = [0i64; 3];
        step[axis] = 1;
        let first = &slices[0];
        let last = &slices[slices.len() - 1];
        let left = Lead { h00: lead.hamiltonian(first), h01: lead.coupling(first, &first.translated(step)) };
        let right = Lead { h00: lead.hamiltonian(last), h01: lead.coupling(last, &last.translated(step)) };
        Self { slices: diag, couplings: off, left, right }
    }

    /// Lead self-energies `(Σ_L, Σ_R)` on the first and last slices.
    pub fn self_energies(&self, energy: f64, method: SelfEnergyMethod) -> Result<(CMat, CMat), GreenError> {
        match method {
            SelfEnergyMethod::Modes => {
                let ml = self.left.modes(energy)?;
                let mr = self.right.modes(energy)?;
                Ok((sigma_left(&self.left.h01, &ml), sigma_right(&self.right.h01, &mr)))
            }
            SelfEnergyMethod::Decimation { eta } => {
                let z = c64::new(energy, eta);
                let away_left = self.left.h01.adjoint().to_owned();
                let gl = decimation::surface_green(&self.left.h00, &away_left, z)?;
                let gr = decimation::surface_green(&self.right.h00, &self.right.h01, z)?;
                let sigma_l = self.left.h01.adjoint() * &gl * &self.left.h01;
                let sigma_r = &self.right.h01 * &gr * self.right.h01.adjoint();
                Ok((sigma_l, sigma_r))
            }
        }
    }

    /// Transmission `T(E) = Tr[Γ_L G_1N Γ_R G_1N†]` with mode-matching self-energies.
    pub fn transmission(&self, energy: f64) -> Result<f64, GreenError> {
        self.transmission_with(energy, SelfEnergyMethod::Modes)
    }

    /// Transmission with a chosen self-energy method.
    pub fn transmission_with(&self, energy: f64, method: SelfEnergyMethod) -> Result<f64, GreenError> {
        let (sigma_l, sigma_r) = self.self_energies(energy, method)?;
        let eta = match method {
            SelfEnergyMethod::Modes => 0.0,
            SelfEnergyMethod::Decimation { eta } => eta,
        };
        let z = c64::new(energy, eta);
        let n = self.slices.len();
        // Left-connected Green's functions g_n = [z − H_nn − V_{n−1}† g_{n−1} V_{n−1} − Σ]⁻¹
        // and the first row G_1n = G_1,n−1 V_{n−1} g_n.
        let mut a = resolvent_matrix(z, &self.slices[0]) - &sigma_l;
        if n == 1 {
            a -= &sigma_r;
        }
        let mut g = inverse(&a);
        let mut g1n = g.clone();
        for k in 1..n {
            let v = &self.couplings[k - 1];
            let mut a = resolvent_matrix(z, &self.slices[k]) - v.adjoint() * &g * v;
            if k == n - 1 {
                a -= &sigma_r;
            }
            g = inverse(&a);
            g1n = solve_right(&(&g1n * v), &a);
        }
        let t = broadening(&sigma_l) * &g1n * broadening(&sigma_r) * g1n.adjoint();
        Ok(trace(&t).re)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(x: f64) -> CMat {
        CMat::from_fn(1, 1, |_, _| c64::new(x, 0.0))
    }

    fn chain(n: usize) -> TwoTerminal {
        TwoTerminal {
            slices: vec![one(0.0); n],
            couplings: vec![one(-1.0); n - 1],
            left: Lead { h00: one(0.0), h01: one(-1.0) },
            right: Lead { h00: one(0.0), h01: one(-1.0) },
        }
    }

    /// Clean chain with hopping −1: T = 1 inside the band |E| < 2, 0 outside, at every energy,
    /// including the band centre where decimation is ill-conditioned.
    #[test]
    fn chain_transmission_modes() {
        let sys = chain(5);
        for e in [-1.9, -1.0, 0.0, 1e-9, 0.3, 1.5] {
            let t = sys.transmission(e).unwrap();
            assert!((t - 1.0).abs() < 1e-12, "T({e}) = {t}");
        }
        assert!(sys.transmission(2.5).unwrap().abs() < 1e-12);
    }

    #[test]
    fn chain_transmission_decimation() {
        let sys = chain(5);
        for e in [-1.9, -1.0, 0.3, 1.5] {
            let t = sys.transmission_with(e, SelfEnergyMethod::Decimation { eta: 1e-12 }).unwrap();
            assert!((t - 1.0).abs() < 1e-9, "T({e}) = {t}");
        }
    }

    /// A single impurity of strength u in a chain: T = 1 / (1 + (u / (2 sin k))²), E = −2 cos k.
    #[test]
    fn chain_impurity() {
        let mut sys = chain(5);
        let u = 0.7;
        sys.slices[2] = one(u);
        for e in [-1.2, 0.0, 0.8] {
            let k = (-e / 2.0f64).acos();
            let exact = 1.0 / (1.0 + (u / (2.0 * k.sin())).powi(2));
            let t = sys.transmission(e).unwrap();
            assert!((t - exact).abs() < 1e-12, "T({e}) = {t}, exact {exact}");
        }
    }

    /// Analytic surface Green's function of the chain: g = (E − i sqrt(4 − E²)) / 2 for |E| < 2,
    /// from decimation and from the modes.
    #[test]
    fn chain_surface_green() {
        let e = 0.7;
        let exact = c64::new(e / 2.0, -(4.0 - e * e).sqrt() / 2.0);
        let g = decimation::surface_green(&one(0.0), &one(-1.0), c64::new(e, 1e-12)).unwrap();
        assert!((g[(0, 0)] - exact).norm() < 1e-8, "{} vs {exact}", g[(0, 0)]);
        let modes = LeadModes::new(&one(0.0), &one(-1.0), e).unwrap();
        let g = crate::modes::surface_green_right(&one(0.0), &one(-1.0), e, &modes);
        assert!((g[(0, 0)] - exact).norm() < 1e-12, "{} vs {exact}", g[(0, 0)]);
        assert_eq!(modes.channels(), 1);
    }
}
