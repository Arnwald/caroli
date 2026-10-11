//! Two-terminal devices: recursive Green's function and Caroli transmission.

use crate::modes::{LeadModes, sigma_left, sigma_right};
use crate::{GreenError, broadening, decimation, inverse, resolvent_matrix, trace};
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

    /// Modes of the lead at a complex energy (see [`LeadModes::at`]).
    pub fn modes_at(&self, z: c64) -> Result<LeadModes, GreenError> {
        LeadModes::at(&self.h00, &self.h01, z)
    }
}

/// How lead self-energies are computed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SelfEnergyMethod {
    /// Mode matching (exact at real energy).
    Modes,
    /// Mode matching at `E + iη` for the leads only, the device staying at real energy (the
    /// thesis codes: generalised Schur decomposition at `E + iη`).
    ModesBroadened { eta: f64 },
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
    /// Sites of each slice (orbital layout of `slices[n]`), when built from a model; empty for
    /// systems assembled by hand.
    pub regions: Vec<Region>,
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
        Self { slices: diag, couplings: off, left, right, regions: slices }
    }

    /// Shifts the electrostatic potential of the left and right leads (bias): adds `v·1` to
    /// their cell Hamiltonians.
    pub fn with_lead_potentials(mut self, left: f64, right: f64) -> Self {
        for (lead, v) in [(&mut self.left, left), (&mut self.right, right)] {
            for i in 0..lead.h00.nrows() {
                lead.h00[(i, i)] += c64::new(v, 0.0);
            }
        }
        self
    }

    /// Transmissions at many energies, in parallel, by independent blocks.
    pub fn transmissions(&self, energies: &[f64], method: SelfEnergyMethod) -> Vec<Result<f64, GreenError>> {
        use rayon::prelude::*;
        let blocks = self.blocks();
        let parts: Vec<Self> = if blocks.len() > 1 {
            blocks.iter().map(|b| self.restricted(b)).collect()
        } else {
            vec![self.clone()]
        };
        energies.par_iter().map(|&e| parts.iter().map(|p| p.transmission_with(e, method)).sum()).collect()
    }

    /// Recursive sweeps from both ends at one energy (see [`Sweeps`]).
    pub fn sweeps(&self, energy: f64, method: SelfEnergyMethod) -> Result<Sweeps, GreenError> {
        let (sigma_l, sigma_r) = self.self_energies(energy, method)?;
        let z = c64::new(energy, device_eta(method));
        let n = self.slices.len();
        let mut left = Vec::with_capacity(n);
        for k in 0..n {
            let mut a = resolvent_matrix(z, &self.slices[k]);
            if k == 0 {
                a -= &sigma_l;
            } else {
                let v = &self.couplings[k - 1];
                a -= v.adjoint() * &left[k - 1] * v;
            }
            left.push(inverse(&a));
        }
        let mut right = vec![CMat::zeros(0, 0); n];
        for k in (0..n).rev() {
            let mut a = resolvent_matrix(z, &self.slices[k]);
            if k == n - 1 {
                a -= &sigma_r;
            } else {
                let v = &self.couplings[k];
                a -= v * &right[k + 1] * v.adjoint();
            }
            right[k] = inverse(&a);
        }
        Ok(Sweeps { z, sigma_l, sigma_r, left, right })
    }

    /// Lead self-energies `(Σ_L, Σ_R)` on the first and last slices.
    pub fn self_energies(&self, energy: f64, method: SelfEnergyMethod) -> Result<(CMat, CMat), GreenError> {
        match method {
            SelfEnergyMethod::Modes => {
                let ml = self.left.modes(energy)?;
                let mr = self.right.modes(energy)?;
                Ok((sigma_left(&self.left.h01, &ml), sigma_right(&self.right.h01, &mr)))
            }
            SelfEnergyMethod::ModesBroadened { eta } => {
                let z = c64::new(energy, eta);
                let ml = self.left.modes_at(z)?;
                let mr = self.right.modes_at(z)?;
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
        Ok(self.solve(energy, method)?.transmission())
    }

    /// Orbitals of the cross-section that no slice, coupling or lead matrix connects, as
    /// independent sets (e.g. the two spin blocks of BHZ with spin-conserving disorder). One set
    /// if the slices differ in size.
    pub fn blocks(&self) -> Vec<Vec<usize>> {
        let w = self.left.h00.nrows();
        let same = self.slices.iter().all(|h| h.nrows() == w) && self.right.h00.nrows() == w;
        if !same {
            return vec![(0..w).collect()];
        }
        let mut parent: Vec<usize> = (0..w).collect();
        fn find(p: &mut [usize], mut i: usize) -> usize {
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        let mats = self.slices.iter().chain(&self.couplings).chain([
            &self.left.h00,
            &self.left.h01,
            &self.right.h00,
            &self.right.h01,
        ]);
        for m in mats {
            for j in 0..m.ncols() {
                for i in 0..m.nrows() {
                    if m[(i, j)] != c64::new(0.0, 0.0) {
                        let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                        parent[a.max(b)] = a.min(b);
                    }
                }
            }
        }
        let mut sets: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
        for i in 0..w {
            let r = find(&mut parent, i);
            sets.entry(r).or_default().push(i);
        }
        sets.into_values().collect()
    }

    /// The system restricted to `orbitals` of every slice (one of [`Self::blocks`]).
    pub fn restricted(&self, orbitals: &[usize]) -> Self {
        let pick =
            |m: &CMat| CMat::from_fn(orbitals.len(), orbitals.len(), |i, j| m[(orbitals[i], orbitals[j])]);
        Self {
            slices: self.slices.iter().map(pick).collect(),
            couplings: self.couplings.iter().map(pick).collect(),
            left: Lead { h00: pick(&self.left.h00), h01: pick(&self.left.h01) },
            right: Lead { h00: pick(&self.right.h00), h01: pick(&self.right.h01) },
            regions: Vec::new(),
        }
    }

    /// Transmission summed over the independent [`Self::blocks`], each solved on its own: for
    /// `b` equal blocks the recursion costs `b²` times less.
    pub fn transmission_by_blocks(&self, energy: f64, method: SelfEnergyMethod) -> Result<f64, GreenError> {
        let blocks = self.blocks();
        if blocks.len() == 1 {
            return self.transmission_with(energy, method);
        }
        blocks.iter().map(|b| self.restricted(b).transmission_with(energy, method)).sum()
    }

    /// Lead self-energies and the Green's function `G_1N` between the first and last slices.
    pub fn solve(&self, energy: f64, method: SelfEnergyMethod) -> Result<EndToEnd, GreenError> {
        let (sigma_l, sigma_r) = self.self_energies(energy, method)?;
        let z = c64::new(energy, device_eta(method));
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
            g1n = &g1n * v * &g;
        }
        Ok(EndToEnd { sigma_l, sigma_r, g1n })
    }
}

/// Imaginary part of the energy used inside the device.
fn device_eta(method: SelfEnergyMethod) -> f64 {
    match method {
        SelfEnergyMethod::Modes | SelfEnergyMethod::ModesBroadened { .. } => 0.0,
        SelfEnergyMethod::Decimation { eta } => eta,
    }
}

/// Recursive sweeps of a two-terminal device at one energy `z`:
/// `left[n]` is the Green's function of slices `0..=n` with the left lead only, `right[n]` that
/// of slices `n..N` with the right lead only. Everything else follows:
/// `G_nn = [z − H_nn − Σ^<-side_n − Σ^>-side_n]⁻¹`, `G_n0 = right[n] V_{n−1}† G_{n−1,0}` and
/// `G_{n,N−1} = left[n] V_n G_{n+1,N−1}`.
#[derive(Clone, Debug)]
pub struct Sweeps {
    pub z: c64,
    pub sigma_l: CMat,
    pub sigma_r: CMat,
    pub left: Vec<CMat>,
    pub right: Vec<CMat>,
}

impl Sweeps {
    /// Full diagonal block `G_nn`.
    pub fn diagonal(&self, sys: &TwoTerminal, n: usize) -> CMat {
        let last = self.left.len() - 1;
        let mut a = resolvent_matrix(self.z, &sys.slices[n]);
        if n == 0 {
            a -= &self.sigma_l;
        } else {
            let v = &sys.couplings[n - 1];
            a -= v.adjoint() * &self.left[n - 1] * v;
        }
        if n == last {
            a -= &self.sigma_r;
        } else {
            let v = &sys.couplings[n];
            a -= v * &self.right[n + 1] * v.adjoint();
        }
        inverse(&a)
    }

    /// Column `G_n0` for every slice `n` (propagation from the left lead).
    pub fn first_column(&self, sys: &TwoTerminal) -> Vec<CMat> {
        let mut col = vec![self.diagonal(sys, 0)];
        for n in 1..self.left.len() {
            let next = &self.right[n] * sys.couplings[n - 1].adjoint() * &col[n - 1];
            col.push(next);
        }
        col
    }

    /// Column `G_{n,N−1}` for every slice `n` (propagation from the right lead).
    pub fn last_column(&self, sys: &TwoTerminal) -> Vec<CMat> {
        let last = self.left.len() - 1;
        let mut col = vec![CMat::zeros(0, 0); last + 1];
        col[last] = self.diagonal(sys, last);
        for n in (0..last).rev() {
            col[n] = &self.left[n] * &sys.couplings[n] * &col[n + 1];
        }
        col
    }
}

/// Result of [`TwoTerminal::solve`]: lead self-energies on the first and last slices and the
/// retarded Green's function between them.
#[derive(Clone, Debug)]
pub struct EndToEnd {
    pub sigma_l: CMat,
    pub sigma_r: CMat,
    pub g1n: CMat,
}

impl EndToEnd {
    /// Caroli transmission `Tr[Γ_L G_1N Γ_R G_1N†]`.
    pub fn transmission(&self) -> f64 {
        let t = broadening(&self.sigma_l) * &self.g1n * broadening(&self.sigma_r) * self.g1n.adjoint();
        trace(&t).re
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
            regions: Vec::new(),
        }
    }

    /// BHZ ribbon with on-site disorder: two spin blocks, same transmission as the full system.
    #[test]
    fn blocks_of_bhz_with_spin_conserving_disorder() {
        use caroli_model::presets::{BhzParams, bhz};
        let clean = bhz(BhzParams::default());
        let pot = |s: &caroli_lattice::Site| {
            let v = ((s.cell[0] * 7 + s.cell[1] * 3) % 5) as f64 * 0.1 - 0.2;
            Some(CMat::from_fn(4, 4, |i, j| if i == j { c64::new(v, 0.0) } else { c64::new(0.0, 0.0) }))
        };
        let device = bhz(BhzParams::default()).with_onsite_term(pot);
        let region = Region::cuboid(clean.lattice(), [6, 5, 0]);
        let sys = TwoTerminal::from_model(&device, &clean, &region, 0);
        let blocks = sys.blocks();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].len(), 10);
        for e in [-0.3, 0.05, 0.4] {
            let full = sys.transmission(e).unwrap();
            let split = sys.transmission_by_blocks(e, SelfEnergyMethod::Modes).unwrap();
            assert!((full - split).abs() < 1e-10, "E {e}: {full} vs {split}");
        }
    }

    /// The sweeps agree with the direct inversion of the whole device matrix.
    #[test]
    fn sweeps_match_full_inversion() {
        let mut sys = chain(6);
        sys.slices[2] = one(0.4);
        sys.slices[4] = one(-0.3);
        let e = 0.37;
        let sw = sys.sweeps(e, SelfEnergyMethod::Modes).unwrap();
        let n = 6;
        let mut a = CMat::from_fn(n, n, |i, j| match i.abs_diff(j) {
            0 => c64::new(e, 0.0) - sys.slices[i][(0, 0)],
            1 => c64::new(1.0, 0.0),
            _ => c64::new(0.0, 0.0),
        });
        a[(0, 0)] -= sw.sigma_l[(0, 0)];
        a[(n - 1, n - 1)] -= sw.sigma_r[(0, 0)];
        let g = inverse(&a);
        let first = sw.first_column(&sys);
        let last = sw.last_column(&sys);
        for k in 0..n {
            assert!((sw.diagonal(&sys, k)[(0, 0)] - g[(k, k)]).norm() < 1e-12);
            assert!((first[k][(0, 0)] - g[(k, 0)]).norm() < 1e-12);
            assert!((last[k][(0, 0)] - g[(k, n - 1)]).norm() < 1e-12);
        }
        let direct = sys.solve(e, SelfEnergyMethod::Modes).unwrap();
        assert!((direct.g1n[(0, 0)] - g[(0, n - 1)]).norm() < 1e-12);
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
