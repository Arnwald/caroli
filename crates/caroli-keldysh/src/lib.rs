//! Non-equilibrium steady state of two-terminal devices (Keldysh formalism).
//!
//! Each lead `p` injects electrons with occupation `f_p(E)`: `Σ^<_p = i f_p Γ_p`, and the lesser
//! Green's function is `G^< = G^r (Σ^<_L + Σ^<_R) G^a`. [`Lesser`] holds, at one energy, the
//! blocks needed for local observables (diagonal `G^<_nn` and next-neighbour `G^<_{n,n+1}`),
//! separately for the electrons injected from each lead.
//!
//! Normalisations (spin included in the orbitals): the particle current through a bond is
//! `J_{i→j} = (e/h) ∫ dE j_{i→j}(E)` with the spectral current
//! `j_{i→j}(E) = 2 Re Tr[H_ij G^<_ji(E)]`; summed over a cross-section it equals
//! `T(E) [f_L(E) − f_R(E)]` (Landauer). The occupied spectral density of orbital `a` is
//! `Im G^<_aa(E) / 2π`.

pub use caroli_green::BondCurrent;
use caroli_green::{GreenError, SelfEnergyMethod, TwoTerminal, broadening};
use caroli_model::{CMat, c64};

/// Fermi–Dirac occupation; at `temperature = 0` a step with `f(μ) = 1`.
pub fn fermi(energy: f64, mu: f64, temperature: f64) -> f64 {
    if temperature == 0.0 {
        return if energy > mu { 0.0 } else { 1.0 };
    }
    let x = (energy - mu) / temperature;
    if x > 0.0 { (-x).exp() / ((-x).exp() + 1.0) } else { 1.0 / (1.0 + x.exp()) }
}

/// Lesser blocks of the electrons injected by one lead.
#[derive(Clone, Debug)]
pub struct Injected {
    /// `G^<_{nn}` for every slice.
    pub diag: Vec<CMat>,
    /// `G^<_{n,n+1}` for `n = 0..N−1`.
    pub next: Vec<CMat>,
}

/// Lesser Green's function blocks of a two-terminal device at one energy.
#[derive(Clone, Debug)]
pub struct Lesser {
    pub energy: f64,
    pub left: Injected,
    pub right: Injected,
    /// Transmission at this energy (from the same Green's functions).
    pub transmission: f64,
    /// Occupations `(f_L, f_R)` used.
    pub occupations: (f64, f64),
}

/// `i f Γ`.
fn sigma_lesser(sigma: &CMat, f: f64) -> CMat {
    let g = broadening(sigma);
    CMat::from_fn(g.nrows(), g.ncols(), |i, j| c64::new(0.0, f) * g[(i, j)])
}

fn injected(column: &[CMat], sigma_lesser: &CMat) -> Injected {
    let n = column.len();
    let diag = column.iter().map(|g| g * sigma_lesser * g.adjoint()).collect();
    let next =
        (0..n.saturating_sub(1)).map(|k| &column[k] * sigma_lesser * column[k + 1].adjoint()).collect();
    Injected { diag, next }
}

impl Lesser {
    /// Lesser blocks at `energy` with lead occupations `f_left`, `f_right`.
    pub fn compute(
        sys: &TwoTerminal,
        energy: f64,
        method: SelfEnergyMethod,
        f_left: f64,
        f_right: f64,
    ) -> Result<Self, GreenError> {
        let sw = sys.sweeps(energy, method)?;
        let first = sw.first_column(sys);
        let last = sw.last_column(sys);
        let left = injected(&first, &sigma_lesser(&sw.sigma_l, f_left));
        let right = injected(&last, &sigma_lesser(&sw.sigma_r, f_right));
        let g1n = &last[0];
        let t = broadening(&sw.sigma_l) * g1n * broadening(&sw.sigma_r) * g1n.adjoint();
        let transmission = caroli_green::trace(&t).re;
        Ok(Self { energy, left, right, transmission, occupations: (f_left, f_right) })
    }

    /// `G^<_{nn}` (both leads).
    pub fn diag(&self, n: usize) -> CMat {
        &self.left.diag[n] + &self.right.diag[n]
    }

    /// `G^<_{n,n+1}` (both leads).
    pub fn next(&self, n: usize) -> CMat {
        &self.left.next[n] + &self.right.next[n]
    }

    /// Spectral particle current from slice `n` to slice `n+1`, `2 Re Tr[V_n G^<_{n+1,n}]`.
    pub fn interface_current(&self, sys: &TwoTerminal, n: usize) -> f64 {
        // G^<_{n+1,n} = −(G^<_{n,n+1})†.
        let v = &sys.couplings[n];
        let g = self.next(n);
        let mut acc = c64::new(0.0, 0.0);
        for i in 0..v.nrows() {
            for j in 0..v.ncols() {
                acc -= v[(i, j)] * g[(i, j)].conj();
            }
        }
        2.0 * acc.re
    }

    /// Occupied spectral density `Im G^<_aa / 2π` of every orbital of slice `n`.
    pub fn occupation(&self, n: usize) -> Vec<f64> {
        let g = self.diag(n);
        (0..g.nrows()).map(|a| g[(a, a)].im / (2.0 * std::f64::consts::PI)).collect()
    }

    /// Spectral bond currents `2 Re Tr[H_ij G^<_ji]` between every pair of coupled sites
    /// (inside a slice and between neighbouring slices), for systems built from a model.
    pub fn bond_currents(&self, sys: &TwoTerminal) -> Vec<BondCurrent> {
        assert!(!sys.regions.is_empty(), "bond currents need the slice regions (TwoTerminal::from_model)");
        let mut out = Vec::new();
        for (n, region) in sys.regions.iter().enumerate() {
            let h = &sys.slices[n];
            let g = self.diag(n);
            for i in 0..region.len() {
                for j in (i + 1)..region.len() {
                    if let Some(c) = bond(h, &g, region, region, i, j, false) {
                        out.push(BondCurrent { from: region.sites()[i], to: region.sites()[j], current: c });
                    }
                }
            }
            if n + 1 < sys.regions.len() {
                let next = &sys.regions[n + 1];
                let v = &sys.couplings[n];
                let g = self.next(n);
                for i in 0..region.len() {
                    for j in 0..next.len() {
                        if let Some(c) = bond(v, &g, region, next, i, j, true) {
                            out.push(BondCurrent {
                                from: region.sites()[i],
                                to: next.sites()[j],
                                current: c,
                            });
                        }
                    }
                }
            }
        }
        out
    }
}

/// `2 Re Tr[H_ij G^<_ji]` for site `i` of region `a` and site `j` of region `b`, or `None` when
/// the sites are not coupled. `h` holds `⟨a| H |b⟩`; `g` is `G^<_{ab}` and `G^<_{ji}` is read
/// as `−conj(G^<_{ij})` when `transposed` (different slices), or directly from `g` otherwise.
fn bond(
    h: &CMat,
    g: &CMat,
    a: &caroli_lattice::Region,
    b: &caroli_lattice::Region,
    i: usize,
    j: usize,
    transposed: bool,
) -> Option<f64> {
    let (oi, ni, oj, nj) = (a.offset(i), a.site_orbitals(i), b.offset(j), b.site_orbitals(j));
    let mut coupled = false;
    let mut acc = c64::new(0.0, 0.0);
    for p in 0..ni {
        for q in 0..nj {
            let hij = h[(oi + p, oj + q)];
            if hij != c64::new(0.0, 0.0) {
                coupled = true;
            }
            let gji = if transposed { -g[(oi + p, oj + q)].conj() } else { g[(oj + q, oi + p)] };
            acc += hij * gji;
        }
    }
    coupled.then_some(2.0 * acc.re)
}

#[cfg(test)]
mod tests {
    use super::*;
    use caroli_green::Lead;

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

    /// Current conservation and Landauer: every interface carries T (f_L − f_R).
    #[test]
    fn interface_currents_equal_landauer() {
        let mut sys = chain(7);
        sys.slices[3] = one(0.8);
        for (e, fl, fr) in [(0.3, 1.0, 0.0), (-0.9, 0.7, 0.2), (1.2, 0.0, 1.0)] {
            let l = Lesser::compute(&sys, e, SelfEnergyMethod::Modes, fl, fr).unwrap();
            let t = sys.transmission(e).unwrap();
            assert!((l.transmission - t).abs() < 1e-12);
            for n in 0..6 {
                let j = l.interface_current(&sys, n);
                assert!((j - t * (fl - fr)).abs() < 1e-12, "E {e}, interface {n}: {j} vs {}", t * (fl - fr));
            }
        }
    }

    /// In equilibrium (f_L = f_R = f), G^< = −f (G^r − G^a) = −2i f Im G^r, so the occupied
    /// density Im G^< / 2π is −f Im G^r / π.
    #[test]
    fn equilibrium_occupation() {
        let mut sys = chain(5);
        sys.slices[2] = one(-0.4);
        let e = 0.45;
        let l = Lesser::compute(&sys, e, SelfEnergyMethod::Modes, 1.0, 1.0).unwrap();
        let sw = sys.sweeps(e, SelfEnergyMethod::Modes).unwrap();
        for n in 0..5 {
            let gr = sw.diagonal(&sys, n)[(0, 0)];
            // G^< = −f (G^r − G^a) = −2i f Im G^r.
            let expected = -2.0 * gr.im / (2.0 * std::f64::consts::PI);
            assert!((l.occupation(n)[0] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn fermi_function() {
        assert_eq!(fermi(0.1, 0.1, 0.0), 1.0);
        assert_eq!(fermi(0.2, 0.1, 0.0), 0.0);
        assert!((fermi(0.1, 0.1, 0.01) - 0.5).abs() < 1e-15);
        assert!(fermi(10.0, 0.0, 0.01) < 1e-300);
    }
}
