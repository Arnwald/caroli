//! Disorder: reproducible random streams, Anderson disorder and classical magnetic impurities.
//!
//! The draws follow the algorithms of the thesis codes (same order, same formulas). With
//! [`rng::Xoshiro256::gfortran`] they rebuild the thesis configurations bit for bit; with any
//! other seed they are ordinary, reproducible random configurations.

pub mod rng;

use caroli_lattice::Site;
use caroli_model::{CMat, OnSite, c64};
use rng::Xoshiro256;
use std::collections::HashMap;

/// `n` distinct sites drawn among `candidates` by a partial Fisher–Yates shuffle (thesis
/// `impurities_set`): for `i = 1..n`, swap entry `i` with a uniform entry of `i..=len`. The
/// order of `candidates` matters for reproducibility; the thesis lists the impurity region
/// with x varying fastest.
pub fn choose_sites(rng: &mut Xoshiro256, candidates: &[Site], n: usize) -> Vec<Site> {
    assert!(n <= candidates.len(), "{n} impurities for {} sites", candidates.len());
    let mut tab: Vec<usize> = (0..candidates.len()).collect();
    for i in 0..n {
        let r = rng.integer(i + 1, candidates.len()) - 1;
        tab.swap(i, r);
    }
    tab[..n].iter().map(|&k| candidates[k]).collect()
}

/// Distribution of classical unit moments, axisymmetric about z: `n_z = a u + b` for a
/// uniform `u`, azimuth `φ = 2π u'` uniform. The formulas are those of the thesis codes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Moments {
    /// Isotropic: `n_z` uniform in [−1, 1] (thesis `spin = 0`).
    Isotropic,
    /// Inside a cone of half-angle `theta0` (radians) around z: `n_z` uniform in [cos θ0, 1]
    /// (thesis `spin ≠ 0`, `theta_0`).
    Cone { theta0: f64 },
    /// Mean polarisation `mz` along z: `n_z` uniform in [2 mz − 1, 1] (thesis 2015 codes, `Mz`).
    Polarized { mz: f64 },
}

impl Moments {
    fn nz(&self, u: f64) -> f64 {
        match *self {
            Moments::Isotropic => 2.0 * u - 1.0,
            Moments::Cone { theta0 } => -(1.0 - theta0.cos()) * u + 1.0,
            Moments::Polarized { mz } => {
                assert!((0.0..=1.0).contains(&mz), "polarisation mz = {mz} outside [0, 1]");
                2.0 * (1.0 - mz) * u + 2.0 * mz - 1.0
            }
        }
    }

    /// `n` moments: all the `n_z` draws first, then all the azimuths (thesis order).
    pub fn draw(&self, rng: &mut Xoshiro256, n: usize) -> Vec<[f64; 3]> {
        let nz: Vec<f64> = rng.uniforms(n).into_iter().map(|u| self.nz(u)).collect();
        let phi: Vec<f64> = rng.uniforms(n).into_iter().map(|u| 2.0 * std::f64::consts::PI * u).collect();
        nz.iter()
            .zip(&phi)
            .map(|(&z, &p)| {
                let perp = (1.0 - z * z).sqrt();
                [perp * p.cos(), perp * p.sin(), z]
            })
            .collect()
    }
}

/// Classical magnetic impurities: distinct sites and their unit moments.
#[derive(Clone, Debug, PartialEq)]
pub struct MagneticImpurities {
    pub sites: Vec<Site>,
    pub moments: Vec<[f64; 3]>,
}

impl MagneticImpurities {
    /// `n` impurities among `candidates` (see [`choose_sites`]), moments drawn from `moments`.
    pub fn generate(rng: &mut Xoshiro256, candidates: &[Site], n: usize, moments: Moments) -> Self {
        let sites = choose_sites(rng, candidates, n);
        let moments = moments.draw(rng, n);
        Self { sites, moments }
    }

    /// Mean moment.
    pub fn magnetisation(&self) -> [f64; 3] {
        let n = self.moments.len().max(1) as f64;
        std::array::from_fn(|a| self.moments.iter().map(|m| m[a]).sum::<f64>() / n)
    }

    /// On-site exchange term `J (n_x S_x + n_y S_y + n_z S_z)` on the impurity sites, for the
    /// spin matrices `s` of the model (e.g. [`caroli_model::presets::bhz_spin`]).
    pub fn exchange(&self, j: f64, s: &[CMat; 3]) -> OnSiteMap {
        let mut map = OnSiteMap::default();
        for (site, n) in self.sites.iter().zip(&self.moments) {
            let (rows, cols) = (s[0].nrows(), s[0].ncols());
            let v = CMat::from_fn(rows, cols, |a, b| {
                c64::new(j, 0.0) * (s[0][(a, b)] * n[0] + s[1][(a, b)] * n[1] + s[2][(a, b)] * n[2])
            });
            map.add(*site, v);
        }
        map
    }
}

/// Anderson disorder: uniform on-site energies in `[−w/2, w/2]` times the identity.
#[derive(Clone, Debug, PartialEq)]
pub struct Anderson {
    pub sites: Vec<Site>,
    pub energies: Vec<f64>,
}

impl Anderson {
    /// One energy per site, drawn in the order of `sites`.
    pub fn generate(rng: &mut Xoshiro256, sites: &[Site], w: f64) -> Self {
        let energies = rng.uniforms(sites.len()).into_iter().map(|u| w * (u - 0.5)).collect();
        Self { sites: sites.to_vec(), energies }
    }

    /// On-site term for sites with `orbitals` orbitals.
    pub fn onsite(&self, orbitals: usize) -> OnSiteMap {
        let mut map = OnSiteMap::default();
        for (site, &e) in self.sites.iter().zip(&self.energies) {
            map.add(
                *site,
                CMat::from_fn(
                    orbitals,
                    orbitals,
                    |a, b| if a == b { c64::new(e, 0.0) } else { c64::new(0.0, 0.0) },
                ),
            );
        }
        map
    }
}

/// Site-dependent on-site matrices stored in a map (sums when a site is added twice).
#[derive(Clone, Debug, Default)]
pub struct OnSiteMap {
    map: HashMap<Site, CMat>,
}

impl OnSiteMap {
    pub fn add(&mut self, site: Site, m: CMat) {
        match self.map.get_mut(&site) {
            Some(v) => *v += m,
            None => {
                self.map.insert(site, m);
            }
        }
    }

    pub fn get(&self, site: &Site) -> Option<&CMat> {
        self.map.get(site)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

impl OnSite for OnSiteMap {
    fn matrix(&self, site: &Site) -> Option<CMat> {
        self.map.get(site).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(n: i64) -> Vec<Site> {
        (0..n).map(|x| Site { cell: [x, 0, 0], sub: 0 }).collect()
    }

    #[test]
    fn distinct_sites() {
        let mut rng = Xoshiro256::seed_from_u64(7);
        let s = choose_sites(&mut rng, &line(50), 30);
        let mut u = s.clone();
        u.sort();
        u.dedup();
        assert_eq!(u.len(), 30);
    }

    #[test]
    fn moment_distributions() {
        let mut rng = Xoshiro256::seed_from_u64(3);
        for m in [Moments::Isotropic, Moments::Cone { theta0: 0.3 }, Moments::Polarized { mz: 0.8 }] {
            for n in m.draw(&mut rng, 200) {
                assert!((n[0] * n[0] + n[1] * n[1] + n[2] * n[2] - 1.0).abs() < 1e-12);
            }
        }
        // Polarisation 1 and cone 0: all moments along +z.
        let up = Moments::Polarized { mz: 1.0 }.draw(&mut rng, 10);
        assert!(up.iter().all(|n| n[2] == 1.0));
        let mean = MagneticImpurities {
            sites: line(1000),
            moments: Moments::Polarized { mz: 0.6 }.draw(&mut rng, 1000),
        }
        .magnetisation();
        assert!((mean[2] - 0.6).abs() < 0.05, "<n_z> = {}", mean[2]);
    }
}
