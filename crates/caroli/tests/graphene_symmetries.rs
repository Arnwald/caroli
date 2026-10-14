//! Exact relations of spinful graphene with classical magnetic impurities (`J n·σ` on a random
//! fraction of the sites, the model of the graphene study): the Hamiltonian anticommutes with the
//! product `SΘ` of the sublattice sign and time reversal, whatever the moments, and maps the field
//! `E` onto `−E`. Hence, configuration by configuration,
//! - `ρ↑(i, ε) = ρ↓(i, −ε)` on every site, and the total local density of states is the same at
//!   `±ε` (no puddles of charge at the neutrality point);
//! - with the moments along ±z (spin conserved), the current of the spin-down valence band is
//!   `j↓(E) = −j↑(−E)`: both spins carry the same part odd in `E`.

use caroli::dynamics::{Evolution, Torus};
use caroli::kpm::Spectrum;
use caroli::prelude::*;
use std::collections::HashMap;

/// Spinful graphene on an `n × n` torus (orbitals A↑, A↓, B↑, B↓) with impurities on 40 % of the
/// sites of each sublattice: isotropic moments, or A along +z and B along −z.
fn graphene(n: usize, isotropic: bool, j: f64, seed: u64) -> Model {
    let mut rng = Xoshiro256::seed_from_u64(seed);
    let mut terms: HashMap<Site, CMat> = HashMap::new();
    for sub in 0..2 {
        let cells: Vec<Site> =
            (0..n as i64).flat_map(|y| (0..n as i64).map(move |x| Site { cell: [x, y, 0], sub })).collect();
        let moments = if isotropic { Moments::Isotropic } else { Moments::Polarized { mz: 1.0 } };
        let imp = MagneticImpurities::generate(&mut rng, &cells, (0.4 * (n * n) as f64) as usize, moments);
        let s = if !isotropic && sub == 1 { -j } else { j };
        for (site, m) in imp.sites.iter().zip(&imp.moments) {
            terms.insert(*site, combine(&[(s * m[0], &sx()), (s * m[1], &sy()), (s * m[2], &sz())]));
        }
    }
    let t = combine(&[(-1.0, &s0())]);
    Model::new(Lattice::honeycomb(1.0 / 3f64.sqrt(), 2))
        .hopping(0, 1, [0, 0, 0], &t)
        .hopping(0, 1, [0, -1, 0], &t)
        .hopping(0, 1, [1, -1, 0], &t)
        .with_onsite_term(move |s: &Site| terms.get(s).cloned())
}

#[test]
fn local_densities_of_states_at_opposite_energies() {
    for isotropic in [true, false] {
        let n = 12;
        let model = graphene(n, isotropic, 1.0, 3);
        let region = Region::cuboid(model.lattice(), [n, n, 0]);
        let ni = n as i64;
        let spectrum =
            Spectrum::periodic(&model, &region, &[[ni, 0, 0], [0, ni, 0]]).moments(256).bounds(-5.0, 5.0);
        let sites: Vec<usize> = (0..region.len()).step_by(5).collect();
        let energies = [0.1, -0.1, 0.4, -0.4];
        let (mut largest, mut asymmetry) = (0.0f64, 0.0f64);
        for site in spectrum.orbital_ldos(&sites, &energies) {
            let (up, down) = (&site[0], &site[1]);
            for e in [0, 2] {
                assert!((up[e] - down[e + 1]).abs() < 1e-12, "ρ↑(ε) = {}, ρ↓(−ε) = {}", up[e], down[e + 1]);
                assert!((up[e] + down[e] - up[e + 1] - down[e + 1]).abs() < 1e-12);
                largest = largest.max(up[e]);
                asymmetry = asymmetry.max((up[e] - up[e + 1]).abs());
            }
        }
        // Each spin on its own is far from symmetric: the relation is not a trivial one.
        assert!(largest > 0.01 && asymmetry > 0.1 * largest, "{largest} {asymmetry}");
    }
}

/// The current of one spin's clean valence band, traced exactly: the sum over every A site of
/// that spin's orbital projected on the band (`P Π_A P = P/2` for graphene's clean bands).
fn band_current(model: &Model, n: usize, field: f64, spin: usize) -> Vec<f64> {
    let sector = [spin, spin + 2];
    let torus = Torus::new(model, n).with_force([0.0, field, 0.0]).with_sector(&sector);
    let ev = Evolution { dt: 0.05, steps: 200, every: 50, fermi: 0.0, densities: false };
    let mut total = vec![0.0; 5];
    for y in 0..n {
        for x in 0..n {
            let r = torus.evolve(&[torus.localized([x, y], spin, Some((-10.0, 0.0)))], &ev);
            total.iter_mut().zip(&r.current).for_each(|(a, j)| *a += j[1]);
        }
    }
    total
}

#[test]
fn spin_currents_under_opposite_fields() {
    // No Dirac point on the k grid (n not a multiple of 3): the band is well defined.
    let n = 8;
    let model = graphene(n, false, 1.0, 7);
    let up = band_current(&model, n, 0.05, 0);
    let up_reversed = band_current(&model, n, -0.05, 0);
    let down = band_current(&model, n, 0.05, 1);
    let scale = up.iter().map(|j| j.abs()).fold(0.0, f64::max);
    assert!(scale > 1e-3, "{up:?}");
    for k in 0..up.len() {
        assert!((down[k] + up_reversed[k]).abs() < 1e-12 * scale.max(1.0), "{down:?} vs {up_reversed:?}");
    }
}
