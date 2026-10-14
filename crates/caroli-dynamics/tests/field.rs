//! Evolution under a uniform field: the k-space split operator ([`Torus`]) against the
//! real-space Magnus integrator ([`Driven`]) on a disordered spinful graphene torus, the Stark
//! breathing of a wave packet, and the Fermi sea as a stochastic trace.

use caroli_disorder::rng::Xoshiro256;
use caroli_dynamics::{Driven, Evolution, Torus};
use caroli_lattice::{Lattice, Region, Site};
use caroli_model::pauli::{s0, sx, sy, sz};
use caroli_model::{CMat, Model, c64, combine};
use std::collections::HashMap;

/// Spinful graphene with Anderson disorder and magnetic impurities on an `n × n` torus.
fn graphene(n: usize, seed: u64) -> Model {
    let mut rng = Xoshiro256::seed_from_u64(seed);
    let mut terms: HashMap<Site, CMat> = HashMap::new();
    for y in 0..n as i64 {
        for x in 0..n as i64 {
            for sub in 0..2 {
                let (u, v, w) = (rng.uniform(), rng.uniform(), rng.uniform());
                let m = if u < 0.3 {
                    let (z, phi) = (2.0 * v - 1.0, 2.0 * std::f64::consts::PI * w);
                    let p = (1.0 - z * z).sqrt();
                    combine(&[(0.8 * p * phi.cos(), &sx()), (0.8 * p * phi.sin(), &sy()), (0.8 * z, &sz())])
                } else {
                    combine(&[(v - 0.5, &s0())])
                };
                terms.insert(Site { cell: [x, y, 0], sub }, m);
            }
        }
    }
    let t = combine(&[(-1.0, &s0())]);
    Model::new(Lattice::honeycomb(1.0, 2))
        .hopping(0, 1, [0, 0, 0], &t)
        .hopping(0, 1, [0, -1, 0], &t)
        .hopping(0, 1, [1, -1, 0], &t)
        .with_onsite_term(move |s: &Site| terms.get(s).cloned())
}

/// The two integrators give the same state and current: different methods (split operator in k
/// space, second order; Magnus with Chebyshev exponentials in real space, fourth order).
#[test]
fn split_operator_matches_magnus() {
    let n = 6;
    let model = graphene(n, 4);
    let force = [0.15, -0.3, 0.0];
    let torus = Torus::new(&model, n).with_force(force);
    let start = torus.localized([2, 3], 1, Some((-10.0, 0.2)));
    let (dt, steps) = (0.004, 500);
    let ev = Evolution { dt, steps, every: steps, fermi: 0.0, densities: true };
    let record = torus.evolve(std::slice::from_ref(&start), &ev);
    // The same state in real space.
    let region = Region::cuboid(model.lattice(), [n, n, 0]);
    let driven = Driven::new(&model, &region, &[[n as i64, 0, 0], [0, n as i64, 0]]).with_force(force);
    let nn = n * n;
    let index = |i: usize| {
        // Torus entry x + n y + n² c, c = 2 sub + spin.
        let (cell, c) = (i % nn, i / nn);
        let (x, y) = (cell % n, cell / n);
        let site = region.index_of(&Site { cell: [x as i64, y as i64, 0], sub: c / 2 }).expect("site");
        region.offset(site) + c % 2
    };
    let mut x0 = start.clone();
    torus.to_x(&mut x0);
    let mut psi = vec![c64::new(0.0, 0.0); driven.len()];
    for (i, z) in x0.iter().enumerate() {
        psi[index(i)] = *z;
    }
    let big = 0.1;
    for k in 0..(dt * steps as f64 / big).round() as usize {
        driven.step(&mut psi, k as f64 * big, big, 1e-13);
    }
    let t = dt * steps as f64;
    let density = &record.densities[1];
    let err =
        density.iter().enumerate().map(|(i, d)| (d - psi[index(i)].norm_sqr()).abs()).fold(0.0, f64::max);
    let peak = density.iter().fold(0.0f64, |m, d| m.max(*d));
    assert!(err < 2e-3 * peak, "densities differ by {err:e} (peak {peak:e})");
    let area = torus.area();
    for (dir, j) in [([1.0, 0.0, 0.0], record.current[1][0]), ([0.0, 1.0, 0.0], record.current[1][1])] {
        let v = driven.velocity(&psi, t, dir);
        assert!((v - j * area).abs() < 2e-3 * v.abs().max(0.1), "velocity {v} vs {}", j * area);
    }
    assert!((record.norm[1] - 1.0).abs() < 1e-12);
}

/// A particle on one site of the square lattice under a force F along x: Bloch oscillations of
/// the width along x, `⟨x²⟩ = (8J²/F²) sin²(Ft/2)`, free spreading `⟨y²⟩ = 2J²t²` along y.
#[test]
fn stark_breathing() {
    let n = 64;
    let model =
        Model::new(Lattice::square(1.0, 1)).hopping(0, 0, [1, 0, 0], -1.0).hopping(0, 0, [0, 1, 0], -1.0);
    let f = 0.5;
    let torus = Torus::new(&model, n).with_force([f, 0.0, 0.0]);
    let start = torus.localized([20, 40], 0, None);
    let ev = Evolution { dt: 0.01, steps: 800, every: 50, fermi: 0.0, densities: false };
    let record = torus.evolve(&[start], &ev);
    for (t, w) in record.times.iter().zip(&record.width) {
        let exact = (8.0 / (f * f) * (0.5 * f * t).sin().powi(2) + 2.0 * t * t).sqrt();
        assert!((w - exact).abs() < 1e-6 * exact.max(1.0), "t = {t}: width {w} vs {exact}");
    }
    // Without a scalar potential the current oscillates around zero: ⟨v_x⟩ = 0 for a symmetric start.
    assert!(record.current.iter().all(|j| j[0].abs() < 1e-12 && j[1].abs() < 1e-12));
}

/// Clean graphene: with one occupied band per k, a single random-phase vector gives the exact
/// trace (no cross terms), whatever the seed; the field creates pairs and drives a current along
/// the force.
#[test]
fn fermi_sea() {
    // No Dirac point on the k grid (n not a multiple of 3): every band state is either filled or empty.
    let n = 26;
    let t = CMat::from_fn(1, 1, |_, _| c64::new(-1.0, 0.0));
    let model = Model::new(Lattice::honeycomb(1.0, 1))
        .hopping(0, 1, [0, 0, 0], &t)
        .hopping(0, 1, [0, -1, 0], &t)
        .hopping(0, 1, [1, -1, 0], &t);
    let torus = Torus::new(&model, n).with_force([0.0, 0.1, 0.0]);
    let ev = Evolution { dt: 0.02, steps: 200, every: 50, fermi: 0.0, densities: false };
    let a = torus.evolve(&torus.filled(0.0, 1, 1), &ev);
    let b = torus.evolve(&torus.filled(0.0, 1, 2), &ev);
    for k in 0..a.times.len() {
        assert!(
            (a.excited[k] - b.excited[k]).abs() < 1e-12 && (a.current[k][1] - b.current[k][1]).abs() < 1e-12
        );
    }
    assert!((a.norm[0] - (n * n) as f64).abs() < 1e-9, "one occupied state per k");
    assert!(a.excited[0].abs() < 1e-14);
    assert!(a.excited.windows(2).all(|w| w[1] > w[0]), "{:?}", a.excited);
    assert!(a.current.iter().skip(1).all(|j| j[1] > 0.0 && j[0].abs() < 1e-12), "{:?}", a.current);
}

/// With disorder, the Fermi sea of the whole Hamiltonian is stationary without field: its
/// carriers above the Fermi energy of the clean bands stay put. The clean Fermi sea is not a
/// stationary state of the disordered Hamiltonian: the disorder excites it at once.
#[test]
fn disordered_fermi_sea_is_stationary() {
    let n = 12;
    let torus = Torus::new(&graphene(n, 2), n);
    let ev = Evolution { dt: 0.05, steps: 200, every: 50, fermi: 0.0, densities: false };
    let ground = torus.evolve(&torus.fermi_sea(0.0, 4, 3, 1024), &ev);
    let quench = torus.evolve(&torus.filled(0.0, 4, 3), &ev);
    let drift =
        |r: &caroli_dynamics::Record| r.excited.iter().map(|x| (x - r.excited[0]).abs()).fold(0.0, f64::max);
    assert!(drift(&ground) < 0.1 * drift(&quench), "{} vs {}", drift(&ground), drift(&quench));
    // Half filling: half of the 4 n² orbitals are occupied (up to the stochastic error).
    let occupied = ground.norm[0] / (4 * n * n) as f64;
    assert!((occupied - 0.5).abs() < 0.02, "{occupied}");
}

/// One spin of clean spinful graphene: half the carriers, current and pairs of both spins.
#[test]
fn spin_sector() {
    let n = 26;
    let t = combine(&[(-1.0, &s0())]);
    let model = Model::new(Lattice::honeycomb(1.0, 2))
        .hopping(0, 1, [0, 0, 0], &t)
        .hopping(0, 1, [0, -1, 0], &t)
        .hopping(0, 1, [1, -1, 0], &t);
    let torus = Torus::new(&model, n).with_force([0.0, 0.1, 0.0]);
    let ev = Evolution { dt: 0.02, steps: 200, every: 50, fermi: 0.0, densities: false };
    let both = torus.evolve(&torus.filled(0.0, 1, 1), &ev);
    // Orbitals A↑, A↓, B↑, B↓: spin up is {0, 2}.
    let up = Torus::new(&model, n).with_force([0.0, 0.1, 0.0]).with_sector(&[0, 2]);
    let one = up.evolve(&up.filled(0.0, 1, 1), &ev);
    assert!((one.norm[0] - (n * n) as f64).abs() < 1e-9, "one occupied spin-up state per k");
    for k in 1..both.times.len() {
        assert!((2.0 * one.excited[k] - both.excited[k]).abs() < 1e-12 * both.excited[k].abs().max(1.0));
        assert!(
            (2.0 * one.current[k][1] - both.current[k][1]).abs() < 1e-12 * both.current[k][1].abs().max(1.0)
        );
        assert!(one.excited[k] > 0.0);
    }
}

#[test]
#[should_panic(expected = "splits")]
fn sector_must_not_split_coupled_orbitals() {
    let t = combine(&[(-1.0, &s0())]);
    let model = Model::new(Lattice::honeycomb(1.0, 2)).hopping(0, 1, [0, 0, 0], &t);
    let _ = Torus::new(&model, 6).with_sector(&[0, 1]);
}
