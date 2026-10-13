//! The SCBA built from a model: the k grid, the Born limit of weak disorder, the moments of the
//! magnetic distributions and the sum rule of the averaged density of states.

use caroli_born::{Disorder, Scba, k_grid};
use caroli_disorder::Moments;
use caroli_disorder::rng::Xoshiro256;
use caroli_lattice::{Lattice, Region};
use caroli_model::presets::{BhzParams, GFactors, bhz, bhz_spin};
use caroli_model::{CMat, Model, c64};

/// `H(k)` on the grid of [`k_grid`], in its order.
#[test]
fn grid_of_the_model() {
    let model = bhz(BhzParams::default());
    let scba = Scba::from_model(&model, &[6, 4], Disorder::anderson(0.0, 4));
    let cell = Region::cuboid(model.lattice(), [1, 1, 0]);
    for (h, k) in scba.hk.iter().zip(k_grid(6, 4)) {
        assert!((h - model.bloch_hamiltonian(&cell, &[[1, 0, 0], [0, 1, 0]], &k)).norm_max() < 1e-15);
    }
    assert_eq!(scba.hk.len(), 24);
}

/// Weak Anderson disorder on the square lattice: `Σ₂ = (W²/12) G₀ + O(W⁴)`.
#[test]
fn weak_disorder_is_the_born_approximation() {
    let model =
        Model::new(Lattice::square(1.0, 1)).hopping(0, 0, [1, 0, 0], -1.0).hopping(0, 0, [0, 1, 0], -1.0);
    let w = 0.3;
    let scba = Scba::from_model(&model, &[48, 48], Disorder::anderson(w, 1));
    let (e, eta) = (-1.3, 0.05);
    let sigma = scba.self_energy(e, eta).sigma[(0, 0)];
    let g0 = scba.local_green(c64::new(e, eta), &CMat::zeros(1, 1))[(0, 0)];
    let born = g0 * (w * w / 12.0);
    assert!((sigma - born).norm() < 0.02 * born.norm(), "Σ = {sigma}, Born {born}");
    assert!(sigma.im < 0.0);
}

/// The mean and second moment of `n_z` used by [`Disorder::magnetic`] are those of the draws.
#[test]
fn moments_of_the_magnetic_distributions() {
    let s = bhz_spin(GFactors::Same);
    let mut rng = Xoshiro256::seed_from_u64(3);
    for moments in [Moments::Isotropic, Moments::Cone { theta0: 0.6 }, Moments::Polarized { mz: 0.7 }] {
        let draws = moments.draw(&mut rng, 200_000);
        let mz = draws.iter().map(|n| n[2]).sum::<f64>() / draws.len() as f64;
        let nz2 = draws.iter().map(|n| n[2] * n[2]).sum::<f64>() / draws.len() as f64;
        let (conc, j) = (0.3, 0.8);
        let d = Disorder::magnetic(conc, j, moments, &s);
        // Mean field conc J ⟨n_z⟩ S_z; S_z weight conc J² (⟨n_z²⟩ − conc ⟨n_z⟩²).
        let mf = d.mean_field[(0, 0)].re / (conc * j * s[2][(0, 0)].re);
        let wz = d.vertices[0].0 / (conc * j * j);
        assert!((mf - mz).abs() < 0.01, "{moments:?}: ⟨n_z⟩ {mf} vs {mz}");
        assert!((wz - (nz2 - conc * mz * mz)).abs() < 0.01, "{moments:?}: S_z weight");
    }
}

/// The averaged density of states holds the four orbitals of a cell (Lorentzian tails aside).
#[test]
fn sum_rule() {
    let model = bhz(BhzParams::default());
    let disorder =
        Disorder::magnetic(0.2, 0.8, Moments::Polarized { mz: 1.0 }, &bhz_spin(GFactors::Opposite))
            .and(Disorder::anderson(0.5, 4));
    let scba = Scba::from_model(&model, &[16, 16], disorder);
    let eta = 0.1;
    let de = 0.05;
    let energies: Vec<f64> = (0..=320).map(|k| -8.0 + de * k as f64).collect();
    let solutions = scba.sweep(&energies, eta);
    assert!(solutions.iter().all(|s| s.residual < 1e-10));
    let dos: Vec<f64> = energies.iter().zip(&solutions).map(|(&e, s)| scba.dos(e, eta, &s.sigma)).collect();
    let integral: f64 = dos.windows(2).map(|w| 0.5 * (w[0] + w[1]) * de).sum();
    // Tails of a Lorentzian beyond ±8 for bands within ±3: about (2/π) η/5 per orbital.
    assert!((integral - 4.0).abs() < 0.08, "∫ρ = {integral}");
    assert!(dos.iter().all(|d| *d >= 0.0));
}
