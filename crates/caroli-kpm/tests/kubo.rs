//! Kubo–Bastin conductivity of the Qi–Wu–Zhang Chern insulator on a torus:
//! `H(k) = sin k_x σx + sin k_y σy + (m + cos k_x + cos k_y) σz`, Chern number ±1 for
//! 0 < |m| < 2. In the gap `σ_xy = ±e²/h` (opposite for ±m) and `σ_xx = 0`.

use caroli_disorder::rng::Xoshiro256;
use caroli_kpm::kubo::{kubo_bastin, kubo_moments};
use caroli_kpm::{jackson, random_phases};
use caroli_lattice::{Lattice, Region};
use caroli_model::pauli::{sx, sy, sz};
use caroli_model::{Model, c64, combine};

fn qwz(m: f64) -> Model {
    let i = c64::new(0.0, 1.0);
    let half = c64::new(0.5, 0.0);
    Model::new(Lattice::square(1.0, 2))
        .onsite(0, combine(&[(m, &sz())]))
        .hopping(0, 0, [1, 0, 0], combine(&[(half, &sz()), (-i * 0.5, &sx())]))
        .hopping(0, 0, [0, 1, 0], combine(&[(half, &sz()), (-i * 0.5, &sy())]))
}

/// σ_xx and σ_xy (e²/h) at the chemical potentials `mus`.
fn conductivities(m: f64, mus: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let l = 32;
    let model = qwz(m);
    let region = Region::cuboid(model.lattice(), [l, l, 0]);
    let periods = [[l as i64, 0, 0], [0, l as i64, 0]];
    let h = model.sparse_bloch_hamiltonian(&region, &periods, &[0.0, 0.0]);
    let vx = model.sparse_velocity(&region, &periods, &[0.0, 0.0], [1.0, 0.0, 0.0]);
    let vy = model.sparse_velocity(&region, &periods, &[0.0, 0.0], [0.0, 1.0, 0.0]);
    let (center, half_width) = (0.0, 3.6);
    let scaled = |x: &[c64], y: &mut [c64]| {
        h.apply(x, y);
        y.iter_mut().for_each(|v| *v = (*v - center) / half_width);
    };
    let mut rng = Xoshiro256::seed_from_u64(11);
    let starts: Vec<Vec<c64>> = (0..6).map(|_| random_phases(&mut rng, h.n)).collect();
    let kernel = jackson(256);
    let omega = (l * l) as f64;
    let sigma = |va: &caroli_model::SparseMatrix, vb: &caroli_model::SparseMatrix| {
        let mu = kubo_moments(scaled, |x, y| va.apply(x, y), |x, y| vb.apply(x, y), &starts, &kernel);
        kubo_bastin(&mu, center, half_width, omega, mus, 1024)
    };
    (sigma(&vx, &vx), sigma(&vx, &vy))
}

#[test]
fn chern_insulator() {
    let mus = [-0.3, 0.0, 0.4];
    let (xx_plus, xy_plus) = conductivities(1.0, &mus);
    let (_, xy_minus) = conductivities(-1.0, &mus);
    println!("σ_xx(m = 1) = {xx_plus:?}, σ_xy(m = 1) = {xy_plus:?}, σ_xy(m = −1) = {xy_minus:?}");
    for k in 0..mus.len() {
        assert!((xy_plus[k].abs() - 1.0).abs() < 0.01, "|σ_xy| = {}", xy_plus[k]);
        assert!((xy_plus[k] + xy_minus[k]).abs() < 1e-3, "opposite Chern numbers");
        assert!(xx_plus[k].abs() < 1e-4, "σ_xx = {} in the gap", xx_plus[k]);
    }
}
