//! The high-level [`Spectrum`] against exact diagonalisation: moments of the exact trace,
//! stochastic estimates, projections and local densities that add up to the total, and the
//! quantised Hall conductivity of a Chern insulator.

use caroli_kpm::{Spectrum, jackson};
use caroli_lattice::{Lattice, Region};
use caroli_model::pauli::{s0, sx, sy, sz};
use caroli_model::presets::{BhzParams, bhz};
use caroli_model::{CMat, Model, c64, combine};

/// `T_n(x)` for `n = 0..m`.
fn chebyshev(x: f64, m: usize) -> Vec<f64> {
    let mut t = vec![1.0, x];
    while t.len() < m {
        let k = t.len();
        t.push(2.0 * x * t[k - 1] - t[k - 2]);
    }
    t.truncate(m);
    t
}

/// The exact trace gives the moments `g_n Σ_i T_n(x_i)` of the eigenvalues.
#[test]
fn exact_trace_matches_eigenvalues() {
    let model = bhz(BhzParams { lambda: 0.1, ..BhzParams::default() });
    let region = Region::cuboid(model.lattice(), [6, 5, 0]);
    let spectrum = Spectrum::new(&model, &region).exact_trace().moments(48);
    let (mu, _) = spectrum.dos_moments();
    let (centre, half) = spectrum.scaling();
    let eig = model.hamiltonian(&region).self_adjoint_eigenvalues(faer::Side::Lower).unwrap();
    let g = jackson(48);
    let mut exact = vec![0.0; 48];
    for e in &eig {
        for (n, t) in chebyshev((e - centre) / half, 48).iter().enumerate() {
            exact[n] += g[n] * t;
        }
    }
    let err = mu.iter().zip(&exact).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max);
    assert!(err < 1e-10 * eig.len() as f64, "{err:e}");
}

/// Random phases: `μ_0` is exactly the number of orbitals, and the other moments agree with the
/// exact trace within the statistical error ~ 1/√(vectors · N).
#[test]
fn stochastic_trace() {
    let model =
        Model::new(Lattice::square(1.0, 1)).hopping(0, 0, [1, 0, 0], -1.0).hopping(0, 0, [0, 1, 0], -1.0);
    let region = Region::cuboid(model.lattice(), [24, 24, 0]);
    let periods = [[24, 0, 0], [0, 24, 0]];
    let n = region.orbitals() as f64;
    let exact = Spectrum::periodic(&model, &region, &periods).moments(32).exact_trace().dos_moments().0;
    let stochastic =
        Spectrum::periodic(&model, &region, &periods).moments(32).vectors(20).seed(5).dos_moments().0;
    assert!((stochastic[0].re - n).abs() < 1e-9 * n);
    let err = stochastic.iter().zip(&exact).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max) / n;
    assert!(err < 0.05, "relative error {err:e}");
    // The density integrates to the number of orbitals (trapezoids on a fine grid).
    let spectrum = Spectrum::periodic(&model, &region, &periods).moments(32).vectors(4);
    let energies: Vec<f64> = (0..=4000).map(|k| -4.5 + 9.0 * k as f64 / 4000.0).collect();
    let rho = spectrum.dos(&energies);
    let integral: f64 = rho.windows(2).map(|w| 0.5 * (w[0] + w[1]) * 9.0 / 4000.0).sum();
    assert!((integral - n).abs() < 2e-3 * n, "∫ρ = {integral}, N = {n}");
}

/// Spin-projected densities add up to the total, and so do the local densities of all the sites
/// (exact traces, BHZ with spin mixing).
#[test]
fn projections_and_local_densities_add_up() {
    let model = bhz(BhzParams { lambda: 0.2, ..BhzParams::default() });
    let region = Region::cuboid(model.lattice(), [5, 4, 0]);
    let spectrum = Spectrum::new(&model, &region).exact_trace().moments(40);
    let energies: Vec<f64> = (0..41).map(|k| -1.5 + 0.075 * k as f64).collect();
    let total = spectrum.dos(&energies);
    let half = CMat::from_fn(2, 2, |i, j| if i == j { c64::new(1.0, 0.0) } else { c64::new(0.0, 0.0) });
    let up = caroli_model::pauli::kron(&combine(&[(0.5, &s0()), (0.5, &sz())]), &half);
    let down = caroli_model::pauli::kron(&combine(&[(0.5, &s0()), (-0.5, &sz())]), &half);
    let (a, b) = (spectrum.projected_dos(&up, &energies), spectrum.projected_dos(&down, &energies));
    let sites: Vec<usize> = (0..region.len()).collect();
    let local = spectrum.ldos(&sites, &energies);
    let scale = total.iter().fold(0.0f64, |m, x| m.max(*x));
    for (k, t) in total.iter().enumerate() {
        assert!((a[k] + b[k] - t).abs() < 1e-9 * scale, "spin projections at {}", energies[k]);
        let sum: f64 = local.iter().map(|l| l[k]).sum();
        assert!((sum - t).abs() < 1e-9 * scale, "local densities at {}", energies[k]);
    }
    // Kramers pairs: the two spins share every energy. The E1 orbitals do not.
    assert!(a.iter().zip(&b).all(|(u, d)| (u - d).abs() < 1e-9 * scale));
    let e1 = caroli_model::pauli::kron(&s0(), &combine(&[(0.5, &s0()), (0.5, &sz())]));
    let share: Vec<f64> =
        spectrum.projected_dos(&e1, &energies).iter().zip(&total).map(|(e, t)| e / t).collect();
    assert!(share.iter().zip(&total).any(|(s, t)| *t > 0.1 * scale && (s - 0.5).abs() > 0.1), "{share:?}");
}

/// Qi–Wu–Zhang Chern insulator: `σ_xy = ±e²/h` and `σ_xx = 0` in the gap.
#[test]
fn chern_insulator() {
    let i = c64::new(0.0, 1.0);
    let qwz = |m: f64| {
        Model::new(Lattice::square(1.0, 2))
            .onsite(0, combine(&[(m, &sz())]))
            .hopping(0, 0, [1, 0, 0], combine(&[(c64::new(0.5, 0.0), &sz()), (-i * 0.5, &sx())]))
            .hopping(0, 0, [0, 1, 0], combine(&[(c64::new(0.5, 0.0), &sz()), (-i * 0.5, &sy())]))
    };
    let (model, flipped) = (qwz(1.0), qwz(-1.0));
    let region = Region::cuboid(model.lattice(), [32, 32, 0]);
    let periods = [[32, 0, 0], [0, 32, 0]];
    let mus = [-0.3, 0.2];
    let (x, y) = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let spectrum = Spectrum::periodic(&model, &region, &periods).moments(192).vectors(8).bounds(-3.0, 3.0);
    let xy = spectrum.conductivity(x, y, &mus, 0.0);
    let xx = spectrum.conductivity(x, x, &mus, 0.0);
    let xy_flipped = Spectrum::periodic(&flipped, &region, &periods)
        .moments(192)
        .vectors(8)
        .bounds(-3.0, 3.0)
        .conductivity(x, y, &mus, 0.0);
    println!("σ_xy = {xy:?}, σ_xx = {xx:?}, σ_xy(−m) = {xy_flipped:?}");
    for k in 0..mus.len() {
        // Statistical error of 8 random vectors on 2048 orbitals: about 1 %.
        assert!(
            (xy[k] + 1.0).abs() < 0.03 && (xy_flipped[k] - 1.0).abs() < 0.03,
            "σ_xy = {}, {}",
            xy[k],
            xy_flipped[k]
        );
        assert!(xx[k].abs() < 0.01, "σ_xx = {}", xx[k]);
    }
}
