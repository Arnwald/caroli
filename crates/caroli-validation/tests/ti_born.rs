//! Self-consistent Born approximation against `scba-anderson`, `scba-magnetic-matrix` and
//! `scba-magnetic-analytic` (deterministic: no random numbers involved).

use caroli_model::presets::BhzParams;
use caroli_validation::golden_f64;
use caroli_validation::thesis::{ScbaAnalytic, ScbaAnderson, ScbaMatrix};

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn golden(case: &str, file: &str) -> Vec<f64> {
    golden_f64(case, file).remove(0)
}

#[test]
fn anderson() {
    let params = BhzParams { mass: 0.013, ..BhzParams::default() };
    let (mass, mu, us, es) =
        ScbaAnderson { params, m: 16, n: 16, u0: (0.2, 6.0), ef: (-0.3, 1.0), eta: 1.0, points: 4 }.run();
    let axis = golden("ti-scba-anderson", "axis_00.dat");
    assert!(max_diff(&[us, es].concat(), &axis) < 1e-14);
    let dm = max_diff(&mass, &golden("ti-scba-anderson", "mass_00.dat"));
    let du = max_diff(&mu, &golden("ti-scba-anderson", "mu_00.dat"));
    println!("scba-anderson: |Δ mass| ≤ {dm:.1e}, |Δ mu| ≤ {du:.1e}");
    assert!(dm < 1e-10 && du < 1e-10);
}

fn matrix(case: &str, gegh: i32) {
    let out = ScbaMatrix {
        params: BhzParams::default(),
        m: 32,
        n: 16,
        conc: 0.2,
        j_range: (0.0, 3.0),
        ef: (-0.2, 0.2),
        eta: 1.0,
        points: 4,
        gegh,
    }
    .run();
    let files = ["mass_u_00.dat", "mu_u_00.dat", "mass_d_00.dat", "mu_d_00.dat"];
    for (o, f) in out.iter().zip(files) {
        let d = max_diff(o, &golden(case, f));
        println!("{case} {f}: |Δ| ≤ {d:.1e}");
        assert!(d < 1e-10, "{case} {f}: {d:e}");
    }
}

#[test]
fn magnetic_matrix_same_g() {
    matrix("ti-scba-matrix-same-g", 1);
}

#[test]
fn magnetic_matrix_opposite_g() {
    matrix("ti-scba-matrix-opposite-g", -1);
}

fn analytic(case: &str, anti: i32) {
    let out = ScbaAnalytic {
        m: 64,
        n: 16,
        conc: 0.4,
        mz: 0.8,
        anti,
        points: 6,
        j_range: (0.0, 0.75),
        ef: (-0.16, 0.16),
        eta: 0.001,
    }
    .run();
    let files = ["mass_u_00.dat", "mu_u_00.dat", "mass_d_00.dat", "mu_d_00.dat"];
    for (o, f) in out.iter().zip(files) {
        let d = max_diff(o, &golden(case, f));
        println!("{case} {f}: |Δ| ≤ {d:.1e}");
        assert!(d < 1e-10, "{case} {f}: {d:e}");
    }
}

#[test]
fn magnetic_analytic_same_g() {
    analytic("ti-scba-analytic-same-g", 0);
}

#[test]
fn magnetic_analytic_opposite_g() {
    analytic("ti-scba-analytic-opposite-g", 1);
}
