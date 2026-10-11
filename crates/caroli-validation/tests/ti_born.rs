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

/// The corrected reduced SCBA (`ScbaAnalytic::point_corrected`) is the general matrix SCBA of
/// `caroli_born` with the second-cumulant vertices (same-g and opposite-g, M_z < 1 so that the
/// spin-flip vertices matter).
#[test]
fn corrected_reduction_is_the_matrix_scba() {
    use caroli_born::{Scba, magnetic};
    use caroli_model::presets::{GFactors, bhz_spin};
    use caroli_model::{CMat, c64};
    use caroli_validation::thesis::bhz_hk;
    let (conc, mz) = (0.2, 0.8);
    let hk = bhz_hk(BhzParams::default(), 64, 16);
    for anti in [0, 1] {
        let reduced = ScbaAnalytic {
            m: 64,
            n: 16,
            conc,
            mz,
            anti,
            points: 2,
            j_range: (0.0, 1.0),
            ef: (-0.1, 0.1),
            eta: 1e-2,
        };
        let s = bhz_spin(if anti == 1 { GFactors::Opposite } else { GFactors::Same });
        for (j, e) in [(0.3, 0.05), (0.9, -0.1)] {
            let r = reduced.point_corrected(j, e);
            let (mf, vertices) = magnetic(conc, j, mz, mz * mz + (1.0 - mz).powi(2) / 3.0, &s);
            let scba = Scba { hk: hk.clone(), mean_field: mf.clone(), vertices };
            let sol = scba.solve_anderson(c64::new(e, 1e-2), &CMat::zeros(4, 4), 8, 1e-12, 400);
            let t = &mf + &sol.sigma;
            let block = |b: usize| {
                let (a, d) = (t[(b, b)], t[(b + 1, b + 1)]);
                [-0.137 + ((a - d) * 0.5).re, -((a + d) * 0.5).re]
            };
            let m = [block(0)[0], block(0)[1], block(2)[0], block(2)[1]];
            let d = r.iter().zip(&m).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
            println!("anti {anti}, J {j}, E {e}: |reduced − matrix| = {d:.1e}");
            assert!(d < 1e-5, "anti {anti}, J {j}, E {e}: {r:?} vs {m:?}");
        }
    }
}
