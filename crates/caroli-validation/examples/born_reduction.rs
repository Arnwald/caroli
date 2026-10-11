//! Compares the thesis reduced SCBA (scba-magnetic-analytic) with the general matrix SCBA using
//! the same weights (v_z on S_z, v_perp/2 on S_x and S_y) and the same mean field.
use caroli_born::Scba;
use caroli_model::presets::{BhzParams, GFactors, bhz_spin};
use caroli_model::{CMat, c64};
use caroli_validation::thesis::{ScbaAnalytic, bhz_hk};

/// Reduced SCBA with a chosen sign of the mass self-energy (thesis: −1), iterated to 1e-12.
#[allow(clippy::too_many_arguments)] // mirrors the thesis variables
fn reduced(
    m: usize,
    n: usize,
    conc: f64,
    mz: f64,
    anti: f64,
    j: f64,
    e: f64,
    eta: f64,
    sign: f64,
) -> (f64, f64) {
    let (mass, b, d) = (-0.137, -0.376406, -0.280933);
    let grid = caroli_born::k_grid(m, n);
    let vz = (1.0 - conc * conc) * mz * mz + (1.0 - mz) * (1.0 - mz) / 3.0;
    let vperp = (2.0f32 / 3.0f32) as f64 * (1.0 - mz) * (1.0 + 2.0 * mz);
    let norm = conc * j * j / (n * m) as f64;
    let z = c64::new(e, eta);
    let mu0 = c64::new(mass + anti * j * conc * mz, 0.0);
    let md0 = c64::new(mass - anti * j * conc * mz, 0.0);
    let muu0 = c64::new(-(1.0 - anti) * j * conc * mz, 0.0);
    let mud0 = c64::new((1.0 - anti) * j * conc * mz, 0.0);
    let (mut mu, mut md, mut muu, mut mud) = (mu0, md0, muu0, mud0);
    for _ in 0..20000 {
        let (mut dmu, mut dmd, mut dmuu, mut dmud) =
            (c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0));
        for k in &grid {
            let c = 4.0 - 2.0 * k[0].cos() - 2.0 * k[1].cos();
            let s = k[0].sin().powi(2) + k[1].sin().powi(2);
            let (eu, ed) = (z + muu + d * c, z + mud + d * c);
            let (mk, dk) = (mu - b * c, md - b * c);
            let (detu, detd) = (eu * eu - mk * mk - s, ed * ed - dk * dk - s);
            dmu += vz * mk / detu + vperp * dk / detd;
            dmd += vz * dk / detd + vperp * mk / detu;
            dmuu += vz * eu / detu + vperp * ed / detd;
            dmud += vz * ed / detd + vperp * eu / detu;
        }
        let nmu = mu0 + dmu * (sign * norm);
        let nmd = md0 + dmd * (sign * norm);
        let nmuu = muu0 - dmuu * norm;
        let nmud = mud0 - dmud * norm;
        let ch = (nmu - mu).norm() + (nmd - md).norm() + (nmuu - muu).norm() + (nmud - mud).norm();
        let p = 0.3;
        mu = nmu * p + mu * (1.0 - p);
        md = nmd * p + md * (1.0 - p);
        muu = nmuu * p + muu * (1.0 - p);
        mud = nmud * p + mud * (1.0 - p);
        if ch < 1e-13 {
            break;
        }
    }
    (mu.re, muu.re)
}

fn main() {
    let (m, n, conc, mz, eta) = (64, 16, 0.4, 0.8, 0.001);
    let params = BhzParams { mass: -0.137, b: -0.376406, d: -0.280933, c: 0.0, a: 1.0, lambda: 0.0 };
    let hk = bhz_hk(params, m, n);
    let vz = (1.0 - conc * conc) * mz * mz + (1.0 - mz) * (1.0 - mz) / 3.0;
    let vperp = (2.0f32 / 3.0f32) as f64 * (1.0 - mz) * (1.0 + 2.0 * mz);
    for (anti, coupling) in [(0, GFactors::Same), (1, GFactors::Opposite)] {
        let analytic =
            ScbaAnalytic { m, n, conc, mz, anti, points: 6, j_range: (0.0, 0.75), ef: (-0.16, 0.16), eta }
                .run();
        let s = bhz_spin(coupling);
        println!("anti = {anti}: (J, E) | M_up analytic vs matrix | mu_up analytic vs matrix");
        for (iu, ie) in [(2, 1), (4, 3), (5, 5)] {
            let j = iu as f64 * 0.75 / 5.0;
            let e = ie as f64 * 0.32 / 5.0 - 0.16;
            let w = conc * j * j;
            let mf = CMat::from_fn(4, 4, |a, b| c64::new(conc * j * mz, 0.0) * s[2][(a, b)]);
            let scba = Scba {
                hk: hk.clone(),
                mean_field: mf.clone(),
                vertices: vec![
                    (w * vz, s[2].clone()),
                    (w * vperp / 2.0, s[0].clone()),
                    (w * vperp / 2.0, s[1].clone()),
                ],
            };
            let sol = scba.solve(c64::new(e, eta), &CMat::zeros(4, 4), 0.3, 1e-12, 5000);
            let tot = &sol.sigma + &mf;
            let m_up = params.mass + ((tot[(0, 0)] - tot[(1, 1)]) / 2.0).re;
            let mu_up = -((tot[(0, 0)] + tot[(1, 1)]) / 2.0).re;
            let k = iu + 6 * ie;
            let (mt, ut) = reduced(m, n, conc, mz, anti as f64, j, e, eta, -1.0);
            let (mp, up) = reduced(m, n, conc, mz, anti as f64, j, e, eta, 1.0);
            println!(
                "  ({j:.2}, {e:+.3}) | {:+.6} vs {m_up:+.6} | {:+.6} vs {mu_up:+.6}   ({} it)",
                analytic[0][k], analytic[1][k], sol.iterations
            );
            println!(
                "      reduced, thesis sign (converged): M {mt:+.6}, mu {ut:+.6};  flipped mass sign: M {mp:+.6}, mu {up:+.6}"
            );
        }
    }
}
