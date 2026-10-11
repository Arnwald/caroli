//! Single precision for the recursive Green's functions: the loss against double precision on
//! the g2g1 systems, to decide whether the Apple GPU (no double precision) can run the
//! transport.
//!
//! As a GPU port would:
//! - the lead self-energies stay in double precision (computed once per energy, outside the
//!   sweep) and are rounded to single precision;
//! - the sweep over the slices runs in complex single precision.
//!
//! The transmission `Tr[Γ_L G_1N Γ_R G_1N†]` is then evaluated in single precision, and, as a
//! variant, in double precision from the single-precision `G_1N`. Two more variants locate the
//! loss:
//! - a double-precision sweep with the self-energies rounded to single precision;
//! - single-precision inversions with the product `G_1n` accumulated in double precision.
//!
//! The last variant is a possible GPU scheme. Each inversion is done in single precision and
//! refined once, `X ← X + X(1 − A X)`, with the residual in double precision (double-single on
//! a GPU) and the correction product in single precision.
//!
//! The systems are the spin blocks of the thesis maps: disordered length 256, width W,
//! same-g n_I = 0.2 and opposite-g n_I = 0.4, M_z = 1. Each block is compared at every (J_I, E_F)
//! of a coarse grid over the phase diagram.
//!
//! `cargo run --release -p caroli-validation --example rgf_precision -- [W ...]`
//! (default 64 128). Writes `target/born_lines/rgf_precision.csv`.

use caroli_disorder::rng::Xoshiro256;
use caroli_disorder::{MagneticImpurities, Moments};
use caroli_green::{EndToEnd, SelfEnergyMethod, TwoTerminal};
use caroli_lattice::Region;
use caroli_model::presets::GFactors;
use caroli_model::{CMat, c64};
use caroli_validation::thesis::{bhz_spin_block, impurity_region};
use faer::linalg::solvers::DenseSolveCore;
use faer::{Mat, c32};
use rayon::prelude::*;
use std::io::Write;

const L: usize = 256;
const ENERGIES: [f64; 6] = [-0.14, -0.06, 0.0, 0.02, 0.06, 0.14];

type C32Mat = Mat<c32>;

/// (name, coupling, conc, disorder strengths, fewer strengths for the largest widths).
type Case = (&'static str, GFactors, f64, &'static [f64], &'static [f64]);

fn to32(m: &CMat) -> C32Mat {
    C32Mat::from_fn(m.nrows(), m.ncols(), |i, j| c32::new(m[(i, j)].re as f32, m[(i, j)].im as f32))
}

fn to64(m: &C32Mat) -> CMat {
    CMat::from_fn(m.nrows(), m.ncols(), |i, j| c64::new(m[(i, j)].re as f64, m[(i, j)].im as f64))
}

/// `(z − H)` in single precision.
fn resolvent32(z: c32, h: &CMat) -> C32Mat {
    C32Mat::from_fn(h.nrows(), h.ncols(), |i, j| {
        let x = c32::new(-(h[(i, j)].re as f32), -(h[(i, j)].im as f32));
        if i == j { x + z } else { x }
    })
}

fn broadening32(s: &C32Mat) -> C32Mat {
    C32Mat::from_fn(s.nrows(), s.ncols(), |i, j| c32::new(0.0, 1.0) * (s[(i, j)] - s[(j, i)].conj()))
}

#[derive(Clone, Copy, PartialEq)]
enum Inverse {
    Double,
    Single,
    /// Single precision, then one refinement step with a double-precision residual.
    Refined,
}

/// Double-precision sweep from the given self-energies, with the inversions done as `mode` says.
fn sweep64(sys: &TwoTerminal, e: f64, sl: &CMat, sr: &CMat, mode: Inverse) -> f64 {
    let inverse = |a: &CMat| -> CMat {
        match mode {
            Inverse::Double => a.partial_piv_lu().inverse(),
            Inverse::Single => to64(&to32(a).partial_piv_lu().inverse()),
            Inverse::Refined => {
                let x = to32(a).partial_piv_lu().inverse();
                let x64 = to64(&x);
                let n = a.nrows();
                let residual = CMat::from_fn(n, n, |i, j| if i == j { c64::new(1.0, 0.0) } else { c64::new(0.0, 0.0) }) - a * &x64;
                x64 + to64(&(&x * to32(&residual)))
            }
        }
    };
    let z = c64::new(e, 0.0);
    let n = sys.slices.len();
    let resolvent = |h: &CMat| CMat::from_fn(h.nrows(), h.ncols(), |i, j| if i == j { z - h[(i, j)] } else { -h[(i, j)] });
    let mut a = resolvent(&sys.slices[0]) - sl;
    if n == 1 {
        a -= sr;
    }
    let mut g = inverse(&a);
    let mut g1n = g.clone();
    for k in 1..n {
        let v = &sys.couplings[k - 1];
        let mut a = resolvent(&sys.slices[k]) - v.adjoint() * &g * v;
        if k == n - 1 {
            a -= sr;
        }
        g = inverse(&a);
        g1n = &g1n * v * &g;
    }
    EndToEnd { sigma_l: sl.clone(), sigma_r: sr.clone(), g1n }.transmission()
}

/// The sweep of `TwoTerminal::solve` in single precision, from double-precision self-energies:
/// transmission in single precision and in double precision from the single `G_1N`.
fn transmission32(sys: &TwoTerminal, e: f64, sol: &EndToEnd) -> (f64, f64) {
    let (sl, sr) = (to32(&sol.sigma_l), to32(&sol.sigma_r));
    let z = c32::new(e as f32, 0.0);
    let n = sys.slices.len();
    let mut a = resolvent32(z, &sys.slices[0]) - &sl;
    if n == 1 {
        a -= &sr;
    }
    let mut g = a.partial_piv_lu().inverse();
    let mut g1n = g.clone();
    for k in 1..n {
        let v = to32(&sys.couplings[k - 1]);
        let mut a = resolvent32(z, &sys.slices[k]) - v.adjoint() * &g * &v;
        if k == n - 1 {
            a -= &sr;
        }
        g = a.partial_piv_lu().inverse();
        g1n = &g1n * &v * &g;
    }
    let t = broadening32(&sl) * &g1n * broadening32(&sr) * g1n.adjoint();
    let single = (0..t.nrows()).map(|i| t[(i, i)].re as f64).sum();
    let mixed = EndToEnd { sigma_l: sol.sigma_l.clone(), sigma_r: sol.sigma_r.clone(), g1n: to64(&g1n) }.transmission();
    (single, mixed)
}

struct Row {
    case: &'static str,
    w: usize,
    j: f64,
    config: i32,
    spin: usize,
    e: f64,
    t64: f64,
    t32: f64,
    mixed: f64,
    sigma32: f64,
    inv32: f64,
    refined: f64,
    /// Seconds in the double, single and refined sweeps.
    times: [f64; 3],
}

fn main() {
    let widths: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().unwrap()).collect();
    let widths = if widths.is_empty() { vec![64, 128] } else { widths };
    let cases: [Case; 2] = [
        ("F", GFactors::Same, 0.2, &[0.0, 0.3, 0.6, 0.9, 1.2, 1.5], &[0.3, 0.6, 1.2]),
        ("AF", GFactors::Opposite, 0.4, &[0.0, 0.2, 0.3, 0.35, 0.4, 0.45, 0.6], &[0.35, 0.45, 0.6]),
    ];
    let mut jobs = Vec::new();
    for &w in &widths {
        let configs = if w <= 64 { 2 } else { 1 };
        for &(case, coupling, conc, js, few) in &cases {
            // Fewer disorder strengths at the largest widths.
            for &j in if w >= 256 { few } else { js } {
                for config in 0..configs {
                    for spin in 0..2 {
                        jobs.push((case, coupling, conc, w, j, config, spin));
                    }
                }
            }
        }
    }
    let t0 = std::time::Instant::now();
    let rows: Vec<Row> = jobs
        .par_iter()
        .flat_map_iter(|&(case, coupling, conc, w, j, config, spin)| {
            let imp = MagneticImpurities::generate(
                &mut Xoshiro256::gfortran(1000 + config),
                &impurity_region(1, L, w),
                (conc * (L * w) as f64) as usize,
                Moments::Cone { theta0: 0.0 },
            );
            let (device, clean) = bhz_spin_block(spin, &imp, j, coupling);
            let sys = TwoTerminal::from_model(&device, &clean, &Region::cuboid(clean.lattice(), [L + 2, w, 0]), 0);
            ENERGIES
                .iter()
                .map(|&e| {
                    let sol = sys.solve(e, SelfEnergyMethod::Modes).unwrap();
                    let (t32, mixed) = transmission32(&sys, e, &sol);
                    let (sl, sr) = (to64(&to32(&sol.sigma_l)), to64(&to32(&sol.sigma_r)));
                    // The diagnostic variants only at the smallest width.
                    let (sigma32, inv32) = if w <= 64 {
                        (sweep64(&sys, e, &sl, &sr, Inverse::Double), sweep64(&sys, e, &sol.sigma_l, &sol.sigma_r, Inverse::Single))
                    } else {
                        (f64::NAN, f64::NAN)
                    };
                    let clock = std::time::Instant::now();
                    let refined = sweep64(&sys, e, &sol.sigma_l, &sol.sigma_r, Inverse::Refined);
                    let t_refined = clock.elapsed().as_secs_f64();
                    let clock = std::time::Instant::now();
                    let _ = sweep64(&sys, e, &sol.sigma_l, &sol.sigma_r, Inverse::Double);
                    let t_double = clock.elapsed().as_secs_f64();
                    let clock = std::time::Instant::now();
                    let _ = transmission32(&sys, e, &sol);
                    let t_single = clock.elapsed().as_secs_f64();
                    let times = [t_double, t_single, t_refined];
                    Row { case, w, j, config, spin, e, t64: sol.transmission(), t32, mixed, sigma32, inv32, refined, times }
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let dir = caroli_validation::repo_root().join("target/born_lines");
    std::fs::create_dir_all(&dir).expect("output directory");
    let mut f = std::fs::File::create(dir.join("rgf_precision.csv")).expect("csv");
    writeln!(f, "case,W,J_I,E_F,config,spin,T64,T32,T32_mixed,T_sigma32,T_inv32,T_refined").unwrap();
    for r in &rows {
        writeln!(
            f,
            "{},{},{},{},{},{},{},{},{},{},{},{}",
            r.case, r.w, r.j, r.e, r.config, r.spin, r.t64, r.t32, r.mixed, r.sigma32, r.inv32, r.refined
        )
        .unwrap();
    }
    println!("{} block transmissions in {:.0?}", rows.len(), t0.elapsed());
    for &w in &widths {
        let sel: Vec<&Row> = rows.iter().filter(|r| r.w == w).collect();
        let stats = |err: &dyn Fn(&Row) -> f64, keep: &dyn Fn(&Row) -> bool| {
            let mut d: Vec<f64> = sel.iter().filter(|r| keep(r)).map(|r| err(r)).collect();
            d.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if d.is_empty() {
                return "—".to_string();
            }
            let q = |p: f64| d[((d.len() - 1) as f64 * p).round() as usize];
            format!("median {:.1e}, 99 % {:.1e}, max {:.1e} ({} points)", q(0.5), q(0.99), d[d.len() - 1], d.len())
        };
        let single = |r: &Row| (r.t32 - r.t64).abs();
        let mixed = |r: &Row| (r.mixed - r.t64).abs();
        let quantized = |r: &Row| (r.t64 - r.t64.round()).abs() < 1e-3;
        let other = |r: &Row| !quantized(r);
        println!("\nW = {w}: |T_single − T_double| per spin block");
        println!("  all points, single trace: {}", stats(&single, &|_| true));
        println!("  all points, double trace: {}", stats(&mixed, &|_| true));
        if w <= 64 {
            println!("  only Σ rounded:           {}", stats(&|r: &Row| (r.sigma32 - r.t64).abs(), &|_| true));
            println!("  only inversions single:   {}", stats(&|r: &Row| (r.inv32 - r.t64).abs(), &|_| true));
        }
        println!("  single + 1 refinement:    {}", stats(&|r: &Row| (r.refined - r.t64).abs(), &|_| true));
        let mean = |k: usize| sel.iter().map(|r| r.times[k]).sum::<f64>() / sel.len() as f64;
        println!(
            "  CPU time per sweep: double {:.2} s, single {:.2} s, refined (emulated) {:.2} s",
            mean(0),
            mean(1),
            mean(2)
        );
        println!("  quantized (|T − round T| < 1e-3): {}", stats(&single, &quantized));
        println!("  not quantized:                    {}", stats(&single, &other));
    }
}
