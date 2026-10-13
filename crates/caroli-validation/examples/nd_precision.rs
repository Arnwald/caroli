//! Single precision for the nested-dissection fronts (`Precision::Single`): the loss against
//! double precision on the spin blocks of the g2g1 maps, and the time it saves.
//!
//! The systems are those of `rgf_precision`: disordered length 256, width W, same-g n_I = 0.2
//! and opposite-g n_I = 0.4, M_z = 1, each spin block at every (J_I, E_F) of a coarse grid over
//! the phase diagram. The lead self-energies are computed once per point, in double precision,
//! and serve both precisions.
//!
//! `cargo run --release -p caroli-validation --example nd_precision -- [W ...]` (default 64 128).
//! Writes `target/born_lines/nd_precision.csv`.

use caroli_disorder::rng::Xoshiro256;
use caroli_disorder::{MagneticImpurities, Moments};
use caroli_green::{NestedTwoTerminal, Precision};
use caroli_lattice::Region;
use caroli_model::presets::GFactors;
use caroli_validation::thesis::{bhz_spin_block, impurity_region};
use rayon::prelude::*;
use std::io::Write;
use std::time::Instant;

const L: usize = 256;
const ENERGIES: [f64; 6] = [-0.14, -0.06, 0.0, 0.02, 0.06, 0.14];

/// (name, coupling, conc, disorder strengths, fewer strengths for the largest widths).
type Case = (&'static str, GFactors, f64, &'static [f64], &'static [f64]);

const CASES: [Case; 2] = [
    ("F", GFactors::Same, 0.2, &[0.0, 0.3, 0.6, 0.9, 1.2, 1.5], &[0.3, 0.6, 1.2]),
    ("AF", GFactors::Opposite, 0.4, &[0.0, 0.2, 0.3, 0.35, 0.4, 0.45, 0.6], &[0.35, 0.45, 0.6]),
];

struct Row {
    case: &'static str,
    w: usize,
    j: f64,
    config: i32,
    spin: usize,
    e: f64,
    t64: f64,
    t32: f64,
}

/// One spin block of the g2g1 ribbon of width `w`.
fn system(coupling: GFactors, conc: f64, w: usize, j: f64, config: i32, spin: usize) -> NestedTwoTerminal {
    let imp = MagneticImpurities::generate(
        &mut Xoshiro256::gfortran(1000 + config),
        &impurity_region(1, L, w),
        (conc * (L * w) as f64) as usize,
        Moments::Cone { theta0: 0.0 },
    );
    let (device, clean) = bhz_spin_block(spin, &imp, j, coupling);
    NestedTwoTerminal::from_model(&device, &clean, &Region::cuboid(clean.lattice(), [L + 2, w, 0]), 0)
}

fn main() {
    let widths: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().unwrap()).collect();
    let widths = if widths.is_empty() { vec![64, 128] } else { widths };
    let mut jobs = Vec::new();
    for &w in &widths {
        let configs = if w <= 64 { 2 } else { 1 };
        for &(case, coupling, conc, js, few) in &CASES {
            for &j in if w >= 256 { few } else { js } {
                for config in 0..configs {
                    for spin in 0..2 {
                        jobs.push((case, coupling, conc, w, j, config, spin));
                    }
                }
            }
        }
    }
    let t0 = Instant::now();
    let rows: Vec<Row> = jobs
        .par_iter()
        .flat_map_iter(|&(case, coupling, conc, w, j, config, spin)| {
            let sys = system(coupling, conc, w, j, config, spin);
            ENERGIES
                .iter()
                .map(|&e| {
                    let sigma = sys.self_energies(e).unwrap();
                    let t64 = sys.transmission_with(&sigma).unwrap();
                    let t32 = sys.transmission_with_precision(&sigma, Precision::Single).unwrap();
                    Row { case, w, j, config, spin, e, t64, t32 }
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let dir = caroli_validation::repo_root().join("target/born_lines");
    std::fs::create_dir_all(&dir).expect("output directory");
    let mut f = std::fs::File::create(dir.join("nd_precision.csv")).expect("csv");
    writeln!(f, "case,W,J_I,E_F,config,spin,T64,T32").unwrap();
    for r in &rows {
        writeln!(f, "{},{},{},{},{},{},{},{}", r.case, r.w, r.j, r.e, r.config, r.spin, r.t64, r.t32)
            .unwrap();
    }
    println!("{} block transmissions in {:.0?}", rows.len(), t0.elapsed());
    for &w in &widths {
        let sel: Vec<&Row> = rows.iter().filter(|r| r.w == w).collect();
        let stats = |keep: &dyn Fn(&Row) -> bool| {
            let mut d: Vec<f64> = sel.iter().filter(|r| keep(r)).map(|r| (r.t32 - r.t64).abs()).collect();
            d.sort_by(f64::total_cmp);
            if d.is_empty() {
                return "—".to_string();
            }
            let q = |p: f64| d[((d.len() - 1) as f64 * p).round() as usize];
            let above = d.iter().filter(|&&x| x > 1e-3).count();
            format!(
                "median {:.1e}, 99 % {:.1e}, max {:.1e} ({} points, {} above 1e-3)",
                q(0.5),
                q(0.99),
                d[d.len() - 1],
                d.len(),
                above
            )
        };
        let quantized = |r: &Row| (r.t64 - r.t64.round()).abs() < 1e-3;
        println!("\nW = {w}: |T_single − T_double| per spin block");
        println!("  all points:                       {}", stats(&|_| true));
        println!("  quantized (|T − round T| < 1e-3): {}", stats(&quantized));
        println!("  not quantized:                    {}", stats(&|r| !quantized(r)));
    }

    // Time per energy on the whole thread pool, one system per case and width.
    println!("\nTime per energy (all threads, best of two runs over the {} energies):", ENERGIES.len());
    for &w in &widths {
        for &(case, coupling, conc, js, few) in &CASES {
            let js = if w >= 256 { few } else { js };
            let j = js[js.len() / 2];
            let sys = system(coupling, conc, w, j, 0, 0);
            let sigmas: Vec<_> = ENERGIES.iter().map(|&e| sys.self_energies(e).unwrap()).collect();
            let time = |p: Precision| {
                (0..2)
                    .map(|_| {
                        let t = Instant::now();
                        for s in &sigmas {
                            sys.transmission_with_precision(s, p).unwrap();
                        }
                        t.elapsed().as_secs_f64() / sigmas.len() as f64
                    })
                    .fold(f64::INFINITY, f64::min)
            };
            let (double, single) = (time(Precision::Double), time(Precision::Single));
            println!(
                "  W = {w:3}, {case:2}, J_I = {j}: double {:.1} ms, single {:.1} ms, {:.2}× faster",
                double * 1e3,
                single * 1e3,
                double / single
            );
        }
    }
}
