//! Spin-resolved conductance maps of the g2g1 systems on ribbons of several widths.
//!
//! With M_z = 1, the exchange conserves s_z. Each spin block is then a two-orbital BHZ model
//! with its own disorder, and T = T↑ + T↓. Building the blocks separately halves the orbitals
//! per slice and quarters the memory of the dense slices.
//!
//! The disordered region has L = 256 slices of width W, with int(n_I·L·W) impurities (M_z = 1),
//! as in the thesis maps. It sits between semi-infinite clean leads.
//!
//! `cargo run --release -p caroli-validation --example born_wide_maps -- <F|AF> <W> <configs> <map|E_F> [J points]`
//!
//! `born_wide_maps check` compares the two-orbital blocks with the four-orbital model split by
//! [`TwoTerminal::blocks`] on a small disordered ribbon.
//!
//! Writes `target/born_lines/wide_<case>_W<W>_<map|E>.csv` with columns
//! J_I, E_F, config, T_up, T_dn. J_I runs over [0, 1.5] for 'F' and [0, 0.75] for 'AF'. `map`
//! takes the 17 energies of [−0.16, 0.16].

use caroli_disorder::rng::Xoshiro256;
use caroli_disorder::{MagneticImpurities, Moments};
use caroli_green::TwoTerminal;
use caroli_lattice::Region;
use caroli_model::presets::{BhzParams, GFactors, bhz, bhz_spin};
use caroli_validation::thesis::{bhz_spin_block, impurity_region};
use rayon::prelude::*;
use std::io::Write;

const L: usize = 256;

/// `(T↑, T↓)` at each energy for one configuration.
fn transmissions(
    coupling: GFactors,
    conc: f64,
    w: usize,
    j: f64,
    config: i32,
    energies: &[f64],
) -> Vec<[f64; 2]> {
    let n_imp = (conc * (L * w) as f64) as usize;
    let imp = MagneticImpurities::generate(
        &mut Xoshiro256::gfortran(1000 + config),
        &impurity_region(1, L, w),
        n_imp,
        Moments::Cone { theta0: 0.0 },
    );
    let mut out = vec![[0.0; 2]; energies.len()];
    for s in 0..2 {
        let (device, clean) = bhz_spin_block(s, &imp, j, coupling);
        let sys =
            TwoTerminal::from_model(&device, &clean, &Region::cuboid(clean.lattice(), [L + 2, w, 0]), 0);
        for (o, &e) in out.iter_mut().zip(energies) {
            o[s] = sys.transmission(e).unwrap();
        }
    }
    out
}

/// The two-orbital blocks against the four-orbital model on a 40 × 16 disordered ribbon.
fn check() {
    let (w, conc) = (16usize, 0.3);
    let imp = MagneticImpurities::generate(
        &mut Xoshiro256::gfortran(7),
        &impurity_region(1, L, w),
        (conc * (L * w) as f64) as usize,
        Moments::Cone { theta0: 0.0 },
    );
    for coupling in [GFactors::Same, GFactors::Opposite] {
        let device = bhz(BhzParams::default()).with_onsite_term(imp.exchange(0.6, &bhz_spin(coupling)));
        let clean = bhz(BhzParams::default());
        let full =
            TwoTerminal::from_model(&device, &clean, &Region::cuboid(clean.lattice(), [L + 2, w, 0]), 0);
        let blocks = full.blocks();
        assert_eq!(blocks.len(), 2);
        for e in [-0.1, 0.02, 0.15] {
            for b in &blocks {
                let s = b[0] % 4 / 2;
                let t4 = full.restricted(b).transmission(e).unwrap();
                let (dev2, clean2) = bhz_spin_block(s, &imp, 0.6, coupling);
                let sys2 = TwoTerminal::from_model(
                    &dev2,
                    &clean2,
                    &Region::cuboid(clean2.lattice(), [L + 2, w, 0]),
                    0,
                );
                let t2 = sys2.transmission(e).unwrap();
                println!("{coupling:?}, E {e:+.2}, spin {s}: four-orbital {t4:.12}, two-orbital {t2:.12}");
                assert!((t4 - t2).abs() < 1e-9);
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("check") {
        return check();
    }
    assert!(args.len() >= 5, "usage: born_wide_maps <F|AF> <W> <configs> <map|E_F> [J points]");
    let (case, w, configs) =
        (args[1].as_str(), args[2].parse::<usize>().unwrap(), args[3].parse::<i32>().unwrap());
    let (coupling, conc, jmax) = match case {
        "F" => (GFactors::Same, 0.2, 1.5),
        "AF" => (GFactors::Opposite, 0.4, 0.75),
        _ => panic!("case F or AF"),
    };
    let energies: Vec<f64> = if args[4] == "map" {
        (0..17).map(|k| -0.16 + 0.02 * k as f64).collect()
    } else {
        vec![args[4].parse().unwrap()]
    };
    let nj: usize = args.get(5).map_or(31, |a| a.parse().unwrap());
    let jobs: Vec<(f64, i32)> = (0..nj)
        .flat_map(|k| (0..configs).map(move |c| (jmax * k as f64 / (nj - 1).max(1) as f64, c)))
        .collect();
    let t0 = std::time::Instant::now();
    let results: Vec<Vec<[f64; 2]>> =
        jobs.par_iter().map(|&(j, c)| transmissions(coupling, conc, w, j, c, &energies)).collect();
    let dir = caroli_validation::repo_root().join("target/born_lines");
    std::fs::create_dir_all(&dir).expect("output directory");
    let tag = if args[4] == "map" { "map".to_string() } else { args[4].clone() };
    let path = dir.join(format!("wide_{case}_W{w}_{tag}.csv"));
    let mut f = std::fs::File::create(&path).expect("csv");
    writeln!(f, "J_I,E_F,config,T_up,T_dn").unwrap();
    for (&(j, c), r) in jobs.iter().zip(&results) {
        for (&e, t) in energies.iter().zip(r) {
            writeln!(f, "{j},{e},{c},{},{}", t[0], t[1]).unwrap();
        }
    }
    println!(
        "{} transmissions per spin in {:.1?}: {}",
        jobs.len() * energies.len(),
        t0.elapsed(),
        path.display()
    );
}
