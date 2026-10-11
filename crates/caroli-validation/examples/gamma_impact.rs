//! Impact of the thesis' Re(Γ) truncation on its anomalous-Hall conductance maps
//! (`anomalous/Results/Ferro/ferro_400(256)x64_0.2_1.0`, the g2g1 paper setup).
//!
//! At a few points of the thesis (J_I, E_F) grid, prints the disorder-averaged conductance of
//! the 50 thesis realisations, and caroli's averages over its own realisations with the thesis
//! formula Tr[Re Γ_L G Re Γ_R G†] and with the exact Caroli formula (same realisations, so their
//! difference is paired).
//!
//! `cargo run --release -p caroli-validation --example gamma_impact -- [realisations] [threads]`

use caroli_disorder::rng::Xoshiro256;
use caroli_disorder::{MagneticImpurities, Moments};
use caroli_green::{SelfEnergyMethod, TwoTerminal};
use caroli_lattice::Region;
use caroli_model::presets::{BhzParams, GFactors, bhz, bhz_spin};
use caroli_validation::read_records;
use caroli_validation::thesis::{impurity_region, thesis_transmission};
use std::sync::Mutex;

const NX: usize = 400;
const NY: usize = 64;
const NX_IMP: usize = 256;
const CONC: f64 = 0.2;
const J_INDICES: [usize; 7] = [0, 10, 21, 31, 42, 52, 63];
const E_INDICES: [usize; 3] = [16, 31, 47];

fn j_of(k: usize) -> f64 {
    1.5 * k as f64 / 63.0
}

fn e_of(m: usize) -> f64 {
    -0.16 + 0.32 * m as f64 / 63.0
}

/// Mean and standard error.
fn stats(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, (var / n).sqrt())
}

/// cond(Nu = 64 J values, Ne = 64 energies) of every thesis realisation.
fn thesis_maps() -> Vec<Vec<f64>> {
    let home = std::env::var("HOME").unwrap_or_default();
    let dir = format!(
        "{home}/Projects/electronic-transport/anomalous/Results/Ferro/ferro_400(256)x64_0.2_1.0/64Jx64e"
    );
    let mut maps = Vec::new();
    for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
        if let Ok(r) = read_records(entry.path().join("cond_00.dat")) {
            maps.push(caroli_validation::as_f64(&r[0]));
        }
    }
    maps
}

fn main() {
    let args: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let realisations = args.first().copied().unwrap_or(8);
    let threads = args.get(1).copied().unwrap_or(6);

    let clean = bhz(BhzParams::default());
    // One clean lead slice at each end (the thesis green_recur); sample slices x = 1..=NX.
    let region = Region::cuboid(clean.lattice(), [NX + 2, NY, 0]);
    let candidates = impurity_region(1 + ((NX - NX_IMP) / 2) as i64, NX_IMP, NY);
    let n_imp = (CONC * NX_IMP as f64 * NY as f64) as usize;
    let spin = bhz_spin(GFactors::Same);

    // results[(j, e)] = (thesis formula, exact) per realisation
    let results: Mutex<Vec<(usize, usize, f64, f64)>> = Mutex::new(Vec::new());
    let tasks: Vec<(usize, usize)> =
        (0..realisations).flat_map(|r| J_INDICES.iter().map(move |&k| (r, k))).collect();
    let next = Mutex::new(0usize);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let task = {
                        let mut n = next.lock().unwrap();
                        if *n >= tasks.len() {
                            break;
                        }
                        *n += 1;
                        tasks[*n - 1]
                    };
                    let (r, k) = task;
                    let mut rng = Xoshiro256::seed_from_u64(1000 + r as u64);
                    let imp = MagneticImpurities::generate(
                        &mut rng,
                        &candidates,
                        n_imp,
                        Moments::Polarized { mz: 1.0 },
                    );
                    let device = bhz(BhzParams::default()).with_onsite_term(imp.exchange(j_of(k), &spin));
                    let sys = TwoTerminal::from_model(&device, &clean, &region, 0);
                    for &m in &E_INDICES {
                        let sol = sys.solve(e_of(m), SelfEnergyMethod::Modes).expect("transport");
                        results.lock().unwrap().push((k, m, thesis_transmission(&sol), sol.transmission()));
                    }
                    eprintln!("done: realisation {r}, J_I = {:.3}", j_of(k));
                }
            });
        }
    });

    let maps = thesis_maps();
    let results = results.into_inner().unwrap();
    println!(
        "{} thesis realisations, {realisations} caroli realisations (400×64, Nx_imp 256, conc 0.2, Mz 1, same-g)",
        maps.len()
    );
    println!(
        "{:>6} {:>7} | {:>15} | {:>15} {:>15} | {:>16}",
        "J_I", "E_F", "thesis data", "caroli Re Γ", "caroli exact", "exact − Re Γ"
    );
    for &m in &E_INDICES {
        for &k in &J_INDICES {
            let data: Vec<f64> = maps.iter().map(|c| c[k + 64 * m]).collect();
            let pts: Vec<&(usize, usize, f64, f64)> =
                results.iter().filter(|p| p.0 == k && p.1 == m).collect();
            let thesis: Vec<f64> = pts.iter().map(|p| p.2).collect();
            let exact: Vec<f64> = pts.iter().map(|p| p.3).collect();
            let diff: Vec<f64> = pts.iter().map(|p| p.3 - p.2).collect();
            let (d, de) = if data.is_empty() { (f64::NAN, f64::NAN) } else { stats(&data) };
            let ((t, te), (x, xe), (dd, dde)) = (stats(&thesis), stats(&exact), stats(&diff));
            println!(
                "{:6.3} {:7.4} | {d:7.3} ± {de:5.3} | {t:7.3} ± {te:5.3} {x:7.3} ± {xe:5.3} | {dd:+7.3} ± {dde:5.3}",
                j_of(k),
                e_of(m)
            );
        }
    }
}
