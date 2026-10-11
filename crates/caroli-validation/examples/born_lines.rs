//! Impact of the reduced-SCBA bugs (sign of the mass self-energy, `1 − conc²` weight) on the Born
//! phase diagrams of the g2g1 paper (Fig. 3, runs 00–07 of `anomalous/Born Approx/run`).
//!
//! For each panel:
//! 1. checks that the thesis replica (`ScbaAnalytic`) gives the published data;
//! 2. solves the corrected SCBA (general matrix SCBA with the second-cumulant vertices,
//!    `caroli_born`) on the same grid;
//! 3. classifies each `(J_I, E_F)` by the conductance the renormalised bands predict (three
//!    readings, see [`RULES`]);
//! 4. where the thesis has exact numerical conductance maps (disorder-averaged transmission of
//!    400×64 ribbons), compares both Born predictions with them under each reading.
//!
//! The comparison with the exact maps depends on the reading: it does not single out one SCBA
//! as the better description. Only the sign error itself is certain.
//!
//! Writes `target/born_lines/panel_NN.csv` and prints a summary.
//!
//! `cargo run --release -p caroli-validation --example born_lines -- [thesis archive]`
//! (default `~/Projects/electronic-transport`).

use caroli_born::{Scba, k_grid, magnetic};
use caroli_model::presets::{BhzParams, GFactors, bhz_spin};
use caroli_model::{CMat, c64};
use caroli_validation::read_records;
use caroli_validation::thesis::{ScbaAnalytic, bhz_hk};
use rayon::prelude::*;
use std::path::{Path, PathBuf};

const MASS: f64 = -0.137;
const B: f64 = -0.376406;
const ETA: f64 = 1e-3;

/// `(conc, M_z, anti)` of runs 00–07 (anti = 1: opposite g, 'A'; 0: same g, 'F').
const PANELS: [(f64, f64, i32); 8] = [
    (0.2, 1.0, 1),
    (0.2, 1.0, 0),
    (0.4, 1.0, 1),
    (0.4, 1.0, 0),
    (0.2, 0.8, 1),
    (0.2, 0.8, 0),
    (0.4, 0.8, 1),
    (0.4, 0.8, 0),
];

/// Numerical maps: folder under `anomalous/Results`, panel. The folder `anti_400(256)x64_0.4_0.8`
/// holds runs with conc = 0.2 according to their `ti_new.ini` and `param_00.dat`.
const MAPS: [(&str, usize); 6] = [
    ("Antiferro/anti_400(256)x64_0.2_1.0/64Jx64e", 0),
    ("Ferro/ferro_400(256)x64_0.2_1.0/64Jx64e", 1),
    ("Antiferro/anti_400(256)x64_0.4_1.0/32Jx64e", 2),
    ("Antiferro/anti_400(256)x64_0.4_0.8/32Jx64e", 4),
    ("Ferro/ferro_400(256)x64_0.2_0.8/64Jx64e", 5),
    ("Antiferro/200(160)x100_0.2_1.0/64jx64e", 0),
];

/// Readings of the renormalised bands `(M↑, μ↑, M↓, μ↓)` at `E_F` as a conductance (`None`:
/// metallic). A spin block is topological if `M_σ/B > 0` and gapped if `|E_F + μ_σ| < |M_σ|`.
/// - strict: metallic as soon as `E_F` lies in a band of either block, else the number of
///   topological blocks;
/// - localized: a block whose band holds `E_F` is localized by the disorder and adds nothing;
/// - topological: the number of topological blocks, whatever `E_F`.
const RULES: [&str; 3] = ["strict", "localized", "topological"];

fn born_g(rule: usize, r: [f64; 4], e: f64) -> Option<u8> {
    let blocks = [(r[0], r[1]), (r[2], r[3])];
    let gapped = |(m, mu): (f64, f64)| (e + mu).abs() < m.abs();
    let topological = |(m, _): (f64, f64)| m / B > 0.0;
    match rule {
        0 => blocks
            .iter()
            .all(|&b| gapped(b))
            .then(|| blocks.iter().filter(|&&b| topological(b)).count() as u8),
        1 => Some(blocks.iter().filter(|&&b| gapped(b) && topological(b)).count() as u8),
        _ => Some(blocks.iter().filter(|&&b| topological(b)).count() as u8),
    }
}

/// Corrected SCBA at one point: `(M↑, μ↑, M↓, μ↓)` from `Σ = Σ_MF + Σ₂`.
fn corrected(hk: &[CMat], conc: f64, mz: f64, anti: i32, j: f64, e: f64) -> ([f64; 4], usize) {
    let s = bhz_spin(if anti == 1 { GFactors::Opposite } else { GFactors::Same });
    let nz2 = mz * mz + (1.0 - mz).powi(2) / 3.0;
    let (mean_field, vertices) = magnetic(conc, j, mz, nz2, &s);
    let scba = Scba { hk: hk.to_vec(), mean_field: mean_field.clone(), vertices };
    let sol = scba.solve_anderson(c64::new(e, ETA), &CMat::zeros(4, 4), 8, 1e-11, 400);
    let total = &mean_field + &sol.sigma;
    let block = |b: usize| {
        let (a, d) = (total[(b, b)], total[(b + 1, b + 1)]);
        (MASS + ((a - d) * 0.5).re, -((a + d) * 0.5).re)
    };
    let ((mu, muu), (md, mud)) = (block(0), block(2));
    ([mu, muu, md, mud], sol.iterations)
}

fn thesis_scba(conc: f64, mz: f64, anti: i32) -> ScbaAnalytic {
    ScbaAnalytic {
        m: 256,
        n: 64,
        conc,
        mz,
        anti,
        points: 64,
        j_range: (0.0, 1.5),
        ef: (-0.16, 0.16),
        eta: ETA,
    }
}

/// Average over the realisations of a numerical map: `(J values, E values, ⟨G⟩[j + nj·e])`.
fn numerical_map(dir: &Path) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let mut sum: Vec<f64> = Vec::new();
    let mut count = 0;
    let mut axes = None;
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let Ok(rec) = read_records(entry.path().join("cond_00.dat")) else { continue };
        let c = caroli_validation::fortran::as_f64(&rec[0]);
        if sum.is_empty() {
            sum = vec![0.0; c.len()];
            let ax = read_records(entry.path().join("axes_00.dat")).ok()?;
            axes = Some(caroli_validation::fortran::as_f64(&ax[0]));
        }
        if c.len() == sum.len() {
            sum.iter_mut().zip(&c).for_each(|(s, x)| *s += x);
            count += 1;
        }
    }
    let axes = axes?;
    let ne = 64;
    let nj = sum.len() / ne;
    assert_eq!(axes.len(), nj + ne, "axes layout");
    let (j, e) = (axes[..nj].to_vec(), axes[nj..].to_vec());
    println!("  {}: {count} realisations, {nj} J × {ne} E", dir.display());
    Some((j, e, sum.iter().map(|s| s / count as f64).collect()))
}

fn main() {
    let archive = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var("HOME").expect("HOME")).join("Projects/electronic-transport")
    });
    let born_dir = archive.join("anomalous/Born Approx/run");
    let out_dir = caroli_validation::repo_root().join("target/born_lines");
    std::fs::create_dir_all(&out_dir).expect("output directory");
    let hk = bhz_hk(BhzParams::default(), 256, 64);
    assert_eq!(hk.len(), k_grid(256, 64).len());
    let (js, es): (Vec<f64>, Vec<f64>) = (
        (0..64).map(|i| 1.5 * i as f64 / 63.0).collect(),
        (0..64).map(|i| -0.16 + 0.32 * i as f64 / 63.0).collect(),
    );

    println!("Born grids (64 J_I in [0, 1.5] × 64 E_F in [−0.16, 0.16]):");
    println!("Strict reading (see RULES):");
    println!(
        "panel  conc  Mz   g  | replica vs published | points whose Born G changes | G=1 points thesis → corrected"
    );
    let mut corrected_grids = Vec::new();
    for (p, &(conc, mz, anti)) in PANELS.iter().enumerate() {
        let read = |name: &str| {
            caroli_validation::fortran::as_f64(
                &read_records(born_dir.join(format!("{name}_{p:02}.dat"))).unwrap()[0],
            )
        };
        let (pm_u, pm_d, pmu_u, pmu_d) = (read("mass_u"), read("mass_d"), read("mu_u"), read("mu_d"));
        // 1. The replica gives the published data (a few points).
        let thesis = thesis_scba(conc, mz, anti);
        let check: f64 = [(5usize, 7usize), (20, 40), (33, 12), (47, 58), (60, 30), (63, 63)]
            .par_iter()
            .map(|&(ij, ie)| {
                let r = thesis.point(js[ij], es[ie]);
                let k = ij + 64 * ie;
                [r[0] - pm_u[k], r[1] - pmu_u[k], r[2] - pm_d[k], r[3] - pmu_d[k]]
                    .iter()
                    .fold(0.0f64, |a, x| a.max(x.abs()))
            })
            .reduce(|| 0.0, f64::max);
        // 2. Corrected SCBA on the same grid.
        let grid: Vec<([f64; 4], usize)> = (0..64 * 64)
            .into_par_iter()
            .map(|k| corrected(&hk, conc, mz, anti, js[k % 64], es[k / 64]))
            .collect();
        // 3. Classification.
        let (mut changed, mut g1_thesis, mut g1_corr) = (0, 0, 0);
        let mut csv = String::from(
            "J_I,E_F,M_up_thesis,mu_up_thesis,M_dn_thesis,mu_dn_thesis,M_up,mu_up,M_dn,mu_dn,G_born_thesis,G_born\n",
        );
        for k in 0..64 * 64 {
            let e = es[k / 64];
            let t = [pm_u[k], pmu_u[k], pm_d[k], pmu_d[k]];
            let c = grid[k].0;
            let (gt, gc) = (born_g(0, t, e), born_g(0, c, e));
            changed += (gt != gc) as usize;
            g1_thesis += (gt == Some(1)) as usize;
            g1_corr += (gc == Some(1)) as usize;
            let g = |x: Option<u8>| x.map_or("metal".to_string(), |v| v.to_string());
            csv += &format!(
                "{},{},{},{},{},{},{},{},{},{},{},{}\n",
                js[k % 64],
                e,
                t[0],
                t[1],
                t[2],
                t[3],
                c[0],
                c[1],
                c[2],
                c[3],
                g(gt),
                g(gc)
            );
        }
        std::fs::write(out_dir.join(format!("panel_{p:02}.csv")), csv).unwrap();
        let iters = grid.iter().map(|g| g.1).max().unwrap();
        println!(
            "  {p:02}   {conc}  {mz}  {}  |      {check:9.1e}       |   {changed:4} / 4096 ({:4.1} %)     |   {g1_thesis} → {g1_corr}   (≤ {iters} Anderson steps)",
            if anti == 1 { 'A' } else { 'F' },
            100.0 * changed as f64 / 4096.0
        );
        corrected_grids.push(grid);
    }

    println!("\nAgainst the exact numerical conductance maps: share of the quantized points");
    println!("(|⟨G⟩ − round⟨G⟩| < 0.1) where the Born reading gives the same G, thesis / corrected SCBA:");
    println!("map{:45}quantized   {:13} {:13} {:13}", "", RULES[0], RULES[1], RULES[2]);
    let results = archive.join("anomalous/Results");
    for &(folder, p) in &MAPS {
        let (conc, mz, anti) = PANELS[p];
        let Some((jn, en, g)) = numerical_map(&results.join(folder)) else {
            println!("  {folder}: not found");
            continue;
        };
        let nj = jn.len();
        let thesis = thesis_scba(conc, mz, anti);
        let values: Vec<([f64; 4], [f64; 4])> = (0..g.len())
            .into_par_iter()
            .map(|k| {
                let (j, e) = (jn[k % nj], en[k / nj]);
                (thesis.point(j, e), corrected(&hk, conc, mz, anti, j, e).0)
            })
            .collect();
        let mut quantized = 0;
        let (mut agree_t, mut agree_c) = ([0usize; 3], [0usize; 3]);
        let mut csv = String::from(
            "J_I,E_F,G_numerical,M_up_thesis,mu_up_thesis,M_dn_thesis,mu_dn_thesis,M_up,mu_up,M_dn,mu_dn\n",
        );
        for (k, (t, c)) in values.iter().enumerate() {
            let (gn, e) = (g[k], en[k / nj]);
            if (gn - gn.round()).abs() < 0.1 {
                quantized += 1;
                for rule in 0..3 {
                    agree_t[rule] += (born_g(rule, *t, e) == Some(gn.round() as u8)) as usize;
                    agree_c[rule] += (born_g(rule, *c, e) == Some(gn.round() as u8)) as usize;
                }
            }
            csv += &format!(
                "{},{},{},{},{},{},{},{},{},{},{}\n",
                jn[k % nj],
                e,
                gn,
                t[0],
                t[1],
                t[2],
                t[3],
                c[0],
                c[1],
                c[2],
                c[3]
            );
        }
        let name = folder.replace(['/', '(', ')'], "_");
        std::fs::write(out_dir.join(format!("numerical_{name}.csv")), csv).unwrap();
        let pct = |a: usize| 100.0 * a as f64 / quantized.max(1) as f64;
        println!(
            "  {folder:44} {quantized:5}/{:5}  {:5.1} / {:5.1}  {:5.1} / {:5.1}  {:5.1} / {:5.1}",
            g.len(),
            pct(agree_t[0]),
            pct(agree_c[0]),
            pct(agree_t[1]),
            pct(agree_c[1]),
            pct(agree_t[2]),
            pct(agree_c[2])
        );
    }
    println!("\nCSV files in {}", out_dir.display());
}
