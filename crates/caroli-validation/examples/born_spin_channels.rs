//! Spin-resolved exact conductance where the published and corrected Born lines of the g2g1
//! paper disagree on the topology of a spin block.
//!
//! With moments along z (M_z = 1), the exchange conserves s_z, so the transmission splits into
//! a spin-↑ block and a spin-↓ block ([`TwoTerminal::blocks`]). A quantized T_σ = 1 needs block
//! σ to be topological, whatever the Born lines say.
//!
//! The systems are those of the thesis maps (`anomalous/ipy/data/{A,F}/G_*.256x64.NI0*.Mz1`):
//! - a 400 × 64 ribbon;
//! - conc·256·64 impurities in the central 256 columns;
//! - eight configurations per point.
//!
//! The 'AF' points are repeated on a ribbon twice as wide (W = 128). An edge state of a BHZ
//! block with mass M decays over about 1.5/|M| sites. When this length reaches W/2, the edge
//! states of a block overlap across the ribbon and its channel is lost before M changes sign;
//! a wider ribbon then restores it. A trivial block carries no channel at any width.
//!
//! The Born states come from the published program (`self_born`) and from its corrected
//! version. A block is topological if M/B > 0, and E_F is in its gap if |E_F + μ| < |M|.
//!
//! `cargo run --release -p caroli-validation --example born_spin_channels`

use caroli_green::TwoTerminal;
use caroli_lattice::Region;
use caroli_model::presets::{BhzParams, GFactors, bhz, bhz_spin};
use caroli_validation::thesis::{Conductance2t, ScbaAnalytic};
use rayon::prelude::*;

const B: f64 = -0.376406;
const SEEDS: [i32; 8] = [11, 23, 37, 41, 53, 67, 79, 97];

/// (title, opposite-g, conc, width, points (J_I, E_F)).
type Case = (&'static str, i32, f64, usize, &'static [(f64, f64)]);
const CASES: [Case; 3] = [
    (
        "'F' (same-g), n_I = 0.2, M_z = 1, W = 64",
        0,
        0.2,
        64,
        &[(0.3, 0.0), (0.9, -0.1), (1.2, -0.05), (1.3, -0.12), (1.45, 0.0)],
    ),
    (
        "'AF' (opposite-g), n_I = 0.4, M_z = 1, W = 64",
        1,
        0.4,
        64,
        &[
            (0.2, 0.02),
            (0.25, 0.02),
            (0.3, 0.02),
            (0.35, 0.02),
            (0.4, 0.02),
            (0.45, 0.02),
            (0.5, 0.02),
            (0.6, 0.05),
        ],
    ),
    (
        "'AF' (opposite-g), n_I = 0.4, M_z = 1, W = 128",
        1,
        0.4,
        128,
        &[(0.25, 0.02), (0.3, 0.02), (0.35, 0.02), (0.4, 0.02)],
    ),
];

/// The Born program of the paper's panels (`born_lines`).
fn scba(conc: f64, anti: i32) -> ScbaAnalytic {
    ScbaAnalytic {
        m: 256,
        n: 64,
        conc,
        mz: 1.0,
        anti,
        points: 64,
        j_range: (0.0, 1.5),
        ef: (-0.16, 0.16),
        eta: 1e-3,
    }
}

/// State of the two spin blocks from `(M↑, μ↑, M↓, μ↓)`.
fn states(r: [f64; 4], e: f64) -> String {
    let block = |m: f64, mu: f64| {
        let topo = if m / B > 0.0 { "topo" } else { "triv" };
        let gap = if (e + mu).abs() < m.abs() { "gap " } else { "band" };
        format!("{topo} {gap} (M {m:+.3})")
    };
    format!("↑ {}  ↓ {}", block(r[0], r[1]), block(r[2], r[3]))
}

/// Exact transmissions `(T↑, T↓)` of one configuration.
fn spin_transmissions(anti: i32, conc: f64, ny: usize, j: f64, e: f64, seed: i32) -> [f64; 2] {
    let (nx, nx_imp) = (400usize, 256usize);
    let imp = Conductance2t { nx, ny, nx_imp, conc, spin: 1, theta0_deg: 0.0, seed }.impurities(1);
    let coupling = if anti == 1 { GFactors::Opposite } else { GFactors::Same };
    let clean = bhz(BhzParams::default());
    let device = bhz(BhzParams::default()).with_onsite_term(imp.exchange(j, &bhz_spin(coupling)));
    let sys = TwoTerminal::from_model(&device, &clean, &Region::cuboid(clean.lattice(), [nx + 2, ny, 0]), 0);
    let blocks = sys.blocks();
    assert_eq!(blocks.len(), 2, "s_z is conserved");
    let mut t = [0.0; 2];
    for b in &blocks {
        // Orbitals (E↑, H↑, E↓, H↓) on each site.
        let spin = b[0] % 4 / 2;
        assert!(b.iter().all(|&o| o % 4 / 2 == spin), "a block mixes the spins");
        t[spin] = sys.restricted(b).transmission(e).unwrap();
    }
    t
}

fn main() {
    for (title, anti, conc, ny, points) in CASES {
        let jobs: Vec<(f64, f64, i32)> =
            points.iter().flat_map(|&(j, e)| SEEDS.iter().map(move |&s| (j, e, s))).collect();
        let t: Vec<[f64; 2]> =
            jobs.par_iter().map(|&(j, e, s)| spin_transmissions(anti, conc, ny, j, e, s)).collect();
        let born = scba(conc, anti);
        println!("\n{title}: exact T per spin, mean [min, max] over {} configurations", SEEDS.len());
        for (p, &(j, e)) in points.iter().enumerate() {
            let runs = &t[p * SEEDS.len()..(p + 1) * SEEDS.len()];
            let stat = |s: usize| {
                let v: Vec<f64> = runs.iter().map(|r| r[s]).collect();
                let mean = v.iter().sum::<f64>() / v.len() as f64;
                let (lo, hi) = v.iter().fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
                format!("{mean:.3} [{lo:.3}, {hi:.3}]")
            };
            println!("  J_I {j:.2}, E_F {e:+.3}: T↑ {}  T↓ {}", stat(0), stat(1));
            println!("      published  {}", states(born.point(j, e), e));
            println!("      corrected  {}", states(born.point_corrected(j, e), e));
        }
    }
}
