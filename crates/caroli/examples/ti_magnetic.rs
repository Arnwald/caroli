//! The 2D topological insulator with magnetic impurities of the author's thesis and of L. Raymond,
//! A. D. Verga and A. Demion, "Anomalous quantum Hall effect induced by disorder in topological
//! insulators": BHZ ribbons of HgTe (400 × 64 sites, between clean leads) whose 256 central
//! columns carry classical magnetic impurities polarised along z, for the two exchange couplings,
//! same-g (`g_E = g_H`) and opposite-g (`g_E = −g_H`).
//!
//! ```text
//! cargo run --release -p caroli --example ti_magnetic -- FIGURE OUT_DIR [--paper] [--configs N]
//! python3 crates/caroli/examples/plot_ti_magnetic.py OUT_DIR
//! ```
//!
//! Figures (CSV files in `OUT_DIR`):
//! - `maps`: disorder-averaged conductance `⟨G⟩(J_I, E_F)` and its spin-resolved parts. With the
//!   moments along z the exchange conserves `s_z`, so `G = T↑ + T↓`, each spin block a two-orbital
//!   BHZ model with its own disorder. Same-g with `n_I = 0.2` (`J_I ≤ 1.5`), opposite-g with
//!   `n_I = 0.4` (`J_I ≤ 0.75`), as the published maps. The quantised `G = 1` region is the
//!   anomalous quantum Hall (Chern insulator) phase.
//! - `local`: local currents, spin densities and local densities of states in six regimes, three
//!   per coupling, from two helical channels to a single chiral one and to none (`n_I = 0.2`).
//!   The current is the one injected by the left lead in a window of ±10⁻³ around `E_F`.
//! - `born`: the Born phase diagram, the renormalised masses `M̄_σ` and chemical potentials
//!   `μ̄_σ` of the two spin blocks from the self-consistent Born approximation of the full BHZ
//!   model (`caroli_born`), on the grids of `maps`. A block is topological where `M̄_σ/B > 0`,
//!   gapped at `E_F` where `|E_F + μ̄_σ| < |M̄_σ|`. The sign of the mass self-energy is the one
//!   of the general SCBA (see `docs/born-addendum`: the published program had it reversed).
//!
//! The default grids (16 `J_I` × 17 `E_F`, 4 configurations) take minutes; `--paper` uses the
//! published ones (64 × 64, 50 configurations: hours on a laptop).

use caroli::born::{Disorder, Scba};
use caroli::model::presets::{Spin, bhz_block};
use caroli::prelude::*;
use rayon::prelude::*;
use std::fmt::Write as _;

/// Ribbon length, width and length of the disordered region.
const L: usize = 400;
const W: usize = 64;
const DISORDERED: usize = 256;

/// The two published maps: coupling, impurity concentration, largest `J_I`.
const MAPS: [(GFactors, &str, f64, f64); 2] =
    [(GFactors::Same, "same-g", 0.2, 1.5), (GFactors::Opposite, "opposite-g", 0.4, 0.75)];

/// Range of the Fermi energy.
const E_F: (f64, f64) = (-0.16, 0.16);

/// The thesis' local-current regimes: coupling, `E_F`, `J_I`.
const REGIMES: [(GFactors, f64, f64); 6] = [
    (GFactors::Opposite, 0.053333, 0.2),
    (GFactors::Opposite, 0.053333, 0.666),
    (GFactors::Opposite, 0.053333, 1.6),
    (GFactors::Same, -0.08889, 0.2),
    (GFactors::Same, -0.08889, 1.0714),
    (GFactors::Same, 0.1, 1.4),
];

struct Options {
    grid: (usize, usize),
    configs: usize,
    k_grid: [usize; 2],
}

fn name(coupling: GFactors) -> &'static str {
    if coupling == GFactors::Same { "same-g" } else { "opposite-g" }
}

fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n).map(|k| if n == 1 { lo } else { lo + (hi - lo) * k as f64 / (n - 1) as f64 }).collect()
}

/// `int(conc · 256 · 64)` impurities with moments along +z in the central columns.
fn impurities(conc: f64, seed: u64) -> MagneticImpurities {
    let buffer = ((L - DISORDERED) / 2) as i64;
    let candidates: Vec<Site> = (buffer..buffer + DISORDERED as i64)
        .flat_map(|x| (0..W as i64).map(move |y| Site { cell: [x, y, 0], sub: 0 }))
        .collect();
    let mut rng = Xoshiro256::seed_from_u64(seed);
    let n = (conc * candidates.len() as f64) as usize;
    MagneticImpurities::generate(&mut rng, &candidates, n, Moments::Polarized { mz: 1.0 })
}

/// The two-orbital spin block `spin` of the ribbon with the exchange `J_I` of the impurities, and
/// its clean leads.
fn spin_block(imp: &MagneticImpurities, coupling: GFactors, j: f64, spin: Spin) -> NestedTwoTerminal {
    let o = if spin == Spin::Up { 0 } else { 2 };
    let exchange = imp.exchange(j, &bhz_spin(coupling));
    let device = bhz_block(BhzParams::default(), spin).with_onsite_term(move |s: &Site| {
        exchange.get(&Site { sub: 0, ..*s }).map(|m| CMat::from_fn(2, 2, |a, b| m[(o + a, o + b)]))
    });
    let lead = bhz_block(BhzParams::default(), spin);
    let region = Region::cuboid(lead.lattice(), [L, W, 0]);
    NestedTwoTerminal::from_model(&device, &lead, &region, 0)
}

fn maps(dir: &str, opt: &Options) {
    let (nj, ne) = opt.grid;
    let energies = linspace(E_F.0, E_F.1, ne);
    let mut csv = String::from("coupling,conc,j_i,e_f,config,t_up,t_dn\n");
    for (coupling, label, conc, j_max) in MAPS {
        let js = linspace(0.0, j_max, nj);
        // The leads are clean: their self-energies serve every J_I and configuration.
        let clean = impurities(0.0, 0);
        let sigma: Vec<[LeadSelfEnergies; 2]> = energies
            .par_iter()
            .map(|&e| {
                [Spin::Up, Spin::Down]
                    .map(|s| spin_block(&clean, coupling, 0.0, s).self_energies(e).expect("leads"))
            })
            .collect();
        let jobs: Vec<(usize, usize)> = (0..opt.configs).flat_map(|c| (0..nj).map(move |k| (c, k))).collect();
        let rows: Vec<String> = jobs
            .par_iter()
            .map(|&(c, k)| {
                let imp = impurities(conc, 1000 + c as u64);
                let [up, down] = [Spin::Up, Spin::Down].map(|s| spin_block(&imp, coupling, js[k], s));
                let mut out = String::new();
                for (ie, &e) in energies.iter().enumerate() {
                    let t_up = up.transmission_with(&sigma[ie][0]).unwrap_or(f64::NAN);
                    let t_dn = down.transmission_with(&sigma[ie][1]).unwrap_or(f64::NAN);
                    writeln!(out, "{label},{conc},{},{e},{c},{t_up},{t_dn}", js[k]).unwrap();
                }
                out
            })
            .collect();
        rows.iter().for_each(|r| csv.push_str(r));
        println!("{label}: {} configurations × {nj} J_I × {ne} E_F", opt.configs);
    }
    write(dir, "maps.csv", &csv);
}

/// Spin projectors in the basis `(E1↑, H1↑, E1↓, H1↓)`.
fn spin_projector(spin: Spin) -> CMat {
    let o = if spin == Spin::Up { 0 } else { 2 };
    CMat::from_fn(4, 4, |a, b| {
        if a == b && (o..o + 2).contains(&a) { c64::new(1.0, 0.0) } else { c64::new(0.0, 0.0) }
    })
}

fn local(dir: &str) {
    let clean = bhz(BhzParams::default());
    let lattice = clean.lattice().clone();
    let region = Region::cuboid(&lattice, [L, W, 0]);
    let imp = impurities(0.2, 7);
    let (up, down) = (spin_projector(Spin::Up), spin_projector(Spin::Down));
    let sz = combine(&[(1.0, &up), (-1.0, &down)]);
    let mut csv = String::from("regime,coupling,e_f,j_i,x,y,jx,jy,sjx,sjy,rho_up,rho_dn,ldos_up,ldos_dn\n");
    let mut totals = String::from("regime,coupling,e_f,j_i,g\n");
    for (r, &(coupling, e_f, j)) in REGIMES.iter().enumerate() {
        let device = bhz(BhzParams::default()).with_onsite_term(imp.exchange(j, &bhz_spin(coupling)));
        let sys = NestedTwoTerminal::from_model(&device, &clean, &region, 0);
        let energies = linspace(e_f - 0.0009, e_f + 0.0009, 10);
        let n = region.len();
        // Per site: current, s_z current, injected spin densities, local densities of states.
        let mut acc = vec![[0.0f64; 8]; n];
        let mut g = 0.0;
        for &e in &energies {
            let states = sys.scattering_states(e).expect("scattering states");
            let mut add_bonds = |bonds: Vec<BondCurrent>, slot: usize| {
                for b in bonds {
                    let (ra, rb) = (lattice.position(b.from), lattice.position(b.to));
                    let (dx, dy) = (rb[0] - ra[0], rb[1] - ra[1]);
                    for s in [b.from, b.to] {
                        let i = region.index_of(&s).expect("site");
                        acc[i][slot] += 0.5 * b.current * dx;
                        acc[i][slot + 1] += 0.5 * b.current * dy;
                    }
                }
            };
            add_bonds(sys.bond_currents(&states, &[1.0, 0.0]), 0);
            add_bonds(sys.operator_currents(&states, &[1.0, 0.0], &sz), 2);
            let columns = [
                sys.site_densities(&states, &[1.0, 0.0], Some(&up)),
                sys.site_densities(&states, &[1.0, 0.0], Some(&down)),
                sys.site_densities(&states, &[1.0, 1.0], Some(&up)),
                sys.site_densities(&states, &[1.0, 1.0], Some(&down)),
            ];
            for (slot, col) in columns.iter().enumerate() {
                acc.iter_mut().zip(col).for_each(|(a, v)| a[4 + slot] += v);
            }
            g += sys.transmission(e).expect("transmission");
        }
        let m = energies.len() as f64;
        for (site, a) in region.sites().iter().zip(&acc) {
            let p = lattice.position(*site);
            let v: Vec<String> = a.iter().map(|x| format!("{:.5e}", x / m)).collect();
            writeln!(csv, "{r},{},{e_f},{j},{},{},{}", name(coupling), p[0], p[1], v.join(",")).unwrap();
        }
        writeln!(totals, "{r},{},{e_f},{j},{}", name(coupling), g / m).unwrap();
        println!("regime {r}: {} J_I = {j}, E_F = {e_f}: G = {:.3}", name(coupling), g / m);
    }
    let positions: String = std::iter::once("x,y\n".to_string())
        .chain(imp.sites.iter().map(|s| {
            let p = lattice.position(*s);
            format!("{},{}\n", p[0], p[1])
        }))
        .collect();
    write(dir, "local.csv", &csv);
    write(dir, "local_totals.csv", &totals);
    write(dir, "impurities.csv", &positions);
}

fn born(dir: &str, opt: &Options) {
    let p = BhzParams::default();
    let model = bhz(p);
    let (nj, ne) = opt.grid;
    let energies = linspace(E_F.0, E_F.1, ne);
    let eta = 1e-3;
    let mut csv = String::from("coupling,conc,j_i,e_f,m_up,mu_up,m_dn,mu_dn,residual\n");
    for (coupling, label, conc, j_max) in MAPS {
        let js = linspace(0.0, j_max, nj);
        let rows: Vec<String> = js
            .par_iter()
            .map(|&j| {
                let disorder =
                    Disorder::magnetic(conc, j, Moments::Polarized { mz: 1.0 }, &bhz_spin(coupling));
                let scba = Scba::from_model(&model, &opt.k_grid, disorder);
                let mut out = String::new();
                for (sol, &e) in scba.sweep(&energies, eta).iter().zip(&energies) {
                    let total = scba.total_self_energy(&sol.sigma);
                    // M̄ = M + Re(Σ_E − Σ_H)/2 and μ̄ = −Re(Σ_E + Σ_H)/2 in each spin block.
                    let block = |o: usize| {
                        let (a, d) = (total[(o, o)].re, total[(o + 1, o + 1)].re);
                        (p.mass + 0.5 * (a - d), -0.5 * (a + d))
                    };
                    let ((mu, muu), (md, mud)) = (block(0), block(2));
                    writeln!(out, "{label},{conc},{j},{e},{mu},{muu},{md},{mud},{:e}", sol.residual).unwrap();
                }
                out
            })
            .collect();
        rows.iter().for_each(|r| csv.push_str(r));
        println!("{label}: Born approximation on {nj} J_I × {ne} E_F (k grid {:?})", opt.k_grid);
    }
    write(dir, "born.csv", &csv);
}

fn write(dir: &str, file: &str, content: &str) {
    let path = std::path::Path::new(dir).join(file);
    std::fs::write(&path, content).expect("write");
    println!("{}", path.display());
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: ti_magnetic (maps|local|born|all) OUT_DIR [--paper] [--configs N]";
    let (figure, dir) = match (args.first(), args.get(1)) {
        (Some(f), Some(d)) => (f.as_str(), d.as_str()),
        _ => panic!("{usage}"),
    };
    let paper = args.iter().any(|a| a == "--paper");
    let mut opt = if paper {
        Options { grid: (64, 64), configs: 50, k_grid: [256, 64] }
    } else {
        Options { grid: (16, 17), configs: 4, k_grid: [128, 128] }
    };
    if let Some(k) = args.iter().position(|a| a == "--configs") {
        opt.configs = args.get(k + 1).and_then(|v| v.parse().ok()).expect("--configs N");
    }
    std::fs::create_dir_all(dir).expect("output directory");
    let t = std::time::Instant::now();
    match figure {
        "maps" => maps(dir, &opt),
        "local" => local(dir),
        "born" => born(dir, &opt),
        "all" => {
            maps(dir, &opt);
            local(dir);
            born(dir, &opt);
        }
        _ => panic!("{usage}"),
    }
    println!("done in {:.1?}", t.elapsed());
}
