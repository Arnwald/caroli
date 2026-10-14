//! Graphene with classical magnetic impurities under a strong electric field: the figures of
//! A. Demion and A. D. Verga, "Nonlinear electric transport in graphene with magnetic disorder",
//! Phys. Rev. B 90, 085412 (2014), arXiv:1406.3743, the second study of the author's thesis.
//!
//! The model of the paper (Sec. II): spinful graphene, hopping ν = 1, lattice constant a = 1
//! (nearest neighbours at 1/√3), on an n × n torus; impurities on a fraction n_I = 0.4 of the A
//! sites and of the B sites (`--conc`), coupled to the electron spin by `J_I n·σ`, either
//! paramagnetic (isotropic moments) or magnetic (A moments along +z, B along −z, which opens a
//! gap). The field E along y enters through a vector potential, `k → k + E t`. Units
//! ν = a = ħ = e = 1: t₀ = ħ/ν ≈ 0.3 fs and E₀ = ν/(e a) ≈ 10¹⁰ V/m for ν = 3 eV, a = 0.25 nm.
//!
//! As in the paper (Eqs. 15–16), the initial state is the spin-up valence band of the clean sheet,
//! `|k, −, ↑⟩`, and N counts the carriers in its spin-up conduction band `|k, +, ↑⟩`: spin-flip
//! scattering by paramagnetic impurities removes carriers from N. The paper averages a spin-up
//! hole started at the origin over 8192 disorder configurations; that average is the trace over
//! the band, estimated here by random-phase vectors on a large torus (each vector averages over
//! every starting site). The current and the pairs are per unit cell (the paper divides by the
//! number of cells, not by the area) and doubled for the two spins. With `--ground-state` the
//! evolution starts from the Fermi sea of the disordered Hamiltonian, both spins, instead: the
//! clean valence band is not a stationary state of the disordered sheet, and the disorder alone
//! excites carriers at `t = 0⁺` (the jump of N in the paper's Fig. 7).
//!
//! ```text
//! cargo run --release -p caroli --example graphene_field -- FIGURE OUT_DIR [--paper] [--ground-state]
//! python3 crates/caroli/examples/plot_graphene_field.py OUT_DIR
//! ```
//!
//! Figures (CSV files in `OUT_DIR`):
//! - `clean` (Fig. 3): current j(t) and pairs N(t) of the clean sheet for E = 0.002 … 0.02; j/E and
//!   N/E against t√E collapse on lines of slopes 0.16 and 0.095, those the paper measures. They
//!   are the slopes of Eq. (18), σ₀√v_F ≈ 0.189 and that over 2v_F (σ₀ = 2/π², v_F = √3/2), per
//!   unit cell: times the area of the cell, √3/2. The initial plateau of j/E is the
//!   low-frequency conductivity 1/4, per cell.
//! - `dos` (Fig. 4): densities of states per cell, paramagnetic and magnetic, J_I = 0.1 … 1.5.
//! - `spreading` (Fig. 6): width w(t) of a hole wave packet without field (the B, spin-up orbital
//!   of one site projected on the valence band), and its exponent β (w ~ t^β) against J_I.
//! - `disorder` (Fig. 7): current and pairs with disorder: paramagnetic and magnetic at J_I = 0.2
//!   for E = 0.002 … 0.02; magnetic at E = 0.02 for J_I = 0 … 1. The statistical error of the
//!   current is about v_F/(n√R) per cell for R vectors (`--vectors`) on an n × n torus, 0.003
//!   with the defaults; the paper's average is worth about 2000 vectors. That of N is far
//!   smaller.
//! - `ldos` (Fig. 8): in the magnetic case, the local densities of states at ε = 0.2 on a sample
//!   of sites for J_I = 0.1 … 1.5 (their logarithmic distribution), and at J_I = 1, ε = ±0.1, per
//!   spin on a patch of 2 × 32² sites, with the impurities. The Hamiltonian anticommutes with the
//!   product of the sublattice sign and time reversal, so the total local density of states is the
//!   same at ±ε on every site: no puddles of charge. With moments along ±z the sublattice sign
//!   alone maps spin up at ε onto spin down at −ε, `ρ↑(i, ε) = ρ↓(i, −ε)`: the electron and hole
//!   puddles of one spin are the spin puddles (the paper's Fig. 8b and c).
//!
//! The default sizes take about half an hour on four cores (`all`); `--paper` uses those of the
//! paper (a 256 × 256 torus, every field and disorder strength of the figures), hours.

use caroli::dynamics::{Evolution, Torus};
use caroli::kpm::Spectrum;
use caroli::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt::Write as _;

#[derive(Clone, Copy, PartialEq)]
enum Phase {
    Clean,
    Para,
    Magnetic,
}

fn name(p: Phase) -> &'static str {
    match p {
        Phase::Clean => "clean",
        Phase::Para => "para",
        Phase::Magnetic => "magnetic",
    }
}

/// Area of the unit cell (lattice constant 1).
const CELL: f64 = 0.866_025_403_784_438_6;

/// Chebyshev moments of the densities of states: a resolution of about 0.02 near ε = 0 at
/// J_I = 0, that of the paper's Fig. 4 (van Hove peaks of 1.7 per cell at J_I = 0.1).
const MOMENTS: usize = 512;

struct Options {
    /// Torus of the dynamics and of the densities of states.
    n: usize,
    /// Impurities per site of each sublattice.
    conc: f64,
    /// Random-phase vectors of the Fermi-sea traces.
    vectors: usize,
    /// Chebyshev moments of the densities of states (`None`: [`MOMENTS`]).
    moments: Option<usize>,
    paper: bool,
    ground_state: bool,
}

/// The impurities of `phase` on an `n × n` torus: sites and on-site matrices `J_I n·σ`.
fn impurities(n: usize, phase: Phase, j: f64, conc: f64, seed: u64) -> HashMap<Site, CMat> {
    let mut terms = HashMap::new();
    if phase == Phase::Clean || j == 0.0 {
        return terms;
    }
    let mut rng = Xoshiro256::seed_from_u64(seed);
    let count = (conc * (n * n) as f64) as usize;
    for sub in 0..2 {
        let cells: Vec<Site> =
            (0..n as i64).flat_map(|y| (0..n as i64).map(move |x| Site { cell: [x, y, 0], sub })).collect();
        let moments = if phase == Phase::Para { Moments::Isotropic } else { Moments::Polarized { mz: 1.0 } };
        let imp = MagneticImpurities::generate(&mut rng, &cells, count, moments);
        let sign = if phase == Phase::Magnetic && sub == 1 { -1.0 } else { 1.0 };
        for (site, m) in imp.sites.iter().zip(&imp.moments) {
            let s = sign * j;
            terms.insert(*site, combine(&[(s * m[0], &sx()), (s * m[1], &sy()), (s * m[2], &sz())]));
        }
    }
    terms
}

/// Spinful graphene (orbitals: spin fastest, A then B) with the impurities `terms`.
fn graphene(terms: HashMap<Site, CMat>) -> Model {
    let t = combine(&[(-1.0, &s0())]);
    let model = Model::new(Lattice::honeycomb(1.0 / 3f64.sqrt(), 2))
        .hopping(0, 1, [0, 0, 0], &t)
        .hopping(0, 1, [0, -1, 0], &t)
        .hopping(0, 1, [1, -1, 0], &t);
    if terms.is_empty() { model } else { model.with_onsite_term(move |s: &Site| terms.get(s).cloned()) }
}

/// The spin-up orbitals of the cell (A↑, B↑).
const SPIN_UP: [usize; 2] = [0, 2];

/// Current and pairs per unit cell against time, under a field `e` along y: `(t, j, N)`. From the
/// clean spin-up valence band, N counting the spin-up conduction band, both doubled for the two
/// spins; or from the Fermi sea of the disordered sheet, both spins.
fn evolve(opt: &Options, phase: Phase, j: f64, e: f64, seed: u64) -> Vec<(f64, f64, f64)> {
    let model = graphene(impurities(opt.n, phase, j, opt.conc, seed));
    let torus = Torus::new(&model, opt.n).with_force([0.0, e, 0.0]);
    let vectors = if phase == Phase::Clean { 1 } else { opt.vectors };
    let (torus, starts, weight) = if opt.ground_state && phase != Phase::Clean {
        let starts = torus.fermi_sea(0.0, vectors, seed + 1, 2048);
        (torus, starts, CELL)
    } else {
        let torus = torus.with_sector(&SPIN_UP);
        let starts = torus.filled(0.0, vectors, seed + 1);
        (torus, starts, 2.0 * CELL)
    };
    let r =
        torus.evolve(&starts, &Evolution { dt: 0.1, steps: 1000, every: 10, fermi: 0.0, densities: false });
    (0..r.times.len()).map(|k| (r.times[k], r.current[k][1] * weight, r.excited[k] * weight)).collect()
}

fn fields(opt: &Options) -> Vec<f64> {
    let count = if opt.paper { 10 } else { 5 };
    (1..=count).map(|k| 0.02 * k as f64 / count as f64).collect()
}

fn clean(dir: &str, opt: &Options) {
    let rows: Vec<String> = fields(opt)
        .par_iter()
        .map(|&e| {
            evolve(opt, Phase::Clean, 0.0, e, 1)
                .iter()
                .map(|(t, j, n)| format!("{e},{t},{j:e},{n:e}\n"))
                .collect()
        })
        .collect();
    write(dir, "clean.csv", &(String::from("field,t,j,pairs\n") + &rows.concat()));
}

fn disorder(dir: &str, opt: &Options) {
    let mut jobs: Vec<(Phase, f64, f64)> = Vec::new();
    for phase in [Phase::Para, Phase::Magnetic] {
        jobs.extend(fields(opt).into_iter().map(|e| (phase, 0.2, e)));
    }
    let step = if opt.paper { 0.05 } else { 0.1 };
    let count = (1.0 / step + 0.5) as usize;
    for k in 0..=count {
        let job = (Phase::Magnetic, round(step * k as f64), 0.02);
        if !jobs.contains(&job) {
            jobs.push(job);
        }
    }
    let rows: Vec<String> = jobs
        .par_iter()
        .map(|&(phase, j, e)| {
            let out: String = evolve(opt, phase, j, e, 7)
                .iter()
                .map(|(t, c, n)| format!("{},{j},{e},{t},{c:e},{n:e}\n", name(phase)))
                .collect();
            println!("disorder: {} J_I = {j}, E = {e}", name(phase));
            out
        })
        .collect();
    write(dir, "disorder.csv", &(String::from("phase,j_i,field,t,j,pairs\n") + &rows.concat()));
}

fn dos(dir: &str, opt: &Options) {
    let energies: Vec<f64> = (0..=800).map(|k| -4.0 + 8.0 * k as f64 / 800.0).collect();
    let jobs: Vec<(Phase, f64)> =
        [Phase::Para, Phase::Magnetic].iter().flat_map(|&p| [0.1, 0.5, 1.0, 1.5].map(|j| (p, j))).collect();
    let rows: Vec<String> = jobs
        .iter()
        .map(|&(phase, j)| {
            let model = graphene(impurities(opt.n, phase, j, opt.conc, 3));
            let region = Region::cuboid(model.lattice(), [opt.n, opt.n, 0]);
            let n = opt.n as i64;
            let moments = opt.moments.unwrap_or(MOMENTS);
            let spectrum = Spectrum::periodic(&model, &region, &[[n, 0, 0], [0, n, 0]])
                .moments(moments)
                .vectors(4)
                .bounds(-3.0 - 2.0 * j, 3.0 + 2.0 * j);
            let cells = (opt.n * opt.n) as f64;
            println!("dos: {} J_I = {j}", name(phase));
            energies
                .iter()
                .zip(spectrum.dos(&energies))
                .map(|(e, rho)| format!("{},{j},{e},{:e}\n", name(phase), rho / cells))
                .collect::<String>()
        })
        .collect();
    write(dir, "dos.csv", &(String::from("phase,j_i,energy,dos\n") + &rows.concat()));
}

fn spreading(dir: &str, opt: &Options) {
    // A wider torus than the dynamics with a field: the packet must not wrap around.
    let n = 2 * opt.n;
    let step = if opt.paper { 0.1 } else { 0.2 };
    let js: Vec<f64> = (0..=(1.6 / step + 0.5) as usize).map(|k| round(step * k as f64)).collect();
    let jobs: Vec<(Phase, f64)> =
        [Phase::Para, Phase::Magnetic].iter().flat_map(|&p| js.iter().map(move |&j| (p, j))).collect();
    let (dt, steps) = (0.1, 2000);
    let series: Vec<(Phase, f64, Vec<f64>, Vec<f64>)> = jobs
        .par_iter()
        .map(|&(phase, j)| {
            let torus = Torus::new(&graphene(impurities(n, phase, j, opt.conc, 5)), n);
            // The B, spin-up orbital (index 2) of the central cell, projected on the valence band.
            let start = torus.localized([n / 2, n / 2], 2, Some((-10.0, 0.0)));
            let r = torus.evolve(&[start], &Evolution { dt, steps, every: 10, fermi: 0.0, densities: false });
            println!("spreading: {} J_I = {j}", name(phase));
            (phase, j, r.times, r.width)
        })
        .collect();
    let mut csv = String::from("phase,j_i,t,width\n");
    let mut fits = String::from("phase,j_i,beta,t_from,t_to\n");
    for (phase, j, times, width) in &series {
        for (t, w) in times.iter().zip(width) {
            writeln!(csv, "{},{j},{t},{w:e}", name(*phase)).unwrap();
        }
        // w ~ t^β on t ≥ 10, before the packet feels the size of the torus (w < n/6).
        let points: Vec<(f64, f64)> = times
            .iter()
            .zip(width)
            .filter(|(t, w)| **t >= 10.0 && **w < n as f64 / 6.0)
            .map(|(t, w)| (t.ln(), w.ln()))
            .collect();
        if points.len() > 3 {
            let m = points.len() as f64;
            let (sx, sy) = points.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
            let (sxx, sxy) = points.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0 * p.0, a.1 + p.0 * p.1));
            let beta = (m * sxy - sx * sy) / (m * sxx - sx * sx);
            let (from, to) = (points[0].0.exp(), points[points.len() - 1].0.exp());
            writeln!(fits, "{},{j},{beta},{from},{to}", name(*phase)).unwrap();
        }
    }
    write(dir, "spreading.csv", &csv);
    write(dir, "spreading_beta.csv", &fits);
}

fn ldos(dir: &str, opt: &Options) {
    let n = 64usize;
    let moments = opt.moments.unwrap_or(MOMENTS);
    let lattice = Lattice::honeycomb(1.0 / 3f64.sqrt(), 2);
    // Histogram: ε = 0.2 on a sample of sites.
    let sample = if opt.paper { 2048 } else { 512 };
    let mut csv = String::from("j_i,site,ldos\n");
    for j in [0.1, 0.5, 1.0, 1.5] {
        let model = graphene(impurities(n, Phase::Magnetic, j, opt.conc, 11));
        let region = Region::cuboid(model.lattice(), [n, n, 0]);
        let ni = n as i64;
        let spectrum = Spectrum::periodic(&model, &region, &[[ni, 0, 0], [0, ni, 0]])
            .moments(moments)
            .bounds(-3.0 - 2.0 * j, 3.0 + 2.0 * j);
        let mut rng = Xoshiro256::seed_from_u64(13);
        let sites: Vec<usize> = (0..sample).map(|_| rng.integer(1, region.len()) - 1).collect();
        for (k, l) in spectrum.ldos(&sites, &[0.2]).iter().enumerate() {
            writeln!(csv, "{j},{},{:e}", sites[k], l[0]).unwrap();
        }
        println!("ldos: J_I = {j}");
    }
    write(dir, "ldos_histogram.csv", &csv);
    // Maps: J_I = 1, ε = ±0.1, on a patch of the torus.
    let patch = 32;
    let terms = impurities(n, Phase::Magnetic, 1.0, opt.conc, 11);
    let imp: Vec<(Site, f64)> = terms.iter().map(|(s, m)| (*s, m[(0, 0)].re)).collect();
    let model = graphene(terms);
    let region = Region::cuboid(model.lattice(), [n, n, 0]);
    let ni = n as i64;
    let spectrum =
        Spectrum::periodic(&model, &region, &[[ni, 0, 0], [0, ni, 0]]).moments(moments).bounds(-5.0, 5.0);
    let inside = |s: &Site| s.cell[0] < patch as i64 && s.cell[1] < patch as i64;
    let sites: Vec<usize> = (0..region.len()).filter(|&i| inside(&region.sites()[i])).collect();
    let mut map = String::from("x,y,sublattice,up_plus,down_plus,up_minus,down_minus\n");
    for (&i, o) in sites.iter().zip(spectrum.orbital_ldos(&sites, &[0.1, -0.1])) {
        let p = lattice.position(region.sites()[i]);
        // Orbitals: spin up, spin down; energies +0.1, −0.1.
        let (up, down) = (&o[0], &o[1]);
        let sub = region.sites()[i].sub;
        writeln!(map, "{},{},{sub},{:e},{:e},{:e},{:e}", p[0], p[1], up[0], down[0], up[1], down[1]).unwrap();
    }
    write(dir, "ldos_map.csv", &map);
    let mut out = String::from("x,y,spin\n");
    for (s, z) in imp.iter().filter(|(s, _)| inside(s)) {
        let p = lattice.position(*s);
        writeln!(out, "{},{},{}", p[0], p[1], if *z > 0.0 { "up" } else { "down" }).unwrap();
    }
    write(dir, "ldos_impurities.csv", &out);
}

/// `x` to two decimals (the disorder strengths of the scans).
fn round(x: f64) -> f64 {
    (100.0 * x).round() / 100.0
}

fn write(dir: &str, file: &str, content: &str) {
    let path = std::path::Path::new(dir).join(file);
    std::fs::write(&path, content).expect("write");
    println!("{}", path.display());
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: graphene_field (clean|dos|spreading|disorder|ldos|all) OUT_DIR [--paper] [--ground-state] [--n N] [--conc C] [--vectors R] [--moments M]";
    let (figure, dir) = match (args.first(), args.get(1)) {
        (Some(f), Some(d)) => (f.as_str(), d.as_str()),
        _ => panic!("{usage}"),
    };
    let paper = args.iter().any(|a| a == "--paper");
    let value =
        |flag: &str| args.iter().position(|a| a == flag).map(|k| args.get(k + 1).expect(usage).as_str());
    let opt = Options {
        n: value("--n").map_or(if paper { 256 } else { 128 }, |v| v.parse().expect("--n N")),
        conc: value("--conc").map_or(0.4, |v| v.parse().expect("--conc C")),
        vectors: value("--vectors").map_or(4, |v| v.parse().expect("--vectors R")),
        moments: value("--moments").map(|v| v.parse().expect("--moments M")),
        paper,
        ground_state: args.iter().any(|a| a == "--ground-state"),
    };
    std::fs::create_dir_all(dir).expect("output directory");
    let t = std::time::Instant::now();
    match figure {
        "clean" => clean(dir, &opt),
        "dos" => dos(dir, &opt),
        "spreading" => spreading(dir, &opt),
        "disorder" => disorder(dir, &opt),
        "ldos" => ldos(dir, &opt),
        "all" => {
            clean(dir, &opt);
            dos(dir, &opt);
            spreading(dir, &opt);
            disorder(dir, &opt);
            ldos(dir, &opt);
        }
        _ => panic!("{usage}"),
    }
    println!("done in {:.1?}", t.elapsed());
}
