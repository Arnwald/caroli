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
//! scattering by paramagnetic impurities removes carriers from N. The current and the pairs are
//! per unit cell (the paper divides by the number of cells, not by the area) and doubled for the
//! two spins. With `--ground-state` the evolution starts from the Fermi sea of the disordered
//! Hamiltonian, both spins, instead: the clean valence band is not a stationary state of the
//! disordered sheet, and the disorder alone excites carriers at `t = 0⁺` (the jump of N in the
//! paper's Fig. 7).
//!
//! The thesis averaged over 8192 disorder configurations, each with one hole started at the
//! origin. The trace over the band is estimated here by random-phase vectors instead, each of
//! which averages over every starting site of the torus: `noise` measures that one vector on a
//! 128 × 128 torus is worth 12 to 140 thesis configurations for the current (about 40 typically;
//! more on a larger torus, in proportion to its area) and 7 to 3000 for N. `--paper` (a 256 × 256
//! torus, 64 vectors) is thus worth 3000 to 36000 configurations: the statistics of the paper's
//! figures.
//!
//! ```text
//! cargo run --release -p caroli --example graphene_field -- estimate [--paper]   # durations here
//! cargo run --release -p caroli --example graphene_field -- all OUT_DIR [--paper] # then the plots
//! ```
//!
//! `all` runs every figure and then the plot script (`plot_graphene_field.py`, numpy and
//! matplotlib). Interrupted, it starts again where it stopped: finished figures and the finished
//! jobs of the long ones (`OUT_DIR/parts`) are kept. On macOS it keeps the machine awake
//! (`caffeinate`). A single figure: `FIGURE OUT_DIR` with `FIGURE` among those below; `--force`
//! recomputes.
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
//!   current is about v_F/(n√R) per cell for R vectors (`--vectors`) on an n × n torus: 0.003
//!   with the defaults, 0.0004 with `--paper`. That of N is far smaller.
//! - `ldos` (Fig. 8): in the magnetic case, the local densities of states at ε = 0.2 on a sample
//!   of sites for J_I = 0.1 … 1.5 (their logarithmic distribution), and at J_I = 1, ε = ±0.1, per
//!   spin on a patch of 2 × 32² sites, with the impurities. The Hamiltonian anticommutes with the
//!   product of the sublattice sign and time reversal, whatever the moments: hence
//!   `ρ↑(i, ε) = ρ↓(i, −ε)` on every site, and the total local density of states is the same at
//!   ±ε (no puddles of charge). The electron and hole puddles of one spin are the spin puddles:
//!   the paper's Fig. 8b and 8c show the same thing.
//!
//! Other commands: `estimate` times the dynamics and the densities of states on this machine and
//! prints the duration of each figure with the options given, and that of the thesis protocol
//! (8192 configurations); `noise OUT_DIR [--samples S]` measures the statistical error of one
//! thesis configuration and of one random-phase vector (`noise.csv`, `noise_summary.csv`).
//!
//! The defaults take about twenty minutes on four cores; `--paper` uses the sizes of the paper
//! (every field and disorder strength of the figures, a 256 × 256 torus, 64 vectors, 32
//! configurations of 2048 sites for the histogram of Fig. 8): `estimate --paper` gives the time.

use caroli::dynamics::{Evolution, Torus};
use caroli::kpm::Spectrum;
use caroli::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

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

/// Cells of the thesis's sheet: its N = 256 grid (steps 4π/N and 4π/(√3 N) in k) holds two
/// decoupled copies of a sheet of N²/2 cells, 128 a × 222 a, and the hole fills one of them.
const THESIS_CELLS: f64 = 32768.0;

/// Disorder configurations of the thesis, per case of Fig. 7.
const THESIS_CONFIGURATIONS: f64 = 8192.0;

#[derive(Debug)]
struct Options {
    /// Torus of the dynamics and of the densities of states.
    n: usize,
    /// Impurities per site of each sublattice.
    conc: f64,
    /// Random-phase vectors of the band traces.
    vectors: usize,
    /// Chebyshev moments of the densities of states (`None`: [`MOMENTS`]).
    moments: Option<usize>,
    /// Disorder configurations averaged per point of `spreading`.
    configurations: usize,
    /// Sites and disorder configurations of the histogram of `ldos`.
    ldos_sites: usize,
    ldos_configurations: usize,
    /// Disorder configurations per case of `noise`.
    samples: usize,
    paper: bool,
    ground_state: bool,
    /// Recompute what `OUT_DIR` already holds.
    force: bool,
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

/// The cases of Fig. 7: `(phase, J_I, E)`.
fn disorder_jobs(opt: &Options) -> Vec<(Phase, f64, f64)> {
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
    jobs
}

fn disorder(dir: &str, opt: &Options) {
    let jobs = disorder_jobs(opt);
    let rows: Vec<String> = jobs
        .par_iter()
        .map(|&(phase, j, e)| {
            part(dir, opt, "disorder", &format!("{}-J{j}-E{e}", name(phase)), || {
                evolve(opt, phase, j, e, 7)
                    .iter()
                    .map(|(t, c, n)| format!("{},{j},{e},{t},{c:e},{n:e}\n", name(phase)))
                    .collect()
            })
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
            let spectrum = Spectrum::periodic(&model, &region, &[[n, 0, 0], [0, n, 0]])
                .moments(opt.moments.unwrap_or(MOMENTS))
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

/// The disorder strengths of Fig. 6.
fn spreading_strengths(opt: &Options) -> Vec<f64> {
    let step = if opt.paper { 0.05 } else { 0.2 };
    (0..=(1.6 / step + 0.5) as usize).map(|k| round(step * k as f64)).collect()
}

/// Steps of `spreading` (dt = 0.1, up to t = 200).
const SPREADING_STEPS: usize = 2000;

fn spreading(dir: &str, opt: &Options) {
    // A wider torus than the dynamics with a field: the packet must not wrap around.
    let n = 2 * opt.n;
    let js = spreading_strengths(opt);
    let jobs: Vec<(Phase, f64, u64)> = [Phase::Para, Phase::Magnetic]
        .iter()
        .flat_map(|&p| js.iter().flat_map(move |&j| (0..opt.configurations as u64).map(move |c| (p, j, c))))
        .collect();
    let ev = Evolution { dt: 0.1, steps: SPREADING_STEPS, every: 10, fermi: 0.0, densities: false };
    let parts: Vec<String> = jobs
        .par_iter()
        .map(|&(phase, j, c)| {
            part(dir, opt, "spreading", &format!("{}-J{j}-c{c}", name(phase)), || {
                let torus = Torus::new(&graphene(impurities(n, phase, j, opt.conc, 5 + 1000 * c)), n);
                // The B, spin-up orbital (index 2) of the central cell, projected on the valence band.
                let start = torus.localized([n / 2, n / 2], 2, Some((-10.0, 0.0)));
                let r = torus.evolve(&[start], &ev);
                r.times.iter().zip(&r.width).map(|(t, w)| format!("{t},{w:e}\n")).collect()
            })
        })
        .collect();
    // The width averaged over the configurations, then w ~ t^β.
    let mut csv = String::from("phase,j_i,t,width\n");
    let mut fits = String::from("phase,j_i,beta,t_from,t_to\n");
    for phase in [Phase::Para, Phase::Magnetic] {
        for &j in &js {
            let mut mean: Vec<(f64, f64)> = Vec::new();
            for (&(p, jj, _), text) in jobs.iter().zip(&parts) {
                if p != phase || jj != j {
                    continue;
                }
                for (k, line) in text.lines().enumerate() {
                    let (t, w) = line.split_once(',').expect("t,width");
                    let (t, w): (f64, f64) = (t.parse().expect("t"), w.parse().expect("width"));
                    if k == mean.len() {
                        mean.push((t, 0.0));
                    }
                    mean[k].1 += w / opt.configurations as f64;
                }
            }
            for (t, w) in &mean {
                writeln!(csv, "{},{j},{t},{w:e}", name(phase)).unwrap();
            }
            // w ~ t^β on t ≥ 10, before the packet feels the size of the torus (w < n/6).
            let points: Vec<(f64, f64)> = mean
                .iter()
                .filter(|(t, w)| *t >= 10.0 && *w < n as f64 / 6.0)
                .map(|(t, w)| (t.ln(), w.ln()))
                .collect();
            if points.len() > 3 {
                let m = points.len() as f64;
                let (sx, sy) = points.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
                let (sxx, sxy) = points.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0 * p.0, a.1 + p.0 * p.1));
                let beta = (m * sxy - sx * sy) / (m * sxx - sx * sx);
                let (from, to) = (points[0].0.exp(), points[points.len() - 1].0.exp());
                writeln!(fits, "{},{j},{beta},{from},{to}", name(phase)).unwrap();
            }
        }
    }
    write(dir, "spreading.csv", &csv);
    write(dir, "spreading_beta.csv", &fits);
}

/// Torus of `ldos` and sites of its maps (cells of the patch).
const LDOS_N: usize = 64;
const LDOS_PATCH: usize = 32;

fn ldos(dir: &str, opt: &Options) {
    let n = LDOS_N;
    let moments = opt.moments.unwrap_or(MOMENTS);
    let lattice = Lattice::honeycomb(1.0 / 3f64.sqrt(), 2);
    let ni = n as i64;
    // Histogram: ε = 0.2 on a sample of sites of each configuration.
    let mut csv = String::from("j_i,configuration,site,ldos\n");
    for j in [0.1, 0.5, 1.0, 1.5] {
        for c in 0..opt.ldos_configurations as u64 {
            csv += &part(dir, opt, "ldos", &format!("J{j}-c{c}"), || {
                let model = graphene(impurities(n, Phase::Magnetic, j, opt.conc, 11 + 1000 * c));
                let region = Region::cuboid(model.lattice(), [n, n, 0]);
                let spectrum = Spectrum::periodic(&model, &region, &[[ni, 0, 0], [0, ni, 0]])
                    .moments(moments)
                    .bounds(-3.0 - 2.0 * j, 3.0 + 2.0 * j);
                let mut rng = Xoshiro256::seed_from_u64(13 + c);
                let sites: Vec<usize> =
                    (0..opt.ldos_sites).map(|_| rng.integer(1, region.len()) - 1).collect();
                let mut out = String::new();
                for (k, l) in spectrum.ldos(&sites, &[0.2]).iter().enumerate() {
                    writeln!(out, "{j},{c},{},{:e}", sites[k], l[0]).unwrap();
                }
                println!("ldos: J_I = {j}, configuration {c}");
                out
            });
        }
    }
    write(dir, "ldos_histogram.csv", &csv);
    // Maps: J_I = 1, ε = ±0.1, on a patch of the torus.
    let terms = impurities(n, Phase::Magnetic, 1.0, opt.conc, 11);
    let imp: Vec<(Site, f64)> = terms.iter().map(|(s, m)| (*s, m[(0, 0)].re)).collect();
    let model = graphene(terms);
    let region = Region::cuboid(model.lattice(), [n, n, 0]);
    let spectrum =
        Spectrum::periodic(&model, &region, &[[ni, 0, 0], [0, ni, 0]]).moments(moments).bounds(-5.0, 5.0);
    let inside = |s: &Site| s.cell[0] < LDOS_PATCH as i64 && s.cell[1] < LDOS_PATCH as i64;
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

/// One recorded value of `noise`: case, estimator, time, current and pairs.
type NoiseRow<'a> = (Phase, f64, &'a str, f64, f64, f64);

/// The statistical error of one sample of the two estimators of the current and the pairs, at
/// E = 0.02, each sample on a new disorder configuration: the thesis's localized spin-up hole
/// (an A↑ orbital projected on the valence band, one per configuration) and one random-phase
/// vector over the spin-up valence band (this example). Both estimate the same averages; the
/// ratio of their variances is the number of thesis configurations that one vector is worth.
fn noise(dir: &str, opt: &Options) {
    let cases = [(Phase::Para, 0.2), (Phase::Magnetic, 0.2), (Phase::Magnetic, 1.0)];
    let ev = Evolution { dt: 0.1, steps: 1000, every: 50, fermi: 0.0, densities: false };
    let jobs: Vec<(Phase, f64, u64)> =
        cases.iter().flat_map(|&(p, j)| (0..opt.samples as u64).map(move |s| (p, j, s))).collect();
    let rows: Vec<Vec<NoiseRow>> = jobs
        .par_iter()
        .map(|&(phase, j, s)| {
            let model = graphene(impurities(opt.n, phase, j, opt.conc, 1000 + s));
            let torus = Torus::new(&model, opt.n).with_force([0.0, 0.02, 0.0]).with_sector(&SPIN_UP);
            let hole = torus.localized([opt.n / 2, opt.n / 2], 0, Some((-10.0, 0.0)));
            let vector = torus.filled(0.0, 1, 2000 + s).remove(0);
            // Per state, then per cell and both spins for every state of the band (the figures).
            let scale = 2.0 * CELL * (opt.n * opt.n) as f64;
            let mut out = Vec::new();
            for (estimator, start) in [("hole", hole), ("vector", vector)] {
                let r = torus.evolve(&[start], &ev);
                for k in 0..r.times.len() {
                    let w = scale / r.norm[k];
                    out.push((phase, j, estimator, r.times[k], r.current[k][1] * w, r.excited[k] * w));
                }
            }
            println!("noise: {} J_I = {j}, configuration {s}", name(phase));
            out
        })
        .collect();
    let mut csv = String::from("phase,j_i,estimator,configuration,t,j,pairs\n");
    for (&(_, _, s), sample) in jobs.iter().zip(&rows) {
        for (phase, j, est, t, c, n) in sample {
            writeln!(csv, "{},{j},{est},{s},{t},{c:e},{n:e}", name(*phase)).unwrap();
        }
    }
    write(dir, "noise.csv", &csv);
    // Mean and standard deviation of one sample, per case, estimator and time.
    let mut summary = String::from("phase,j_i,estimator,t,j_mean,j_std,pairs_mean,pairs_std\n");
    for &(phase, j) in &cases {
        for est in ["hole", "vector"] {
            let pick: Vec<&NoiseRow> =
                rows.iter().flatten().filter(|r| r.0 == phase && r.1 == j && r.2 == est).collect();
            let mut times: Vec<f64> = pick.iter().map(|r| r.3).collect();
            times.dedup();
            for t in times {
                let at: Vec<(f64, f64)> = pick.iter().filter(|r| r.3 == t).map(|r| (r.4, r.5)).collect();
                let m = at.len() as f64;
                let (mj, mn) = at.iter().fold((0.0, 0.0), |a, x| (a.0 + x.0 / m, a.1 + x.1 / m));
                let (vj, vn) =
                    at.iter().fold((0.0, 0.0), |a, x| (a.0 + (x.0 - mj).powi(2), a.1 + (x.1 - mn).powi(2)));
                let (sj, sn) = ((vj / (m - 1.0)).sqrt(), (vn / (m - 1.0)).sqrt());
                writeln!(summary, "{},{j},{est},{t},{mj:e},{sj:e},{mn:e},{sn:e}", name(phase)).unwrap();
            }
        }
    }
    write(dir, "noise_summary.csv", &summary);
}

/// Seconds per step of one evolution on an `n × n` torus with disorder, the machine busy with
/// as many evolutions as it has threads (as the figures run them).
fn step_cost(opt: &Options, n: usize) -> f64 {
    let threads = rayon::current_num_threads();
    let model = graphene(impurities(n, Phase::Magnetic, 0.2, opt.conc, 1));
    let torus = Torus::new(&model, n).with_force([0.0, 0.02, 0.0]).with_sector(&SPIN_UP);
    let starts = torus.filled(0.0, threads, 1);
    let steps = 20;
    let ev = Evolution { dt: 0.1, steps, every: steps, fermi: 0.0, densities: false };
    let t = Instant::now();
    starts.par_iter().for_each(|s| {
        torus.evolve(std::slice::from_ref(s), &ev);
    });
    t.elapsed().as_secs_f64() / (threads * steps) as f64
}

/// Seconds per local density of states (one orbital, the KPM recursion) on the torus of `ldos`.
fn kpm_cost(opt: &Options) -> f64 {
    let model = graphene(impurities(LDOS_N, Phase::Magnetic, 1.0, opt.conc, 11));
    let region = Region::cuboid(model.lattice(), [LDOS_N, LDOS_N, 0]);
    let ni = LDOS_N as i64;
    let spectrum = Spectrum::periodic(&model, &region, &[[ni, 0, 0], [0, ni, 0]])
        .moments(opt.moments.unwrap_or(MOMENTS))
        .bounds(-5.0, 5.0);
    let sites: Vec<usize> = (0..16).collect();
    let t = Instant::now();
    spectrum.orbital_ldos(&sites, &[0.2]);
    t.elapsed().as_secs_f64() / 64.0
}

/// Durations of the figures on this machine with `opt`, and of the thesis protocol.
fn estimate(opt: &Options) {
    println!(
        "{} threads; measuring the dynamics on {}² and {}² tori and the KPM…",
        rayon::current_num_threads(),
        opt.n,
        2 * opt.n
    );
    let (c1, c2, kpm) = (step_cost(opt, opt.n), step_cost(opt, 2 * opt.n), kpm_cost(opt));
    let hours = |s: f64| s / 3600.0;
    let clean = fields(opt).len() as f64 * 1000.0 * c1;
    let disorder = (disorder_jobs(opt).len() * opt.vectors) as f64 * 1000.0 * c1;
    let spreading =
        (2 * spreading_strengths(opt).len() * opt.configurations) as f64 * SPREADING_STEPS as f64 * c2;
    let orbitals = 4.0 * (4 * opt.ldos_sites * opt.ldos_configurations + 2 * LDOS_PATCH * LDOS_PATCH) as f64;
    let ldos = orbitals * kpm;
    // The densities of states: 8 cases × 4 vectors on the n × n torus.
    let dos = 32.0 * kpm * (opt.n * opt.n) as f64 / (LDOS_N * LDOS_N) as f64;
    println!("figure      duration (h)");
    for (figure, s) in
        [("clean", clean), ("dos", dos), ("ldos", ldos), ("disorder", disorder), ("spreading", spreading)]
    {
        println!("{figure:<11} {:>8.2}", hours(s));
    }
    let total = clean + dos + ldos + disorder + spreading;
    println!("all         {:>8.2}", hours(total));
    // The thesis: 40 cases of Fig. 7 (10 fields para and magnetic, 20 more J_I), 8192 holes each,
    // on its sheet of 32768 cells (its grid carried a second, empty copy: twice that).
    let thesis = 40.0 * THESIS_CONFIGURATIONS * 1000.0 * c1 * THESIS_CELLS / (opt.n * opt.n) as f64;
    println!(
        "the thesis protocol for Fig. 7 alone (8192 configurations × 40 cases): {:.0} h ({:.1} days)",
        hours(thesis),
        hours(thesis) / 24.0
    );
}

/// `x` to two decimals (the disorder strengths of the scans).
fn round(x: f64) -> f64 {
    (100.0 * x).round() / 100.0
}

/// Writes `content` to `dir/file` through a temporary file, so that an interrupted run leaves
/// no partial table.
fn write(dir: &str, file: &str, content: &str) {
    let path = Path::new(dir).join(file);
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, content).expect("write");
    std::fs::rename(&tmp, &path).expect("rename");
    println!("{}", path.display());
}

/// One job of a long figure, kept in `dir/parts/figure/key.csv` so that an interrupted run
/// resumes: computed by `compute` unless already there (or with `--force`).
fn part(dir: &str, opt: &Options, figure: &str, key: &str, compute: impl FnOnce() -> String) -> String {
    let folder: PathBuf = Path::new(dir).join("parts").join(figure);
    let path = folder.join(format!("{key}.csv"));
    if !opt.force
        && let Ok(text) = std::fs::read_to_string(&path)
    {
        return text;
    }
    let text = compute();
    std::fs::create_dir_all(&folder).expect("parts directory");
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, &text).expect("write");
    std::fs::rename(&tmp, &path).expect("rename");
    text
}

/// The options of the results in `dir`: kept on the first run, and a later run with other
/// options stops rather than mix results (`--force` replaces them).
fn check_options(dir: &str, opt: &Options) {
    let signature = format!(
        "n = {}\nconc = {}\nvectors = {}\nmoments = {}\nconfigurations = {}\nldos = {} sites × {} configurations\npaper = {}\nground_state = {}\n",
        opt.n,
        opt.conc,
        opt.vectors,
        opt.moments.unwrap_or(MOMENTS),
        opt.configurations,
        opt.ldos_sites,
        opt.ldos_configurations,
        opt.paper,
        opt.ground_state
    );
    let path = Path::new(dir).join("options.txt");
    match std::fs::read_to_string(&path) {
        Ok(old) if old != signature && !opt.force => panic!(
            "{} holds results with other options:\n{old}\nuse another directory, the same options, or --force",
            dir
        ),
        _ => std::fs::write(&path, signature).expect("options"),
    }
}

/// Every figure, skipping those already in `dir`, then the plots.
fn all(dir: &str, opt: &Options) {
    // Keeps a Mac awake as long as this process runs.
    #[cfg(target_os = "macos")]
    let _awake =
        std::process::Command::new("caffeinate").args(["-i", "-w", &std::process::id().to_string()]).spawn();
    type Figure = (&'static str, fn(&str, &Options), &'static [&'static str]);
    let figures: [Figure; 5] = [
        ("clean", clean, &["clean.csv"]),
        ("dos", dos, &["dos.csv"]),
        ("ldos", ldos, &["ldos_histogram.csv", "ldos_map.csv", "ldos_impurities.csv"]),
        ("disorder", disorder, &["disorder.csv"]),
        ("spreading", spreading, &["spreading.csv", "spreading_beta.csv"]),
    ];
    let start = Instant::now();
    for (figure, run, outputs) in figures {
        let line = if !opt.force && outputs.iter().all(|f| Path::new(dir).join(f).exists()) {
            format!("{figure}: already done")
        } else {
            let t = Instant::now();
            run(dir, opt);
            format!("{figure}: {:.0} s", t.elapsed().as_secs_f64())
        };
        println!("[{:.0} s] {line}", start.elapsed().as_secs_f64());
        let previous = std::fs::read_to_string(Path::new(dir).join("campaign.log")).unwrap_or_default();
        std::fs::write(Path::new(dir).join("campaign.log"), previous + &format!("{line}\n")).expect("log");
    }
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples").join("plot_graphene_field.py");
    match std::process::Command::new("python3").arg(&script).arg(dir).status() {
        Ok(s) if s.success() => {}
        _ => println!("plots: python3 {} {dir} (needs numpy and matplotlib)", script.display()),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: graphene_field estimate [OPTIONS]\n       graphene_field (all|clean|dos|ldos|disorder|spreading|noise) OUT_DIR [OPTIONS]\noptions: --paper --ground-state --force --n N --conc C --vectors R --moments M --configurations C --ldos-sites S --ldos-configurations C --samples S";
    let command = args.first().map(String::as_str).unwrap_or_else(|| panic!("{usage}"));
    let paper = args.iter().any(|a| a == "--paper");
    let value =
        |flag: &str| args.iter().position(|a| a == flag).map(|k| args.get(k + 1).expect(usage).as_str());
    let number = |flag: &str, default: usize| value(flag).map_or(default, |v| v.parse().expect(usage));
    let opt = Options {
        n: number("--n", if paper { 256 } else { 128 }),
        conc: value("--conc").map_or(0.4, |v| v.parse().expect("--conc C")),
        vectors: number("--vectors", if paper { 64 } else { 4 }),
        moments: value("--moments").map(|v| v.parse().expect("--moments M")),
        configurations: number("--configurations", if paper { 4 } else { 1 }),
        ldos_sites: number("--ldos-sites", if paper { 2048 } else { 512 }),
        ldos_configurations: number("--ldos-configurations", if paper { 32 } else { 1 }),
        samples: number("--samples", 16),
        paper,
        ground_state: args.iter().any(|a| a == "--ground-state"),
        force: args.iter().any(|a| a == "--force"),
    };
    if command == "estimate" {
        estimate(&opt);
        return;
    }
    let dir = args.get(1).filter(|d| !d.starts_with("--")).unwrap_or_else(|| panic!("{usage}"));
    std::fs::create_dir_all(dir).expect("output directory");
    if command != "noise" {
        check_options(dir, &opt);
    }
    let t = Instant::now();
    match command {
        "clean" => clean(dir, &opt),
        "dos" => dos(dir, &opt),
        "spreading" => spreading(dir, &opt),
        "disorder" => disorder(dir, &opt),
        "ldos" => ldos(dir, &opt),
        "noise" => noise(dir, &opt),
        "all" => all(dir, &opt),
        _ => panic!("{usage}"),
    }
    println!("done in {:.1?}", t.elapsed());
}
