//! Graphene with classical magnetic impurities under a strong electric field, the second study
//! of the author's thesis (A. Demion and A. D. Verga, "Nonlinear electric transport in graphene
//! with magnetic disorder", Phys. Rev. B 90, 085412 (2014)).
//!
//! Spinful graphene (hopping 1, lattice constant 1) on an `n × n` torus, with impurities on a
//! fraction of the A sites and as many B sites, coupled to the electron spin by `V_imp n·σ`:
//! paramagnetic (isotropic moments) or antiferromagnetic (A moments along +z, B along −z, which
//! opens a gap). The field is a constant force along y, `A(t) = −F t`.
//!
//! ```text
//! cargo run --release -p caroli --example graphene_field -- FIGURE OUT_DIR [--n N]
//! python3 crates/caroli/examples/plot_graphene_field.py OUT_DIR
//! ```
//!
//! Figures (CSV files in `OUT_DIR`):
//! - `current`: current density and density of carriers above the Dirac point of the
//!   instantaneous clean bands against time, for the clean, paramagnetic and antiferromagnetic
//!   samples and several fields; the pairs created by the field are its growth. The Fermi sea at
//!   half filling is a stochastic trace: random-phase superpositions of the occupied states
//!   (clean bands; with impurities, the ground state of the disordered Hamiltonian, by a
//!   Chebyshev projection). The plot subtracts the current of the same vectors evolved without
//!   field: it removes the noise of the trace at early times, before the two evolutions
//!   decorrelate; later, only more vectors or a larger torus reduce it.
//! - `spreading`: width of a wave packet, the B orbital of one site projected on the valence
//!   band, without field: ballistic in clean graphene, slower with impurities.
//! - `dos`: densities of states (kernel polynomial method, per unit cell).
//! - `conductivity`: Kubo–Bastin `σ_xx(μ)` (e²/h) on a torus.
//!
//! Parameters as in the thesis runs: `V_imp = 0.5`, 10 % of the sites of each sublattice,
//! `dt = 0.1`, fields up to 0.02.

use caroli::dynamics::{Evolution, Torus};
use caroli::kpm::Spectrum;
use caroli::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt::Write as _;

const V_IMP: f64 = 0.5;
const CONC: f64 = 0.1;

#[derive(Clone, Copy, PartialEq)]
enum Phase {
    Clean,
    Para,
    Af,
}

const PHASES: [(Phase, &str); 3] = [(Phase::Clean, "clean"), (Phase::Para, "para"), (Phase::Af, "af")];

/// Spinful graphene with lattice constant 1 (nearest neighbours at 1/√3) and the impurities of
/// `phase` on an `n × n` torus.
fn graphene(n: usize, phase: Phase, seed: u64) -> Model {
    let t = combine(&[(-1.0, &s0())]);
    let model = Model::new(Lattice::honeycomb(1.0 / 3f64.sqrt(), 2))
        .hopping(0, 1, [0, 0, 0], &t)
        .hopping(0, 1, [0, -1, 0], &t)
        .hopping(0, 1, [1, -1, 0], &t);
    if phase == Phase::Clean {
        return model;
    }
    let mut rng = Xoshiro256::seed_from_u64(seed);
    let count = (CONC * (n * n) as f64) as usize;
    let mut terms: HashMap<Site, CMat> = HashMap::new();
    for sub in 0..2 {
        let cells: Vec<Site> =
            (0..n as i64).flat_map(|y| (0..n as i64).map(move |x| Site { cell: [x, y, 0], sub })).collect();
        let moments = if phase == Phase::Para { Moments::Isotropic } else { Moments::Polarized { mz: 1.0 } };
        let imp = MagneticImpurities::generate(&mut rng, &cells, count, moments);
        let sign = if phase == Phase::Af && sub == 1 { -1.0 } else { 1.0 };
        for (site, m) in imp.sites.iter().zip(&imp.moments) {
            let j = sign * V_IMP;
            terms.insert(*site, combine(&[(j * m[0], &sx()), (j * m[1], &sy()), (j * m[2], &sz())]));
        }
    }
    model.with_onsite_term(move |s: &Site| terms.get(s).cloned())
}

struct Options {
    n: usize,
}

fn current(dir: &str, opt: &Options) {
    // F = 0 evolves the same random vectors in the same disorder: the plot subtracts its current
    // (a control variate for the noise of the stochastic trace, effective at early times).
    let fields = [0.0, 0.005, 0.01, 0.02];
    let (dt, steps, every) = (0.1, 2000, 10);
    let jobs: Vec<(Phase, &str, f64)> =
        PHASES.iter().flat_map(|&(p, name)| fields.iter().map(move |&f| (p, name, f))).collect();
    let rows: Vec<String> = jobs
        .par_iter()
        .map(|&(phase, name, f)| {
            let torus = Torus::new(&graphene(opt.n, phase, 3), opt.n).with_force([0.0, f, 0.0]);
            // Clean: one random vector is exact (one occupied band state per k and spin pair);
            // disordered: the ground state of the disordered Hamiltonian, several vectors.
            let starts = if phase == Phase::Clean {
                torus.filled(0.0, 1, 11)
            } else {
                torus.fermi_sea(0.0, 4, 11, 1024)
            };
            let ev = Evolution { dt, steps, every, fermi: 0.0, densities: false };
            let r = torus.evolve(&starts, &ev);
            let mut out = String::new();
            for k in 0..r.times.len() {
                writeln!(out, "{name},{f},{},{:e},{:e}", r.times[k], r.current[k][1], r.excited[k]).unwrap();
            }
            println!("current: {name}, F = {f}");
            out
        })
        .collect();
    write(dir, "current.csv", &(String::from("phase,field,t,j,pairs\n") + &rows.concat()));
}

fn spreading(dir: &str, opt: &Options) {
    let (dt, steps, every) = (0.1, 3000, 20);
    let rows: Vec<String> = PHASES
        .par_iter()
        .map(|&(phase, name)| {
            let torus = Torus::new(&graphene(opt.n, phase, 5), opt.n);
            // The B orbital (spin up) of the central site, projected on the valence band.
            let start = torus.localized([opt.n / 2, opt.n / 2], 2, Some((-10.0, 0.0)));
            let ev = Evolution { dt, steps, every, fermi: 0.0, densities: false };
            let r = torus.evolve(&[start], &ev);
            let mut out = String::new();
            for k in 0..r.times.len() {
                writeln!(out, "{name},{},{:e}", r.times[k], r.width[k]).unwrap();
            }
            println!("spreading: {name}");
            out
        })
        .collect();
    write(dir, "spreading.csv", &(String::from("phase,t,width\n") + &rows.concat()));
}

fn dos(dir: &str, opt: &Options) {
    let energies: Vec<f64> = (0..=600).map(|k| -3.3 + 6.6 * k as f64 / 600.0).collect();
    let mut csv = String::from("phase,energy,dos\n");
    for (phase, name) in PHASES {
        let model = graphene(opt.n, phase, 7);
        let region = Region::cuboid(model.lattice(), [opt.n, opt.n, 0]);
        let n = opt.n as i64;
        let spectrum = Spectrum::periodic(&model, &region, &[[n, 0, 0], [0, n, 0]]).moments(1024).vectors(4);
        let cells = (opt.n * opt.n) as f64;
        for (e, rho) in energies.iter().zip(spectrum.dos(&energies)) {
            writeln!(csv, "{name},{e},{:e}", rho / cells).unwrap();
        }
        println!("dos: {name}");
    }
    write(dir, "dos.csv", &csv);
}

fn conductivity(dir: &str, opt: &Options) {
    let n = opt.n.min(64);
    let mus: Vec<f64> = (0..=60).map(|k| -1.5 + 3.0 * k as f64 / 60.0).collect();
    let mut csv = String::from("phase,mu,sigma_xx\n");
    for (phase, name) in PHASES {
        let model = graphene(n, phase, 9);
        let region = Region::cuboid(model.lattice(), [n, n, 0]);
        let ni = n as i64;
        let spectrum = Spectrum::periodic(&model, &region, &[[ni, 0, 0], [0, ni, 0]]).moments(384).vectors(4);
        let x = [1.0, 0.0, 0.0];
        for (mu, s) in mus.iter().zip(spectrum.conductivity(x, x, &mus, 0.0)) {
            writeln!(csv, "{name},{mu},{s:e}").unwrap();
        }
        println!("conductivity: {name}");
    }
    write(dir, "conductivity.csv", &csv);
}

fn write(dir: &str, file: &str, content: &str) {
    let path = std::path::Path::new(dir).join(file);
    std::fs::write(&path, content).expect("write");
    println!("{}", path.display());
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: graphene_field (current|spreading|dos|conductivity|all) OUT_DIR [--n N]";
    let (figure, dir) = match (args.first(), args.get(1)) {
        (Some(f), Some(d)) => (f.as_str(), d.as_str()),
        _ => panic!("{usage}"),
    };
    let mut opt = Options { n: 128 };
    if let Some(k) = args.iter().position(|a| a == "--n") {
        opt.n = args.get(k + 1).and_then(|v| v.parse().ok()).expect("--n N");
    }
    std::fs::create_dir_all(dir).expect("output directory");
    let t = std::time::Instant::now();
    match figure {
        "current" => current(dir, &opt),
        "spreading" => spreading(dir, &opt),
        "dos" => dos(dir, &opt),
        "conductivity" => conductivity(dir, &opt),
        "all" => {
            current(dir, &opt);
            spreading(dir, &opt);
            dos(dir, &opt);
            conductivity(dir, &opt);
        }
        _ => panic!("{usage}"),
    }
    println!("done in {:.1?}", t.elapsed());
}
