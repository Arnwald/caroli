//! The input files of `caroli` against the library: each calculation on a small system, compared
//! with the direct API call or with the physics; the example inputs build; the errors name what
//! is wrong.

use caroli::prelude::*;
use caroli_cli::{describe, parse, run_all};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Runs `text` in a fresh directory and returns the rows of each table (header first).
fn run(text: &str) -> Vec<Vec<Vec<String>>> {
    let input = parse(text).unwrap_or_else(|e| panic!("parse: {e}"));
    let dir = std::env::temp_dir().join(format!("caroli-cli-test-{}-{:x}", std::process::id(), hash(text)));
    run_all(&input, &dir, &BTreeMap::new(), None).unwrap_or_else(|e| panic!("run: {e}"));
    let mut tables = Vec::new();
    for k in 0..input.runs.len() {
        let name = caroli_cli::output_name(k, &input.runs[k]);
        let text = std::fs::read_to_string(dir.join(name)).expect("table");
        tables.push(text.lines().map(|l| l.split(',').map(String::from).collect()).collect());
    }
    std::fs::remove_dir_all(&dir).ok();
    tables
}

fn hash(s: &str) -> u64 {
    s.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

/// Column `name` of a table as numbers.
fn col(table: &[Vec<String>], name: &str) -> Vec<f64> {
    let k = table[0]
        .iter()
        .position(|h| h == name)
        .unwrap_or_else(|| panic!("no column {name} in {:?}", table[0]));
    table[1..].iter().map(|r| r[k].parse().expect("number")).collect()
}

fn error(text: &str) -> String {
    match parse(text) {
        Err(e) => e.to_string(),
        Ok(input) => {
            let dir = std::env::temp_dir().join(format!("caroli-cli-error-{:x}", hash(text)));
            let e = run_all(&input, &dir, &BTreeMap::new(), None)
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default();
            std::fs::remove_dir_all(&dir).ok();
            e
        }
    }
}

/// Two-terminal transmission of a clean BHZ ribbon, as the library computes it; the faces in the
/// other order go through the multi-terminal solver and give the same numbers.
#[test]
fn transmission_matches_the_library() {
    let t = run(r#"
        [model]
        preset = "bhz"
        [system]
        size = [16, 10]
        [[run]]
        type = "transmission"
        energies = [0.0, 0.05, 0.25]
        [[run]]
        type = "transmission"
        energies = [0.0, 0.05, 0.25]
        sectors = [0, 0, 1, 1]
    "#);
    let model = bhz(BhzParams::default());
    let region = Region::cuboid(model.lattice(), [16, 10, 0]);
    let direct = NestedTwoTerminal::from_model(&model, &model, &region, 0);
    for (e, cli) in [0.0, 0.05, 0.25].iter().zip(col(&t[0], "T")) {
        assert!((cli - direct.transmission(*e).unwrap()).abs() < 1e-10, "E = {e}");
    }
    assert!((col(&t[0], "T")[0] - 2.0).abs() < 1e-8);
    // Spin sectors: no spin flip, and the spin-conserving parts add up to T.
    let (up, down, flip) = (col(&t[1], "T_1.0_0.0"), col(&t[1], "T_1.1_0.1"), col(&t[1], "T_1.0_0.1"));
    for k in 0..3 {
        assert!((up[k] + down[k] - col(&t[0], "T")[k]).abs() < 1e-8 && flip[k].abs() < 1e-10);
    }
    let reversed = run(r#"
        [model]
        preset = "bhz"
        [system]
        size = [16, 10]
        leads = ["+x", "-x"]
        [[run]]
        type = "transmission"
        energies = [0.0, 0.05, 0.25]
    "#);
    for (a, b) in col(&reversed[0], "T_1_0").iter().zip(col(&t[0], "T")) {
        assert!((a - b).abs() < 1e-8);
    }
}

/// A scan with a parameter that depends on the scanned one, and disorder configurations: one row
/// per (scan point, configuration, energy); the Anderson disorder lowers the conductance.
#[test]
fn scans_and_configurations() {
    let t = run(r#"
        [parameters]
        W = 0.0
        half = "W/2"
        [model]
        preset = "square"
        [[disorder]]
        type = "anderson"
        strength = "2*half"
        [system]
        size = [12, 8]
        [[run]]
        type = "transmission"
        energies = { from = -1.0, to = 1.0, step = 0.5 }
        scan = { W = [0.0, 3.0] }
        configurations = 2
    "#);
    assert_eq!(t[0][0], ["W", "config", "energy", "T"]);
    assert_eq!(t[0].len() - 1, 2 * 2 * 5);
    let (w, g) = (col(&t[0], "W"), col(&t[0], "T"));
    let mean = |x: f64| {
        let v: Vec<f64> = w.iter().zip(&g).filter(|(a, _)| **a == x).map(|(_, b)| *b).collect();
        v.iter().sum::<f64>() / v.len() as f64
    };
    assert!(mean(3.0) < mean(0.0) - 0.5, "{} vs {}", mean(3.0), mean(0.0));
    // Clean: integer conductance away from the subband edges, the same for both configurations.
    assert!(g[..10].iter().all(|x| (x - x.round()).abs() < 1e-6), "{:?}", &g[..10]);
}

/// Ribbon bands and the dense spectrum against the library.
#[test]
fn bands_and_spectrum() {
    let t = run(r#"
        [model]
        preset = "graphene"
        [system]
        size = [3, 4]
        [[run]]
        type = "bands"
        k = [0.0, 1.0]
        [[run]]
        type = "spectrum"
        [[run]]
        type = "bands"
        path = [[0, 0], ["4*pi/3", "2*pi/3"]]
        points = 3
    "#);
    let model = Model::new(Lattice::honeycomb(1.0, 1))
        .hopping(0, 1, [0, 0, 0], -1.0)
        .hopping(0, 1, [0, -1, 0], -1.0)
        .hopping(0, 1, [1, -1, 0], -1.0);
    let region = Region::cuboid(model.lattice(), [3, 4, 0]);
    let slice = &region.slices(0)[0];
    let e = model
        .bloch_hamiltonian(slice, &[[1, 0, 0]], &[1.0])
        .self_adjoint_eigenvalues(faer::Side::Lower)
        .unwrap();
    let cli: Vec<f64> = col(&t[0], "energy")[8..].to_vec();
    assert!(e.iter().zip(&cli).all(|(a, b)| (a - b).abs() < 1e-12) && cli.len() == 8);
    let exact = model.hamiltonian(&region).self_adjoint_eigenvalues(faer::Side::Lower).unwrap();
    assert!(exact.iter().zip(col(&t[1], "energy")).all(|(a, b)| (a - b).abs() < 1e-12));
    // The Dirac point K at the end of the path (Bloch phases of the honeycomb vectors): two zero
    // energies.
    let last: Vec<f64> = col(&t[2], "energy").into_iter().rev().take(2).collect();
    assert!(last.iter().all(|e| e.abs() < 1e-12), "{last:?}");
}

/// KPM densities of states hold every orbital; the projections on the two spins add up.
#[test]
fn densities_of_states() {
    let t = run(r#"
        [model]
        preset = "square"
        spin = true
        [[disorder]]
        type = "magnetic"
        concentration = 0.2
        exchange = 1.0
        [system]
        size = [24, 24]
        periodic = true
        [[run]]
        type = "dos"
        energies = { from = -6.0, to = 6.0, count = 1201 }
        moments = 128
        vectors = 4
        projections = { up = "(s0 + sz)/2", down = "(s0 - sz)/2" }
    "#);
    let (e, rho, up, down) = (col(&t[0], "energy"), col(&t[0], "dos"), col(&t[0], "up"), col(&t[0], "down"));
    let integral: f64 =
        rho.windows(2).zip(e.windows(2)).map(|(r, x)| 0.5 * (r[0] + r[1]) * (x[1] - x[0])).sum();
    assert!((integral - 2.0 * 576.0).abs() < 0.01 * 1152.0, "∫ρ = {integral}");
    let scale = rho.iter().fold(0.0f64, |m, x| m.max(*x));
    for k in 0..rho.len() {
        assert!((up[k] + down[k] - rho[k]).abs() < 0.05 * scale, "E = {}", e[k]);
    }
}

/// The Born approximation of the input against the library's.
#[test]
fn scba_matches_the_library() {
    let t = run(r#"
        [model]
        preset = "bhz"
        [[disorder]]
        type = "anderson"
        strength = 1.0
        [[run]]
        type = "scba"
        energies = [0.05]
        k_grid = [32, 32]
        eta = 0.001
    "#);
    let scba = caroli::born::Scba::from_model(
        &bhz(BhzParams::default()),
        &[32, 32],
        caroli::born::Disorder::anderson(1.0, 4),
    );
    let sol = scba.self_energy(0.05, 0.001);
    assert!((col(&t[0], "sigma_re_0")[0] - sol.sigma[(0, 0)].re).abs() < 1e-9);
    assert!((col(&t[0], "sigma_im_1")[0] - sol.sigma[(1, 1)].im).abs() < 1e-9);
    assert!((col(&t[0], "dos")[0] - scba.dos(0.05, 0.001, &sol.sigma)).abs() < 1e-8);
}

/// A wave packet under a force: the Stark breathing of the width along the force.
#[test]
fn evolution_of_a_wave_packet() {
    let t = run(r#"
        [model]
        preset = "square"
        [system]
        size = [48, 48]
        [[run]]
        type = "evolution"
        force = [0.5, 0.0]
        dt = 0.01
        steps = 600
        every = 100
        initial = { type = "localized", cell = [24, 24] }
    "#);
    for (t, w) in col(&t[0], "t").iter().zip(col(&t[0], "width")) {
        let exact = (8.0 / 0.25 * (0.25 * t).sin().powi(2) + 2.0 * t * t).sqrt();
        assert!((w - exact).abs() < 1e-5 * exact.max(1.0), "t = {t}: {w} vs {exact}");
    }
}

/// The local run: currents, densities and an operator (the spin current of the helical edges).
#[test]
fn local_currents() {
    let t = run(r#"
        [model]
        preset = "bhz"
        [system]
        size = [8, 12]
        [[run]]
        type = "local"
        energies = 0.02
        operators = { sz = "kron(sz, s0)" }
    "#);
    let (y, jx, jsz) = (col(&t[0], "y"), col(&t[0], "jx"), col(&t[0], "jsz_x"));
    // Through a column the current is the transmission (2), and each edge carries one spin.
    let column: Vec<usize> =
        col(&t[0], "x").iter().enumerate().filter(|(_, x)| **x == 4.0).map(|(k, _)| k).collect();
    let total: f64 = column.iter().map(|&k| jx[k]).sum();
    // Bond currents per unit energy: through a cross-section they add up to T(f_L − f_R) = 2.
    assert!((total - 2.0).abs() < 1e-6, "current {total}");
    let bottom: f64 = column.iter().filter(|&&k| y[k] < 6.0).map(|&k| jsz[k]).sum();
    let top: f64 = column.iter().filter(|&&k| y[k] >= 6.0).map(|&k| jsz[k]).sum();
    assert!(bottom * top < 0.0, "opposite spin currents on the two edges: {bottom}, {top}");
}

/// Every example input reads and builds.
#[test]
fn example_inputs_build() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("inputs");
    let mut count = 0;
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "toml") {
            let input =
                parse(&std::fs::read_to_string(&path).unwrap()).unwrap_or_else(|e| panic!("{path:?}: {e}"));
            describe(&input, &BTreeMap::new()).unwrap_or_else(|e| panic!("{path:?}: {e}"));
            count += 1;
        }
    }
    assert!(count >= 5);
}

/// The errors say what is wrong.
#[test]
fn errors_name_the_problem() {
    let base = "[model]\npreset = \"square\"\n[system]\nsize = [4, 4]\n";
    let e = error(&format!("{base}[[run]]\ntype = \"transmission\"\nenergies = 0\nenergy = 1\n"));
    assert!(e.contains("energy"), "{e}");
    let e = error(&format!("{base}[[run]]\ntype = \"transmision\"\nenergies = 0\n"));
    assert!(e.contains("transmision"), "{e}");
    let e = error(
        "[model]\npreset = \"square\"\nt = \"q\"\n[system]\nsize = [4, 4]\n[[run]]\ntype = \"spectrum\"\n",
    );
    assert!(e.contains("unknown name `q`"), "{e}");
    let e = error("[model]\npreset = \"square\"\n[[run]]\ntype = \"spectrum\"\n");
    assert!(e.contains("size"), "{e}");
    let e = error(
        "[model]\npreset = \"bhz\"\nspin = true\n[system]\nsize = [4, 4]\n[[run]]\ntype = \"spectrum\"\n",
    );
    assert!(e.contains("spin"), "{e}");
    let e = error(&format!("{base}[[run]]\ntype = \"conductivity\"\nenergies = 0\n"));
    assert!(e.contains("periodic"), "{e}");
}
