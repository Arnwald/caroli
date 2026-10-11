//! Graphene split-operator dynamics against `current-pair-creation` (clean, AF and paramagnetic
//! impurities), with the thesis impurity draws (gfortran-compatible stream, fixed seed).

use caroli_validation::thesis::GrapheneCurrent;
use caroli_validation::{golden_dir, read_records};

/// A golden series written one value per record.
fn series(case: &str, file: &str) -> Vec<f64> {
    read_records(golden_dir(case).join(file))
        .unwrap()
        .iter()
        .map(|r| f64::from_le_bytes(r[..8].try_into().unwrap()))
        .collect()
}

fn check(case: &str, vimp: f64, conc: f64, samples: usize, spin: i32) {
    let p = GrapheneCurrent {
        n: 16,
        vimp,
        imp_conc: conc,
        elec_max: 0.02,
        nelec: 2,
        t_total: 2.0,
        samples,
        dt: 0.1,
        ndiag: 1,
        choice_spin: spin,
    };
    let out = p.run();
    for (k, (cur, np)) in out.iter().enumerate() {
        let (gc, gn) = (series(case, &format!("curr_0{k}.dat")), series(case, &format!("np_0{k}.dat")));
        let dc = cur.iter().zip(&gc).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let dn = np.iter().zip(&gn).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let sc = gc.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        println!("{case} field {k}: |Δ current| ≤ {dc:.1e} (max {sc:.1e}), |Δ pairs| ≤ {dn:.1e}");
        assert_eq!((cur.len(), np.len()), (gc.len(), gn.len()));
        assert!(dc < 1e-10 && dn < 1e-10, "{case} field {k}");
    }
}

#[test]
fn clean() {
    check("graphene-current-clean", 0.0, 0.4, 1, 1);
}

#[test]
fn antiferromagnetic() {
    check("graphene-current-af", 0.5, 0.1, 2, 1);
}

#[test]
fn paramagnetic() {
    check("graphene-current-para", 0.5, 0.1, 2, 0);
}

/// Wave-packet width program (adiabatic basis with Berry connection), paramagnetic impurities.
#[test]
fn wavepacket_width() {
    use caroli_validation::thesis::GrapheneWidth;
    let p = GrapheneWidth {
        n: 16,
        vimp: 0.5,
        imp_conc: 10.0,
        elec: 0.0,
        t_total: 2.0,
        samples: 1,
        dt: 0.1,
        ndiag: 1,
        choice_spin: 0,
    };
    let (cur, np, w) = p.run();
    for (name, ours, file) in
        [("current", &cur, "curr_00.dat"), ("pairs", &np, "np_00.dat"), ("width", &w, "w_00.dat")]
    {
        let g = series("graphene-width", file);
        assert_eq!(ours.len(), g.len(), "{name}: length");
        let d = ours.iter().zip(&g).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let sc = g.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        println!("graphene-width {name}: |Δ| ≤ {d:.1e} (max {sc:.1e})");
        assert!(d < 1e-10 * sc.max(1.0), "{name}: {d:e}");
    }
}

/// Wave-packet density program: current, pair creation and sublattice density maps.
#[test]
fn wavepacket_density() {
    use caroli_validation::golden_f64;
    use caroli_validation::thesis::GrapheneDensity;
    let p = GrapheneDensity {
        n: 16,
        vimp: 0.5,
        imp_conc: 10.0,
        elec: 0.01,
        t_total: 0.4,
        dt: 0.1,
        ndiag: 1,
        choice_spin: 1,
    };
    let out = p.run();
    let maxd = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    let (gc, gn) = (series("graphene-density", "curr_00.dat"), series("graphene-density", "np_00.dat"));
    assert_eq!((out.current.len(), out.pairs.len()), (gc.len(), gn.len()));
    let (dc, dn) = (maxd(&out.current, &gc), maxd(&out.pairs, &gn));
    let ga = golden_f64("graphene-density", "density1_00.dat");
    let gb = golden_f64("graphene-density", "density2_00.dat");
    assert_eq!((out.density_a.len(), out.density_b.len()), (ga.len(), gb.len()));
    let da = out.density_a.iter().zip(&ga).map(|(a, b)| maxd(a, b)).fold(0.0, f64::max);
    let db = out.density_b.iter().zip(&gb).map(|(a, b)| maxd(a, b)).fold(0.0, f64::max);
    println!(
        "graphene-density: |Δ current| ≤ {dc:.1e}, |Δ pairs| ≤ {dn:.1e}, |Δ ρ_A| ≤ {da:.1e}, |Δ ρ_B| ≤ {db:.1e}"
    );
    assert!(dc < 1e-12 && dn < 1e-12 && da < 1e-12 && db < 1e-12);
}
