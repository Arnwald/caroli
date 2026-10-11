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
