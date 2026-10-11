//! KPM densities of states against `dos-kpm-magnetic` (TI, same-g and opposite-g) and graphene
//! `dos-kpm`, with the thesis random vectors and impurities (gfortran-compatible stream).

use caroli_validation::golden_f64;
use caroli_validation::thesis::Kpm;

fn check(case: &str, out: Vec<f64>) {
    let golden = &golden_f64(case, "dos_00.dat")[0];
    assert_eq!(out.len(), golden.len(), "{case}: record length");
    let scale = golden.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let d = out.iter().zip(golden).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("{case}: max |Δ| = {d:.1e} (max |value| {scale:.2})");
    assert!(d < 1e-9 * scale, "{case}: {d:e}");
}

#[test]
fn ti_same_g() {
    let p = Kpm { n: 16, m: 64, n_state: 2, imp_conc: 0.1, v_exc: 0.55, choice_spin: 1, opposite: false };
    check("ti-dos-kpm-same-g", p.ti());
}

#[test]
fn ti_opposite_g() {
    let p = Kpm { n: 16, m: 64, n_state: 2, imp_conc: 0.1, v_exc: 0.55, choice_spin: 1, opposite: true };
    check("ti-dos-kpm-opposite-g", p.ti());
}

#[test]
fn graphene_af() {
    let p = Kpm { n: 8, m: 64, n_state: 2, imp_conc: 0.4, v_exc: 1.0, choice_spin: 1, opposite: false };
    check("graphene-dos-kpm", p.graphene());
}
