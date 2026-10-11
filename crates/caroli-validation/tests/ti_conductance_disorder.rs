//! Two-terminal conductance with magnetic impurities against `conductance-2t`
//! (`ti-cond2t-same-g`, `-isotropic`, `-opposite-g`), with the thesis' own random
//! configurations.
//!
//! The thesis truncates the broadenings to their real part (`gamma_p = real(IM*(Σ − Σ†))`).
//! With that formula caroli reproduces the golden values to round-off; the exact Caroli
//! transmission differs from them by up to ~0.2 e²/h here.

use caroli_green::{SelfEnergyMethod, TwoTerminal};
use caroli_lattice::Region;
use caroli_model::presets::{BhzParams, GFactors, bhz, bhz_spin};
use caroli_validation::golden_f64;
use caroli_validation::thesis::{Conductance2t, thesis_transmission};

/// Golden axes (U0 values, energies) and cond(Nu, Ne), column-major.
fn golden(case: &str, nu: usize, ne: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let axes = &golden_f64(case, "axes_00.dat")[0];
    let cond = golden_f64(case, "cond_00.dat")[0].clone();
    assert_eq!((axes.len(), cond.len()), (nu + ne, nu * ne));
    (axes[..nu].to_vec(), axes[nu..].to_vec(), cond)
}

/// Case parameters (`reference/baseline/cases/ti-cond2t-*/ti.ini`): 16×10 ribbon, impurity
/// region 8 long, conc 0.3, J_s in {0.5, 1, 1.5}, 5 energies in [−0.05, 0.05], seed 12345.
fn check(case: &str, spin: i32, coupling: GFactors) {
    let (nx, ny, nx_imp) = (16usize, 10usize, 8usize);
    let (us, energies, cond) = golden(case, 3, 5);
    // One clean lead slice at each end, as in the thesis green_recur: sample slices are x = 1..=nx.
    let params = Conductance2t { nx, ny, nx_imp, conc: 0.3, spin, theta0_deg: 0.0, seed: 12345 };
    let imp = params.impurities(1);
    assert_eq!(imp.sites.len(), 24);
    let clean = bhz(BhzParams::default());
    let region = Region::cuboid(clean.lattice(), [nx + 2, ny, 0]);
    let mut worst_thesis: f64 = 0.0;
    let mut worst_exact: f64 = 0.0;
    let mut worst_methods: f64 = 0.0;
    for (iu, &u) in us.iter().enumerate() {
        let device = bhz(BhzParams::default()).with_onsite_term(imp.exchange(u, &bhz_spin(coupling)));
        let sys = TwoTerminal::from_model(&device, &clean, &region, 0);
        for (ie, &e) in energies.iter().enumerate() {
            let g_ref = cond[iu + 3 * ie];
            // Thesis: lead self-energies at E + iη, η = 1e-5 (ti.ini), device at real E.
            let sol = sys.solve(e, SelfEnergyMethod::ModesBroadened { eta: 1e-5 }).unwrap();
            worst_thesis = worst_thesis.max((thesis_transmission(&sol) - g_ref).abs());
            let exact = sys.transmission(e).unwrap();
            worst_exact = worst_exact.max((exact - g_ref).abs());
            // The exact transmission does not depend on how the self-energies are computed.
            let decimated = sys.transmission_with(e, SelfEnergyMethod::Decimation { eta: 1e-9 }).unwrap();
            worst_methods = worst_methods.max((exact - decimated).abs());
        }
    }
    // Same configuration, same self-energies and the thesis formula: equal to round-off.
    assert!(worst_thesis < 1e-9, "{case}: thesis formula off by {worst_thesis:e}");
    assert!(worst_methods < 1e-6, "{case}: modes and decimation differ by {worst_methods:e}");
    println!("{case}: |thesis formula − golden| ≤ {worst_thesis:.1e}, |exact − golden| ≤ {worst_exact:.2}");
}

#[test]
fn same_g_along_z() {
    check("ti-cond2t-same-g", 1, GFactors::Same);
}

#[test]
fn same_g_isotropic() {
    check("ti-cond2t-isotropic", 0, GFactors::Same);
}

#[test]
fn opposite_g_along_z() {
    check("ti-cond2t-opposite-g", 1, GFactors::Opposite);
}
