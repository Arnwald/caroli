//! Four-terminal conductances with Anderson disorder against `conductance-4t`
//! (`ti-cond4t-anderson`): 8×8 square, arms of 4, three U0 (log grid) × three energies, spin-up
//! BHZ block (the thesis `dim = 2`).

use caroli_green::{MultiTerminal, SelfEnergyMethod};
use caroli_model::presets::{BhzParams, Spin, bhz_block};
use caroli_validation::golden_f64;
use caroli_validation::thesis::{FourTerminal, thesis_transmission_pq};

#[test]
fn anderson_cross() {
    let p = FourTerminal { nx: 8, ny: 8, ntp: 4, seed: 12347 };
    let (region, interfaces) = p.geometry();
    let axes = &golden_f64("ti-cond4t-anderson", "axes_00.dat")[0];
    let gxx = &golden_f64("ti-cond4t-anderson", "sigma_xx_00.dat")[0];
    let gxy = &golden_f64("ti-cond4t-anderson", "sigma_xy_00.dat")[0];
    let nu = 3;
    let (us, es) = axes.split_at(nu);
    let clean = bhz_block(BhzParams::default(), Spin::Up);
    let (mut worst_thesis, mut worst_exact): (f64, f64) = (0.0, 0.0);
    for (iu, &u) in us.iter().enumerate() {
        let device = bhz_block(BhzParams::default(), Spin::Up).with_onsite_term(p.anderson(u));
        let sys = MultiTerminal::from_model(&device, &clean, &region, &interfaces);
        for (ie, &e) in es.iter().enumerate() {
            let sol = sys.solve(e, SelfEnergyMethod::ModesBroadened { eta: 1e-5 }).unwrap();
            let (txx, txy) =
                (thesis_transmission_pq(&sys, &sol, 0, 2), thesis_transmission_pq(&sys, &sol, 1, 3));
            let k = iu + nu * ie;
            worst_thesis = worst_thesis.max((txx - gxx[k]).abs()).max((txy - gxy[k]).abs());
            let exact = sys.solve(e, SelfEnergyMethod::Modes).unwrap();
            worst_exact = worst_exact.max((exact.transmission(&sys, 0, 2) - gxx[k]).abs());
        }
    }
    println!(
        "ti-cond4t-anderson: |thesis formula − golden| ≤ {worst_thesis:.1e}, |exact − golden| ≤ {worst_exact:.1e}"
    );
    // The thesis puts E + iη (η = 1e-5) on its last recursion slice; caroli's device is at real E.
    assert!(worst_thesis < 1e-4, "thesis formula off by {worst_thesis:e}");
}
