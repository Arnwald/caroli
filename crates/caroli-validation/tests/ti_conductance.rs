//! Two-terminal conductance of a clean BHZ ribbon against `conductance-2t` (`ti-cond2t-clean`).

use caroli_green::{SelfEnergyMethod, TwoTerminal};
use caroli_lattice::Region;
use caroli_model::presets::{BhzParams, bhz};
use caroli_validation::golden_f64;

/// Case `ti-cond2t-clean`: 16×10 ribbon, Nu = 3 identical columns (no impurity), 13 energies.
/// The axes file holds the 3 values of U0 then the 13 energies; cond(Nu, Ne) is column-major.
fn golden() -> (Vec<f64>, Vec<f64>) {
    let axes = &golden_f64("ti-cond2t-clean", "axes_00.dat")[0];
    let cond = &golden_f64("ti-cond2t-clean", "cond_00.dat")[0];
    let (nu, ne) = (3, 13);
    assert_eq!((axes.len(), cond.len()), (nu + ne, nu * ne));
    let energies = axes[nu..].to_vec();
    let g = (0..ne).map(|ie| cond[ie * nu]).collect();
    (energies, g)
}

fn ribbon() -> TwoTerminal {
    let model = bhz(BhzParams::default());
    let region = Region::cuboid(model.lattice(), [16, 10, 0]);
    TwoTerminal::from_model(&model, &model, &region, 0)
}

/// A clean wire matched to its leads transmits every open channel: T is an integer.
/// The thesis code deviates from integers by up to ~1 % (reference/baseline/README.md,
/// "Findings"); we check that caroli agrees with it within that margin.
#[test]
fn clean_ribbon_conductance() {
    let (energies, g_ref) = golden();
    let sys = ribbon();
    for (e, gr) in energies.iter().zip(&g_ref) {
        let t = sys.transmission(*e).unwrap();
        assert!((t - t.round()).abs() < 1e-8, "T({e}) = {t} is not an integer");
        assert!((t - gr).abs() < 0.025, "T({e}) = {t}, thesis {gr}");
    }
}

/// Inside the bulk gap the ribbon carries the two helical edge channels: T = 2.
#[test]
fn quantum_spin_hall_plateau() {
    let sys = ribbon();
    for e in [-0.05, 0.0, 0.02, 0.05] {
        let t = sys.transmission(e).unwrap();
        assert!((t - 2.0).abs() < 1e-8, "T({e}) = {t}");
    }
}

/// Mode matching and decimation agree away from cell eigenvalues.
#[test]
fn modes_agree_with_decimation() {
    let sys = ribbon();
    for e in [-0.27, 0.13, 0.31] {
        let a = sys.transmission(e).unwrap();
        let b = sys.transmission_with(e, SelfEnergyMethod::Decimation { eta: 1e-10 }).unwrap();
        assert!((a - b).abs() < 1e-6, "T({e}): modes {a}, decimation {b}");
    }
}
