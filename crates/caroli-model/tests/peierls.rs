//! Peierls phases: the spectrum of a finite flake in a uniform field does not depend on the
//! gauge, and a field changes it.

use caroli_lattice::{Lattice, Region};
use caroli_model::{CMat, Model, c64, gauge};
use faer::Side;

fn one(x: f64) -> CMat {
    CMat::from_fn(1, 1, |_, _| c64::new(x, 0.0))
}

fn spectrum(model: &Model, region: &Region) -> Vec<f64> {
    let h = model.hamiltonian(region);
    let mut e: Vec<f64> = h.self_adjoint_eigenvalues(Side::Lower).unwrap();
    e.sort_by(f64::total_cmp);
    e
}

fn max_difference(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

#[test]
fn flake_spectra_are_gauge_invariant() {
    let b = 0.37;
    let square = || {
        Model::new(Lattice::square(1.0, 1)).hopping(0, 0, [1, 0, 0], one(-1.0)).hopping(
            0,
            0,
            [0, 1, 0],
            one(-1.0),
        )
    };
    let graphene = || {
        Model::new(Lattice::honeycomb(1.0, 1))
            .hopping(0, 1, [0, 0, 0], one(-1.0))
            .hopping(0, 1, [0, -1, 0], one(-1.0))
            .hopping(0, 1, [1, -1, 0], one(-1.0))
    };
    for model in [square as fn() -> Model, graphene] {
        let region = Region::cuboid(model().lattice(), [8, 7, 0]);
        let x = spectrum(&model().with_peierls(gauge::landau_x(b)), &region);
        let y = spectrum(&model().with_peierls(gauge::landau_y(b)), &region);
        let s = spectrum(&model().with_peierls(gauge::symmetric(b)), &region);
        assert!(max_difference(&x, &y) < 1e-12 && max_difference(&x, &s) < 1e-12);
        assert!(max_difference(&x, &spectrum(&model(), &region)) > 1e-2, "the field changes the spectrum");
    }
}
