//! Site-dependent hopping terms: a term with the Peierls phase gives the Hamiltonian of
//! `with_peierls`, and bond disorder keeps the Hamiltonian Hermitian, the same in its dense and
//! sparse forms.

use caroli_lattice::{Lattice, Region, Site};
use caroli_model::{CMat, Model, c64, gauge};

fn square() -> Model {
    Model::new(Lattice::square(1.0, 1)).hopping(0, 0, [1, 0, 0], -1.0).hopping(0, 0, [0, 1, 0], -1.0)
}

#[test]
fn peierls_as_a_hopping_term() {
    let b = 0.31;
    let lattice = Lattice::square(1.0, 1);
    let phase = gauge::landau_x(b);
    let term = move |from: &Site, to: &Site, t: &CMat| {
        let (ri, rj) = (lattice.position(*from), lattice.position(*to));
        Some(t * faer::Scale(c64::from_polar(1.0, phase(ri, rj))))
    };
    let region = Region::cuboid(square().lattice(), [7, 6, 0]);
    let a = square().with_peierls(gauge::landau_x(b)).hamiltonian(&region);
    let h = square().with_hopping_term(term).hamiltonian(&region);
    assert!((a - h).norm_max() < 1e-14);
}

#[test]
fn bond_disorder() {
    // A reproducible factor per bond, the same in both directions.
    let factor = |a: &Site, b: &Site| {
        let (p, q) = if a < b { (a, b) } else { (b, a) };
        let k = (p.cell[0] * 7919 + p.cell[1] * 104_729 + q.cell[0] * 31 + q.cell[1] * 977) as u64;
        1.0 + 0.4 * (((k.wrapping_mul(6_364_136_223_846_793_005) >> 11) as f64 / (1u64 << 53) as f64) - 0.5)
    };
    let model = square().with_hopping_term(move |from: &Site, to: &Site, t: &CMat| {
        Some(t * faer::Scale(c64::new(factor(from, to), 0.0)))
    });
    let region = Region::cuboid(model.lattice(), [8, 5, 0]);
    let h = model.hamiltonian(&region);
    assert!((&h - h.adjoint()).norm_max() < 1e-15, "Hermitian");
    assert!((model.sparse_hamiltonian(&region).to_dense() - &h).norm_max() < 1e-15, "dense and sparse");
    assert!((&h - square().hamiltonian(&region)).norm_max() > 0.01, "the bonds changed");
}
