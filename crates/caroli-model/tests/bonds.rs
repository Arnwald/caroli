//! [`Model::bonds`] rebuilds the Bloch Hamiltonian, and the displacements of the bonds give the
//! Peierls phases of a uniform vector potential.

use caroli_lattice::{Lattice, Region, Site};
use caroli_model::presets::{BhzParams, bhz};
use caroli_model::{CMat, Model, c64};

/// `Σ_bonds e^{i n·φ} t + on-site` against [`Model::bloch_hamiltonian`].
fn rebuild(model: &Model, region: &Region, periods: &[[i64; 3]], phases: &[f64]) -> CMat {
    let mut h = CMat::zeros(region.orbitals(), region.orbitals());
    for (i, site) in region.sites().iter().enumerate() {
        let m = model.onsite_matrix(site);
        let o = region.offset(i);
        for a in 0..m.nrows() {
            for b in 0..m.ncols() {
                h[(o + a, o + b)] += m[(a, b)];
            }
        }
    }
    for bond in model.bonds(region, periods) {
        let phase: f64 = bond.images.iter().zip(phases).map(|(n, p)| *n as f64 * p).sum();
        let (r, c) = (region.offset(bond.from), region.offset(bond.to));
        for a in 0..bond.matrix.nrows() {
            for b in 0..bond.matrix.ncols() {
                h[(r + a, c + b)] += c64::from_polar(1.0, phase) * bond.matrix[(a, b)];
            }
        }
    }
    h
}

#[test]
fn bonds_rebuild_the_hamiltonian() {
    let model = bhz(BhzParams { lambda: 0.2, ..BhzParams::default() }).with_field(0.05);
    let region = Region::cuboid(model.lattice(), [4, 3, 0]);
    let open = rebuild(&model, &region, &[], &[]);
    assert!((open - model.hamiltonian(&region)).norm_max() < 1e-14);
    let one = |x: f64| CMat::from_fn(1, 1, |_, _| c64::new(x, 0.0));
    let graphene = Model::new(Lattice::honeycomb(1.0, 1))
        .hopping(0, 1, [0, 0, 0], one(-1.0))
        .hopping(0, 1, [0, -1, 0], one(-1.0))
        .hopping(0, 1, [1, -1, 0], one(-1.0))
        .with_onsite_term(move |s: &Site| (s.sub == 1 && s.cell == [1, 1, 0]).then(|| one(0.3)));
    let cells = Region::cuboid(graphene.lattice(), [3, 3, 0]);
    let (periods, phases) = ([[3, 0, 0], [0, 3, 0]], [0.4, -1.3]);
    let periodic = rebuild(&graphene, &cells, &periods, &phases);
    assert!((periodic - graphene.bloch_hamiltonian(&cells, &periods, &phases)).norm_max() < 1e-14);
}

/// A uniform `A` through the bonds equals the same `A` as a Peierls phase of the model.
#[test]
fn displacements_give_the_peierls_phases() {
    let a = [0.3, -0.7, 0.0];
    let model = bhz(BhzParams::default());
    let peierls = bhz(BhzParams::default())
        .with_peierls(move |ri: [f64; 3], rj: [f64; 3]| a[0] * (ri[0] - rj[0]) + a[1] * (ri[1] - rj[1]));
    let region = Region::cuboid(model.lattice(), [3, 4, 0]);
    let mut h = CMat::zeros(region.orbitals(), region.orbitals());
    for (i, site) in region.sites().iter().enumerate() {
        let m = model.onsite_matrix(site);
        let o = region.offset(i);
        for p in 0..4 {
            for q in 0..4 {
                h[(o + p, o + q)] += m[(p, q)];
            }
        }
    }
    for bond in model.bonds(&region, &[]) {
        let d = bond.displacement;
        let phase = c64::from_polar(1.0, -(a[0] * d[0] + a[1] * d[1]));
        let (r, c) = (region.offset(bond.from), region.offset(bond.to));
        for p in 0..4 {
            for q in 0..4 {
                h[(r + p, c + q)] += phase * bond.matrix[(p, q)];
            }
        }
    }
    assert!((h - peierls.hamiltonian(&region)).norm_max() < 1e-14);
}
