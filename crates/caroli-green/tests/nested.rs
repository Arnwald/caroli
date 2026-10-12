//! Nested dissection against the recursive Green's function on varied two-terminal systems:
//! clean and disordered square lattices (E = 0 included, where interior blocks of a clean
//! lattice are singular), BHZ (two spin blocks), graphene (two sublattices and diagonal
//! hoppings), an irregular constriction, a long strip and devices without interior slices.

use caroli_green::{NestedTwoTerminal, SelfEnergyMethod, TwoTerminal};
use caroli_lattice::{Lattice, Region, Site};
use caroli_model::presets::{BhzParams, bhz};
use caroli_model::{CMat, Model, c64};

fn scalar(x: f64, n: usize) -> CMat {
    CMat::from_fn(n, n, |i, j| if i == j { c64::new(x, 0.0) } else { c64::new(0.0, 0.0) })
}

fn square() -> Model {
    Model::new(Lattice::square(1.0, 1)).hopping(0, 0, [1, 0, 0], scalar(-1.0, 1)).hopping(
        0,
        0,
        [0, 1, 0],
        scalar(-1.0, 1),
    )
}

fn graphene() -> Model {
    Model::new(Lattice::honeycomb(1.0, 1))
        .hopping(0, 1, [0, 0, 0], scalar(-1.0, 1))
        .hopping(0, 1, [0, -1, 0], scalar(-1.0, 1))
        .hopping(0, 1, [1, -1, 0], scalar(-1.0, 1))
}

/// Reproducible on-site disorder of strength `w` on every orbital of a site.
fn disordered(model: Model, w: f64, n: usize) -> Model {
    model.with_onsite_term(move |s: &Site| {
        let k = (s.cell[0] * 7919 + s.cell[1] * 104_729 + s.sub as i64 * 31) as u64;
        let u = ((k.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407) >> 11)
            as f64)
            / (1u64 << 53) as f64;
        Some(scalar(w * (u - 0.5), n))
    })
}

fn compare(device: &Model, lead: &Model, region: &Region, energies: &[f64], label: &str) {
    let rgf = TwoTerminal::from_model(device, lead, region, 0);
    let nd = NestedTwoTerminal::from_model(device, lead, region, 0);
    for &e in energies {
        let t_rgf = rgf.transmission_by_blocks(e, SelfEnergyMethod::Modes).unwrap();
        let t_nd = nd.transmission(e).unwrap();
        assert!((t_rgf - t_nd).abs() < 1e-9 * (1.0 + t_rgf), "{label}, E {e}: RGF {t_rgf}, nested {t_nd}");
    }
}

#[test]
fn square_lattice() {
    let (m, lead) = (square(), square());
    let region = Region::cuboid(m.lattice(), [14, 9, 0]);
    compare(&m, &lead, &region, &[-3.1, -1.3, 0.0, 0.4, 2.2], "clean square");
    let dirty = disordered(square(), 2.0, 1);
    compare(&dirty, &lead, &region, &[-2.7, -0.5, 0.0, 1.1, 3.0], "disordered square");
}

#[test]
fn bhz_with_disorder() {
    let lead = bhz(BhzParams::default());
    let dirty = disordered(bhz(BhzParams::default()), 0.6, 4);
    let region = Region::cuboid(lead.lattice(), [12, 7, 0]);
    let nd = NestedTwoTerminal::from_model(&dirty, &lead, &region, 0);
    assert_eq!(nd.block_count(), 2, "the two spin blocks");
    compare(&dirty, &lead, &region, &[-0.4, -0.05, 0.0, 0.08, 0.5], "BHZ");
}

#[test]
fn graphene_ribbons() {
    let lead = graphene();
    let region = Region::cuboid(lead.lattice(), [10, 6, 0]);
    compare(&graphene(), &lead, &region, &[-2.3, -0.7, 0.3, 1.6], "clean graphene");
    compare(&disordered(graphene(), 1.0, 1), &lead, &region, &[-1.1, 0.25, 0.9], "disordered graphene");
}

#[test]
fn constriction() {
    // A square sample pinched to three sites in the middle, with full-width leads.
    let lead = square();
    let region = Region::shaped(lead.lattice(), [20, 11, 0], |p| {
        let (x, y) = (p[0], p[1]);
        (x - 9.5).abs() > 3.0 || (y - 5.0).abs() <= 1.0
    });
    compare(&disordered(square(), 1.0, 1), &lead, &region, &[-2.9, -0.8, 0.6, 2.4], "constriction");
}

#[test]
fn long_strip_and_short_devices() {
    let lead = square();
    let dirty = disordered(square(), 1.5, 1);
    compare(&dirty, &lead, &Region::cuboid(lead.lattice(), [80, 5, 0]), &[-1.9, 0.3, 2.6], "long strip");
    for length in [1usize, 2, 3] {
        compare(&dirty, &lead, &Region::cuboid(lead.lattice(), [length, 6, 0]), &[-1.2, 0.7], "short device");
    }
}
