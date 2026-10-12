//! Scattering states of the nested-dissection solver against the recursive lesser Green's
//! function: occupied densities and bond currents on a disordered square sample, BHZ (two spin
//! blocks), graphene and a sample in a magnetic field; the bond currents through every
//! cross-section add up to the transmission.

use caroli_green::{NestedTwoTerminal, SelfEnergyMethod, TwoTerminal};
use caroli_keldysh::Lesser;
use caroli_lattice::{Lattice, Region, Site};
use caroli_model::presets::{BhzParams, bhz};
use caroli_model::{CMat, Model, c64};
use std::collections::HashMap;

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

fn compare(device: &Model, lead: &Model, region: &Region, energies: &[f64]) {
    let rgf = TwoTerminal::from_model(device, lead, region, 0);
    let nested = NestedTwoTerminal::from_model(device, lead, region, 0);
    for &e in energies {
        let states = nested.scattering_states(e).unwrap();
        for (fl, fr) in [(1.0, 0.0), (0.3, 0.8)] {
            let lesser = Lesser::compute(&rgf, e, SelfEnergyMethod::Modes, fl, fr).unwrap();
            // Occupied densities, slice by slice.
            let occupation = states.occupation(fl, fr);
            for (n, slice) in rgf.regions.iter().enumerate() {
                let reference = lesser.occupation(n);
                let mut k = 0;
                for site in slice.sites() {
                    let i = region.index_of(site).unwrap();
                    let (o0, no) = (region.offset(i), region.site_orbitals(i));
                    for (a, occ) in occupation[o0..o0 + no].iter().enumerate() {
                        assert!(
                            (occ - reference[k + a]).abs() < 1e-10,
                            "E = {e}: occupation of orbital {}",
                            o0 + a
                        );
                    }
                    k += no;
                }
            }
            // Bond currents, each bond in a fixed direction.
            let canonical =
                |from: Site, to: Site, j: f64| if from < to { ((from, to), j) } else { ((to, from), -j) };
            let reference: HashMap<(Site, Site), f64> =
                lesser.bond_currents(&rgf).into_iter().map(|b| canonical(b.from, b.to, b.current)).collect();
            let ours = nested.bond_currents(&states, fl, fr);
            assert_eq!(ours.len(), reference.len(), "E = {e}: bonds");
            for b in ours {
                let (key, j) = canonical(b.from, b.to, b.current);
                assert!(
                    (j - reference[&key]).abs() < 1e-10,
                    "E = {e}: bond {key:?}: {j} vs {}",
                    reference[&key]
                );
            }
            // Through each cross-section between slices, the currents add up to T (f_L − f_R).
            let t = nested.transmission(e).unwrap();
            let currents = nested.bond_currents(&states, fl, fr);
            for x in 0..rgf.regions.len() as i64 - 1 {
                let through: f64 = currents
                    .iter()
                    .map(|b| match (b.from.cell[0] <= x, b.to.cell[0] <= x) {
                        (true, false) => b.current,
                        (false, true) => -b.current,
                        _ => 0.0,
                    })
                    .sum();
                assert!(
                    (through - t * (fl - fr)).abs() < 1e-9,
                    "E = {e}: cut {x}: {through} vs {}",
                    t * (fl - fr)
                );
            }
        }
    }
}

#[test]
fn square_sample() {
    let region = Region::cuboid(square().lattice(), [10, 7, 0]);
    compare(&disordered(square(), 1.5, 1), &square(), &region, &[-2.6, -0.3, 0.0, 1.4]);
}

#[test]
fn bhz_sample() {
    let clean = bhz(BhzParams::default());
    let region = Region::cuboid(clean.lattice(), [8, 6, 0]);
    compare(&disordered(bhz(BhzParams::default()), 0.6, 4), &clean, &region, &[-0.15, 0.03, 0.35]);
}

#[test]
fn graphene_sample() {
    let region = Region::cuboid(graphene().lattice(), [7, 5, 0]);
    compare(&disordered(graphene(), 0.8, 1), &graphene(), &region, &[-1.3, 0.45, 1.1]);
}

#[test]
fn sample_in_a_field() {
    let b = 0.15;
    let region = Region::cuboid(square().lattice(), [9, 8, 0]);
    compare(
        &disordered(square(), 1.0, 1).with_field(b),
        &square().with_field(b),
        &region,
        &[-3.3, -1.1, 0.6],
    );
}
