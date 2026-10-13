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

/// Lead self-energies computed once serve other disorder configurations with the same leads:
/// the transmission is the same as when each system solves its own leads.
#[test]
fn reused_self_energies() {
    let clean = bhz(BhzParams::default());
    let region = Region::cuboid(clean.lattice(), [12, 9, 0]);
    let reference =
        NestedTwoTerminal::from_model(&disordered(bhz(BhzParams::default()), 0.5, 4), &clean, &region, 0);
    let other =
        NestedTwoTerminal::from_model(&disordered(bhz(BhzParams::default()), 1.3, 4), &clean, &region, 0);
    for e in [-0.2, 0.05, 0.4] {
        let sigma = reference.self_energies(e).unwrap();
        assert_eq!(sigma.energy(), e);
        let (t, own) = (other.transmission_with(&sigma).unwrap(), other.transmission(e).unwrap());
        assert!((t - own).abs() < 1e-12, "E = {e}: {t} vs {own}");
        assert!(
            (reference.transmission_with(&sigma).unwrap() - reference.transmission(e).unwrap()).abs() < 1e-12
        );
    }
}

#[test]
#[should_panic(expected = "other leads")]
fn self_energies_of_other_leads() {
    let region = Region::cuboid(square().lattice(), [6, 5, 0]);
    let sys = NestedTwoTerminal::from_model(&square(), &square(), &region, 0);
    let shifted = square().onsite(0, scalar(0.3, 1));
    let other = NestedTwoTerminal::from_model(&shifted, &shifted, &region, 0);
    let _ = other.transmission_with(&sys.self_energies(0.4).unwrap());
}

/// Transmission eigenvalues add up to the transmission and lie in [0, 1]; a clean ribbon has
/// perfect channels (Fano factor 0), a high barrier only small ones (Fano factor near 1).
#[test]
fn transmission_eigenvalues_and_shot_noise() {
    let region = Region::cuboid(square().lattice(), [12, 7, 0]);
    let dirty = NestedTwoTerminal::from_model(&disordered(square(), 2.0, 1), &square(), &region, 0);
    let clean = NestedTwoTerminal::from_model(&square(), &square(), &region, 0);
    let barrier = square().with_onsite_term(|s: &Site| (s.cell[0] == 6).then(|| scalar(40.0, 1)));
    let tunnel = NestedTwoTerminal::from_model(&barrier, &square(), &region, 0);
    for e in [-2.3, -0.7, 0.9] {
        let taus = dirty.transmission_eigenvalues(e).unwrap();
        let t = dirty.transmission(e).unwrap();
        assert!(
            (taus.iter().sum::<f64>() - t).abs() < 1e-10,
            "E = {e}: Σ τ = {} vs T = {t}",
            taus.iter().sum::<f64>()
        );
        assert!(taus.iter().all(|&x| (-1e-10..=1.0 + 1e-10).contains(&x)), "E = {e}: {taus:?}");
        let perfect = clean.transmission_eigenvalues(e).unwrap();
        assert!(
            perfect.iter().all(|&x| (x - 1.0).abs() < 1e-9) && clean.fano_factor(e).unwrap().abs() < 1e-8
        );
        assert!(tunnel.fano_factor(e).unwrap() > 0.99, "E = {e}: tunnel barrier");
    }
}

/// Three dimensions: a disordered cubic wire (cross-section 6 × 5) and a slab cut along its
/// thin direction, against the recursive Green's function.
#[test]
fn cubic_samples() {
    let cubic = || {
        Model::new(Lattice::cubic(1.0, 1))
            .hopping(0, 0, [1, 0, 0], scalar(-1.0, 1))
            .hopping(0, 0, [0, 1, 0], scalar(-1.0, 1))
            .hopping(0, 0, [0, 0, 1], scalar(-1.0, 1))
    };
    for (extent, axis) in [([14, 6, 5], 0), ([4, 9, 8], 0), ([7, 10, 6], 2)] {
        let region = Region::cuboid(cubic().lattice(), extent);
        let device = disordered(cubic(), 2.5, 1);
        let nested = NestedTwoTerminal::from_model(&device, &cubic(), &region, axis);
        let rgf = TwoTerminal::from_model(&device, &cubic(), &region, axis);
        for e in [-4.1, -1.3, 0.0, 2.2] {
            let (t, t2) = (nested.transmission(e).unwrap(), rgf.transmission(e).unwrap());
            assert!((t - t2).abs() < 1e-9 * (1.0 + t2), "{extent:?} along {axis}, E = {e}: {t} vs {t2}");
        }
    }
}

/// The two-terminal scattering matrix is unitary and equals the multi-terminal one of the same
/// ribbon (disordered, and in a field).
#[test]
fn two_terminal_scattering_matrix() {
    use caroli_green::{Interface, NestedMultiTerminal};
    let region = Region::cuboid(square().lattice(), [10, 7, 0]);
    let slices = region.slices(0);
    let interfaces = [
        Interface { slice: slices[0].clone(), outward: [-1, 0, 0] },
        Interface { slice: slices[9].clone(), outward: [1, 0, 0] },
    ];
    for (device, lead) in [
        (disordered(square(), 2.0, 1), square()),
        (disordered(square(), 1.0, 1).with_field(0.2), square().with_field(0.2)),
    ] {
        let two = NestedTwoTerminal::from_model(&device, &lead, &region, 0);
        let multi = NestedMultiTerminal::from_model(&device, &lead, &region, &interfaces);
        for e in [-2.6, -0.9, 0.4, 1.8] {
            let (s2, sm) = (two.smatrix(e).unwrap().full(), multi.smatrix(e).unwrap().full());
            let n = s2.ncols();
            assert!((s2.adjoint() * &s2 - CMat::identity(n, n)).norm_max() < 1e-9, "E = {e}: unitarity");
            assert!((&s2 - &sm).norm_max() < 1e-9, "E = {e}: two-terminal and multi-terminal S differ");
            assert!((two.smatrix(e).unwrap().transmission(1, 0) - two.transmission(e).unwrap()).abs() < 1e-9);
        }
    }
}

/// Bond disorder (a site-dependent hopping term): nested dissection and the recursive Green's
/// function agree.
#[test]
fn bond_disorder() {
    let factor = |a: &Site, b: &Site| {
        let (p, q) = if a < b { (a, b) } else { (b, a) };
        let k = (p.cell[0] * 7919 + p.cell[1] * 104_729 + q.cell[0] * 31 + q.cell[1] * 977) as u64;
        1.0 + 0.6 * (((k.wrapping_mul(6_364_136_223_846_793_005) >> 11) as f64 / (1u64 << 53) as f64) - 0.5)
    };
    let region = Region::cuboid(square().lattice(), [12, 7, 0]);
    // Bond disorder inside the device only: the leads stay clean.
    let device = square().with_hopping_term(move |from: &Site, to: &Site, t: &CMat| {
        let inside = |s: &Site| (1..11).contains(&s.cell[0]);
        (inside(from) && inside(to)).then(|| CMat::from_fn(1, 1, |_, _| t[(0, 0)] * factor(from, to)))
    });
    let (nested, rgf) = (
        NestedTwoTerminal::from_model(&device, &square(), &region, 0),
        TwoTerminal::from_model(&device, &square(), &region, 0),
    );
    for e in [-2.4, -0.5, 0.0, 1.9] {
        let (t, t2) = (nested.transmission(e).unwrap(), rgf.transmission(e).unwrap());
        assert!((t - t2).abs() < 1e-10, "E = {e}: {t} vs {t2}");
    }
}

/// The fronts in single precision follow double precision, and a checked batch measures the
/// difference and falls back to double precision when it is too large.
#[test]
fn single_precision() {
    use caroli_green::{Precision, PrecisionOptions};
    // Large enough for fronts above the small-front threshold (the dense library's kernels).
    let lead = square();
    let region = Region::cuboid(lead.lattice(), [48, 40, 0]);
    let nd = NestedTwoTerminal::from_model(&disordered(square(), 2.0, 1), &lead, &region, 0);
    let energies = [-2.7, -1.1, 0.0, 0.6, 1.9];
    for &e in &energies {
        let sigma = nd.self_energies(e).unwrap();
        let double = nd.transmission_with(&sigma).unwrap();
        let single = nd.transmission_with_precision(&sigma, Precision::Single).unwrap();
        assert!((single - double).abs() < 1e-4, "square, E {e}: double {double}, single {single}");
    }
    let bhz_lead = bhz(BhzParams::default());
    let bhz_region = Region::cuboid(bhz_lead.lattice(), [24, 20, 0]);
    let bhz_nd = NestedTwoTerminal::from_model(
        &disordered(bhz(BhzParams::default()), 0.6, 4),
        &bhz_lead,
        &bhz_region,
        0,
    );
    for e in [-0.3, 0.02, 0.4] {
        let sigma = bhz_nd.self_energies(e).unwrap();
        let double = bhz_nd.transmission_with(&sigma).unwrap();
        let single = bhz_nd.transmission_with_precision(&sigma, Precision::Single).unwrap();
        assert!((single - double).abs() < 1e-4, "BHZ, E {e}: double {double}, single {single}");
    }
    // Checked: every other energy recomputed in double precision.
    let exact: Vec<f64> = nd.transmissions(&energies).into_iter().map(|t| t.unwrap()).collect();
    let options = PrecisionOptions { every: 2, ..PrecisionOptions::single() };
    let (t, report) = nd.transmissions_checked(&energies, &options);
    assert!(report.fallback.is_none() && report.checked == 3 && report.max < 1e-4, "{report:?}");
    assert_eq!(report.precision, Precision::Single);
    assert_eq!(t[0].as_ref().unwrap(), &exact[0], "checked points return their double-precision value");
    // A tolerance no single-precision run meets: the batch is recomputed in double precision.
    let strict = PrecisionOptions { tolerance: 0.0, ..options };
    let (t, report) = nd.transmissions_checked(&energies, &strict);
    assert!(report.fallback.is_some() && report.precision == Precision::Double, "{report:?}");
    assert!(t.iter().zip(&exact).all(|(a, b)| a.as_ref().unwrap() == b));
}
