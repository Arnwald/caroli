//! Transport in a uniform magnetic field (Peierls phases): quantum Hall plateaus in a
//! disordered ribbon, gauge invariance of the transmission, and the check that the leads see a
//! gauge invariant along them.

use caroli_green::{NestedTwoTerminal, TwoTerminal};
use caroli_lattice::{Lattice, Region, Site};
use caroli_model::{CMat, Model, c64, gauge};

fn one(x: f64) -> CMat {
    CMat::from_fn(1, 1, |_, _| c64::new(x, 0.0))
}

fn square() -> Model {
    Model::new(Lattice::square(1.0, 1)).hopping(0, 0, [1, 0, 0], one(-1.0)).hopping(
        0,
        0,
        [0, 1, 0],
        one(-1.0),
    )
}

/// Reproducible on-site disorder of strength `w`.
fn disordered(model: Model, w: f64) -> Model {
    model.with_onsite_term(move |s: &Site| {
        let k = (s.cell[0] * 7919 + s.cell[1] * 104_729) as u64;
        let u = ((k.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407) >> 11)
            as f64)
            / (1u64 << 53) as f64;
        Some(one(w * (u - 0.5)))
    })
}

/// Flux b = 0.1 per plaquette: Landau levels near E_n = −4 + 0.1 (2n + 1), magnetic length 3.2,
/// on a ribbon 60 wide. Between levels the edge states carry exactly one channel per level
/// below, despite the disorder.
#[test]
fn quantum_hall_plateaus() {
    let b = 0.1;
    let clean = square().with_peierls(gauge::landau_x(b));
    let device = disordered(square(), 0.2).with_peierls(gauge::landau_x(b));
    let region = Region::cuboid(clean.lattice(), [40, 60, 0]);
    let nested = NestedTwoTerminal::from_model(&device, &clean, &region, 0);
    let rgf = TwoTerminal::from_model(&device, &clean, &region, 0);
    for (e, plateau) in [(-3.8, 1.0), (-3.6, 2.0), (-3.4, 3.0)] {
        let (t, t2) = (nested.transmission(e).unwrap(), rgf.transmission(e).unwrap());
        assert!((t - plateau).abs() < 1e-6, "E = {e}: T = {t}, plateau {plateau}");
        assert!((t - t2).abs() < 1e-9, "E = {e}: nested {t}, RGF {t2}");
    }
}

/// A uniform shift of the vector potential, A = −b (y − y0) x̂, is a gauge transformation: the
/// transmission does not change.
#[test]
fn transmission_is_gauge_invariant() {
    let (b, y0) = (0.23, 3.7);
    let landau = gauge::landau_x(b);
    let shifted = move |ri: [f64; 3], rj: [f64; 3]| landau(ri, rj) + b * y0 * (ri[0] - rj[0]);
    let region = Region::cuboid(square().lattice(), [12, 9, 0]);
    let t = |phase: Box<dyn Fn([f64; 3], [f64; 3]) -> f64 + Send + Sync>,
             phase2: Box<dyn Fn([f64; 3], [f64; 3]) -> f64 + Send + Sync>,
             e: f64| {
        let clean = square().with_peierls(phase);
        let device = disordered(square(), 1.0).with_peierls(phase2);
        NestedTwoTerminal::from_model(&device, &clean, &region, 0).transmission(e).unwrap()
    };
    for e in [-2.9, -1.2, 0.4, 1.7] {
        let a = t(Box::new(gauge::landau_x(b)), Box::new(gauge::landau_x(b)), e);
        let s = t(Box::new(shifted.clone()), Box::new(shifted.clone()), e);
        assert!((a - s).abs() < 1e-10, "E = {e}: {a} vs {s}");
    }
}

/// Leads along x in the gauge A = b x ŷ: the lead Hamiltonian changes from cell to cell.
#[test]
#[should_panic(expected = "changes along the lead")]
fn leads_need_an_invariant_gauge() {
    let model = square().with_peierls(gauge::landau_y(0.1));
    let region = Region::cuboid(model.lattice(), [6, 5, 0]);
    let _ = NestedTwoTerminal::from_model(&model, &model, &region, 0);
}
