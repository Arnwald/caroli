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

/// A `w × w` square with arms of length `arm` and a lead at the end of each: left, right,
/// bottom, top.
fn cross(lat: &Lattice, w: i64, arm: i64) -> (Region, Vec<caroli_green::Interface>) {
    let site = |x: i64, y: i64| Site { cell: [x, y, 0], sub: 0 };
    let cells = (-arm..w + arm)
        .flat_map(|x| (-arm..w + arm).map(move |y| (x, y)))
        .filter(|&(x, y)| (0..w).contains(&x) || (0..w).contains(&y));
    let region = Region::from_sites(lat, cells.map(|(x, y)| site(x, y)).collect());
    let column = |x: i64| Region::from_sites(lat, (0..w).map(|y| site(x, y)).collect());
    let row = |y: i64| Region::from_sites(lat, (0..w).map(|x| site(x, y)).collect());
    let interfaces = vec![
        caroli_green::Interface { slice: column(-arm), outward: [-1, 0, 0] },
        caroli_green::Interface { slice: column(w + arm - 1), outward: [1, 0, 0] },
        caroli_green::Interface { slice: row(-arm), outward: [0, -1, 0] },
        caroli_green::Interface { slice: row(w + arm - 1), outward: [0, 1, 0] },
    ];
    (region, interfaces)
}

/// Transport along y in a uniform field: the leads are taken to the Landau gauge invariant
/// along y, and the plateaus are those of the ribbon along x.
#[test]
fn quantum_hall_plateaus_along_y() {
    let b = 0.1;
    let clean = square().with_field(b);
    let device = disordered(square(), 0.2).with_field(b);
    let region = Region::cuboid(clean.lattice(), [60, 40, 0]);
    let nested = NestedTwoTerminal::from_model(&device, &clean, &region, 1);
    let rgf = TwoTerminal::from_model(&device, &clean, &region, 1);
    for (e, plateau) in [(-3.8, 1.0), (-3.6, 2.0), (-3.4, 3.0)] {
        let (t, t2) = (nested.transmission(e).unwrap(), rgf.transmission(e).unwrap());
        assert!((t - plateau).abs() < 1e-6 && (t - t2).abs() < 1e-9, "E = {e}: T = {t} (RGF {t2})");
    }
}

/// A clean cross in a uniform field is symmetric under a quarter turn, which takes left to
/// bottom, bottom to right, right to top and top to left: the leads along x and along y, in
/// different gauges, give the same transmissions. Dense and nested solvers agree.
#[test]
fn hall_cross_quarter_turn() {
    use caroli_green::{MultiTerminal, NestedMultiTerminal, SelfEnergyMethod};
    let b = 0.05;
    let model = square().with_field(b);
    let (region, interfaces) = cross(model.lattice(), 12, 4);
    let nested = NestedMultiTerminal::from_model(&model, &model, &region, &interfaces);
    let dense = MultiTerminal::from_model(&model, &model, &region, &interfaces);
    let turn = [2, 3, 1, 0]; // left → bottom, right → top, bottom → right, top → left
    for e in [-3.1, -2.5, -1.2] {
        let t = nested.transmissions(e).unwrap();
        let sol = dense.solve(e, SelfEnergyMethod::Modes).unwrap();
        for p in 0..4 {
            for q in 0..4 {
                assert!((t[p][q] - sol.transmission(&dense, p, q)).abs() < 1e-9, "E = {e}: dense and nested");
                if p != q {
                    let rotated = t[turn[p]][turn[q]];
                    assert!(
                        (t[p][q] - rotated).abs() < 1e-9,
                        "E = {e}: T[{p}][{q}] = {} vs {rotated}",
                        t[p][q]
                    );
                }
            }
        }
    }
}

/// Onsager: reversing the field transposes the transmissions of a disordered cross.
#[test]
fn hall_cross_onsager() {
    use caroli_green::NestedMultiTerminal;
    let (region, interfaces) = cross(square().lattice(), 12, 4);
    let t = |b: f64, e: f64| {
        let (clean, device) = (square().with_field(b), disordered(square(), 1.0).with_field(b));
        NestedMultiTerminal::from_model(&device, &clean, &region, &interfaces).transmissions(e).unwrap()
    };
    for e in [-2.7, -0.8, 0.9] {
        let (plus, minus) = (t(0.07, e), t(-0.07, e));
        for p in 0..4 {
            for q in 0..4 {
                assert!(
                    (plus[p][q] - minus[q][p]).abs() < 1e-9,
                    "E = {e}: T[{p}][{q}](b) vs T[{q}][{p}](−b)"
                );
            }
        }
    }
}

/// Quantum Hall regime in a disordered cross: the edge states carry ν channels from each lead to
/// the next one around the cross, and nothing elsewhere.
#[test]
fn hall_cross_chirality() {
    use caroli_green::NestedMultiTerminal;
    let b = 0.1;
    let (clean, device) = (square().with_field(b), disordered(square(), 0.2).with_field(b));
    let (region, interfaces) = cross(clean.lattice(), 40, 10);
    let sys = NestedMultiTerminal::from_model(&device, &clean, &region, &interfaces);
    for (e, nu) in [(-3.8, 1.0), (-3.6, 2.0)] {
        let t = sys.transmissions(e).unwrap();
        // Column q: what lead q sends to the others.
        let columns: Vec<Vec<f64>> = (0..4).map(|q| t.iter().map(|row| row[q]).collect()).collect();
        for (q, column) in columns.iter().enumerate() {
            let targets: Vec<usize> = (0..4).filter(|&p| p != q && column[p] > 0.5).collect();
            assert_eq!(targets.len(), 1, "E = {e}: lead {q} feeds {targets:?}");
            let p = targets[0];
            assert!((column[p] - nu).abs() < 1e-6, "E = {e}: T[{p}][{q}] = {}", column[p]);
            assert!((0..4).filter(|&r| r != q && r != p).all(|r| column[r].abs() < 1e-6));
        }
    }
}
