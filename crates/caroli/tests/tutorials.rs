//! The classic transport tutorials (those of Kwant's documentation), written with caroli, with
//! the physics they show checked: conductance steps of a quantum wire, a Rashba wire in a
//! Zeeman field, the Aharonov–Bohm effect in a ring, the edge channel of a zigzag graphene
//! ribbon, and the quantum Hall effect in a Hall bar.

use caroli::prelude::*;

/// Conductance steps: a clean wire 10 sites wide transmits one channel per open subband,
/// `T(E) = #{q : E > 2 − 2 cos(qπ/11)}` near the bottom of the band.
#[test]
fn quantum_wire() {
    let (w, t) = (10, 1.0);
    let wire = Model::new(Lattice::square(1.0, 1)).onsite(0, 4.0 * t).hopping(0, 0, [1, 0, 0], -t).hopping(
        0,
        0,
        [0, 1, 0],
        -t,
    );
    let region = Region::cuboid(wire.lattice(), [30, w, 0]);
    let sys = NestedTwoTerminal::from_model(&wire, &wire, &region, 0);
    for e in [0.05, 0.2, 0.45, 0.7, 0.95] {
        let open = (1..=w)
            .filter(|&q| e > 2.0 - 2.0 * (q as f64 * std::f64::consts::PI / (w + 1) as f64).cos())
            .count();
        assert!((sys.transmission(e).unwrap() - open as f64).abs() < 1e-9, "E = {e}");
    }
}

/// A Rashba wire in a Zeeman field (spin in the orbitals): `H = 4t σ0 + e_z σz` on site,
/// `−t σ0 + i α σy / 2` along x and `−t σ0 − i α σx / 2` along y. The clean wire still transmits
/// one per channel of its leads, and the two solvers agree.
#[test]
fn rashba_wire() {
    let (t, alpha, e_z) = (1.0, 0.5, 0.08);
    let i = c64::new(0.0, 1.0);
    let wire = Model::new(Lattice::square(1.0, 2))
        .onsite(0, combine(&[(4.0 * t, &s0()), (e_z, &sz())]))
        .hopping(0, 0, [1, 0, 0], combine(&[(c64::from(-t), &s0()), (i * alpha / 2.0, &sy())]))
        .hopping(0, 0, [0, 1, 0], combine(&[(c64::from(-t), &s0()), (-i * alpha / 2.0, &sx())]));
    let region = Region::cuboid(wire.lattice(), [30, 10, 0]);
    let nested = NestedTwoTerminal::from_model(&wire, &wire, &region, 0);
    let rgf = TwoTerminal::from_model(&wire, &wire, &region, 0);
    for e in [0.03, 0.12, 0.31, 0.58] {
        let channels = rgf.left.modes(e).unwrap().channels() as f64;
        let (tn, tr) = (nested.transmission(e).unwrap(), rgf.transmission(e).unwrap());
        assert!(
            (tn - channels).abs() < 1e-8 && (tn - tr).abs() < 1e-9,
            "E = {e}: {tn}, {tr}, {channels} channels"
        );
    }
}

/// Aharonov–Bohm ring: the flux φ goes through the hole on a branch cut (the bonds crossing the
/// half-line above the centre carry the phase), so the transmission is periodic in φ with the
/// flux quantum 2π, and the flux changes it.
#[test]
fn aharonov_bohm_ring() {
    let (r_in, r_out, arm) = (6.0, 11.0, 4i64);
    let centre = [0.5, 0.5];
    let inside = move |x: f64, y: f64| {
        let r = (x - centre[0]).hypot(y - centre[1]);
        (r_in..=r_out).contains(&r) || ((y - centre[1]).abs() < 2.0 && (x - centre[0]).abs() >= r_in)
    };
    let lat = Lattice::square(1.0, 1);
    let reach = r_out as i64 + arm;
    let sites: Vec<Site> = (-reach..=reach + 1)
        .flat_map(|x| (-reach..=reach + 1).map(move |y| (x, y)))
        .filter(|&(x, y)| inside(x as f64, y as f64))
        .map(|(x, y)| Site { cell: [x, y, 0], sub: 0 })
        .collect();
    let region = Region::from_sites(&lat, sites);
    let column = |x: i64| {
        let cells: Vec<Site> = region.sites().iter().filter(|s| s.cell[0] == x).copied().collect();
        Region::from_sites(&lat, cells)
    };
    let (left, right) = (-reach, reach + 1);
    let interfaces = [
        Interface { slice: column(left), outward: [-1, 0, 0] },
        Interface { slice: column(right), outward: [1, 0, 0] },
    ];
    let ring = |phi: f64| {
        // Horizontal bonds that cross x = 0.5 above the hole carry the flux.
        let cut = move |ri: [f64; 3], rj: [f64; 3]| {
            let crosses = ri[1] > centre[1] + 1.0 && (ri[0] - centre[0]) * (rj[0] - centre[0]) < 0.0;
            if crosses { phi * (ri[0] - rj[0]).signum() } else { 0.0 }
        };
        let model = |p: bool| {
            let m = Model::new(lat.clone()).onsite(0, 4.0).hopping(0, 0, [1, 0, 0], -1.0).hopping(
                0,
                0,
                [0, 1, 0],
                -1.0,
            );
            if p { m.with_peierls(cut) } else { m }
        };
        NestedMultiTerminal::from_model(&model(true), &model(false), &region, &interfaces)
    };
    let e = 0.6;
    let t = |phi: f64| ring(phi).transmissions(e).unwrap()[1][0];
    let (t0, t_pi) = (t(0.3), t(0.3 + std::f64::consts::PI));
    assert!((t0 - t(0.3 + 2.0 * std::f64::consts::PI)).abs() < 1e-9, "period 2π");
    assert!((t0 - t_pi).abs() > 1e-3, "the flux changes T: {t0} vs {t_pi}");
}

/// Graphene ribbon with zigzag edges: near the Dirac point it transmits the single edge channel.
#[test]
fn zigzag_graphene_ribbon() {
    let graphene = Model::new(Lattice::honeycomb(1.0, 1))
        .hopping(0, 1, [0, 0, 0], -1.0)
        .hopping(0, 1, [0, -1, 0], -1.0)
        .hopping(0, 1, [1, -1, 0], -1.0);
    let region = Region::cuboid(graphene.lattice(), [20, 20, 0]);
    let sys = NestedTwoTerminal::from_model(&graphene, &graphene, &region, 0);
    for e in [0.02, -0.05, 0.08] {
        assert!((sys.transmission(e).unwrap() - 1.0).abs() < 1e-8, "E = {e}");
    }
}

/// Quantum Hall effect in a Hall bar (a cross, leads: left, right, bottom, top). A current from
/// left to right, with the bottom and top leads as voltage probes: Landauer–Büttiker gives
/// the Hall resistance `h/(ν e²)` and no longitudinal resistance on the plateaus.
#[test]
fn hall_bar() {
    let b = 0.1;
    let clean = Model::new(Lattice::square(1.0, 1))
        .hopping(0, 0, [1, 0, 0], -1.0)
        .hopping(0, 0, [0, 1, 0], -1.0)
        .with_field(b);
    let (w, arm) = (40i64, 10i64);
    let lat = clean.lattice().clone();
    let cell = |x: i64, y: i64| Site { cell: [x, y, 0], sub: 0 };
    let region = Region::from_sites(
        &lat,
        (-arm..w + arm)
            .flat_map(|x| (-arm..w + arm).map(move |y| (x, y)))
            .filter(|&(x, y)| (0..w).contains(&x) || (0..w).contains(&y))
            .map(|(x, y)| cell(x, y))
            .collect(),
    );
    let column = |x: i64| Region::from_sites(&lat, (0..w).map(|y| cell(x, y)).collect());
    let row = |y: i64| Region::from_sites(&lat, (0..w).map(|x| cell(x, y)).collect());
    let interfaces = [
        Interface { slice: column(-arm), outward: [-1, 0, 0] },
        Interface { slice: column(w + arm - 1), outward: [1, 0, 0] },
        Interface { slice: row(-arm), outward: [0, -1, 0] },
        Interface { slice: row(w + arm - 1), outward: [0, 1, 0] },
    ];
    let sys = NestedMultiTerminal::from_model(&clean, &clean, &region, &interfaces);
    for (e, nu) in [(-3.8, 1.0), (-3.6, 2.0)] {
        let t = sys.transmissions(e).unwrap();
        // A unit current from the left lead to the grounded right one; the bottom and top leads
        // are voltage probes (no current).
        let v = lead_voltages(&t, &[1.0, -1.0, 0.0, 0.0], 1);
        let (v0, v2, v3) = (v[0], v[2], v[3]);
        let hall = v3 - v2;
        assert!((hall.abs() - 1.0 / nu).abs() < 1e-6, "E = {e}: R_xy = {hall}, expected ±{}", 1.0 / nu);
        // Two-terminal resistance h/(ν e²) between source and drain: R_xx is zero.
        assert!((v0 - 1.0 / nu).abs() < 1e-6, "E = {e}: R_2t = {v0}");
    }
}
