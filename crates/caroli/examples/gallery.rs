//! The figures of the benchmark page, computed with caroli:
//!
//! - local spin currents along the helical edges of a BHZ ribbon (quantum spin Hall effect), with
//!   Anderson disorder, then with magnetic impurities polarised along z;
//! - the conductance of a clean BHZ ribbon;
//! - a Hall bar in the quantum Hall regime: the local current and the Hall resistance;
//! - the edge states of a zigzag graphene ribbon.
//!
//! ```text
//! cargo run --release -p caroli --example gallery -- OUT_DIR
//! ```
//!
//! Each figure is a JSON array of rows in `OUT_DIR`.

use caroli::prelude::*;
use serde_json::{Value, json};
use std::collections::BTreeMap;

fn write(dir: &str, name: &str, rows: &[Value]) {
    let path = format!("{dir}/{name}.json");
    std::fs::write(&path, serde_json::to_string(rows).expect("JSON")).expect("write");
    println!("{path}: {} rows", rows.len());
}

/// The current vector at each site: every bond gives half its current, along the bond, to each
/// of its two sites.
fn site_currents(lat: &Lattice, bonds: &[BondCurrent]) -> BTreeMap<Site, [f64; 2]> {
    let mut out: BTreeMap<Site, [f64; 2]> = BTreeMap::new();
    for b in bonds {
        let (ra, rb) = (lat.position(b.from), lat.position(b.to));
        let (dx, dy) = (rb[0] - ra[0], rb[1] - ra[1]);
        let norm = dx.hypot(dy);
        for site in [b.from, b.to] {
            let v = out.entry(site).or_insert([0.0, 0.0]);
            v[0] += 0.5 * b.current * dx / norm;
            v[1] += 0.5 * b.current * dy / norm;
        }
    }
    out
}

/// Spin-up and spin-down projectors in the BHZ basis `(E1↑, H1↑, E1↓, H1↓)`.
fn spin_projectors() -> [CMat; 2] {
    let up =
        CMat::from_fn(4, 4, |a, b| if a == b && a < 2 { c64::new(1.0, 0.0) } else { c64::new(0.0, 0.0) });
    let down =
        CMat::from_fn(4, 4, |a, b| if a == b && a >= 2 { c64::new(1.0, 0.0) } else { c64::new(0.0, 0.0) });
    [up, down]
}

/// Quantum spin Hall ribbon: spin-resolved local currents injected by the left lead, and the
/// spin-resolved transmissions, for an Anderson-disordered device and for one with polarised
/// magnetic impurities.
fn spin_currents(dir: &str) {
    let (l, w, e) = (48i64, 24i64, -0.05);
    let clean = bhz(BhzParams::default());
    let lat = clean.lattice().clone();
    let region = Region::cuboid(&lat, [l as usize, w as usize, 0]);
    let slices = region.slices(0);
    let interfaces = [
        Interface { slice: slices[0].clone(), outward: [-1, 0, 0] },
        Interface { slice: slices[l as usize - 1].clone(), outward: [1, 0, 0] },
    ];
    let interior: Vec<Site> =
        region.sites().iter().copied().filter(|s| s.cell[0] > 0 && s.cell[0] < l - 1).collect();
    let mut rng = Xoshiro256::seed_from_u64(11);
    let anderson =
        bhz(BhzParams::default()).with_onsite_term(Anderson::generate(&mut rng, &interior, 0.5).onsite(4));
    let mut rng = Xoshiro256::seed_from_u64(7);
    let impurities = MagneticImpurities::generate(
        &mut rng,
        &interior,
        (0.2 * interior.len() as f64).round() as usize,
        Moments::Polarized { mz: 1.0 },
    );
    let magnetic =
        bhz(BhzParams::default()).with_onsite_term(impurities.exchange(1.0, &bhz_spin(GFactors::Same)));
    let (mut rows, mut totals) = (Vec::new(), Vec::new());
    for (case, device) in [("anderson", anderson), ("magnetic", magnetic)] {
        let sys = NestedMultiTerminal::from_model(&device, &clean, &region, &interfaces);
        let states = sys.scattering_states(e).expect("scattering states");
        let sectors =
            sys.sector_transmissions(e, &[Some(vec![0, 0, 1, 1]), Some(vec![0, 0, 1, 1])]).expect("sectors");
        let at = |p: (usize, usize)| sectors.labels.iter().position(|&x| x == p).expect("label");
        for (k, (spin, op)) in ["up", "down"].into_iter().zip(spin_projectors()).enumerate() {
            totals
                .push(json!({"case": case, "spin": spin, "transmission": sectors.t[at((1, k))][at((0, k))]}));
            let bonds = sys.operator_currents(&states, &[1.0, 0.0], &op);
            for (site, j) in site_currents(&lat, &bonds) {
                let r = lat.position(site);
                rows.push(json!({"case": case, "spin": spin, "x": r[0], "y": r[1], "jx": j[0], "jy": j[1]}));
            }
        }
    }
    let imp: Vec<Value> = impurities
        .sites
        .iter()
        .map(|s| {
            let r = lat.position(*s);
            json!({"x": r[0], "y": r[1]})
        })
        .collect();
    write(dir, "spin_currents", &rows);
    write(dir, "spin_transmissions", &totals);
    write(dir, "impurities", &imp);
}

/// Transmission of a clean BHZ ribbon against energy: two helical edge channels in the gap.
fn bhz_conductance(dir: &str) {
    let model = bhz(BhzParams::default());
    let region = Region::cuboid(model.lattice(), [40, 40, 0]);
    let sys = NestedTwoTerminal::from_model(&model, &model, &region, 0);
    let energies: Vec<f64> = (0..=400).map(|k| -0.5 + 0.0025 * k as f64).collect();
    let rows: Vec<Value> = energies
        .iter()
        .zip(sys.transmissions(&energies))
        .filter_map(|(e, t)| t.ok().map(|t| json!({"energy": e, "transmission": t})))
        .collect();
    write(dir, "bhz_conductance", &rows);
}

/// Hall bar on a square lattice in a field `b` (flux per plaquette): the local current of a
/// unit current from the left lead to the grounded right one, the bottom and top leads being
/// voltage probes, and the Hall and two-terminal resistances against energy.
fn hall_bar(dir: &str) {
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
    let source = [1.0, -1.0, 0.0, 0.0];
    let e = -3.8;
    let voltages = lead_voltages(&sys.transmissions(e).expect("transmissions"), &source, 1);
    let states = sys.scattering_states(e).expect("scattering states");
    let currents: Vec<Value> = site_currents(&lat, &sys.bond_currents(&states, &voltages))
        .into_iter()
        .map(|(site, j)| json!({"x": site.cell[0], "y": site.cell[1], "jx": j[0], "jy": j[1]}))
        .collect();
    write(dir, "hall_currents", &currents);
    let plateaus: Vec<Value> = (0..=140)
        .map(|k| -3.95 + 0.01 * k as f64)
        .filter_map(|e| {
            let t = sys.transmissions(e).ok()?;
            let v = lead_voltages(&t, &source, 1);
            // Below the lowest Landau level nothing is transmitted and the voltages diverge.
            let (r_xy, r_2t) = (v[3] - v[2], v[0] - v[1]);
            (r_2t.is_finite() && r_2t.abs() < 10.0).then(|| json!({"energy": e, "r_xy": r_xy, "r_2t": r_2t}))
        })
        .collect();
    write(dir, "hall_plateaus", &plateaus);
}

/// Local density of states of the scattering states in a zigzag graphene ribbon near the Dirac
/// point: the edge states.
fn graphene_edges(dir: &str) {
    let graphene = Model::new(Lattice::honeycomb(1.0, 1))
        .hopping(0, 1, [0, 0, 0], -1.0)
        .hopping(0, 1, [0, -1, 0], -1.0)
        .hopping(0, 1, [1, -1, 0], -1.0);
    let lat = graphene.lattice().clone();
    let region = Region::cuboid(&lat, [24, 16, 0]);
    let sys = NestedTwoTerminal::from_model(&graphene, &graphene, &region, 0);
    let states = sys.scattering_states(0.05).expect("scattering states");
    let ldos = sys.site_densities(&states, &[1.0, 1.0], None);
    let rows: Vec<Value> = region
        .sites()
        .iter()
        .zip(&ldos)
        .map(|(site, rho)| {
            let r = lat.position(*site);
            json!({"x": r[0], "y": r[1], "sublattice": if site.sub == 0 { "A" } else { "B" }, "ldos": rho})
        })
        .collect();
    write(dir, "graphene_ldos", &rows);
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| ".".into());
    std::fs::create_dir_all(&dir).expect("output directory");
    spin_currents(&dir);
    bhz_conductance(&dir);
    hall_bar(&dir);
    graphene_edges(&dir);
}
