//! The figures of the benchmark page, computed with caroli:
//!
//! - local currents in a BHZ ribbon (quantum spin Hall effect) with magnetic impurities polarised
//!   along z, as in the thesis: the current injected by the left lead over the spin-up fraction of
//!   the injected density, for the same-g and opposite-g couplings, three regimes each;
//! - the conductance of a clean BHZ ribbon;
//! - a Hall bar in the quantum Hall regime: the local current and the Hall resistance;
//! - the edge states of a zigzag graphene ribbon.
//!
//! ```text
//! cargo run --release -p caroli --example gallery -- OUT_DIR [FIGURE...]
//! ```
//!
//! Each figure is a JSON array of rows in `OUT_DIR`. The figures are `spin_currents`,
//! `bhz_conductance`, `hall_bar` and `graphene_edges`; all of them by default.

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

/// `v` rounded to four significant digits, to keep the JSON small.
fn round4(v: f64) -> f64 {
    if v == 0.0 || !v.is_finite() {
        return v;
    }
    let scale = 10f64.powi(3 - v.abs().log10().floor() as i32);
    (v * scale).round() / scale
}

/// Local currents in a quantum spin Hall ribbon with magnetic impurities, as in the local-current
/// figure of the thesis: a 400 × 64 BHZ ribbon whose 256 central columns carry impurities on 20 %
/// of the sites, moments along z, between two clean buffers of 72 columns. The left lead injects
/// the current in a bias window of ±0.001 around the Fermi energy (ten energies). For the same-g
/// and opposite-g couplings, three exchange strengths J_I and Fermi energies each, from two
/// channels to almost none: on every third site, as means over 3 × 3 blocks, the current and the
/// spin-up fraction ρ↑/(ρ↑ + ρ↓) of the injected density; and the transmission and its
/// spin-resolved parts, means over the window.
fn spin_currents(dir: &str) {
    const BLOCK: i64 = 3;
    let (l, w, disordered) = (400i64, 64i64, 256i64);
    let buffer = (l - disordered) / 2;
    let clean = bhz(BhzParams::default());
    let lat = clean.lattice().clone();
    let region = Region::cuboid(&lat, [l as usize, w as usize, 0]);
    let slices = region.slices(0);
    let interfaces = [
        Interface { slice: slices[0].clone(), outward: [-1, 0, 0] },
        Interface { slice: slices[l as usize - 1].clone(), outward: [1, 0, 0] },
    ];
    let candidates: Vec<Site> = region
        .sites()
        .iter()
        .copied()
        .filter(|s| (buffer..buffer + disordered).contains(&s.cell[0]))
        .collect();
    let mut rng = Xoshiro256::seed_from_u64(7);
    let impurities = MagneticImpurities::generate(
        &mut rng,
        &candidates,
        (0.2 * candidates.len() as f64) as usize,
        Moments::Polarized { mz: 1.0 },
    );
    let [up, down] = spin_projectors();
    let block = |s: Site| (s.cell[0] / BLOCK, s.cell[1] / BLOCK);
    let (mut rows, mut totals) = (Vec::new(), Vec::new());
    // The panels of the thesis figure: coupling, panel (top to bottom), E_F, J_I.
    let panels = [
        (GFactors::Opposite, 1, 0.053333, 0.2),
        (GFactors::Opposite, 2, 0.053333, 0.666),
        (GFactors::Opposite, 3, 0.053333, 1.6),
        (GFactors::Same, 1, -0.08889, 0.2),
        (GFactors::Same, 2, -0.08889, 1.0714),
        (GFactors::Same, 3, 0.1, 1.4),
    ];
    for (coupling, panel, e_f, j_i) in panels {
        let name = if coupling == GFactors::Same { "same" } else { "opposite" };
        let device =
            bhz(BhzParams::default()).with_onsite_term(impurities.exchange(j_i, &bhz_spin(coupling)));
        let sys = NestedMultiTerminal::from_model(&device, &clean, &region, &interfaces);
        let energies: Vec<f64> = (0..10).map(|k| e_f - 0.0009 + 0.0002 * k as f64).collect();
        // Per block: x, y, jx, jy, ρ↑, ρ↓ summed over its sites (and the currents and densities
        // over the energies), and the number of sites.
        let mut sums: BTreeMap<(i64, i64), [f64; 7]> = BTreeMap::new();
        for site in region.sites() {
            let r = lat.position(*site);
            let b = sums.entry(block(*site)).or_insert([0.0; 7]);
            b[0] += r[0];
            b[1] += r[1];
            b[6] += 1.0;
        }
        let (mut t, mut t_up, mut t_down) = (0.0, 0.0, 0.0);
        for &e in &energies {
            let states = sys.scattering_states(e).expect("scattering states");
            for (site, j) in site_currents(&lat, &sys.bond_currents(&states, &[1.0, 0.0])) {
                let b = sums.get_mut(&block(site)).expect("block");
                b[2] += j[0];
                b[3] += j[1];
            }
            let rho_up = sys.site_densities(&states, &[1.0, 0.0], Some(&up));
            let rho_down = sys.site_densities(&states, &[1.0, 0.0], Some(&down));
            for ((site, ru), rd) in region.sites().iter().zip(&rho_up).zip(&rho_down) {
                let b = sums.get_mut(&block(*site)).expect("block");
                b[4] += ru;
                b[5] += rd;
            }
            let sectors = sys
                .sector_transmissions(e, &[Some(vec![0, 0, 1, 1]), Some(vec![0, 0, 1, 1])])
                .expect("sectors");
            let at = |p: (usize, usize)| sectors.labels.iter().position(|&x| x == p).expect("label");
            // From spin b of the left lead to spin a of the right one.
            let pair = |a: usize, b: usize| sectors.t[at((1, a))][at((0, b))];
            t += pair(0, 0) + pair(0, 1) + pair(1, 0) + pair(1, 1);
            t_up += pair(0, 0);
            t_down += pair(1, 1);
        }
        let ne = energies.len() as f64;
        totals.push(json!({
            "coupling": name, "panel": panel, "e_f": e_f, "j_i": j_i,
            "g": t / ne, "t_up": t_up / ne, "t_down": t_down / ne,
        }));
        for b in sums.values() {
            let n = b[6];
            rows.push(json!({
                "coupling": name, "panel": panel, "x": b[0] / n, "y": b[1] / n,
                "jx": round4(b[2] / (n * ne)), "jy": round4(b[3] / (n * ne)), "up": round4(b[4] / (b[4] + b[5])),
            }));
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

/// A figure: its name and the function that writes its rows.
type Figure = (&'static str, fn(&str));

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = args.first().cloned().unwrap_or_else(|| ".".into());
    let only = args.get(1..).unwrap_or_default();
    std::fs::create_dir_all(&dir).expect("output directory");
    let figures: [Figure; 4] = [
        ("spin_currents", spin_currents),
        ("bhz_conductance", bhz_conductance),
        ("hall_bar", hall_bar),
        ("graphene_edges", graphene_edges),
    ];
    for (name, figure) in figures {
        if only.is_empty() || only.iter().any(|o| o == name) {
            figure(&dir);
        }
    }
}
