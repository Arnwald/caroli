//! Multi-terminal nested dissection against the dense direct solve (`MultiTerminal`): a
//! disordered square cross with four leads, a BHZ cross (two spin blocks), graphene with a
//! narrow lead (slanted bonds), and two leads at the ends of a ribbon against
//! `NestedTwoTerminal`.

use caroli_green::{Interface, MultiTerminal, NestedMultiTerminal, NestedTwoTerminal, SelfEnergyMethod};
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

fn sites(lat: &Lattice, cells: impl Iterator<Item = (i64, i64)>) -> Vec<Site> {
    let subs = lat.sublattices().len();
    cells.flat_map(|(x, y)| (0..subs).map(move |s| Site { cell: [x, y, 0], sub: s })).collect()
}

/// An `nx × ny` square with arms of length `arm` on its four edges, as wide as the edges, and a
/// lead at the end of each arm.
fn cross(lat: &Lattice, nx: i64, ny: i64, arm: i64) -> (Region, Vec<Interface>) {
    let cells = (-arm..nx + arm)
        .flat_map(|x| (-arm..ny + arm).map(move |y| (x, y)))
        .filter(|&(x, y)| (0..ny).contains(&y) || (0..nx).contains(&x));
    let region = Region::from_sites(lat, sites(lat, cells));
    let column = |x: i64| Region::from_sites(lat, sites(lat, (0..ny).map(|y| (x, y))));
    let row = |y: i64| Region::from_sites(lat, sites(lat, (0..nx).map(|x| (x, y))));
    let interfaces = vec![
        Interface { slice: column(-arm), outward: [-1, 0, 0] },
        Interface { slice: row(ny + arm - 1), outward: [0, 1, 0] },
        Interface { slice: column(nx + arm - 1), outward: [1, 0, 0] },
        Interface { slice: row(-arm), outward: [0, -1, 0] },
    ];
    (region, interfaces)
}

/// Every `T_pq` of the nested solver against the dense one.
fn compare(
    device: &Model,
    lead: &Model,
    region: &Region,
    interfaces: &[Interface],
    energies: &[f64],
    blocks: usize,
) {
    let dense = MultiTerminal::from_model(device, lead, region, interfaces);
    let nested = NestedMultiTerminal::from_model(device, lead, region, interfaces);
    assert_eq!((nested.block_count(), nested.lead_count()), (blocks, interfaces.len()));
    for &e in energies {
        let sol = dense.solve(e, SelfEnergyMethod::Modes).unwrap();
        let t = nested.transmissions(e).unwrap();
        for (p, row) in t.iter().enumerate() {
            for (q, &tpq) in row.iter().enumerate() {
                let d = sol.transmission(&dense, p, q);
                assert!((tpq - d).abs() < 1e-9 * (1.0 + d.abs()), "E = {e}, T[{p}][{q}]: {tpq} vs {d}");
            }
        }
    }
}

#[test]
fn square_cross() {
    let (region, interfaces) = cross(square().lattice(), 9, 7, 3);
    compare(&disordered(square(), 1.5, 1), &square(), &region, &interfaces, &[-2.7, -0.4, 0.0, 1.3], 1);
}

#[test]
fn bhz_cross() {
    let clean = bhz(BhzParams::default());
    let (region, interfaces) = cross(clean.lattice(), 8, 6, 2);
    let device = disordered(bhz(BhzParams::default()), 0.5, 4);
    compare(&device, &clean, &region, &interfaces, &[-0.2, 0.02, 0.3], 2);
}

/// A narrow lead (two rows of cells) on a graphene flake, and a wide one on the other side:
/// device sites outside the interface slice bond to the first lead cell.
#[test]
fn graphene_narrow_lead() {
    let g = graphene();
    let lat = g.lattice().clone();
    let cells = (0..7)
        .flat_map(|x| (0..6).map(move |y| (x, y)))
        .chain((-2..0).flat_map(|x| (2..4).map(move |y| (x, y))));
    let region = Region::from_sites(&lat, sites(&lat, cells));
    let interfaces = [
        Interface {
            slice: Region::from_sites(&lat, sites(&lat, (2..4).map(|y| (-2, y)))),
            outward: [-1, 0, 0],
        },
        Interface {
            slice: Region::from_sites(&lat, sites(&lat, (0..6).map(|y| (6, y)))),
            outward: [1, 0, 0],
        },
    ];
    compare(&disordered(graphene(), 0.8, 1), &g, &region, &interfaces, &[-1.4, -0.3, 0.45, 1.3], 1);
}

/// Two leads at the ends of a disordered ribbon: the same transmission as the two-terminal
/// solver, in both directions.
#[test]
fn two_leads_as_two_terminal() {
    let (sq, device) = (square(), disordered(square(), 2.0, 1));
    let region = Region::cuboid(sq.lattice(), [14, 6, 0]);
    let slices = region.slices(0);
    let interfaces = [
        Interface { slice: slices[0].clone(), outward: [-1, 0, 0] },
        Interface { slice: slices[13].clone(), outward: [1, 0, 0] },
    ];
    let multi = NestedMultiTerminal::from_model(&device, &sq, &region, &interfaces);
    let two = NestedTwoTerminal::from_model(&device, &sq, &region, 0);
    for e in [-2.9, -1.1, 0.0, 0.7] {
        let (t, t2) = (multi.transmissions(e).unwrap(), two.transmission(e).unwrap());
        assert!((t[1][0] - t2).abs() < 1e-10 && (t[0][1] - t2).abs() < 1e-10, "E = {e}: {:?} vs {t2}", t);
    }
}
