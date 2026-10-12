//! Lead modes on the hard cases found in review: wide graphene ribbons cut along a slanted
//! direction (low-rank couplings), subband edges, bound states of the lead surface (graphene at
//! E = 0) and nearly degenerate channels. Clean ribbons are checked against the number of bands
//! crossing the energy with positive velocity, counted on the band structure of the lead.

use caroli_green::{Lead, SelfEnergyMethod, TwoTerminal};
use caroli_lattice::{Lattice, Region};
use caroli_model::{CMat, Model, c64};

fn one(x: f64) -> CMat {
    CMat::from_fn(1, 1, |_, _| c64::new(x, 0.0))
}

/// Right-moving channels of a lead at `e`: upward crossings of `e` by the sorted bands of
/// `H(k) = h00 + h01 e^{ik} + h01† e^{−ik}`.
fn band_count(lead: &Lead, e: f64, nk: usize) -> usize {
    let ks: Vec<f64> =
        (0..=nk).map(|j| -std::f64::consts::PI + 2.0 * std::f64::consts::PI * j as f64 / nk as f64).collect();
    let all: Vec<Vec<f64>> = ks.iter().map(|&k| lead.bands(k)).collect();
    all.windows(2).map(|w| w[0].iter().zip(&w[1]).filter(|(a, b)| **a < e && **b > e).count()).sum()
}

fn graphene() -> Model {
    Model::new(Lattice::honeycomb(1.0, 1))
        .hopping(0, 1, [0, 0, 0], one(-1.0))
        .hopping(0, 1, [0, -1, 0], one(-1.0))
        .hopping(0, 1, [1, -1, 0], one(-1.0))
}

/// 100-site-wide graphene ribbon cut along the honeycomb axis 0 (slanted slices, coupling of
/// low rank): a clean ribbon transmits one per channel.
#[test]
fn wide_graphene_ribbon() {
    let g = graphene();
    let sys = TwoTerminal::from_model(&g, &g, &Region::cuboid(g.lattice(), [3, 100, 0]), 0);
    for e in [-1.7504, 2.7043, -0.1697, 0.83] {
        let t = sys.transmission(e).unwrap();
        let channels = band_count(&sys.left, e, 1500);
        assert!((t - channels as f64).abs() < 1e-6, "E {e}: T = {t}, {channels} channels");
    }
}

/// Energies on subband edges of a square ribbon (and the bottom of a chain band): the
/// transmission is the limit from above.
#[test]
fn subband_edges() {
    let sq = Model::new(Lattice::square(1.0, 1)).hopping(0, 0, [1, 0, 0], one(-1.0)).hopping(
        0,
        0,
        [0, 1, 0],
        one(-1.0),
    );
    let sys = TwoTerminal::from_model(&sq, &sq, &Region::cuboid(sq.lattice(), [4, 5, 0]), 0);
    for e in [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0] {
        let t = sys.transmission(e).unwrap();
        let channels = band_count(&sys.left, e + 1e-7, 4000);
        assert!((t - channels as f64).abs() < 1e-6, "E {e}: T = {t}, {channels} channels just above");
    }
    let chain = Model::new(Lattice::square(1.0, 1)).hopping(0, 0, [1, 0, 0], one(-1.0));
    let sys = TwoTerminal::from_model(&chain, &chain, &Region::cuboid(chain.lattice(), [5, 1, 0]), 0);
    let t = sys.transmission(2.0 - 1e-13).unwrap();
    assert!((t - 1.0).abs() < 1e-6, "chain just below the top of the band: T = {t}");
}

/// BHZ ribbon at eigenvalues of H(k = 0) (subband edges of the lead).
#[test]
fn bhz_subband_edges() {
    use caroli_model::presets::{BhzParams, bhz};
    let m = bhz(BhzParams::default());
    let sys = TwoTerminal::from_model(&m, &m, &Region::cuboid(m.lattice(), [3, 6, 0]), 0);
    for e in [-0.570410700056883, -0.660747457925355] {
        let t = sys.transmission(e).unwrap();
        let channels = band_count(&sys.left, e + 1e-7, 4000);
        assert!((t - channels as f64).abs() < 1e-5, "E {e}: T = {t}, {channels} channels just above");
    }
}

/// Armchair graphene ribbons on a rectangular 4-site cell at E = 0, where the zigzag ends of
/// the lead slices carry bound states: metallic (W = 4) and gapped (W = 5), clean and
/// disordered.
#[test]
fn graphene_at_zero_energy() {
    let s3 = 3f64.sqrt();
    let lat = Lattice::new(&[[s3, 0.0, 0.0], [0.0, 3.0, 0.0]])
        .with_sublattice("A1", [0.0, 0.0, 0.0], 1)
        .with_sublattice("B1", [0.0, 1.0, 0.0], 1)
        .with_sublattice("A2", [s3 / 2.0, 1.5, 0.0], 1)
        .with_sublattice("B2", [s3 / 2.0, 2.5, 0.0], 1);
    let t = one(-1.0);
    let armchair = || {
        Model::new(lat.clone())
            .hopping(0, 1, [0, 0, 0], t.clone())
            .hopping(0, 3, [0, -1, 0], t.clone())
            .hopping(0, 3, [-1, -1, 0], t.clone())
            .hopping(2, 1, [0, 0, 0], t.clone())
            .hopping(2, 1, [1, 0, 0], t.clone())
            .hopping(2, 3, [0, 0, 0], t.clone())
    };
    for (w, expected) in [(4usize, 1.0), (5, 0.0)] {
        let m = armchair();
        let sys = TwoTerminal::from_model(&m, &m, &Region::cuboid(m.lattice(), [w, 3, 0]), 1);
        let t0 = sys.transmission(0.0).unwrap();
        assert!((t0 - expected).abs() < 1e-6, "W = {w}: T(0) = {t0}, expected {expected}");
    }
    // Disordered metallic ribbon: T(0) is the limit of T(E) at E → 0.
    let pot = |s: &caroli_lattice::Site| {
        Some(one(0.3 * (((s.cell[0] * 7 + s.cell[1] * 3 + s.sub as i64) % 5) as f64 - 2.0) / 2.0))
    };
    let clean = armchair();
    let dirty = armchair().with_onsite_term(pot);
    let sys = TwoTerminal::from_model(&dirty, &clean, &Region::cuboid(clean.lattice(), [4, 3, 0]), 1);
    let (t0, t1) = (sys.transmission(0.0).unwrap(), sys.transmission(1e-7).unwrap());
    assert!((t0 - t1).abs() < 1e-5 && (0.0..=1.0 + 1e-9).contains(&t0), "T(0) = {t0}, T(1e-7) = {t1}");
}

/// Two decoupled chains whose channels have nearly equal Bloch phases: the transmission is the
/// sum of the two chains, whether the phases are grouped (Δk = 5e-8) or not (Δk = 9e-7).
#[test]
fn nearly_degenerate_channels() {
    let e = 0.37;
    let k1 = (-e / 2.0f64).acos();
    for dk in [5e-8, 9e-7] {
        let eps2 = e + 2.6 * (k1 + dk).cos();
        let diag = |a: f64, b: f64| {
            CMat::from_fn(2, 2, |i, j| {
                if i != j { c64::new(0.0, 0.0) } else { c64::new(if i == 0 { a } else { b }, 0.0) }
            })
        };
        let h00 = diag(0.0, eps2);
        // A uniform phase on the coupling keeps the transmission and the near degeneracy, and
        // makes the coupling non-Hermitian: the general algorithm and its grouping are tested.
        let h01 = diag(-1.0, -1.3) * faer::Scale(c64::from_polar(1.0, 0.4));
        let mut slices = vec![h00.clone(); 5];
        slices[2] = diag(0.9, eps2 - 1.1);
        let sys = TwoTerminal {
            slices,
            couplings: vec![h01.clone(); 4],
            left: Lead { h00: h00.clone(), h01: h01.clone() },
            right: Lead { h00, h01 },
            regions: Vec::new(),
        };
        let t = sys.transmission(e).unwrap();
        let split = sys.transmission_by_blocks(e, SelfEnergyMethod::Modes).unwrap();
        assert!((t - split).abs() < 1e-11, "Δk = {dk}: {t} vs {split}");
    }
}

/// Leads with a Hermitian definite coupling take the fast path: its self-energies, channels and
/// velocities agree with the general algorithm on a square strip (also on a subband edge), with
/// degenerate (spinful) channels, and on random Hermitian cells with couplings of either sign.
#[test]
fn definite_couplings_fast_path() {
    use caroli_green::modes::{LeadModes, sigma_left, sigma_right};
    let zero = c64::new(0.0, 0.0);
    let n = 7;
    let strip = CMat::from_fn(n, n, |i, j| if i.abs_diff(j) == 1 { c64::new(-1.0, 0.0) } else { zero });
    let minus_one = CMat::from_fn(n, n, |i, j| if i == j { c64::new(-1.0, 0.0) } else { zero });
    let spinful = |m: &CMat| {
        CMat::from_fn(
            2 * m.nrows(),
            2 * m.ncols(),
            |i, j| if i % 2 == j % 2 { m[(i / 2, j / 2)] } else { zero },
        )
    };
    // Bottom of the second subband of the strip: ε_2 − 2, ε_q = −2 cos(qπ/(n + 1)).
    let edge = -2.0 * (2.0 * std::f64::consts::PI / (n + 1) as f64).cos() - 2.0;
    let entry = |seed: usize, i: usize, j: usize| {
        c64::new(
            ((i * 7 + j * 13 + seed) % 17) as f64 / 17.0 - 0.5,
            ((i * 3 + j * 5 + 2 * seed) % 11) as f64 / 11.0 - 0.5,
        )
    };
    let m = 6;
    let a = CMat::from_fn(m, m, |i, j| entry(1, i, j));
    let cell = CMat::from_fn(m, m, |i, j| a[(i, j)] + a[(j, i)].conj());
    let b = CMat::from_fn(m, m, |i, j| entry(2, i, j));
    let bb = &b * b.adjoint();
    let p = CMat::from_fn(m, m, |i, j| bb[(i, j)] + if i == j { c64::new(0.5, 0.0) } else { zero });
    let minus_p = CMat::from_fn(m, m, |i, j| -p[(i, j)]);
    let random = vec![-2.0, -0.7, 0.0, 0.4, 1.3, 2.5];
    let cases = [
        (
            "strip",
            strip.clone(),
            minus_one.clone(),
            vec![-4.5, -3.9, -2.5, -1.0, 0.0, 0.3, 1.7, 3.2, 4.5, edge],
        ),
        ("spinful strip", spinful(&strip), spinful(&minus_one), vec![-2.5, 0.3, 1.7]),
        ("random, H1 = -P", cell.clone(), minus_p, random.clone()),
        ("random, H1 = +P", cell, p, random),
    ];
    let sorted = |v: &[f64]| {
        let mut v = v.to_vec();
        v.sort_by(f64::total_cmp);
        v
    };
    for (name, h0, h1, energies) in cases {
        for e in energies {
            let fast = LeadModes::new(&h0, &h1, e).unwrap();
            let general = LeadModes::at_general(&h0, &h1, fast.energy).unwrap();
            let k = h0.nrows();
            // The fast path factors the coupling by Cholesky: u is lower triangular.
            assert!((0..k).all(|i| (i + 1..k).all(|j| fast.u[(i, j)] == zero)), "{name}: general path taken");
            assert_eq!(fast.energy, general.energy, "{name}, E = {e}");
            let dr = (sigma_right(&h1, &fast) - sigma_right(&h1, &general)).norm_max();
            let dl = (sigma_left(&h1, &fast) - sigma_left(&h1, &general)).norm_max();
            assert!(dr < 1e-10 && dl < 1e-10, "{name}, E = {e}: ΔΣ_R = {dr:e}, ΔΣ_L = {dl:e}");
            assert_eq!(fast.channels(), general.channels(), "{name}, E = {e}");
            for (x, y) in [
                (&fast.right_velocities, &general.right_velocities),
                (&fast.left_velocities, &general.left_velocities),
            ] {
                assert!(
                    sorted(x).iter().zip(sorted(y)).all(|(a, b)| (a - b).abs() < 1e-9),
                    "{name}, E = {e}: velocities"
                );
            }
        }
    }
}

/// The propagating modes solve the Bloch equation `H1† λ⁻¹ φ + (H0 − E) φ + H1 λ φ = 0` and are
/// orthonormal for the current `J(a, b) = i(φ_a† H1 φ_b λ_b − λ̄_a φ_a† H1† φ_b)`: +1 going right,
/// −1 going left, 0 between different modes. Fast path (strip), degenerate channels (spinful
/// strip, BHZ) and the general algorithm (graphene, a field).
#[test]
fn propagating_modes_carry_unit_current() {
    use caroli_green::modes::LeadModes;
    let zero = c64::new(0.0, 0.0);
    let square = Model::new(Lattice::square(1.0, 1)).hopping(0, 0, [1, 0, 0], one(-1.0)).hopping(
        0,
        0,
        [0, 1, 0],
        one(-1.0),
    );
    let bhz = caroli_model::presets::bhz(caroli_model::presets::BhzParams::default());
    let field = Model::new(Lattice::square(1.0, 1))
        .hopping(0, 0, [1, 0, 0], one(-1.0))
        .hopping(0, 0, [0, 1, 0], one(-1.0))
        .with_field(0.2);
    let lead_of = |m: &Model, w: usize| {
        let region = Region::cuboid(m.lattice(), [3, w, 0]);
        TwoTerminal::from_model(m, m, &region, 0).left
    };
    let leads = [
        ("strip", lead_of(&square, 9), vec![-2.7, -0.4, 1.1]),
        ("BHZ", lead_of(&bhz, 7), vec![-0.3, 0.02, 0.6]),
        ("graphene", lead_of(&graphene(), 6), vec![-1.3, 0.45, 2.1]),
        ("field", lead_of(&field, 12), vec![-3.6, -2.1, 0.7]),
    ];
    for (name, lead, energies) in leads {
        for e in energies {
            let m = LeadModes::new(&lead.h00, &lead.h01, e).unwrap();
            let (h0, h1) = (&lead.h00, &lead.h01);
            let n = h0.nrows();
            let modes: Vec<(Vec<c64>, c64, f64)> = (0..m.right_modes.ncols())
                .map(|c| (m.right_modes.col_as_slice(c).to_vec(), m.right_lambdas[c], 1.0))
                .chain(
                    (0..m.left_modes.ncols())
                        .map(|c| (m.left_modes.col_as_slice(c).to_vec(), m.left_lambdas[c], -1.0)),
                )
                .collect();
            assert_eq!(m.right_modes.ncols(), m.channels());
            for (phi, lambda, _) in &modes {
                let residual = (0..n)
                    .map(|i| {
                        (0..n)
                            .map(|j| {
                                h1[(j, i)].conj() * phi[j] / lambda
                                    + (h0[(i, j)] - if i == j { c64::new(e, 0.0) } else { zero }) * phi[j]
                                    + h1[(i, j)] * phi[j] * lambda
                            })
                            .sum::<c64>()
                            .norm()
                    })
                    .fold(0.0, f64::max);
                assert!(residual < 1e-8, "{name}, E = {e}: Bloch residual {residual:e}");
            }
            let current = |a: &(Vec<c64>, c64, f64), b: &(Vec<c64>, c64, f64)| {
                let (pa, la, _) = a;
                let (pb, lb, _) = b;
                let h1b: Vec<c64> = (0..n).map(|i| (0..n).map(|j| h1[(i, j)] * pb[j]).sum()).collect();
                let h1db: Vec<c64> =
                    (0..n).map(|i| (0..n).map(|j| h1[(j, i)].conj() * pb[j]).sum()).collect();
                let dot = |x: &[c64], y: &[c64]| x.iter().zip(y).map(|(p, q)| p.conj() * q).sum::<c64>();
                c64::new(0.0, 1.0) * (dot(pa, &h1b) * lb - la.conj() * dot(pa, &h1db))
            };
            for (k, a) in modes.iter().enumerate() {
                for (l, b) in modes.iter().enumerate() {
                    let expected = if k == l { a.2 } else { 0.0 };
                    let j = current(a, b);
                    assert!(
                        (j - c64::new(expected, 0.0)).norm() < 1e-7,
                        "{name}, E = {e}: J[{k}][{l}] = {j}"
                    );
                }
            }
        }
    }
}
