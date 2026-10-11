//! BHZ spectra against the thesis codes: exact diagonalisation of a flake (`ti-diag`) and
//! ribbon bands (`ti-ribbon-bands`). Both are deterministic, so they pin down the Hamiltonian
//! conventions (Γ matrices, hoppings, on-site terms) to machine precision.

use caroli_lattice::Region;
use caroli_model::gamma::gamma12;
use caroli_model::presets::{BhzParams, bhz};
use caroli_model::{CMat, c64};
use caroli_validation::golden_f64;
use faer::Side;

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// `ti-diag`: clean 6×6 BHZ flake, open boundaries, 144 eigenvalues.
#[test]
fn flake_spectrum_matches_ti_diag() {
    let golden = &golden_f64("ti-diag", "valpropres_00.dat")[0];
    let model = bhz(BhzParams::default());
    let region = Region::cuboid(model.lattice(), [6, 6, 0]);
    let h = model.hamiltonian(&region);
    let e = h.self_adjoint_eigenvalues(Side::Lower).unwrap();
    let d = max_diff(&e, golden);
    assert!(d < 1e-10, "max |ΔE| = {d:e}");
}

/// `ti-ribbon-bands`: 16-site ribbon (open in x, periodic in y) with a uniform exchange
/// J_s Γ12, J_s = 0.17. The golden file holds ky(Nx, Ny) then the eigenvalues (4Nx, Ny).
#[test]
fn ribbon_bands_match_ti_ribbon_bands() {
    let (nx, ny) = (16usize, 32usize);
    let rec = &golden_f64("ti-ribbon-bands", "valpropres_00.dat")[0];
    assert_eq!(rec.len(), nx * ny + 4 * nx * ny);
    let (ky, bands) = rec.split_at(nx * ny);
    let js = 0.17;
    let model = bhz(BhzParams::default()).with_onsite_term(move |_: &caroli_lattice::Site| {
        Some(CMat::from_fn(4, 4, |i, j| c64::new(js, 0.0) * gamma12()[(i, j)]))
    });
    let ribbon = Region::cuboid(model.lattice(), [nx, 1, 0]);
    let mut worst: f64 = 0.0;
    for iy in 0..ny {
        let k = ky[iy * nx]; // ky(1, iy)
        let h = model.bloch_hamiltonian(&ribbon, &[[0, 1, 0]], &[k]);
        let e = h.self_adjoint_eigenvalues(Side::Lower).unwrap();
        worst = worst.max(max_diff(&e, &bands[iy * 4 * nx..(iy + 1) * 4 * nx]));
    }
    assert!(worst < 1e-10, "max |ΔE| = {worst:e}");
}
