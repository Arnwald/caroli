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

/// `ti-ribbon-bands-fourier-y` (2013 ancestor): 16-site ribbon, M = −0.25, inter-block term
/// λ = −0.1. The 2013 Γ matrices are the transposes of the later ones (Γ44 = −τ_y), and its
/// Γ14 is the anti-Hermitian −i s_y; LAPACK's `heev` reads the upper triangle only, so the
/// program diagonalises the Hermitian matrix with y term `sin k_y τ_y − λ sin k_y s_x`. In
/// caroli terms the y hopping is `DΓ0 + BΓ55 − (i/2)(Γ44 − λ Γ23)`, the rest is the BHZ preset.
#[test]
fn ribbon_bands_match_2013_fourier_y() {
    use caroli_model::gamma::{gamma0, gamma23, gamma44, gamma55};
    use caroli_model::{Model, combine};
    let n = 16usize;
    let rec = &golden_f64("ti-ribbon-bands-fourier-y", "valpropres_00.dat")[0];
    assert_eq!(rec.len(), n * n + 4 * n * n);
    let (ky, bands) = rec.split_at(n * n);
    let p = BhzParams { mass: -0.25, lambda: -0.1, ..BhzParams::default() };
    let preset = bhz(p);
    let (r, i) = (|x: f64| c64::new(x, 0.0), |x: f64| c64::new(0.0, x));
    let ty = combine(&[
        (r(p.d), &gamma0()),
        (r(p.b), &gamma55()),
        (i(-0.5 * p.a), &gamma44()),
        (i(0.5 * p.a * p.lambda), &gamma23()),
    ]);
    let mut model = Model::new(preset.lattice().clone());
    for hop in preset.hoppings().iter().filter(|h| h.offset != [0, 1, 0]) {
        model = model.hopping(hop.from, hop.to, hop.offset, hop.matrix.clone());
    }
    let cell = caroli_lattice::Site { cell: [0, 0, 0], sub: 0 };
    let model = model.onsite(0, preset.onsite_matrix(&cell)).hopping(0, 0, [0, 1, 0], ty);
    let ribbon = Region::cuboid(model.lattice(), [n, 1, 0]);
    let mut worst: f64 = 0.0;
    for iy in 0..n {
        let k = ky[iy * n]; // ky(1, iy); the program uses ky(iy, iy), equal for Nx = Ny
        let h = model.bloch_hamiltonian(&ribbon, &[[0, 1, 0]], &[k]);
        let e = h.self_adjoint_eigenvalues(Side::Lower).unwrap();
        worst = worst.max(max_diff(&e, &bands[iy * 4 * n..(iy + 1) * 4 * n]));
    }
    println!("ribbon-bands-fourier-y: max |ΔE| = {worst:.1e}");
    assert!(worst < 1e-10, "max |ΔE| = {worst:e}");
}

/// `ti-dos-spin-lesser` (2013 recursive code), replayed as it actually computes (see
/// `caroli_validation::thesis::DosSpinLesser2013`): clean 8×8 sample, 16 energies.
#[test]
fn dos_2013_recursion() {
    use caroli_validation::thesis::DosSpinLesser2013;
    let rec = &golden_f64("ti-dos-spin-lesser", "dos_00.dat")[0];
    let ne = 16;
    assert_eq!(rec.len(), 3 * ne);
    let p =
        DosSpinLesser2013 { nx: 8, ny: 8, ne, params: BhzParams { mass: -0.137, ..BhzParams::default() } };
    let (e, up, down) = p.run();
    let scale = rec.iter().fold(0f64, |m, x| m.max(x.abs()));
    let rel = |a: &[f64], b: &[f64]| max_diff(a, b) / scale;
    let (de, du, dd) = (max_diff(&e, &rec[..ne]), rel(&up, &rec[ne..2 * ne]), rel(&down, &rec[2 * ne..]));
    println!("dos-spin-lesser: |ΔE| ≤ {de:.1e}, relative |Δ up| ≤ {du:.1e}, |Δ down| ≤ {dd:.1e}");
    assert!(de < 1e-14 && du < 1e-9 && dd < 1e-9);
}
