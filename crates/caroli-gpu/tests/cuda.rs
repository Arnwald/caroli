//! CUDA kernels against the f64 CPU implementations (`--features cuda`). Skipped where there is
//! no CUDA device.
#![cfg(feature = "cuda")]

use caroli_disorder::rng::Xoshiro256;
use caroli_gpu::CudaDevice;
use caroli_lattice::{Region, Site};
use caroli_model::presets::{BhzParams, bhz};
use caroli_model::{CMat, SparseMatrix, c64};

/// Disordered BHZ torus (Anderson, W = 2), rescaled: `(H̃, centre, half width)`.
fn system(n: usize) -> (SparseMatrix, f64, f64) {
    let mut rng = Xoshiro256::seed_from_u64(11);
    let v: Vec<f64> = (0..n * n).map(|_| 2.0 * rng.uniform() - 1.0).collect();
    let anderson = move |s: &Site| {
        let x = v[s.cell[0] as usize + n * s.cell[1] as usize];
        Some(CMat::from_fn(4, 4, |a, b| if a == b { c64::new(x, 0.0) } else { c64::new(0.0, 0.0) }))
    };
    let model = bhz(BhzParams { lambda: 0.1, ..BhzParams::default() }).with_onsite_term(anderson);
    let region = Region::cuboid(model.lattice(), [n, n, 0]);
    let h = model.sparse_bloch_hamiltonian(&region, &[[n as i64, 0, 0], [0, n as i64, 0]], &[0.0, 0.0]);
    let (lo, hi) = h.spectral_bounds();
    let (centre, half) = ((hi + lo) / 2.0, (hi - lo) / 2.0 * 1.01);
    (h.rescaled(centre, half), centre, half)
}

fn device() -> Option<CudaDevice> {
    match CudaDevice::new(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("skipped: {e}");
            None
        }
    }
}

#[test]
fn kpm_moments_match_cpu() {
    let Some(gpu) = device() else { return };
    let (h, _, _) = system(24);
    let mut rng = Xoshiro256::seed_from_u64(5);
    let starts: Vec<Vec<c64>> = (0..3).map(|_| caroli_kpm::random_phases(&mut rng, h.n)).collect();
    let kernel = caroli_kpm::jackson(400);
    let cpu = caroli_kpm::block_moments(|x, y| h.apply_block(x, y, 3), &starts, &kernel);
    let mu = gpu.upload(&h).unwrap().kpm_moments(&starts, &kernel).unwrap();
    let err = mu.iter().zip(&cpu).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max) / cpu[0].norm();
    println!("{}: max |Δμ_n|/μ_0 = {err:.1e}", gpu.name());
    assert!(err < 1e-12, "{err:e}");
}

#[test]
fn propagation_matches_cpu() {
    let Some(gpu) = device() else { return };
    let (h, centre, half) = system(16);
    let cheb = caroli_dynamics::Chebyshev::new(0.5, centre, half, 1e-14);
    let norm = (h.n as f64).sqrt();
    let psi0: Vec<c64> = (0..h.n).map(|i| c64::from_polar(1.0 / norm, 0.37 * (i * i) as f64)).collect();
    let steps = 20;
    let mut cpu = psi0.clone();
    for _ in 0..steps {
        cheb.step(|x, y| h.apply(x, y), &mut cpu);
    }
    let mut psi = caroli_kpm::interleave(&[psi0.clone(), psi0]);
    gpu.upload(&h).unwrap().propagate(&cheb.coefficients, &mut psi, 2, steps).unwrap();
    let err = caroli_kpm::deinterleave(&psi, 2)
        .iter()
        .map(|p| p.iter().zip(&cpu).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>().sqrt())
        .fold(0.0, f64::max);
    println!("{}: ‖Δψ‖ after {steps} steps = {err:.1e}", gpu.name());
    assert!(err < 1e-12, "{err:e}");
}
