//! Timings of the Chebyshev kernels: CPU (f64, all cores) against Metal (f32, double-single).
//!
//! `cargo run --release -p caroli-gpu --example bench -- [n] [moments] [vectors]`
//! (defaults: 512×512 BHZ torus with Anderson disorder, 512 moments, 4 random vectors).

use caroli_disorder::rng::Xoshiro256;
use caroli_lattice::{Region, Site};
use caroli_model::presets::{BhzParams, bhz};
use caroli_model::{CMat, c64};
use std::time::Instant;

fn main() {
    let args: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().expect("integer argument")).collect();
    let n = args.first().copied().unwrap_or(512);
    let moments = args.get(1).copied().unwrap_or(512);
    let vectors = args.get(2).copied().unwrap_or(4);

    let mut rng = Xoshiro256::seed_from_u64(1);
    let v: Vec<f64> = (0..n * n).map(|_| 2.0 * rng.uniform() - 1.0).collect();
    let anderson = move |s: &Site| {
        let x = v[s.cell[0] as usize + n * s.cell[1] as usize];
        Some(CMat::from_fn(4, 4, |a, b| if a == b { c64::new(x, 0.0) } else { c64::new(0.0, 0.0) }))
    };
    let model = bhz(BhzParams::default()).with_onsite_term(anderson);
    let region = Region::cuboid(model.lattice(), [n, n, 0]);
    let t = Instant::now();
    let h = model.sparse_bloch_hamiltonian(&region, &[[n as i64, 0, 0], [0, n as i64, 0]], &[0.0, 0.0]);
    let (lo, hi) = h.spectral_bounds();
    let h = h.rescaled((hi + lo) / 2.0, (hi - lo) / 2.0 * 1.01);
    println!("{n}×{n} BHZ torus: {} orbitals, {} non-zeros, built in {:.2?}", h.n, h.nnz(), t.elapsed());
    let starts: Vec<Vec<c64>> = (0..vectors).map(|_| caroli_kpm::random_phases(&mut rng, h.n)).collect();
    let kernel = caroli_kpm::jackson(moments);

    let t = Instant::now();
    let cpu = caroli_kpm::block_moments(|x, y| h.apply_block(x, y, vectors), &starts, &kernel);
    let cpu_time = t.elapsed();
    println!(
        "CPU f64 ({} threads): {moments} moments × {vectors} vectors in {cpu_time:.2?}",
        rayon_threads()
    );

    #[cfg(target_os = "macos")]
    {
        use caroli_gpu::{MetalDevice, Precision};
        let gpu = MetalDevice::new().expect("Metal device");
        for precision in [Precision::Single, Precision::DoubleSingle] {
            let op = gpu.upload(&h, precision).unwrap();
            let _ = op.kpm_moments(&starts, &kernel[..2]).unwrap(); // warm-up
            let t = Instant::now();
            let mu = op.kpm_moments(&starts, &kernel).unwrap();
            let time = t.elapsed();
            let err = mu.iter().zip(&cpu).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max) / cpu[0].norm();
            println!(
                "{} {precision:?}: {time:.2?} ({:.1}× the CPU), max |Δμ|/μ_0 = {err:.1e}",
                gpu.name(),
                cpu_time.as_secs_f64() / time.as_secs_f64()
            );
        }
    }
}

fn rayon_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}
