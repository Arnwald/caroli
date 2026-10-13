//! Kernel-polynomial moments on the GPU, checked against the CPU.
//!
//! The GPU computes the moments of every start vector; the CPU recomputes those of a sample
//! (one vector in fifty, at least one) in f64, and the GPU those of the same sample. If the two
//! differ by more than the tolerance (relative to `μ_0`), or the GPU fails, the CPU computes
//! everything. With a single start vector the check costs a full CPU run: for a local density of
//! states, batch the sites as start vectors.

use crate::block_moments;
use caroli_gpu::{Device, Options, Report};
use caroli_model::{SparseMatrix, c64};

/// The checked start vectors: one in fifty, evenly spaced, at least one.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn sample(vectors: usize) -> Vec<usize> {
    let count = vectors.div_ceil(50);
    (0..count).map(|k| k * vectors / count).collect()
}

/// `max_n |a_n − b_n| / |b_0|`.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn deviation(a: &[c64], b: &[c64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).norm()).fold(0.0, f64::max) / b[0].norm()
}

/// Moments `μ_n = kernel_n · mean_v ⟨r_v| T_n(H̃) |r_v⟩` of the rescaled Hamiltonian `h` (as
/// [`block_moments`]), on the device of `options`.
pub fn moments(
    h: &SparseMatrix,
    starts: &[Vec<c64>],
    kernel: &[f64],
    options: &Options,
) -> (Vec<c64>, Report) {
    let cpu = |starts: &[Vec<c64>]| block_moments(|x, y| h.apply_block(x, y, starts.len()), starts, kernel);
    let on_cpu = |fallback: Option<String>, deviation: Option<f64>| {
        (cpu(starts), Report { ran_on: "CPU".into(), deviation, fallback })
    };
    let precision = match options.device {
        Device::Cpu => return on_cpu(None, None),
        Device::Gpu(p) => p,
    };
    #[cfg(target_os = "macos")]
    {
        let gpu = match caroli_gpu::MetalDevice::new() {
            Ok(g) => g,
            Err(e) => return on_cpu(Some(format!("no GPU: {e}")), None),
        };
        let run = || -> Result<(Vec<c64>, f64), caroli_gpu::GpuError> {
            let op = gpu.upload(h, precision)?;
            let mu = op.kpm_moments(starts, kernel)?;
            let picked: Vec<Vec<c64>> = sample(starts.len()).into_iter().map(|i| starts[i].clone()).collect();
            let reference = cpu(&picked);
            let checked = op.kpm_moments(&picked, kernel)?;
            Ok((mu, deviation(&checked, &reference)))
        };
        match run() {
            Err(e) => on_cpu(Some(format!("GPU: {e}")), None),
            Ok((mu, dev)) if dev.is_nan() || dev > options.tolerance || mu.iter().any(|m| m.is_nan()) => {
                on_cpu(
                    Some(format!("deviation {dev:.1e} above the tolerance {:.1e}", options.tolerance)),
                    Some(dev),
                )
            }
            Ok((mu, dev)) => (
                mu,
                Report {
                    ran_on: format!("{} ({precision:?})", gpu.name()),
                    deviation: Some(dev),
                    fallback: None,
                },
            ),
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = precision;
        on_cpu(Some("no GPU backend for the KPM on this platform (Metal only)".into()), None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{jackson, random_phases};
    use caroli_disorder::rng::Xoshiro256;
    use caroli_gpu::Precision;

    /// A ring with random on-site energies, rescaled into (−1, 1).
    fn ring(n: usize) -> SparseMatrix {
        let mut rng = Xoshiro256::seed_from_u64(3);
        let mut entries = Vec::new();
        for i in 0..n {
            entries.push((i, i, c64::new(0.5 * (2.0 * rng.uniform() - 1.0), 0.0)));
            entries.push((i, (i + 1) % n, c64::new(-0.6, 0.0)));
            entries.push(((i + 1) % n, i, c64::new(-0.6, 0.0)));
        }
        SparseMatrix::from_triplets(n, entries).rescaled(0.0, 1.8)
    }

    #[test]
    fn checked_moments() {
        let h = ring(4096);
        let mut rng = Xoshiro256::seed_from_u64(9);
        let starts: Vec<Vec<c64>> = (0..60).map(|_| random_phases(&mut rng, h.n)).collect();
        let kernel = jackson(256);
        let (exact, report) = moments(&h, &starts, &kernel, &Options::cpu());
        assert_eq!(report.ran_on, "CPU");
        // double-single: on the GPU when there is one, within 1e-10 of the CPU.
        let (mu, report) = moments(&h, &starts, &kernel, &Options::accurate());
        assert!(deviation(&mu, &exact) < 1e-10, "{report:?}");
        if report.fallback.is_some() {
            return; // no GPU here
        }
        assert!(report.deviation.unwrap() < 1e-10);
        // f32 within 1e-3; an impossible tolerance sends it back to the CPU, which is exact.
        let (mu, report) = moments(&h, &starts, &kernel, &Options::exploration());
        assert!(report.fallback.is_none() && deviation(&mu, &exact) < 1e-4, "{report:?}");
        let strict = Options { device: Device::Gpu(Precision::Single), tolerance: 1e-12 };
        let (mu, report) = moments(&h, &starts, &kernel, &strict);
        assert_eq!(report.ran_on, "CPU");
        assert!(report.fallback.is_some() && report.deviation.unwrap() > 1e-12);
        assert_eq!(mu, exact);
        assert_eq!(sample(60), vec![0, 30]);
        assert_eq!(sample(1), vec![0]);
    }
}
