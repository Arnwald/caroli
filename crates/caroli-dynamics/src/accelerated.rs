//! Chebyshev propagation on the GPU, checked against the CPU.
//!
//! - **Calibration.** The first 2 % of the steps (at least one, at most 50) run on the CPU in
//!   f64 and on the GPU from the same state. Their distance, relative to `‖ψ‖` and scaled
//!   linearly to the whole run, predicts the final error; above the tolerance the CPU does the
//!   rest. Otherwise the GPU continues from the CPU state.
//! - **Final check.** The norm of every wave function must stay within the tolerance; if it does
//!   not (or the GPU fails), the CPU recomputes the whole run.

use crate::Chebyshev;
use caroli_gpu::{Device, Options, Report};
use caroli_model::{SparseMatrix, c64};

fn norms(psi: &[c64], vectors: usize) -> Vec<f64> {
    let mut n = vec![0.0; vectors];
    for (i, z) in psi.iter().enumerate() {
        n[i % vectors] += z.norm_sqr();
    }
    n
}

/// `steps` steps `ψ ← exp(−iH dt) ψ` of the propagator `cheb` with the rescaled Hamiltonian `h`,
/// for a block of `vectors` wave functions stored interleaved (as `SparseMatrix::apply_block`),
/// on the device of `options`.
pub fn propagate(
    h: &SparseMatrix,
    cheb: &Chebyshev,
    psi: &mut [c64],
    vectors: usize,
    steps: usize,
    options: &Options,
) -> Report {
    assert!(
        vectors > 0 && psi.len() == h.n * vectors,
        "ψ must hold {vectors} wave functions of {} entries",
        h.n
    );
    let cpu = |psi: &mut [c64], steps: usize| {
        for _ in 0..steps {
            cheb.step(|x, y| h.apply_block(x, y, vectors), psi);
        }
    };
    let precision = match options.device {
        Device::Cpu => {
            cpu(psi, steps);
            return Report { ran_on: "CPU".into(), ..Report::default() };
        }
        Device::Gpu(p) => p,
    };
    #[cfg(target_os = "macos")]
    {
        let start = psi.to_vec();
        let before = norms(psi, vectors);
        let mut report = Report::default();
        let gpu = match caroli_gpu::MetalDevice::new() {
            Ok(g) => g,
            Err(e) => {
                cpu(psi, steps);
                return Report {
                    ran_on: "CPU".into(),
                    deviation: None,
                    fallback: Some(format!("no GPU: {e}")),
                };
            }
        };
        let k = (steps / 50).clamp(1, 50).min(steps);
        let run = |psi: &mut [c64], report: &mut Report| -> Result<bool, caroli_gpu::GpuError> {
            let op = gpu.upload(h, precision)?;
            let mut on_gpu = psi.to_vec();
            op.propagate(&cheb.coefficients, &mut on_gpu, vectors, k)?;
            cpu(psi, k);
            let scale: f64 = psi.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            let distance = on_gpu.iter().zip(psi.iter()).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>().sqrt();
            let predicted = distance / scale * steps as f64 / k as f64;
            report.deviation = Some(predicted);
            if predicted.is_nan() || predicted > options.tolerance {
                report.fallback = Some(format!(
                    "predicted error {predicted:.1e} above the tolerance {:.1e}",
                    options.tolerance
                ));
                cpu(psi, steps - k);
                return Ok(false);
            }
            op.propagate(&cheb.coefficients, psi, vectors, steps - k)?;
            Ok(true)
        };
        match run(psi, &mut report) {
            Ok(false) => report.ran_on = "CPU".into(),
            Ok(true) => {
                let drift = norms(psi, vectors)
                    .iter()
                    .zip(&before)
                    .map(|(a, b)| (a - b).abs() / b)
                    .fold(0.0, f64::max);
                if drift.is_nan() || drift > options.tolerance {
                    psi.copy_from_slice(&start);
                    cpu(psi, steps);
                    report.ran_on = "CPU".into();
                    report.fallback =
                        Some(format!("norm drift {drift:.1e} above the tolerance {:.1e}", options.tolerance));
                } else {
                    report.ran_on = format!("{} ({precision:?})", gpu.name());
                }
            }
            Err(e) => {
                psi.copy_from_slice(&start);
                cpu(psi, steps);
                report =
                    Report { ran_on: "CPU".into(), deviation: None, fallback: Some(format!("GPU: {e}")) };
            }
        }
        report
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = precision;
        cpu(psi, steps);
        Report {
            ran_on: "CPU".into(),
            deviation: None,
            fallback: Some("no GPU backend for the propagation on this platform (Metal only)".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use caroli_gpu::Precision;

    #[test]
    fn checked_propagation() {
        // A ring with a linear potential, rescaled into (−1, 1).
        let n = 2048;
        let mut entries = Vec::new();
        for i in 0..n {
            entries.push((i, i, c64::new(0.3 * (i as f64 / n as f64 - 0.5), 0.0)));
            entries.push((i, (i + 1) % n, c64::new(-0.5, 0.0)));
            entries.push(((i + 1) % n, i, c64::new(-0.5, 0.0)));
        }
        let h = SparseMatrix::from_triplets(n, entries);
        let (centre, half) = (0.0, 1.2);
        let scaled = h.rescaled(centre, half);
        let cheb = Chebyshev::new(0.4, centre, half, 1e-15);
        let norm = (n as f64).sqrt();
        let psi0: Vec<c64> = (0..2 * n).map(|i| c64::from_polar(1.0 / norm, 0.01 * (i * i) as f64)).collect();
        let steps = 100;
        let run = |options: &Options| {
            let mut psi = psi0.clone();
            let report = propagate(&scaled, &cheb, &mut psi, 2, steps, options);
            (psi, report)
        };
        let (exact, report) = run(&Options::cpu());
        assert_eq!(report.ran_on, "CPU");
        let distance = |a: &[c64]| a.iter().zip(&exact).map(|(x, y)| (x - y).norm_sqr()).sum::<f64>().sqrt();
        let (psi, report) = run(&Options::accurate());
        assert!(distance(&psi) < 1e-10, "{report:?}");
        if report.fallback.is_some() {
            return; // no GPU here
        }
        let (psi, report) = run(&Options::exploration());
        assert!(report.fallback.is_none() && distance(&psi) < 1e-3, "{report:?}");
        let strict = Options { device: Device::Gpu(Precision::Single), tolerance: 1e-12 };
        let (psi, report) = run(&strict);
        assert_eq!(report.ran_on, "CPU");
        assert!(report.fallback.is_some() && distance(&psi) < 1e-13, "{report:?}");
    }
}
