//! Running a walk on the GPU in reduced precision, checked against the CPU, which takes over
//! when the error would exceed the tolerance.
//!
//! - **Calibration.** The first 2 % of the steps (at least one, at most 200) run on the CPU in
//!   f64 and on the GPU from the same state; the distance between the two states, scaled
//!   linearly to the whole run, predicts the final error. Above the tolerance, f32 gives way to
//!   double-single, then to the CPU.
//! - **Checks.** At every checkpoint the CPU sums `‖ψ‖²` in f64; a drift above the tolerance
//!   (or a NaN) sends the run back to the CPU from the last good checkpoint.
//! - **Report.** Every run says where it ran, the calibrated error and the norm drift.

use crate::walk::SpinWalk;
use caroli_gpu::Precision;
use caroli_model::c64;

/// Where a walk runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    /// f64 on the CPU.
    Cpu,
    /// On the GPU in this precision (Metal), checked against the CPU.
    Gpu(Precision),
}

/// How a walk runs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    pub device: Device,
    /// Largest error `‖ψ − ψ_exact‖` (of a normalized state) accepted at the end.
    pub tolerance: f64,
    /// Steps between checkpoints (observations and checks); 0 for the start and the end only.
    pub every: usize,
}

impl Options {
    /// f64 on the CPU.
    pub fn cpu(every: usize) -> Self {
        Self { device: Device::Cpu, tolerance: 0.0, every }
    }

    /// Exploration: the GPU in f32, errors up to 10⁻³ accepted.
    pub fn exploration(every: usize) -> Self {
        Self { device: Device::Gpu(Precision::Single), tolerance: 1e-3, every }
    }

    /// The GPU in double-single, errors up to 10⁻⁹ accepted.
    pub fn accurate(every: usize) -> Self {
        Self { device: Device::Gpu(Precision::DoubleSingle), tolerance: 1e-9, every }
    }
}

/// What a run did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Report {
    /// "CPU", or the GPU and the precision of its last steps.
    pub ran_on: String,
    /// Steps of each calibration and the measured distance between GPU and CPU after them.
    pub calibrations: Vec<(Precision, usize, f64)>,
    /// Error predicted at the last step by the accepted calibration.
    pub predicted_error: Option<f64>,
    /// Largest relative drift of `‖ψ‖²` seen at the GPU checkpoints.
    pub norm_drift: f64,
    /// Step from which the CPU took over, and why.
    pub fallback: Option<(usize, String)>,
    pub gpu_steps: usize,
    pub cpu_steps: usize,
}

/// The tables of a walk for the GPU kernels.
#[cfg(target_os = "macos")]
pub(crate) struct Tables {
    bits: usize,
    node_rows: Vec<usize>,
    coins: Vec<Vec<c64>>,
    halves: Vec<(usize, usize)>,
    exchange: [c64; 3],
}

#[cfg(target_os = "macos")]
impl Tables {
    pub(crate) fn of(walk: &SpinWalk) -> Self {
        let graph = walk.graph();
        let (p, alpha, beta) = walk.exchange();
        Self {
            bits: graph.edges(),
            node_rows: (0..graph.nodes())
                .map(|x| graph.half_edges_of(x).start)
                .chain([graph.half_edges()])
                .collect(),
            coins: walk
                .coins()
                .iter()
                .map(|c| (0..c.nrows()).flat_map(|i| (0..c.ncols()).map(move |j| c[(i, j)])).collect())
                .collect(),
            halves: (0..graph.edges()).map(|e| graph.halves(e)).collect(),
            exchange: [p, alpha, beta],
        }
    }

    pub(crate) fn view(&self) -> caroli_gpu::WalkTables<'_> {
        caroli_gpu::WalkTables {
            bits: self.bits,
            node_rows: &self.node_rows,
            coins: &self.coins,
            halves: &self.halves,
            exchange: self.exchange,
        }
    }
}

/// Steps per calibration: 2 % of the run, at least one, at most 200.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn calibration_steps(steps: usize) -> usize {
    (steps / 50).clamp(1, 200)
}

/// Runs `steps` steps from `psi` (left at the last step) and calls `observe(t, ψ)` at `t = 0`,
/// every `options.every` steps and at the end.
pub fn evolve(
    walk: &SpinWalk,
    psi: &mut [c64],
    steps: usize,
    options: &Options,
    mut observe: impl FnMut(usize, &[c64]),
) -> Report {
    let every = if options.every == 0 { steps.max(1) } else { options.every };
    let mut report = Report::default();
    observe(0, psi);
    let mut t = 0;
    if let Device::Gpu(precision) = options.device {
        t = run_gpu(walk, psi, steps, every, precision, options.tolerance, &mut observe, &mut report);
    }
    if t < steps {
        report.ran_on =
            if report.ran_on.is_empty() { "CPU".into() } else { format!("{}, then the CPU", report.ran_on) };
        cpu_segment(walk, psi, t, steps, steps, every, &mut observe);
        report.cpu_steps += steps - t;
    } else if report.ran_on.is_empty() {
        report.ran_on = "CPU".into();
    }
    report
}

/// CPU steps from `t` to `end`, observing at the checkpoints of a run that ends at `last`.
fn cpu_segment(
    walk: &SpinWalk,
    psi: &mut [c64],
    mut t: usize,
    end: usize,
    last: usize,
    every: usize,
    observe: &mut impl FnMut(usize, &[c64]),
) {
    while t < end {
        let next = next_checkpoint(t, last, every).min(end);
        walk.steps(psi, next - t);
        t = next;
        if t % every == 0 || t == last {
            observe(t, psi);
        }
    }
}

/// The checkpoint after `t`: the next multiple of `every`, or the end.
fn next_checkpoint(t: usize, end: usize, every: usize) -> usize {
    ((t / every + 1) * every).min(end)
}

/// The GPU part of [`evolve`]: returns the step reached; the CPU does the rest.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn run_gpu(
    walk: &SpinWalk,
    psi: &mut [c64],
    steps: usize,
    every: usize,
    precision: Precision,
    tolerance: f64,
    observe: &mut impl FnMut(usize, &[c64]),
    report: &mut Report,
) -> usize {
    use caroli_gpu::MetalDevice;
    let gpu = match MetalDevice::new() {
        Ok(g) => g,
        Err(e) => {
            report.fallback = Some((0, format!("no GPU: {e}")));
            return 0;
        }
    };
    let owned = Tables::of(walk);
    let tables = owned.view();
    let norm0: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
    let candidates: &[Precision] = match precision {
        Precision::Single => &[Precision::Single, Precision::DoubleSingle],
        Precision::DoubleSingle => &[Precision::DoubleSingle],
    };
    let mut t = 0;
    for &candidate in candidates {
        if t >= steps {
            break;
        }
        let mut device = match gpu.walk(&tables, psi, candidate) {
            Ok(w) => w,
            Err(e) => {
                report.fallback = Some((t, format!("GPU: {e}")));
                return t;
            }
        };
        // Calibration: the same steps on both, from the same state.
        let k = calibration_steps(steps).min(steps - t);
        let start = t;
        cpu_segment(walk, psi, start, start + k, steps, every, observe);
        report.cpu_steps += k;
        t = start + k;
        if let Err(e) = device.steps(k) {
            report.fallback = Some((t, format!("GPU: {e}")));
            return t;
        }
        let error = device.distance(psi) / norm0.sqrt();
        report.calibrations.push((candidate, k, error));
        let predicted = error * steps as f64 / k as f64;
        if predicted.is_nan() || predicted > tolerance {
            report.fallback = Some((
                t,
                format!("{candidate:?}: predicted error {predicted:.1e} above the tolerance {tolerance:.1e}"),
            ));
            continue;
        }
        report.fallback = None;
        report.predicted_error = Some(predicted);
        report.ran_on = format!("{} ({candidate:?})", gpu.name());
        // Accepted: continue on the GPU from the exact state.
        device.set_state(psi);
        while t < steps {
            let next = next_checkpoint(t, steps, every);
            if let Err(e) = device.steps(next - t) {
                report.fallback = Some((t, format!("GPU: {e}")));
                return t;
            }
            report.gpu_steps += next - t;
            let drift = (device.norm_sqr() - norm0).abs() / norm0;
            report.norm_drift = report.norm_drift.max(if drift.is_nan() { f64::INFINITY } else { drift });
            if drift.is_nan() || drift > tolerance {
                report.fallback =
                    Some((t, format!("norm drift {drift:.1e} above the tolerance {tolerance:.1e}")));
                return t;
            }
            device.read_into(psi);
            t = next;
            observe(t, psi);
        }
        return t;
    }
    t
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
fn run_gpu(
    _walk: &SpinWalk,
    _psi: &mut [c64],
    _steps: usize,
    _every: usize,
    _precision: Precision,
    _tolerance: f64,
    _observe: &mut impl FnMut(usize, &[c64]),
    report: &mut Report,
) -> usize {
    report.fallback = Some((0, "no GPU walk backend on this platform (Metal only)".into()));
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::walk::PLUS;

    fn run(options: &Options, steps: usize) -> (Vec<c64>, Vec<(usize, f64)>, Report) {
        let walk = SpinWalk::ring(6, 0.5, 0.7);
        let mut psi = walk.product_state(&walk.localized(0, 0), &[PLUS; 6]);
        let mut seen = Vec::new();
        let report = evolve(&walk, &mut psi, steps, options, |t, psi| {
            seen.push((t, walk.node_probabilities(psi)[0]));
        });
        (psi, seen, report)
    }

    #[test]
    fn cpu_checkpoints() {
        let (_, seen, report) = run(&Options::cpu(40), 100);
        assert_eq!(seen.iter().map(|s| s.0).collect::<Vec<_>>(), vec![0, 40, 80, 100]);
        assert_eq!((report.ran_on.as_str(), report.cpu_steps, report.gpu_steps), ("CPU", 100, 0));
    }

    /// On a Mac with a GPU: double-single agrees with the CPU to ~1e-12 and stays on the GPU;
    /// an impossible tolerance sends f32 and double-single back to the CPU, which then gives the
    /// exact result.
    #[test]
    fn gpu_checked_against_cpu() {
        let (exact, cpu_seen, _) = run(&Options::cpu(40), 300);
        let (psi, seen, report) = run(&Options::accurate(40), 300);
        let distance =
            |a: &[c64], b: &[c64]| a.iter().zip(b).map(|(x, y)| (x - y).norm_sqr()).sum::<f64>().sqrt();
        assert_eq!(seen.len(), cpu_seen.len());
        if report.gpu_steps == 0 {
            // No GPU here: the CPU ran everything.
            assert!(report.fallback.is_some() && distance(&psi, &exact) < 1e-14);
            return;
        }
        assert!(report.fallback.is_none(), "{report:?}");
        assert!(distance(&psi, &exact) < 1e-10, "double-single: {}", distance(&psi, &exact));
        assert!(report.norm_drift < 1e-10);
        let strict = Options { device: Device::Gpu(Precision::Single), tolerance: 1e-15, every: 40 };
        let (psi, seen, report) = run(&strict, 300);
        assert_eq!(report.ran_on, "CPU");
        assert_eq!(report.calibrations.len(), 2);
        assert!(report.calibrations[0].2 > 1e-9 && report.calibrations[1].2 < 1e-10);
        assert!(distance(&psi, &exact) < 1e-14);
        assert_eq!(
            seen.iter().map(|s| s.0).collect::<Vec<_>>(),
            cpu_seen.iter().map(|s| s.0).collect::<Vec<_>>()
        );
        // f32 with an exploration tolerance stays on the GPU.
        let (psi, _, report) = run(&Options::exploration(40), 300);
        assert!(report.fallback.is_none() && report.ran_on.contains("Single"), "{report:?}");
        assert!(distance(&psi, &exact) < 1e-4, "f32: {}", distance(&psi, &exact));
    }
}
