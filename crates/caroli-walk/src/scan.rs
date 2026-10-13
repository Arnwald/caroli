//! Many walks of one graph side by side, as in a scan of parameters: one walk per CPU thread
//! and, optionally, a stream of walks on the GPU fed by one more thread, all taking the next walk
//! from one queue. Each walk returns the probability of every half-edge summed over its steps
//! (time averages, Kolmogorov–Smirnov distances).
//!
//! Walks run in reduced precision (f32 on the CPU or on the GPU) are checked: one in fifty, at
//! least one, is recomputed in f64 on the CPU, and if the sums differ by more than the tolerance
//! (relative to their total), all the walks of that kind are recomputed in f64.

use crate::walk::{SpinWalk, to_single};
use caroli_model::c64;
use rayon::prelude::*;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// How a scan runs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScanOptions {
    /// f32 states on the CPU: half the memory traffic.
    pub single: bool,
    /// Also run walks on the GPU (f32), fed by one of the threads.
    pub gpu: bool,
    /// Largest deviation accepted between a reduced-precision walk and its f64 recomputation.
    pub tolerance: f64,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self { single: false, gpu: false, tolerance: 1e-3 }
    }
}

/// What a scan did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScanReport {
    /// Walks run on the CPU and on the GPU.
    pub cpu_walks: usize,
    pub gpu_walks: usize,
    /// The GPU, or why it was not used.
    pub gpu: String,
    /// Reduced-precision walks recomputed in f64, and the largest relative deviation found.
    pub checked: usize,
    pub deviation: f64,
    /// Why the f64 CPU recomputed the reduced-precision walks.
    pub fallback: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Double,
    Single,
    Gpu,
}

/// The result of each walk, filled by the threads that run them.
type Slots = [Mutex<Option<(Vec<f64>, Kind)>>];

/// The sums over `steps` steps of the half-edge probabilities, on the calling thread.
fn on_cpu(walk: &SpinWalk, psi: Vec<c64>, steps: usize, single: bool) -> Vec<f64> {
    let mut acc = vec![0.0; walk.graph().half_edges()];
    if single {
        let mut psi = to_single(&psi);
        for _ in 0..steps {
            walk.step_serial_accumulate(&mut psi, &mut acc);
        }
    } else {
        let mut psi = psi;
        for _ in 0..steps {
            walk.step_serial_accumulate(&mut psi, &mut acc);
        }
    }
    acc
}

/// `max_h |a_h − b_h| / Σ b`.
fn deviation(a: &[f64], b: &[f64]) -> f64 {
    let total: f64 = b.iter().sum();
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max) / total
}

/// For each walk, the probability of every half-edge summed over `steps` steps from
/// `initial(i)`. The walks must share their graph.
pub fn scan(
    walks: &[SpinWalk],
    initial: impl Fn(usize) -> Vec<c64> + Sync,
    steps: usize,
    options: &ScanOptions,
) -> (Vec<Vec<f64>>, ScanReport) {
    let mut report = ScanReport::default();
    if walks.is_empty() {
        return (Vec::new(), report);
    }
    assert!(walks.iter().all(|w| w.graph() == walks[0].graph()), "the walks of a scan share their graph");
    let results: Box<Slots> = walks.iter().map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    let threads = rayon::current_num_threads().max(1);
    let gpu_name = Mutex::new(String::from("not used"));
    // The CPU threads wait for the GPU to be ready (or to have failed), so that it gets its share.
    let ready = AtomicBool::new(!options.gpu);
    let cpu_kind = if options.single { Kind::Single } else { Kind::Double };
    std::thread::scope(|s| {
        if options.gpu {
            s.spawn(|| {
                *gpu_name.lock().unwrap() = gpu_worker(walks, &initial, steps, &next, &results, &ready);
                ready.store(true, Ordering::Release);
            });
        }
        let workers = if options.gpu { threads.saturating_sub(1).max(1) } else { threads };
        for _ in 0..workers {
            s.spawn(|| {
                while !ready.load(Ordering::Acquire) {
                    std::thread::yield_now();
                }
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= walks.len() {
                        break;
                    }
                    let acc = on_cpu(&walks[i], initial(i), steps, options.single);
                    *results[i].lock().unwrap() = Some((acc, cpu_kind));
                }
            });
        }
    });
    report.gpu = gpu_name.into_inner().unwrap();
    let mut out: Vec<(Vec<f64>, Kind)> =
        results.into_iter().map(|m| m.into_inner().unwrap().expect("every walk ran")).collect();
    report.gpu_walks = out.iter().filter(|r| r.1 == Kind::Gpu).count();
    report.cpu_walks = out.len() - report.gpu_walks;
    // Check the reduced-precision walks against f64, kind by kind.
    for kind in [Kind::Single, Kind::Gpu] {
        let indices: Vec<usize> = (0..out.len()).filter(|&i| out[i].1 == kind).collect();
        if indices.is_empty() {
            continue;
        }
        let picked: Vec<usize> = (0..indices.len().div_ceil(50)).map(|k| indices[k * 50]).collect();
        let worst = picked
            .par_iter()
            .map(|&i| deviation(&out[i].0, &on_cpu(&walks[i], initial(i), steps, false)))
            .reduce(|| 0.0, f64::max);
        report.checked += picked.len();
        report.deviation = report.deviation.max(worst);
        if worst.is_nan() || worst > options.tolerance {
            let name = if kind == Kind::Gpu { "GPU" } else { "CPU f32" };
            report.fallback = Some(format!(
                "{name}: deviation {worst:.1e} above the tolerance {:.1e}; {} walks recomputed in f64",
                options.tolerance,
                indices.len()
            ));
            let redone: Vec<(usize, Vec<f64>)> =
                indices.par_iter().map(|&i| (i, on_cpu(&walks[i], initial(i), steps, false))).collect();
            for (i, acc) in redone {
                out[i] = (acc, Kind::Double);
            }
        }
    }
    (out.into_iter().map(|r| r.0).collect(), report)
}

/// The GPU's share of a scan: walks taken from the queue until it is empty. Returns the GPU's
/// name, or why it could not run; a walk it fails is run on the CPU by this thread.
#[cfg(target_os = "macos")]
fn gpu_worker(
    walks: &[SpinWalk],
    initial: &(impl Fn(usize) -> Vec<c64> + Sync),
    steps: usize,
    next: &AtomicUsize,
    results: &Slots,
    ready: &AtomicBool,
) -> String {
    use caroli_gpu::{MetalDevice, Precision};
    let device = MetalDevice::new();
    ready.store(true, Ordering::Release);
    let device = match device {
        Ok(d) => d,
        Err(e) => return format!("no GPU: {e}"),
    };
    // Walks per batch: about 8 MB of f32 states, so that a batch stays in the GPU's cache (larger
    // batches spill to memory and run slower per walk on an M5).
    let per_walk = walks[0].dimension() * 8;
    let size = ((8usize << 20) / per_walk).clamp(1, 64);
    let rows = walks[0].graph().half_edges();
    let mut failed = None;
    loop {
        let start = next.fetch_add(size, Ordering::Relaxed);
        if start >= walks.len() {
            break;
        }
        let picked: Vec<usize> = (start..(start + size).min(walks.len())).collect();
        let tables: Vec<crate::gpu::Tables> =
            picked.iter().map(|&i| crate::gpu::Tables::of(&walks[i])).collect();
        let views: Vec<caroli_gpu::WalkTables> = tables.iter().map(|t| t.view()).collect();
        let states: Vec<c64> = picked.iter().flat_map(|&i| initial(i)).collect();
        let run = device.walks(&views, &states, Precision::Single).and_then(|mut w| {
            w.steps_accumulate(steps)?;
            Ok(w.accumulated())
        });
        match run {
            Ok(acc) => {
                for (k, &i) in picked.iter().enumerate() {
                    *results[i].lock().unwrap() = Some((acc[k * rows..(k + 1) * rows].to_vec(), Kind::Gpu));
                }
            }
            Err(e) => {
                for &i in &picked {
                    *results[i].lock().unwrap() =
                        Some((on_cpu(&walks[i], initial(i), steps, false), Kind::Double));
                }
                failed = Some(format!("GPU failed ({e}); the CPU took over"));
                break;
            }
        }
    }
    failed.unwrap_or_else(|| device.name())
}

#[cfg(not(target_os = "macos"))]
fn gpu_worker(
    _walks: &[SpinWalk],
    _initial: &(impl Fn(usize) -> Vec<c64> + Sync),
    _steps: usize,
    _next: &AtomicUsize,
    _results: &Slots,
    _ready: &AtomicBool,
) -> String {
    "no GPU walk backend on this platform (Metal only)".into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::walk::PLUS;

    #[test]
    fn scans_agree() {
        let n = 9;
        let walks: Vec<SpinWalk> =
            (0..12).map(|k| SpinWalk::ring(n, 0.1 + 0.12 * k as f64, 0.3 + 0.2 * k as f64)).collect();
        let initial = |i: usize| walks[i].product_state(&walks[i].localized(4, 0), &[PLUS; 9]);
        let steps = 200;
        let (exact, report) = scan(&walks, initial, steps, &ScanOptions::default());
        assert_eq!((report.cpu_walks, report.gpu_walks, report.checked), (12, 0, 0));
        let total: f64 = exact[0].iter().sum();
        assert!((total - steps as f64).abs() < 1e-9, "the probabilities add up to one at each step");
        for options in [
            ScanOptions { single: true, ..ScanOptions::default() },
            ScanOptions { gpu: true, ..ScanOptions::default() },
            ScanOptions { single: true, gpu: true, ..ScanOptions::default() },
        ] {
            let (sums, report) = scan(&walks, initial, steps, &options);
            assert!(report.fallback.is_none(), "{report:?}");
            // Without a GPU (other platforms, virtual machines) the CPU runs every walk, and
            // only the f32 ones need checking.
            let reduced = options.single || report.gpu_walks > 0;
            assert_eq!(report.checked >= 1, reduced, "{report:?}");
            assert!(report.deviation < 1e-4, "{report:?}");
            for (a, b) in sums.iter().zip(&exact) {
                assert!(deviation(a, b) < 1e-4, "{options:?}");
            }
        }
        // An impossible tolerance: the reduced-precision walks are redone in f64.
        let strict = ScanOptions { single: true, gpu: false, tolerance: 1e-15 };
        let (sums, report) = scan(&walks, initial, steps, &strict);
        assert!(report.fallback.is_some());
        assert_eq!(sums, exact);
    }
}
