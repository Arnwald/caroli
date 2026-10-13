//! Reduced precision for transport (experimental), chosen by the user and checked against double
//! precision.
//!
//! The nested-dissection solvers can run their fronts in complex f32 ([`Precision::Single`]).
//! The root of the elimination (the lead self-energies, the delayed variables and the final
//! solve) and the trace stay in f64. [`checked`] runs a batch of points in the chosen precision,
//! recomputes a sample in double precision and falls back to double precision for the whole
//! batch when the sample differs by more than the tolerance: every run reports the error it
//! measured.
//!
//! Measured on Apple silicon (`ARCHITECTURE.md`, item 12), single precision is a preview tool:
//! - 1.1–1.6× faster on one thread on the benchmark problems (less where the lead modes, always
//!   in f64, take most of the time: BHZ, graphene), 1.4–1.9× on all threads;
//! - the median error stays near 10⁻⁵, but the largest grows with the width: on the spin blocks
//!   of the g2g1 ribbons, 3·10⁻² at W = 64, 6·10⁻² at W = 128, 0.4 at W = 256 (below 10⁻² on
//!   the conductance plateaus). Nested dissection eliminates closed subdomains at a real energy,
//!   and close to one of their eigenvalues a block keeps too few digits in f32.

use crate::GreenError;
use rayon::prelude::*;

/// Floating-point precision of the eliminations of a nested-dissection solve.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Precision {
    /// Complex f64 throughout (the default).
    #[default]
    Double,
    /// Experimental: the fronts below the root in complex f32, the root and the trace in f64.
    /// 1.1–1.9× faster, with errors up to 10⁻² and more in wide samples (see the module
    /// documentation): for previews, checked with [`checked`].
    Single,
}

/// How a batch of points runs in reduced precision (see [`checked`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrecisionOptions {
    pub precision: Precision,
    /// Largest difference accepted between a point in `precision` and its double-precision value.
    pub tolerance: f64,
    /// One point in `every` (at least one) is recomputed in double precision.
    pub every: usize,
}

impl PrecisionOptions {
    /// Double precision: nothing to check.
    pub fn double() -> Self {
        Self { precision: Precision::Double, tolerance: 0.0, every: 0 }
    }

    /// Single precision for previews: one point in fifty recomputed in double precision,
    /// differences up to 10⁻² accepted.
    pub fn single() -> Self {
        Self { precision: Precision::Single, tolerance: 1e-2, every: 50 }
    }
}

impl Default for PrecisionOptions {
    fn default() -> Self {
        Self::double()
    }
}

/// What a checked batch did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PrecisionReport {
    /// The precision of the results returned: `Double` after a fallback.
    pub precision: Precision,
    /// Points recomputed in double precision to check the batch, and the median and largest
    /// difference found among them.
    pub checked: usize,
    pub median: f64,
    pub max: f64,
    /// Points that failed in reduced precision (a singular front) and were computed in double
    /// precision instead.
    pub failed: usize,
    /// Why the whole batch was recomputed in double precision.
    pub fallback: Option<String>,
}

/// Runs `run(point, precision)` on every point, in parallel, in `options.precision`. In reduced
/// precision, one point in `options.every` is recomputed in double precision and compared with
/// `distance`; if one of them differs by more than `options.tolerance`, the whole batch is
/// recomputed in double precision. Points that fail in reduced precision are computed in double
/// precision. The checked points return their double-precision values.
///
/// ```ignore
/// let (t, report) = checked(&energies, &PrecisionOptions::single(), |&e, p| {
///     sys.transmission_with_precision(&sys.self_energies(e)?, p)
/// }, |a, b| (a - b).abs());
/// ```
pub fn checked<P: Sync, O: Send>(
    points: &[P],
    options: &PrecisionOptions,
    run: impl Fn(&P, Precision) -> Result<O, GreenError> + Sync,
    distance: impl Fn(&O, &O) -> f64,
) -> (Vec<Result<O, GreenError>>, PrecisionReport) {
    let mut report = PrecisionReport { precision: options.precision, ..PrecisionReport::default() };
    let mut out: Vec<Result<O, GreenError>> = points.par_iter().map(|p| run(p, options.precision)).collect();
    if options.precision == Precision::Double || points.is_empty() {
        return (out, report);
    }
    // Points that failed in reduced precision, and the sample, in double precision.
    let every = options.every.max(1);
    let redo: Vec<usize> = (0..points.len()).filter(|&i| out[i].is_err() || i % every == 0).collect();
    let double: Vec<(usize, Result<O, GreenError>)> =
        redo.par_iter().map(|&i| (i, run(&points[i], Precision::Double))).collect();
    let mut differences = Vec::new();
    for (i, d) in double {
        match (&out[i], &d) {
            (Ok(single), Ok(exact)) => differences.push(distance(single, exact)),
            (Err(_), _) => report.failed += 1,
            (Ok(_), Err(_)) => differences.push(f64::NAN),
        }
        out[i] = d;
    }
    report.checked = differences.len();
    differences.sort_by(f64::total_cmp);
    if let Some(&last) = differences.last() {
        report.median = differences[differences.len() / 2];
        report.max = if differences.iter().any(|d| d.is_nan()) { f64::NAN } else { last };
    }
    // A NaN (a point that failed in double precision only) fails the check too.
    if report.max.is_nan() || report.max > options.tolerance {
        report.fallback = Some(format!(
            "largest difference {:.1e} above the tolerance {:.1e}: all {} points recomputed in double precision",
            report.max,
            options.tolerance,
            points.len()
        ));
        report.precision = Precision::Double;
        out = points.par_iter().map(|p| run(p, Precision::Double)).collect();
    }
    (out, report)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A point "computed" in single precision is off by 1e-4 at index 7.
    fn run(i: &usize, p: Precision) -> Result<f64, GreenError> {
        let exact = *i as f64;
        match (p, *i) {
            (Precision::Single, 7) => Ok(exact + 1e-4),
            (Precision::Single, 3) => Err(GreenError::Singular(0.0)),
            _ => Ok(exact),
        }
    }

    #[test]
    fn sample_failures_and_fallback() {
        let points: Vec<usize> = (0..10).collect();
        let d = |a: &f64, b: &f64| (a - b).abs();
        // Index 7 is not sampled (every 5): kept; the failed index 3 is redone in double.
        let options = PrecisionOptions { precision: Precision::Single, tolerance: 1e-6, every: 5 };
        let (out, report) = checked(&points, &options, run, d);
        assert_eq!((report.checked, report.failed, report.fallback.is_none()), (2, 1, true));
        assert_eq!(out[3].as_ref().ok(), Some(&3.0));
        assert_eq!(out[7].as_ref().ok(), Some(&(7.0 + 1e-4)));
        // Index 7 sampled (every 7): above the tolerance, everything redone in double.
        let options = PrecisionOptions { every: 7, ..options };
        let (out, report) = checked(&points, &options, run, d);
        assert!(report.fallback.is_some() && report.precision == Precision::Double);
        assert!(out.iter().enumerate().all(|(i, t)| t.as_ref().ok() == Some(&(i as f64))));
        // Double precision: nothing checked.
        let (_, report) = checked(&points, &PrecisionOptions::double(), run, d);
        assert_eq!(report, PrecisionReport::default());
    }
}
