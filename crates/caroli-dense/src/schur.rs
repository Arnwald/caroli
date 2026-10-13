//! In-house complex Schur decomposition, and the reordering of a Schur form.
//!
//! - Hessenberg reduction: faer.
//! - QR iterations: implicit single-shift, with Wilkinson shifts and deflation (LAPACK zlahqr).
//! - Reordering: adjacent swaps by Givens rotations (LAPACK ztrexc). [`partition`] makes them in
//!   windows and applies their accumulated rotations to the rest of the matrix as products
//!   (Kressner's block algorithm, ACM TOMS 32, 2006): most of the work becomes matrix products.

use crate::{Op, matmul};
use faer::{Accum, Mat, c64};

const ZERO: c64 = c64::new(0.0, 0.0);
const ONE: c64 = c64::new(1.0, 0.0);

/// Window of the blocked reordering, and the largest number of eigenvalues moved through it at
/// once.
const WINDOW: usize = 64;
const BATCH: usize = 32;

/// Givens rotation `(c, s)` with `[c s; −s̄ c] [f; g] = [r; 0]` (LAPACK zlartg).
fn givens(f: c64, g: c64) -> (f64, c64) {
    if g == ZERO {
        return (1.0, ZERO);
    }
    if f == ZERO {
        return (0.0, g.conj() / g.norm());
    }
    let d = f.norm().hypot(g.norm());
    (f.norm() / d, f / f.norm() * g.conj() / d)
}

/// `H ← G H G†` and `Z ← Z G†` for the rotation `G = [c s; −s̄ c]` on indices `k, k+1`, rows
/// restricted to `cols` and columns to `rows`.
fn rotate(
    h: &mut Mat<c64>,
    z: &mut Mat<c64>,
    k: usize,
    c: f64,
    s: c64,
    cols: std::ops::Range<usize>,
    rows: std::ops::Range<usize>,
) {
    for j in cols {
        let (x, y) = (h[(k, j)], h[(k + 1, j)]);
        h[(k, j)] = x * c + s * y;
        h[(k + 1, j)] = y * c - s.conj() * x;
    }
    for i in rows {
        let (x, y) = (h[(i, k)], h[(i, k + 1)]);
        h[(i, k)] = x * c + s.conj() * y;
        h[(i, k + 1)] = y * c - s * x;
    }
    for i in 0..z.nrows() {
        let (x, y) = (z[(i, k)], z[(i, k + 1)]);
        z[(i, k)] = x * c + s.conj() * y;
        z[(i, k + 1)] = y * c - s * x;
    }
}

/// Complex Schur decomposition `a = q t q†` (`t` upper triangular). Returns `(q, t)`.
#[cfg_attr(all(target_os = "macos", feature = "accelerate"), allow(dead_code))]
pub(crate) fn schur(a: faer::MatRef<'_, c64>) -> Result<(Mat<c64>, Mat<c64>), crate::DenseError> {
    use crate::DenseError;
    use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
    use faer::linalg::evd::hessenberg;
    use faer::linalg::householder;
    use faer::{Conj, Par, Spec};
    let n = a.nrows();
    let mut h = a.to_owned();
    let mut q = Mat::<c64>::identity(n, n);
    if n <= 1 {
        return Ok((q, h));
    }
    if !(h.norm_max().is_finite()) {
        return Err(DenseError::Schur);
    }
    // Hessenberg form h = q h' q†.
    let par = Par::Seq;
    let bs = faer::linalg::qr::no_pivoting::factor::recommended_block_size::<c64>(n - 1, n - 1);
    let req = StackReq::any_of(&[
        hessenberg::hessenberg_in_place_scratch::<c64>(n, bs, par, Spec::default()),
        householder::apply_block_householder_sequence_on_the_right_in_place_scratch::<c64>(n - 1, bs, n - 1),
    ]);
    let mut mem = MemBuffer::new(req);
    let stack = MemStack::new(&mut mem);
    let mut hh = Mat::<c64>::zeros(bs, n - 1);
    hessenberg::hessenberg_in_place(h.as_mut(), hh.as_mut(), par, stack, Spec::default());
    householder::apply_block_householder_sequence_on_the_right_in_place_with_conj(
        h.as_ref().submatrix(1, 0, n - 1, n - 1),
        hh.as_ref(),
        Conj::No,
        q.as_mut().submatrix_mut(1, 1, n - 1, n - 1),
        par,
        stack,
    );
    for j in 0..n {
        for i in j + 2..n {
            h[(i, j)] = ZERO;
        }
    }

    // Implicit single-shift QR on the active block [lo, hi]. A subdiagonal entry is negligible
    // next to its two diagonal neighbours. When an eigenvalue takes more than ten iterations, it
    // is also negligible next to the norm of the matrix (a normwise backward error of one ulp):
    // this deflates clusters of tiny eigenvalues, such as the zero eigenvalues of a low-rank
    // matrix, whose diagonal entries shrink with the subdiagonal ones.
    let ulp = f64::EPSILON;
    let frobenius = h.norm_l2();
    let mut hi = n - 1;
    let max_iter = 30 * n.max(10);
    let mut total = 0;
    loop {
        // Deflate negligible subdiagonal entries at the bottom.
        let mut its = 0;
        loop {
            let mut lo = hi;
            while lo > 0 {
                let sub = h[(lo, lo - 1)].norm();
                let mut scale = h[(lo - 1, lo - 1)].norm() + h[(lo, lo)].norm();
                if scale == 0.0 {
                    scale = h.norm_max();
                }
                if sub <= ulp * scale || (its >= 10 && sub <= ulp * frobenius) {
                    h[(lo, lo - 1)] = ZERO;
                    break;
                }
                lo -= 1;
            }
            if lo == hi {
                break; // h[hi][hi] has converged
            }
            total += 1;
            if total > max_iter {
                return Err(DenseError::Schur);
            }
            // Shift: eigenvalue of the trailing 2×2 block closest to its last entry, or an
            // exceptional shift every 10 iterations.
            let shift = if its > 0 && its % 10 == 0 {
                h[(hi, hi)] + c64::new(0.75 * h[(hi, hi - 1)].norm(), 0.0)
            } else {
                let (aa, bb, cc, dd) = (h[(hi - 1, hi - 1)], h[(hi - 1, hi)], h[(hi, hi - 1)], h[(hi, hi)]);
                let half = (aa - dd) * 0.5;
                let disc = (half * half + bb * cc).sqrt();
                let (l1, l2) = ((aa + dd) * 0.5 + disc, (aa + dd) * 0.5 - disc);
                if (l1 - dd).norm() < (l2 - dd).norm() { l1 } else { l2 }
            };
            // Bulge chase from lo to hi.
            let (mut x, mut y) = (h[(lo, lo)] - shift, h[(lo + 1, lo)]);
            for k in lo..hi {
                if k > lo {
                    x = h[(k, k - 1)];
                    y = h[(k + 1, k - 1)];
                }
                let (c, s) = givens(x, y);
                let first = if k > lo { k - 1 } else { lo };
                rotate(&mut h, &mut q, k, c, s, first..n, 0..(k + 3).min(hi + 1));
                if k > lo {
                    h[(k + 1, k - 1)] = ZERO;
                }
            }
            its += 1;
        }
        if hi == 0 {
            break;
        }
        hi -= 1;
    }
    for j in 0..n {
        for i in j + 1..n {
            h[(i, j)] = ZERO;
        }
    }
    Ok((q, h))
}

/// Swaps the adjacent eigenvalues `k`, `k + 1` of the Schur form `(t, q)` (LAPACK ztrexc).
fn swap_adjacent(t: &mut Mat<c64>, q: &mut Mat<c64>, k: usize) {
    let n = t.nrows();
    let (t11, t22) = (t[(k, k)], t[(k + 1, k + 1)]);
    let (c, s) = givens(t[(k, k + 1)], t22 - t11);
    rotate(t, q, k, c, s, k + 2..n, 0..k);
    t[(k, k)] = t22;
    t[(k + 1, k + 1)] = t11;
}

/// Stable reordering of the Schur form `(t, q)` by increasing `keys`, by adjacent swaps on the
/// whole matrix. Cheap when few eigenvalues move; [`partition`] is the fast way to move many.
pub fn reorder(t: &mut Mat<c64>, q: &mut Mat<c64>, keys: &mut [usize]) {
    for i in 1..keys.len() {
        let mut j = i;
        while j > 0 && keys[j - 1] > keys[j] {
            swap_adjacent(t, q, j - 1);
            keys.swap(j - 1, j);
            j -= 1;
        }
    }
}

/// Moves the selected eigenvalues of the Schur form `(t, q)` to the leading positions, keeping
/// the order among the selected and among the others (as LAPACK ztrsen). Returns their number.
///
/// Batches of up to `BATCH` eigenvalues move up together through windows of `WINDOW` positions;
/// the rotations of a window are applied to the rest of `t` and to `q` as matrix products.
pub fn partition(t: &mut Mat<c64>, q: &mut Mat<c64>, select: &[bool]) -> usize {
    assert_eq!(select.len(), t.nrows(), "partition: one flag per eigenvalue");
    let positions: Vec<usize> = (0..select.len()).filter(|&k| select[k]).collect();
    let (mut placed, mut next) = (0, 0);
    while next < positions.len() {
        let first = positions[next];
        if first == placed {
            placed += 1;
            next += 1;
            continue;
        }
        // The batch: the next selected eigenvalues, spread over half a window at most. Those
        // before it are in place, so everything between `placed` and the batch is unselected.
        let mut k = 1;
        while next + k < positions.len() && k < BATCH && positions[next + k] < first + WINDOW / 2 {
            k += 1;
        }
        let batch = &positions[next..next + k];
        let mut end = batch[k - 1] + 1;
        let mut gathered = false;
        loop {
            let start = placed.max(end.saturating_sub(WINDOW));
            let local: Vec<bool> = (start..end)
                .map(|p| if gathered { p >= end - k } else { batch.binary_search(&p).is_ok() })
                .collect();
            window_partition(t, q, start, &local);
            gathered = true;
            if start == placed {
                break;
            }
            end = start + k;
        }
        placed += k;
        next += k;
    }
    positions.len()
}

/// Stable partition of the window `start..start + select.len()` of `t` (selected first). The
/// swaps act on a copy of the diagonal block, their product `u` on the rest.
fn window_partition(t: &mut Mat<c64>, q: &mut Mat<c64>, start: usize, select: &[bool]) {
    let (n, w) = (t.nrows(), select.len());
    let end = start + w;
    let mut tw = t.as_ref().submatrix(start, start, w, w).to_owned();
    let mut u = Mat::<c64>::identity(w, w);
    let (mut target, mut swaps) = (0, 0);
    for (j, &selected) in select.iter().enumerate() {
        if selected {
            for p in (target..j).rev() {
                swap_adjacent(&mut tw, &mut u, p);
                swaps += 1;
            }
            target += 1;
        }
    }
    if swaps == 0 {
        return;
    }
    t.as_mut().submatrix_mut(start, start, w, w).copy_from(&tw);
    if end < n {
        let right = t.as_ref().submatrix(start, end, w, n - end).to_owned();
        let dst = t.as_mut().submatrix_mut(start, end, w, n - end);
        matmul(dst, Accum::Replace, u.as_ref(), Op::Adjoint, right.as_ref(), Op::Plain, ONE);
    }
    if start > 0 {
        let above = t.as_ref().submatrix(0, start, start, w).to_owned();
        let dst = t.as_mut().submatrix_mut(0, start, start, w);
        matmul(dst, Accum::Replace, above.as_ref(), Op::Plain, u.as_ref(), Op::Plain, ONE);
    }
    let m = q.nrows();
    let cols = q.as_ref().submatrix(0, start, m, w).to_owned();
    matmul(
        q.as_mut().submatrix_mut(0, start, m, w),
        Accum::Replace,
        cols.as_ref(),
        Op::Plain,
        u.as_ref(),
        Op::Plain,
        ONE,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_matrix(n: usize) -> Mat<c64> {
        Mat::from_fn(n, n, |i, j| {
            c64::new(((i * 7 + j * 13) % 17) as f64 / 17.0 - 0.5, ((i * 3 + j * 5) % 11) as f64 / 11.0 - 0.5)
        })
    }

    fn unitarity(q: &Mat<c64>) -> f64 {
        let n = q.ncols();
        (q.adjoint() * q - Mat::<c64>::identity(n, n)).norm_max()
    }

    fn check_schur(a: &Mat<c64>, q: &Mat<c64>, t: &Mat<c64>, tol: f64) {
        let n = a.nrows();
        let err = (q * t * q.adjoint() - a).norm_max();
        assert!(err < tol && unitarity(q) < tol, "n {n}: {err:e}, {:e}", unitarity(q));
        for j in 0..n {
            for i in j + 1..n {
                assert_eq!(t[(i, j)], ZERO);
            }
        }
    }

    /// Both Schur decompositions (in-house and the build's backend): `a = q t q†`, eigenvalues as
    /// faer's.
    #[test]
    fn schur_decompositions() {
        for n in [1usize, 2, 5, 40, 97, 200] {
            let a = test_matrix(n);
            for (name, (q, t)) in
                [("in-house", schur(a.as_ref()).unwrap()), ("backend", crate::schur(a.as_ref()).unwrap())]
            {
                check_schur(&a, &q, &t, 1e-11);
                let mut ours: Vec<c64> = (0..n).map(|k| t[(k, k)]).collect();
                let mut theirs: Vec<c64> = a.eigenvalues().unwrap();
                let key = |z: &c64| (z.re * 1e6).round() as i64 * 1_000_000_000 + (z.im * 1e6).round() as i64;
                ours.sort_by_key(key);
                theirs.sort_by_key(key);
                assert!(
                    ours.iter().zip(&theirs).all(|(x, y)| (x - y).norm() < 1e-9),
                    "{name} n {n}: eigenvalues differ"
                );
            }
        }
    }

    /// The blocked partition moves the selected eigenvalues to the front, stably, and the leading
    /// Schur vectors span their invariant subspace; the result matches the unblocked swaps.
    #[test]
    fn partition_and_reorder() {
        for n in [1usize, 2, 5, 40, 97, 300] {
            let a = test_matrix(n);
            let (q0, t0) = crate::schur(a.as_ref()).unwrap();
            let eig: Vec<c64> = (0..n).map(|k| t0[(k, k)]).collect();
            let mut mods: Vec<f64> = eig.iter().map(|z| z.norm()).collect();
            mods.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for cut in [mods[n / 2], mods[n / 5], mods[(4 * n) / 5]] {
                let select: Vec<bool> = eig.iter().map(|z| z.norm() < cut).collect();
                let (mut q, mut t) = (q0.clone(), t0.clone());
                let m = partition(&mut t, &mut q, &select);
                check_schur(&a, &q, &t, 1e-10);
                let expected: Vec<c64> = eig
                    .iter()
                    .zip(&select)
                    .filter(|(_, s)| **s)
                    .chain(eig.iter().zip(&select).filter(|(_, s)| !**s))
                    .map(|(z, _)| *z)
                    .collect();
                assert!((0..n).all(|k| (t[(k, k)] - expected[k]).norm() < 1e-9 * (1.0 + expected[k].norm())));
                let q1 = q.as_ref().submatrix(0, 0, n, m).to_owned();
                let t11 = t.as_ref().submatrix(0, 0, m, m).to_owned();
                assert!((&a * &q1 - &q1 * &t11).norm_max() < 1e-10, "n {n}: invariant subspace");
                // Unblocked swaps give the same subspace.
                let (mut qu, mut tu) = (q0.clone(), t0.clone());
                let mut keys: Vec<usize> = select.iter().map(|&s| (!s) as usize).collect();
                reorder(&mut tu, &mut qu, &mut keys);
                let qu1 = qu.as_ref().submatrix(0, 0, n, m).to_owned();
                let overlap = qu1.adjoint() * &q1; // unitary when the subspaces agree
                assert!(unitarity(&overlap) < 1e-9, "n {n}: subspaces differ");
            }
        }
    }

    #[test]
    #[ignore]
    fn timing() {
        for n in [256usize, 512, 1024] {
            let a = test_matrix(n);
            let t0 = std::time::Instant::now();
            let (q0, t0m) = crate::schur(a.as_ref()).unwrap();
            let ts = t0.elapsed();
            let mut mods: Vec<f64> = (0..n).map(|k| t0m[(k, k)].norm()).collect();
            mods.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let select: Vec<bool> = (0..n).map(|k| t0m[(k, k)].norm() < mods[n / 2]).collect();
            let (mut q, mut t) = (q0.clone(), t0m.clone());
            let start = std::time::Instant::now();
            partition(&mut t, &mut q, &select);
            let tp = start.elapsed();
            let (mut q, mut t) = (q0.clone(), t0m.clone());
            let mut keys: Vec<usize> = select.iter().map(|&s| (!s) as usize).collect();
            let start = std::time::Instant::now();
            reorder(&mut t, &mut q, &mut keys);
            let tr = start.elapsed();
            println!(
                "n {n}: schur ({}) {ts:.1?} | half to the front: blocked {tp:.1?}, unblocked {tr:.1?}",
                crate::BACKEND
            );
        }
    }
}
