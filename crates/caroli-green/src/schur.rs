//! Complex Schur decomposition `a = q t q†` with reordering, for the invariant subspaces of lead
//! modes. Eigenvectors of the transfer problem can be nearly dependent (strongly decaying
//! evanescent modes); orthonormal Schur vectors span the same invariant subspaces stably.
//!
//! - Hessenberg reduction: faer.
//! - QR iterations: implicit single-shift, with Wilkinson shifts and deflation (LAPACK zlahqr).
//! - Reordering: adjacent swaps by Givens rotations (LAPACK ztrexc).

use crate::GreenError;
use caroli_model::{CMat, c64};

/// Givens rotation `(c, s)` with `[c s; −s̄ c] [f; g] = [r; 0]` (LAPACK zlartg).
fn givens(f: c64, g: c64) -> (f64, c64) {
    if g == c64::new(0.0, 0.0) {
        return (1.0, c64::new(0.0, 0.0));
    }
    if f == c64::new(0.0, 0.0) {
        return (0.0, g.conj() / g.norm());
    }
    let d = f.norm().hypot(g.norm());
    (f.norm() / d, f / f.norm() * g.conj() / d)
}

/// `H ← G H G†` and `Z ← Z G†` for the rotation `G = [c s; −s̄ c]` on indices `k, k+1`, rows
/// restricted to `cols` and columns to `rows`.
fn rotate(
    h: &mut CMat,
    z: &mut CMat,
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

/// Complex Schur decomposition `a = q t q†` (`t` upper triangular).
pub(crate) fn schur(a: &CMat) -> Result<(CMat, CMat), GreenError> {
    use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
    use faer::linalg::evd::hessenberg;
    use faer::linalg::householder;
    use faer::{Conj, Par, Spec};
    let n = a.nrows();
    let zero = c64::new(0.0, 0.0);
    let mut h = a.clone();
    let mut q = CMat::from_fn(n, n, |i, j| if i == j { c64::new(1.0, 0.0) } else { zero });
    if n <= 1 {
        return Ok((q, h));
    }
    if !(h.norm_max().is_finite()) {
        return Err(GreenError::Eigen);
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
    let mut hh = CMat::zeros(bs, n - 1);
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
            h[(i, j)] = zero;
        }
    }

    // Implicit single-shift QR on the active block [lo, hi].
    let ulp = f64::EPSILON;
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
                if sub <= ulp * scale {
                    h[(lo, lo - 1)] = zero;
                    break;
                }
                lo -= 1;
            }
            if lo == hi {
                break; // h[hi][hi] has converged
            }
            total += 1;
            if total > max_iter {
                return Err(GreenError::Eigen);
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
                    h[(k + 1, k - 1)] = zero;
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
            h[(i, j)] = zero;
        }
    }
    Ok((q, h))
}

/// Swaps the adjacent eigenvalues `k`, `k + 1` of the Schur form `(t, q)` (LAPACK ztrexc).
fn swap_adjacent(t: &mut CMat, q: &mut CMat, k: usize) {
    let n = t.nrows();
    let (t11, t22) = (t[(k, k)], t[(k + 1, k + 1)]);
    let (c, s) = givens(t[(k, k + 1)], t22 - t11);
    rotate(t, q, k, c, s, k + 2..n, 0..k);
    t[(k, k)] = t22;
    t[(k + 1, k + 1)] = t11;
}

/// Stable reordering of the Schur form by increasing `keys` (adjacent swaps).
pub(crate) fn reorder(t: &mut CMat, q: &mut CMat, keys: &mut [usize]) {
    for i in 1..keys.len() {
        let mut j = i;
        while j > 0 && keys[j - 1] > keys[j] {
            swap_adjacent(t, q, j - 1);
            keys.swap(j - 1, j);
            j -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_matrix(n: usize) -> CMat {
        CMat::from_fn(n, n, |i, j| {
            c64::new(((i * 7 + j * 13) % 17) as f64 / 17.0 - 0.5, ((i * 3 + j * 5) % 11) as f64 / 11.0 - 0.5)
        })
    }

    #[test]
    #[ignore]
    fn schur_timing() {
        for n in [80usize, 198, 512] {
            let a = test_matrix(n);
            let t0 = std::time::Instant::now();
            let _ = schur(&a).unwrap();
            let ts = t0.elapsed();
            let t0 = std::time::Instant::now();
            let _ = a.eigen().unwrap();
            println!("n {n}: schur {ts:.2?}, faer eigen {:.2?}", t0.elapsed());
        }
    }

    /// `a = q t q†` with `q` unitary and `t` upper triangular, eigenvalues as faer's; then a
    /// reordering keeps the decomposition and moves the selected eigenvalues to the front.
    #[test]
    fn schur_and_reorder() {
        for n in [1usize, 2, 5, 40, 97] {
            let a = test_matrix(n);
            let (mut q, mut t) = schur(&a).unwrap();
            let err = (&q * &t * q.adjoint() - &a).norm_max();
            let orth = (q.adjoint() * &q
                - CMat::from_fn(n, n, |i, j| if i == j { c64::new(1.0, 0.0) } else { c64::new(0.0, 0.0) }))
            .norm_max();
            assert!(err < 1e-12 && orth < 1e-12, "n {n}: {err:e}, {orth:e}");
            for j in 0..n {
                for i in j + 1..n {
                    assert_eq!(t[(i, j)], c64::new(0.0, 0.0));
                }
            }
            let mut ours: Vec<c64> = (0..n).map(|k| t[(k, k)]).collect();
            let mut theirs: Vec<c64> = a.eigenvalues().unwrap();
            let key = |z: &c64| (z.re * 1e6).round() as i64 * 1_000_000_000 + (z.im * 1e6).round() as i64;
            ours.sort_by_key(key);
            theirs.sort_by_key(key);
            assert!(
                ours.iter().zip(&theirs).all(|(x, y)| (x - y).norm() < 1e-9),
                "n {n}: eigenvalues differ"
            );
            // Move the eigenvalues of modulus below the median to the front.
            let mut mods: Vec<f64> = ours.iter().map(|z| z.norm()).collect();
            mods.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = mods[n / 2];
            let mut keys: Vec<usize> = (0..n).map(|k| (t[(k, k)].norm() >= median) as usize).collect();
            let small = keys.iter().filter(|&&k| k == 0).count();
            reorder(&mut t, &mut q, &mut keys);
            assert!((&q * &t * q.adjoint() - &a).norm_max() < 1e-11);
            assert!(
                (0..small).all(|k| t[(k, k)].norm() < median)
                    && (small..n).all(|k| t[(k, k)].norm() >= median)
            );
            // The leading Schur vectors span an invariant subspace: a q₁ = q₁ t₁₁.
            let q1 = CMat::from_fn(n, small, |i, j| q[(i, j)]);
            let t11 = CMat::from_fn(small, small, |i, j| t[(i, j)]);
            assert!((&a * &q1 - &q1 * &t11).norm_max() < 1e-11, "n {n}: invariant subspace");
        }
    }
}
