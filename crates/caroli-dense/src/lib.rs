//! Dense complex kernels for caroli: the Schur decomposition, its reordering, LU solves, QR and
//! SVD, and matrix products. They dominate the lead modes and the nested-dissection fronts.
//!
//! On macOS, with the `accelerate` feature (on by default), they can call Apple's Accelerate
//! framework (LAPACK and BLAS, which run on the matrix units of Apple silicon); elsewhere, or
//! with the feature off, they use faer and an in-house QR algorithm. The reordering is in-house
//! everywhere: adjacent swaps (LAPACK ztrexc) made in small windows, whose rotations reach the
//! rest of the matrix as products.
//!
//! [`set_backend`] chooses the library at run time (see [`Backend`]); the environment variable
//! `CAROLI_DENSE_BACKEND` (`accelerate`, `faer`) sets the initial choice.

#[cfg(all(target_os = "macos", feature = "accelerate"))]
mod accelerate;
mod field;
mod schur;

pub use faer::{Accum, Mat, MatMut, MatRef, c32, c64};
pub use field::Field;
pub use schur::{partition, reorder};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Errors of the dense kernels.
#[derive(Debug, thiserror::Error)]
pub enum DenseError {
    #[error("the Schur decomposition did not converge")]
    Schur,
    #[error("the singular value decomposition did not converge")]
    Svd,
    #[error("the Hermitian eigensolver did not converge")]
    Eigen,
}

/// Libraries this build can use: `"accelerate"` (with faer) on macOS with the feature, else
/// `"faer"`.
pub const BACKEND: &str =
    if cfg!(all(target_os = "macos", feature = "accelerate")) { "accelerate" } else { "faer" };

/// Which library runs the dense kernels. Without Accelerate in the build, both choices run faer.
///
/// Accelerate is the default: its kernels run on the matrix units of Apple silicon, several
/// times faster than faer's on the vector units, and it stays ahead on every core at once
/// (`benchmarks/README.md`, multi-core throughput).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Accelerate for every call.
    Accelerate,
    /// faer (and the in-house Schur decomposition) for every call.
    Faer,
}

/// `0`: not set yet; `1`: Accelerate; `2`: faer.
static POLICY: AtomicUsize = AtomicUsize::new(0);

fn encode(b: Backend) -> usize {
    match b {
        Backend::Accelerate => 1,
        Backend::Faer => 2,
    }
}

/// Chooses the library of the dense kernels for the whole process.
pub fn set_backend(b: Backend) {
    POLICY.store(encode(b), Ordering::Relaxed);
}

/// The current choice (from `CAROLI_DENSE_BACKEND` at the first call, Accelerate by default).
pub fn backend() -> Backend {
    let mut code = POLICY.load(Ordering::Relaxed);
    if code == 0 {
        let chosen = std::env::var("CAROLI_DENSE_BACKEND").ok().and_then(|v| match v.as_str() {
            "accelerate" => Some(Backend::Accelerate),
            "faer" => Some(Backend::Faer),
            _ => None,
        });
        code = encode(chosen.unwrap_or(Backend::Accelerate));
        // Keep a choice made meanwhile by set_backend.
        code = match POLICY.compare_exchange(0, code, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => code,
            Err(current) => current,
        };
    }
    if code == 2 { Backend::Faer } else { Backend::Accelerate }
}

/// Whether Accelerate runs the next call.
#[cfg_attr(not(all(target_os = "macos", feature = "accelerate")), allow(dead_code))]
fn use_accelerate() -> bool {
    cfg!(all(target_os = "macos", feature = "accelerate")) && backend() == Backend::Accelerate
}

/// Complex Schur decomposition `a = q t q†`: `q` unitary, `t` upper triangular with exact zeros
/// below the diagonal. Returns `(q, t)`.
pub fn schur(a: MatRef<'_, c64>) -> Result<(Mat<c64>, Mat<c64>), DenseError> {
    assert_eq!(a.nrows(), a.ncols(), "schur: square matrix expected");
    #[cfg(all(target_os = "macos", feature = "accelerate"))]
    if use_accelerate() {
        return accelerate::schur(a);
    }
    schur::schur(a)
}

/// Complex Schur decomposition `a = q t q†` of a real matrix (given with zero imaginary parts):
/// the real Schur form (a quarter of the arithmetic of the complex one), whose 2 × 2 blocks of
/// complex pairs are then split by complex rotations (`rsf2csf`). Same output as [`schur`].
pub fn schur_of_real(a: MatRef<'_, c64>) -> Result<(Mat<c64>, Mat<c64>), DenseError> {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "schur_of_real: square matrix expected");
    #[cfg(all(target_os = "macos", feature = "accelerate"))]
    if use_accelerate() {
        let mut t: Vec<f64> = (0..n * n).map(|k| a[(k % n, k / n)].re).collect();
        if !t.iter().all(|x| x.is_finite()) {
            return Err(DenseError::Schur);
        }
        let mut q = vec![0.0; n * n];
        accelerate::real_schur(&mut t, &mut q, n)?;
        let mut tc = Mat::<c64>::from_fn(n, n, |i, j| c64::new(t[i + j * n], 0.0));
        let mut qc = Mat::<c64>::from_fn(n, n, |i, j| c64::new(q[i + j * n], 0.0));
        split_pairs(&mut tc, &mut qc);
        return Ok((qc, tc));
    }
    schur(a)
}

/// Turns a real quasi-triangular Schur form into a complex triangular one (as MATLAB's
/// `rsf2csf`): each 2 × 2 block of a complex pair is triangularised by a complex rotation applied
/// to its two rows, its two columns and the Schur vectors.
#[cfg_attr(not(all(target_os = "macos", feature = "accelerate")), allow(dead_code))]
fn split_pairs(t: &mut Mat<c64>, q: &mut Mat<c64>) {
    let n = t.nrows();
    let zero = c64::new(0.0, 0.0);
    for m in (1..n).rev() {
        let sub = t[(m, m - 1)];
        if sub == zero {
            continue;
        }
        // An eigenvalue of the block [[a, b], [c, d]], relative to d.
        let (a, b, c, d) = (t[(m - 1, m - 1)], t[(m - 1, m)], sub, t[(m, m)]);
        let half = (a - d) * 0.5;
        let mu = half + (half * half + b * c).sqrt(); // eigenvalue − d
        let r = (mu.norm_sqr() + c.norm_sqr()).sqrt();
        if r == 0.0 {
            continue;
        }
        let (cs, sn) = (mu / r, c / r);
        // G = [[cs*, sn*], [−sn, cs]] on rows m−1, m (columns m−1..n), G† on columns m−1, m.
        for j in m - 1..n {
            let (x, y) = (t[(m - 1, j)], t[(m, j)]);
            t[(m - 1, j)] = cs.conj() * x + sn.conj() * y;
            t[(m, j)] = -sn * x + cs * y;
        }
        for i in 0..=m {
            let (x, y) = (t[(i, m - 1)], t[(i, m)]);
            t[(i, m - 1)] = x * cs + y * sn;
            t[(i, m)] = -x * sn.conj() + y * cs.conj();
        }
        for i in 0..n {
            let (x, y) = (q[(i, m - 1)], q[(i, m)]);
            q[(i, m - 1)] = x * cs + y * sn;
            q[(i, m)] = -x * sn.conj() + y * cs.conj();
        }
        t[(m, m - 1)] = zero;
    }
}

/// LU factorisation with partial pivoting of a square matrix.
#[derive(Clone, Debug)]
pub struct Lu {
    inner: LuInner,
}

#[derive(Clone, Debug)]
enum LuInner {
    #[cfg(all(target_os = "macos", feature = "accelerate"))]
    Accelerate(accelerate::Lu),
    Faer(faer::linalg::solvers::PartialPivLu<c64>),
}

impl Lu {
    pub fn new(a: MatRef<'_, c64>) -> Self {
        assert_eq!(a.nrows(), a.ncols(), "lu: square matrix expected");
        #[cfg(all(target_os = "macos", feature = "accelerate"))]
        if use_accelerate() {
            return Self { inner: LuInner::Accelerate(accelerate::Lu::new(a)) };
        }
        Self { inner: LuInner::Faer(a.partial_piv_lu()) }
    }

    /// Smallest and largest modulus of the pivots (the diagonal of `U`); `(∞, 0)` for an empty
    /// matrix.
    pub fn pivot_range(&self) -> (f64, f64) {
        let diagonal: Vec<c64> = match &self.inner {
            #[cfg(all(target_os = "macos", feature = "accelerate"))]
            LuInner::Accelerate(lu) => lu.diagonal().collect(),
            LuInner::Faer(lu) => {
                let u = lu.U();
                (0..u.nrows()).map(|i| u[(i, i)]).collect()
            }
        };
        diagonal.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), d| (lo.min(d.norm()), hi.max(d.norm())))
    }

    /// `b ← a⁻¹ b`.
    pub fn solve_in_place(&self, b: MatMut<'_, c64>) {
        self.solve_with(b, false);
    }

    /// `b ← a⁻† b`.
    pub fn solve_adjoint_in_place(&self, b: MatMut<'_, c64>) {
        self.solve_with(b, true);
    }

    fn solve_with(&self, b: MatMut<'_, c64>, adjoint: bool) {
        match &self.inner {
            #[cfg(all(target_os = "macos", feature = "accelerate"))]
            LuInner::Accelerate(lu) => {
                if let Err(mut b) = lu.solve_in_place(b, adjoint) {
                    // A stride LAPACK cannot take: solve a column-major copy.
                    let mut copy = b.to_owned();
                    let _ = lu.solve_in_place(copy.as_mut(), adjoint);
                    b.copy_from(&copy);
                }
            }
            LuInner::Faer(lu) => {
                use faer::linalg::solvers::SolveCore;
                if adjoint {
                    lu.solve_transpose_in_place_with_conj(faer::Conj::Yes, b);
                } else {
                    lu.solve_in_place_with_conj(faer::Conj::No, b);
                }
            }
        }
    }

    /// Estimate of the reciprocal condition number `1 / (‖a‖₁ ‖a⁻¹‖₁)`, given `‖a‖₁`: 0 for a
    /// singular matrix. As LAPACK zgecon, `‖a⁻¹‖₁` comes from a few solves with `a` and `a†`
    /// (Hager's method with Higham's refinements, zlacn2). The 2-norm ratio `σ_min / σ_max` is
    /// within a factor n of it.
    pub fn inverse_condition_1(&self, norm_1: f64) -> f64 {
        let n = self.pivots_len();
        if n == 0 {
            return 1.0;
        }
        let (lo, _) = self.pivot_range();
        let invertible = lo > 0.0 && norm_1 > 0.0; // false for NaN
        if !invertible {
            return 0.0;
        }
        let estimate = self.inverse_norm_1();
        if estimate.is_finite() && estimate > 0.0 { 1.0 / (norm_1 * estimate) } else { 0.0 }
    }

    fn pivots_len(&self) -> usize {
        match &self.inner {
            #[cfg(all(target_os = "macos", feature = "accelerate"))]
            LuInner::Accelerate(lu) => lu.diagonal().count(),
            LuInner::Faer(lu) => lu.U().nrows(),
        }
    }
    /// Lower estimate of `‖a⁻¹‖₁` (LAPACK zlacn2, at most 5 iterations).
    fn inverse_norm_1(&self) -> f64 {
        let n = self.pivots_len();
        let one = c64::new(1.0, 0.0);
        let apply = |x: Mat<c64>, adjoint: bool| {
            let mut y = x;
            self.solve_with(y.as_mut(), adjoint);
            y
        };
        let norm = |y: &Mat<c64>| (0..n).map(|i| y[(i, 0)].norm()).sum::<f64>();
        let sign = |y: &Mat<c64>| {
            Mat::from_fn(n, 1, |i, _| {
                let a = y[(i, 0)].norm();
                if a > f64::MIN_POSITIVE { y[(i, 0)] / a } else { one }
            })
        };
        let argmax =
            |z: &Mat<c64>| (0..n).max_by(|&a, &b| z[(a, 0)].norm().total_cmp(&z[(b, 0)].norm())).unwrap_or(0);
        let y = apply(Mat::from_fn(n, 1, |_, _| c64::new(1.0 / n as f64, 0.0)), false);
        if n == 1 {
            return y[(0, 0)].norm();
        }
        let mut estimate = norm(&y);
        let z = apply(sign(&y), true);
        let mut j = argmax(&z);
        for _ in 2..=5 {
            let y = apply(Mat::from_fn(n, 1, |i, _| if i == j { one } else { c64::new(0.0, 0.0) }), false);
            let previous = estimate;
            estimate = estimate.max(norm(&y));
            if estimate <= previous {
                break;
            }
            let z = apply(sign(&y), true);
            let last = j;
            j = argmax(&z);
            if z[(last, 0)].norm() == z[(j, 0)].norm() {
                break;
            }
        }
        // Alternative estimate from x_i = ±(1 + i/(n − 1)).
        let x = Mat::from_fn(n, 1, |i, _| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            c64::new(sign * (1.0 + i as f64 / (n - 1) as f64), 0.0)
        });
        estimate.max(2.0 * norm(&apply(x, false)) / (3.0 * n as f64))
    }

    /// `a⁻¹ b`.
    pub fn solve(&self, b: MatRef<'_, c64>) -> Mat<c64> {
        let mut x = b.to_owned();
        self.solve_in_place(x.as_mut());
        x
    }
}

/// `‖a‖₁`, the largest column sum of moduli.
pub fn norm_1(a: MatRef<'_, c64>) -> f64 {
    (0..a.ncols()).map(|j| (0..a.nrows()).map(|i| a[(i, j)].norm()).sum::<f64>()).fold(0.0, f64::max)
}

/// Orthonormal basis of the columns of a full-rank `m × n` matrix, `m ≥ n`: the thin `Q` of its
/// QR factorisation.
pub fn orthonormal_columns(a: MatRef<'_, c64>) -> Mat<c64> {
    assert!(a.nrows() >= a.ncols(), "orthonormal_columns: more columns than rows");
    #[cfg(all(target_os = "macos", feature = "accelerate"))]
    if use_accelerate() {
        return accelerate::orthonormal_columns(a);
    }
    a.qr().compute_thin_Q()
}

/// Thin singular value decomposition `a = u diag(s) v†`.
#[derive(Clone, Debug)]
pub struct Svd {
    pub u: Mat<c64>,
    /// Singular values, in decreasing order.
    pub s: Vec<f64>,
    pub v: Mat<c64>,
}

/// Thin singular value decomposition of `a`.
pub fn svd(a: MatRef<'_, c64>) -> Result<Svd, DenseError> {
    #[cfg(all(target_os = "macos", feature = "accelerate"))]
    if use_accelerate() {
        return accelerate::svd(a);
    }
    let svd = a.thin_svd().map_err(|_| DenseError::Svd)?;
    let s = (0..a.nrows().min(a.ncols())).map(|k| svd.S()[k].re).collect();
    Ok(Svd { u: svd.U().to_owned(), s, v: svd.V().to_owned() })
}

/// Eigen-decomposition `a = v diag(w) v†` of a Hermitian matrix, from its lower triangle:
/// eigenvalues in increasing order, orthonormal eigenvectors in columns.
pub fn hermitian_eigen(a: MatRef<'_, c64>) -> Result<(Vec<f64>, Mat<c64>), DenseError> {
    assert_eq!(a.nrows(), a.ncols(), "hermitian_eigen: square matrix expected");
    #[cfg(all(target_os = "macos", feature = "accelerate"))]
    if use_accelerate() {
        return accelerate::hermitian_eigen(a);
    }
    let evd = a.self_adjoint_eigen(faer::Side::Lower).map_err(|_| DenseError::Eigen)?;
    Ok(((0..a.nrows()).map(|k| evd.S()[k].re).collect(), evd.U().to_owned()))
}

/// Whether a matrix has no imaginary part at all.
pub fn is_real(a: MatRef<'_, c64>) -> bool {
    (0..a.ncols()).all(|j| (0..a.nrows()).all(|i| a[(i, j)].im == 0.0))
}

/// Thin singular value decomposition of a real matrix (given with zero imaginary parts), with
/// real singular vectors (a complex decomposition may mix degenerate ones with phases). Without
/// Accelerate, [`svd`].
pub fn svd_of_real(a: MatRef<'_, c64>) -> Result<Svd, DenseError> {
    #[cfg(all(target_os = "macos", feature = "accelerate"))]
    if use_accelerate() {
        let (m, n) = (a.nrows(), a.ncols());
        let k = m.min(n);
        let mut re: Vec<f64> = (0..m * n).map(|x| a[(x % m, x / m)].re).collect();
        let (u, s, vt) = accelerate::real_svd(&mut re, m, n)?;
        return Ok(Svd {
            u: Mat::from_fn(m, k, |i, j| c64::new(u[i + j * m], 0.0)),
            s,
            v: Mat::from_fn(n, k, |i, j| c64::new(vt[j + i * k], 0.0)),
        });
    }
    svd(a)
}

/// How a factor enters a product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Plain,
    Transpose,
    Adjoint,
}

/// `dst = α op(a) op(b)` (`Accum::Replace`) or `dst += α op(a) op(b)` (`Accum::Add`).
pub fn matmul(
    dst: MatMut<'_, c64>,
    accum: Accum,
    a: MatRef<'_, c64>,
    op_a: Op,
    b: MatRef<'_, c64>,
    op_b: Op,
    alpha: c64,
) {
    let shape = |m: MatRef<'_, c64>, op: Op| {
        if op == Op::Plain { (m.nrows(), m.ncols()) } else { (m.ncols(), m.nrows()) }
    };
    let ((ra, ka), (kb, cb)) = (shape(a, op_a), shape(b, op_b));
    assert!(ra == dst.nrows() && cb == dst.ncols() && ka == kb, "matmul: dimension mismatch");
    #[cfg(all(target_os = "macos", feature = "accelerate"))]
    let dst = if use_accelerate() {
        match accelerate::matmul(dst, accum, a, op_a, b, op_b, alpha) {
            Ok(()) => return,
            Err(dst) => dst, // strides BLAS cannot take
        }
    } else {
        dst
    };
    faer_matmul(dst, accum, a, op_a, b, op_b, alpha);
}

fn faer_matmul(
    dst: MatMut<'_, c64>,
    accum: Accum,
    a: MatRef<'_, c64>,
    op_a: Op,
    b: MatRef<'_, c64>,
    op_b: Op,
    alpha: c64,
) {
    use faer::linalg::matmul::matmul as mm;
    let par = faer::get_global_parallelism();
    match (op_a, op_b) {
        (Op::Plain, Op::Plain) => mm(dst, accum, a, b, alpha, par),
        (Op::Plain, Op::Transpose) => mm(dst, accum, a, b.transpose(), alpha, par),
        (Op::Plain, Op::Adjoint) => mm(dst, accum, a, b.adjoint(), alpha, par),
        (Op::Transpose, Op::Plain) => mm(dst, accum, a.transpose(), b, alpha, par),
        (Op::Transpose, Op::Transpose) => mm(dst, accum, a.transpose(), b.transpose(), alpha, par),
        (Op::Transpose, Op::Adjoint) => mm(dst, accum, a.transpose(), b.adjoint(), alpha, par),
        (Op::Adjoint, Op::Plain) => mm(dst, accum, a.adjoint(), b, alpha, par),
        (Op::Adjoint, Op::Transpose) => mm(dst, accum, a.adjoint(), b.transpose(), alpha, par),
        (Op::Adjoint, Op::Adjoint) => mm(dst, accum, a.adjoint(), b.adjoint(), alpha, par),
    }
}

/// `op(a) op(b)` as a new matrix.
pub fn product(a: MatRef<'_, c64>, op_a: Op, b: MatRef<'_, c64>, op_b: Op) -> Mat<c64> {
    let rows = if op_a == Op::Plain { a.nrows() } else { a.ncols() };
    let cols = if op_b == Op::Plain { b.ncols() } else { b.nrows() };
    let mut c = Mat::zeros(rows, cols);
    matmul(c.as_mut(), Accum::Replace, a, op_a, b, op_b, c64::new(1.0, 0.0));
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_matrix(m: usize, n: usize, seed: usize) -> Mat<c64> {
        Mat::from_fn(m, n, |i, j| {
            c64::new(
                ((i * 7 + j * 13 + seed) % 17) as f64 / 17.0 - 0.5,
                ((i * 3 + j * 5 + 2 * seed) % 11) as f64 / 11.0 - 0.5,
            )
        })
    }

    /// Every backend gives the same results.
    #[test]
    fn backends_agree() {
        let a = test_matrix(60, 60, 3) + Mat::<c64>::identity(60, 60) * faer::Scale(c64::new(1.5, 0.2));
        let b = test_matrix(60, 4, 4);
        let run =
            || {
                let x = Lu::new(a.as_ref()).solve(b.as_ref());
                let p = product(a.as_ref(), Op::Adjoint, b.as_ref(), Op::Plain);
                let s = svd(a.as_ref()).unwrap().s;
                let h = &a + a.adjoint();
                let (w, v) = hermitian_eigen(h.as_ref()).unwrap();
                let d = Mat::<c64>::from_fn(60, 60, |i, j| {
                    if i == j { c64::new(w[i], 0.0) } else { c64::new(0.0, 0.0) }
                });
                assert!((&v * d * v.adjoint() - &h).norm_max() < 1e-11);
                assert!(w.windows(2).all(|p| p[0] <= p[1]));
                (x, p, s, w)
            };
        let saved = backend();
        let reference = run();
        for choice in [Backend::Accelerate, Backend::Faer] {
            set_backend(choice);
            assert_eq!(backend(), choice);
            let (x, p, s, w) = run();
            assert!(
                (&x - &reference.0).norm_max() < 1e-11 && (&p - &reference.1).norm_max() < 1e-11,
                "{choice:?}"
            );
            assert!(s.iter().zip(&reference.2).all(|(u, v)| (u - v).abs() < 1e-11), "{choice:?}");
            assert!(w.iter().zip(&reference.3).all(|(u, v)| (u - v).abs() < 1e-11), "{choice:?}");
        }
        set_backend(saved);
    }

    /// The Schur form of a real matrix through its real Schur form, against the complex one.
    #[test]
    fn schur_of_real_matrices() {
        for n in [1usize, 2, 7, 60, 257] {
            let a = Mat::<c64>::from_fn(n, n, |i, j| c64::new(test_matrix(n, n, 5)[(i, j)].re, 0.0));
            let (q, t) = schur_of_real(a.as_ref()).unwrap_or_else(|e| panic!("n {n}: {e:?}"));
            assert!((&q * &t * q.adjoint() - &a).norm_max() < 1e-12 * n as f64, "n {n}: q t q† = a");
            assert!(
                (q.adjoint() * &q - Mat::<c64>::identity(n, n)).norm_max() < 1e-12 * n as f64,
                "n {n}: unitary"
            );
            assert!((0..n).all(|j| (j + 1..n).all(|i| t[(i, j)] == c64::new(0.0, 0.0))), "n {n}: triangular");
            let (_, tc) = schur(a.as_ref()).unwrap();
            // Each eigenvalue matches one of the complex decomposition.
            let mut theirs: Vec<c64> = (0..n).map(|k| tc[(k, k)]).collect();
            for k in 0..n {
                let mine = t[(k, k)];
                let (best, d) = theirs
                    .iter()
                    .enumerate()
                    .map(|(i, x)| (i, (x - mine).norm()))
                    .min_by(|x, y| x.1.total_cmp(&y.1))
                    .unwrap();
                assert!(d < 1e-9, "n {n}: eigenvalue {mine}");
                theirs.swap_remove(best);
            }
        }
    }

    /// The real SVD reconstructs the matrix with real factors.
    #[test]
    fn svd_of_real_matrices() {
        for (m, n) in [(7usize, 3usize), (40, 40), (30, 50)] {
            let a = Mat::<c64>::from_fn(m, n, |i, j| c64::new(test_matrix(m, n, 9)[(i, j)].re, 0.0));
            let Svd { u, s, v } = svd_of_real(a.as_ref()).unwrap();
            let k = m.min(n);
            let sigma =
                Mat::<c64>::from_fn(
                    k,
                    k,
                    |i, j| if i == j { c64::new(s[i], 0.0) } else { c64::new(0.0, 0.0) },
                );
            assert!((&u * sigma * v.adjoint() - &a).norm_max() < 1e-12, "{m}x{n}");
            assert!(is_real(u.as_ref()) && is_real(v.as_ref()));
        }
    }

    /// Time of the real path against the complex one (`cargo test --release -- --ignored`).
    #[test]
    #[ignore]
    fn schur_of_real_timing() {
        for n in [256usize, 512, 1024] {
            let a = Mat::<c64>::from_fn(n, n, |i, j| {
                c64::new(((i * 7919 + j * 104729) % 1009) as f64 / 1009.0 - 0.5, 0.0)
            });
            let t0 = std::time::Instant::now();
            let _ = schur(a.as_ref()).unwrap();
            let complex = t0.elapsed();
            let t0 = std::time::Instant::now();
            let _ = schur_of_real(a.as_ref()).unwrap();
            println!("n {n}: complex {complex:.2?}, real {:.2?}", t0.elapsed());
        }
    }

    /// LU solves against faer, with a strided right-hand side, and the pivots of a singular matrix.
    #[test]
    fn lu_solves() {
        for n in [1usize, 4, 37, 120] {
            let a = test_matrix(n, n, 7) + Mat::<c64>::identity(n, n) * faer::Scale(c64::new(2.0, 0.5));
            let b = test_matrix(n, 5, 8);
            let lu = Lu::new(a.as_ref());
            let x = lu.solve(b.as_ref());
            assert!((&a * &x - &b).norm_max() < 1e-12, "n {n}");
            let mut big = test_matrix(n + 2, 9, 9);
            big.as_mut().submatrix_mut(1, 2, n, 5).copy_from(&b);
            lu.solve_in_place(big.as_mut().submatrix_mut(1, 2, n, 5));
            assert!((big.as_ref().submatrix(1, 2, n, 5).to_owned() - &x).norm_max() < 1e-12);
            let (lo, hi) = lu.pivot_range();
            assert!(lo > 0.0 && hi >= lo);
        }
        let mut s = test_matrix(6, 6, 1);
        for i in 0..6 {
            s[(i, 5)] = s[(i, 0)] * 2.0; // dependent columns
        }
        let (lo, hi) = Lu::new(s.as_ref()).pivot_range();
        assert!(lo < 1e-14 * hi, "singular: {lo:e} / {hi:e}");
    }

    /// The condition estimate against the singular values, and solves with the adjoint.
    #[test]
    fn condition_estimate() {
        for n in [1usize, 5, 40, 150] {
            let mut a = test_matrix(n, n, 3);
            for i in 0..n {
                a[(i, i)] += c64::new(0.05 * i as f64, 0.0);
            }
            let lu = Lu::new(a.as_ref());
            let b = test_matrix(n, 3, 4);
            let mut x = b.clone();
            lu.solve_adjoint_in_place(x.as_mut());
            assert!((a.adjoint() * &x - &b).norm_max() < 1e-9 * (1.0 + x.norm_max()), "n {n}: adjoint solve");
            let s = a.singular_values().unwrap();
            let ratio =
                s.iter().cloned().fold(f64::INFINITY, f64::min) / s.iter().cloned().fold(0.0, f64::max);
            let rcond = lu.inverse_condition_1(norm_1(a.as_ref()));
            assert!(
                rcond <= n as f64 * ratio * (1.0 + 1e-9) && rcond >= ratio / (n as f64 * n as f64),
                "n {n}: {rcond:e} vs {ratio:e}"
            );
        }
        let mut s = test_matrix(6, 6, 1);
        for i in 0..6 {
            s[(i, 5)] = s[(i, 0)] * 2.0;
        }
        assert!(Lu::new(s.as_ref()).inverse_condition_1(norm_1(s.as_ref())) < 1e-14);
    }

    #[test]
    fn qr_and_svd() {
        for (m, n) in [(7usize, 3usize), (60, 60), (90, 40)] {
            let a = test_matrix(m, n, 5)
                + Mat::<c64>::from_fn(
                    m,
                    n,
                    |i, j| if i == j { c64::new(1.0, 0.0) } else { c64::new(0.0, 0.0) },
                );
            let q = orthonormal_columns(a.as_ref());
            assert!((q.adjoint() * &q - Mat::<c64>::identity(n, n)).norm_max() < 1e-13);
            // Same column space: a = q (q† a).
            assert!((&q * (q.adjoint() * &a) - &a).norm_max() < 1e-12);
            let Svd { u, s, v } = svd(a.as_ref()).unwrap();
            let k = m.min(n);
            let sigma =
                Mat::<c64>::from_fn(
                    k,
                    k,
                    |i, j| if i == j { c64::new(s[i], 0.0) } else { c64::new(0.0, 0.0) },
                );
            assert!((&u * sigma * v.adjoint() - &a).norm_max() < 1e-12, "{m}x{n}");
            assert!(s.windows(2).all(|w| w[0] >= w[1]));
        }
    }

    /// Every combination of operations, both accumulation modes, and strided views.
    #[test]
    fn products_match_faer() {
        let ops = [Op::Plain, Op::Transpose, Op::Adjoint];
        let alpha = c64::new(0.7, -0.4);
        for (m, n, k) in [(5, 7, 3), (40, 33, 65), (1, 9, 4), (6, 1, 1), (3, 4, 0)] {
            for &op_a in &ops {
                for &op_b in &ops {
                    let a = if op_a == Op::Plain { test_matrix(m, k, 1) } else { test_matrix(k, m, 1) };
                    let b = if op_b == Op::Plain { test_matrix(k, n, 2) } else { test_matrix(n, k, 2) };
                    let opm = |x: &Mat<c64>, op: Op| match op {
                        Op::Plain => x.clone(),
                        Op::Transpose => x.transpose().to_owned(),
                        Op::Adjoint => x.adjoint().to_owned(),
                    };
                    let reference = (opm(&a, op_a) * opm(&b, op_b)) * faer::Scale(alpha);
                    // Replace, into a view of a larger matrix (column stride above the row count).
                    let mut big = test_matrix(m + 3, n + 2, 3);
                    matmul(
                        big.as_mut().submatrix_mut(1, 2, m, n),
                        Accum::Replace,
                        a.as_ref(),
                        op_a,
                        b.as_ref(),
                        op_b,
                        alpha,
                    );
                    let got = big.as_ref().submatrix(1, 2, m, n).to_owned();
                    assert!((&got - &reference).norm_max() < 1e-13, "replace {op_a:?} {op_b:?} {m}x{n}x{k}");
                    // Add.
                    let mut c = test_matrix(m, n, 4);
                    let expected = &c + &reference;
                    matmul(c.as_mut(), Accum::Add, a.as_ref(), op_a, b.as_ref(), op_b, alpha);
                    assert!((&c - &expected).norm_max() < 1e-13, "add {op_a:?} {op_b:?} {m}x{n}x{k}");
                }
            }
        }
        // A row-major view goes through the fallback.
        let a = test_matrix(6, 5, 5);
        let b = test_matrix(6, 4, 6);
        let at = a.transpose(); // row stride ≠ 1
        let mut c = Mat::zeros(5, 4);
        matmul(c.as_mut(), Accum::Replace, at, Op::Plain, b.as_ref(), Op::Plain, c64::new(1.0, 0.0));
        assert!((&c - a.transpose() * &b).norm_max() < 1e-13);
    }
}
