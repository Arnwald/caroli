//! Dense complex kernels for caroli: the Schur decomposition, its reordering, and matrix
//! products. They dominate the lead modes and the nested-dissection fronts.
//!
//! On macOS, with the `accelerate` feature (on by default), the Schur decomposition and the
//! products call Apple's Accelerate framework (LAPACK and BLAS, which run on the matrix units of
//! Apple silicon). Elsewhere, or with the feature off, they use faer and an in-house QR algorithm.
//! The reordering is in-house everywhere: adjacent swaps (LAPACK ztrexc) made in small windows,
//! whose rotations reach the rest of the matrix as products.

#[cfg(all(target_os = "macos", feature = "accelerate"))]
mod accelerate;
mod schur;

pub use faer::{Accum, Mat, MatMut, MatRef, c64};
pub use schur::{partition, reorder};

/// Errors of the dense kernels.
#[derive(Debug, thiserror::Error)]
pub enum DenseError {
    #[error("the Schur decomposition did not converge")]
    Schur,
}

/// Backend of the Schur decomposition and of the products in this build.
pub const BACKEND: &str =
    if cfg!(all(target_os = "macos", feature = "accelerate")) { "accelerate" } else { "faer" };

/// Complex Schur decomposition `a = q t q†`: `q` unitary, `t` upper triangular with exact zeros
/// below the diagonal. Returns `(q, t)`.
pub fn schur(a: MatRef<'_, c64>) -> Result<(Mat<c64>, Mat<c64>), DenseError> {
    assert_eq!(a.nrows(), a.ncols(), "schur: square matrix expected");
    #[cfg(all(target_os = "macos", feature = "accelerate"))]
    return accelerate::schur(a);
    #[cfg(not(all(target_os = "macos", feature = "accelerate")))]
    return schur::schur(a);
}

/// LU factorisation with partial pivoting of a square matrix.
#[derive(Clone, Debug)]
pub struct Lu {
    #[cfg(all(target_os = "macos", feature = "accelerate"))]
    inner: accelerate::Lu,
    #[cfg(not(all(target_os = "macos", feature = "accelerate")))]
    inner: faer::linalg::solvers::PartialPivLu<c64>,
}

impl Lu {
    pub fn new(a: MatRef<'_, c64>) -> Self {
        assert_eq!(a.nrows(), a.ncols(), "lu: square matrix expected");
        #[cfg(all(target_os = "macos", feature = "accelerate"))]
        return Self { inner: accelerate::Lu::new(a) };
        #[cfg(not(all(target_os = "macos", feature = "accelerate")))]
        return Self { inner: a.partial_piv_lu() };
    }

    /// Smallest and largest modulus of the pivots (the diagonal of `U`); `(∞, 0)` for an empty
    /// matrix.
    pub fn pivot_range(&self) -> (f64, f64) {
        #[cfg(all(target_os = "macos", feature = "accelerate"))]
        let diagonal: Vec<c64> = self.inner.diagonal().collect();
        #[cfg(not(all(target_os = "macos", feature = "accelerate")))]
        let diagonal: Vec<c64> = {
            let u = self.inner.U();
            (0..u.nrows()).map(|i| u[(i, i)]).collect()
        };
        diagonal.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), d| (lo.min(d.norm()), hi.max(d.norm())))
    }

    /// `b ← a⁻¹ b`.
    pub fn solve_in_place(&self, b: MatMut<'_, c64>) {
        #[cfg(all(target_os = "macos", feature = "accelerate"))]
        if let Err(mut b) = self.inner.solve_in_place(b) {
            // A stride LAPACK cannot take: solve a column-major copy.
            let mut copy = b.to_owned();
            let _ = self.inner.solve_in_place(copy.as_mut());
            b.copy_from(&copy);
        }
        #[cfg(not(all(target_os = "macos", feature = "accelerate")))]
        {
            use faer::linalg::solvers::SolveCore;
            self.inner.solve_in_place_with_conj(faer::Conj::No, b);
        }
    }

    /// `a⁻¹ b`.
    pub fn solve(&self, b: MatRef<'_, c64>) -> Mat<c64> {
        let mut x = b.to_owned();
        self.solve_in_place(x.as_mut());
        x
    }
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
    let dst = match accelerate::matmul(dst, accum, a, op_a, b, op_b, alpha) {
        Ok(()) => return,
        Err(dst) => dst, // strides BLAS cannot take
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
