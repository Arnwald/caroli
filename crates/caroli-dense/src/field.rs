//! The scalar of the elimination kernels: complex f64, or complex f32 for reduced precision.

use crate::{Lu, Op, matmul};
use faer::{Accum, Conj, Mat, MatRef, c32, c64};

/// A complex scalar the elimination kernels run in: [`c64`], or [`c32`] for reduced precision.
/// Through Accelerate on Apple silicon, the Schur step of a front of 128 to 2048 variables runs
/// 2.7–3.7× faster in `c32`.
pub trait Field:
    faer::traits::ComplexField
    + Copy
    + Send
    + Sync
    + PartialEq
    + std::fmt::Debug
    + 'static
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
{
    const ZERO: Self;
    const ONE: Self;
    /// The pivot test of [`Field::schur`]: a pivot below this fraction of the largest entry of
    /// the block counts as zero. About 10⁵ times the rounding error of `c64`; for `c32`, about
    /// 100 times its rounding error. A stricter `c32` test would refuse more ill-conditioned
    /// blocks, which nested dissection then delays to larger fronts: more accurate, but up to
    /// 3× slower than `c64` (`ARCHITECTURE.md`, item 12).
    const PIVOT_TOL: f64;

    fn from_c64(z: c64) -> Self;
    fn to_c64(self) -> c64;
    /// `|z|`, in f64.
    fn modulus(self) -> f64;
    /// The matrix in complex f64 (a move for `c64`).
    fn into_c64(m: Mat<Self>) -> Mat<c64>;

    /// Schur complement of the leading `ne × ne` block `f11` of the square matrix `f`:
    /// `(S, X)` with `X = f11⁻¹ f12` and `S = f22 − f21 X`, or `None` if a pivot of the LU of
    /// `f11` falls below [`Field::PIVOT_TOL`] times its largest entry (numerically singular).
    fn schur(f: MatRef<'_, Self>, ne: usize) -> Option<(Mat<Self>, Mat<Self>)>;
}

/// Largest modulus of the entries of `a`.
fn largest<T: Field>(a: MatRef<'_, T>) -> f64 {
    let mut m = 0.0f64;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            m = m.max(a[(i, j)].modulus());
        }
    }
    m
}

impl Field for c64 {
    const ZERO: Self = c64::new(0.0, 0.0);
    const ONE: Self = c64::new(1.0, 0.0);
    const PIVOT_TOL: f64 = 1e-11;

    fn from_c64(z: c64) -> Self {
        z
    }

    fn to_c64(self) -> c64 {
        self
    }

    fn modulus(self) -> f64 {
        self.norm()
    }

    fn into_c64(m: Mat<Self>) -> Mat<c64> {
        m
    }

    fn schur(f: MatRef<'_, Self>, ne: usize) -> Option<(Mat<Self>, Mat<Self>)> {
        let n = f.nrows();
        let f11 = f.submatrix(0, 0, ne, ne);
        let scale = f11.norm_max();
        let lu = Lu::new(f11);
        let (small, _) = lu.pivot_range();
        let stable = small > Self::PIVOT_TOL * scale; // false for NaN
        if !stable {
            return None;
        }
        let nb = n - ne;
        if nb == 0 {
            return Some((Mat::zeros(0, 0), Mat::zeros(ne, 0)));
        }
        let x = lu.solve(f.submatrix(0, ne, ne, nb));
        let mut s = f.submatrix(ne, ne, nb, nb).to_owned();
        matmul(
            s.as_mut(),
            Accum::Add,
            f.submatrix(ne, 0, nb, ne),
            Op::Plain,
            x.as_ref(),
            Op::Plain,
            -Self::ONE,
        );
        Some((s, x))
    }
}

impl Field for c32 {
    const ZERO: Self = c32::new(0.0, 0.0);
    const ONE: Self = c32::new(1.0, 0.0);
    const PIVOT_TOL: f64 = 1e-5;

    fn from_c64(z: c64) -> Self {
        c32::new(z.re as f32, z.im as f32)
    }

    fn to_c64(self) -> c64 {
        c64::new(self.re as f64, self.im as f64)
    }

    fn modulus(self) -> f64 {
        (self.re as f64).hypot(self.im as f64)
    }

    fn into_c64(m: Mat<Self>) -> Mat<c64> {
        Mat::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)].to_c64())
    }

    fn schur(f: MatRef<'_, Self>, ne: usize) -> Option<(Mat<Self>, Mat<Self>)> {
        #[cfg(all(target_os = "macos", feature = "accelerate"))]
        if crate::use_accelerate() {
            if let Some(result) = crate::accelerate::schur_c32(f, ne, Self::PIVOT_TOL) {
                return result;
            }
        }
        faer_schur(f, ne)
    }
}

/// [`Field::schur`] with faer's LU and product.
fn faer_schur<T: Field>(f: MatRef<'_, T>, ne: usize) -> Option<(Mat<T>, Mat<T>)> {
    use faer::linalg::solvers::SolveCore;
    let n = f.nrows();
    let f11 = f.submatrix(0, 0, ne, ne);
    let scale = largest(f11);
    let lu = f11.partial_piv_lu();
    let u = lu.U();
    let small = (0..ne).map(|i| u[(i, i)].modulus()).fold(f64::INFINITY, f64::min);
    let stable = small > T::PIVOT_TOL * scale; // false for NaN
    if !stable {
        return None;
    }
    let nb = n - ne;
    if nb == 0 {
        return Some((Mat::zeros(0, 0), Mat::zeros(ne, 0)));
    }
    let mut x = f.submatrix(0, ne, ne, nb).to_owned();
    lu.solve_in_place_with_conj(Conj::No, x.as_mut());
    let mut s = f.submatrix(ne, ne, nb, nb).to_owned();
    let minus_one = T::from_c64(c64::new(-1.0, 0.0));
    faer::linalg::matmul::matmul(
        s.as_mut(),
        Accum::Add,
        f.submatrix(ne, 0, nb, ne),
        x.as_ref(),
        minus_one,
        faer::get_global_parallelism(),
    );
    Some((s, x))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn front(n: usize) -> Mat<c64> {
        Mat::from_fn(n, n, |i, j| {
            let v = ((i * 7 + j * 13) % 17) as f64 / 17.0 - 0.5;
            c64::new(v + if i == j { 3.0 } else { 0.0 }, ((i + 2 * j) % 5) as f64 / 10.0)
        })
    }

    #[test]
    fn single_schur_follows_double() {
        for (n, ne) in [(12, 5), (90, 40)] {
            let f = front(n);
            let (s64, x64) = c64::schur(f.as_ref(), ne).unwrap();
            let f32m = Mat::from_fn(n, n, |i, j| c32::from_c64(f[(i, j)]));
            let (s32, x32) = c32::schur(f32m.as_ref(), ne).unwrap();
            let (s32, x32) = (c32::into_c64(s32), c32::into_c64(x32));
            let err = |a: &Mat<c64>, b: &Mat<c64>| (a - b).norm_max() / b.norm_max();
            assert!(err(&s32, &s64) < 1e-5 && err(&x32, &x64) < 1e-5, "n = {n}");
            // The faer path gives the same.
            let (s, _) = faer_schur(f32m.as_ref(), ne).unwrap();
            assert!(err(&c32::into_c64(s), &s64) < 1e-5);
        }
    }

    #[test]
    fn singular_block_is_refused() {
        let mut f = front(10);
        for j in 0..10 {
            f[(1, j)] = f[(0, j)]; // two equal rows in the leading block
        }
        assert!(c64::schur(f.as_ref(), 4).is_none());
        let f32m = Mat::from_fn(10, 10, |i, j| c32::from_c64(f[(i, j)]));
        assert!(c32::schur(f32m.as_ref(), 4).is_none());
    }
}
