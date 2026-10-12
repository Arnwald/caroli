//! Apple Accelerate: LAPACK 3.9+ (the `$NEWLAPACK` symbols, 32-bit integers) and BLAS.

use crate::{DenseError, Op, Svd};
use faer::{Accum, Mat, MatMut, MatRef, c64};

type Int = i32;

unsafe extern "C" {
    #[link_name = "zgehrd$NEWLAPACK"]
    fn zgehrd(
        n: *const Int,
        ilo: *const Int,
        ihi: *const Int,
        a: *mut c64,
        lda: *const Int,
        tau: *mut c64,
        work: *mut c64,
        lwork: *const Int,
        info: *mut Int,
    );
    #[link_name = "zunghr$NEWLAPACK"]
    fn zunghr(
        n: *const Int,
        ilo: *const Int,
        ihi: *const Int,
        a: *mut c64,
        lda: *const Int,
        tau: *const c64,
        work: *mut c64,
        lwork: *const Int,
        info: *mut Int,
    );
    #[link_name = "zhseqr$NEWLAPACK"]
    fn zhseqr(
        job: *const u8,
        compz: *const u8,
        n: *const Int,
        ilo: *const Int,
        ihi: *const Int,
        h: *mut c64,
        ldh: *const Int,
        w: *mut c64,
        z: *mut c64,
        ldz: *const Int,
        work: *mut c64,
        lwork: *const Int,
        info: *mut Int,
    );
    #[link_name = "zgetrf$NEWLAPACK"]
    fn zgetrf(m: *const Int, n: *const Int, a: *mut c64, lda: *const Int, ipiv: *mut Int, info: *mut Int);
    #[link_name = "zgetrs$NEWLAPACK"]
    fn zgetrs(
        trans: *const u8,
        n: *const Int,
        nrhs: *const Int,
        a: *const c64,
        lda: *const Int,
        ipiv: *const Int,
        b: *mut c64,
        ldb: *const Int,
        info: *mut Int,
    );
    #[link_name = "zgeqrf$NEWLAPACK"]
    fn zgeqrf(
        m: *const Int,
        n: *const Int,
        a: *mut c64,
        lda: *const Int,
        tau: *mut c64,
        work: *mut c64,
        lwork: *const Int,
        info: *mut Int,
    );
    #[link_name = "zungqr$NEWLAPACK"]
    fn zungqr(
        m: *const Int,
        n: *const Int,
        k: *const Int,
        a: *mut c64,
        lda: *const Int,
        tau: *const c64,
        work: *mut c64,
        lwork: *const Int,
        info: *mut Int,
    );
    #[link_name = "zgesdd$NEWLAPACK"]
    fn zgesdd(
        jobz: *const u8,
        m: *const Int,
        n: *const Int,
        a: *mut c64,
        lda: *const Int,
        s: *mut f64,
        u: *mut c64,
        ldu: *const Int,
        vt: *mut c64,
        ldvt: *const Int,
        work: *mut c64,
        lwork: *const Int,
        rwork: *mut f64,
        iwork: *mut Int,
        info: *mut Int,
    );
    #[link_name = "zheevd$NEWLAPACK"]
    fn zheevd(
        jobz: *const u8,
        uplo: *const u8,
        n: *const Int,
        a: *mut c64,
        lda: *const Int,
        w: *mut f64,
        work: *mut c64,
        lwork: *const Int,
        rwork: *mut f64,
        lrwork: *const Int,
        iwork: *mut Int,
        liwork: *const Int,
        info: *mut Int,
    );
    #[link_name = "zgemm$NEWLAPACK"]
    fn zgemm(
        transa: *const u8,
        transb: *const u8,
        m: *const Int,
        n: *const Int,
        k: *const Int,
        alpha: *const c64,
        a: *const c64,
        lda: *const Int,
        b: *const c64,
        ldb: *const Int,
        beta: *const c64,
        c: *mut c64,
        ldc: *const Int,
    );
}

const ZERO: c64 = c64::new(0.0, 0.0);

/// Leading dimension of a column-major view, if BLAS can take it.
fn leading_dimension(rows: usize, cols: usize, row_stride: isize, col_stride: isize) -> Option<Int> {
    let min = rows.max(1);
    if cols <= 1 && (row_stride == 1 || rows <= 1) {
        return Int::try_from(min).ok(); // a single column: the leading dimension is never used
    }
    if row_stride != 1 && rows > 1 {
        return None;
    }
    let ld = usize::try_from(col_stride).ok().filter(|&ld| ld >= min)?;
    Int::try_from(ld).ok()
}

/// Hessenberg reduction (zgehrd), its unitary factor (zunghr), then the multishift QR algorithm
/// with aggressive early deflation (zhseqr).
pub(crate) fn schur(a: MatRef<'_, c64>) -> Result<(Mat<c64>, Mat<c64>), DenseError> {
    let n = a.nrows();
    let mut t = a.to_owned();
    if n <= 1 {
        return Ok((Mat::identity(n, n), t));
    }
    if !t.norm_max().is_finite() {
        return Err(DenseError::Schur);
    }
    let ni = Int::try_from(n).map_err(|_| DenseError::Schur)?;
    let one: Int = 1;
    let ld = Int::try_from(t.col_stride()).map_err(|_| DenseError::Schur)?;
    let mut q = Mat::<c64>::zeros(n, n);
    let ldq = Int::try_from(q.col_stride()).map_err(|_| DenseError::Schur)?;
    let mut tau = vec![ZERO; n - 1];
    let mut w = vec![ZERO; n];
    let mut info: Int = 0;

    // Workspace: the largest of the three queries.
    let query: Int = -1;
    let mut sizes = [ZERO; 3];
    // SAFETY: valid pointers and dimensions; lwork = −1 only writes the optimal size.
    unsafe {
        zgehrd(&ni, &one, &ni, t.as_ptr_mut(), &ld, tau.as_mut_ptr(), &mut sizes[0], &query, &mut info);
        zunghr(&ni, &one, &ni, q.as_ptr_mut(), &ldq, tau.as_ptr(), &mut sizes[1], &query, &mut info);
        zhseqr(
            b"S".as_ptr(),
            b"V".as_ptr(),
            &ni,
            &one,
            &ni,
            t.as_ptr_mut(),
            &ld,
            w.as_mut_ptr(),
            q.as_ptr_mut(),
            &ldq,
            &mut sizes[2],
            &query,
            &mut info,
        );
    }
    let lwork = sizes.iter().map(|s| s.re as usize).max().unwrap_or(0).max(n);
    let mut work = vec![ZERO; lwork];
    let lw = Int::try_from(lwork).map_err(|_| DenseError::Schur)?;

    // SAFETY: as above, with a workspace of the queried size.
    unsafe {
        zgehrd(&ni, &one, &ni, t.as_ptr_mut(), &ld, tau.as_mut_ptr(), work.as_mut_ptr(), &lw, &mut info);
    }
    if info != 0 {
        return Err(DenseError::Schur);
    }
    // The reflectors are below the subdiagonal of t: copy them to q, then clear them.
    q.copy_from(&t);
    for j in 0..n {
        for i in j + 2..n {
            t[(i, j)] = ZERO;
        }
    }
    // SAFETY: as above.
    unsafe {
        zunghr(&ni, &one, &ni, q.as_ptr_mut(), &ldq, tau.as_ptr(), work.as_mut_ptr(), &lw, &mut info);
    }
    if info != 0 {
        return Err(DenseError::Schur);
    }
    // SAFETY: as above.
    unsafe {
        zhseqr(
            b"S".as_ptr(),
            b"V".as_ptr(),
            &ni,
            &one,
            &ni,
            t.as_ptr_mut(),
            &ld,
            w.as_mut_ptr(),
            q.as_ptr_mut(),
            &ldq,
            work.as_mut_ptr(),
            &lw,
            &mut info,
        );
    }
    if info != 0 {
        return Err(DenseError::Schur);
    }
    for j in 0..n {
        for i in j + 1..n {
            t[(i, j)] = ZERO;
        }
    }
    Ok((q, t))
}

/// zgemm, or the destination back when a stride does not fit BLAS.
pub(crate) fn matmul<'a>(
    dst: MatMut<'a, c64>,
    accum: Accum,
    a: MatRef<'_, c64>,
    op_a: Op,
    b: MatRef<'_, c64>,
    op_b: Op,
    alpha: c64,
) -> Result<(), MatMut<'a, c64>> {
    let (m, n) = (dst.nrows(), dst.ncols());
    let k = if op_a == Op::Plain { a.ncols() } else { a.nrows() };
    if m == 0 || n == 0 {
        return Ok(());
    }
    let (Some(lda), Some(ldb), Some(ldc)) = (
        leading_dimension(a.nrows(), a.ncols(), a.row_stride(), a.col_stride()),
        leading_dimension(b.nrows(), b.ncols(), b.row_stride(), b.col_stride()),
        leading_dimension(m, n, dst.row_stride(), dst.col_stride()),
    ) else {
        return Err(dst);
    };
    let (Ok(mi), Ok(ni), Ok(ki)) = (Int::try_from(m), Int::try_from(n), Int::try_from(k)) else {
        return Err(dst);
    };
    if k == 0 {
        if accum == Accum::Replace {
            let mut dst = dst;
            dst.fill(ZERO);
        }
        return Ok(());
    }
    let code = |op: Op| match op {
        Op::Plain => b'N',
        Op::Transpose => b'T',
        Op::Adjoint => b'C',
    };
    let (ta, tb) = (code(op_a), code(op_b));
    let beta = if accum == Accum::Add { c64::new(1.0, 0.0) } else { ZERO };
    // SAFETY: column-major views with checked leading dimensions; with β = 0, BLAS does not read
    // the destination.
    unsafe {
        zgemm(
            &ta,
            &tb,
            &mi,
            &ni,
            &ki,
            &alpha,
            a.as_ptr(),
            &lda,
            b.as_ptr(),
            &ldb,
            &beta,
            dst.as_ptr_mut(),
            &ldc,
        );
    }
    Ok(())
}

/// LU factorisation with partial pivoting (zgetrf): the factors in place, LAPACK pivots.
#[derive(Clone, Debug)]
pub(crate) struct Lu {
    factors: Mat<c64>,
    pivots: Vec<Int>,
}

impl Lu {
    pub(crate) fn new(a: MatRef<'_, c64>) -> Self {
        let n = a.nrows();
        let mut factors = a.to_owned();
        let mut pivots: Vec<Int> = (1..=n as Int).collect();
        if n > 0 {
            let ni = Int::try_from(n).expect("matrix too large for LAPACK");
            let ld = Int::try_from(factors.col_stride()).expect("matrix too large for LAPACK");
            let mut info: Int = 0;
            // SAFETY: a square column-major matrix and n pivots. info > 0 flags an exactly zero
            // pivot; the factorisation is complete and the caller sees it in the pivots.
            unsafe { zgetrf(&ni, &ni, factors.as_ptr_mut(), &ld, pivots.as_mut_ptr(), &mut info) };
        }
        Self { factors, pivots }
    }

    pub(crate) fn diagonal(&self) -> impl Iterator<Item = c64> + '_ {
        (0..self.factors.nrows()).map(|i| self.factors[(i, i)])
    }

    /// `b ← a⁻¹ b`, or `b ← a⁻† b` (zgetrs), or `b` back when its stride does not fit LAPACK.
    pub(crate) fn solve_in_place<'a>(
        &self,
        b: MatMut<'a, c64>,
        adjoint: bool,
    ) -> Result<(), MatMut<'a, c64>> {
        let (n, nrhs) = (b.nrows(), b.ncols());
        if n == 0 || nrhs == 0 {
            return Ok(());
        }
        let Some(ldb) = leading_dimension(n, nrhs, b.row_stride(), b.col_stride()) else {
            return Err(b);
        };
        let ni = Int::try_from(n).expect("matrix too large for LAPACK");
        let nr = Int::try_from(nrhs).expect("matrix too large for LAPACK");
        let lda = Int::try_from(self.factors.col_stride()).expect("matrix too large for LAPACK");
        let mut info: Int = 0;
        // SAFETY: the factors of an n × n matrix and a column-major n × nrhs right-hand side.
        unsafe {
            zgetrs(
                if adjoint { b"C".as_ptr() } else { b"N".as_ptr() },
                &ni,
                &nr,
                self.factors.as_ptr(),
                &lda,
                self.pivots.as_ptr(),
                b.as_ptr_mut(),
                &ldb,
                &mut info,
            )
        };
        Ok(())
    }
}

/// Thin `Q` of the QR factorisation of an `m × n` matrix, `m ≥ n` (zgeqrf, zungqr).
pub(crate) fn orthonormal_columns(a: MatRef<'_, c64>) -> Mat<c64> {
    let (m, n) = (a.nrows(), a.ncols());
    let mut q = a.to_owned();
    if n == 0 {
        return q;
    }
    let (mi, ni) = (Int::try_from(m).expect("matrix too large"), Int::try_from(n).expect("matrix too large"));
    let ld = Int::try_from(q.col_stride()).expect("matrix too large");
    let mut tau = vec![ZERO; n];
    let mut info: Int = 0;
    let query: Int = -1;
    let mut sizes = [ZERO; 2];
    // SAFETY: valid pointers and dimensions; lwork = −1 only writes the optimal size.
    unsafe {
        zgeqrf(&mi, &ni, q.as_ptr_mut(), &ld, tau.as_mut_ptr(), &mut sizes[0], &query, &mut info);
        zungqr(&mi, &ni, &ni, q.as_ptr_mut(), &ld, tau.as_ptr(), &mut sizes[1], &query, &mut info);
    }
    let lwork = sizes.iter().map(|s| s.re as usize).max().unwrap_or(0).max(n);
    let mut work = vec![ZERO; lwork];
    let lw = Int::try_from(lwork).expect("workspace too large");
    // SAFETY: as above, with a workspace of the queried size.
    unsafe {
        zgeqrf(&mi, &ni, q.as_ptr_mut(), &ld, tau.as_mut_ptr(), work.as_mut_ptr(), &lw, &mut info);
        zungqr(&mi, &ni, &ni, q.as_ptr_mut(), &ld, tau.as_ptr(), work.as_mut_ptr(), &lw, &mut info);
    }
    q
}

/// Thin singular value decomposition `a = u diag(s) v†` (zgesdd).
pub(crate) fn svd(a: MatRef<'_, c64>) -> Result<Svd, DenseError> {
    let (m, n) = (a.nrows(), a.ncols());
    let k = m.min(n);
    if k == 0 {
        return Ok(Svd { u: Mat::zeros(m, 0), s: Vec::new(), v: Mat::zeros(n, 0) });
    }
    let mut a = a.to_owned();
    let (mut u, mut vt) = (Mat::<c64>::zeros(m, k), Mat::<c64>::zeros(k, n));
    let mut s = vec![0.0; k];
    let int = |x: usize| Int::try_from(x).map_err(|_| DenseError::Svd);
    let (mi, ni) = (int(m)?, int(n)?);
    let (lda, ldu, ldvt) =
        (int(a.col_stride() as usize)?, int(u.col_stride() as usize)?, int(vt.col_stride() as usize)?);
    let mut rwork = vec![0.0; (5 * k * k + 5 * k).max(2 * m.max(n) * k + 2 * k * k + k)];
    let mut iwork = vec![0 as Int; 8 * k];
    let mut info: Int = 0;
    let query: Int = -1;
    let mut size = ZERO;
    // SAFETY: valid pointers and dimensions; lwork = −1 only writes the optimal size.
    unsafe {
        zgesdd(
            b"S".as_ptr(),
            &mi,
            &ni,
            a.as_ptr_mut(),
            &lda,
            s.as_mut_ptr(),
            u.as_ptr_mut(),
            &ldu,
            vt.as_ptr_mut(),
            &ldvt,
            &mut size,
            &query,
            rwork.as_mut_ptr(),
            iwork.as_mut_ptr(),
            &mut info,
        );
    }
    let lwork = (size.re as usize).max(1);
    let mut work = vec![ZERO; lwork];
    let lw = int(lwork)?;
    // SAFETY: as above, with a workspace of the queried size.
    unsafe {
        zgesdd(
            b"S".as_ptr(),
            &mi,
            &ni,
            a.as_ptr_mut(),
            &lda,
            s.as_mut_ptr(),
            u.as_ptr_mut(),
            &ldu,
            vt.as_ptr_mut(),
            &ldvt,
            work.as_mut_ptr(),
            &lw,
            rwork.as_mut_ptr(),
            iwork.as_mut_ptr(),
            &mut info,
        );
    }
    if info != 0 {
        return Err(DenseError::Svd);
    }
    Ok(Svd { u, s, v: vt.adjoint().to_owned() })
}

/// Eigenvalues and eigenvectors of a Hermitian matrix, from its lower triangle (zheevd, divide
/// and conquer).
pub(crate) fn hermitian_eigen(a: MatRef<'_, c64>) -> Result<(Vec<f64>, Mat<c64>), DenseError> {
    let n = a.nrows();
    let mut v = a.to_owned();
    let mut w = vec![0.0; n];
    if n == 0 {
        return Ok((w, v));
    }
    let ni = Int::try_from(n).map_err(|_| DenseError::Eigen)?;
    let ld = Int::try_from(v.col_stride()).map_err(|_| DenseError::Eigen)?;
    let mut info: Int = 0;
    let query: Int = -1;
    let (mut size, mut rsize, mut isize) = (ZERO, 0.0, 0 as Int);
    // SAFETY: valid pointers and dimensions; the workspace queries only write the optimal sizes.
    unsafe {
        zheevd(
            b"V".as_ptr(),
            b"L".as_ptr(),
            &ni,
            v.as_ptr_mut(),
            &ld,
            w.as_mut_ptr(),
            &mut size,
            &query,
            &mut rsize,
            &query,
            &mut isize,
            &query,
            &mut info,
        );
    }
    let lwork = (size.re as usize).max(1);
    let lrwork = (rsize as usize).max(1);
    let liwork = (isize as usize).max(1);
    let mut work = vec![ZERO; lwork];
    let mut rwork = vec![0.0; lrwork];
    let mut iwork = vec![0 as Int; liwork];
    let int = |x: usize| Int::try_from(x).map_err(|_| DenseError::Eigen);
    let (lw, lrw, liw) = (int(lwork)?, int(lrwork)?, int(liwork)?);
    // SAFETY: as above, with workspaces of the queried sizes.
    unsafe {
        zheevd(
            b"V".as_ptr(),
            b"L".as_ptr(),
            &ni,
            v.as_ptr_mut(),
            &ld,
            w.as_mut_ptr(),
            work.as_mut_ptr(),
            &lw,
            rwork.as_mut_ptr(),
            &lrw,
            iwork.as_mut_ptr(),
            &liw,
            &mut info,
        );
    }
    if info != 0 {
        return Err(DenseError::Eigen);
    }
    Ok((w, v))
}
