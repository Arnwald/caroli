//! Compressed sparse row (CSR) matrices for Hamiltonians of large regions: the operator of the
//! kernel polynomial method and of Chebyshev time evolution on any geometry, and the format the
//! GPU backends upload.

use crate::{CMat, c64};
use rayon::prelude::*;

/// Rows above which [`SparseMatrix::apply`] runs in parallel.
const PARALLEL_ROWS: usize = 1 << 14;

/// Square complex matrix in CSR form; column indices are `u32` (as on the GPU).
#[derive(Clone, Debug, PartialEq)]
pub struct SparseMatrix {
    pub n: usize,
    /// Entries of row `r` are `row_ptr[r]..row_ptr[r + 1]`.
    pub row_ptr: Vec<usize>,
    pub cols: Vec<u32>,
    pub vals: Vec<c64>,
}

impl SparseMatrix {
    /// From `(row, column, value)` entries; duplicates are summed, columns sorted in each row.
    pub fn from_triplets(n: usize, mut entries: Vec<(usize, usize, c64)>) -> Self {
        assert!(n <= u32::MAX as usize, "dimension {n} exceeds the u32 column index");
        entries.sort_unstable_by_key(|&(r, c, _)| (r, c));
        let mut row_ptr = vec![0usize; n + 1];
        let (mut cols, mut vals) = (Vec::with_capacity(entries.len()), Vec::with_capacity(entries.len()));
        let mut last: Option<(usize, usize)> = None;
        for (r, c, v) in entries {
            assert!(r < n && c < n, "entry ({r}, {c}) outside a {n}×{n} matrix");
            if last == Some((r, c)) {
                *vals.last_mut().unwrap() += v;
            } else {
                cols.push(c as u32);
                vals.push(v);
                row_ptr[r + 1] += 1;
                last = Some((r, c));
            }
        }
        for r in 0..n {
            row_ptr[r + 1] += row_ptr[r];
        }
        Self { n, row_ptr, cols, vals }
    }

    pub fn nnz(&self) -> usize {
        self.vals.len()
    }

    /// Entries of row `r` as `(column, value)`.
    pub fn row(&self, r: usize) -> impl Iterator<Item = (usize, c64)> + '_ {
        (self.row_ptr[r]..self.row_ptr[r + 1]).map(|k| (self.cols[k] as usize, self.vals[k]))
    }

    /// `y = A x` (rows in parallel for large matrices).
    pub fn apply(&self, x: &[c64], y: &mut [c64]) {
        assert_eq!((x.len(), y.len()), (self.n, self.n));
        let row = |(r, out): (usize, &mut c64)| {
            let mut acc = c64::new(0.0, 0.0);
            for k in self.row_ptr[r]..self.row_ptr[r + 1] {
                acc += self.vals[k] * x[self.cols[k] as usize];
            }
            *out = acc;
        };
        if self.n >= PARALLEL_ROWS {
            y.par_iter_mut().enumerate().with_min_len(4096).for_each(row);
        } else {
            y.iter_mut().enumerate().for_each(row);
        }
    }

    pub fn to_dense(&self) -> CMat {
        let mut m = CMat::zeros(self.n, self.n);
        for r in 0..self.n {
            for (c, v) in self.row(r) {
                m[(r, c)] += v;
            }
        }
        m
    }

    /// Gershgorin bounds `(lower, upper)` on the spectrum of a Hermitian matrix.
    pub fn spectral_bounds(&self) -> (f64, f64) {
        let mut bounds = (f64::INFINITY, f64::NEG_INFINITY);
        for r in 0..self.n {
            let (mut centre, mut radius) = (0.0, 0.0);
            for (c, v) in self.row(r) {
                if c == r { centre += v.re } else { radius += v.norm() }
            }
            bounds = (bounds.0.min(centre - radius), bounds.1.max(centre + radius));
        }
        bounds
    }

    /// `(A − centre·1) / half_width`, e.g. to map a spectrum into (−1, 1) for Chebyshev
    /// expansions.
    pub fn rescaled(&self, centre: f64, half_width: f64) -> Self {
        let mut entries: Vec<(usize, usize, c64)> =
            (0..self.n).flat_map(|r| self.row(r).map(move |(c, v)| (r, c, v / half_width))).collect();
        entries.extend((0..self.n).map(|r| (r, r, c64::new(-centre / half_width, 0.0))));
        Self::from_triplets(self.n, entries)
    }
}

#[cfg(test)]
mod tests {
    use crate::presets::{BhzParams, bhz};
    use caroli_lattice::Region;

    /// The sparse builders give the dense Hamiltonians, open and twisted-periodic.
    #[test]
    fn sparse_matches_dense() {
        let model = bhz(BhzParams { lambda: 0.1, ..BhzParams::default() });
        let region = Region::cuboid(model.lattice(), [5, 4, 0]);
        let open = model.sparse_hamiltonian(&region);
        assert!((open.to_dense() - model.hamiltonian(&region)).norm_max() < 1e-15);
        let (periods, phases) = ([[5, 0, 0], [0, 4, 0]], [0.3, -1.1]);
        let bloch = model.sparse_bloch_hamiltonian(&region, &periods, &phases);
        assert!((bloch.to_dense() - model.bloch_hamiltonian(&region, &periods, &phases)).norm_max() < 1e-15);
        // y = A x against the dense product.
        let x: Vec<_> = (0..bloch.n).map(|i| super::c64::new((i as f64).sin(), 0.3 * i as f64)).collect();
        let mut y = vec![super::c64::new(0.0, 0.0); bloch.n];
        bloch.apply(&x, &mut y);
        let dense = bloch.to_dense();
        for (r, yr) in y.iter().enumerate() {
            let d: super::c64 = (0..bloch.n).map(|c| dense[(r, c)] * x[c]).sum();
            assert!((yr - d).norm() < 1e-12);
        }
        // Gershgorin bounds contain the spectrum.
        let (lo, hi) = bloch.spectral_bounds();
        let e = dense.self_adjoint_eigenvalues(faer::Side::Lower).unwrap();
        assert!(lo <= e[0] && e[e.len() - 1] <= hi);
    }
}
