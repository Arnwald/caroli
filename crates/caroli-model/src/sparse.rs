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

    /// `Y = A X` for a block of `m` vectors stored interleaved (`x[row·m + v]`): the matrix is
    /// read once for the whole block.
    pub fn apply_block(&self, x: &[c64], y: &mut [c64], m: usize) {
        assert_eq!((x.len(), y.len()), (self.n * m, self.n * m));
        let row = |(r, out): (usize, &mut [c64])| {
            out.iter_mut().for_each(|o| *o = c64::new(0.0, 0.0));
            for k in self.row_ptr[r]..self.row_ptr[r + 1] {
                let (c, v) = (self.cols[k] as usize, self.vals[k]);
                for (o, xv) in out.iter_mut().zip(&x[c * m..(c + 1) * m]) {
                    *o += v * xv;
                }
            }
        };
        if self.n >= PARALLEL_ROWS {
            y.par_chunks_exact_mut(m).enumerate().with_min_len(4096).for_each(row);
        } else {
            y.chunks_exact_mut(m).enumerate().for_each(row);
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

    /// A one-cell supercell of a chain with first- and second-neighbour hoppings:
    /// `H(k) = 2 t1 cos k + 2 t2 cos 2k` (the second neighbour lies two periods away).
    #[test]
    fn long_hoppings_reach_far_images() {
        use crate::{CMat, Model, c64};
        use caroli_lattice::Lattice;
        let one = |x: f64| CMat::from_fn(1, 1, |_, _| c64::new(x, 0.0));
        let (t1, t2, k) = (-1.0, 0.3, 0.7);
        let model = Model::new(Lattice::square(1.0, 1)).hopping(0, 0, [1, 0, 0], one(t1)).hopping(
            0,
            0,
            [2, 0, 0],
            one(t2),
        );
        let cell = Region::cuboid(model.lattice(), [1, 1, 0]);
        let h = model.bloch_hamiltonian(&cell, &[[1, 0, 0]], &[k]);
        let exact = 2.0 * t1 * k.cos() + 2.0 * t2 * (2.0 * k).cos();
        assert!((h[(0, 0)].re - exact).abs() < 1e-14, "{} vs {exact}", h[(0, 0)].re);
        let s = model.sparse_bloch_hamiltonian(&cell, &[[1, 0, 0]], &[k]);
        assert!((s.to_dense()[(0, 0)].re - exact).abs() < 1e-14);
    }

    /// Sheared periods: one cell with periods [1,0,0] and [1,1,0]; the hopping [1,−1,0] is
    /// 2 T₁ − T₂. `H(k)` at Bloch phases (φ₁, φ₂) is `Σ_offsets t e^{i n·φ}` over both directions.
    #[test]
    fn sheared_periods() {
        use crate::{CMat, Model, c64};
        use caroli_lattice::Lattice;
        let one = |x: f64| CMat::from_fn(1, 1, |_, _| c64::new(x, 0.0));
        let (t, tp) = (-1.0, 0.3);
        let model = Model::new(Lattice::square(1.0, 1))
            .hopping(0, 0, [1, 0, 0], one(t))
            .hopping(0, 0, [0, 1, 0], one(t))
            .hopping(0, 0, [1, 1, 0], one(tp))
            .hopping(0, 0, [1, -1, 0], one(tp));
        let cell = Region::cuboid(model.lattice(), [1, 1, 0]);
        let (p1, p2) = (0.4, -0.9);
        let h = model.bloch_hamiltonian(&cell, &[[1, 0, 0], [1, 1, 0]], &[p1, p2]);
        // Offsets in the period basis: [1,0]=T1, [0,1]=T2−T1, [1,1]=T2, [1,−1]=2T1−T2.
        let phase = |n1: f64, n2: f64| 2.0 * (n1 * p1 + n2 * p2).cos();
        let exact = t * phase(1.0, 0.0) + t * phase(-1.0, 1.0) + tp * phase(0.0, 1.0) + tp * phase(2.0, -1.0);
        assert!((h[(0, 0)].re - exact).abs() < 1e-14, "{} vs {exact}", h[(0, 0)].re);
    }

    /// Two fundamental domains of the same periodic system give the same spectrum: a slanted
    /// square supercell {(j, j)} against the straight one {(0, j)}, and a graphene armchair cell
    /// with half of its sites moved by one period; the Hamiltonians stay Hermitian.
    #[test]
    fn any_fundamental_domain() {
        use crate::{CMat, Model, c64};
        use caroli_lattice::{Lattice, Site};
        let one = |x: f64| CMat::from_fn(1, 1, |_, _| c64::new(x, 0.0));
        let spectrum = |h: &CMat| h.self_adjoint_eigenvalues(faer::Side::Lower).unwrap();
        let close = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12);
        let sq = Model::new(Lattice::square(1.0, 1)).hopping(0, 0, [1, 0, 0], one(-1.0)).hopping(
            0,
            0,
            [0, 1, 0],
            one(-1.0),
        );
        let site = |x: i64, y: i64, sub: usize| Site { cell: [x, y, 0], sub };
        let straight = Region::from_sites(sq.lattice(), (0..6).map(|j| site(0, j, 0)).collect());
        let slanted = Region::from_sites(sq.lattice(), (0..6).map(|j| site(j, j, 0)).collect());
        for k in [0.0, 0.4, 2.1] {
            let (a, b) = (
                sq.bloch_hamiltonian(&straight, &[[1, 0, 0]], &[k]),
                sq.bloch_hamiltonian(&slanted, &[[1, 0, 0]], &[k]),
            );
            assert!((&b - b.adjoint()).norm_max() < 1e-15);
            assert!(close(&spectrum(&a), &spectrum(&b)), "square, k = {k}");
        }
        let g = Model::new(Lattice::honeycomb(1.0, 1))
            .hopping(0, 1, [0, 0, 0], one(-1.0))
            .hopping(0, 1, [0, -1, 0], one(-1.0))
            .hopping(0, 1, [1, -1, 0], one(-1.0));
        let w = 30;
        let cell: Vec<Site> =
            (0..w).flat_map(|i| (0..2).flat_map(move |j| (0..2).map(move |s| site(i, j, s)))).collect();
        let moved: Vec<Site> =
            cell.iter()
                .enumerate()
                .map(|(k, s)| {
                    if k % 2 == 0 { Site { cell: [s.cell[0] - 1, s.cell[1] + 2, 0], sub: s.sub } } else { *s }
                })
                .collect();
        let (r1, r2) = (Region::from_sites(g.lattice(), cell), Region::from_sites(g.lattice(), moved));
        for k in [0.0, 0.7] {
            let (a, b) = (
                g.bloch_hamiltonian(&r1, &[[-1, 2, 0]], &[k]),
                g.bloch_hamiltonian(&r2, &[[-1, 2, 0]], &[k]),
            );
            assert!((&a - a.adjoint()).norm_max() < 1e-15 && (&b - b.adjoint()).norm_max() < 1e-15);
            assert!(close(&spectrum(&a), &spectrum(&b)), "graphene, k = {k}");
        }
    }

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
        // Block product: two interleaved vectors.
        let x2: Vec<_> = (0..2 * bloch.n).map(|i| super::c64::new((0.1 * i as f64).cos(), 1.0)).collect();
        let mut y2 = vec![super::c64::new(0.0, 0.0); 2 * bloch.n];
        bloch.apply_block(&x2, &mut y2, 2);
        for v in 0..2 {
            let xv: Vec<_> = x2.iter().skip(v).step_by(2).copied().collect();
            let mut yv = vec![super::c64::new(0.0, 0.0); bloch.n];
            bloch.apply(&xv, &mut yv);
            assert!(yv.iter().zip(y2.iter().skip(v).step_by(2)).all(|(a, b)| (a - b).norm() < 1e-13));
        }
        // Gershgorin bounds contain the spectrum.
        let (lo, hi) = bloch.spectral_bounds();
        let e = dense.self_adjoint_eigenvalues(faer::Side::Lower).unwrap();
        assert!(lo <= e[0] && e[e.len() - 1] <= hi);
    }
}
