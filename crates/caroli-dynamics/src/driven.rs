//! Time evolution of any region of a [`Model`] under a uniform vector potential `A(t)`.
//!
//! Each hopping block of the Hamiltonian picks up the Peierls factor `exp(−i A(t)·d)`, `d` being
//! the displacement of its bond ([`Model::bonds`]), so that a constant force `F` (an electric
//! field, the charge absorbed) is `A(t) = −F t` and the crystal momentum moves as `k + F t`. The
//! steps are those of the fourth-order commutator-free Magnus integrator ([`magnus4_step`]): any
//! geometry, open or periodic, with any disorder, magnetic field or hopping term of the model.

use crate::{TimeDependent, magnus4_step};
use caroli_lattice::{Cell, Region, Vec3};
use caroli_model::{Model, SparseMatrix, c64};
use rayon::prelude::*;
use std::sync::Arc;

/// A uniform vector potential `A(t)` (cartesian, in the inverse length unit of the lattice).
pub type VectorPotential = Arc<dyn Fn(f64) -> Vec3 + Send + Sync>;

/// The vector potential of a constant force `F` switched on at `t = 0`: `A(t) = −F t`.
pub fn constant_force(force: Vec3) -> VectorPotential {
    Arc::new(move |t: f64| [-force[0] * t, -force[1] * t, -force[2] * t])
}

/// Rows above which the products run in parallel.
const PARALLEL_ROWS: usize = 1 << 14;

/// The Hamiltonian `H(t)` of a region under a uniform vector potential, as a sparse matrix whose
/// entries carry the displacement class of their bond.
pub struct Driven {
    n: usize,
    row_ptr: Vec<usize>,
    cols: Vec<u32>,
    vals: Vec<c64>,
    /// Displacement class of each entry; class 0 is the on-site one.
    class: Vec<u16>,
    displacements: Vec<Vec3>,
    bounds: (f64, f64),
    potential: VectorPotential,
}

impl Driven {
    /// The region repeated along `periods` (open if empty), without field.
    pub fn new(model: &Model, region: &Region, periods: &[Cell]) -> Self {
        let mut displacements: Vec<Vec3> = vec![[0.0; 3]];
        let mut keys: std::collections::HashMap<[i64; 3], u16> = [([0; 3], 0u16)].into();
        let key = |d: Vec3| d.map(|x| (x * 1e9).round() as i64);
        let mut entries: Vec<(usize, usize, u16, c64)> = Vec::new();
        for (i, site) in region.sites().iter().enumerate() {
            let m = model.onsite_matrix(site);
            let o = region.offset(i);
            for a in 0..m.nrows() {
                for b in 0..m.ncols() {
                    if m[(a, b)] != c64::new(0.0, 0.0) {
                        entries.push((o + a, o + b, 0, m[(a, b)]));
                    }
                }
            }
        }
        for bond in model.bonds(region, periods) {
            let class = *keys.entry(key(bond.displacement)).or_insert_with(|| {
                displacements.push(bond.displacement);
                (displacements.len() - 1) as u16
            });
            let (r, c) = (region.offset(bond.from), region.offset(bond.to));
            for a in 0..bond.matrix.nrows() {
                for b in 0..bond.matrix.ncols() {
                    if bond.matrix[(a, b)] != c64::new(0.0, 0.0) {
                        entries.push((r + a, c + b, class, bond.matrix[(a, b)]));
                    }
                }
            }
        }
        entries.sort_unstable_by_key(|&(r, c, k, _)| (r, c, k));
        let n = region.orbitals();
        let mut row_ptr = vec![0usize; n + 1];
        let (mut cols, mut vals, mut class) = (Vec::new(), Vec::new(), Vec::new());
        let mut last = None;
        for (r, c, k, v) in entries {
            if last == Some((r, c, k)) {
                *vals.last_mut().expect("entry") += v;
            } else {
                cols.push(c as u32);
                vals.push(v);
                class.push(k);
                row_ptr[r + 1] += 1;
                last = Some((r, c, k));
            }
        }
        for r in 0..n {
            row_ptr[r + 1] += row_ptr[r];
        }
        // Gershgorin bounds: the Peierls factors leave the moduli unchanged.
        let mut bounds = (f64::INFINITY, f64::NEG_INFINITY);
        for r in 0..n {
            let (mut centre, mut radius) = (0.0, 0.0);
            for k in row_ptr[r]..row_ptr[r + 1] {
                if cols[k] as usize == r && class[k] == 0 {
                    centre += vals[k].re
                } else {
                    radius += vals[k].norm()
                }
            }
            bounds = (bounds.0.min(centre - radius), bounds.1.max(centre + radius));
        }
        Self { n, row_ptr, cols, vals, class, displacements, bounds, potential: Arc::new(|_| [0.0; 3]) }
    }

    /// A constant force `F` from `t = 0` (see [`constant_force`]).
    pub fn with_force(self, force: Vec3) -> Self {
        self.with_vector_potential(constant_force(force))
    }

    /// Any uniform vector potential `A(t)`.
    pub fn with_vector_potential(mut self, potential: VectorPotential) -> Self {
        self.potential = potential;
        self
    }

    /// Dimension of the wave functions.
    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// `Σ_i w_i exp(−i A(t_i)·d_c)` for each displacement class.
    fn phases(&self, stages: &[(f64, f64)]) -> Vec<c64> {
        let potentials: Vec<(Vec3, f64)> = stages.iter().map(|&(t, w)| ((self.potential)(t), w)).collect();
        self.displacements
            .iter()
            .map(|d| {
                potentials
                    .iter()
                    .map(|(a, w)| c64::from_polar(*w, -(a[0] * d[0] + a[1] * d[1] + a[2] * d[2])))
                    .sum()
            })
            .collect()
    }

    /// `y = Σ_r M_r x` over the rows, with `M = Σ_entries v · table[class]`.
    fn product(&self, table: &[c64], x: &[c64], y: &mut [c64]) {
        assert_eq!((x.len(), y.len()), (self.n, self.n));
        let row = |(r, out): (usize, &mut c64)| {
            let mut acc = c64::new(0.0, 0.0);
            for k in self.row_ptr[r]..self.row_ptr[r + 1] {
                acc += self.vals[k] * table[self.class[k] as usize] * x[self.cols[k] as usize];
            }
            *out = acc;
        };
        if self.n >= PARALLEL_ROWS {
            y.par_iter_mut().enumerate().with_min_len(4096).for_each(row);
        } else {
            y.iter_mut().enumerate().for_each(row);
        }
    }

    /// `H(t)` as a sparse matrix.
    pub fn hamiltonian(&self, t: f64) -> SparseMatrix {
        let table = &self.phases(&[(t, 1.0)]);
        let entries = (0..self.n)
            .flat_map(|r| {
                (self.row_ptr[r]..self.row_ptr[r + 1])
                    .map(move |k| (r, self.cols[k] as usize, self.vals[k] * table[self.class[k] as usize]))
            })
            .collect();
        SparseMatrix::from_triplets(self.n, entries)
    }

    /// `⟨ψ| v |ψ⟩` along `direction` at time `t`, with the velocity `v = i[H(t), r]`
    /// (`v_ij = i H_ij(t) d_ij·direction`).
    pub fn velocity(&self, psi: &[c64], t: f64, direction: Vec3) -> f64 {
        let table = self.phases(&[(t, 1.0)]);
        let along: Vec<f64> = self
            .displacements
            .iter()
            .map(|d| d[0] * direction[0] + d[1] * direction[1] + d[2] * direction[2])
            .collect();
        (0..self.n)
            .into_par_iter()
            .with_min_len(4096)
            .map(|r| {
                let mut acc = c64::new(0.0, 0.0);
                for k in self.row_ptr[r]..self.row_ptr[r + 1] {
                    let c = self.class[k] as usize;
                    acc += self.vals[k] * table[c] * along[c] * psi[self.cols[k] as usize];
                }
                (psi[r].conj() * c64::new(0.0, 1.0) * acc).re
            })
            .sum()
    }

    /// One Magnus step `ψ(t) → ψ(t + dt)`, the exponentials exact to `tol`.
    pub fn step(&self, psi: &mut [c64], t: f64, dt: f64, tol: f64) {
        magnus4_step(self, psi, t, dt, tol);
    }
}

impl TimeDependent for Driven {
    fn apply(&self, stages: &[(f64, f64)], x: &[c64], y: &mut [c64]) {
        self.product(&self.phases(stages), x, y);
    }

    fn bounds(&self) -> (f64, f64) {
        self.bounds
    }
}
