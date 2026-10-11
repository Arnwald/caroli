//! Tight-binding models on a [`Lattice`] and the matrices they produce on a [`Region`].
//!
//! A [`Model`] holds an on-site matrix per sublattice and a list of [`Hopping`]s
//! `⟨(R, from)| H |(R + offset, to)⟩`. Extra on-site terms that depend on the site (disorder,
//! potentials) are added through [`OnSite`]. The model builds dense Hamiltonians of finite
//! regions, Bloch Hamiltonians of partially periodic systems and the blocks used by transport.

use caroli_lattice::{Cell, Lattice, Region, Site};
use faer::Mat;
pub use faer::c64;

pub mod gamma;
pub mod presets;
pub mod sparse;

pub use sparse::SparseMatrix;

/// Complex dense matrix (orbital blocks and Hamiltonians).
pub type CMat = Mat<c64>;

/// A hopping term: `⟨(R, from)| H |(R + offset, to)⟩ = matrix`. Its Hermitian conjugate (the
/// reverse hop) is added automatically.
#[derive(Clone, Debug)]
pub struct Hopping {
    pub from: usize,
    pub to: usize,
    pub offset: Cell,
    pub matrix: CMat,
}

/// Site-dependent on-site term, added to the sublattice on-site matrix.
pub trait OnSite: Send + Sync {
    /// Extra on-site matrix of `site`, or `None` for no contribution.
    fn matrix(&self, site: &Site) -> Option<CMat>;
}

impl<F: Fn(&Site) -> Option<CMat> + Send + Sync> OnSite for F {
    fn matrix(&self, site: &Site) -> Option<CMat> {
        self(site)
    }
}

/// Tight-binding model on a lattice.
pub struct Model {
    lattice: Lattice,
    onsite: Vec<CMat>,
    hoppings: Vec<Hopping>,
    extra: Vec<Box<dyn OnSite>>,
}

impl Model {
    /// Model with zero on-site matrices and no hopping.
    pub fn new(lattice: Lattice) -> Self {
        let onsite = lattice.sublattices().iter().map(|s| CMat::zeros(s.orbitals, s.orbitals)).collect();
        Self { lattice, onsite, hoppings: Vec::new(), extra: Vec::new() }
    }

    pub fn lattice(&self) -> &Lattice {
        &self.lattice
    }

    pub fn hoppings(&self) -> &[Hopping] {
        &self.hoppings
    }

    /// Sets the on-site matrix of a sublattice.
    pub fn onsite(mut self, sub: usize, matrix: CMat) -> Self {
        let n = self.lattice.sublattices()[sub].orbitals;
        assert_eq!((matrix.nrows(), matrix.ncols()), (n, n), "on-site matrix of sublattice {sub}");
        self.onsite[sub] = matrix;
        self
    }

    /// Adds a hopping `⟨(R, from)| H |(R + offset, to)⟩ = matrix` (and its conjugate).
    pub fn hopping(mut self, from: usize, to: usize, offset: Cell, matrix: CMat) -> Self {
        let subs = self.lattice.sublattices();
        assert_eq!((matrix.nrows(), matrix.ncols()), (subs[from].orbitals, subs[to].orbitals));
        assert!(offset != [0; 3] || from != to, "an on-site term is not a hopping");
        self.hoppings.push(Hopping { from, to, offset, matrix });
        self
    }

    /// Adds a site-dependent on-site term (disorder, external potential...).
    pub fn with_onsite_term(mut self, term: impl OnSite + 'static) -> Self {
        self.extra.push(Box::new(term));
        self
    }

    /// Full on-site matrix of a site (sublattice term plus site-dependent terms).
    pub fn onsite_matrix(&self, site: &Site) -> CMat {
        let mut m = self.onsite[site.sub].clone();
        for term in &self.extra {
            if let Some(v) = term.matrix(site) {
                m += v;
            }
        }
        m
    }

    /// Dense Hamiltonian of a finite region (open boundaries).
    pub fn hamiltonian(&self, region: &Region) -> CMat {
        let mut h = CMat::zeros(region.orbitals(), region.orbitals());
        self.build(&mut h, region, &[], &[]);
        h
    }

    /// Bloch Hamiltonian of a region repeated periodically: `periods` are translations (in
    /// cells) and `phases` the matching Bloch phases `k·T` in radians. Hoppings that leave the
    /// region by `n` periods pick up `exp(i Σ n·phase)`.
    pub fn bloch_hamiltonian(&self, region: &Region, periods: &[Cell], phases: &[f64]) -> CMat {
        let mut h = CMat::zeros(region.orbitals(), region.orbitals());
        self.build(&mut h, region, periods, phases);
        h
    }

    /// Sparse Hamiltonian of a finite region (open boundaries), for regions too large for
    /// dense matrices (KPM, time evolution).
    pub fn sparse_hamiltonian(&self, region: &Region) -> SparseMatrix {
        self.sparse_bloch_hamiltonian(region, &[], &[])
    }

    /// Sparse version of [`Self::bloch_hamiltonian`] (twisted periodic boundaries).
    pub fn sparse_bloch_hamiltonian(
        &self,
        region: &Region,
        periods: &[Cell],
        phases: &[f64],
    ) -> SparseMatrix {
        let mut t = Triplets(Vec::new());
        self.build(&mut t, region, periods, phases);
        SparseMatrix::from_triplets(region.orbitals(), t.0)
    }

    /// On-site terms and couplings, each hopping partner reduced into the region modulo the
    /// period lattice: a partner `p = q + Σ n_i T_i` with `q` in the region couples with the
    /// Bloch phase `exp(i Σ n_i phase_i)`. Partners with no image in the region (open
    /// directions) are dropped. Hoppings of any range and sheared periods are handled.
    fn build(&self, h: &mut impl Sink, region: &Region, periods: &[Cell], phases: &[f64]) {
        assert_eq!(periods.len(), phases.len());
        for i in 0..region.len() {
            h.block(
                region.offset(i),
                region.offset(i),
                &self.onsite_matrix(&region.sites()[i]),
                c64::new(1.0, 0.0),
                false,
            );
        }
        if periods.is_empty() {
            self.add_couplings_one_way(h, region, region, c64::new(1.0, 0.0));
            return;
        }
        let images = Images::new(region, periods);
        for (i, site) in region.sites().iter().enumerate() {
            for hop in &self.hoppings {
                // Forward: site is `from`, partner is `to` at +offset; backward: site is `to`,
                // partner is `from` at −offset, block matrix†.
                let ends =
                    [(hop.from == site.sub, 1, hop.to, false), (hop.to == site.sub, -1, hop.from, true)];
                for (applies, sign, sub, adjoint) in ends {
                    if !applies {
                        continue;
                    }
                    if let Some((j, n)) = images.locate(region, &shifted(site, hop.offset, sign, sub)) {
                        let phase: f64 = n.iter().zip(phases).map(|(n, k)| *n as f64 * k).sum();
                        let (s, cs) = phase.sin_cos();
                        h.block(region.offset(i), region.offset(j), &hop.matrix, c64::new(cs, s), adjoint);
                    }
                }
            }
        }
    }

    /// Coupling matrix `⟨a| H |b⟩` between two disjoint regions (rows: `a`, columns: `b`).
    pub fn coupling(&self, a: &Region, b: &Region) -> CMat {
        let mut h = CMat::zeros(a.orbitals(), b.orbitals());
        self.add_couplings_one_way(&mut h, a, b, c64::new(1.0, 0.0));
        h
    }

    /// Adds `⟨a| H |b⟩` (hoppings from a site of `a` to a site of `b`, in either direction of
    /// the stored hoppings) times `factor` into `h`, rows indexed by `a` and columns by `b`.
    fn add_couplings_one_way(&self, h: &mut impl Sink, a: &Region, b: &Region, factor: c64) {
        for (i, site) in a.sites().iter().enumerate() {
            for hop in &self.hoppings {
                // Forward: site is `from`, partner is `to` at +offset.
                if hop.from == site.sub {
                    let partner = shifted(site, hop.offset, 1, hop.to);
                    if let Some(j) = b.index_of(&partner) {
                        h.block(a.offset(i), b.offset(j), &hop.matrix, factor, false);
                    }
                }
                // Backward: site is `to`, partner is `from` at -offset; block is matrix†.
                if hop.to == site.sub {
                    let partner = shifted(site, hop.offset, -1, hop.from);
                    if let Some(j) = b.index_of(&partner) {
                        h.block(a.offset(i), b.offset(j), &hop.matrix, factor, true);
                    }
                }
            }
        }
    }

    /// Transport blocks of a region cut into slices along primitive vector `axis`: the on-slice
    /// Hamiltonians `H_nn` and the couplings `V_n = ⟨n| H |n+1⟩`. Panics if a hopping skips a
    /// slice (range longer than one slice).
    pub fn slice_blocks(&self, region: &Region, axis: usize) -> (Vec<Region>, Vec<CMat>, Vec<CMat>) {
        let slices = region.slices(axis);
        let diag = slices.iter().map(|s| self.hamiltonian(s)).collect();
        let off = slices.windows(2).map(|w| self.coupling(&w[0], &w[1])).collect();
        for hop in &self.hoppings {
            assert!(hop.offset[axis].abs() <= 1, "hopping range exceeds one slice along axis {axis}");
        }
        (slices, diag, off)
    }
}

fn shifted(site: &Site, offset: Cell, sign: i64, sub: usize) -> Site {
    Site {
        cell: [
            site.cell[0] + sign * offset[0],
            site.cell[1] + sign * offset[1],
            site.cell[2] + sign * offset[2],
        ],
        sub,
    }
}

/// Reduction of sites into a region modulo a lattice of periods.
struct Images {
    periods: Vec<Cell>,
    /// Pseudo-inverse of the period matrix (rows: periods), to estimate the image of a site.
    pinv: Vec<[f64; 3]>,
    /// Mean cell of the region.
    centre: [f64; 3],
}

impl Images {
    fn new(region: &Region, periods: &[Cell]) -> Self {
        let m = periods.len();
        // Gram matrix G = P Pᵀ (m × m), pinv = G⁻¹ P.
        let p: Vec<[f64; 3]> = periods.iter().map(|t| [t[0] as f64, t[1] as f64, t[2] as f64]).collect();
        let mut g = vec![0.0; m * m];
        for a in 0..m {
            for b in 0..m {
                g[a * m + b] = (0..3).map(|c| p[a][c] * p[b][c]).sum();
            }
        }
        let ginv = invert(&g, m).expect("periods must be linearly independent");
        let pinv = (0..m)
            .map(|a| {
                let mut row = [0.0; 3];
                for (b, pb) in p.iter().enumerate() {
                    for c in 0..3 {
                        row[c] += ginv[a * m + b] * pb[c];
                    }
                }
                row
            })
            .collect();
        let count = region.len().max(1) as f64;
        let mut centre = [0.0; 3];
        for s in region.sites() {
            for (c, x) in centre.iter_mut().zip(s.cell) {
                *c += x as f64 / count;
            }
        }
        Self { periods: periods.to_vec(), pinv, centre }
    }

    /// `(index in the region, period counts n)` with `site = region[index] + Σ n_i T_i`.
    fn locate(&self, region: &Region, site: &Site) -> Option<(usize, Vec<i64>)> {
        let m = self.periods.len();
        if let Some(j) = region.index_of(site) {
            return Some((j, vec![0; m]));
        }
        let guess: Vec<i64> = self
            .pinv
            .iter()
            .map(|row| {
                (0..3).map(|c| row[c] * (site.cell[c] as f64 - self.centre[c])).sum::<f64>().round() as i64
            })
            .collect();
        // Candidates around the estimate, nearest first.
        for radius in 0..=2i64 {
            let side = (2 * radius + 1) as usize;
            for code in 0..side.pow(m as u32) {
                let mut n = guess.clone();
                let (mut c, mut on_shell) = (code, false);
                for v in n.iter_mut() {
                    let d = (c % side) as i64 - radius;
                    c /= side;
                    on_shell |= d.abs() == radius;
                    *v += d;
                }
                if !on_shell || n.iter().all(|&v| v == 0) {
                    continue;
                }
                let mut cell = site.cell;
                for (k, t) in n.iter().zip(&self.periods) {
                    for c in 0..3 {
                        cell[c] -= k * t[c];
                    }
                }
                if let Some(j) = region.index_of(&Site { cell, sub: site.sub }) {
                    return Some((j, n));
                }
            }
        }
        None
    }
}

/// Inverse of a small dense matrix (row-major), by Gauss–Jordan with partial pivoting.
fn invert(a: &[f64], m: usize) -> Option<Vec<f64>> {
    let mut a = a.to_vec();
    let mut inv: Vec<f64> = (0..m * m).map(|k| if k % (m + 1) == 0 { 1.0 } else { 0.0 }).collect();
    for col in 0..m {
        let piv = (col..m).max_by(|&r, &s| a[r * m + col].abs().total_cmp(&a[s * m + col].abs()))?;
        if a[piv * m + col].abs() < 1e-12 {
            return None;
        }
        for j in 0..m {
            a.swap(col * m + j, piv * m + j);
            inv.swap(col * m + j, piv * m + j);
        }
        let d = a[col * m + col];
        for j in 0..m {
            a[col * m + j] /= d;
            inv[col * m + j] /= d;
        }
        for r in 0..m {
            if r != col {
                let f = a[r * m + col];
                for j in 0..m {
                    a[r * m + j] -= f * a[col * m + j];
                    inv[r * m + j] -= f * inv[col * m + j];
                }
            }
        }
    }
    Some(inv)
}

/// Destination of Hamiltonian blocks: a dense matrix or a list of sparse entries.
trait Sink {
    /// Adds `factor · m` (or `factor · m†`) with its top-left corner at `(r0, c0)`.
    fn block(&mut self, r0: usize, c0: usize, m: &CMat, factor: c64, adjoint: bool);
}

impl Sink for CMat {
    fn block(&mut self, r0: usize, c0: usize, m: &CMat, factor: c64, adjoint: bool) {
        let (rows, cols) = if adjoint { (m.ncols(), m.nrows()) } else { (m.nrows(), m.ncols()) };
        for j in 0..cols {
            for i in 0..rows {
                let v = if adjoint { m[(j, i)].conj() } else { m[(i, j)] };
                self[(r0 + i, c0 + j)] += factor * v;
            }
        }
    }
}

struct Triplets(Vec<(usize, usize, c64)>);

impl Sink for Triplets {
    fn block(&mut self, r0: usize, c0: usize, m: &CMat, factor: c64, adjoint: bool) {
        let (rows, cols) = if adjoint { (m.ncols(), m.nrows()) } else { (m.nrows(), m.ncols()) };
        for j in 0..cols {
            for i in 0..rows {
                let v = if adjoint { m[(j, i)].conj() } else { m[(i, j)] };
                if v != c64::new(0.0, 0.0) {
                    self.0.push((r0 + i, c0 + j, factor * v));
                }
            }
        }
    }
}

/// `Σ_i c_i M_i` for small matrices of the same shape.
pub fn combine(terms: &[(c64, &CMat)]) -> CMat {
    let (n, m) = (terms[0].1.nrows(), terms[0].1.ncols());
    CMat::from_fn(n, m, |i, j| terms.iter().map(|(c, mat)| *c * mat[(i, j)]).sum())
}
