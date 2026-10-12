//! Tight-binding models on a [`Lattice`] and the matrices they produce on a [`Region`].
//!
//! A [`Model`] holds an on-site matrix per sublattice and a list of [`Hopping`]s
//! `⟨(R, from)| H |(R + offset, to)⟩`. Extra on-site terms that depend on the site (disorder,
//! potentials) are added through [`OnSite`], and a magnetic field through [`Peierls`] phases
//! (see [`gauge`]). The model builds dense Hamiltonians of finite regions, Bloch Hamiltonians of
//! partially periodic systems and the blocks used by transport.

use caroli_lattice::{Cell, Lattice, Region, Site, Vec3};
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

/// Peierls phase of the hopping `⟨i| H |j⟩`: `φ(r_i, r_j) = ∫ A · dl` from `r_j` to `r_i`, in
/// radians (charge and ħ absorbed in `A`). It must be antisymmetric: `φ(a, b) = −φ(b, a)`.
pub trait Peierls: Send + Sync {
    fn phase(&self, to: Vec3, from: Vec3) -> f64;
}

impl<F: Fn(Vec3, Vec3) -> f64 + Send + Sync> Peierls for F {
    fn phase(&self, to: Vec3, from: Vec3) -> f64 {
        self(to, from)
    }
}

/// Uniform magnetic fields `B ẑ` as [`Peierls`] phases, for [`Model::with_peierls`].
///
/// `b` is the field in units of `ħ/(e a²)`, `a` being the length unit of the lattice: the phase
/// around a loop is `b` times the area it encloses (counterclockwise), and the flux quantum is
/// `2π`. On the square lattice of constant 1, `b` is the flux per plaquette and the Landau levels
/// near the band bottom sit at `E_n ≈ −4t + (2n + 1) t b`.
///
/// The leads must see a gauge that is invariant along their axis (the transport constructors
/// check it): [`landau_x`] for leads along x, [`landau_y`] for leads along y.
pub mod gauge {
    use caroli_lattice::Vec3;

    /// `A = (−b y, 0, 0)`: invariant along x.
    pub fn landau_x(b: f64) -> impl Fn(Vec3, Vec3) -> f64 + Send + Sync + Clone {
        move |ri, rj| -b * 0.5 * (ri[1] + rj[1]) * (ri[0] - rj[0])
    }

    /// `A = (0, b x, 0)`: invariant along y.
    pub fn landau_y(b: f64) -> impl Fn(Vec3, Vec3) -> f64 + Send + Sync + Clone {
        move |ri, rj| b * 0.5 * (ri[0] + rj[0]) * (ri[1] - rj[1])
    }

    /// `A = b (−y, x, 0) / 2`.
    pub fn symmetric(b: f64) -> impl Fn(Vec3, Vec3) -> f64 + Send + Sync + Clone {
        move |ri, rj| {
            let (xm, ym) = (0.5 * (ri[0] + rj[0]), 0.5 * (ri[1] + rj[1]));
            0.5 * b * (-ym * (ri[0] - rj[0]) + xm * (ri[1] - rj[1]))
        }
    }
}

/// Tight-binding model on a lattice.
pub struct Model {
    lattice: Lattice,
    onsite: Vec<CMat>,
    hoppings: Vec<Hopping>,
    extra: Vec<Box<dyn OnSite>>,
    peierls: Option<Box<dyn Peierls>>,
}

impl Model {
    /// Model with zero on-site matrices and no hopping.
    pub fn new(lattice: Lattice) -> Self {
        let onsite = lattice.sublattices().iter().map(|s| CMat::zeros(s.orbitals, s.orbitals)).collect();
        Self { lattice, onsite, hoppings: Vec::new(), extra: Vec::new(), peierls: None }
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

    /// Multiplies every hopping `⟨i| H |j⟩` by `exp(i φ(r_i, r_j))`: a vector potential through
    /// the Peierls substitution (see [`gauge`] for uniform fields). Replaces any previous phase.
    pub fn with_peierls(mut self, phase: impl Peierls + 'static) -> Self {
        self.peierls = Some(Box::new(phase));
        self
    }

    /// `factor · exp(i φ(r_i, r_j))` for the block `⟨i| H |j⟩`.
    fn phased(&self, factor: c64, i: &Site, j: &Site) -> c64 {
        match &self.peierls {
            None => factor,
            Some(p) => {
                let (s, c) = p.phase(self.lattice.position(*i), self.lattice.position(*j)).sin_cos();
                factor * c64::new(c, s)
            }
        }
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
                    let partner = shifted(site, hop.offset, sign, sub);
                    if let Some((j, n)) = images.locate(&partner) {
                        let phase: f64 = n.iter().zip(phases).map(|(n, k)| *n as f64 * k).sum();
                        let (s, cs) = phase.sin_cos();
                        let factor = self.phased(c64::new(cs, s), site, &partner);
                        h.block(region.offset(i), region.offset(j), &hop.matrix, factor, adjoint);
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
                        h.block(
                            a.offset(i),
                            b.offset(j),
                            &hop.matrix,
                            self.phased(factor, site, &partner),
                            false,
                        );
                    }
                }
                // Backward: site is `to`, partner is `from` at -offset; block is matrix†.
                if hop.to == site.sub {
                    let partner = shifted(site, hop.offset, -1, hop.from);
                    if let Some(j) = b.index_of(&partner) {
                        h.block(
                            a.offset(i),
                            b.offset(j),
                            &hop.matrix,
                            self.phased(factor, site, &partner),
                            true,
                        );
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

/// Exact reduction of sites into a region modulo the lattice of the periods.
///
/// The periods (rows) are brought to row-echelon (Hermite) form `H = W P` by unimodular integer
/// row operations. Reducing a cell along the pivots of `H` gives a canonical representative, so
/// every site of the region is filed under its class, and a hopping partner is found by its
/// class with the exact number of periods it lies away.
struct Images {
    /// Echelon rows and their pivot columns.
    rows: Vec<[i64; 3]>,
    pivots: Vec<usize>,
    /// `H = W P`.
    w: Vec<Vec<i64>>,
    /// Class `(representative cell, sublattice)` → (index in the region, echelon coefficients).
    classes: std::collections::HashMap<([i64; 3], usize), (usize, Vec<i64>)>,
}

impl Images {
    fn new(region: &Region, periods: &[Cell]) -> Self {
        let m = periods.len();
        let mut rows: Vec<[i64; 3]> = periods.to_vec();
        let mut w: Vec<Vec<i64>> = (0..m).map(|i| (0..m).map(|j| (i == j) as i64).collect()).collect();
        let mut pivots = Vec::new();
        let mut top = 0;
        for col in 0..3 {
            if top == m {
                break;
            }
            // Euclid on the rows top.. in this column.
            while let Some(p) = (top..m).filter(|&i| rows[i][col] != 0).min_by_key(|&i| rows[i][col].abs()) {
                rows.swap(top, p);
                w.swap(top, p);
                let (pivot_row, pivot_w) = (rows[top], w[top].clone());
                let mut done = true;
                for i in top + 1..m {
                    let f = rows[i][col] / pivot_row[col];
                    if f != 0 {
                        rows[i].iter_mut().zip(&pivot_row).for_each(|(x, y)| *x -= f * y);
                        w[i].iter_mut().zip(&pivot_w).for_each(|(x, y)| *x -= f * y);
                    }
                    done &= rows[i][col] == 0;
                }
                if done {
                    break;
                }
            }
            if rows[top][col] != 0 {
                if rows[top][col] < 0 {
                    rows[top].iter_mut().for_each(|x| *x = -*x);
                    w[top].iter_mut().for_each(|x| *x = -*x);
                }
                pivots.push(col);
                top += 1;
            }
        }
        assert_eq!(top, m, "periods must be linearly independent");
        let mut images = Self { rows, pivots, w, classes: Default::default() };
        for (i, site) in region.sites().iter().enumerate() {
            let (rep, k) = images.reduce(site.cell);
            let previous = images.classes.insert((rep, site.sub), (i, k));
            assert!(previous.is_none(), "the region holds two sites a whole number of periods apart");
        }
        images
    }

    /// Canonical representative of a cell and its echelon coefficients `k`: `cell = rep + Σ k_i H_i`.
    fn reduce(&self, cell: [i64; 3]) -> ([i64; 3], Vec<i64>) {
        let mut x = cell;
        let mut k = vec![0; self.rows.len()];
        for (i, (&c, row)) in self.pivots.iter().zip(&self.rows).enumerate() {
            let q = x[c].div_euclid(row[c]);
            for a in 0..3 {
                x[a] -= q * row[a];
            }
            k[i] = q;
        }
        (x, k)
    }

    /// `(index in the region, period counts n)` with `site = region[index] + Σ n_l T_l`, or `None`
    /// if the class of the site is not in the region (open directions).
    fn locate(&self, site: &Site) -> Option<(usize, Vec<i64>)> {
        let (rep, k) = self.reduce(site.cell);
        let (j, kq) = self.classes.get(&(rep, site.sub))?;
        let m = self.rows.len();
        let n = (0..m).map(|l| (0..m).map(|i| (k[i] - kq[i]) * self.w[i][l]).sum()).collect();
        Some((*j, n))
    }
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
