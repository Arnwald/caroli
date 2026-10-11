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
        for i in 0..region.len() {
            add_block(
                &mut h,
                region.offset(i),
                region.offset(i),
                &self.onsite_matrix(&region.sites()[i]),
                c64::new(1.0, 0.0),
            );
        }
        self.add_couplings_one_way(&mut h, region, region, c64::new(1.0, 0.0));
        h
    }

    /// Bloch Hamiltonian of a region repeated periodically: `periods` are translations (in
    /// cells) and `phases` the matching Bloch phases `k·T` in radians. Hoppings that leave the
    /// region by `n` periods pick up `exp(i Σ n·phase)`.
    pub fn bloch_hamiltonian(&self, region: &Region, periods: &[Cell], phases: &[f64]) -> CMat {
        assert_eq!(periods.len(), phases.len());
        let mut h = self.hamiltonian(region);
        // Neighbouring images: every combination of -1, 0, +1 period, except the region itself.
        let combos = 3usize.pow(periods.len() as u32);
        for c in 0..combos {
            let mut shift = [0i64; 3];
            let mut phase = 0.0;
            let mut code = c;
            let mut any = false;
            for (t, k) in periods.iter().zip(phases) {
                let n = (code % 3) as i64 - 1;
                code /= 3;
                any |= n != 0;
                for a in 0..3 {
                    shift[a] += n * t[a];
                }
                phase += n as f64 * k;
            }
            if any {
                let image = region.translated(shift);
                let (s, cs) = phase.sin_cos();
                self.add_couplings_one_way(&mut h, region, &image, c64::new(cs, s));
            }
        }
        h
    }

    /// Coupling matrix `⟨a| H |b⟩` between two disjoint regions (rows: `a`, columns: `b`).
    pub fn coupling(&self, a: &Region, b: &Region) -> CMat {
        let mut h = CMat::zeros(a.orbitals(), b.orbitals());
        self.add_couplings_one_way(&mut h, a, b, c64::new(1.0, 0.0));
        h
    }

    /// Adds `⟨a| H |b⟩` (hoppings from a site of `a` to a site of `b`, in either direction of
    /// the stored hoppings) times `factor` into `h`, rows indexed by `a` and columns by `b`.
    fn add_couplings_one_way(&self, h: &mut CMat, a: &Region, b: &Region, factor: c64) {
        for (i, site) in a.sites().iter().enumerate() {
            for hop in &self.hoppings {
                // Forward: site is `from`, partner is `to` at +offset.
                if hop.from == site.sub {
                    let partner = shifted(site, hop.offset, 1, hop.to);
                    if let Some(j) = b.index_of(&partner) {
                        add_block(h, a.offset(i), b.offset(j), &hop.matrix, factor);
                    }
                }
                // Backward: site is `to`, partner is `from` at -offset; block is matrix†.
                if hop.to == site.sub {
                    let partner = shifted(site, hop.offset, -1, hop.from);
                    if let Some(j) = b.index_of(&partner) {
                        add_block_adjoint(h, a.offset(i), b.offset(j), &hop.matrix, factor);
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

fn add_block(h: &mut CMat, r0: usize, c0: usize, m: &CMat, factor: c64) {
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            h[(r0 + i, c0 + j)] += factor * m[(i, j)];
        }
    }
}

fn add_block_adjoint(h: &mut CMat, r0: usize, c0: usize, m: &CMat, factor: c64) {
    for j in 0..m.nrows() {
        for i in 0..m.ncols() {
            h[(r0 + i, c0 + j)] += factor * m[(j, i)].conj();
        }
    }
}

/// `Σ_i c_i M_i` for small matrices of the same shape.
pub fn combine(terms: &[(c64, &CMat)]) -> CMat {
    let (n, m) = (terms[0].1.nrows(), terms[0].1.ncols());
    CMat::from_fn(n, m, |i, j| terms.iter().map(|(c, mat)| *c * mat[(i, j)]).sum())
}
