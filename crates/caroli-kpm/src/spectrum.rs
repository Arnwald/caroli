//! Densities of states and conductivities of a [`Model`] by the kernel polynomial method.
//!
//! [`Spectrum`] builds the sparse Hamiltonian of a region, open or periodic, rescales it into
//! (−1, 1) and computes Chebyshev moments: the density of states `Tr δ(E − H)`, its projection on
//! an on-site operator (spin, orbital, sublattice), local densities of states, and the
//! Kubo–Bastin conductivity of a periodic sample.
//!
//! ```
//! use caroli_kpm::Spectrum;
//! use caroli_lattice::{Lattice, Region};
//! use caroli_model::Model;
//!
//! // Square lattice, 64 × 64 torus: the band runs from −4 to 4.
//! let model = Model::new(Lattice::square(1.0, 1)).hopping(0, 0, [1, 0, 0], -1.0).hopping(0, 0, [0, 1, 0], -1.0);
//! let region = Region::cuboid(model.lattice(), [64, 64, 0]);
//! let spectrum = Spectrum::periodic(&model, &region, &[[64, 0, 0], [0, 64, 0]]).moments(128).vectors(4);
//! let rho = spectrum.dos(&[-5.0, 0.5]);
//! assert!(rho[0] == 0.0 && rho[1] > 0.0);
//! ```

use crate::kubo::{kubo_bastin, kubo_moments};
use crate::{density, interleave, jackson, random_phases};
use caroli_disorder::rng::Xoshiro256;
use caroli_gpu::{Options, Report};
use caroli_lattice::{Cell, Region, Vec3};
use caroli_model::{CMat, Model, SparseMatrix, c64};

/// The kernel polynomial method on the Hamiltonian of a [`Model`] on a [`Region`].
///
/// Energies are those of the model. Densities are per unit energy and per sample: the density of
/// states integrates to the number of orbitals of the region. The energy resolution of the
/// Jackson kernel is about `π · half_width / moments` ([`Self::resolution`]).
pub struct Spectrum<'a> {
    model: &'a Model,
    region: &'a Region,
    periods: Vec<Cell>,
    /// `H` and `H̃ = (H − centre)/half_width`.
    h: SparseMatrix,
    scaled: SparseMatrix,
    centre: f64,
    half_width: f64,
    moments: usize,
    trace: Trace,
    seed: u64,
    options: Options,
}

/// How traces are estimated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Trace {
    /// Random-phase vectors (`⟨r|A|r⟩` estimates `Tr A`, relative error ~ 1/√(vectors · N)).
    Stochastic(usize),
    /// Every basis vector: exact, for small systems and tests.
    Exact,
}

impl<'a> Spectrum<'a> {
    /// The region with open boundaries.
    pub fn new(model: &'a Model, region: &'a Region) -> Self {
        Self::periodic(model, region, &[])
    }

    /// The region repeated along `periods` (cells), e.g. `[[n, 0, 0], [0, n, 0]]` for an
    /// `n × n` torus. The spectral bounds come from Gershgorin's theorem, widened by 1 %.
    pub fn periodic(model: &'a Model, region: &'a Region, periods: &[Cell]) -> Self {
        let h = model.sparse_bloch_hamiltonian(region, periods, &vec![0.0; periods.len()]);
        let (lo, hi) = h.spectral_bounds();
        let (centre, half_width) = scale(lo, hi);
        let scaled = h.rescaled(centre, half_width);
        Self {
            model,
            region,
            periods: periods.to_vec(),
            h,
            scaled,
            centre,
            half_width,
            moments: 256,
            trace: Trace::Stochastic(10),
            seed: 1,
            options: Options::cpu(),
        }
    }

    /// Bounds on the spectrum, tighter than Gershgorin's for a finer resolution. They must
    /// contain every eigenvalue (they are widened by 1 %).
    pub fn bounds(mut self, lower: f64, upper: f64) -> Self {
        assert!(lower < upper, "empty spectral window [{lower}, {upper}]");
        (self.centre, self.half_width) = scale(lower, upper);
        self.scaled = self.h.rescaled(self.centre, self.half_width);
        self
    }

    /// Number of Chebyshev moments (default 256).
    pub fn moments(mut self, moments: usize) -> Self {
        assert!(moments >= 2, "at least two moments");
        self.moments = moments;
        self
    }

    /// Number of random-phase vectors of the stochastic trace (default 10).
    pub fn vectors(mut self, vectors: usize) -> Self {
        assert!(vectors > 0, "at least one random vector");
        self.trace = Trace::Stochastic(vectors);
        self
    }

    /// Exact traces over every basis vector instead of random vectors (small systems).
    pub fn exact_trace(mut self) -> Self {
        self.trace = Trace::Exact;
        self
    }

    /// Seed of the random vectors (default 1).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Where the density-of-states moments run (CPU by default; see [`caroli_gpu::Options`]).
    pub fn device(mut self, options: Options) -> Self {
        self.options = options;
        self
    }

    /// `(centre, half_width)` of the rescaling `H̃ = (H − centre)/half_width`.
    pub fn scaling(&self) -> (f64, f64) {
        (self.centre, self.half_width)
    }

    /// Width of the Jackson kernel, `π · half_width / moments`.
    pub fn resolution(&self) -> f64 {
        std::f64::consts::PI * self.half_width / self.moments as f64
    }

    /// The Hamiltonian (unscaled).
    pub fn hamiltonian(&self) -> &SparseMatrix {
        &self.h
    }

    /// The start vectors of a trace over the whole basis.
    fn starts(&self) -> Vec<Vec<c64>> {
        let n = self.h.n;
        match self.trace {
            Trace::Stochastic(count) => {
                let mut rng = Xoshiro256::seed_from_u64(self.seed);
                (0..count).map(|_| random_phases(&mut rng, n)).collect()
            }
            Trace::Exact => (0..n).map(|i| unit(n, i)).collect(),
        }
    }

    /// Weight that turns a mean over the start vectors into a trace.
    fn trace_weight(&self, count: usize) -> f64 {
        match self.trace {
            Trace::Stochastic(_) => 1.0,
            Trace::Exact => count as f64,
        }
    }

    /// Moments `μ_n = g_n Tr T_n(H̃)` of the density of states, and where they ran.
    pub fn dos_moments(&self) -> (Vec<c64>, Report) {
        let kernel = jackson(self.moments);
        match self.trace {
            Trace::Stochastic(_) => {
                crate::accelerated::moments(&self.scaled, &self.starts(), &kernel, &self.options)
            }
            Trace::Exact => {
                // Blocks of basis vectors, so that the memory stays that of a few vectors.
                let n = self.h.n;
                let mut mu = vec![c64::new(0.0, 0.0); self.moments];
                for first in (0..n).step_by(BLOCK) {
                    let block: Vec<Vec<c64>> = (first..(first + BLOCK).min(n)).map(|i| unit(n, i)).collect();
                    for m in cross_moments(&self.scaled, &block, &block, &kernel) {
                        mu.iter_mut().zip(m).for_each(|(a, b)| *a += b);
                    }
                }
                (mu, Report { ran_on: "CPU".into(), deviation: None, fallback: None })
            }
        }
    }

    /// Density per unit energy at `energies` from moments of `H̃` (zero outside the bounds).
    pub fn density(&self, mu: &[c64], energies: &[f64]) -> Vec<f64> {
        let x: Vec<f64> = energies.iter().map(|e| (e - self.centre) / self.half_width).collect();
        let inside: Vec<f64> = x.iter().copied().filter(|x| x.abs() < 1.0).collect();
        let mut values = density(mu, &inside).into_iter();
        x.iter()
            .map(|x| if x.abs() < 1.0 { values.next().expect("value") / self.half_width } else { 0.0 })
            .collect()
    }

    /// Density of states `Tr δ(E − H)` at `energies`.
    pub fn dos(&self, energies: &[f64]) -> Vec<f64> {
        self.density(&self.dos_moments().0, energies)
    }

    /// `Tr[O δ(E − H)]` for a Hermitian on-site operator `O` (the same matrix on every site, e.g.
    /// a spin or sublattice projector; all sites must carry `O.nrows()` orbitals).
    pub fn projected_dos(&self, op: &CMat, energies: &[f64]) -> Vec<f64> {
        let starts = self.starts();
        let left: Vec<Vec<c64>> = starts.iter().map(|r| self.on_site(op, r)).collect();
        let kernel = jackson(self.moments);
        let per_vector = cross_moments(&self.scaled, &left, &starts, &kernel);
        let w = self.trace_weight(starts.len()) / starts.len() as f64;
        let mu: Vec<c64> =
            (0..self.moments).map(|n| per_vector.iter().map(|m| m[n]).sum::<c64>() * w).collect();
        self.density(&mu, energies)
    }

    /// Local densities of states `Σ_a ⟨i, a| δ(E − H) |i, a⟩` of the sites `sites` (indices into
    /// the region), summed over their orbitals: one curve per site.
    pub fn ldos(&self, sites: &[usize], energies: &[f64]) -> Vec<Vec<f64>> {
        let n = self.h.n;
        let mut owner = Vec::new();
        let mut starts = Vec::new();
        for (k, &i) in sites.iter().enumerate() {
            for a in 0..self.region.site_orbitals(i) {
                starts.push(unit(n, self.region.offset(i) + a));
                owner.push(k);
            }
        }
        let kernel = jackson(self.moments);
        let per_vector = cross_moments(&self.scaled, &starts, &starts, &kernel);
        let mut mu = vec![vec![c64::new(0.0, 0.0); self.moments]; sites.len()];
        for (m, &k) in per_vector.iter().zip(&owner) {
            mu[k].iter_mut().zip(m).for_each(|(a, b)| *a += b);
        }
        mu.iter().map(|m| self.density(m, energies)).collect()
    }

    /// Kubo–Bastin conductivity `σ_ab` in units of `e²/h` at the chemical potentials `mus` and the
    /// temperature `temperature` (energy units; 0 for a step), for a region periodic in every
    /// direction of the lattice (an `n × n` torus in 2D). `a` and `b` are directions (unit
    /// vectors). The cost grows as `moments²` per random vector.
    pub fn conductivity(&self, a: Vec3, b: Vec3, mus: &[f64], temperature: f64) -> Vec<f64> {
        let dim = self.model.lattice().dim();
        assert_eq!(
            self.periods.len(),
            dim,
            "the conductivity needs a region periodic in all {dim} directions"
        );
        let zero = vec![0.0; dim];
        let va = self.model.sparse_velocity(self.region, &self.periods, &zero, a);
        let vb = self.model.sparse_velocity(self.region, &self.periods, &zero, b);
        let starts = self.starts();
        let kernel = jackson(self.moments);
        let mut mu = kubo_moments(
            |x, y| self.scaled.apply(x, y),
            |x, y| va.apply(x, y),
            |x, y| vb.apply(x, y),
            &starts,
            &kernel,
        );
        let w = self.trace_weight(starts.len());
        mu.iter_mut().flatten().for_each(|m| *m *= w);
        let grid = (4 * self.moments).max(1024);
        kubo_bastin(&mu, self.centre, self.half_width, self.volume(), mus, temperature, grid)
    }

    /// Area (2D) or volume (3D) of the periodic supercell, in the length unit of the lattice.
    fn volume(&self) -> f64 {
        let vectors = self.model.lattice().vectors();
        let t: Vec<Vec3> = self
            .periods
            .iter()
            .map(|p| std::array::from_fn(|c| (0..vectors.len()).map(|i| p[i] as f64 * vectors[i][c]).sum()))
            .collect();
        let cross = |u: Vec3, v: Vec3| {
            [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        };
        let norm = |u: Vec3| (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        match t.len() {
            1 => norm(t[0]),
            2 => norm(cross(t[0], t[1])),
            3 => {
                let c = cross(t[1], t[2]);
                (t[0][0] * c[0] + t[0][1] * c[1] + t[0][2] * c[2]).abs()
            }
            _ => unreachable!("periods of a lattice of dimension 1 to 3"),
        }
    }

    /// `O` applied on every site of a vector.
    fn on_site(&self, op: &CMat, x: &[c64]) -> Vec<c64> {
        let mut y = vec![c64::new(0.0, 0.0); x.len()];
        for i in 0..self.region.len() {
            let (o, m) = (self.region.offset(i), self.region.site_orbitals(i));
            assert_eq!(op.nrows(), m, "on-site operator of size {} on a site with {m} orbitals", op.nrows());
            for r in 0..m {
                y[o + r] = (0..m).map(|c| op[(r, c)] * x[o + c]).sum();
            }
        }
        y
    }
}

/// `(centre, half_width)` mapping `[lower, upper]`, widened by 1 %, onto [−1, 1].
fn scale(lower: f64, upper: f64) -> (f64, f64) {
    let centre = 0.5 * (lower + upper);
    let half = 0.5 * (upper - lower) * 1.01;
    (centre, half.max(1e-12 * centre.abs().max(1.0)))
}

/// Basis vector `i` of length `n`.
fn unit(n: usize, i: usize) -> Vec<c64> {
    let mut v = vec![c64::new(0.0, 0.0); n];
    v[i] = c64::new(1.0, 0.0);
    v
}

/// Vectors per block of [`cross_moments`] (bounds the memory to a few blocks of this size).
const BLOCK: usize = 32;

/// Moments `kernel_n ⟨left_v| T_n(h) |right_v⟩` of each pair of vectors, the Chebyshev vectors of
/// the right ones computed in blocks with [`SparseMatrix::apply_block`].
fn cross_moments(h: &SparseMatrix, left: &[Vec<c64>], right: &[Vec<c64>], kernel: &[f64]) -> Vec<Vec<c64>> {
    assert_eq!(left.len(), right.len());
    let mut out = Vec::with_capacity(right.len());
    for (lb, rb) in left.chunks(BLOCK).zip(right.chunks(BLOCK)) {
        let m = rb.len();
        let l = interleave(lb);
        let mut prev = interleave(rb);
        let mut cur = vec![c64::new(0.0, 0.0); prev.len()];
        let mut next = vec![c64::new(0.0, 0.0); prev.len()];
        let mut mu = vec![Vec::with_capacity(kernel.len()); m];
        let dots = |x: &[c64], mu: &mut Vec<Vec<c64>>, g: f64| {
            let mut acc = vec![c64::new(0.0, 0.0); m];
            for (lrow, xrow) in l.chunks_exact(m).zip(x.chunks_exact(m)) {
                acc.iter_mut().zip(lrow.iter().zip(xrow)).for_each(|(a, (p, q))| *a += p.conj() * q);
            }
            mu.iter_mut().zip(acc).for_each(|(v, a)| v.push(a * g));
        };
        dots(&prev, &mut mu, kernel[0]);
        if kernel.len() > 1 {
            h.apply_block(&prev, &mut cur, m);
            dots(&cur, &mut mu, kernel[1]);
        }
        for &g in kernel.iter().skip(2) {
            h.apply_block(&cur, &mut next, m);
            next.iter_mut().zip(&prev).for_each(|(x, p)| *x = 2.0 * *x - p);
            dots(&next, &mut mu, g);
            std::mem::swap(&mut prev, &mut cur);
            std::mem::swap(&mut cur, &mut next);
        }
        out.extend(mu);
    }
    out
}
