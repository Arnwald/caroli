//! Disorder-averaged Green's functions in the self-consistent Born approximation (SCBA).
//!
//! For a periodic model `H(k)` with a random on-site potential `V`, the averaged Green's
//! function `⟨G⟩(z) = (1/N_k) Σ_k [z − H(k) − Σ(z)]⁻¹` is built with the self-energy
//!
//! ```text
//! Σ = Σ_MF + Σ₂,   Σ_MF = ⟨V⟩,   Σ₂ = Σ_t w_t V_t ⟨G⟩ V_t,
//! ```
//!
//! where the vertices `(w_t, V_t)` decompose the second cumulant `⟨V⊗V⟩ − ⟨V⟩⊗⟨V⟩` of the
//! disorder. [`Scba::update`] is the map `Σ₂ ↦ Σ_t w_t V_t ⟨G⟩ V_t`; [`Scba::solve`] iterates it.
//! [`anderson`] and [`magnetic`] build the vertices of the two thesis disorders.
//!
//! From a [`Model`]: [`Scba::from_model`] samples `H(k)` on a uniform grid, a [`Disorder`]
//! describes the impurities, and [`Scba::self_energy`], [`Scba::sweep`] and [`Scba::dos`] give
//! the self-energy and the averaged density of states.
//!
//! ```
//! use caroli_born::{Disorder, Scba};
//! use caroli_model::presets::{BhzParams, bhz};
//!
//! // BHZ with Anderson disorder W = 1 on a 64 × 64 k grid, at E = 0.05 + 0.001 i.
//! let model = bhz(BhzParams::default());
//! let scba = Scba::from_model(&model, &[64, 64], Disorder::anderson(1.0, 4));
//! let sol = scba.self_energy(0.05, 1e-3);
//! assert!(sol.residual < 1e-10);
//! // The renormalised mass M̄ = M + Re(Σ_E − Σ_H)/2 moves further to the topological side
//! // (M̄/B > M/B), as in the topological Anderson insulator.
//! let shift = 0.5 * (sol.sigma[(0, 0)].re - sol.sigma[(1, 1)].re);
//! assert!(shift / BhzParams::default().b > 0.0);
//! ```

use caroli_disorder::Moments;
use caroli_lattice::Region;
use caroli_model::{CMat, Model, c64};
use faer::linalg::solvers::DenseSolveCore;
use rayon::prelude::*;

/// Uniform k grid of `m × n` points `2π (i/m, j/n)` in the first Brillouin zone, listed as the
/// thesis `k_FBZ` (negative values first: `−π, …, −δ, 0, …, π − δ` for even sizes,
/// `−π + δ/2 … π − δ/2` symmetric for odd ones); returned as `(kx, ky)` with `kx` varying
/// fastest.
pub fn k_grid(m: usize, n: usize) -> Vec<[f64; 2]> {
    let axis = |len: usize| -> Vec<f64> {
        let k0 = 2.0 * std::f64::consts::PI / len as f64;
        // k_FBZ lists (i − len) k0 for the upper indices, then i k0 for i = 0, 1, …; the
        // positive half has ⌈len/2⌉ values so that odd sizes stay inside (−π, π).
        let half = len.div_ceil(2);
        (half..len).map(|i| (i as f64 - len as f64) * k0).chain((0..half).map(|i| i as f64 * k0)).collect()
    };
    let (kx, ky) = (axis(m), axis(n));
    ky.iter().flat_map(|&y| kx.iter().map(move |&x| [x, y])).collect()
}

/// SCBA problem: `H(k)` on a k grid, mean field `⟨V⟩` and second-order vertices.
#[derive(Clone, Debug)]
pub struct Scba {
    pub hk: Vec<CMat>,
    pub mean_field: CMat,
    pub vertices: Vec<(f64, CMat)>,
}

/// Result of [`Scba::solve`].
#[derive(Clone, Debug)]
pub struct Solution {
    /// Second-order self-energy `Σ₂` (the mean field is not included).
    pub sigma: CMat,
    pub iterations: usize,
    /// Largest element of the last change of `Σ₂`.
    pub residual: f64,
}

fn identity_minus(z: c64, h: &CMat) -> CMat {
    let n = h.nrows();
    CMat::from_fn(n, n, |i, j| if i == j { z - h[(i, j)] } else { -h[(i, j)] })
}

impl Scba {
    /// `⟨G⟩(z) = (1/N_k) Σ_k [z − H(k) − Σ_MF − Σ₂]⁻¹`. Large grids are summed in parallel, in
    /// fixed chunks added in order, so the result does not depend on the number of threads.
    pub fn local_green(&self, z: c64, sigma: &CMat) -> CMat {
        let n = self.mean_field.nrows();
        let shift = &self.mean_field + sigma;
        let partial = |hk: &[CMat]| -> Vec<c64> {
            let mut acc = vec![c64::new(0.0, 0.0); n * n];
            let mut a = vec![c64::new(0.0, 0.0); n * n];
            let mut inv = vec![c64::new(0.0, 0.0); n * n];
            for h in hk {
                if n <= SMALL {
                    for j in 0..n {
                        for i in 0..n {
                            let d = if i == j { z } else { c64::new(0.0, 0.0) };
                            a[i + n * j] = d - h[(i, j)] - shift[(i, j)];
                        }
                    }
                    gauss_jordan_inverse(n, &mut a, &mut inv);
                    acc.iter_mut().zip(&inv).for_each(|(s, x)| *s += x);
                } else {
                    let m = identity_minus(z, h) - &shift;
                    let g = m.partial_piv_lu().inverse();
                    for j in 0..n {
                        for i in 0..n {
                            acc[i + n * j] += g[(i, j)];
                        }
                    }
                }
            }
            acc
        };
        let acc = if self.hk.len() >= 2 * K_CHUNK {
            let parts: Vec<Vec<c64>> = self.hk.par_chunks(K_CHUNK).map(partial).collect();
            let mut acc = vec![c64::new(0.0, 0.0); n * n];
            for p in &parts {
                acc.iter_mut().zip(p).for_each(|(s, x)| *s += x);
            }
            acc
        } else {
            partial(&self.hk)
        };
        let scale = 1.0 / self.hk.len() as f64;
        CMat::from_fn(n, n, |i, j| acc[i + n * j] * scale)
    }

    /// One SCBA map: `Σ_t w_t V_t ⟨G⟩(z; Σ₂) V_t`.
    pub fn update(&self, z: c64, sigma: &CMat) -> CMat {
        let g = self.local_green(z, sigma);
        let n = g.nrows();
        let mut out = CMat::zeros(n, n);
        for (w, v) in &self.vertices {
            let t = v * &g * v;
            out += CMat::from_fn(n, n, |i, j| c64::new(*w, 0.0) * t[(i, j)]);
        }
        out
    }

    /// Fixed-point iteration `Σ ← (1 − p) Σ + p F(Σ)` from `guess` until the largest change
    /// of an element is below `tol` (at most `max_iter` steps). A non-finite update stops the
    /// iteration with a NaN residual.
    pub fn solve(&self, z: c64, guess: &CMat, mixing: f64, tol: f64, max_iter: usize) -> Solution {
        let mut sigma = guess.clone();
        let mut residual = f64::INFINITY;
        let mut iterations = 0;
        while residual > tol && iterations < max_iter {
            let next = self.update(z, &sigma);
            residual = max_abs_diff(&next, &sigma);
            sigma = mix(&next, &sigma, mixing);
            iterations += 1;
        }
        Solution { sigma, iterations, residual }
    }
}

impl Scba {
    /// Fixed point `Σ₂ = F(Σ₂)` by Anderson acceleration (Anderson mixing, also known as DIIS)
    /// on the residual `F(Σ) − Σ` with the last `depth` iterates, from `guess`, until the
    /// largest change of an element is below `tol`. It needs far fewer k sums than
    /// [`Self::solve`]. When the extrapolation misbehaves (non-finite values, a residual that
    /// grows tenfold), the history is dropped and a plain step taken.
    pub fn solve_anderson(&self, z: c64, guess: &CMat, depth: usize, tol: f64, max_iter: usize) -> Solution {
        let n = guess.nrows();
        let flat = |m: &CMat| -> Vec<f64> {
            (0..n * n).flat_map(|k| [m[(k % n, k / n)].re, m[(k % n, k / n)].im]).collect()
        };
        let unflat =
            |v: &[f64]| CMat::from_fn(n, n, |i, j| c64::new(v[2 * (i + n * j)], v[2 * (i + n * j) + 1]));
        // NaN-propagating maximum: a failed update must not look converged.
        let residual = |f: &[f64]| {
            f.chunks_exact(2)
                .map(|c| c[0].hypot(c[1]))
                .fold(0.0, |m: f64, d| if d.is_nan() || m.is_nan() { f64::NAN } else { m.max(d) })
        };
        let mut x = flat(guess);
        let mut g = flat(&self.update(z, guess));
        let mut f: Vec<f64> = g.iter().zip(&x).map(|(a, b)| a - b).collect();
        let (mut dfs, mut dgs): (Vec<Vec<f64>>, Vec<Vec<f64>>) = (Vec::new(), Vec::new());
        let mut iterations = 1;
        let mut res = residual(&f);
        while res > tol && iterations < max_iter {
            let mut next = g.clone();
            if !dfs.is_empty() {
                if let Some(gamma) = least_squares(&dfs, &f) {
                    for (k, gk) in gamma.iter().enumerate() {
                        next.iter_mut().zip(&dgs[k]).for_each(|(a, d)| *a -= gk * d);
                    }
                }
            }
            if next.iter().any(|v| !v.is_finite()) {
                next = g.clone();
                dfs.clear();
                dgs.clear();
            }
            let g_next = flat(&self.update(z, &unflat(&next)));
            let f_next: Vec<f64> = g_next.iter().zip(&next).map(|(a, b)| a - b).collect();
            let res_next = residual(&f_next);
            iterations += 1;
            if res_next.is_nan() || res_next >= 10.0 * res {
                // Restart from a plain step.
                dfs.clear();
                dgs.clear();
            } else {
                dfs.push(f_next.iter().zip(&f).map(|(a, b)| a - b).collect());
                dgs.push(g_next.iter().zip(&g).map(|(a, b)| a - b).collect());
                if dfs.len() > depth {
                    dfs.remove(0);
                    dgs.remove(0);
                }
            }
            (x, g, f, res) = (next, g_next, f_next, res_next);
        }
        let _ = x;
        Solution { sigma: unflat(&g), iterations, residual: res }
    }
}

/// `γ` minimising `‖f − Σ_k γ_k df_k‖` (normal equations, lightly regularised), or `None` if
/// singular.
fn least_squares(dfs: &[Vec<f64>], f: &[f64]) -> Option<Vec<f64>> {
    let m = dfs.len();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let mut a = vec![0.0; m * m];
    let mut b = vec![0.0; m];
    for i in 0..m {
        b[i] = dot(&dfs[i], f);
        for j in 0..m {
            a[i * m + j] = dot(&dfs[i], &dfs[j]);
        }
    }
    let trace: f64 = (0..m).map(|i| a[i * m + i]).sum();
    if trace.is_nan() || trace <= 0.0 {
        return None;
    }
    for i in 0..m {
        a[i * m + i] += 1e-12 * trace / m as f64;
    }
    // Gaussian elimination with partial pivoting.
    for col in 0..m {
        let piv = (col..m).max_by(|&r, &s| a[r * m + col].abs().total_cmp(&a[s * m + col].abs()))?;
        if a[piv * m + col].abs() < 1e-300 {
            return None;
        }
        for j in 0..m {
            a.swap(col * m + j, piv * m + j);
        }
        b.swap(col, piv);
        for r in col + 1..m {
            let factor = a[r * m + col] / a[col * m + col];
            for j in col..m {
                a[r * m + j] -= factor * a[col * m + j];
            }
            b[r] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; m];
    for i in (0..m).rev() {
        let s: f64 = (i + 1..m).map(|j| a[i * m + j] * x[j]).sum();
        x[i] = (b[i] - s) / a[i * m + i];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// Blocks up to this size are inverted with [`gauss_jordan_inverse`] (no allocation).
const SMALL: usize = 16;

/// k points per parallel task in [`Scba::local_green`].
const K_CHUNK: usize = 256;

/// Inverse of the column-major `n × n` matrix `a` (destroyed) into `inv`, by Gauss–Jordan
/// elimination with partial pivoting.
fn gauss_jordan_inverse(n: usize, a: &mut [c64], inv: &mut [c64]) {
    inv.iter_mut()
        .enumerate()
        .for_each(|(k, x)| *x = if k % (n + 1) == 0 { c64::new(1.0, 0.0) } else { c64::new(0.0, 0.0) });
    for col in 0..n {
        let pivot =
            (col..n).max_by(|&r, &s| a[r + n * col].norm().total_cmp(&a[s + n * col].norm())).unwrap();
        if pivot != col {
            for j in 0..n {
                a.swap(pivot + n * j, col + n * j);
                inv.swap(pivot + n * j, col + n * j);
            }
        }
        let d = c64::new(1.0, 0.0) / a[col + n * col];
        for j in 0..n {
            a[col + n * j] *= d;
            inv[col + n * j] *= d;
        }
        for r in 0..n {
            if r != col {
                let f = a[r + n * col];
                if f != c64::new(0.0, 0.0) {
                    for j in 0..n {
                        let (ac, ic) = (a[col + n * j], inv[col + n * j]);
                        a[r + n * j] -= f * ac;
                        inv[r + n * j] -= f * ic;
                    }
                }
            }
        }
    }
}

/// Disorder seen by the SCBA: the mean field `⟨V⟩` and the vertices `(w_t, V_t)` of the second
/// cumulant (see the crate documentation).
#[derive(Clone, Debug)]
pub struct Disorder {
    pub mean_field: CMat,
    pub vertices: Vec<(f64, CMat)>,
}

impl Disorder {
    /// Anderson disorder: on-site energies uniform in `[−w/2, w/2]` on every site, the same on
    /// all `orbitals` of a site.
    pub fn anderson(w: f64, orbitals: usize) -> Self {
        let (mean_field, vertices) = anderson(w, orbitals);
        Self { mean_field, vertices }
    }

    /// Classical magnetic impurities on a fraction `conc` of the sites, of exchange `j`, with unit
    /// moments drawn from `moments` (as [`caroli_disorder::MagneticImpurities`]) and coupled
    /// through the spin matrices `s = (S_x, S_y, S_z)` of the model (e.g.
    /// [`caroli_model::presets::bhz_spin`]).
    pub fn magnetic(conc: f64, j: f64, moments: Moments, s: &[CMat; 3]) -> Self {
        // Every distribution of `Moments` draws n_z uniformly in [a, 1] and a uniform azimuth.
        let a = match moments {
            Moments::Isotropic => -1.0,
            Moments::Cone { theta0 } => theta0.cos(),
            Moments::Polarized { mz } => 2.0 * mz - 1.0,
        };
        let (mz, nz2) = ((1.0 + a) / 2.0, (1.0 + a + a * a) / 3.0);
        let (mean_field, vertices) = magnetic(conc, j, mz, nz2, s);
        Self { mean_field, vertices }
    }

    /// Two independent disorders together: the mean fields add up, and so do the cumulants.
    pub fn and(mut self, other: Disorder) -> Self {
        self.mean_field += &other.mean_field;
        self.vertices.extend(other.vertices);
        self
    }
}

impl Scba {
    /// The SCBA of `model` with `disorder`, `H(k)` sampled on a uniform grid of the Brillouin
    /// zone with `grid[a]` points along reciprocal direction `a` (one size per lattice
    /// direction; the 2D grid is [`k_grid`]). The unit cell holds every sublattice, so the
    /// matrices of `disorder` act on all the orbitals of a cell.
    pub fn from_model(model: &Model, grid: &[usize], disorder: Disorder) -> Self {
        let dim = model.lattice().dim();
        assert_eq!(grid.len(), dim, "one grid size per lattice direction");
        let cell = Region::cuboid(model.lattice(), [1, 1, 1]);
        assert_eq!(
            disorder.mean_field.nrows(),
            cell.orbitals(),
            "disorder matrices of size {} for a cell of {} orbitals",
            disorder.mean_field.nrows(),
            cell.orbitals()
        );
        let periods: Vec<[i64; 3]> = (0..dim).map(|a| std::array::from_fn(|c| (a == c) as i64)).collect();
        let points: Vec<Vec<f64>> = match dim {
            1 => k_grid(grid[0], 1).into_iter().map(|k| vec![k[0]]).collect(),
            2 => k_grid(grid[0], grid[1]).into_iter().map(|k| k.to_vec()).collect(),
            _ => {
                let plane = k_grid(grid[0], grid[1]);
                k_grid(grid[2], 1)
                    .into_iter()
                    .flat_map(|kz| plane.iter().map(move |k| vec![k[0], k[1], kz[0]]))
                    .collect()
            }
        };
        let hk = points.par_iter().map(|k| model.bloch_hamiltonian(&cell, &periods, k)).collect();
        Self { hk, mean_field: disorder.mean_field, vertices: disorder.vertices }
    }

    /// Second-order self-energy `Σ₂` at `energy + iη`, from zero, by Anderson acceleration to a
    /// largest change of 1e-11 (at most 500 k sums).
    pub fn self_energy(&self, energy: f64, eta: f64) -> Solution {
        let n = self.mean_field.nrows();
        self.solve_anderson(c64::new(energy, eta), &CMat::zeros(n, n), 8, 1e-11, 500)
    }

    /// [`Self::self_energy`] at each energy, each solution starting from the previous one.
    pub fn sweep(&self, energies: &[f64], eta: f64) -> Vec<Solution> {
        let n = self.mean_field.nrows();
        let mut guess = CMat::zeros(n, n);
        energies
            .iter()
            .map(|&e| {
                let mut sol = self.solve_anderson(c64::new(e, eta), &guess, 8, 1e-11, 500);
                if sol.residual.is_nan() || sol.residual > 1e-11 {
                    // The previous point can lead astray across a band edge: start again from zero.
                    let fresh = self.self_energy(e, eta);
                    if fresh.residual < sol.residual || sol.residual.is_nan() {
                        sol = fresh;
                    }
                }
                if sol.residual.is_finite() {
                    guess = sol.sigma.clone();
                }
                sol
            })
            .collect()
    }

    /// The whole self-energy `Σ_MF + Σ₂`.
    pub fn total_self_energy(&self, sigma: &CMat) -> CMat {
        &self.mean_field + sigma
    }

    /// Averaged density of states per unit cell, `−Im Tr ⟨G⟩(E + iη) / π`, with the self-energy
    /// `Σ₂ = sigma` (zero for the clean model).
    pub fn dos(&self, energy: f64, eta: f64, sigma: &CMat) -> f64 {
        let g = self.local_green(c64::new(energy, eta), sigma);
        -(0..g.nrows()).map(|i| g[(i, i)].im).sum::<f64>() / std::f64::consts::PI
    }
}

/// `p a + (1 − p) b`.
pub fn mix(a: &CMat, b: &CMat, p: f64) -> CMat {
    CMat::from_fn(a.nrows(), a.ncols(), |i, j| {
        c64::new(p, 0.0) * a[(i, j)] + c64::new(1.0 - p, 0.0) * b[(i, j)]
    })
}

/// Largest `|a_ij − b_ij|`; NaN if any difference is NaN (a failed update must not look
/// converged).
pub fn max_abs_diff(a: &CMat, b: &CMat) -> f64 {
    let mut m: f64 = 0.0;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            let d = (a[(i, j)] - b[(i, j)]).norm();
            if d.is_nan() {
                return f64::NAN;
            }
            m = m.max(d);
        }
    }
    m
}

/// Anderson disorder: uniform on-site energies in `[−w/2, w/2]` on every site, variance `w²/12`,
/// no mean field.
pub fn anderson(w: f64, orbitals: usize) -> (CMat, Vec<(f64, CMat)>) {
    let id = CMat::from_fn(
        orbitals,
        orbitals,
        |i, j| if i == j { c64::new(1.0, 0.0) } else { c64::new(0.0, 0.0) },
    );
    (CMat::zeros(orbitals, orbitals), vec![(w * w / 12.0, id)])
}

/// Classical magnetic impurities of exchange `j` on a fraction `conc` of the sites, with moments
/// of mean `⟨n_z⟩ = mz` and second moments `⟨n_z²⟩ = nz2`, `⟨n_x²⟩ = ⟨n_y²⟩ = (1 − nz2)/2`
/// (axisymmetric about z), coupled through the spin matrices `s = (S_x, S_y, S_z)`.
///
/// Second cumulant of `V = p J n·S` with a Bernoulli occupation `p` of mean `conc`:
/// `conc J² [⟨n_a n_b⟩ − conc ⟨n_a⟩⟨n_b⟩] S_a ⊗ S_b`. For moments uniform in `n_z ∈ [2 mz − 1, 1]`
/// (the thesis distribution), `nz2 = mz² + (1 − mz)²/3`.
pub fn magnetic(conc: f64, j: f64, mz: f64, nz2: f64, s: &[CMat; 3]) -> (CMat, Vec<(f64, CMat)>) {
    let n = s[2].nrows();
    let mf = CMat::from_fn(n, n, |a, b| c64::new(conc * j * mz, 0.0) * s[2][(a, b)]);
    let w = conc * j * j;
    let perp = (1.0 - nz2) / 2.0;
    let vertices =
        vec![(w * (nz2 - conc * mz * mz), s[2].clone()), (w * perp, s[0].clone()), (w * perp, s[1].clone())];
    (mf, vertices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_inverse() {
        let n = 4;
        let m = CMat::from_fn(n, n, |i, j| {
            c64::new(
                ((i * 7 + j * 3) % 5) as f64 + if i == j { 3.0 } else { 0.0 },
                ((i + 2 * j) % 3) as f64 - 1.0,
            )
        });
        let mut a: Vec<c64> = (0..n * n).map(|k| m[(k % n, k / n)]).collect();
        let mut inv = vec![c64::new(0.0, 0.0); n * n];
        gauss_jordan_inverse(n, &mut a, &mut inv);
        let g = CMat::from_fn(n, n, |i, j| inv[i + n * j]);
        let id = &m * &g;
        for i in 0..n {
            for j in 0..n {
                let e = if i == j { 1.0 } else { 0.0 };
                assert!((id[(i, j)] - c64::new(e, 0.0)).norm() < 1e-13);
            }
        }
    }

    #[test]
    fn grid_order_and_extent() {
        let g = k_grid(4, 2);
        assert_eq!(g.len(), 8);
        let pi = std::f64::consts::PI;
        assert_eq!(g[0], [-pi, -pi]);
        assert_eq!(g[2][0], 0.0);
        assert!((g[3][0] - pi / 2.0).abs() < 1e-15);
        assert_eq!(g[4][1], 0.0);
        // Odd sizes: symmetric inside (−π, π); a single point is k = 0.
        let odd: Vec<f64> = k_grid(5, 1).iter().map(|k| k[0] / pi).collect();
        assert!(odd.iter().zip([-0.8, -0.4, 0.0, 0.4, 0.8]).all(|(a, b)| (a - b).abs() < 1e-15));
        assert_eq!(k_grid(1, 1), vec![[0.0, 0.0]]);
    }

    /// Anderson acceleration reaches the fixed point of plain mixing in far fewer steps
    /// (BHZ with magnetic disorder, a matrix problem).
    #[test]
    fn anderson_matches_mixing() {
        use caroli_model::presets::{BhzParams, GFactors, bhz, bhz_spin};
        let model = bhz(BhzParams::default());
        let cell = caroli_lattice::Region::cuboid(model.lattice(), [1, 1, 0]);
        let hk = k_grid(32, 16)
            .iter()
            .map(|k| model.bloch_hamiltonian(&cell, &[[1, 0, 0], [0, 1, 0]], k))
            .collect();
        let (mz, conc, j) = (0.8, 0.3, 0.9);
        let (mean_field, vertices) =
            magnetic(conc, j, mz, mz * mz + (1.0 - mz).powi(2) / 3.0, &bhz_spin(GFactors::Opposite));
        let scba = Scba { hk, mean_field, vertices };
        let z = c64::new(0.05, 1e-3);
        let zero = CMat::zeros(4, 4);
        let slow = scba.solve(z, &zero, 0.3, 1e-12, 20000);
        let fast = scba.solve_anderson(z, &zero, 6, 1e-12, 200);
        assert!(fast.residual < 1e-12 && slow.residual < 1e-12);
        assert!(max_abs_diff(&slow.sigma, &fast.sigma) < 1e-10);
        assert!(fast.iterations * 5 < slow.iterations, "{} vs {}", fast.iterations, slow.iterations);
    }

    /// A singular local Green's function (real z on an eigenvalue) gives a NaN residual, not a
    /// converged solution.
    #[test]
    fn nan_is_not_convergence() {
        let one = |x: c64| CMat::from_fn(1, 1, |_, _| x);
        let scba = Scba {
            hk: vec![one(c64::new(0.0, 0.0)); 4],
            mean_field: one(c64::new(0.0, 0.0)),
            vertices: vec![(0.3, one(c64::new(1.0, 0.0)))],
        };
        let zero = one(c64::new(0.0, 0.0));
        assert!(scba.solve(c64::new(0.0, 0.0), &zero, 0.5, 1e-12, 50).residual.is_nan());
        assert!(scba.solve_anderson(c64::new(0.0, 0.0), &zero, 4, 1e-12, 50).residual.is_nan());
    }

    /// Single band, constant H(k) = 0: ⟨G⟩ = 1/(z − Σ) and the SCBA equation
    /// Σ = w/(z − Σ) has the closed form Σ = (z − sqrt(z² − 4w))/2 (retarded branch).
    #[test]
    fn scalar_fixed_point() {
        let one = |x: c64| CMat::from_fn(1, 1, |_, _| x);
        let w = 0.3;
        let scba = Scba {
            hk: vec![one(c64::new(0.0, 0.0)); 3],
            mean_field: one(c64::new(0.0, 0.0)),
            vertices: vec![(w, one(c64::new(1.0, 0.0)))],
        };
        let z = c64::new(0.4, 1e-3);
        let sol = scba.solve(z, &one(c64::new(0.0, -0.1)), 0.5, 1e-13, 2000);
        let exact = (z - (z * z - c64::new(4.0 * w, 0.0)).sqrt()) / 2.0;
        assert!((sol.sigma[(0, 0)] - exact).norm() < 1e-10, "{} vs {exact}", sol.sigma[(0, 0)]);
        assert!(sol.sigma[(0, 0)].im < 0.0);
    }
}
