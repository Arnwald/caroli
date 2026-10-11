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

use caroli_model::{CMat, c64};
use faer::linalg::solvers::DenseSolveCore;

/// Uniform k grid of `m × n` points `2π (i/m, j/n)`, in FFT order along each axis
/// (`0, …, π − δ, −π, …, −δ`), as the thesis `k_FBZ`; returned as `(kx, ky)` with `kx`
/// varying fastest.
pub fn k_grid(m: usize, n: usize) -> Vec<[f64; 2]> {
    let axis = |len: usize| -> Vec<f64> {
        let k0 = 2.0 * std::f64::consts::PI / len as f64;
        // k_FBZ lists (i − len) k0 for i = len/2..len−1, then i k0 for i = 0..len/2−1.
        (len / 2..len)
            .map(|i| (i as f64 - len as f64) * k0)
            .chain((0..len / 2).map(|i| i as f64 * k0))
            .collect()
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
    /// `⟨G⟩(z) = (1/N_k) Σ_k [z − H(k) − Σ_MF − Σ₂]⁻¹`.
    pub fn local_green(&self, z: c64, sigma: &CMat) -> CMat {
        let n = self.mean_field.nrows();
        let shift = &self.mean_field + sigma;
        let mut acc = vec![c64::new(0.0, 0.0); n * n];
        let mut a = vec![c64::new(0.0, 0.0); n * n];
        let mut inv = vec![c64::new(0.0, 0.0); n * n];
        for h in &self.hk {
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
    /// of an element is below `tol` (at most `max_iter` steps).
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

/// Blocks up to this size are inverted with [`gauss_jordan_inverse`] (no allocation).
const SMALL: usize = 16;

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

/// `p a + (1 − p) b`.
pub fn mix(a: &CMat, b: &CMat, p: f64) -> CMat {
    CMat::from_fn(a.nrows(), a.ncols(), |i, j| {
        c64::new(p, 0.0) * a[(i, j)] + c64::new(1.0 - p, 0.0) * b[(i, j)]
    })
}

/// Largest `|a_ij − b_ij|`.
pub fn max_abs_diff(a: &CMat, b: &CMat) -> f64 {
    let mut m: f64 = 0.0;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            m = m.max((a[(i, j)] - b[(i, j)]).norm());
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
