//! Time evolution on periodic 2D supercells by the split-operator method:
//! `ψ(t + dt) = U_V U_K(t) U_V ψ(t)` with `U_V = exp(−i V dt/2)` local in real space and
//! `U_K(k, t) = exp(−i H_K(k, t) dt)` diagonal in k (FFTs from [`caroli_torus`]).
//!
//! The k-space step is supplied per time step, so time-dependent fields (vector potentials,
//! `k → k − A(t)`) only change `U_K`. [`exp_i`] builds the exponentials of Hermitian blocks.
//!
//! [`Chebyshev`] propagates with any Hamiltonian given as a product `H̃ψ` (sparse matrices of
//! any geometry, GPU operators): `exp(−iH dt)` expanded in Chebyshev polynomials of the
//! rescaled `H̃ = (H − centre)/half_width`, exact to the requested tolerance for any `dt`.

pub mod accelerated;

use caroli_model::{CMat, c64};
use caroli_torus::Fft2;
use faer::Side;

/// `exp(−i h τ)` for a Hermitian block `h` (by diagonalisation).
pub fn exp_i(h: &CMat, tau: f64) -> CMat {
    let evd = h.self_adjoint_eigen(Side::Lower).expect("Hermitian eigenproblem");
    let (s, u) = (evd.S(), evd.U());
    let n = h.nrows();
    let phases: Vec<c64> = (0..n).map(|k| c64::from_polar(1.0, -s[k].re * tau)).collect();
    CMat::from_fn(n, n, |i, j| (0..n).map(|k| u[(i, k)] * phases[k] * u[(j, k)].conj()).sum())
}

/// Bessel functions `J_0(x) … J_{count−1}(x)` (Miller's backward recurrence, normalised by
/// `J_0 + 2 Σ J_2k = 1`).
pub fn bessel_j(x: f64, count: usize) -> Vec<f64> {
    if x == 0.0 {
        return (0..count).map(|n| if n == 0 { 1.0 } else { 0.0 }).collect();
    }
    let ax = x.abs();
    if ax < 1e-30 {
        // Leading term (x/2)^n / n!; the next one is smaller by (x/2)² < 1e-60.
        let mut term = 1.0;
        return (0..count)
            .map(|n| {
                if n > 0 {
                    term *= x / 2.0 / n as f64;
                }
                term
            })
            .collect();
    }
    let top = count.max(ax as usize) + 30 + (10.0 * ax.cbrt()) as usize;
    let mut j = vec![0.0; top + 2];
    j[top] = 1e-300;
    for n in (1..=top).rev() {
        j[n - 1] = 2.0 * n as f64 / ax * j[n] - j[n + 1];
        if j[n - 1].abs() > 1e250 {
            j.iter_mut().for_each(|v| *v *= 1e-250);
        }
    }
    let norm = j[0] + 2.0 * j.iter().skip(2).step_by(2).sum::<f64>();
    j.truncate(count);
    j.iter_mut().enumerate().for_each(|(n, v)| {
        *v /= norm;
        // J_n(−x) = (−1)^n J_n(x).
        if x < 0.0 && n % 2 == 1 {
            *v = -*v;
        }
    });
    j
}

/// Propagator `ψ ← exp(−iH dt) ψ` by the Chebyshev expansion
/// `exp(−iH dt) = e^{−i·centre·dt} Σ_n (2 − δ_n0) (−i)^n J_n(half_width·dt) T_n(H̃)`.
#[derive(Clone, Debug)]
pub struct Chebyshev {
    /// Expansion coefficients, the global phase included.
    pub coefficients: Vec<c64>,
}

impl Chebyshev {
    /// Terms up to `|J_n| < tol` (the spectrum of `H̃` must lie in [−1, 1]).
    pub fn new(dt: f64, centre: f64, half_width: f64, tol: f64) -> Self {
        let tau = half_width * dt;
        let guess = (1.3 * tau.abs()) as usize + 40 + (8.0 * tau.abs().cbrt()) as usize;
        let j = bessel_j(tau, guess);
        let last = j.iter().rposition(|v| v.abs() > tol).unwrap_or(0);
        let phase = c64::from_polar(1.0, -centre * dt);
        let coefficients = (0..=last.max(1))
            .map(|n| {
                let minus_i_n =
                    [c64::new(1.0, 0.0), c64::new(0.0, -1.0), c64::new(-1.0, 0.0), c64::new(0.0, 1.0)][n % 4];
                phase * minus_i_n * if n == 0 { j[0] } else { 2.0 * j[n] }
            })
            .collect();
        Self { coefficients }
    }

    /// One step: `ψ ← Σ_n c_n T_n(H̃) ψ` with `apply(x, y)` setting `y = H̃ x`. `ψ` may hold a
    /// block of wave functions if `apply` acts on the block (e.g. `SparseMatrix::apply_block`).
    pub fn step(&self, apply: impl Fn(&[c64], &mut [c64]), psi: &mut [c64]) {
        use rayon::prelude::*;
        const CHUNK: usize = 1 << 14;
        let c = &self.coefficients;
        let n = psi.len();
        let mut t0 = psi.to_vec();
        let mut t1 = vec![c64::new(0.0, 0.0); n];
        apply(&t0, &mut t1);
        psi.par_chunks_mut(CHUNK).zip(t0.par_chunks(CHUNK)).zip(t1.par_chunks(CHUNK)).for_each(
            |((p, a), b)| {
                for ((p, a), b) in p.iter_mut().zip(a).zip(b) {
                    *p = c[0] * a + c[1] * b;
                }
            },
        );
        let mut t2 = vec![c64::new(0.0, 0.0); n];
        for &ck in &c[2..] {
            apply(&t1, &mut t2);
            psi.par_chunks_mut(CHUNK).zip(t2.par_chunks_mut(CHUNK)).zip(t0.par_chunks(CHUNK)).for_each(
                |((p, next), prev)| {
                    for ((p, x), q) in p.iter_mut().zip(next.iter_mut()).zip(prev) {
                        *x = 2.0 * *x - q;
                        *p += ck * *x;
                    }
                },
            );
            // (T_{n−1}, T_n) ← (T_n, T_{n+1}).
            std::mem::swap(&mut t0, &mut t1);
            std::mem::swap(&mut t1, &mut t2);
        }
    }
}

/// A time-dependent Hamiltonian for [`magnus4_step`].
pub trait TimeDependent {
    /// `y = Σ_i w_i H(t_i) x` for `stages = [(t_i, w_i)]`.
    fn apply(&self, stages: &[(f64, f64)], x: &[c64], y: &mut [c64]);
    /// Bounds `(lower, upper)` on the spectrum of `H(t)` at every `t`.
    fn bounds(&self) -> (f64, f64);
}

/// One step of the fourth-order commutator-free Magnus integrator (Blanes & Moan, 2006):
/// `U(t + dt, t) ≈ exp(−i dt B₂) exp(−i dt B₁)`, `B₁ = α₂ H(t₁) + α₁ H(t₂)`,
/// `B₂ = α₁ H(t₁) + α₂ H(t₂)`, `t₁,₂ = t + (1/2 ∓ √3/6) dt`, `α₁,₂ = (3 ∓ 2√3)/12`.
/// Each exponential is a [`Chebyshev`] expansion to `tol`: the global error is O(dt⁴), with no
/// splitting error between kinetic and potential terms.
pub fn magnus4_step(h: &impl TimeDependent, psi: &mut [c64], t: f64, dt: f64, tol: f64) {
    let s3 = 3f64.sqrt();
    let (t1, t2) = (t + (0.5 - s3 / 6.0) * dt, t + (0.5 + s3 / 6.0) * dt);
    let (a1, a2) = ((3.0 - 2.0 * s3) / 12.0, (3.0 + 2.0 * s3) / 12.0);
    let (lo, hi) = h.bounds();
    for stages in [[(t1, a2), (t2, a1)], [(t1, a1), (t2, a2)]] {
        // Spectrum of Σ w_i H(t_i) (a weight is negative).
        let (blo, bhi) = stages
            .iter()
            .fold((0.0, 0.0), |(l, u), &(_, w)| (l + (w * lo).min(w * hi), u + (w * lo).max(w * hi)));
        let (centre, half) = ((bhi + blo) / 2.0, ((bhi - blo) / 2.0 * 1.01).max(1e-300));
        let cheb = Chebyshev::new(dt, centre, half, tol);
        cheb.step(
            |x, y| {
                h.apply(&stages, x, y);
                y.iter_mut().zip(x).for_each(|(y, x)| *y = (*y - centre * x) / half);
            },
            psi,
        );
    }
}

/// Multiplies the wave function (`ψ[site + n²·c]`) by a block on each listed site.
pub fn apply_local(psi: &mut [c64], nn: usize, comps: usize, blocks: &[(usize, CMat)]) {
    let mut tmp = vec![c64::new(0.0, 0.0); comps];
    for (site, b) in blocks {
        for (j, t) in tmp.iter_mut().enumerate() {
            *t = (0..comps).map(|i| b[(j, i)] * psi[site + nn * i]).sum();
        }
        for (j, t) in tmp.iter().enumerate() {
            psi[site + nn * j] = *t;
        }
    }
}

/// Multiplies a k-space wave function by one block per k.
pub fn apply_k(psi_k: &mut [c64], comps: usize, blocks: &[CMat]) {
    let nn = blocks.len();
    let mut tmp = vec![c64::new(0.0, 0.0); comps];
    for (k, b) in blocks.iter().enumerate() {
        for (j, t) in tmp.iter_mut().enumerate() {
            *t = (0..comps).map(|i| b[(j, i)] * psi_k[k + nn * i]).sum();
        }
        for (j, t) in tmp.iter().enumerate() {
            psi_k[k + nn * j] = *t;
        }
    }
}

/// Split-operator propagator on an `n × n` torus with `comps` orbitals per site.
#[derive(Clone, Debug)]
pub struct SplitOperator {
    pub n: usize,
    pub comps: usize,
    /// `exp(−i V dt/2)` on the sites carrying a potential (identity elsewhere).
    pub half_potential: Vec<(usize, CMat)>,
    fft: Fft2,
}

impl SplitOperator {
    /// Propagator with the local potential `v` (site, Hermitian block) and time step `dt`.
    /// Blocks listed for the same site are summed before exponentiation.
    pub fn new(n: usize, comps: usize, v: &[(usize, CMat)], dt: f64) -> Self {
        let mut summed: std::collections::BTreeMap<usize, CMat> = std::collections::BTreeMap::new();
        for (s, b) in v {
            summed.entry(*s).and_modify(|m| *m += b).or_insert_with(|| b.clone());
        }
        let half_potential = summed.into_iter().map(|(s, b)| (s, exp_i(&b, dt / 2.0))).collect();
        Self { n, comps, half_potential, fft: Fft2::new(n) }
    }

    /// Same, with the half-step blocks given directly.
    pub fn with_half_steps(n: usize, comps: usize, half_potential: Vec<(usize, CMat)>) -> Self {
        Self { n, comps, half_potential, fft: Fft2::new(n) }
    }

    /// Forward FFT of every component (k space, unnormalised).
    pub fn to_k(&self, psi: &mut [c64]) {
        psi.chunks_exact_mut(self.n * self.n).for_each(|c| self.fft.forward(c));
    }

    /// Backward FFT of every component divided by n².
    pub fn to_x(&self, psi: &mut [c64]) {
        let norm = (self.n * self.n) as f64;
        psi.chunks_exact_mut(self.n * self.n).for_each(|c| {
            self.fft.backward(c);
            c.iter_mut().for_each(|z| *z /= norm);
        });
    }

    /// One step on a real-space wave function: `U_V`, FFT, `U_K`, inverse FFT, `U_V`.
    pub fn step(&self, psi_x: &mut [c64], kinetic_step: &[CMat]) {
        let nn = self.n * self.n;
        apply_local(psi_x, nn, self.comps, &self.half_potential);
        self.to_k(psi_x);
        apply_k(psi_x, self.comps, kinetic_step);
        self.to_x(psi_x);
        apply_local(psi_x, nn, self.comps, &self.half_potential);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fourth order: halving dt divides the error by about 16 (dense driven 6-level system,
    /// reference with dt/16).
    #[test]
    fn magnus4_is_fourth_order() {
        struct Driven(CMat, CMat);
        #[allow(clippy::needless_range_loop)] // dense products, indices read best
        impl TimeDependent for Driven {
            fn apply(&self, stages: &[(f64, f64)], x: &[c64], y: &mut [c64]) {
                let n = x.len();
                y.iter_mut().for_each(|v| *v = c64::new(0.0, 0.0));
                for &(t, w) in stages {
                    let f = (1.3 * t).cos();
                    for i in 0..n {
                        for j in 0..n {
                            y[i] += w * (self.0[(i, j)] + f * self.1[(i, j)]) * x[j];
                        }
                    }
                }
            }
            fn bounds(&self) -> (f64, f64) {
                let r = |m: &CMat| {
                    (0..m.nrows())
                        .map(|i| (0..m.ncols()).map(|j| m[(i, j)].norm()).sum::<f64>())
                        .fold(0.0, f64::max)
                };
                let b = r(&self.0) + r(&self.1);
                (-b, b)
            }
        }
        let n = 6;
        let herm = |seed: f64| {
            let a = CMat::from_fn(n, n, |i, j| {
                c64::new((seed * (i * 7 + j) as f64).sin(), (seed * (i + 3 * j) as f64).cos())
            });
            CMat::from_fn(n, n, |i, j| (a[(i, j)] + a[(j, i)].conj()) * 0.5)
        };
        let h = Driven(herm(0.37), herm(1.11));
        let psi0: Vec<c64> =
            (0..n).map(|i| c64::new(1.0 / (n as f64).sqrt(), 0.0) * c64::from_polar(1.0, i as f64)).collect();
        let evolve = |dt: f64| {
            let mut psi = psi0.clone();
            let steps = (2.0 / dt).round() as usize;
            for k in 0..steps {
                magnus4_step(&h, &mut psi, k as f64 * dt, dt, 1e-15);
            }
            psi
        };
        let reference = evolve(0.0125);
        let err =
            |dt: f64| evolve(dt).iter().zip(&reference).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>().sqrt();
        let (e1, e2) = (err(0.2), err(0.1));
        assert!(e1 / e2 > 12.0 && e1 / e2 < 20.0, "errors {e1:e}, {e2:e}");
        let norm: f64 = reference.iter().map(|z| z.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-13);
    }

    /// Blocks listed twice for a site act as their sum.
    #[test]
    fn split_operator_sums_duplicate_sites() {
        let m = |a: f64, b: f64| {
            CMat::from_fn(2, 2, |i, j| {
                c64::new(if i == j { a * if i == 0 { 1.0 } else { -1.0 } } else { b }, 0.0)
            })
        };
        let twice = SplitOperator::new(4, 2, &[(5, m(0.0, 0.8)), (5, m(0.6, 0.0))], 0.1);
        let once = SplitOperator::new(4, 2, &[(5, m(0.6, 0.8))], 0.1);
        assert_eq!(twice.half_potential.len(), 1);
        assert!((&twice.half_potential[0].1 - &once.half_potential[0].1).norm_max() < 1e-15);
    }

    #[test]
    fn bessel_values() {
        // J_0(1), J_1(1), J_5(1) and J_0(10), J_12(10) (Abramowitz & Stegun).
        let j = bessel_j(1.0, 6);
        assert!((j[0] - 0.765_197_686_557_966_6).abs() < 1e-15);
        assert!((j[1] - 0.440_050_585_744_933_5).abs() < 1e-15);
        assert!((j[5] - 2.497_577_302_112_344e-4).abs() < 1e-17);
        let tiny = bessel_j(1e-60, 3);
        assert!(
            (tiny[0] - 1.0).abs() < 1e-15
                && (tiny[1] - 5e-61).abs() < 1e-75
                && (tiny[2] - 1.25e-121).abs() < 1e-135
        );
        let j = bessel_j(10.0, 13);
        assert!((j[0] + 0.245_935_764_451_348_3).abs() < 1e-14);
        assert!((j[12] - 6.337_025_497_015_6e-2).abs() < 1e-13);
    }

    /// Chebyshev propagation of a BHZ flake against exact exponentiation, for a short and a
    /// long time step.
    #[test]
    fn chebyshev_matches_exact_exponential() {
        use caroli_lattice::Region;
        use caroli_model::presets::{BhzParams, bhz};
        let model = bhz(BhzParams { lambda: 0.2, ..BhzParams::default() });
        let region = Region::cuboid(model.lattice(), [5, 4, 0]);
        let h = model.sparse_hamiltonian(&region);
        let (lo, hi) = h.spectral_bounds();
        let (centre, half) = ((hi + lo) / 2.0, (hi - lo) / 2.0 * 1.01);
        let scaled = h.rescaled(centre, half);
        let dense = h.to_dense();
        let psi0: Vec<c64> =
            (0..h.n).map(|i| c64::new((0.7 * i as f64).cos(), (0.2 * i as f64).sin())).collect();
        for dt in [0.05, 7.5] {
            let mut psi = psi0.clone();
            Chebyshev::new(dt, centre, half, 1e-15).step(|x, y| scaled.apply(x, y), &mut psi);
            let u = exp_i(&dense, dt);
            let exact: Vec<c64> = (0..h.n).map(|r| (0..h.n).map(|c| u[(r, c)] * psi0[c]).sum()).collect();
            let err = psi.iter().zip(&exact).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max);
            assert!(err < 1e-12, "dt {dt}: {err:e}");
        }
    }

    /// exp(−i σ_x τ) = cos τ − i σ_x sin τ.
    #[test]
    fn exponential_of_pauli() {
        let sx = CMat::from_fn(2, 2, |i, j| if i != j { c64::new(1.0, 0.0) } else { c64::new(0.0, 0.0) });
        let u = exp_i(&sx, 0.7);
        assert!((u[(0, 0)] - c64::new(0.7f64.cos(), 0.0)).norm() < 1e-14);
        assert!((u[(0, 1)] - c64::new(0.0, -0.7f64.sin())).norm() < 1e-14);
    }

    /// A free particle: without potential, a step is exact and conserves the norm, and the
    /// k-space phases accumulate as exp(−i ε_k t).
    #[test]
    fn unitary_steps() {
        let n = 8;
        let prop = SplitOperator::new(n, 1, &[(3, CMat::from_fn(1, 1, |_, _| c64::new(0.4, 0.0)))], 0.05);
        let kin: Vec<CMat> = (0..n * n)
            .map(|k| exp_i(&CMat::from_fn(1, 1, |_, _| c64::new((k as f64 * 0.1).cos(), 0.0)), 0.05))
            .collect();
        let mut psi: Vec<c64> =
            (0..n * n).map(|i| c64::new((i as f64).sin(), (0.5 * i as f64).cos())).collect();
        let norm0: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
        for _ in 0..50 {
            prop.step(&mut psi, &kin);
        }
        let norm: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
        assert!((norm - norm0).abs() < 1e-11 * norm0);
    }
}
