//! Kubo–Bastin conductivity by the kernel polynomial method (García, Covaci and Rappoport,
//! Phys. Rev. Lett. 114, 116602 (2015)), as Kwant's `kpm.conductivity` and KITE.
//!
//! With the Hamiltonian rescaled to `H̃ = (H − b)/a` (spectrum inside (−1, 1)) and the
//! velocities `v_α = i[H, r_α]` (ħ = 1),
//!
//! ```text
//! σ_αβ(μ) = (4 e² / (π ħ Ω a²)) ∫_{−1}^{1} dε f(ε) Σ_nm Γ_nm(ε) μ_nm / (1 − ε²)²,
//! μ_nm = g_n g_m / ((1 + δ_n0)(1 + δ_m0)) Tr[v_α T_n(H̃) v_β T_m(H̃)],
//! Γ_nm(ε) = (ε − i n s) e^{i n θ} T_m(ε) + (ε + i m s) e^{−i m θ} T_n(ε),  ε = cos θ, s = sin θ,
//! ```
//!
//! with `Ω` the area (2D) or volume of the sample and `g_n` the kernel. The trace is a stochastic
//! average over random-phase vectors.

use caroli_dense::{Accum, Mat, Op, matmul};
use caroli_model::c64;

/// Moments `μ_nm` (`n, m < kernel.len()`), averaged over the start vectors (random phases, so that
/// `⟨r|A|r⟩` estimates `Tr A`). `h(x, y)` sets `y = H̃ x`; `va` and `vb` apply the velocities.
///
/// Memory: one `N × M` matrix of `v_β T_m(H̃) |r⟩` per start vector, plus columns of
/// `T_n(H̃) v_α |r⟩` in batches.
pub fn kubo_moments(
    h: impl Fn(&[c64], &mut [c64]),
    va: impl Fn(&[c64], &mut [c64]),
    vb: impl Fn(&[c64], &mut [c64]),
    starts: &[Vec<c64>],
    kernel: &[f64],
) -> Vec<Vec<c64>> {
    const BATCH: usize = 64;
    let m = kernel.len();
    assert!(m > 0 && !starts.is_empty());
    let n = starts[0].len();
    let one = c64::new(1.0, 0.0);
    let mut mu = Mat::<c64>::zeros(m, m);
    // Chebyshev vectors T_k(H̃) x for k = 0..m, column by column.
    let chebyshev = |x: &[c64], columns: &mut dyn FnMut(usize, &[c64])| {
        let mut prev = x.to_vec();
        let mut cur = vec![c64::new(0.0, 0.0); n];
        columns(0, &prev);
        if m == 1 {
            return;
        }
        h(&prev, &mut cur);
        columns(1, &cur);
        let mut next = vec![c64::new(0.0, 0.0); n];
        for k in 2..m {
            h(&cur, &mut next);
            for (x, p) in next.iter_mut().zip(&prev) {
                *x = 2.0 * *x - p;
            }
            columns(k, &next);
            std::mem::swap(&mut prev, &mut cur);
            std::mem::swap(&mut cur, &mut next);
        }
    };
    let mut buffer = vec![c64::new(0.0, 0.0); n];
    for r in starts {
        // R[:, k] = v_β T_k(H̃) |r⟩.
        let mut right = Mat::<c64>::zeros(n, m);
        chebyshev(r, &mut |k, t| {
            vb(t, &mut buffer);
            right.col_as_slice_mut(k).copy_from_slice(&buffer);
        });
        // L[:, k] = T_k(H̃) v_α |r⟩, in batches of columns: μ[rows] += L† R.
        let mut start = vec![c64::new(0.0, 0.0); n];
        va(r, &mut start);
        let mut left = Mat::<c64>::zeros(n, BATCH.min(m));
        let mut first = 0;
        chebyshev(&start, &mut |k, t| {
            let c = k - first;
            left.col_as_slice_mut(c).copy_from_slice(t);
            if c + 1 == left.ncols() || k + 1 == m {
                let rows = c + 1;
                let block = left.as_ref().subcols(0, rows);
                matmul(
                    mu.as_mut().subrows_mut(first, rows),
                    Accum::Add,
                    block,
                    Op::Adjoint,
                    right.as_ref(),
                    Op::Plain,
                    one,
                );
                first = k + 1;
            }
        });
    }
    let count = starts.len() as f64;
    (0..m)
        .map(|a| {
            (0..m)
                .map(|b| {
                    let weight = kernel[a] * kernel[b]
                        / ((1.0 + (a == 0) as u8 as f64) * (1.0 + (b == 0) as u8 as f64));
                    mu[(a, b)] * (weight / count)
                })
                .collect()
        })
        .collect()
}

/// Conductivity `σ_αβ(μ, T)` in units of `e²/h` at the chemical potentials `mus` and the
/// temperature `temperature` (unscaled energies; 0 for a step), from the moments of
/// [`kubo_moments`] for `H̃ = (H − center)/half_width` and a sample of area (volume) `omega`.
/// The energy integral runs on `grid` Chebyshev nodes.
pub fn kubo_bastin(
    mu: &[Vec<c64>],
    center: f64,
    half_width: f64,
    omega: f64,
    mus: &[f64],
    temperature: f64,
    grid: usize,
) -> Vec<f64> {
    let m = mu.len();
    let pi = std::f64::consts::PI;
    // The integrand Σ_nm Re[Γ_nm(ε) μ_nm] / (1 − ε²)² on the nodes ε_k = cos θ_k.
    let thetas: Vec<f64> = (0..grid).map(|k| pi * (k as f64 + 0.5) / grid as f64).collect();
    let integrand: Vec<f64> = thetas
        .iter()
        .map(|&theta| {
            let (eps, s) = (theta.cos(), theta.sin());
            let t: Vec<f64> = (0..m).map(|k| (k as f64 * theta).cos()).collect(); // T_k(ε)
            let mut acc = c64::new(0.0, 0.0);
            for (a, mu_a) in mu.iter().enumerate() {
                // Σ_m μ_am T_m and Σ_n μ_na T_n.
                let row: c64 = mu_a.iter().zip(&t).map(|(x, tb)| x * tb).sum();
                let col: c64 = mu.iter().zip(&t).map(|(mu_b, tb)| mu_b[a] * tb).sum();
                let phase = c64::from_polar(1.0, a as f64 * theta);
                let ka = a as f64;
                acc += c64::new(eps, -ka * s) * phase * row + c64::new(eps, ka * s) * phase.conj() * col;
            }
            acc.re / (s * s).powi(2)
        })
        .collect();
    let prefactor = 8.0 / (omega * half_width * half_width);
    // ∫ dε f(ε) F(ε) = ∫ dθ sin θ f(cos θ) F(cos θ), midpoint rule on the nodes.
    let weight = pi / grid as f64;
    mus.iter()
        .map(|&chemical| {
            let occupation = |theta: f64| {
                let energy = center + half_width * theta.cos();
                if temperature == 0.0 {
                    if energy <= chemical { 1.0 } else { 0.0 }
                } else {
                    let x = (energy - chemical) / temperature;
                    if x > 0.0 { (-x).exp() / (1.0 + (-x).exp()) } else { 1.0 / (1.0 + x.exp()) }
                }
            };
            prefactor
                * thetas
                    .iter()
                    .zip(&integrand)
                    .map(|(&theta, f)| occupation(theta) * theta.sin() * f * weight)
                    .sum::<f64>()
        })
        .collect()
}
