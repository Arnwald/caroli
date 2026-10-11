//! Kernel polynomial method (Weiße, Wellein, Alvermann & Fehske, Rev. Mod. Phys. 78, 275 (2006)).
//!
//! For a Hamiltonian rescaled to `H̃ = (H − e0)/ΔE` with spectrum inside (−1, 1), the moments
//! `μ_n = g_n ⟨r| T_n(H̃) |r⟩` (Jackson kernel `g_n`) give the spectral density at `x ∈ (−1, 1)`
//! as `ρ(x) = [μ_0 + 2 Σ_{n≥1} μ_n T_n(x)] / (π sqrt(1 − x²))`. With random-phase vectors `|r⟩`
//! this estimates the density of states (stochastic trace); with a site orbital it gives the
//! local density of states.

use caroli_model::c64;

/// Jackson kernel `g_n`, `n = 0..m`.
pub fn jackson(m: usize) -> Vec<f64> {
    let mf = m as f64;
    let b1 = std::f64::consts::PI / (mf + 1.0);
    (0..m)
        .map(|n| {
            let bn = n as f64 * std::f64::consts::PI / (mf + 1.0);
            1.0 / (mf + 1.0) * ((mf - n as f64 + 1.0) * bn.cos() + 1.0 / b1.tan() * bn.sin())
        })
        .collect()
}

/// `⟨a|b⟩`.
pub fn dot(a: &[c64], b: &[c64]) -> c64 {
    a.iter().zip(b).map(|(x, y)| x.conj() * y).sum()
}

/// Moments `μ_n = kernel_n ⟨start| T_n(H̃) |start⟩` for `n = 0..kernel.len()`, with `apply` the
/// rescaled Hamiltonian (`out = H̃ ψ`). Chebyshev recursion `T_{n+1} = 2 H̃ T_n − T_{n−1}`.
pub fn moments(apply: impl Fn(&[c64], &mut [c64]), start: &[c64], kernel: &[f64]) -> Vec<c64> {
    let len = start.len();
    let mut s0 = start.to_vec();
    let mut s1 = vec![c64::new(0.0, 0.0); len];
    apply(&s0, &mut s1);
    let mut hs = vec![c64::new(0.0, 0.0); len];
    let mut mu = Vec::with_capacity(kernel.len());
    for &g in kernel {
        mu.push(g * dot(start, &s0));
        apply(&s1, &mut hs);
        for i in 0..len {
            let s2 = hs[i] * 2.0 - s0[i];
            s0[i] = s1[i];
            s1[i] = s2;
        }
    }
    mu
}

/// `μ_0 + 2 Σ_{n≥1} μ_n T_n(x)` at every `x`.
pub fn chebyshev_series(mu: &[c64], x: &[f64]) -> Vec<c64> {
    x.iter()
        .map(|&e| {
            let mut acc = mu[0] + mu[1] * (2.0 * e);
            let (mut tm2, mut tm1) = (1.0, e);
            for &m in &mu[2..] {
                let t = 2.0 * e * tm1 - tm2;
                acc += m * (2.0 * t);
                tm2 = tm1;
                tm1 = t;
            }
            acc
        })
        .collect()
}

/// Density at `x` from the series: `Re[series(x)] / (π sqrt(1 − x²))`.
pub fn density(mu: &[c64], x: &[f64]) -> Vec<f64> {
    chebyshev_series(mu, x)
        .iter()
        .zip(x)
        .map(|(s, e)| s.re / (std::f64::consts::PI * (1.0 - e * e).sqrt()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Diagonal "Hamiltonian" with known eigenvalues: the KPM density of a basis vector integrates
    /// to 1, and the moments are exactly g_n T_n(ε) for an eigenvector of energy ε.
    #[test]
    fn single_level() {
        let eps = 0.3;
        let m = 200;
        let kernel = jackson(m);
        let mu = moments(
            |v, out| out.iter_mut().zip(v).for_each(|(o, x)| *o = x * eps),
            &[c64::new(1.0, 0.0)],
            &kernel,
        );
        let t3 = 4.0 * eps.powi(3) - 3.0 * eps;
        assert!((mu[3].re - kernel[3] * t3).abs() < 1e-14);
        // Integral of the density over (−1, 1) is μ_0 = 1 (Gauss–Chebyshev quadrature).
        let k = 2000;
        let xs: Vec<f64> =
            (0..k).map(|j| ((j as f64 + 0.5) * std::f64::consts::PI / k as f64).cos()).collect();
        let rho = density(&mu, &xs);
        let integral: f64 = rho
            .iter()
            .zip(&xs)
            .map(|(r, x)| r * (1.0 - x * x).sqrt() * std::f64::consts::PI / k as f64)
            .sum();
        assert!((integral - 1.0).abs() < 1e-10);
        // The peak sits at ε.
        let peak = xs.iter().zip(&rho).max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        assert!((peak - eps).abs() < 0.01);
    }

    #[test]
    fn jackson_endpoints() {
        let g = jackson(64);
        assert!((g[0] - 1.0).abs() < 1e-15);
        assert!(g.windows(2).all(|w| w[1] <= w[0] + 1e-15));
    }
}
