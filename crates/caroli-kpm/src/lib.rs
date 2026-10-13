//! Kernel polynomial method (Weiße, Wellein, Alvermann & Fehske, Rev. Mod. Phys. 78, 275 (2006)).
//!
//! For a Hamiltonian rescaled to `H̃ = (H − e0)/ΔE` with spectrum inside (−1, 1), the moments
//! `μ_n = g_n ⟨r| T_n(H̃) |r⟩` (Jackson kernel `g_n`) give the spectral density at `x ∈ (−1, 1)`
//! as `ρ(x) = [μ_0 + 2 Σ_{n≥1} μ_n T_n(x)] / (π sqrt(1 − x²))`. With random-phase vectors `|r⟩`
//! this estimates the density of states (stochastic trace); with a site orbital it gives the
//! local density of states.

pub mod accelerated;
pub mod kubo;
pub mod spectrum;

pub use spectrum::Spectrum;

use caroli_disorder::rng::Xoshiro256;
use caroli_model::c64;
use rayon::prelude::*;

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

/// Random-phase vector `r_j = e^{2πiφ_j}` (`⟨r|r⟩ = n`): `⟨r|A|r⟩` is an unbiased estimate of
/// `Tr A`.
pub fn random_phases(rng: &mut Xoshiro256, n: usize) -> Vec<c64> {
    (0..n).map(|_| c64::from_polar(1.0, 2.0 * std::f64::consts::PI * rng.uniform())).collect()
}

/// Interleaves vectors of equal length into one block: `x[row·m + v] = vectors[v][row]`.
pub fn interleave(vectors: &[Vec<c64>]) -> Vec<c64> {
    let (m, n) = (vectors.len(), vectors.first().map_or(0, Vec::len));
    assert!(vectors.iter().all(|v| v.len() == n), "vectors of different lengths");
    (0..n * m).map(|i| vectors[i % m][i / m]).collect()
}

/// Inverse of [`interleave`] for a block of `m` vectors.
pub fn deinterleave(x: &[c64], m: usize) -> Vec<Vec<c64>> {
    (0..m).map(|v| x.iter().skip(v).step_by(m).copied().collect()).collect()
}

/// Entries per parallel task for the elementwise work and the sums.
const CHUNK: usize = 1 << 14;

/// Stochastic moments `μ_n = kernel_n · mean_v ⟨r_v| T_n(H̃) |r_v⟩` for a block of start
/// vectors. `apply(x, y)` sets `y = H̃ x` for the whole block stored interleaved
/// (`x[row·m + v]`, see [`interleave`]), e.g. `SparseMatrix::apply_block`.
///
/// Uses `T_m T_n = (T_{m+n} + T_{|m−n|})/2`, i.e. `μ_{2n} = 2⟨T_n|T_n⟩ − μ_0` and
/// `μ_{2n+1} = 2⟨T_n|T_{n+1}⟩ − μ_1`: `⌈M/2⌉` products with `H̃` give `M` moments (half of
/// [`moments`]). The vector work runs in parallel in fixed chunks, so the result does not depend
/// on the number of threads.
pub fn block_moments(apply: impl Fn(&[c64], &mut [c64]), starts: &[Vec<c64>], kernel: &[f64]) -> Vec<c64> {
    let (m, count) = (starts.len(), kernel.len());
    assert!(m > 0 && count > 0);
    let mut prev = interleave(starts);
    let mut cur = vec![c64::new(0.0, 0.0); prev.len()];
    apply(&prev, &mut cur);
    let pdot = |a: &[c64], b: &[c64]| -> c64 {
        a.par_chunks(CHUNK).zip(b.par_chunks(CHUNK)).map(|(x, y)| dot(x, y)).collect::<Vec<_>>().iter().sum()
    };
    let steps = count.div_ceil(2);
    // s1[n] = ⟨T_n|T_n⟩, s2[n] = ⟨T_n|T_{n+1}⟩ (summed over the block).
    let (mut s1, mut s2) = (vec![pdot(&prev, &prev)], vec![pdot(&prev, &cur)]);
    let mut next = vec![c64::new(0.0, 0.0); prev.len()];
    for _ in 1..steps {
        apply(&cur, &mut next);
        let parts: Vec<(c64, c64)> = next
            .par_chunks_mut(CHUNK)
            .zip(prev.par_chunks(CHUNK))
            .zip(cur.par_chunks(CHUNK))
            .map(|((nx, pv), cu)| {
                let (mut a, mut b) = (c64::new(0.0, 0.0), c64::new(0.0, 0.0));
                for ((x, p), c) in nx.iter_mut().zip(pv).zip(cu) {
                    *x = 2.0 * *x - p;
                    a += c.conj() * c;
                    b += c.conj() * *x;
                }
                (a, b)
            })
            .collect();
        s1.push(parts.iter().map(|p| p.0).sum());
        s2.push(parts.iter().map(|p| p.1).sum());
        std::mem::swap(&mut prev, &mut cur);
        std::mem::swap(&mut cur, &mut next);
    }
    let (mu0, mu1) = (s1[0], s2[0]);
    (0..count)
        .map(|k| {
            let raw = match k {
                0 => mu0,
                1 => mu1,
                _ if k % 2 == 0 => 2.0 * s1[k / 2] - mu0,
                _ => 2.0 * s2[k / 2] - mu1,
            };
            raw * (kernel[k] / m as f64)
        })
        .collect()
}

/// `μ_0 + 2 Σ_{n≥1} μ_n T_n(x)` at every `x`.
pub fn chebyshev_series(mu: &[c64], x: &[f64]) -> Vec<c64> {
    assert!(!mu.is_empty(), "no moments");
    x.iter()
        .map(|&e| {
            let mut acc = mu[0] + mu.get(1).map_or(c64::new(0.0, 0.0), |m| m * (2.0 * e));
            let (mut tm2, mut tm1) = (1.0, e);
            for &m in mu.iter().skip(2) {
                let t = 2.0 * e * tm1 - tm2;
                acc += m * (2.0 * t);
                tm2 = tm1;
                tm1 = t;
            }
            acc
        })
        .collect()
}

/// Density at `x` from the series: `Re[series(x)] / (π sqrt(1 − x²))` (`x` strictly inside
/// (−1, 1)).
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

    /// The stochastic trace of a diagonal matrix is exact with random phases, and the doubling
    /// relations give the moments of the plain recursion.
    #[test]
    fn block_moments_of_diagonal() {
        let eps = [0.1, -0.4, 0.75];
        let mut rng = Xoshiro256::seed_from_u64(7);
        let starts: Vec<Vec<c64>> = (0..4).map(|_| random_phases(&mut rng, 3)).collect();
        let kernel = jackson(9);
        let m = starts.len();
        let apply_block = |x: &[c64], y: &mut [c64]| {
            y.iter_mut().zip(x).enumerate().for_each(|(i, (o, v))| *o = v * eps[i / m]);
        };
        let mu = block_moments(apply_block, &starts, &kernel);
        let t2: f64 = eps.iter().map(|e| 2.0 * e * e - 1.0).sum();
        assert!((mu[0].re - 3.0).abs() < 1e-14 && (mu[2].re - kernel[2] * t2).abs() < 1e-14);
        let apply =
            |x: &[c64], y: &mut [c64]| y.iter_mut().zip(x).zip(&eps).for_each(|((o, v), e)| *o = v * *e);
        for (k, mk) in mu.iter().enumerate() {
            let plain: c64 = starts.iter().map(|s| moments(apply, s, &kernel)[k]).sum::<c64>() / m as f64;
            assert!((mk - plain).norm() < 1e-14, "moment {k}");
        }
        assert_eq!(deinterleave(&interleave(&starts), m), starts);
    }

    #[test]
    fn jackson_endpoints() {
        let g = jackson(64);
        assert!((g[0] - 1.0).abs() < 1e-15);
        assert!(g.windows(2).all(|w| w[1] <= w[0] + 1e-15));
    }
}
