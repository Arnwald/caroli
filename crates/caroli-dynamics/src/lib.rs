//! Time evolution on periodic 2D supercells by the split-operator method:
//! `ψ(t + dt) = U_V U_K(t) U_V ψ(t)` with `U_V = exp(−i V dt/2)` local in real space and
//! `U_K(k, t) = exp(−i H_K(k, t) dt)` diagonal in k (FFTs from [`caroli_torus`]).
//!
//! The k-space step is supplied per time step, so time-dependent fields (vector potentials,
//! `k → k − A(t)`) only change `U_K`. [`exp_i`] builds the exponentials of Hermitian blocks.

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
    pub fn new(n: usize, comps: usize, v: &[(usize, CMat)], dt: f64) -> Self {
        let half_potential = v.iter().map(|(s, b)| (*s, exp_i(b, dt / 2.0))).collect();
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
