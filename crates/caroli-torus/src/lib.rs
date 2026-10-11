//! Tight-binding Hamiltonians on an `n × n` torus (periodic 2D supercell of a one-cell lattice):
//! the translation-invariant part is diagonal in k and applied through 2D FFTs, local terms
//! (disorder, potentials) are applied in real space. This is the operator of the kernel
//! polynomial method and of split-operator time evolution.
//!
//! Wave functions are stored component by component, `ψ[x + n·y + n²·c]` for orbital `c`
//! (x fastest, as the thesis Fortran arrays `psi(N, N, N_spin)`).
//!
//! FFT conventions (those of FFTW, used by the thesis): forward `ψ̂(k) = Σ_x e^{−ik·x} ψ(x)`,
//! backward `Σ_k e^{+ik·x} ψ̂(k)`, both unnormalised. With these, the k-space multiplier of a
//! model with hoppings `H_{0,n} = ⟨x| H |x+n⟩` is `M(k) = Σ_n H_{0,n} e^{+ik·n}`
//! ([`kinetic_from_model`]). k indices are in FFT order: `k_i = 2π i/n` for `i < n/2`, else
//! `2π (i − n)/n`.

use caroli_lattice::Region;
use caroli_model::{CMat, Model, c64};
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// k value of FFT index `i` on a ring of `n` sites.
pub fn k_value(i: usize, n: usize) -> f64 {
    let i = if i < n / 2 { i as f64 } else { i as f64 - n as f64 };
    2.0 * std::f64::consts::PI * i / n as f64
}

/// 2D FFTs of `n × n` arrays stored x fastest.
#[derive(Clone)]
pub struct Fft2 {
    n: usize,
    forward: Arc<dyn Fft<f64>>,
    backward: Arc<dyn Fft<f64>>,
}

impl std::fmt::Debug for Fft2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Fft2({0}×{0})", self.n)
    }
}

impl Fft2 {
    pub fn new(n: usize) -> Self {
        let mut planner = FftPlanner::new();
        Self { n, forward: planner.plan_fft_forward(n), backward: planner.plan_fft_inverse(n) }
    }

    fn run(&self, plan: &Arc<dyn Fft<f64>>, data: &mut [c64]) {
        let n = self.n;
        // Along x: contiguous rows.
        for row in data.chunks_exact_mut(n) {
            plan.process(row);
        }
        // Along y: gather each column.
        let mut col = vec![c64::new(0.0, 0.0); n];
        for x in 0..n {
            for y in 0..n {
                col[y] = data[x + n * y];
            }
            plan.process(&mut col);
            for y in 0..n {
                data[x + n * y] = col[y];
            }
        }
    }

    /// Forward transform in place (unnormalised).
    pub fn forward(&self, data: &mut [c64]) {
        self.run(&self.forward, data);
    }

    /// Backward transform in place (unnormalised: forward then backward multiplies by n²).
    pub fn backward(&self, data: &mut [c64]) {
        self.run(&self.backward, data);
    }
}

/// `M(k) = Σ_n H_{0,n} e^{+ik·n}` on the `n × n` k grid (index `kx + n·ky`, FFT order) for a
/// model whose lattice has a single site per cell.
pub fn kinetic_from_model(model: &Model, n: usize) -> Vec<CMat> {
    let cell = Region::cuboid(model.lattice(), [1, 1, 0]);
    (0..n * n)
        .map(|k| {
            model.bloch_hamiltonian(&cell, &[[1, 0, 0], [0, 1, 0]], &[k_value(k % n, n), k_value(k / n, n)])
        })
        .collect()
}

/// `H = M(k) + V(x)` on an `n × n` torus with `comps` orbitals per site.
#[derive(Clone, Debug)]
pub struct TorusOperator {
    pub n: usize,
    pub comps: usize,
    /// k-space multiplier per k index (`kx + n·ky`).
    pub kinetic: Vec<CMat>,
    /// Local matrices on sites `x + n·y`.
    pub local: Vec<(usize, CMat)>,
    fft: Fft2,
}

impl TorusOperator {
    pub fn new(n: usize, comps: usize, kinetic: Vec<CMat>, local: Vec<(usize, CMat)>) -> Self {
        assert_eq!(kinetic.len(), n * n);
        Self { n, comps, kinetic, local, fft: Fft2::new(n) }
    }

    /// Number of entries of a wave function.
    pub fn len(&self) -> usize {
        self.n * self.n * self.comps
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn fft(&self) -> &Fft2 {
        &self.fft
    }

    /// Forward FFT of every component.
    pub fn to_k(&self, psi: &mut [c64]) {
        psi.chunks_exact_mut(self.n * self.n).for_each(|c| self.fft.forward(c));
    }

    /// Backward FFT of every component, divided by n² (inverse of [`Self::to_k`]).
    pub fn to_x(&self, psi: &mut [c64]) {
        let norm = (self.n * self.n) as f64;
        psi.chunks_exact_mut(self.n * self.n).for_each(|c| {
            self.fft.backward(c);
            c.iter_mut().for_each(|z| *z /= norm);
        });
    }

    /// `out = H ψ`: `to_x(M(k) to_k(ψ))` plus the local terms.
    pub fn apply(&self, psi: &[c64], out: &mut [c64]) {
        let (nn, m) = (self.n * self.n, self.comps);
        let mut k = psi.to_vec();
        self.to_k(&mut k);
        out.iter_mut().for_each(|z| *z = c64::new(0.0, 0.0));
        for (idx, mk) in self.kinetic.iter().enumerate() {
            for j in 0..m {
                let mut acc = c64::new(0.0, 0.0);
                for i in 0..m {
                    acc += mk[(j, i)] * k[idx + nn * i];
                }
                out[idx + nn * j] = acc;
            }
        }
        self.to_x(out);
        for (site, v) in &self.local {
            for j in 0..m {
                let mut acc = c64::new(0.0, 0.0);
                for i in 0..m {
                    acc += v[(j, i)] * psi[site + nn * i];
                }
                out[site + nn * j] += acc;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use caroli_lattice::{Lattice, Site};

    /// FFTs agree with the direct sums and backward(forward(ψ)) = n² ψ.
    #[test]
    fn fft_conventions() {
        let n = 6;
        let f = Fft2::new(n);
        let psi: Vec<c64> =
            (0..n * n).map(|i| c64::new((i as f64 * 0.37).sin(), (i as f64 * 0.11).cos())).collect();
        let mut a = psi.clone();
        f.forward(&mut a);
        let (kx, ky) = (2, 5);
        let direct: c64 = (0..n * n)
            .map(|i| {
                let (x, y) = ((i % n) as f64, (i / n) as f64);
                psi[i] * c64::from_polar(1.0, -(k_value(kx, n) * x + k_value(ky, n) * y))
            })
            .sum();
        assert!((a[kx + n * ky] - direct).norm() < 1e-12);
        f.backward(&mut a);
        assert!(a.iter().zip(&psi).all(|(u, v)| (u / (n * n) as f64 - v).norm() < 1e-13));
    }

    /// The FFT operator equals the real-space Hamiltonian of the periodic supercell.
    #[test]
    fn matches_real_space_hamiltonian() {
        let n = 4;
        let lat = Lattice::square(1.0, 2);
        let t = CMat::from_fn(2, 2, |i, j| c64::new(-1.0 + 0.3 * i as f64, 0.2 * j as f64 - 0.1));
        let e = CMat::from_fn(2, 2, |i, j| {
            if i == j { c64::new(0.5 - i as f64, 0.0) } else { c64::new(0.2, if i < j { 0.1 } else { -0.1 }) }
        });
        let model = Model::new(lat.clone()).onsite(0, e).hopping(0, 0, [1, 0, 0], t.clone()).hopping(
            0,
            0,
            [0, 1, 0],
            t.adjoint().to_owned(),
        );
        let local = vec![(5, CMat::from_fn(2, 2, |i, j| c64::new(if i == j { 0.7 } else { 0.0 }, 0.0)))];
        let op = TorusOperator::new(n, 2, kinetic_from_model(&model, n), local.clone());
        // Real space: the n × n supercell repeated in x and y.
        let cells = Region::from_sites(
            &lat,
            (0..n as i64)
                .flat_map(|y| (0..n as i64).map(move |x| Site { cell: [x, y, 0], sub: 0 }))
                .collect(),
        );
        let mut h = model.bloch_hamiltonian(&cells, &[[n as i64, 0, 0], [0, n as i64, 0]], &[0.0, 0.0]);
        for (s, v) in &local {
            for i in 0..2 {
                for j in 0..2 {
                    h[(2 * s + i, 2 * s + j)] += v[(i, j)];
                }
            }
        }
        let psi: Vec<c64> =
            (0..2 * n * n).map(|i| c64::new((i as f64).cos(), (0.3 * i as f64).sin())).collect();
        let mut out = vec![c64::new(0.0, 0.0); psi.len()];
        op.apply(&psi, &mut out);
        // Region order: site s = x + n y, orbitals interleaved; torus order: component-major.
        for s in 0..n * n {
            for c in 0..2 {
                let direct: c64 = (0..n * n)
                    .flat_map(|s2| (0..2).map(move |c2| (s2, c2)))
                    .map(|(s2, c2)| h[(2 * s + c, 2 * s2 + c2)] * psi[s2 + n * n * c2])
                    .sum();
                assert!((out[s + n * n * c] - direct).norm() < 1e-12, "site {s} comp {c}");
            }
        }
    }
}
