//! Observables of a walk state (Sellapillay & Verga, section 2.2): position and color
//! probabilities, edge spins, entanglement entropies (base-2 logarithm) and concurrence.

use crate::walk::SpinWalk;
use caroli_dense::{Op, hermitian_eigen, product};
use caroli_model::{CMat, c64};
use faer::MatRef;
use rayon::prelude::*;

/// `−Σ λ log₂ λ` over the eigenvalues of a density matrix.
pub fn von_neumann(rho: &CMat) -> f64 {
    let (w, _) = hermitian_eigen(rho.as_ref()).expect("eigenvalues of a density matrix");
    w.iter().filter(|&&l| l > 1e-300).map(|&l| -l * l.log2()).sum()
}

impl SpinWalk {
    /// Probability on each half-edge, `Σ_s |ψ(h, s)|²`.
    pub fn half_edge_probabilities(&self, psi: &[c64]) -> Vec<f64> {
        psi.par_chunks(self.row_len()).map(|row| row.iter().map(|z| z.norm_sqr()).sum()).collect()
    }

    /// Particle distribution `p(x) = Σ_{c,s} |ψ(x, c, s)|²`.
    pub fn node_probabilities(&self, psi: &[c64]) -> Vec<f64> {
        let ph = self.half_edge_probabilities(psi);
        (0..self.graph().nodes()).map(|x| self.graph().half_edges_of(x).map(|h| ph[h]).sum()).collect()
    }

    /// Spin expectation `(⟨X⟩, ⟨Y⟩, ⟨Z⟩)` of every edge (`Z = +1` for spin up).
    pub fn spins(&self, psi: &[c64]) -> Vec<[f64; 3]> {
        let l = self.row_len();
        (0..self.graph().edges())
            .into_par_iter()
            .map(|e| {
                let bit = 1usize << e;
                psi.par_chunks(l)
                    .map(|row| {
                        let mut acc = [0.0; 3];
                        for block in row.chunks(2 * bit) {
                            let (up, down) = block.split_at(bit);
                            for (a, b) in up.iter().zip(down) {
                                let ab = a.conj() * b;
                                acc[0] += 2.0 * ab.re;
                                acc[1] += 2.0 * ab.im;
                                acc[2] += a.norm_sqr() - b.norm_sqr();
                            }
                        }
                        acc
                    })
                    .reduce(|| [0.0; 3], |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]])
            })
            .collect()
    }

    /// Gram matrix of the half-edge rows, `G[h, h'] = Σ_s ψ(h, s) ψ*(h', s)`: the reduced
    /// density matrix of the walker (position and color).
    pub fn walker_density(&self, psi: &[c64]) -> CMat {
        let rows = MatRef::from_row_major_slice(psi, self.graph().half_edges(), self.row_len());
        product(rows, Op::Plain, rows, Op::Adjoint)
    }

    /// Reduced density matrix of the position, colors padded to the largest degree.
    pub fn position_density(&self, psi: &[c64]) -> CMat {
        let g = self.walker_density(psi);
        let graph = self.graph();
        CMat::from_fn(graph.nodes(), graph.nodes(), |x, y| {
            (0..graph.degree(x).min(graph.degree(y)))
                .map(|c| g[(graph.half_edge(x, c), graph.half_edge(y, c))])
                .sum()
        })
    }

    /// Reduced density matrix of the color, padded to the largest degree.
    pub fn color_density(&self, psi: &[c64]) -> CMat {
        let g = self.walker_density(psi);
        let graph = self.graph();
        let d = graph.max_degree();
        CMat::from_fn(d, d, |c, k| {
            (0..graph.nodes())
                .filter(|&x| c.max(k) < graph.degree(x))
                .map(|x| g[(graph.half_edge(x, c), graph.half_edge(x, k))])
                .sum()
        })
    }

    /// Entropies `(S_x, S_c, S_s)` of the position, the color and all the spins; `S_s` is that of
    /// the walker (position and color together), the state being pure.
    pub fn entropies(&self, psi: &[c64]) -> [f64; 3] {
        [
            von_neumann(&self.position_density(psi)),
            von_neumann(&self.color_density(psi)),
            von_neumann(&self.walker_density(psi)),
        ]
    }

    /// Reduced density matrix of the spins of `edges`: basis index `Σ_k bit(edges[k]) 2^k`.
    pub fn spin_density(&self, psi: &[c64], edges: &[usize]) -> CMat {
        let total = self.graph().edges();
        assert!(edges.iter().all(|&e| e < total), "no such edge");
        let mut kept = edges.to_vec();
        kept.sort_unstable();
        kept.dedup();
        assert_eq!(kept.len(), edges.len(), "repeated edge");
        let rest: Vec<usize> = (0..total).filter(|e| !edges.contains(e)).collect();
        let spread = |bits: &[usize], i: usize| -> usize {
            bits.iter().enumerate().filter(|(k, _)| i >> k & 1 == 1).map(|(_, &e)| 1usize << e).sum()
        };
        let a: Vec<usize> = (0..1usize << edges.len()).map(|i| spread(edges, i)).collect();
        let r: Vec<usize> = (0..1usize << rest.len()).map(|i| spread(&rest, i)).collect();
        let dim = a.len();
        // Φ_h[i, j] = ψ(h, a[i] + r[j]); ρ = Σ_h Φ_h Φ_h†.
        psi.par_chunks(self.row_len())
            .map(|row| {
                let phi = CMat::from_fn(dim, r.len(), |i, j| row[a[i] | r[j]]);
                product(phi.as_ref(), Op::Plain, phi.as_ref(), Op::Adjoint)
            })
            .reduce(|| CMat::zeros(dim, dim), |x, y| x + y)
    }

    /// Entanglement entropy `S_A` of the spins of `edges` with the rest of the system.
    pub fn spin_entropy(&self, psi: &[c64], edges: &[usize]) -> f64 {
        von_neumann(&self.spin_density(psi, edges))
    }

    /// Concurrence of the spins of edges `e1`, `e2` (Wootters).
    pub fn concurrence(&self, psi: &[c64], e1: usize, e2: usize) -> f64 {
        concurrence(&self.spin_density(psi, &[e1, e2]))
    }
}

/// Concurrence `max(0, √λ₁ − √λ₂ − √λ₃ − √λ₄)` of a two-qubit density matrix, `λ` the
/// eigenvalues of `ρ (Y⊗Y) ρ* (Y⊗Y)` in decreasing order.
pub fn concurrence(rho: &CMat) -> f64 {
    assert!(rho.nrows() == 4 && rho.ncols() == 4, "two-qubit density matrix expected");
    // Y⊗Y is real: (Y⊗Y)[i, j] = −1, 1, 1, −1 on the antidiagonal.
    let yy = CMat::from_fn(4, 4, |i, j| {
        if i + j == 3 { c64::new(if i == 0 || i == 3 { -1.0 } else { 1.0 }, 0.0) } else { c64::new(0.0, 0.0) }
    });
    let conj = CMat::from_fn(4, 4, |i, j| rho[(i, j)].conj());
    let tilde = &yy * &conj * &yy;
    // √λ are the eigenvalues of √(√ρ ρ̃ √ρ).
    let (w, v) = hermitian_eigen(rho.as_ref()).expect("eigenvalues");
    let sqrt_rho = &v
        * CMat::from_fn(4, 4, |i, j| c64::new(if i == j { w[i].max(0.0).sqrt() } else { 0.0 }, 0.0))
        * v.adjoint();
    let m = &sqrt_rho * &tilde * &sqrt_rho;
    let (mut l, _) = hermitian_eigen(m.as_ref()).expect("eigenvalues");
    l.iter_mut().for_each(|x| *x = x.max(0.0).sqrt());
    l.sort_by(|a, b| b.total_cmp(a));
    (l[0] - l[1] - l[2] - l[3]).max(0.0)
}

/// Kolmogorov–Smirnov distance between a distribution over the nodes and the uniform one,
/// `max_x |Σ_{y ≤ x} p(y) − (x + 1)/N|` (Sellapillay & Verga, Eq. 31): 0 for a walker spread
/// evenly, larger for a localized one.
pub fn ks_distance(p: &[f64]) -> f64 {
    let (n, total) = (p.len() as f64, p.iter().sum::<f64>());
    let mut cumulative = 0.0;
    p.iter()
        .enumerate()
        .map(|(x, v)| {
            cumulative += v / total;
            (cumulative - (x + 1) as f64 / n).abs()
        })
        .fold(0.0, f64::max)
}

impl SpinWalk {
    /// Adds `p(x)` to `acc` on the calling thread (for runs side by side, see
    /// [`SpinWalk::step_serial`]).
    pub fn accumulate_node_probabilities(&self, psi: &[c64], acc: &mut [f64]) {
        let graph = self.graph();
        for (h, row) in psi.chunks(self.row_len()).enumerate() {
            acc[graph.node_and_color(h).0] += row.iter().map(|z| z.norm_sqr()).sum::<f64>();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::walk::{PLUS, UP};

    #[test]
    fn product_state_observables() {
        let walk = SpinWalk::ring(4, 0.3, 0.5);
        let psi = walk.product_state(&walk.localized(1, 1), &[UP, PLUS, UP, PLUS]);
        let p = walk.node_probabilities(&psi);
        assert!((p[1] - 1.0).abs() < 1e-15 && p[0].abs() < 1e-15);
        let s = walk.spins(&psi);
        assert!((s[0][2] - 1.0).abs() < 1e-15 && (s[1][0] - 1.0).abs() < 1e-15 && s[1][2].abs() < 1e-15);
        let [sx, sc, ss] = walk.entropies(&psi);
        assert!(sx.abs() < 1e-12 && sc.abs() < 1e-12 && ss.abs() < 1e-12);
        assert!(walk.spin_entropy(&psi, &[0, 3]).abs() < 1e-12);
        assert!(walk.concurrence(&psi, 1, 2).abs() < 1e-12);
    }

    #[test]
    fn bell_pair() {
        // (|00⟩ + |11⟩)/√2 on spins 0 and 2, walker on one half-edge: one bit of entanglement
        // between spin 0 and the rest, concurrence 1, and no entanglement of the walker.
        let walk = SpinWalk::ring(3, 0.3, 0.5);
        let l = walk.row_len();
        let h = walk.graph().half_edge(0, 0);
        let mut psi = vec![c64::new(0.0, 0.0); walk.dimension()];
        psi[h * l] = c64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
        psi[h * l + 0b101] = c64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
        assert!((walk.spin_entropy(&psi, &[0]) - 1.0).abs() < 1e-12);
        assert!(walk.spin_entropy(&psi, &[0, 2]).abs() < 1e-12);
        assert!((walk.concurrence(&psi, 0, 2) - 1.0).abs() < 1e-12);
        assert!(walk.concurrence(&psi, 0, 1).abs() < 1e-12);
        assert!(walk.entropies(&psi)[2].abs() < 1e-12);
    }

    /// The coupling entangles the walker and the spins, and `S_s` equals the entropy of all the
    /// spins computed directly.
    #[test]
    fn walker_spin_entanglement() {
        let walk = SpinWalk::ring(4, std::f64::consts::FRAC_PI_4, 1.0);
        let mut psi = walk.product_state(&walk.localized(0, 0), &[PLUS; 4]);
        walk.steps(&mut psi, 12);
        let [_, _, ss] = walk.entropies(&psi);
        assert!(ss > 0.1);
        assert!((walk.spin_entropy(&psi, &[0, 1, 2, 3]) - ss).abs() < 1e-10);
        let norm: f64 = walk.node_probabilities(&psi).iter().sum();
        assert!((norm - 1.0).abs() < 1e-13);
    }
}
