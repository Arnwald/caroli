//! Lead modes and mode-matching self-energies.
//!
//! A Bloch solution `ψ_j = λ^j u` of a lead with cell Hamiltonian `H0` and coupling
//! `H1 = ⟨j| H |j+1⟩` satisfies `H1† u + λ (H0 − E) u + λ² H1 u = 0`. The linearisation
//!
//! ```text
//! [ 0    1      ] [u ]     [ 1  0  ] [u ]
//! [ −H1†  E − H0 ] [λu] = λ [ 0  H1 ] [λu]
//! ```
//!
//! gives `2n` pairs `λ = α/β` (with `β = 0` for the infinite ones when `H1` is singular). A mode
//! goes right if it decays towards `+∞` (`|λ| < 1`) or propagates (`|λ| = 1`) with positive
//! velocity `v = u† i(λ H1 − λ* H1†) u`. No imaginary part of the energy is needed.

use crate::{GreenError, inverse, solve_right};
use caroli_model::{CMat, c64};
use faer::linalg::solvers::Solve;

/// Propagating if `||λ| − 1|` is below this.
const UNIT_TOL: f64 = 1e-8;
/// Propagating modes whose `λ` differ by less than this are treated as degenerate.
const DEGENERATE_TOL: f64 = 1e-6;

/// Modes of a lead at one energy, split by direction. Each mode is a pair of cell vectors
/// `(u, w) = (ψ_j, ψ_{j+1})`, so that modes with `λ = 0` or `λ = ∞` (singular couplings) are
/// represented without dividing by `λ`.
#[derive(Clone, Debug)]
pub struct LeadModes {
    /// Right-going modes (decaying towards +∞ or propagating with v > 0), as columns.
    pub right_u: CMat,
    pub right_w: CMat,
    /// Left-going modes (decaying towards −∞ or propagating with v < 0), as columns.
    pub left_u: CMat,
    pub left_w: CMat,
    /// Velocities of the propagating modes, right-going (> 0) and left-going (< 0).
    pub right_velocities: Vec<f64>,
    pub left_velocities: Vec<f64>,
}

/// Möbius shift used to turn the pencil `(A, B)` into a standard eigenproblem
/// `(A − σB)⁻¹ B x = μ x`, `λ = σ + 1/μ`: a generic point, away from the unit circle.
/// (faer's QZ is two orders of magnitude slower than its standard eigensolver.)
const SHIFT: c64 = c64::new(0.2732, 0.1879);

impl LeadModes {
    /// Solves for the modes of the lead `(h0, h1)` at energy `energy`.
    pub fn new(h0: &CMat, h1: &CMat, energy: f64) -> Result<Self, GreenError> {
        Self::at(h0, h1, c64::new(energy, 0.0))
    }

    /// Modes at a complex energy `z`. With `Im z > 0` no mode propagates (`|λ| ≠ 1`): the split
    /// is by `|λ|` alone, as in a Schur decomposition ordered by `|λ| < 1`.
    pub fn at(h0: &CMat, h1: &CMat, z: c64) -> Result<Self, GreenError> {
        let energy = z;
        let n = h0.nrows();
        let zero = c64::new(0.0, 0.0);
        let one = c64::new(1.0, 0.0);
        // A − σB and B for A = [[0, 1], [−H1†, E − H0]], B = [[1, 0], [0, H1]].
        let shifted = CMat::from_fn(2 * n, 2 * n, |i, j| match (i < n, j < n) {
            (true, true) => {
                if i == j {
                    -SHIFT
                } else {
                    zero
                }
            }
            (true, false) => {
                if j - n == i {
                    one
                } else {
                    zero
                }
            }
            (false, true) => -h1[(j, i - n)].conj(),
            (false, false) => {
                let (r, c) = (i - n, j - n);
                (if r == c { energy } else { zero }) - h0[(r, c)] - SHIFT * h1[(r, c)]
            }
        });
        let b = CMat::from_fn(2 * n, 2 * n, |i, j| match (i < n, j < n) {
            (true, true) => {
                if i == j {
                    one
                } else {
                    zero
                }
            }
            (false, false) => h1[(i - n, j - n)],
            _ => zero,
        });
        let c = shifted.partial_piv_lu().solve(&b);
        let evd = c.eigen().map_err(|_| GreenError::Eigen)?;
        let (mu, x) = (evd.S(), evd.U());

        type Mode = (Vec<c64>, Vec<c64>);
        let (mut right, mut left): (Vec<Mode>, Vec<Mode>) = (Vec::new(), Vec::new());
        let mut propagating: Vec<(c64, Vec<c64>)> = Vec::new();
        for k in 0..2 * n {
            // λ = α/β with α = 1 + σμ, β = μ (β = 0: λ = ∞).
            let (al, be) = (one + SHIFT * mu[k], mu[k]);
            let u: Vec<c64> = (0..n).map(|i| x[(i, k)]).collect();
            let w: Vec<c64> = (0..n).map(|i| x[(n + i, k)]).collect();
            let (an, bn) = (al.norm(), be.norm());
            if (an - bn).abs() <= UNIT_TOL * an.max(bn) {
                propagating.push((al / be, u));
            } else if an < bn {
                right.push((u, w));
            } else {
                left.push((u, w));
            }
        }

        // Group degenerate propagating modes and diagonalise the velocity in each group.
        let (mut rv, mut lv) = (Vec::new(), Vec::new());
        let mut used = vec![false; propagating.len()];
        for i in 0..propagating.len() {
            if used[i] {
                continue;
            }
            let lam = propagating[i].0 / propagating[i].0.norm();
            let group: Vec<usize> = (i..propagating.len())
                .filter(|&j| {
                    !used[j] && (propagating[j].0 / propagating[j].0.norm() - lam).norm() < DEGENERATE_TOL
                })
                .collect();
            for &j in &group {
                used[j] = true;
            }
            for (v, u) in velocity_basis(h1, lam, group.iter().map(|&j| &propagating[j].1)) {
                let w = u.iter().map(|z| lam * z).collect();
                if v > 0.0 {
                    right.push((u, w));
                    rv.push(v);
                } else {
                    left.push((u, w));
                    lv.push(v);
                }
            }
        }
        if right.len() != n || left.len() != n {
            return Err(GreenError::ModeCount { right: right.len(), left: left.len(), expected: n });
        }
        let cols = |modes: &[Mode], second: bool| {
            CMat::from_fn(n, n, |i, j| if second { modes[j].1[i] } else { modes[j].0[i] })
        };
        Ok(Self {
            right_u: cols(&right, false),
            right_w: cols(&right, true),
            left_u: cols(&left, false),
            left_w: cols(&left, true),
            right_velocities: rv,
            left_velocities: lv,
        })
    }

    /// Number of propagating channels in each direction.
    pub fn channels(&self) -> usize {
        self.right_velocities.len()
    }

    /// `F₊ = W₊ U₊⁻¹`: maps the wave function of a cell to the next one (towards +∞).
    pub fn forward_map(&self) -> CMat {
        solve_right(&self.right_w, &self.right_u)
    }

    /// `F₋⁻¹ = U₋ W₋⁻¹`: maps the wave function of a cell to the previous one (towards −∞).
    pub fn backward_map(&self) -> CMat {
        solve_right(&self.left_u, &self.left_w)
    }
}

/// Orthonormal basis of the span of `vectors` diagonalising the current operator
/// `W = i(λ H1 − λ* H1†)`; returns `(velocity, mode)` pairs.
fn velocity_basis<'a>(
    h1: &CMat,
    lam: c64,
    vectors: impl Iterator<Item = &'a Vec<c64>>,
) -> Vec<(f64, Vec<c64>)> {
    // Gram–Schmidt (twice for stability).
    let mut basis: Vec<Vec<c64>> = Vec::new();
    for v in vectors {
        let mut w = v.clone();
        for _ in 0..2 {
            for b in &basis {
                let p: c64 = b.iter().zip(&w).map(|(bi, wi)| bi.conj() * wi).sum();
                for (wi, bi) in w.iter_mut().zip(b) {
                    *wi -= p * bi;
                }
            }
        }
        let norm = w.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        if norm > 1e-10 {
            basis.push(w.into_iter().map(|z| z / norm).collect());
        }
    }
    let n = h1.nrows();
    let m = basis.len();
    let i = c64::new(0.0, 1.0);
    let w = CMat::from_fn(n, n, |r, c| i * (lam * h1[(r, c)] - lam.conj() * h1[(c, r)].conj()));
    let q = CMat::from_fn(n, m, |r, c| basis[c][r]);
    let wq = q.adjoint() * &w * &q;
    let evd = wq.self_adjoint_eigen(faer::Side::Lower).expect("Hermitian eigenproblem");
    let (s, z) = (evd.S(), evd.U());
    let modes = &q * z;
    (0..m).map(|k| (s[k].re, (0..n).map(|r| modes[(r, k)]).collect())).collect()
}

/// Self-energy of a right lead (towards +∞) on the slice it touches: `Σ = H1 F₊`.
pub fn sigma_right(h1: &CMat, modes: &LeadModes) -> CMat {
    h1 * modes.forward_map()
}

/// Self-energy of a left lead (towards −∞) on the slice it touches: `Σ = H1† F₋⁻¹`.
pub fn sigma_left(h1: &CMat, modes: &LeadModes) -> CMat {
    h1.adjoint() * modes.backward_map()
}

/// Surface Green's function of a right lead from its modes, `g = (E − H0 − H1 F₊)⁻¹`.
pub fn surface_green_right(h0: &CMat, h1: &CMat, energy: f64, modes: &LeadModes) -> CMat {
    let n = h0.nrows();
    let s = sigma_right(h1, modes);
    inverse(&CMat::from_fn(n, n, |i, j| {
        (if i == j { c64::new(energy, 0.0) } else { c64::new(0.0, 0.0) }) - h0[(i, j)] - s[(i, j)]
    }))
}
