//! Lead modes and mode-matching self-energies.
//!
//! A lead has a cell Hamiltonian `H0` and a coupling `H1 = ⟨j| H |j+1⟩`. Its Bloch solutions
//! `ψ_{j+1} = λ ψ_j` are found from the compact SVD `H1 = u v†` (`u = U√S`, `v = V√S`, rank
//! `r`). The vectors `x_j = (α_j, γ_j) = (u† ψ_{j−1}, v† ψ_j)` obey a regular `2r × 2r` pencil
//! `R x = λ L x`. This is the formulation of Wimmer and Kwant, stabilised by `i(uu† + vv†)` so that
//! it holds even when `H0 − E` is singular. Couplings of low rank (graphene ribbons cut along a
//! slanted direction, ...) therefore give small, well-conditioned problems.
//!
//! Right-going modes (`|λ| < 1`, or `|λ| = 1` with a positive velocity) span a subspace with
//! orthonormal basis `X`, left-going ones a subspace with basis `Y`, and
//!
//! ```text
//! Σ_R = u X_γ X_α⁻¹ u†   (lead towards +∞, on the slice it touches),
//! Σ_L = v Y_α Y_γ⁻¹ v†   (lead towards −∞).
//! ```
//!
//! Degenerate propagating modes are separated by diagonalising their current. Two cases have no
//! self-energy:
//! - a subband edge (a propagating mode with vanishing velocity);
//! - a bound state of the lead surface (singular `X_α` or `Y_γ`, e.g. graphene zigzag ends at
//!   E = 0).
//!
//! The modes are then computed at `E + δ`, with `δ` from 1e-9 to 1e-6 of the bandwidth (the limit
//! from above), and [`LeadModes::energy`] records the energy used.

use crate::{GreenError, solve_right};
use caroli_model::{CMat, c64};
use faer::Side;
use faer::linalg::solvers::Solve;

/// Singular values of `H1` below this fraction of the largest are dropped.
const RANK_TOL: f64 = 1e-12;
/// `|λ|` within this of 1: a propagating candidate.
const UNIT_TOL: f64 = 1e-6;
/// Propagating `λ` closer than this form one degenerate group.
const DEGENERATE_TOL: f64 = 1e-7;
/// Velocities (of unit cell vectors) below this fraction of the bandwidth scale: a subband edge.
const VELOCITY_TOL: f64 = 1e-8;
/// Propagating modes of one group whose `(α, γ)` vectors are closer to parallel than this
/// (angle): a Jordan block, i.e. a subband edge.
const DEFECTIVE_TOL: f64 = 1e-7;
/// `X_α` or `Y_γ` with `σ_min/σ_max` below this: a bound state of the lead surface.
const SINGULAR_TOL: f64 = 1e-11;
/// Energy shifts tried in turn, in units of the bandwidth scale.
const SHIFTS: [f64; 5] = [0.0, 1e-9, 1e-8, 1e-7, 1e-6];

/// Modes of a lead at one energy, as subspaces of the reduced coordinates `(α, γ)`.
#[derive(Clone, Debug)]
pub struct LeadModes {
    /// Compact factors of the coupling, `H1 = u v†` (`n × r`).
    pub u: CMat,
    pub v: CMat,
    /// Orthonormal basis of the right-going modes in `(α, γ)` coordinates (`2r × r`).
    pub right: CMat,
    /// Orthonormal basis of the left-going modes (`2r × r`).
    pub left: CMat,
    /// Velocities of the propagating modes: right-going (> 0) and left-going (< 0).
    pub right_velocities: Vec<f64>,
    pub left_velocities: Vec<f64>,
    /// Energy at which the modes were computed (shifted off a band edge or a surface state).
    pub energy: c64,
}

/// `∞`-norm, a bound on the spectral radius.
fn row_norm(m: &CMat) -> f64 {
    (0..m.nrows()).map(|i| (0..m.ncols()).map(|j| m[(i, j)].norm()).sum::<f64>()).fold(0.0, f64::max)
}

fn rows(m: &CMat, range: std::ops::Range<usize>) -> CMat {
    CMat::from_fn(range.len(), m.ncols(), |i, j| m[(range.start + i, j)])
}

/// Smallest over largest singular value (0 for an empty or zero matrix).
fn inverse_condition(m: &CMat) -> f64 {
    if m.nrows() == 0 {
        return 1.0;
    }
    match m.singular_values() {
        Ok(s) => {
            let (lo, hi) = s.iter().fold((f64::INFINITY, 0.0f64), |(l, h), &x| (l.min(x), h.max(x)));
            if hi > 0.0 { lo / hi } else { 0.0 }
        }
        Err(_) => 0.0,
    }
}

impl LeadModes {
    /// Modes of the lead `(h0, h1)` at the real energy `energy`.
    pub fn new(h0: &CMat, h1: &CMat, energy: f64) -> Result<Self, GreenError> {
        Self::at(h0, h1, c64::new(energy, 0.0))
    }

    /// Modes at a complex energy `z`. With `Im z > 0` no mode propagates (`|λ| ≠ 1`).
    pub fn at(h0: &CMat, h1: &CMat, z: c64) -> Result<Self, GreenError> {
        let scale = (row_norm(h0) + 2.0 * row_norm(h1)).max(f64::MIN_POSITIVE);
        let mut last = GreenError::Eigen;
        for shift in SHIFTS {
            match Self::solve(h0, h1, z + shift * scale, scale) {
                Ok(m) => return Ok(m),
                Err(e) => last = e,
            }
        }
        Err(last)
    }

    fn solve(h0: &CMat, h1: &CMat, z: c64, scale: f64) -> Result<Self, GreenError> {
        let n = h0.nrows();
        let i = c64::new(0.0, 1.0);
        let one = c64::new(1.0, 0.0);
        let zero = c64::new(0.0, 0.0);

        // Compact SVD of the coupling.
        let svd = h1.svd().map_err(|_| GreenError::Eigen)?;
        let (su, ss, sv) = (svd.U(), svd.S(), svd.V());
        let smax = (0..n).map(|k| ss[k].re).fold(0.0, f64::max);
        let kept: Vec<usize> = (0..n).filter(|&k| ss[k].re > RANK_TOL * smax).collect();
        let r = kept.len();
        let u = CMat::from_fn(n, r, |a, k| su[(a, kept[k])] * ss[kept[k]].re.sqrt());
        let v = CMat::from_fn(n, r, |a, k| sv[(a, kept[k])] * ss[kept[k]].re.sqrt());
        if r == 0 {
            return Ok(Self {
                u,
                v,
                right: CMat::zeros(0, 0),
                left: CMat::zeros(0, 0),
                right_velocities: Vec::new(),
                left_velocities: Vec::new(),
                energy: z,
            });
        }

        // Stabilised resolvent G̃ = (H0 − z + i(uu† + vv†))⁻¹, applied to u and v.
        let stab = &u * u.adjoint() + &v * v.adjoint();
        let a = CMat::from_fn(n, n, |p, q| h0[(p, q)] - if p == q { z } else { zero } + i * stab[(p, q)]);
        let rhs = CMat::from_fn(n, 2 * r, |p, k| if k < r { u[(p, k)] } else { v[(p, k - r)] });
        let g = a.partial_piv_lu().solve(&rhs);
        let gu = CMat::from_fn(n, r, |p, k| g[(p, k)]);
        let gv = CMat::from_fn(n, r, |p, k| g[(p, r + k)]);
        if gu.norm_max().is_nan() || gv.norm_max().is_nan() {
            return Err(GreenError::Singular(z.re));
        }
        let (ugu, ugv) = (u.adjoint() * &gu, u.adjoint() * &gv);
        let (vgu, vgv) = (v.adjoint() * &gu, v.adjoint() * &gv);

        // Pencil R x = λ L x:
        //   L = [[1 − i u†G̃u, u†G̃u], [−i v†G̃u, v†G̃u]],  R = [[−u†G̃v, i u†G̃v], [−v†G̃v, i v†G̃v − 1]].
        let lmat = CMat::from_fn(2 * r, 2 * r, |p, q| match (p < r, q < r) {
            (true, true) => (if p == q { one } else { zero }) - i * ugu[(p, q)],
            (true, false) => ugu[(p, q - r)],
            (false, true) => -i * vgu[(p - r, q)],
            (false, false) => vgu[(p - r, q - r)],
        });
        let rmat = CMat::from_fn(2 * r, 2 * r, |p, q| match (p < r, q < r) {
            (true, true) => -ugv[(p, q)],
            (true, false) => i * ugv[(p, q - r)],
            (false, true) => -vgv[(p - r, q)],
            (false, false) => i * vgv[(p - r, q - r)] - if p == q { one } else { zero },
        });

        // A standard eigenproblem with the same invariant subspaces: L⁻¹R (eigenvalue λ) or R⁻¹L
        // (1/λ) when one of them is well conditioned, else (R − σL)⁻¹L (λ = σ + 1/μ) with σ off
        // the unit circle.
        let pivot_ratio = |d: faer::MatRef<'_, c64>| {
            let (lo, hi) = (0..d.nrows()).fold((f64::INFINITY, 0.0f64), |(l, h), p| {
                let x = d[(p, p)].norm();
                (l.min(x), h.max(x))
            });
            if hi > 0.0 { lo / hi } else { 0.0 }
        };
        #[derive(Clone, Copy)]
        enum Map {
            Direct,
            Inverse,
            Mobius(c64),
        }
        let (lu_l, lu_r) = (lmat.partial_piv_lu(), rmat.partial_piv_lu());
        let (pl, pr) = (pivot_ratio(lu_l.U()), pivot_ratio(lu_r.U()));
        let (c, map) = if pl >= pr && pl > 1e-10 {
            (lu_l.solve(&rmat), Map::Direct)
        } else if pr > 1e-10 {
            (lu_r.solve(&lmat), Map::Inverse)
        } else {
            let mut best: Option<(f64, c64, CMat)> = None;
            for k in 0..8 {
                let sigma = c64::from_polar(if k % 2 == 0 { 0.5 } else { 2.0 }, 0.3 + 0.9 * k as f64);
                let m = &rmat - CMat::from_fn(2 * r, 2 * r, |p, q| sigma * lmat[(p, q)]);
                let lu = m.partial_piv_lu();
                let ratio = pivot_ratio(lu.U());
                if best.as_ref().is_none_or(|b| ratio > b.0) {
                    best = Some((ratio, sigma, lu.solve(&lmat)));
                }
            }
            let (ratio, sigma, c) = best.expect("candidates");
            if ratio.is_nan() || ratio <= 1e-14 {
                return Err(GreenError::Eigen);
            }
            (c, Map::Mobius(sigma))
        };

        // Schur form and classes: right-evanescent, propagating, left-evanescent (with λ = ∞).
        let m2 = 2 * r;
        let (q0, t0) = crate::schur::schur(&c)?;
        let mu_max = (0..m2).map(|k| t0[(k, k)].norm()).fold(0.0, f64::max);
        let to_lambda = |mu: c64| -> Option<c64> {
            match map {
                Map::Direct => Some(mu),
                _ if mu.norm() <= 1e-13 * mu_max => None,
                Map::Inverse => Some(one / mu),
                Map::Mobius(sigma) => Some(sigma + one / mu),
            }
        };
        let lams: Vec<Option<c64>> = (0..m2).map(|k| to_lambda(t0[(k, k)])).collect();
        const RIGHT: usize = 0;
        const LEFT: usize = usize::MAX;
        let mut keys: Vec<usize> = lams
            .iter()
            .map(|l| match l {
                Some(l) if l.norm() < 1.0 - UNIT_TOL => RIGHT,
                Some(l) if l.norm() <= 1.0 + UNIT_TOL => 1, // propagating, grouped below
                _ => LEFT,
            })
            .collect();
        // Degenerate propagating groups (single linkage), numbered 1, 2, ... in order.
        let prop: Vec<usize> = (0..m2).filter(|&k| keys[k] == 1).collect();
        let mut group: Vec<usize> = (0..prop.len()).collect();
        for a in 0..prop.len() {
            for b in 0..a {
                if (lams[prop[a]].unwrap() - lams[prop[b]].unwrap()).norm() < DEGENERATE_TOL {
                    let (ga, gb) = (group[a], group[b]);
                    let (lo, hi) = (ga.min(gb), ga.max(gb));
                    group.iter_mut().for_each(|g| {
                        if *g == hi {
                            *g = lo
                        }
                    });
                }
            }
        }
        let mut ids: Vec<usize> = group.clone();
        ids.sort_unstable();
        ids.dedup();
        for (a, &k) in prop.iter().enumerate() {
            keys[k] = 1 + ids.binary_search(&group[a]).expect("group id");
        }

        // Ordering A: right-evanescent, then the propagating groups.
        let (mut qa, mut ta) = (q0.clone(), t0.clone());
        let mut keys_a = keys.clone();
        crate::schur::reorder(&mut ta, &mut qa, &mut keys_a);
        let n_re = keys_a.iter().filter(|&&k| k == RIGHT).count();
        let (mut right, mut left): (Vec<Vec<c64>>, Vec<Vec<c64>>) =
            ((0..n_re).map(|k| (0..m2).map(|p| qa[(p, k)]).collect()).collect(), Vec::new());
        let (mut rv, mut lv) = (Vec::new(), Vec::new());
        let dot = |x: &[c64], y: &[c64]| x.iter().zip(y).map(|(a, b)| a.conj() * b).sum::<c64>();
        let mut g0 = n_re;
        while g0 < m2 && keys_a[g0] != LEFT {
            let mut g1 = g0 + 1;
            while g1 < m2 && keys_a[g1] == keys_a[g0] {
                g1 += 1;
            }
            let m = g1 - g0;
            // Each member keeps its own eigenvalue. A numerically scalar group block (a true
            // degeneracy) takes the Schur basis e_k on the group; otherwise each eigenvector is
            // back-substituted within the group too, and a Jordan block (subband edge) shows up as
            // nearly parallel vectors.
            let tnorm = ta.norm_max();
            let off = (g0..g1)
                .flat_map(|a| (a + 1..g1).map(move |b| (a, b)))
                .map(|(a, b)| ta[(a, b)].norm())
                .fold(0.0, f64::max);
            let scalar = off <= 1e-8 * tnorm;
            let mut lams_g: Vec<c64> = Vec::with_capacity(m);
            let mut xs: Vec<Vec<c64>> = Vec::with_capacity(m);
            let mut phis: Vec<Vec<c64>> = Vec::with_capacity(m);
            for k in g0..g1 {
                let mu = ta[(k, k)];
                let lam = to_lambda(mu).ok_or(GreenError::Singular(z.re))?;
                let mut y = vec![zero; g1];
                y[k] = one;
                let start = if scalar { g0 } else { k };
                for i in (0..start).rev() {
                    let s: c64 = (i + 1..g1).map(|j| ta[(i, j)] * y[j]).sum();
                    let d = ta[(i, i)] - mu;
                    if d == zero {
                        return Err(GreenError::Singular(z.re));
                    }
                    y[i] = -s / d;
                }
                let x: Vec<c64> = (0..m2).map(|p| (0..g1).map(|j| qa[(p, j)] * y[j]).sum()).collect();
                let norm = x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
                if !(norm.is_finite() && norm > 0.0) {
                    return Err(GreenError::Singular(z.re));
                }
                let x: Vec<c64> = x.iter().map(|z| z / norm).collect();
                // φ = G̃(−v α + i v γ + λ(i u α − u γ)).
                let (al, ga) = (&x[..r], &x[r..]);
                let bb: Vec<c64> = (0..r).map(|q| -al[q] + i * ga[q]).collect();
                let cc: Vec<c64> = (0..r).map(|q| lam * (i * al[q] - ga[q])).collect();
                let phi: Vec<c64> =
                    (0..n).map(|p| (0..r).map(|q| gv[(p, q)] * bb[q] + gu[(p, q)] * cc[q]).sum()).collect();
                lams_g.push(lam);
                xs.push(x);
                phis.push(phi);
            }
            // Current J_ab = i(λ_b φ_a† H1 φ_b − λ_a* φ_a† H1† φ_b) in the metric of the (α, γ)
            // vectors; its inertia gives the right- and left-going modes.
            let h1phi: Vec<Vec<c64>> = phis
                .iter()
                .map(|f| (0..n).map(|p| (0..n).map(|q| h1[(p, q)] * f[q]).sum()).collect())
                .collect();
            let h1dphi: Vec<Vec<c64>> = phis
                .iter()
                .map(|f| (0..n).map(|p| (0..n).map(|q| h1[(q, p)].conj() * f[q]).sum()).collect())
                .collect();
            let gram = CMat::from_fn(m, m, |p, q| dot(&xs[p], &xs[q]));
            let cur = CMat::from_fn(m, m, |p, q| {
                i * (lams_g[q] * dot(&phis[p], &h1phi[q]) - lams_g[p].conj() * dot(&phis[p], &h1dphi[q]))
            });
            let llt = gram.llt(Side::Lower).map_err(|_| GreenError::Singular(z.re))?;
            let lch = llt.L().to_owned();
            if inverse_condition(&lch) < DEFECTIVE_TOL {
                return Err(GreenError::Singular(z.re)); // a Jordan block: subband edge
            }
            let linv = crate::inverse(&lch);
            let jp = &linv * &cur * linv.adjoint();
            let jh = CMat::from_fn(m, m, |p, q| (jp[(p, q)] + jp[(q, p)].conj()) * 0.5);
            let evd = jh.self_adjoint_eigen(Side::Lower).map_err(|_| GreenError::Eigen)?;
            let (vel, w) = (evd.S(), evd.U());
            let coeffs = linv.adjoint() * w;
            for k in 0..m {
                let phi: Vec<c64> =
                    (0..n).map(|p| (0..m).map(|q| phis[q][p] * coeffs[(q, k)]).sum()).collect();
                let vk = vel[k].re / phi.iter().map(|z| z.norm_sqr()).sum::<f64>();
                if vk.is_nan() || vk.abs() < VELOCITY_TOL * scale {
                    return Err(GreenError::Singular(z.re)); // subband edge
                }
                let x: Vec<c64> = (0..m2).map(|p| (0..m).map(|q| xs[q][p] * coeffs[(q, k)]).sum()).collect();
                if vk > 0.0 {
                    right.push(x);
                    rv.push(vk);
                } else {
                    left.push(x);
                    lv.push(vk);
                }
            }
            g0 = g1;
        }

        // Ordering B: left-evanescent first.
        let (mut qb, mut tb) = (q0, t0);
        let mut keys_b: Vec<usize> = keys.iter().map(|&k| if k == LEFT { 0 } else { 1 }).collect();
        crate::schur::reorder(&mut tb, &mut qb, &mut keys_b);
        let n_le = keys_b.iter().filter(|&&k| k == 0).count();
        left.extend((0..n_le).map(|k| (0..m2).map(|p| qb[(p, k)]).collect::<Vec<c64>>()));

        if right.len() != r || left.len() != r {
            return Err(GreenError::ModeCount { right: right.len(), left: left.len(), expected: r });
        }
        // Orthonormal bases; columns scaled to unit norm first (Householder QR errors are
        // relative to the whole matrix).
        let basis = |cols: &[Vec<c64>]| -> CMat {
            let norms: Vec<f64> =
                cols.iter().map(|c| c.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()).collect();
            CMat::from_fn(m2, cols.len(), |p, q| cols[q][p] / norms[q]).qr().compute_thin_Q()
        };
        let (xb, yb) = (basis(&right), basis(&left));
        if inverse_condition(&rows(&xb, 0..r)) < SINGULAR_TOL
            || inverse_condition(&rows(&yb, r..m2)) < SINGULAR_TOL
        {
            return Err(GreenError::Singular(z.re));
        }
        Ok(Self { u, v, right: xb, left: yb, right_velocities: rv, left_velocities: lv, energy: z })
    }

    /// Number of propagating channels in each direction.
    pub fn channels(&self) -> usize {
        self.right_velocities.len()
    }

    /// `Σ_R = u X_γ X_α⁻¹ u†`.
    fn sigma_right(&self) -> CMat {
        let (n, r) = (self.u.nrows(), self.u.ncols());
        if r == 0 {
            return CMat::zeros(n, n);
        }
        let t = solve_right(&rows(&self.right, r..2 * r), &rows(&self.right, 0..r));
        &self.u * t * self.u.adjoint()
    }

    /// `Σ_L = v Y_α Y_γ⁻¹ v†`.
    fn sigma_left(&self) -> CMat {
        let (n, r) = (self.v.nrows(), self.v.ncols());
        if r == 0 {
            return CMat::zeros(n, n);
        }
        let t = solve_right(&rows(&self.left, 0..r), &rows(&self.left, r..2 * r));
        &self.v * t * self.v.adjoint()
    }
}

/// Self-energy of a right lead (towards +∞) on the slice it touches. `h1` is the coupling the
/// modes were computed with (kept for the signature; the modes hold its factors).
pub fn sigma_right(h1: &CMat, modes: &LeadModes) -> CMat {
    debug_assert_eq!(h1.nrows(), modes.u.nrows());
    modes.sigma_right()
}

/// Self-energy of a left lead (towards −∞) on the slice it touches.
pub fn sigma_left(h1: &CMat, modes: &LeadModes) -> CMat {
    debug_assert_eq!(h1.nrows(), modes.v.nrows());
    modes.sigma_left()
}

/// Surface Green's function of a right lead from its modes, `g = (E − H0 − Σ_R)⁻¹`.
pub fn surface_green_right(h0: &CMat, h1: &CMat, energy: f64, modes: &LeadModes) -> CMat {
    let n = h0.nrows();
    let s = sigma_right(h1, modes);
    crate::inverse(&CMat::from_fn(n, n, |i, j| {
        (if i == j { c64::new(energy, 0.0) } else { c64::new(0.0, 0.0) }) - h0[(i, j)] - s[(i, j)]
    }))
}
