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
use caroli_dense::{Accum, Lu, Op, matmul, norm_1, product};
use caroli_model::{CMat, c64};
use faer::Side;

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
/// `X_α` or `Y_γ` with an inverse condition number (1-norm estimate) below this: a bound state
/// of the lead surface.
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

/// Stable partition of the Schur form `(t, q)` and of its keys: selected keys first.
fn partition_by(t: &mut CMat, q: &mut CMat, keys: &mut Vec<usize>, select: impl Fn(usize) -> bool) {
    let flags: Vec<bool> = keys.iter().map(|&k| select(k)).collect();
    caroli_dense::partition(t, q, &flags);
    let (mut first, rest): (Vec<usize>, Vec<usize>) = keys.iter().copied().partition(|&k| select(k));
    first.extend(rest);
    *keys = first;
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
    ///
    /// At a real energy, a Hermitian definite coupling takes the fast path of
    /// [`DefiniteCoupling`]; everything else takes the general algorithm.
    pub fn at(h0: &CMat, h1: &CMat, z: c64) -> Result<Self, GreenError> {
        let definite = if z.im == 0.0 { DefiniteCoupling::new(h0, h1) } else { None };
        Self::shifted(h0, h1, z, |e, scale| match &definite {
            Some(d) => d.modes(h0, e.re, scale),
            None => Self::solve(h0, h1, e, scale),
        })
    }

    /// Modes at `z` by the general algorithm (the `2r` Schur decomposition), whatever the
    /// coupling: a cross-check of the fast path.
    pub fn at_general(h0: &CMat, h1: &CMat, z: c64) -> Result<Self, GreenError> {
        Self::shifted(h0, h1, z, |e, scale| Self::solve(h0, h1, e, scale))
    }

    /// `solve` at `z`, then at `z + δ` for the shifts δ of [`SHIFTS`] until it succeeds.
    fn shifted(
        h0: &CMat,
        h1: &CMat,
        z: c64,
        solve: impl Fn(c64, f64) -> Result<Self, GreenError>,
    ) -> Result<Self, GreenError> {
        let scale = (row_norm(h0) + 2.0 * row_norm(h1)).max(f64::MIN_POSITIVE);
        let mut last = GreenError::Eigen;
        for shift in SHIFTS {
            match solve(z + shift * scale, scale) {
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
        let caroli_dense::Svd { u: su, s: ss, v: sv } =
            caroli_dense::svd(h1.as_ref()).map_err(|_| GreenError::Eigen)?;
        let smax = ss.iter().copied().fold(0.0, f64::max);
        let kept: Vec<usize> = (0..ss.len()).filter(|&k| ss[k] > RANK_TOL * smax).collect();
        let r = kept.len();
        let u = CMat::from_fn(n, r, |a, k| su[(a, kept[k])] * ss[kept[k]].sqrt());
        let v = CMat::from_fn(n, r, |a, k| sv[(a, kept[k])] * ss[kept[k]].sqrt());
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
        let uv = CMat::from_fn(n, 2 * r, |p, k| if k < r { u[(p, k)] } else { v[(p, k - r)] });
        let stab = product(uv.as_ref(), Op::Plain, uv.as_ref(), Op::Adjoint);
        let a = CMat::from_fn(n, n, |p, q| h0[(p, q)] - if p == q { z } else { zero } + i * stab[(p, q)]);
        let g = Lu::new(a.as_ref()).solve(uv.as_ref());
        if g.norm_max().is_nan() {
            return Err(GreenError::Singular(z.re));
        }
        let (gu, gv) = (g.as_ref().subcols(0, r), g.as_ref().subcols(r, r));
        // [u v]† G̃ [u v], in blocks.
        let blocks = product(uv.as_ref(), Op::Adjoint, g.as_ref(), Op::Plain);
        let (ugu, ugv) = (blocks.as_ref().submatrix(0, 0, r, r), blocks.as_ref().submatrix(0, r, r, r));
        let (vgu, vgv) = (blocks.as_ref().submatrix(r, 0, r, r), blocks.as_ref().submatrix(r, r, r, r));

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
        let pivot_ratio = |lu: &Lu| {
            let (lo, hi) = lu.pivot_range();
            if hi > 0.0 { lo / hi } else { 0.0 }
        };
        #[derive(Clone, Copy)]
        enum Map {
            Direct,
            Inverse,
            Mobius(c64),
        }
        let (lu_l, lu_r) = (Lu::new(lmat.as_ref()), Lu::new(rmat.as_ref()));
        let (pl, pr) = (pivot_ratio(&lu_l), pivot_ratio(&lu_r));
        let (c, map) = if pl >= pr && pl > 1e-10 {
            (lu_l.solve(rmat.as_ref()), Map::Direct)
        } else if pr > 1e-10 {
            (lu_r.solve(lmat.as_ref()), Map::Inverse)
        } else {
            let mut best: Option<(f64, c64, CMat)> = None;
            for k in 0..8 {
                let sigma = c64::from_polar(if k % 2 == 0 { 0.5 } else { 2.0 }, 0.3 + 0.9 * k as f64);
                let m = &rmat - CMat::from_fn(2 * r, 2 * r, |p, q| sigma * lmat[(p, q)]);
                let lu = Lu::new(m.as_ref());
                let ratio = pivot_ratio(&lu);
                if best.as_ref().is_none_or(|b| ratio > b.0) {
                    best = Some((ratio, sigma, lu.solve(lmat.as_ref())));
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
        let (q0, t0) = caroli_dense::schur(c.as_ref()).map_err(|_| GreenError::Eigen)?;
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

        // Ordering A: right-evanescent, then the propagating groups (blocked partitions, then
        // the few swaps that make each group contiguous).
        let (mut qa, mut ta) = (q0.clone(), t0.clone());
        let mut keys_a = keys.clone();
        partition_by(&mut ta, &mut qa, &mut keys_a, |k| k == RIGHT);
        partition_by(&mut ta, &mut qa, &mut keys_a, |k| k != LEFT);
        caroli_dense::reorder(&mut ta, &mut qa, &mut keys_a);
        let n_re = keys_a.iter().filter(|&&k| k == RIGHT).count();
        let (mut right, mut left): (Vec<Vec<c64>>, Vec<Vec<c64>>) =
            ((0..n_re).map(|k| (0..m2).map(|p| qa[(p, k)]).collect()).collect(), Vec::new());
        let (mut rv, mut lv) = (Vec::new(), Vec::new());

        // Propagating modes, in groups of (nearly) equal λ. Each member keeps its own eigenvalue.
        // A numerically scalar group block (a true degeneracy) takes the Schur basis e_k on the
        // group; otherwise each eigenvector is back-substituted within the group too, and a Jordan
        // block (subband edge) shows up as nearly parallel vectors.
        let tnorm = ta.norm_max();
        let mut groups: Vec<(usize, usize, bool)> = Vec::new();
        let mut g0 = n_re;
        while g0 < m2 && keys_a[g0] != LEFT {
            let mut g1 = g0 + 1;
            while g1 < m2 && keys_a[g1] == keys_a[g0] {
                g1 += 1;
            }
            let off = (g0..g1)
                .flat_map(|a| (a + 1..g1).map(move |b| (a, b)))
                .map(|(a, b)| ta[(a, b)].norm())
                .fold(0.0, f64::max);
            groups.push((g0, g1, off <= 1e-8 * tnorm));
            g0 = g1;
        }
        let np = g0 - n_re;
        // Eigenvectors of the Schur form (the columns of y), by column-oriented back-substitution.
        let mut y = CMat::zeros(m2, np);
        let mut lams: Vec<c64> = Vec::with_capacity(np);
        let mut acc: Vec<c64> = Vec::with_capacity(m2);
        for &(g0, g1, scalar) in &groups {
            for k in g0..g1 {
                let mu = ta[(k, k)];
                lams.push(to_lambda(mu).ok_or(GreenError::Singular(z.re))?);
                let col = y.col_as_slice_mut(k - n_re);
                col[k] = one;
                let start = if scalar { g0 } else { k };
                acc.clear();
                acc.extend(ta.col_as_slice(k)[..start].iter().map(|t| -t));
                for j in (0..start).rev() {
                    let d = ta[(j, j)] - mu;
                    if d == zero {
                        return Err(GreenError::Singular(z.re));
                    }
                    let yj = acc[j] / d;
                    col[j] = yj;
                    for (a, t) in acc[..j].iter_mut().zip(&ta.col_as_slice(j)[..j]) {
                        *a -= yj * t;
                    }
                }
            }
        }
        // x = Q y, normalised, and φ = G̃(−v α + i v γ + λ(i u α − u γ)) with (α, γ) = x.
        let mut x = product(qa.as_ref(), Op::Plain, y.as_ref(), Op::Plain);
        for p in 0..np {
            let col = x.col_as_slice_mut(p);
            let norm = col.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            if !(norm.is_finite() && norm > 0.0) {
                return Err(GreenError::Singular(z.re));
            }
            col.iter_mut().for_each(|z| *z /= norm);
        }
        let bb = CMat::from_fn(r, np, |q, p| -x[(q, p)] + i * x[(r + q, p)]);
        let cc = CMat::from_fn(r, np, |q, p| lams[p] * (i * x[(q, p)] - x[(r + q, p)]));
        let mut phis = CMat::zeros(n, np);
        matmul(phis.as_mut(), Accum::Replace, gv, Op::Plain, bb.as_ref(), Op::Plain, one);
        matmul(phis.as_mut(), Accum::Add, gu, Op::Plain, cc.as_ref(), Op::Plain, one);
        let h1phi = product(h1.as_ref(), Op::Plain, phis.as_ref(), Op::Plain);
        let h1dphi = product(h1.as_ref(), Op::Adjoint, phis.as_ref(), Op::Plain);
        // Current J_ab = i(λ_b φ_a† H1 φ_b − λ_a* φ_a† H1† φ_b) of each group, in the metric of the
        // (α, γ) vectors; its inertia gives the right- and left-going modes.
        for &(g0, g1, _) in &groups {
            let (p0, m) = (g0 - n_re, g1 - g0);
            let (xg, fg) = (x.as_ref().subcols(p0, m), phis.as_ref().subcols(p0, m));
            let gram = product(xg, Op::Adjoint, xg, Op::Plain);
            let a = product(fg, Op::Adjoint, h1phi.as_ref().subcols(p0, m), Op::Plain);
            let b = product(fg, Op::Adjoint, h1dphi.as_ref().subcols(p0, m), Op::Plain);
            let cur =
                CMat::from_fn(m, m, |p, q| i * (lams[p0 + q] * a[(p, q)] - lams[p0 + p].conj() * b[(p, q)]));
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
            let (phi_k, x_k) = (
                product(fg, Op::Plain, coeffs.as_ref(), Op::Plain),
                product(xg, Op::Plain, coeffs.as_ref(), Op::Plain),
            );
            for k in 0..m {
                let vk = vel[k].re / phi_k.col_as_slice(k).iter().map(|z| z.norm_sqr()).sum::<f64>();
                if vk.is_nan() || vk.abs() < VELOCITY_TOL * scale {
                    return Err(GreenError::Singular(z.re)); // subband edge
                }
                if vk > 0.0 {
                    right.push(x_k.col_as_slice(k).to_vec());
                    rv.push(vk);
                } else {
                    left.push(x_k.col_as_slice(k).to_vec());
                    lv.push(vk);
                }
            }
        }

        // Ordering B: left-evanescent first.
        let (mut qb, mut tb) = (q0, t0);
        let left_evanescent: Vec<bool> = keys.iter().map(|&k| k == LEFT).collect();
        let n_le = caroli_dense::partition(&mut tb, &mut qb, &left_evanescent);
        left.extend((0..n_le).map(|k| (0..m2).map(|p| qb[(p, k)]).collect::<Vec<c64>>()));

        if right.len() != r || left.len() != r {
            return Err(GreenError::ModeCount { right: right.len(), left: left.len(), expected: r });
        }
        // Orthonormal bases; columns scaled to unit norm first (Householder QR errors are
        // relative to the whole matrix).
        let basis = |cols: &[Vec<c64>]| -> CMat {
            let norms: Vec<f64> =
                cols.iter().map(|c| c.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()).collect();
            caroli_dense::orthonormal_columns(
                CMat::from_fn(m2, cols.len(), |p, q| cols[q][p] / norms[q]).as_ref(),
            )
        };
        let (xb, yb) = (basis(&right), basis(&left));
        let invertible =
            |m: CMat| Lu::new(m.as_ref()).inverse_condition_1(norm_1(m.as_ref())) >= SINGULAR_TOL;
        if !invertible(rows(&xb, 0..r)) || !invertible(rows(&yb, r..m2)) {
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

/// A Hermitian definite coupling `H1 = s P` (`s = ±1`, `P = L L† > 0`) with a Hermitian cell
/// `H0`: square lattices, and multi-orbital leads without spin-orbit coupling or magnetic field
/// along the axis.
///
/// At a real energy `E`, with `μ = λ + 1/λ`, the Bloch equation `H1† λ⁻¹ + H0 − E + H1 λ`
/// becomes the Hermitian problem `(H0 − E) ψ = −s μ P ψ`. With `ψ = L⁻† y` it reads
/// `L⁻¹ (H0 − E) L⁻† y = ν y`, `μ = −s ν`: one Hermitian eigenproblem of size n instead of a
/// Schur decomposition of size 2n.
/// - Each transverse mode `y` gives the two roots of `λ² − μλ + 1 = 0`. If `|μ| < 2` they are
///   on the unit circle, and the current `J = −2 s Im λ ψ†Pψ` sorts them. Otherwise they are
///   real, `|λ| < 1` going right.
/// - With the factors `u = s L`, `v = L` of the coupling, a mode is `(α, γ) ∝ (s y, λ y)`, so
///   the bases of right- and left-going modes are orthonormal as they stand, and `X_α`, `Y_γ`
///   are never singular.
#[derive(Clone, Debug)]
struct DefiniteCoupling {
    s: f64,
    l: CMat,
}

impl DefiniteCoupling {
    /// Largest `‖m − m†‖ / ‖m‖` for a Hermitian matrix.
    const HERMITIAN_TOL: f64 = 1e-14;
    /// Smallest ratio of the diagonal entries of `L` (a condition number of `P` below ~1e8).
    const DEFINITE_TOL: f64 = 1e-4;

    fn new(h0: &CMat, h1: &CMat) -> Option<Self> {
        let n = h0.nrows();
        let hermitian = |m: &CMat| {
            let asym = (0..n)
                .flat_map(|i| (0..n).map(move |j| (i, j)))
                .map(|(i, j)| (m[(i, j)] - m[(j, i)].conj()).norm())
                .fold(0.0, f64::max);
            asym <= Self::HERMITIAN_TOL * m.norm_max()
        };
        if n == 0 || h1.norm_max() == 0.0 || !hermitian(h0) || !hermitian(h1) {
            return None;
        }
        [-1.0, 1.0].into_iter().find_map(|s| {
            let p = CMat::from_fn(n, n, |i, j| (h1[(i, j)] + h1[(j, i)].conj()) * (0.5 * s));
            let l = p.llt(Side::Lower).ok()?.L().to_owned();
            let (lo, hi) = (0..n).fold((f64::INFINITY, 0.0f64), |(lo, hi), i| {
                (lo.min(l[(i, i)].norm()), hi.max(l[(i, i)].norm()))
            });
            (lo > Self::DEFINITE_TOL * hi).then_some(Self { s, l })
        })
    }

    /// Modes at the real energy `e`.
    fn modes(&self, h0: &CMat, e: f64, scale: f64) -> Result<LeadModes, GreenError> {
        use faer::linalg::triangular_solve::{
            solve_lower_triangular_in_place, solve_upper_triangular_in_place,
        };
        let (n, s, l) = (h0.nrows(), self.s, &self.l);
        let par = faer::get_global_parallelism();
        // M = L⁻¹ (H0 − E) L⁻†.
        let mut b = CMat::from_fn(n, n, |i, j| {
            (h0[(i, j)] + h0[(j, i)].conj()) * 0.5
                - if i == j { c64::new(e, 0.0) } else { c64::new(0.0, 0.0) }
        });
        solve_lower_triangular_in_place(l.as_ref(), b.as_mut(), par);
        let mut m = b.adjoint().to_owned();
        solve_lower_triangular_in_place(l.as_ref(), m.as_mut(), par);
        let m = CMat::from_fn(n, n, |i, j| (m[(i, j)] + m[(j, i)].conj()) * 0.5);
        let evd = m.self_adjoint_eigen(Side::Lower).map_err(|_| GreenError::Eigen)?;
        let (nu, y) = (evd.S(), evd.U());

        // The right- and left-going root of each transverse mode.
        let (mut right, mut left, mut propagating) =
            (Vec::with_capacity(n), Vec::with_capacity(n), Vec::new());
        for k in 0..n {
            let mu = -s * nu[k].re;
            if mu.abs() < 2.0 {
                // J ∝ −s Im λ: the root with s Im λ < 0 goes right. Roots closer than the
                // degeneracy tolerance form a Jordan block, as in the general algorithm.
                let root = c64::new(mu / 2.0, (4.0 - mu * mu).sqrt() / 2.0);
                if 2.0 * root.im < DEGENERATE_TOL {
                    return Err(GreenError::Singular(e)); // subband edge
                }
                let (r, lft) = if s < 0.0 { (root, root.conj()) } else { (root.conj(), root) };
                right.push(r);
                left.push(lft);
                propagating.push(k);
            } else {
                let big = (mu + mu.signum() * (mu * mu - 4.0).sqrt()) / 2.0;
                if big.abs() <= 1.0 + UNIT_TOL {
                    return Err(GreenError::Singular(e)); // subband edge (λ = ±1)
                }
                right.push(c64::new(1.0 / big, 0.0));
                left.push(c64::new(big, 0.0));
            }
        }
        // Velocities J / ψ†ψ of the propagating modes, ψ = L⁻† y (P-normalised).
        let mut psi = CMat::from_fn(n, propagating.len(), |i, c| y[(i, propagating[c])]);
        solve_upper_triangular_in_place(l.adjoint(), psi.as_mut(), par);
        let (mut rv, mut lv) = (Vec::with_capacity(propagating.len()), Vec::with_capacity(propagating.len()));
        for (c, &k) in propagating.iter().enumerate() {
            let v = -2.0 * s * right[k].im / psi.col_as_slice(c).iter().map(|z| z.norm_sqr()).sum::<f64>();
            let fast = v.abs() >= VELOCITY_TOL * scale; // false for NaN
            if !fast {
                return Err(GreenError::Singular(e)); // subband edge
            }
            rv.push(v);
            lv.push(-v);
        }
        let basis = |lams: &[c64]| {
            CMat::from_fn(2 * n, n, |p, k| {
                let c = 1.0 / (1.0 + lams[k].norm_sqr()).sqrt();
                if p < n { y[(p, k)] * (s * c) } else { y[(p - n, k)] * lams[k] * c }
            })
        };
        Ok(LeadModes {
            u: CMat::from_fn(n, n, |i, j| l[(i, j)] * s),
            v: l.clone(),
            right: basis(&right),
            left: basis(&left),
            right_velocities: rv,
            left_velocities: lv,
            energy: c64::new(e, 0.0),
        })
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
