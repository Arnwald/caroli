//! The walk operator `U = V(J) M C` and its application on the CPU, in f64 or f32.

use crate::graph::Graph;
use caroli_model::{CMat, SparseMatrix, c64};
use num_complex::Complex;
use num_traits::Float;
use rayon::prelude::*;

/// Spin configurations per parallel task.
const CHUNK: usize = 1 << 14;

/// The real type of a state on the CPU: `f64`, or `f32` for half the memory traffic and twice
/// the SIMD width, at about 10⁻⁷ of relative error per step (see [`crate::gpu`] for checked
/// runs in reduced precision).
pub trait Real: Float + Send + Sync + std::fmt::Debug + 'static {
    fn of(x: f64) -> Self;
    fn wide(self) -> f64;
}

impl Real for f64 {
    fn of(x: f64) -> Self {
        x
    }
    fn wide(self) -> f64 {
        self
    }
}

impl Real for f32 {
    fn of(x: f64) -> Self {
        x as f32
    }
    fn wide(self) -> f64 {
        self as f64
    }
}

fn narrow<T: Real>(z: c64) -> Complex<T> {
    Complex::new(T::of(z.re), T::of(z.im))
}

/// A state converted to f32.
pub fn to_single(psi: &[c64]) -> Vec<Complex<f32>> {
    psi.par_iter().map(|&z| narrow(z)).collect()
}

/// A state converted to f64.
pub fn to_double(psi: &[Complex<f32>]) -> Vec<c64> {
    psi.par_iter().map(|z| c64::new(z.re as f64, z.im as f64)).collect()
}

/// A raw pointer shared by tasks that touch disjoint entries.
struct Shared<T>(*mut Complex<T>);

impl<T> Clone for Shared<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Shared<T> {}

// SAFETY: every task writes only the entries it owns (see the callers).
unsafe impl<T> Send for Shared<T> {}
unsafe impl<T> Sync for Shared<T> {}

impl<T> Shared<T> {
    /// # Safety
    ///
    /// `start..start + len` must be inside the buffer and touched by no other live reference.
    #[allow(clippy::mut_from_ref)]
    unsafe fn slice(&self, start: usize, len: usize) -> &mut [Complex<T>] {
        // SAFETY: per the caller.
        unsafe { std::slice::from_raw_parts_mut(self.0.add(start), len) }
    }
}

/// Runs `f(task)` for `tasks` tasks, in parallel or on the calling thread.
fn run_tasks(parallel: bool, tasks: usize, f: impl Fn(usize) + Sync + Send) {
    if parallel {
        (0..tasks).into_par_iter().for_each(f);
    } else {
        (0..tasks).for_each(f);
    }
}

/// `(p, α, β)` of the edge update, in the precision of the state.
type Exchange<T> = (Complex<T>, Complex<T>, Complex<T>);

/// The edge update on runs of quadruples: `a0, a1` (half-edge `a`, edge spin up and down),
/// `b0, b1` (half-edge `b`); `(p, α, β)` from [`SpinWalk::exchange`]. With `ACC`, returns the
/// probabilities `Σ |a'|²`, `Σ |b'|²` of the updated entries.
#[inline]
fn exchange_run<T: Real, const ACC: bool>(
    a0: &mut [Complex<T>],
    a1: &mut [Complex<T>],
    b0: &mut [Complex<T>],
    b1: &mut [Complex<T>],
    (p, alpha, beta): Exchange<T>,
) -> (f64, f64) {
    let (mut sa, mut sb) = (T::zero(), T::zero());
    for (((x0, x1), y0), y1) in a0.iter_mut().zip(a1.iter_mut()).zip(b0.iter_mut()).zip(b1.iter_mut()) {
        let (u0, u1, v0, v1) = (*x0, *x1, *y0, *y1);
        *x0 = p * v0;
        *x1 = alpha * v1 + beta * u0;
        *y0 = beta * v1 + alpha * u0;
        *y1 = p * u1;
        if ACC {
            sa = sa + x0.norm_sqr() + x1.norm_sqr();
            sb = sb + y0.norm_sqr() + y1.norm_sqr();
        }
    }
    (sa.wide(), sb.wide())
}

/// The edge update for a low spin bit `B`: the rows `a`, `b` in blocks of `2B` configurations,
/// the first `B` with the spin up.
#[inline]
fn edge_blocks<T: Real, const B: usize, const ACC: bool>(
    a: &mut [Complex<T>],
    b: &mut [Complex<T>],
    (p, alpha, beta): Exchange<T>,
) -> (f64, f64) {
    let (mut sa, mut sb) = (T::zero(), T::zero());
    for (a, b) in a.chunks_exact_mut(2 * B).zip(b.chunks_exact_mut(2 * B)) {
        for i in 0..B {
            let (u0, u1, v0, v1) = (a[i], a[i + B], b[i], b[i + B]);
            a[i] = p * v0;
            a[i + B] = alpha * v1 + beta * u0;
            b[i] = beta * v1 + alpha * u0;
            b[i + B] = p * u1;
            if ACC {
                sa = sa + a[i].norm_sqr() + a[i + B].norm_sqr();
                sb = sb + b[i].norm_sqr() + b[i + B].norm_sqr();
            }
        }
    }
    (sa.wide(), sb.wide())
}

/// Coin `R(θ) = [[cos θ, −sin θ], [sin θ, cos θ]]` on the two colors of a node.
///
/// The motion leaves the walker on the edge it came along, so on a ring the coin angle `θ` sets
/// what it does next: it turns back with amplitude `cos θ` and goes on with amplitude `sin θ`.
/// At `θ = 0` the walker bounces on one edge forever, at `θ = π/2` it runs around the ring.
pub fn rotation(theta: f64) -> CMat {
    let (s, c) = theta.sin_cos();
    CMat::from_fn(2, 2, |i, j| c64::new([[c, -s], [s, c]][i][j], 0.0))
}

/// Grover coin `2/d 𝟙 − I` (`𝟙` the matrix of ones).
pub fn grover(d: usize) -> CMat {
    CMat::from_fn(d, d, |i, j| c64::new(2.0 / d as f64 - if i == j { 1.0 } else { 0.0 }, 0.0))
}

/// Fourier coin `e^{2iπ jk/d} / √d`.
pub fn fourier(d: usize) -> CMat {
    let norm = 1.0 / (d as f64).sqrt();
    CMat::from_fn(d, d, |j, k| c64::from_polar(norm, 2.0 * std::f64::consts::PI * (j * k) as f64 / d as f64))
}

/// A walker on a graph coupled to one spin per edge (K. Sellapillay and A. D. Verga, Phys. Rev.
/// E 103, 032123 (2021)). One step is `U = V(J) M C`:
///
/// - the coin `C` applies `coins[x]` to the colors of each node `x`;
/// - the motion `M` swaps the two half-edges of every edge, `|x c_y s⟩ ↔ |y c_x s⟩`;
/// - the interaction `V(J) = exp(i J τ·σ / 4)` couples, on every edge, the walker's side of the
///   edge (`τ`) to the edge spin (`σ`): an exchange that can move the walker across the edge
///   while flipping the spin.
///
/// The state is `ψ[h · 2^|E| + s]`: half-edge `h`, spin configuration `s` (bit `e` of `s` is the
/// spin of edge `e`, 0 for up).
#[derive(Clone, Debug)]
pub struct SpinWalk {
    graph: Graph,
    coins: Vec<CMat>,
    j: f64,
}

impl SpinWalk {
    /// Walk with one coin per node (a `d_x × d_x` unitary) and coupling `j`.
    pub fn new(graph: Graph, coins: Vec<CMat>, j: f64) -> Self {
        assert_eq!(coins.len(), graph.nodes(), "one coin per node");
        for (x, c) in coins.iter().enumerate() {
            let d = graph.degree(x);
            assert!(c.nrows() == d && c.ncols() == d, "node {x}: the coin must be {d} × {d}");
        }
        assert!(graph.edges() < usize::BITS as usize - 8, "too many spins");
        Self { graph, coins, j }
    }

    /// The one-dimensional lattice of the paper: a ring of `n` nodes, coin `R(θ)` everywhere.
    pub fn ring(n: usize, theta: f64, j: f64) -> Self {
        Self::new(Graph::cycle(n), vec![rotation(theta); n], j)
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn coins(&self) -> &[CMat] {
        &self.coins
    }

    /// Particle–spin coupling `J`.
    pub fn coupling(&self) -> f64 {
        self.j
    }

    /// Spin configurations per half-edge, `2^|E|`.
    pub fn row_len(&self) -> usize {
        1usize << self.graph.edges()
    }

    /// Dimension of the Hilbert space, `2|E| · 2^|E|`.
    pub fn dimension(&self) -> usize {
        self.graph.half_edges() * self.row_len()
    }

    /// The edge part `V(J) M` on the four amplitudes `(a0, a1, b0, b1)` of an edge: half-edges
    /// `a`, `b`, edge spin 0 or 1. Returns `(p, α, β)` such that
    /// `a0' = p b0`, `a1' = α b1 + β a0`, `b0' = β b1 + α a0`, `b1' = p a1`.
    pub fn exchange(&self) -> (c64, c64, c64) {
        let j = self.j;
        let q = c64::from_polar(1.0, -j / 4.0);
        (c64::from_polar(1.0, j / 4.0), q * (j / 2.0).cos(), q * c64::new(0.0, (j / 2.0).sin()))
    }

    /// One step `ψ ← U ψ`, on all the threads of the rayon pool (f64 or f32 state).
    pub fn step<T: Real>(&self, psi: &mut [Complex<T>]) {
        assert_eq!(psi.len(), self.dimension(), "state of the wrong dimension");
        self.coin(psi, true);
        self.edges(psi, true);
    }

    /// `steps` steps.
    pub fn steps<T: Real>(&self, psi: &mut [Complex<T>], steps: usize) {
        for _ in 0..steps {
            self.step(psi);
        }
    }

    /// One step on the calling thread only: for many small walks run side by side (a scan of
    /// parameters), each state staying in its core's cache.
    pub fn step_serial<T: Real>(&self, psi: &mut [Complex<T>]) {
        assert_eq!(psi.len(), self.dimension(), "state of the wrong dimension");
        self.coin(psi, false);
        self.edges(psi, false);
    }

    /// `steps` steps on the calling thread only.
    pub fn steps_serial<T: Real>(&self, psi: &mut [Complex<T>], steps: usize) {
        for _ in 0..steps {
            self.step_serial(psi);
        }
    }

    /// `ψ ← C ψ`, the coin part of a step.
    pub fn coin_pass<T: Real>(&self, psi: &mut [Complex<T>]) {
        self.coin(psi, true);
    }

    /// `ψ ← V(J) M ψ`, the motion and interaction part of a step.
    pub fn edge_pass<T: Real>(&self, psi: &mut [Complex<T>]) {
        self.edges(psi, true);
    }

    /// `ψ ← C ψ`: tasks are (node, range of spin configurations).
    fn coin<T: Real>(&self, psi: &mut [Complex<T>], parallel: bool) {
        let l = self.row_len();
        let chunks = l.div_ceil(CHUNK);
        let ptr = Shared(psi.as_mut_ptr());
        run_tasks(parallel, self.graph.nodes() * chunks, |task| {
            let (x, k) = (task / chunks, task % chunks);
            let (s0, s1) = (k * CHUNK, ((k + 1) * CHUNK).min(l));
            let (rows, coin) = (self.graph.half_edges_of(x), &self.coins[x]);
            let len = s1 - s0;
            // SAFETY: task (x, k) owns the entries s0..s1 of the rows of node x.
            let row = |h: usize| unsafe { ptr.slice(h * l + s0, len) };
            let c = |i: usize, j: usize| narrow::<T>(coin[(i, j)]);
            match rows.len() {
                1 => {
                    let c = c(0, 0);
                    row(rows.start).iter_mut().for_each(|v| *v = *v * c);
                }
                2 => {
                    let (u, v) = (row(rows.start), row(rows.start + 1));
                    if (0..4).all(|k| coin[(k / 2, k % 2)].im == 0.0) {
                        // A real coin (the rotation R(θ)): half the multiplications.
                        let (c00, c01, c10, c11) = (c(0, 0).re, c(0, 1).re, c(1, 0).re, c(1, 1).re);
                        for (u, v) in u.iter_mut().zip(v.iter_mut()) {
                            let (a, b) = (*u, *v);
                            *u = a * c00 + b * c01;
                            *v = a * c10 + b * c11;
                        }
                    } else {
                        let (c00, c01, c10, c11) = (c(0, 0), c(0, 1), c(1, 0), c(1, 1));
                        for (u, v) in u.iter_mut().zip(v.iter_mut()) {
                            let (a, b) = (*u, *v);
                            *u = c00 * a + c01 * b;
                            *v = c10 * a + c11 * b;
                        }
                    }
                }
                d => {
                    let matrix: Vec<Complex<T>> = (0..d * d).map(|k| c(k / d, k % d)).collect();
                    let mut views: Vec<&mut [Complex<T>]> = rows.map(row).collect();
                    let zero = Complex::new(T::zero(), T::zero());
                    let mut buf = vec![zero; d];
                    for i in 0..len {
                        for (b, v) in buf.iter_mut().zip(&views) {
                            *b = v[i];
                        }
                        for (r, v) in views.iter_mut().enumerate() {
                            v[i] = (0..d).fold(zero, |acc, k| acc + matrix[r * d + k] * buf[k]);
                        }
                    }
                }
            }
        });
    }

    /// `(p, α, β)` in the precision of the state.
    fn exchange_in<T: Real>(&self) -> Exchange<T> {
        let (p, alpha, beta) = self.exchange();
        (narrow(p), narrow(alpha), narrow(beta))
    }

    /// `ψ ← V(J) M ψ`: tasks are (edge, range of configurations), see [`Self::edge_task`].
    fn edges<T: Real>(&self, psi: &mut [Complex<T>], parallel: bool) {
        let l = self.row_len();
        let chunks = l / l.min(2 * CHUNK);
        let k = self.exchange_in::<T>();
        let ptr = Shared(psi.as_mut_ptr());
        run_tasks(parallel, self.graph.edges() * chunks, |task| {
            // SAFETY: the tasks own disjoint entries.
            unsafe { self.edge_task::<T, false>(ptr, task, k) };
        });
    }

    /// One step on the calling thread, adding the probability of every half-edge after the step
    /// to `acc` (`Σ_s |ψ(h, s)|²`, one entry per half-edge) at no extra sweep over memory: for
    /// time averages in scans of parameters.
    pub fn step_serial_accumulate<T: Real>(&self, psi: &mut [Complex<T>], acc: &mut [f64]) {
        assert_eq!(psi.len(), self.dimension(), "state of the wrong dimension");
        assert_eq!(acc.len(), self.graph.half_edges(), "one accumulator per half-edge");
        self.coin(psi, false);
        let l = self.row_len();
        let chunks = l / l.min(2 * CHUNK);
        let k = self.exchange_in::<T>();
        let ptr = Shared(psi.as_mut_ptr());
        for task in 0..self.graph.edges() * chunks {
            // SAFETY: one task at a time.
            let (ha, hb, pa, pb) = unsafe { self.edge_task::<T, true>(ptr, task, k) };
            acc[ha] += pa;
            acc[hb] += pb;
        }
    }

    /// Task `task` of the edge part: edge `e`, range `c` of configurations. Every task updates
    /// `CHUNK` quadruples: a whole range of `2 CHUNK` configurations when the edge spin is a low
    /// bit, or `CHUNK` configurations with the spin up and their partners when it is a high
    /// bit. Returns the edge's half-edges and, with `ACC`, the probabilities of their updated
    /// entries.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a state of this walk, and no other live reference may touch the
    /// entries of this task.
    unsafe fn edge_task<T: Real, const ACC: bool>(
        &self,
        ptr: Shared<T>,
        task: usize,
        k: Exchange<T>,
    ) -> (usize, usize, f64, f64) {
        let l = self.row_len();
        let span = l.min(2 * CHUNK);
        let chunks = l / span;
        let (e, c) = (task / chunks, task % chunks);
        let (ha, hb) = self.graph.halves(e);
        let bit = 1usize << e;
        let (pa, pb) = if 2 * bit <= span {
            // SAFETY: per the caller.
            let (a, b) = unsafe { (ptr.slice(ha * l + c * span, span), ptr.slice(hb * l + c * span, span)) };
            match bit {
                1 => edge_blocks::<T, 1, ACC>(a, b, k),
                2 => edge_blocks::<T, 2, ACC>(a, b, k),
                4 => edge_blocks::<T, 4, ACC>(a, b, k),
                _ => {
                    let (mut sa, mut sb) = (0.0, 0.0);
                    for (a, b) in a.chunks_exact_mut(2 * bit).zip(b.chunks_exact_mut(2 * bit)) {
                        let ((a0, a1), (b0, b1)) = (a.split_at_mut(bit), b.split_at_mut(bit));
                        let (x, y) = exchange_run::<T, ACC>(a0, a1, b0, b1, k);
                        sa += x;
                        sb += y;
                    }
                    (sa, sb)
                }
            }
        } else {
            // Spin up at s..s + run, down at s + bit..: one run of a block of 2 bit.
            let run = span / 2;
            let first = c * run;
            let s = ((first >> e) << (e + 1)) | (first & (bit - 1));
            // SAFETY: per the caller.
            let (a0, a1, b0, b1) = unsafe {
                (
                    ptr.slice(ha * l + s, run),
                    ptr.slice(ha * l + s + bit, run),
                    ptr.slice(hb * l + s, run),
                    ptr.slice(hb * l + s + bit, run),
                )
            };
            exchange_run::<T, ACC>(a0, a1, b0, b1, k)
        };
        (ha, hb, pa, pb)
    }

    /// The one-step operator `U` as a sparse matrix (at most `2 d_max` entries per column).
    pub fn matrix(&self) -> SparseMatrix {
        let l = self.row_len();
        let (p, alpha, beta) = self.exchange();
        // V M on |h, s⟩.
        let vm = |h: usize, s: usize| -> Vec<(usize, usize, c64)> {
            let e = self.graph.edge_of(h);
            let (ha, hb) = self.graph.halves(e);
            let bit = 1usize << e;
            let down = s & bit != 0;
            match (h == ha, down) {
                (true, false) => vec![(ha, s | bit, beta), (hb, s, alpha)],
                (true, true) => vec![(hb, s, p)],
                (false, false) => vec![(ha, s, p)],
                (false, true) => vec![(ha, s, alpha), (hb, s & !bit, beta)],
            }
        };
        let mut entries = Vec::with_capacity(self.dimension() * 2 * self.graph.max_degree());
        for x in 0..self.graph.nodes() {
            let rows = self.graph.half_edges_of(x);
            for (c, h) in rows.clone().enumerate() {
                for s in 0..l {
                    for (r, hr) in rows.clone().enumerate() {
                        let w = self.coins[x][(r, c)];
                        if w == c64::new(0.0, 0.0) {
                            continue;
                        }
                        for (ho, so, v) in vm(hr, s) {
                            entries.push((ho * l + so, h * l + s, v * w));
                        }
                    }
                }
            }
        }
        SparseMatrix::from_triplets(self.dimension(), entries)
    }

    /// The product state `walker ⊗ spins[0] ⊗ … ⊗ spins[|E| − 1]`, with `walker` the amplitude
    /// on each half-edge and `spins[e]` the (up, down) amplitudes of spin `e`.
    pub fn product_state(&self, walker: &[c64], spins: &[[c64; 2]]) -> Vec<c64> {
        assert_eq!(walker.len(), self.graph.half_edges(), "one walker amplitude per half-edge");
        assert_eq!(spins.len(), self.graph.edges(), "one spin state per edge");
        let mut config = vec![c64::new(1.0, 0.0)];
        for spin in spins {
            let down: Vec<c64> = config.iter().map(|v| v * spin[1]).collect();
            config.iter_mut().for_each(|v| *v *= spin[0]);
            config.extend(down);
        }
        let l = self.row_len();
        let mut psi = vec![c64::new(0.0, 0.0); self.dimension()];
        psi.par_chunks_mut(l).zip(walker.par_iter()).for_each(|(row, &w)| {
            if w != c64::new(0.0, 0.0) {
                row.iter_mut().zip(&config).for_each(|(v, &c)| *v = w * c);
            }
        });
        psi
    }

    /// Walker on color `c` of node `x` (a half-edge), as the `walker` of [`Self::product_state`].
    pub fn localized(&self, x: usize, c: usize) -> Vec<c64> {
        let mut w = vec![c64::new(0.0, 0.0); self.graph.half_edges()];
        w[self.graph.half_edge(x, c)] = c64::new(1.0, 0.0);
        w
    }
}

/// Spin up `|0⟩`.
pub const UP: [c64; 2] = [c64::new(1.0, 0.0), c64::new(0.0, 0.0)];

/// Spin along +x, `|+⟩ = (|0⟩ + |1⟩)/√2`.
pub const PLUS: [c64; 2] =
    [c64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0), c64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0)];

#[cfg(test)]
mod tests {
    use super::*;

    fn random_state(n: usize, seed: u64) -> Vec<c64> {
        let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let v: Vec<c64> = (0..n).map(|_| c64::new(next(), next())).collect();
        let norm = v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        v.iter().map(|z| z / norm).collect()
    }

    fn distance(a: &[c64], b: &[c64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).norm_sqr()).sum::<f64>().sqrt()
    }

    /// The structured step and the explicit matrix agree, on rings and on a graph of degree 3.
    #[test]
    fn step_matches_matrix() {
        let walks = [
            SpinWalk::ring(3, 0.4, 0.9),
            SpinWalk::ring(5, std::f64::consts::FRAC_PI_3, 1.0),
            SpinWalk::new(
                Graph::from_edges(4, &[(0, 1), (0, 2), (0, 3), (1, 2)]),
                vec![
                    fourier(3),
                    grover(2),
                    rotation(0.3),
                    CMat::from_fn(1, 1, |_, _| c64::from_polar(1.0, 0.7)),
                ],
                0.6,
            ),
        ];
        for (k, walk) in walks.iter().enumerate() {
            let u = walk.matrix();
            let psi = random_state(walk.dimension(), k as u64 + 1);
            let mut stepped = psi.clone();
            walk.step(&mut stepped);
            let mut product = vec![c64::new(0.0, 0.0); psi.len()];
            u.apply(&psi, &mut product);
            assert!(distance(&stepped, &product) < 1e-13, "walk {k}");
        }
    }

    /// `U` is unitary: its columns are orthonormal.
    #[test]
    fn unitary() {
        let walk = SpinWalk::ring(3, 0.8, 2.1);
        let n = walk.dimension();
        let columns: Vec<Vec<c64>> = (0..n)
            .map(|j| {
                let mut v = vec![c64::new(0.0, 0.0); n];
                v[j] = c64::new(1.0, 0.0);
                walk.step(&mut v);
                v
            })
            .collect();
        for i in 0..n {
            for j in 0..n {
                let dot: c64 = columns[i].iter().zip(&columns[j]).map(|(a, b)| a.conj() * b).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).norm() < 1e-13, "columns {i}, {j}");
            }
        }
    }

    /// Without coupling the spins are spectators: the walker follows the free walk on 2|E|
    /// amplitudes, whatever the spin state.
    #[test]
    fn free_walk() {
        let (n, theta) = (6, 0.7);
        let walk = SpinWalk::ring(n, theta, 0.0);
        let spins = vec![PLUS; n];
        let mut psi = walk.product_state(&walk.localized(2, 0), &spins);
        // The free walk alone: no spins, the same graph.
        let free = SpinWalk::ring(n, theta, 0.0);
        let mut w = walk.localized(2, 0);
        for _ in 0..25 {
            walk.step(&mut psi);
            // Free step on the walker amplitudes: coin then swap of the edge half-edges.
            let mut next = w.clone();
            for x in 0..n {
                let (a, b) = (w[2 * x], w[2 * x + 1]);
                let c = &free.coins()[x];
                next[2 * x] = c[(0, 0)] * a + c[(0, 1)] * b;
                next[2 * x + 1] = c[(1, 0)] * a + c[(1, 1)] * b;
            }
            for e in 0..n {
                let (ha, hb) = free.graph().halves(e);
                next.swap(ha, hb);
            }
            w = next;
            let expected = walk.product_state(&w, &spins);
            assert!(distance(&psi, &expected) < 1e-12);
        }
    }

    #[test]
    fn single_precision() {
        let walk = SpinWalk::ring(8, 0.7, 1.1);
        let psi = random_state(walk.dimension(), 13);
        let mut double = psi.clone();
        let mut single = to_single(&psi);
        walk.steps(&mut double, 50);
        walk.steps(&mut single, 50);
        let error = distance(&to_double(&single), &double);
        assert!(error > 1e-12 && error < 1e-4, "f32 after 50 steps: {error:e}");
    }

    #[test]
    fn accumulated_probabilities() {
        let walk = SpinWalk::ring(7, 0.9, 0.4);
        let mut psi = random_state(walk.dimension(), 11);
        let mut acc = vec![0.0; walk.graph().half_edges()];
        walk.step_serial_accumulate(&mut psi, &mut acc);
        let mut stepped = random_state(walk.dimension(), 11);
        walk.step(&mut stepped);
        assert!(distance(&psi, &stepped) < 1e-15);
        for (a, b) in acc.iter().zip(&walk.half_edge_probabilities(&psi)) {
            assert!((a - b).abs() < 1e-14);
        }
    }

    #[test]
    fn serial_matches_parallel() {
        let walk = SpinWalk::ring(9, 0.6, 1.3);
        let mut a = random_state(walk.dimension(), 5);
        let mut b = a.clone();
        walk.steps(&mut a, 7);
        walk.steps_serial(&mut b, 7);
        assert!(distance(&a, &b) < 1e-15);
    }

    #[test]
    fn product_states() {
        let walk = SpinWalk::ring(3, 0.1, 0.2);
        let psi = walk.product_state(&walk.localized(1, 0), &[UP, PLUS, UP]);
        let l = walk.row_len();
        let h = walk.graph().half_edge(1, 0);
        // Spin 1 is the middle bit: configurations 0 and 2 carry 1/√2.
        assert!((psi[h * l] - PLUS[0]).norm() < 1e-15 && (psi[h * l + 2] - PLUS[1]).norm() < 1e-15);
        assert!(psi.iter().map(|z| z.norm_sqr()).sum::<f64>() - 1.0 < 1e-15);
    }
}
