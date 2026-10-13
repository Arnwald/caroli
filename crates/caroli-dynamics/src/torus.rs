//! Time evolution of a periodic 2D [`Model`] on an `n × n` torus under a uniform vector potential,
//! by the split-operator method.
//!
//! The translation-invariant part `M(k, A)` (every hopping, with the Peierls factor
//! `exp(−i A·d)` of its bond) is diagonal in k; the site-dependent on-site terms of the model
//! (disorder, potentials) act in real space. A step is `U_V(dt/2) U_K U_V(dt/2)` with
//! `U_K(k) = exp(−i M(k, A(t + dt/2)) dt)`: second order in `dt`. Without site-dependent terms the
//! wave functions stay in k space and every k evolves on its own.
//!
//! The observables are those of a non-interacting electron gas: the current density, the density
//! of carriers excited above the Fermi energy of the instantaneous clean bands (pair creation by
//! a field), the width of a wave packet and the densities. A filled Fermi sea enters as a
//! stochastic trace ([`Torus::filled`]): a random-phase superposition of the occupied band
//! states, whose expectation values estimate the traces over the occupied states.
//!
//! Wave functions are stored component by component, `ψ[x + n·y + n²·c]`, `c` running over the
//! orbitals of the unit cell (every sublattice), as in [`caroli_torus`].

use crate::driven::{VectorPotential, constant_force};
use crate::{apply_local, exp_i};
use caroli_disorder::rng::Xoshiro256;
use caroli_lattice::{Region, Site, Vec3};
use caroli_model::{CMat, Model, c64};
use caroli_torus::{Fft2, k_value};
use faer::Side;
use rayon::prelude::*;

/// A hopping block of the unit cell: `⟨row| H |col⟩` towards the cell `images` away.
#[derive(Clone, Debug)]
struct Hop {
    row: usize,
    col: usize,
    images: [i64; 2],
    displacement: Vec3,
    matrix: CMat,
}

/// A band state at one k: energy, orbitals of its group, amplitudes on them.
type BandState = (f64, Vec<usize>, Vec<c64>);

/// A periodic 2D model on an `n × n` torus of unit cells.
pub struct Torus {
    n: usize,
    comps: usize,
    vectors: [Vec3; 2],
    /// Position, in the cell, of the site of each orbital.
    offsets: Vec<Vec3>,
    onsite: CMat,
    hops: Vec<Hop>,
    /// Groups of orbitals that `M(k)` never couples to one another.
    groups: Vec<Vec<usize>>,
    /// Site-dependent on-site terms of each cell (`x + n·y`), on all the orbitals of the cell.
    local: Vec<(usize, CMat)>,
    potential: VectorPotential,
    fft: Fft2,
}

/// The parameters of [`Torus::evolve`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Evolution {
    pub dt: f64,
    pub steps: usize,
    /// Observables every `every` steps (and at the start).
    pub every: usize,
    /// Fermi energy of the excited-carrier count.
    pub fermi: f64,
    /// Whether to record the densities.
    pub densities: bool,
}

/// The observables recorded by [`Torus::evolve`], averaged over the start vectors.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Record {
    pub times: Vec<f64>,
    /// `⟨ψ|ψ⟩`.
    pub norm: Vec<f64>,
    /// Current density `⟨v⟩/Ω` (`v = i[H, r]`, `Ω` the area of the torus), x and y.
    pub current: Vec<[f64; 2]>,
    /// Density of carriers in the instantaneous clean bands above the Fermi energy, per unit
    /// area.
    pub excited: Vec<f64>,
    /// `sqrt(⟨Δr²⟩ − ⟨Δr⟩²)`, positions taken around the densest orbital at the start (nearest
    /// image).
    pub width: Vec<f64>,
    /// `|ψ|²` of every orbital (`x + n·y + n²·c`), if asked.
    pub densities: Vec<Vec<f64>>,
}

impl Torus {
    /// The model on an `n × n` torus, without field. Its hoppings must be the same in every cell
    /// (no hopping term, no Peierls phase: use [`crate::Driven`] otherwise); its site-dependent
    /// on-site terms are read on every site of the torus.
    pub fn new(model: &Model, n: usize) -> Self {
        let lattice = model.lattice();
        assert_eq!(lattice.dim(), 2, "a torus of a 2D lattice");
        assert!(
            model.has_uniform_hoppings(),
            "the torus needs hoppings without site-dependent terms or Peierls phases"
        );
        let cell = Region::cuboid(lattice, [1, 1, 0]);
        let comps = cell.orbitals();
        let mut onsite = CMat::zeros(comps, comps);
        let mut offsets = Vec::with_capacity(comps);
        for (i, site) in cell.sites().iter().enumerate() {
            let (o, m) = (cell.offset(i), model.sublattice_onsite(site.sub));
            add_block(&mut onsite, o, o, m, c64::new(1.0, 0.0));
            offsets.extend(std::iter::repeat_n(lattice.position(*site), m.nrows()));
        }
        let hops: Vec<Hop> = model
            .bonds(&cell, &[[1, 0, 0], [0, 1, 0]])
            .into_iter()
            .map(|b| Hop {
                row: cell.offset(b.from),
                col: cell.offset(b.to),
                images: [b.images[0], b.images[1]],
                displacement: b.displacement,
                matrix: b.matrix,
            })
            .collect();
        // Orbitals coupled by M(k): union-find on the non-zero entries.
        let mut parent: Vec<usize> = (0..comps).collect();
        fn root(p: &mut [usize], mut i: usize) -> usize {
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        let mut link = |a: usize, b: usize| {
            let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
            parent[ra] = rb;
        };
        for a in 0..comps {
            for b in 0..comps {
                if onsite[(a, b)] != c64::new(0.0, 0.0) {
                    link(a, b);
                }
            }
        }
        for h in &hops {
            for a in 0..h.matrix.nrows() {
                for b in 0..h.matrix.ncols() {
                    if h.matrix[(a, b)] != c64::new(0.0, 0.0) {
                        link(h.row + a, h.col + b);
                    }
                }
            }
        }
        let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
        for a in 0..comps {
            groups.entry(root(&mut parent, a)).or_default().push(a);
        }
        let mut local = Vec::new();
        for y in 0..n as i64 {
            for x in 0..n as i64 {
                let mut block: Option<CMat> = None;
                for (i, s) in cell.sites().iter().enumerate() {
                    if let Some(m) = model.onsite_terms(&Site { cell: [x, y, 0], sub: s.sub }) {
                        let b = block.get_or_insert_with(|| CMat::zeros(comps, comps));
                        add_block(b, cell.offset(i), cell.offset(i), &m, c64::new(1.0, 0.0));
                    }
                }
                if let Some(b) = block {
                    local.push((x as usize + n * y as usize, b));
                }
            }
        }
        let v = lattice.vectors();
        Self {
            n,
            comps,
            vectors: [v[0], v[1]],
            offsets,
            onsite,
            hops,
            groups: groups.into_values().collect(),
            local,
            potential: std::sync::Arc::new(|_| [0.0; 3]),
            fft: Fft2::new(n),
        }
    }

    /// A constant force `F` from `t = 0`: `A(t) = −F t`, the crystal momentum moves as `k + F t`.
    pub fn with_force(self, force: Vec3) -> Self {
        self.with_vector_potential(constant_force(force))
    }

    /// Any uniform vector potential `A(t)`.
    pub fn with_vector_potential(mut self, potential: VectorPotential) -> Self {
        self.potential = potential;
        self
    }

    pub fn size(&self) -> usize {
        self.n
    }

    /// Orbitals per unit cell.
    pub fn components(&self) -> usize {
        self.comps
    }

    /// Area of the torus.
    pub fn area(&self) -> f64 {
        let [a, b] = self.vectors;
        (a[0] * b[1] - a[1] * b[0]).abs() * (self.n * self.n) as f64
    }

    /// `M(k, A)` at the k point of FFT index `k` (`kx + n·ky`).
    pub fn kinetic(&self, k: usize, a: Vec3) -> CMat {
        self.k_matrix(k, a, None)
    }

    /// The velocity `∂M/∂k` along `direction` at k index `k`: `Σ i (d·direction) t e^{i(k·n − A·d)}`.
    pub fn velocity(&self, k: usize, a: Vec3, direction: Vec3) -> CMat {
        self.k_matrix(k, a, Some(direction))
    }

    fn k_matrix(&self, k: usize, a: Vec3, direction: Option<Vec3>) -> CMat {
        let n = self.n;
        let (p1, p2) = (k_value(k % n, n), k_value(k / n, n));
        let mut m =
            if direction.is_some() { CMat::zeros(self.comps, self.comps) } else { self.onsite.clone() };
        for h in &self.hops {
            let d = h.displacement;
            let phase =
                p1 * h.images[0] as f64 + p2 * h.images[1] as f64 - (a[0] * d[0] + a[1] * d[1] + a[2] * d[2]);
            let factor = match direction {
                None => c64::from_polar(1.0, phase),
                Some(u) => {
                    c64::new(0.0, d[0] * u[0] + d[1] * u[1] + d[2] * u[2]) * c64::from_polar(1.0, phase)
                }
            };
            add_block(&mut m, h.row, h.col, &h.matrix, factor);
        }
        m
    }

    /// Band energies and states of `M(k, A)`, group by group: `(energy, orbitals of the group,
    /// state on them)`.
    fn bands(&self, m: &CMat) -> Vec<(f64, &[usize], Vec<c64>)> {
        let mut out = Vec::with_capacity(self.comps);
        for g in &self.groups {
            let sub = CMat::from_fn(g.len(), g.len(), |i, j| m[(g[i], g[j])]);
            let evd = sub.self_adjoint_eigen(Side::Lower).expect("Hermitian eigenproblem");
            for b in 0..g.len() {
                out.push((evd.S()[b].re, g.as_slice(), (0..g.len()).map(|i| evd.U()[(i, b)]).collect()));
            }
        }
        out
    }

    /// `exp(−i M dt)`, group by group (two-orbital groups in closed form).
    fn propagator(&self, m: &CMat, dt: f64) -> CMat {
        let mut u = CMat::zeros(self.comps, self.comps);
        for g in &self.groups {
            match g.len() {
                1 => u[(g[0], g[0])] = c64::from_polar(1.0, -m[(g[0], g[0])].re * dt),
                2 => {
                    let (a, b, d) = (m[(g[0], g[0])].re, m[(g[0], g[1])], m[(g[1], g[1])].re);
                    let (mean, z) = (0.5 * (a + d), 0.5 * (a - d));
                    let r = (z * z + b.norm_sqr()).sqrt();
                    let (c, s) = ((r * dt).cos(), if r > 0.0 { (r * dt).sin() / r } else { dt });
                    let phase = c64::from_polar(1.0, -mean * dt);
                    let mi = c64::new(0.0, -s);
                    u[(g[0], g[0])] = phase * (c + mi * z);
                    u[(g[1], g[1])] = phase * (c - mi * z);
                    u[(g[0], g[1])] = phase * mi * b;
                    u[(g[1], g[0])] = phase * mi * b.conj();
                }
                _ => {
                    let sub = CMat::from_fn(g.len(), g.len(), |i, j| m[(g[i], g[j])]);
                    let e = exp_i(&sub, dt);
                    for (i, &gi) in g.iter().enumerate() {
                        for (j, &gj) in g.iter().enumerate() {
                            u[(gi, gj)] = e[(i, j)];
                        }
                    }
                }
            }
        }
        u
    }

    /// Forward FFT of every component.
    pub fn to_k(&self, psi: &mut [c64]) {
        psi.chunks_exact_mut(self.n * self.n).for_each(|c| self.fft.forward(c));
    }

    /// Backward FFT of every component divided by n² (inverse of [`Self::to_k`]).
    pub fn to_x(&self, psi: &mut [c64]) {
        let norm = (self.n * self.n) as f64;
        psi.chunks_exact_mut(self.n * self.n).for_each(|c| {
            self.fft.backward(c);
            c.iter_mut().for_each(|z| *z /= norm);
        });
    }

    /// `ψ_k ← f(k, ψ_k)` for every k, in parallel (`ψ` in k space, component by component).
    fn per_k(&self, psi: &mut [c64], f: impl Fn(usize, &mut [c64]) + Sync) {
        let (nn, m) = (self.n * self.n, self.comps);
        let mut by_k = vec![c64::new(0.0, 0.0); nn * m];
        for c in 0..m {
            for k in 0..nn {
                by_k[k * m + c] = psi[k + nn * c];
            }
        }
        by_k.par_chunks_exact_mut(m).enumerate().with_min_len(64).for_each(|(k, v)| f(k, v));
        for c in 0..m {
            for k in 0..nn {
                psi[k + nn * c] = by_k[k * m + c];
            }
        }
    }

    /// `Σ_k f(k, ψ_k)` in parallel.
    fn sum_k<T: Send + std::iter::Sum<T>>(&self, psi: &[c64], f: impl Fn(usize, &[c64]) -> T + Sync) -> T {
        let (nn, m) = (self.n * self.n, self.comps);
        (0..nn)
            .into_par_iter()
            .with_min_len(64)
            .map(|k| {
                let v: Vec<c64> = (0..m).map(|c| psi[k + nn * c]).collect();
                f(k, &v)
            })
            .sum()
    }

    /// The Fermi sea below `fermi` of the clean bands at `t = 0`, as `vectors` random-phase
    /// superpositions of the occupied band states (k space): for each of them `⟨ψ|O|ψ⟩`
    /// estimates `Tr[P O]`, `P` the projector on the occupied states, and `⟨ψ|ψ⟩` is their number.
    pub fn filled(&self, fermi: f64, vectors: usize, seed: u64) -> Vec<Vec<c64>> {
        let (nn, a0) = (self.n * self.n, (self.potential)(0.0));
        let bands: Vec<Vec<BandState>> = (0..nn)
            .into_par_iter()
            .map(|k| {
                let m = self.kinetic(k, a0);
                self.bands(&m).into_iter().map(|(e, g, u)| (e, g.to_vec(), u)).collect()
            })
            .collect();
        let mut rng = Xoshiro256::seed_from_u64(seed);
        (0..vectors)
            .map(|_| {
                let mut psi = vec![c64::new(0.0, 0.0); nn * self.comps];
                for (k, bk) in bands.iter().enumerate() {
                    for (e, g, u) in bk {
                        let phase =
                            c64::from_polar(self.n as f64, 2.0 * std::f64::consts::PI * rng.uniform());
                        if *e < fermi {
                            for (i, &c) in g.iter().enumerate() {
                                psi[k + nn * c] += phase * u[i];
                            }
                        }
                    }
                }
                psi
            })
            .collect()
    }

    /// The orbital `orbital` of cell `cell` (k space, unit norm), projected on the clean band
    /// states at `t = 0` whose energy lies in `window` if one is given.
    pub fn localized(&self, cell: [usize; 2], orbital: usize, window: Option<(f64, f64)>) -> Vec<c64> {
        let nn = self.n * self.n;
        let mut psi = vec![c64::new(0.0, 0.0); nn * self.comps];
        psi[cell[0] + self.n * cell[1] + nn * orbital] = c64::new(1.0, 0.0);
        self.to_k(&mut psi);
        if let Some((lo, hi)) = window {
            let a0 = (self.potential)(0.0);
            self.per_k(&mut psi, |k, v| {
                let bands = self.bands(&self.kinetic(k, a0));
                let mut out = vec![c64::new(0.0, 0.0); v.len()];
                for (e, g, u) in bands {
                    if e >= lo && e <= hi {
                        let overlap: c64 = g.iter().zip(&u).map(|(&c, x)| x.conj() * v[c]).sum();
                        g.iter().zip(&u).for_each(|(&c, x)| out[c] += overlap * x);
                    }
                }
                v.copy_from_slice(&out);
            });
            let norm = (psi.iter().map(|z| z.norm_sqr()).sum::<f64>() / nn as f64).sqrt();
            assert!(norm > 0.0, "no band state in the window");
            psi.iter_mut().for_each(|z| *z /= norm);
        }
        psi
    }

    /// One step `ψ(t) → ψ(t + dt)` of a state in k space (no site-dependent term).
    fn kinetic_step(&self, psi_k: &mut [c64], t: f64, dt: f64) {
        let a = (self.potential)(t + 0.5 * dt);
        self.per_k(psi_k, |k, v| {
            let u = self.propagator(&self.kinetic(k, a), dt);
            let w: Vec<c64> = (0..v.len()).map(|i| (0..v.len()).map(|j| u[(i, j)] * v[j]).sum()).collect();
            v.copy_from_slice(&w);
        });
    }

    /// Evolves each start state (k space, e.g. from [`Self::filled`] or [`Self::localized`]) from
    /// `t = 0` and records the observables, averaged over the start states.
    pub fn evolve(&self, starts: &[Vec<c64>], ev: &Evolution) -> Record {
        assert!(!starts.is_empty() && ev.every > 0);
        let nn = self.n * self.n;
        let half: Vec<(usize, CMat)> = self.local.iter().map(|(s, v)| (*s, exp_i(v, 0.5 * ev.dt))).collect();
        let reference = {
            let mut x = starts[0].clone();
            self.to_x(&mut x);
            let (best, _) = x
                .iter()
                .enumerate()
                .fold((0, -1.0), |(b, m), (i, z)| if z.norm_sqr() > m { (i, z.norm_sqr()) } else { (b, m) });
            best
        };
        let mut record = Record::default();
        for (s, start) in starts.iter().enumerate() {
            let mut psi = start.clone();
            let in_x = !self.local.is_empty();
            if in_x {
                self.to_x(&mut psi);
            }
            let mut row = 0;
            for step in 0..=ev.steps {
                let t = step as f64 * ev.dt;
                if step % ev.every == 0 {
                    let (mut k, mut x) = (psi.clone(), psi.clone());
                    if in_x {
                        self.to_k(&mut k)
                    } else {
                        self.to_x(&mut x)
                    }
                    let m = self.measure(&k, &x, t, ev, reference);
                    if s == 0 {
                        record.times.push(t);
                        record.norm.push(0.0);
                        record.current.push([0.0; 2]);
                        record.excited.push(0.0);
                        record.width.push(0.0);
                        if ev.densities {
                            record.densities.push(vec![0.0; nn * self.comps]);
                        }
                    }
                    record.norm[row] += m.0;
                    record.current[row][0] += m.1[0];
                    record.current[row][1] += m.1[1];
                    record.excited[row] += m.2;
                    record.width[row] += m.3;
                    if ev.densities {
                        record.densities[row].iter_mut().zip(&x).for_each(|(d, z)| *d += z.norm_sqr());
                    }
                    row += 1;
                }
                if step == ev.steps {
                    break;
                }
                if in_x {
                    apply_local(&mut psi, nn, self.comps, &half);
                    self.to_k(&mut psi);
                    self.kinetic_step(&mut psi, t, ev.dt);
                    self.to_x(&mut psi);
                    apply_local(&mut psi, nn, self.comps, &half);
                } else {
                    self.kinetic_step(&mut psi, t, ev.dt);
                }
            }
        }
        let count = starts.len() as f64;
        record
            .norm
            .iter_mut()
            .chain(record.excited.iter_mut())
            .chain(record.width.iter_mut())
            .for_each(|x| *x /= count);
        record.current.iter_mut().flatten().for_each(|x| *x /= count);
        record.densities.iter_mut().flatten().for_each(|x| *x /= count);
        record
    }

    /// `(norm, current density, excited density, width)` of a state given in k and x space.
    fn measure(
        &self,
        psi_k: &[c64],
        psi_x: &[c64],
        t: f64,
        ev: &Evolution,
        reference: usize,
    ) -> (f64, [f64; 2], f64, f64) {
        let nn = (self.n * self.n) as f64;
        let a = (self.potential)(t);
        let (area, (ex, ey)) = (self.area(), ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]));
        let expectation = |m: &CMat, v: &[c64]| -> f64 {
            (0..v.len())
                .map(|i| (0..v.len()).map(|j| v[i].conj() * m[(i, j)] * v[j]).sum::<c64>())
                .sum::<c64>()
                .re
        };
        let (jx, jy, excited): (f64, f64, f64) = self
            .sum_k(psi_k, |k, v| {
                let (vx, vy) = (self.velocity(k, a, ex), self.velocity(k, a, ey));
                let up: f64 = self
                    .bands(&self.kinetic(k, a))
                    .into_iter()
                    .filter(|(e, _, _)| *e > ev.fermi)
                    .map(|(_, g, u)| g.iter().zip(&u).map(|(&c, x)| x.conj() * v[c]).sum::<c64>().norm_sqr())
                    .sum();
                Triple(expectation(&vx, v), expectation(&vy, v), up)
            })
            .into();
        let norm = psi_x.iter().map(|z| z.norm_sqr()).sum::<f64>();
        // Positions around the reference orbital, nearest image on the torus.
        let n = self.n as i64;
        let (rx, ry, rc) = (
            (reference % (n * n) as usize) as i64 % n,
            (reference % (n * n) as usize) as i64 / n,
            reference / (n * n) as usize,
        );
        let wrap = |d: i64| (d + n / 2).rem_euclid(n) - n / 2;
        let [a1, a2] = self.vectors;
        let (mut m1, mut m2) = ([0.0; 2], 0.0);
        for (i, z) in psi_x.iter().enumerate() {
            let p = z.norm_sqr();
            if p == 0.0 {
                continue;
            }
            let (cell, c) = (i % (n * n) as usize, i / (n * n) as usize);
            let (dx, dy) = (wrap(cell as i64 % n - rx) as f64, wrap(cell as i64 / n - ry) as f64);
            let r = [
                dx * a1[0] + dy * a2[0] + self.offsets[c][0] - self.offsets[rc][0],
                dx * a1[1] + dy * a2[1] + self.offsets[c][1] - self.offsets[rc][1],
            ];
            m1[0] += p * r[0];
            m1[1] += p * r[1];
            m2 += p * (r[0] * r[0] + r[1] * r[1]);
        }
        let width = if norm > 0.0 {
            (m2 / norm - (m1[0] * m1[0] + m1[1] * m1[1]) / (norm * norm)).max(0.0).sqrt()
        } else {
            0.0
        };
        (norm, [jx / (nn * area), jy / (nn * area)], excited / (nn * area), width)
    }
}

/// Three sums at once.
struct Triple(f64, f64, f64);

impl std::iter::Sum for Triple {
    fn sum<I: Iterator<Item = Triple>>(iter: I) -> Self {
        iter.fold(Triple(0.0, 0.0, 0.0), |a, b| Triple(a.0 + b.0, a.1 + b.1, a.2 + b.2))
    }
}

impl From<Triple> for (f64, f64, f64) {
    fn from(t: Triple) -> Self {
        (t.0, t.1, t.2)
    }
}

/// `m[r0.., c0..] += factor · b`.
fn add_block(m: &mut CMat, r0: usize, c0: usize, b: &CMat, factor: c64) {
    for i in 0..b.nrows() {
        for j in 0..b.ncols() {
            m[(r0 + i, c0 + j)] += factor * b[(i, j)];
        }
    }
}
