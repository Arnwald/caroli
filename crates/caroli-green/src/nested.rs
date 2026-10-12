//! Two-terminal transport by geometric nested dissection.
//!
//! The device is cut recursively by separator layers into a binary tree of domains. Each node
//! eliminates its separator from a dense frontal matrix and passes the Schur complement on its
//! boundary to its parent (the multifrontal method). The two lead slices are never eliminated.
//! The root therefore holds the Schur complement `S` of `z − H − Σ` on them, and
//! `G_LR = (S⁻¹)_LR` gives the Caroli transmission.
//!
//! Compared with the recursive Green's function ([`crate::TwoTerminal`]):
//! - the cost of an n × n sample is O(n³) instead of O(n⁴), and that of a long strip
//!   O(L W²) instead of O(L W³);
//! - only the pending Schur complements are stored, no factors;
//! - the tree is built once, and each energy redoes the numeric phase, in parallel over
//!   independent branches.
//!
//! If a separator block cannot be eliminated stably, its variables are delayed: they join the
//! separator of the parent node. This happens for an exactly singular interior block, such as a
//! piece of a clean lattice at one of its eigenvalues.
//!
//! Independent blocks (orbitals that neither the device nor the leads couple, e.g. the two
//! spin blocks of BHZ) are solved separately on the same tree.
//!
//! The elimination itself (`SiteTree`, `Tree`) also serves [`crate::NestedMultiTerminal`].

use crate::gauge::Gauge;
use crate::modes::{ModeSolver, sigma_left, sigma_right};
use crate::{GreenError, Lead, broadening, caroli_trace};
use caroli_dense::{Accum, Lu, Op, matmul, product};
use caroli_lattice::{Region, Site};
use caroli_model::{CMat, Model, SparseMatrix, c64};

/// Domains with at most this many orbitals are not split further.
const LEAF_ORBITALS: usize = 16;
/// A pivot below this fraction of the largest entry of a separator block delays the block.
const PIVOT_TOL: f64 = 1e-11;
/// Fronts with fewer variables than this are eliminated by [`small_schur`].
const SMALL_FRONT: usize = 64;

/// One node of the elimination tree, for one block of orbitals.
#[derive(Clone, Debug)]
struct Node {
    /// Orbitals eliminated at this node (sorted).
    eliminate: Vec<usize>,
    /// Orbitals of the front that are eliminated later or never (sorted).
    boundary: Vec<usize>,
    children: Vec<usize>,
}

/// Geometric tree over the sites of a region: separator layers as thick as the hopping range,
/// down to leaves of `LEAF_ORBITALS` orbitals. Shared by the independent blocks of orbitals.
#[derive(Clone, Debug)]
pub(crate) struct SiteTree {
    /// `(sites eliminated, children)` of each node, in post-order.
    nodes: Vec<(Vec<usize>, Vec<usize>)>,
    /// Children of the root.
    top: Vec<usize>,
}

impl SiteTree {
    /// Tree over the `interior` sites (indices in `region`); the other sites stay at the root.
    pub(crate) fn new(device: &Model, region: &Region, interior: Vec<usize>) -> Self {
        let mut range = [1i64; 3];
        for hop in device.hoppings() {
            for (d, r) in range.iter_mut().enumerate() {
                *r = (*r).max(hop.offset[d].abs());
            }
        }
        let mut nodes = Vec::new();
        let top = split(region, interior, &range, &mut nodes).into_iter().collect();
        Self { nodes, top }
    }

    /// Elimination tree of the orbitals for which `mine` holds.
    pub(crate) fn tree(&self, region: &Region, h: &SparseMatrix, mine: impl Fn(usize) -> bool) -> Tree {
        let block_orbitals = |sites: &[usize]| -> Vec<usize> {
            let mut v: Vec<usize> = sites
                .iter()
                .flat_map(|&i| region.offset(i)..region.offset(i) + region.site_orbitals(i))
                .filter(|&o| mine(o))
                .collect();
            v.sort_unstable();
            v
        };
        let mut nodes: Vec<Node> = self
            .nodes
            .iter()
            .map(|(sites, children)| Node {
                eliminate: block_orbitals(sites),
                boundary: Vec::new(),
                children: children.clone(),
            })
            .collect();
        symbolic(h, &mut nodes);
        let plans = (0..nodes.len()).map(|k| Plan::new(h, &nodes, k)).collect();
        Tree { nodes, top: self.top.clone(), plans }
    }
}

/// Elimination tree of one block of orbitals.
#[derive(Clone, Debug)]
pub(crate) struct Tree {
    nodes: Vec<Node>,
    /// Children of the root.
    top: Vec<usize>,
    /// The assembly of each front when nothing below it was delayed.
    plans: Vec<Plan>,
}

/// How a front is assembled when its children hand over exactly their boundaries (no delayed
/// variable): computed once with the tree, instead of sorting and searching the variables at
/// every energy.
#[derive(Clone, Debug)]
struct Plan {
    /// Variables of the front: those eliminated here (sorted), then the boundary.
    vars: Vec<usize>,
    /// Number of variables eliminated here.
    ne: usize,
    /// Positions of the own separator (they get `z` on the diagonal).
    diagonal: Vec<usize>,
    /// Entries of `H` in the front: `(row, column, value)`.
    entries: Vec<(usize, usize, c64)>,
    /// For each child, where its Schur complement goes (runs, see [`runs`]).
    children: Vec<Vec<(usize, usize, usize)>>,
}

impl Plan {
    fn new(h: &SparseMatrix, nodes: &[Node], k: usize) -> Self {
        let node = &nodes[k];
        let children: Vec<&[usize]> = node.children.iter().map(|&c| nodes[c].boundary.as_slice()).collect();
        let (vars, ne) = front_vars(node, &children);
        let pos = Positions::new(&vars);
        let mut entries = Vec::new();
        separator_entries(h, node, &pos, |a, b, v| entries.push((a, b, v)));
        Self {
            diagonal: node.eliminate.iter().map(|&i| pos.get(i).unwrap()).collect(),
            entries,
            children: children.iter().map(|vars| runs(vars, &pos)).collect(),
            vars,
            ne,
        }
    }
}

/// Variables of the front of `node` whose children hand over `children`: the own separator and
/// what the children could not eliminate (sorted), then the boundary; and how many are
/// eliminated here.
fn front_vars(node: &Node, children: &[&[usize]]) -> (Vec<usize>, usize) {
    let in_boundary = |o: usize| node.boundary.binary_search(&o).is_ok();
    let mut elim: Vec<usize> = node.eliminate.clone();
    for vars in children {
        elim.extend(vars.iter().copied().filter(|&o| !in_boundary(o)));
    }
    elim.sort_unstable();
    elim.dedup();
    let ne = elim.len();
    elim.extend_from_slice(&node.boundary);
    (elim, ne)
}

/// Entries of `H` that the separator of `node` brings to its front: with every later variable
/// (the rest of the separator after it, and the boundary); the earlier ones were assembled where
/// those were eliminated, and a child's delayed variables already carry theirs.
fn separator_entries(h: &SparseMatrix, node: &Node, pos: &Positions, mut add: impl FnMut(usize, usize, c64)) {
    let in_boundary = |o: usize| node.boundary.binary_search(&o).is_ok();
    for &i in &node.eliminate {
        let pi = pos.get(i).unwrap();
        for (j, hij) in h.row(i) {
            if j == i {
                add(pi, pi, hij);
            } else if let Some(pj) = pos.get(j) {
                let later = if node.eliminate.binary_search(&j).is_ok() { j > i } else { in_boundary(j) };
                if later {
                    add(pi, pj, hij);
                    add(pj, pi, hij.conj());
                }
            }
        }
    }
}

/// What a front passes to its parent: its remaining variables and their Schur complement, and,
/// when the factors are kept, what the back-substitution needs.
struct Front {
    vars: Vec<usize>,
    s: CMat,
    factor: Option<Factor>,
}

/// Back-substitution data of a front: with a right-hand side that vanishes on the eliminated
/// variables, `ψ_elim = −X ψ_boundary`. A delayed front eliminates nothing and only holds its
/// children.
pub(crate) struct Factor {
    pub(crate) elim: Vec<usize>,
    pub(crate) boundary: Vec<usize>,
    pub(crate) x: CMat,
    pub(crate) children: Vec<Factor>,
}

impl Factor {
    /// Fills the rows of the eliminated variables of `psi` (rows: orbitals) from those of their
    /// boundaries, from this front down to the leaves.
    pub(crate) fn back_substitute(&self, psi: &mut CMat) {
        if !self.elim.is_empty() {
            let cols = psi.ncols();
            let b = CMat::from_fn(self.boundary.len(), cols, |a, c| psi[(self.boundary[a], c)]);
            let mut e = CMat::zeros(self.elim.len(), cols);
            matmul(
                e.as_mut(),
                Accum::Replace,
                self.x.as_ref(),
                Op::Plain,
                b.as_ref(),
                Op::Plain,
                c64::new(-1.0, 0.0),
            );
            for (a, &o) in self.elim.iter().enumerate() {
                for c in 0..cols {
                    psi[(o, c)] = e[(a, c)];
                }
            }
        }
        for child in &self.children {
            child.back_substitute(psi);
        }
    }
}

/// The root front of a [`Tree`] (see [`Tree::root`]).
pub(crate) struct Root {
    /// Variables delayed up to the root (first in `f`), then the kept orbitals.
    pub(crate) delayed: Vec<usize>,
    pub(crate) f: CMat,
    /// Back-substitution data of the top fronts, when kept.
    pub(crate) factors: Vec<Factor>,
}

impl Tree {
    /// Front of the root on the `kept` orbitals, which are never eliminated: `f` holds the
    /// variables delayed up to the root, then `kept` in order. The entries of `z − H` between
    /// kept orbitals are included; the caller adds the leads. With `keep`, the fronts also keep
    /// their back-substitution data.
    pub(crate) fn root(
        &self,
        h: &SparseMatrix,
        z: c64,
        kept: &[usize],
        keep: bool,
    ) -> Result<Root, GreenError> {
        let kids = self.fronts(h, &self.top, z, keep)?;
        let in_kept = Positions::new(kept);
        let mut delayed: Vec<usize> =
            kids.iter().flat_map(|k| k.vars.iter().copied()).filter(|&o| in_kept.get(o).is_none()).collect();
        delayed.sort_unstable();
        delayed.dedup();
        let vars: Vec<usize> = delayed.iter().chain(kept).copied().collect();
        let pos = Positions::new(&vars);
        let mut f = CMat::zeros(vars.len(), vars.len());
        for &i in kept {
            let pi = pos.get(i).unwrap();
            f[(pi, pi)] += z;
            for (j, hij) in h.row(i) {
                if in_kept.get(j).is_some() {
                    f[(pi, pos.get(j).unwrap())] -= hij;
                }
            }
        }
        let factors = extend_add(&mut f, &pos, kids);
        Ok(Root { delayed, f, factors })
    }

    /// Fronts of the given nodes, in parallel.
    fn fronts(
        &self,
        h: &SparseMatrix,
        nodes: &[usize],
        z: c64,
        keep: bool,
    ) -> Result<Vec<Front>, GreenError> {
        match nodes {
            [] => Ok(Vec::new()),
            [a] => Ok(vec![self.front(h, *a, z, keep)?]),
            [a, b] => {
                let (x, y) = rayon::join(|| self.front(h, *a, z, keep), || self.front(h, *b, z, keep));
                Ok(vec![x?, y?])
            }
            _ => nodes.iter().map(|&k| self.front(h, k, z, keep)).collect(),
        }
    }

    /// Eliminates node `k` (after its children) and returns the variables left in its front and
    /// their Schur complement.
    fn front(&self, h: &SparseMatrix, k: usize, z: c64, keep: bool) -> Result<Front, GreenError> {
        let node = &self.nodes[k];
        let kids = self.fronts(h, &node.children, z, keep)?;
        let plan = &self.plans[k];
        let planned = kids.iter().zip(&node.children).all(|(kid, &c)| kid.vars == self.nodes[c].boundary);
        let (vars, ne, f, children) = if planned {
            let n = plan.vars.len();
            let mut f = CMat::zeros(n, n);
            for &(a, b, v) in &plan.entries {
                f[(a, b)] -= v;
            }
            for &a in &plan.diagonal {
                f[(a, a)] += z;
            }
            let mut factors = Vec::new();
            for (kid, runs) in kids.into_iter().zip(&plan.children) {
                add_runs(&mut f, runs, &kid.s);
                factors.extend(kid.factor);
            }
            (None, plan.ne, f, factors)
        } else {
            // Variables delayed below: assemble from the actual variables.
            let handed: Vec<&[usize]> = kids.iter().map(|kid| kid.vars.as_slice()).collect();
            let (vars, ne) = front_vars(node, &handed);
            let pos = Positions::new(&vars);
            let mut f = CMat::zeros(vars.len(), vars.len());
            separator_entries(h, node, &pos, |a, b, v| f[(a, b)] -= v);
            for &i in &node.eliminate {
                let pi = pos.get(i).unwrap();
                f[(pi, pi)] += z;
            }
            let children = extend_add(&mut f, &pos, kids);
            (Some(vars), ne, f, children)
        };
        let vars = || vars.clone().unwrap_or_else(|| plan.vars.clone());
        let delayed = |vars: Vec<usize>, f: CMat, children: Vec<Factor>| Front {
            factor: keep.then(|| Factor {
                elim: Vec::new(),
                boundary: Vec::new(),
                x: CMat::zeros(0, 0),
                children,
            }),
            vars,
            s: f,
        };
        if ne == 0 {
            return Ok(delayed(vars(), f, children));
        }
        let eliminated = if f.nrows() < SMALL_FRONT { small_schur(&f, ne, keep) } else { schur(&f, ne) };
        match eliminated {
            Some((s, x)) => Ok(Front {
                vars: node.boundary.clone(),
                s,
                factor: keep.then(|| Factor {
                    elim: vars()[..ne].to_vec(),
                    boundary: node.boundary.clone(),
                    x,
                    children,
                }),
            }),
            // Singular separator block: hand the whole front to the parent.
            None => Ok(delayed(vars(), f, children)),
        }
    }
}

/// Independent sets of indices (union-find).
pub(crate) struct Components(Vec<usize>);

impl Components {
    pub(crate) fn new(n: usize) -> Self {
        Self((0..n).collect())
    }

    fn find(&mut self, mut i: usize) -> usize {
        while self.0[i] != i {
            self.0[i] = self.0[self.0[i]];
            i = self.0[i];
        }
        i
    }

    pub(crate) fn join(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.0[ra.max(rb)] = ra.min(rb);
        }
    }

    /// Joins `orbitals[a]` and `orbitals[b]` for every nonzero `m[(a, b)]`.
    pub(crate) fn join_matrix(&mut self, m: &CMat, rows: &[usize], cols: &[usize]) {
        for b in 0..m.ncols() {
            for a in 0..m.nrows() {
                if m[(a, b)] != c64::new(0.0, 0.0) {
                    self.join(rows[a], cols[b]);
                }
            }
        }
    }

    /// The label of every index (its smallest member), and the sorted labels.
    pub(crate) fn labels(mut self) -> (Vec<usize>, Vec<usize>) {
        let n = self.0.len();
        let roots: Vec<usize> = (0..n).map(|i| self.find(i)).collect();
        let mut labels = roots.clone();
        labels.sort_unstable();
        labels.dedup();
        (roots, labels)
    }
}

/// One independent block of orbitals with its elimination tree and its leads.
#[derive(Clone, Debug)]
struct Block {
    tree: Tree,
    left: Lead,
    right: Lead,
    /// The energy-independent part of the leads' mode problems.
    left_modes: ModeSolver,
    right_modes: ModeSolver,
    /// Device orbitals of the first and last slices, in the order of the lead cells.
    left_orbitals: Vec<usize>,
    right_orbitals: Vec<usize>,
}

/// Lead self-energies of a [`NestedTwoTerminal`] at one energy, block by block (see
/// [`NestedTwoTerminal::self_energies`]).
#[derive(Clone, Debug)]
pub struct LeadSelfEnergies {
    energy: f64,
    blocks: Vec<BlockSelfEnergies>,
}

impl LeadSelfEnergies {
    /// Energy of the self-energies.
    pub fn energy(&self) -> f64 {
        self.energy
    }
}

#[derive(Clone, Debug)]
struct BlockSelfEnergies {
    left: Lead,
    right: Lead,
    left_orbitals: Vec<usize>,
    right_orbitals: Vec<usize>,
    sigma_l: CMat,
    sigma_r: CMat,
}

fn same_lead(a: &Lead, b: &Lead) -> bool {
    a.h00 == b.h00 && a.h01 == b.h01
}

/// A two-terminal device solved by nested dissection (see the module documentation).
#[derive(Clone, Debug)]
pub struct NestedTwoTerminal {
    h: SparseMatrix,
    region: Region,
    blocks: Vec<Block>,
}

/// Scattering states at one energy (see [`NestedTwoTerminal::scattering_states`] and
/// [`crate::NestedMultiTerminal::scattering_states`]).
#[derive(Clone, Debug)]
pub struct ScatteringStates {
    pub energy: f64,
    /// `ψ_p = G W_p` for each lead `p`, on every device orbital (rows, in the order of the
    /// region), one column per channel of the lead: `W_p W_p† = Γ_p`, so `ψ_p ψ_p† = G Γ_p G†`
    /// is the spectral function of the electrons lead `p` injects. Two-terminal devices list the
    /// left lead, then the right one.
    pub leads: Vec<CMat>,
}

impl ScatteringStates {
    /// Occupied spectral density `Im G^<_aa / 2π = Σ_p f_p |ψ_p,a|² / 2π` of every orbital, for
    /// the lead occupations `f` (`G^< = i Σ_p f_p ψ_p ψ_p†`). With every `f_p = 1` it is the local
    /// density of states of the scattering states (bound states of the device are not included).
    pub fn occupation(&self, f: &[f64]) -> Vec<f64> {
        assert_eq!(f.len(), self.leads.len(), "one occupation per lead");
        let n = self.leads.first().map_or(0, |m| m.nrows());
        (0..n)
            .map(|a| {
                let weight = |m: &CMat| (0..m.ncols()).map(|c| m[(a, c)].norm_sqr()).sum::<f64>();
                self.leads.iter().zip(f).map(|(m, fp)| fp * weight(m)).sum::<f64>()
                    / (2.0 * std::f64::consts::PI)
            })
            .collect()
    }
}

/// Spectral particle current through a bond, `j_{i→j}(E) = 2 Re Tr[H_ij G^<_ji(E)]`: summed over
/// a cross-section, it equals `T(E) [f_L(E) − f_R(E)]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BondCurrent {
    pub from: Site,
    pub to: Site,
    pub current: f64,
}

/// Position of each item of `vars` (any order): a sorted index for binary search.
pub(crate) struct Positions(Vec<(usize, usize)>);

impl Positions {
    pub(crate) fn new(vars: &[usize]) -> Self {
        let mut v: Vec<(usize, usize)> = vars.iter().enumerate().map(|(k, &x)| (x, k)).collect();
        v.sort_unstable();
        Self(v)
    }

    pub(crate) fn get(&self, x: usize) -> Option<usize> {
        self.0.binary_search_by_key(&x, |&(y, _)| y).ok().map(|k| self.0[k].1)
    }
}

impl NestedTwoTerminal {
    /// Two-terminal system built from a model on a region cut along primitive vector `axis`,
    /// as [`crate::TwoTerminal::from_model`]: both leads continue the first and last slices of
    /// the region with the clean `lead` model.
    pub fn from_model(device: &Model, lead: &Model, region: &Region, axis: usize) -> Self {
        for hop in lead.hoppings() {
            assert!(hop.offset[axis].abs() <= 1, "a lead hopping skips a cell along axis {axis}");
        }
        crate::gauge::assert_same_field(device, lead);
        let slices = region.slices(axis);
        let mut step = [0i64; 3];
        step[axis] = 1;
        // In a uniform field, everything in the gauge invariant along the axis.
        let mut h = device.sparse_hamiltonian(region);
        Gauge::along(device, step).transform_sparse(&mut h, region);
        let lead_gauge = Gauge::along(lead, step);
        let (first, last) = (&slices[0], &slices[slices.len() - 1]);
        let (h00, h01) = lead_gauge.lead(first, step, [-step[0], -step[1], -step[2]]);
        let left = Lead { h00, h01 };
        let (h00, h01) = lead_gauge.lead(last, step, step);
        let right = Lead { h00, h01 };
        let orbitals_of = |s: &Region| -> Vec<usize> {
            s.sites()
                .iter()
                .flat_map(|site| {
                    let i = region.index_of(site).expect("slice site in the region");
                    let o = region.offset(i);
                    o..o + region.site_orbitals(i)
                })
                .collect()
        };
        let (left_orbitals, right_orbitals) = (orbitals_of(first), orbitals_of(last));

        // Geometric tree over the interior sites (all but the two lead slices).
        let ends = [first.sites()[0].cell[axis], last.sites()[0].cell[axis]];
        let interior: Vec<usize> =
            (0..region.len()).filter(|&i| !ends.contains(&region.sites()[i].cell[axis])).collect();
        let sites = SiteTree::new(device, region, interior);

        // Independent blocks of orbitals: connected through the device or the leads.
        let mut components = Components::new(h.n);
        for r in 0..h.n {
            for (c, v) in h.row(r) {
                if v != c64::new(0.0, 0.0) {
                    components.join(r, c);
                }
            }
        }
        for (lead, orbs) in [(&left, &left_orbitals), (&right, &right_orbitals)] {
            components.join_matrix(&lead.h00, orbs, orbs);
            components.join_matrix(&lead.h01, orbs, orbs);
        }
        let (roots, labels) = components.labels();

        let blocks = labels
            .iter()
            .map(|&label| {
                let mine = |o: usize| roots[o] == label;
                let pick =
                    |orbs: &[usize]| -> Vec<usize> { (0..orbs.len()).filter(|&k| mine(orbs[k])).collect() };
                let (kl, kr) = (pick(&left_orbitals), pick(&right_orbitals));
                let restrict =
                    |m: &CMat, k: &[usize]| CMat::from_fn(k.len(), k.len(), |a, b| m[(k[a], k[b])]);
                let block_left = Lead { h00: restrict(&left.h00, &kl), h01: restrict(&left.h01, &kl) };
                let block_right = Lead { h00: restrict(&right.h00, &kr), h01: restrict(&right.h01, &kr) };
                let left_modes = ModeSolver::new(&block_left.h00, &block_left.h01);
                let right_modes = if same_lead(&block_left, &block_right) {
                    left_modes.clone()
                } else {
                    ModeSolver::new(&block_right.h00, &block_right.h01)
                };
                Block {
                    tree: sites.tree(region, &h, mine),
                    left: block_left,
                    right: block_right,
                    left_modes,
                    right_modes,
                    left_orbitals: kl.iter().map(|&k| left_orbitals[k]).collect(),
                    right_orbitals: kr.iter().map(|&k| right_orbitals[k]).collect(),
                }
            })
            .filter(|b: &Block| !b.left_orbitals.is_empty() && !b.right_orbitals.is_empty())
            .collect();
        Self { h, region: region.clone(), blocks }
    }

    /// Number of independent blocks solved separately.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Caroli transmission `T(E) = Tr[Γ_L G_LR Γ_R G_LR†]`, with mode-matching self-energies.
    pub fn transmission(&self, energy: f64) -> Result<f64, GreenError> {
        self.transmission_with(&self.self_energies(energy)?)
    }

    /// Transmissions at many energies, in parallel.
    pub fn transmissions(&self, energies: &[f64]) -> Vec<Result<f64, GreenError>> {
        use rayon::prelude::*;
        energies.par_iter().map(|&e| self.transmission(e)).collect()
    }

    /// Lead self-energies at `energy`, for each independent block. They serve every system with
    /// the same leads and blocks: in a disorder average, the leads are solved once per energy
    /// instead of once per configuration.
    ///
    /// ```ignore
    /// let sigma = reference.self_energies(e)?;
    /// let mean = configurations.iter().map(|sys| sys.transmission_with(&sigma)).sum::<Result<f64, _>>()?
    ///     / configurations.len() as f64;
    /// ```
    pub fn self_energies(&self, energy: f64) -> Result<LeadSelfEnergies, GreenError> {
        use rayon::prelude::*;
        let z = c64::new(energy, 0.0);
        let blocks = self
            .blocks
            .par_iter()
            .map(|block| {
                let ml = block.left_modes.modes_at(z)?;
                let mr = if same_lead(&block.left, &block.right) {
                    ml.clone()
                } else {
                    block.right_modes.modes_at(z)?
                };
                Ok(BlockSelfEnergies {
                    left: block.left.clone(),
                    right: block.right.clone(),
                    left_orbitals: block.left_orbitals.clone(),
                    right_orbitals: block.right_orbitals.clone(),
                    sigma_l: sigma_left(&block.left.h01, &ml),
                    sigma_r: sigma_right(&block.right.h01, &mr),
                })
            })
            .collect::<Result<Vec<_>, GreenError>>()?;
        Ok(LeadSelfEnergies { energy, blocks })
    }

    /// Transmission at the energy of `sigma`, with its lead self-energies, computed by
    /// [`Self::self_energies`] on this system or on one with the same leads and blocks.
    ///
    /// Panics if the leads or the blocks differ.
    pub fn transmission_with(&self, sigma: &LeadSelfEnergies) -> Result<f64, GreenError> {
        use rayon::prelude::*;
        assert_eq!(sigma.blocks.len(), self.blocks.len(), "self-energies of a system with other blocks");
        let z = c64::new(sigma.energy, 0.0);
        let parts: Vec<Result<f64, GreenError>> = self
            .blocks
            .par_iter()
            .zip(&sigma.blocks)
            .map(|(block, s)| {
                assert!(
                    same_lead(&block.left, &s.left)
                        && same_lead(&block.right, &s.right)
                        && block.left_orbitals == s.left_orbitals
                        && block.right_orbitals == s.right_orbitals,
                    "self-energies of a system with other leads"
                );
                let g_lr = self.lead_green(block, z, &s.sigma_l, &s.sigma_r)?;
                Ok(caroli_trace(&broadening(&s.sigma_l), &g_lr, &broadening(&s.sigma_r)))
            })
            .collect();
        parts.into_iter().sum()
    }

    /// Scattering matrix at `energy` in the basis of the propagating lead modes (lead 0 on the
    /// left, lead 1 on the right), as `kwant.smatrix` (see [`crate::SMatrix`]).
    pub fn smatrix(&self, energy: f64) -> Result<crate::SMatrix, GreenError> {
        use crate::smatrix::{BlockScattering, LeadSolution, ScatteringLead, scattering_blocks};
        use std::rc::Rc;
        let z = c64::new(energy, 0.0);
        let mut parts: Vec<BlockScattering> = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            // Each lead seen from the device: cells going away from it, coupled to the end slice.
            let left_out = block.left.h01.adjoint().to_owned();
            let left = Rc::new(LeadSolution::new(&block.left.h00, &left_out, z)?);
            let right = Rc::new(LeadSolution::new(&block.right.h00, &block.right.h01, z)?);
            let (sigma_l, sigma_r) = (left.self_energy(&left_out), right.self_energy(&block.right.h01));
            let root = self.root_with_leads(block, z, &sigma_l, &sigma_r, false)?;
            let k = root.s.nrows();
            let green = Lu::new(root.s.as_ref()).solve(CMat::identity(k, k).as_ref());
            if !green.norm_max().is_finite() {
                return Err(GreenError::Singular(energy));
            }
            let (nl, nr) = (block.left_orbitals.len(), block.right_orbitals.len());
            let leads = [
                Some(ScatteringLead {
                    at: (0..nl).collect(),
                    coupling: left_out.clone(),
                    h01: left_out,
                    solution: left,
                }),
                Some(ScatteringLead {
                    at: (root.right..root.right + nr).collect(),
                    coupling: block.right.h01.clone(),
                    h01: block.right.h01.clone(),
                    solution: right,
                }),
            ];
            parts.push(scattering_blocks(&green, &leads));
        }
        Ok(crate::SMatrix::assemble(energy, parts))
    }

    /// Transmission eigenvalues `τ_n` at `energy`, in decreasing order: the squared singular
    /// values of `t = W_L† G_LR W_R` (`W_p W_p† = Γ_p`), the same as those of the transmission
    /// block of the scattering matrix. Their sum is the transmission.
    pub fn transmission_eigenvalues(&self, energy: f64) -> Result<Vec<f64>, GreenError> {
        let sigma = self.self_energies(energy)?;
        let z = c64::new(energy, 0.0);
        let mut taus = Vec::new();
        for (block, s) in self.blocks.iter().zip(&sigma.blocks) {
            let g_lr = self.lead_green(block, z, &s.sigma_l, &s.sigma_r)?;
            let (wl, wr) = (gamma_root(&s.sigma_l)?, gamma_root(&s.sigma_r)?);
            let gw = product(g_lr.as_ref(), Op::Plain, wr.as_ref(), Op::Plain);
            let t = product(wl.as_ref(), Op::Adjoint, gw.as_ref(), Op::Plain);
            let svd = caroli_dense::svd(t.as_ref()).map_err(|_| GreenError::Eigen)?;
            taus.extend(svd.s.iter().map(|x| x * x));
        }
        taus.sort_by(|a, b| b.total_cmp(a));
        Ok(taus)
    }

    /// Fano factor `F = Σ τ (1 − τ) / Σ τ` at `energy`: the shot noise over the Poisson value, at
    /// zero temperature and small bias (0 for perfect channels, 1 for a tunnel barrier, 1/3 for
    /// a diffusive wire).
    pub fn fano_factor(&self, energy: f64) -> Result<f64, GreenError> {
        let taus = self.transmission_eigenvalues(energy)?;
        let total: f64 = taus.iter().sum();
        Ok(taus.iter().map(|t| t * (1.0 - t)).sum::<f64>() / total)
    }

    /// `G_LR`, the block of `(z − H − Σ)⁻¹` between the first and last slices.
    fn lead_green(&self, block: &Block, z: c64, sigma_l: &CMat, sigma_r: &CMat) -> Result<CMat, GreenError> {
        let root = self.root_with_leads(block, z, sigma_l, sigma_r, false)?;
        let (nl, nr) = (block.left_orbitals.len(), block.right_orbitals.len());
        // Columns of the right slice of S⁻¹, rows of the left slice.
        let rhs = CMat::from_fn(root.s.nrows(), nr, |i, j| {
            if i == root.right + j { c64::new(1.0, 0.0) } else { c64::new(0.0, 0.0) }
        });
        let x = Lu::new(root.s.as_ref()).solve(rhs.as_ref());
        if !x.norm_max().is_finite() {
            return Err(GreenError::Singular(z.re));
        }
        Ok(CMat::from_fn(nl, nr, |i, j| x[(i, j)]))
    }

    /// The Schur complement `S` of `z − H − Σ` on the lead slices of a block (left slice first,
    /// then the right one unless they coincide), with the delayed variables eliminated.
    fn root_with_leads(
        &self,
        block: &Block,
        z: c64,
        sigma_l: &CMat,
        sigma_r: &CMat,
        keep: bool,
    ) -> Result<LeadRoot, GreenError> {
        let same = block.left_orbitals == block.right_orbitals;
        let mut kept: Vec<usize> = block.left_orbitals.clone();
        if !same {
            kept.extend_from_slice(&block.right_orbitals);
        }
        let Root { delayed, mut f, factors } = block.tree.root(&self.h, z, &kept, keep)?;
        let nd = delayed.len();
        let (nl, nr) = (block.left_orbitals.len(), block.right_orbitals.len());
        let right = if same { 0 } else { nl };
        for a in 0..nl {
            for c in 0..nl {
                f[(nd + a, nd + c)] -= sigma_l[(a, c)];
            }
        }
        for a in 0..nr {
            for c in 0..nr {
                f[(nd + right + a, nd + right + c)] -= sigma_r[(a, c)];
            }
        }
        let (s, x) = if nd > 0 {
            schur(&f, nd).ok_or(GreenError::Singular(z.re))?
        } else {
            (f, CMat::zeros(0, kept.len()))
        };
        Ok(LeadRoot {
            kept,
            right,
            s,
            delayed: Factor { elim: delayed, boundary: Vec::new(), x, children: factors },
        })
    }

    /// Scattering states at `energy`: `ψ_p = G W_p` on every device orbital for each lead `p`,
    /// with `W_p W_p† = Γ_p` (one column per channel). The elimination keeps its factors, and a
    /// back-substitution from the lead slices gives the states everywhere: memory grows with the
    /// factors (O(N log N) for a 2D sample of N orbitals) and with N times the channels.
    pub fn scattering_states(&self, energy: f64) -> Result<ScatteringStates, GreenError> {
        let sigma = self.self_energies(energy)?;
        let z = c64::new(energy, 0.0);
        let n = self.h.n;
        let (mut left, mut right): (Vec<CMat>, Vec<CMat>) = (Vec::new(), Vec::new());
        for (block, s) in self.blocks.iter().zip(&sigma.blocks) {
            let (wl, wr) = (gamma_root(&s.sigma_l)?, gamma_root(&s.sigma_r)?);
            let (cl, cr) = (wl.ncols(), wr.ncols());
            let root = self.root_with_leads(block, z, &s.sigma_l, &s.sigma_r, true)?;
            // Sources on the lead slices: W_L on the left slice, W_R on the right one.
            let mut rhs = CMat::zeros(root.kept.len(), cl + cr);
            for a in 0..wl.nrows() {
                for c in 0..cl {
                    rhs[(a, c)] = wl[(a, c)];
                }
            }
            for a in 0..wr.nrows() {
                for c in 0..cr {
                    rhs[(root.right + a, cl + c)] = wr[(a, c)];
                }
            }
            let x = Lu::new(root.s.as_ref()).solve(rhs.as_ref());
            if !x.norm_max().is_finite() {
                return Err(GreenError::Singular(energy));
            }
            let mut psi = CMat::zeros(n, cl + cr);
            for (a, &o) in root.kept.iter().enumerate() {
                for c in 0..cl + cr {
                    psi[(o, c)] = x[(a, c)];
                }
            }
            let LeadRoot { kept, delayed, .. } = root;
            let delayed = Factor { boundary: kept, ..delayed };
            delayed.back_substitute(&mut psi);
            left.push(psi.as_ref().subcols(0, cl).to_owned());
            right.push(psi.as_ref().subcols(cl, cr).to_owned());
        }
        let concat = |parts: Vec<CMat>| {
            let cols: usize = parts.iter().map(|m| m.ncols()).sum();
            let mut out = CMat::zeros(n, cols);
            let mut c0 = 0;
            for m in parts {
                out.as_mut().subcols_mut(c0, m.ncols()).copy_from(&m);
                c0 += m.ncols();
            }
            out
        };
        Ok(ScatteringStates { energy, leads: vec![concat(left), concat(right)] })
    }

    /// Spectral bond currents `2 Re Tr[H_ij G^<_ji]` between every pair of coupled sites, from
    /// the scattering states and the lead occupations `f` (`G^< = i Σ_p f_p ψ_p ψ_p†`). Each bond
    /// appears once, from the site of lower index in the region.
    pub fn bond_currents(&self, states: &ScatteringStates, f: &[f64]) -> Vec<BondCurrent> {
        bond_currents(&self.h, &self.region, states, f, None)
    }

    /// Bond currents of an on-site operator `O` (a spin current for `O = s_z`):
    /// `Re Tr[(O H_ij + H_ij O) G^<_ji]`, `2 Re Tr[H_ij G^<_ji]` for `O = 1`.
    pub fn operator_currents(&self, states: &ScatteringStates, f: &[f64], op: &CMat) -> Vec<BondCurrent> {
        bond_currents(&self.h, &self.region, states, f, Some(op))
    }

    /// Spectral density `Σ_p f_p ψ_p† O ψ_p / 2π` of every site (`O = 1` for `None`): the occupied
    /// density, or a spin density.
    pub fn site_densities(&self, states: &ScatteringStates, f: &[f64], op: Option<&CMat>) -> Vec<f64> {
        site_densities(&self.region, states, f, op)
    }

    /// The device region (orbital layout of the scattering states).
    pub fn region(&self) -> &Region {
        &self.region
    }
}

/// Spectral bond currents between every pair of coupled sites of `region`, with
/// `G^< = i Σ_p f_p ψ_p ψ_p†` from the scattering states: `2 Re Tr[H_ij G^<_ji]`, or with an
/// on-site operator `O` (spin...), `Re Tr[(O H_ij + H_ij O) G^<_ji]`. Each bond appears once,
/// from the site of lower index in the region.
pub(crate) fn bond_currents(
    h: &SparseMatrix,
    region: &Region,
    states: &ScatteringStates,
    f: &[f64],
    op: Option<&CMat>,
) -> Vec<BondCurrent> {
    assert_eq!(f.len(), states.leads.len(), "one occupation per lead");
    // Rows of ψ_p as contiguous vectors: (ψ_p ψ_p†)_ba = Σ_k ψ_bk ψ_ak*.
    let rows: Vec<CMat> = states.leads.iter().map(|m| m.transpose().to_owned()).collect();
    let lesser = |b: usize, a: usize| -> c64 {
        let mut acc = c64::new(0.0, 0.0);
        for (t, fp) in rows.iter().zip(f) {
            acc += (0..t.nrows()).map(|k| t[(k, b)] * t[(k, a)].conj()).sum::<c64>() * *fp;
        }
        c64::new(0.0, 1.0) * acc
    };
    let mut owner = vec![0usize; h.n];
    for i in 0..region.len() {
        owner[region.offset(i)..region.offset(i) + region.site_orbitals(i)].fill(i);
    }
    let mut currents: std::collections::BTreeMap<(usize, usize), c64> = Default::default();
    match op {
        None => {
            for a in 0..h.n {
                for (b, hab) in h.row(a) {
                    let (i, j) = (owner[a], owner[b]);
                    if i < j {
                        *currents.entry((i, j)).or_default() += hab * lesser(b, a) * 2.0;
                    }
                }
            }
        }
        Some(o) => {
            // Blocks H_ij, then Tr[(O H_ij + H_ij O) G^<_ji].
            let mut blocks: std::collections::BTreeMap<(usize, usize), CMat> = Default::default();
            for a in 0..h.n {
                for (b, hab) in h.row(a) {
                    let (i, j) = (owner[a], owner[b]);
                    if i < j {
                        let (ni, nj) = (region.site_orbitals(i), region.site_orbitals(j));
                        assert!(
                            o.nrows() == ni && o.nrows() == nj,
                            "the operator acts on the orbitals of a site"
                        );
                        let block = blocks.entry((i, j)).or_insert_with(|| CMat::zeros(ni, nj));
                        block[(a - region.offset(i), b - region.offset(j))] += hab;
                    }
                }
            }
            for ((i, j), hij) in blocks {
                let (oi, oj) = (region.offset(i), region.offset(j));
                let g = CMat::from_fn(hij.ncols(), hij.nrows(), |b, a| lesser(oj + b, oi + a)); // G^<_ji
                let weighted = o * &hij + &hij * o;
                let acc: c64 = (0..weighted.nrows())
                    .flat_map(|a| (0..weighted.ncols()).map(move |b| (a, b)))
                    .map(|(a, b)| weighted[(a, b)] * g[(b, a)])
                    .sum();
                currents.insert((i, j), acc);
            }
        }
    }
    currents
        .into_iter()
        .map(|((i, j), acc)| BondCurrent { from: region.sites()[i], to: region.sites()[j], current: acc.re })
        .collect()
}

/// Spectral densities of every site of `region`, `Σ_p f_p ψ_p† O ψ_p / 2π` on its orbitals (`O = 1`
/// without an operator): the occupied density, or a spin density.
pub(crate) fn site_densities(
    region: &Region,
    states: &ScatteringStates,
    f: &[f64],
    op: Option<&CMat>,
) -> Vec<f64> {
    assert_eq!(f.len(), states.leads.len(), "one occupation per lead");
    (0..region.len())
        .map(|i| {
            let (o0, n) = (region.offset(i), region.site_orbitals(i));
            let mut acc = c64::new(0.0, 0.0);
            for (m, fp) in states.leads.iter().zip(f) {
                for k in 0..m.ncols() {
                    for a in 0..n {
                        let left = m[(o0 + a, k)].conj();
                        acc += match op {
                            None => left * m[(o0 + a, k)] * *fp,
                            Some(o) => (0..n).map(|b| left * o[(a, b)] * m[(o0 + b, k)]).sum::<c64>() * *fp,
                        };
                    }
                }
            }
            acc.re / (2.0 * std::f64::consts::PI)
        })
        .collect()
}

/// The root of a two-terminal block: `S` on the kept lead orbitals (`right` is the position of
/// the right slice), and the delayed variables with their back-substitution data.
struct LeadRoot {
    kept: Vec<usize>,
    right: usize,
    s: CMat,
    delayed: Factor,
}

/// `W` with `W W† = Γ = i(Σ − Σ†)`: the eigenvectors of `Γ` with positive eigenvalues, scaled by
/// their square roots (one column per channel).
pub(crate) fn gamma_root(sigma: &CMat) -> Result<CMat, GreenError> {
    let g = broadening(sigma);
    let n = g.nrows();
    let g = CMat::from_fn(n, n, |i, j| (g[(i, j)] + g[(j, i)].conj()) * 0.5);
    let evd = g.self_adjoint_eigen(faer::Side::Lower).map_err(|_| GreenError::Eigen)?;
    let (values, vectors) = (evd.S(), evd.U());
    let top = (0..n).map(|k| values[k].re.abs()).fold(0.0, f64::max);
    let channels: Vec<usize> = (0..n).filter(|&k| values[k].re > 1e-12 * top).collect();
    Ok(CMat::from_fn(n, channels.len(), |i, c| vectors[(i, channels[c])] * values[channels[c]].re.sqrt()))
}

/// Builds the elimination tree over `sites` (indices in `region`) in post-order: each node is
/// `(sites eliminated, children)`. Returns the root of the subtree, if any.
fn split(
    region: &Region,
    sites: Vec<usize>,
    range: &[i64; 3],
    nodes: &mut Vec<(Vec<usize>, Vec<usize>)>,
) -> Option<usize> {
    if sites.is_empty() {
        return None;
    }
    let orbitals: usize = sites.iter().map(|&i| region.site_orbitals(i)).sum();
    let dim = region.lattice().dim();
    let extent = |d: usize| {
        let (lo, hi) = sites.iter().fold((i64::MAX, i64::MIN), |(lo, hi), &i| {
            let c = region.sites()[i].cell[d];
            (lo.min(c), hi.max(c))
        });
        (lo, hi)
    };
    let (d, (lo, hi)) = (0..dim)
        .map(|d| (d, extent(d)))
        .max_by_key(|&(d, (lo, hi))| (hi - lo, std::cmp::Reverse(d)))
        .unwrap();
    if orbitals <= LEAF_ORBITALS || hi - lo < range[d] {
        nodes.push((sites, Vec::new()));
        return Some(nodes.len() - 1);
    }
    let mid = lo + (hi - lo + 1 - range[d]) / 2;
    let (mut a, mut sep, mut b) = (Vec::new(), Vec::new(), Vec::new());
    for i in sites {
        let c = region.sites()[i].cell[d];
        if c < mid {
            a.push(i);
        } else if c < mid + range[d] {
            sep.push(i);
        } else {
            b.push(i);
        }
    }
    let children: Vec<usize> =
        [split(region, a, range, nodes), split(region, b, range, nodes)].into_iter().flatten().collect();
    nodes.push((sep, children));
    Some(nodes.len() - 1)
}

/// Fills the boundaries of the nodes (in post-order): the variables of the front that are
/// eliminated later, from the sparsity of `h` and the children's boundaries.
fn symbolic(h: &SparseMatrix, nodes: &mut [Node]) {
    const NEVER: usize = usize::MAX;
    let mut rank = vec![NEVER; h.n];
    let mut next = 0;
    let mut last_rank = vec![0usize; nodes.len()];
    for (k, node) in nodes.iter().enumerate() {
        for &o in &node.eliminate {
            rank[o] = next;
            next += 1;
        }
        last_rank[k] = next;
    }
    for k in 0..nodes.len() {
        let mut bnd: Vec<usize> = Vec::new();
        for &o in &nodes[k].eliminate {
            bnd.extend(h.row(o).map(|(j, _)| j));
        }
        for &c in &nodes[k].children.clone() {
            bnd.extend(nodes[c].boundary.iter().copied());
        }
        bnd.retain(|&o| rank[o] == NEVER || rank[o] >= last_rank[k]);
        bnd.sort_unstable();
        bnd.dedup();
        nodes[k].boundary = bnd;
    }
}

/// Adds the children's Schur complements into the front `f` (variables at `pos`), and returns
/// their back-substitution data (when kept).
fn extend_add(f: &mut CMat, pos: &Positions, kids: Vec<Front>) -> Vec<Factor> {
    let mut factors = Vec::new();
    for kid in kids {
        add_runs(f, &runs(&kid.vars, pos), &kid.s);
        factors.extend(kid.factor);
    }
    factors
}

/// Where the variables `vars` sit in a front with positions `pos`, as runs of consecutive
/// positions `(first in vars, first in the front, length)`: both sides are sorted, so a few runs
/// cover a column.
fn runs(vars: &[usize], pos: &Positions) -> Vec<(usize, usize, usize)> {
    let mut runs: Vec<(usize, usize, usize)> = Vec::new();
    for (a, &o) in vars.iter().enumerate() {
        let pa = pos.get(o).expect("child variable in the parent front");
        match runs.last_mut() {
            Some((a0, p0, len)) if *a0 + *len == a && *p0 + *len == pa => *len += 1,
            _ => runs.push((a, pa, 1)),
        }
    }
    runs
}

/// `f[p, q] += s[a, b]` for the positions `p`, `q` of `a`, `b` given by `runs`.
fn add_runs(f: &mut CMat, runs: &[(usize, usize, usize)], s: &CMat) {
    for &(b0, q0, width) in runs {
        for b in 0..width {
            let (src, dst) = (s.col_as_slice(b0 + b), f.col_as_slice_mut(q0 + b));
            for &(a0, p0, len) in runs {
                for (d, x) in dst[p0..p0 + len].iter_mut().zip(&src[a0..a0 + len]) {
                    *d += x;
                }
            }
        }
    }
}

/// [`schur`] for a small front: Gaussian elimination in a copy of `f`, without calls to the
/// dense library or other allocations; `X` only when `want_x`. Partial pivoting within the
/// separator rows, with the same stability test.
pub(crate) fn small_schur(f: &CMat, ne: usize, want_x: bool) -> Option<(CMat, CMat)> {
    let n = f.nrows();
    let nb = n - ne;
    let scale = (0..ne)
        .map(|j| f.col_as_slice(j)[..ne].iter().map(|x| x.norm()).fold(0.0, f64::max))
        .fold(0.0, f64::max);
    let mut w = f.clone();
    let mut lk: Vec<c64> = Vec::with_capacity(n);
    for k in 0..ne {
        let (p, big) =
            (k..ne).map(|i| (i, w[(i, k)].norm())).fold((k, -1.0), |a, b| if b.1 > a.1 { b } else { a });
        let stable = big > PIVOT_TOL * scale; // false for NaN
        if !stable {
            return None;
        }
        if p != k {
            for j in 0..n {
                w.col_as_slice_mut(j).swap(p, k);
            }
        }
        let inv = c64::new(1.0, 0.0) / w[(k, k)];
        for v in &mut w.col_as_slice_mut(k)[k + 1..] {
            *v *= inv;
        }
        // Rank-one update of the trailing columns.
        lk.clear();
        lk.extend_from_slice(&w.col_as_slice(k)[k + 1..]);
        for j in k + 1..n {
            let col = w.col_as_slice_mut(j);
            let ukj = col[k];
            if ukj != c64::new(0.0, 0.0) {
                for (c, l) in col[k + 1..].iter_mut().zip(&lk) {
                    *c -= l * ukj;
                }
            }
        }
    }
    let s = CMat::from_fn(nb, nb, |i, j| w[(ne + i, ne + j)]);
    let x = if want_x {
        // X = U⁻¹ U12, with U12 = L⁻¹ P f12 in the top right block.
        let mut x = CMat::from_fn(ne, nb, |i, j| w[(i, ne + j)]);
        for j in 0..nb {
            let col = x.col_as_slice_mut(j);
            for i in (0..ne).rev() {
                let v = col[i] / w[(i, i)];
                col[i] = v;
                for (l, c) in col[..i].iter_mut().enumerate() {
                    *c -= w[(l, i)] * v;
                }
            }
        }
        x
    } else {
        CMat::zeros(ne, 0)
    };
    Some((s, x))
}

/// Schur complement `S = f22 − f21 X` of the leading `ne × ne` block of `f`, with
/// `X = f11⁻¹ f12`; `None` if that block is numerically singular.
pub(crate) fn schur(f: &CMat, ne: usize) -> Option<(CMat, CMat)> {
    let n = f.nrows();
    let f11 = f.as_ref().submatrix(0, 0, ne, ne);
    let scale = f11.norm_max();
    let lu = Lu::new(f11);
    let (small, _) = lu.pivot_range();
    let stable = small > PIVOT_TOL * scale; // false for NaN
    if !stable {
        return None;
    }
    let nb = n - ne;
    if nb == 0 {
        return Some((CMat::zeros(0, 0), CMat::zeros(ne, 0)));
    }
    let x = lu.solve(f.as_ref().submatrix(0, ne, ne, nb));
    let mut s = f.as_ref().submatrix(ne, ne, nb, nb).to_owned();
    let minus_one = c64::new(-1.0, 0.0);
    matmul(
        s.as_mut(),
        Accum::Add,
        f.as_ref().submatrix(ne, 0, nb, ne),
        Op::Plain,
        x.as_ref(),
        Op::Plain,
        minus_one,
    );
    Some((s, x))
}
