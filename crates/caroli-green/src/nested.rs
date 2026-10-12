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

use crate::modes::{sigma_left, sigma_right};
use crate::{GreenError, Lead, broadening, trace};
use caroli_dense::{Accum, Lu, Op, matmul};
use caroli_lattice::Region;
use caroli_model::{CMat, Model, SparseMatrix, c64};

/// Domains with at most this many orbitals are not split further.
const LEAF_ORBITALS: usize = 16;
/// A pivot below this fraction of the largest entry of a separator block delays the block.
const PIVOT_TOL: f64 = 1e-11;

/// One node of the elimination tree, for one block of orbitals.
#[derive(Clone, Debug)]
struct Node {
    /// Orbitals eliminated at this node (sorted).
    eliminate: Vec<usize>,
    /// Orbitals of the front that are eliminated later or never (sorted).
    boundary: Vec<usize>,
    children: Vec<usize>,
}

/// One independent block of orbitals with its elimination tree and its leads.
#[derive(Clone, Debug)]
struct Block {
    nodes: Vec<Node>,
    /// Children of the root.
    top: Vec<usize>,
    left: Lead,
    right: Lead,
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
    blocks: Vec<Block>,
}

/// Position of each item of `vars` (any order): a sorted index for binary search.
struct Positions(Vec<(usize, usize)>);

impl Positions {
    fn new(vars: &[usize]) -> Self {
        let mut v: Vec<(usize, usize)> = vars.iter().enumerate().map(|(k, &x)| (x, k)).collect();
        v.sort_unstable();
        Self(v)
    }

    fn get(&self, x: usize) -> Option<usize> {
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
        let h = device.sparse_hamiltonian(region);
        let slices = region.slices(axis);
        let mut step = [0i64; 3];
        step[axis] = 1;
        let (first, last) = (&slices[0], &slices[slices.len() - 1]);
        let left = Lead { h00: lead.hamiltonian(first), h01: lead.coupling(first, &first.translated(step)) };
        let right = Lead { h00: lead.hamiltonian(last), h01: lead.coupling(last, &last.translated(step)) };
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
        let mut range = [1i64; 3];
        for hop in device.hoppings() {
            for (d, r) in range.iter_mut().enumerate() {
                *r = (*r).max(hop.offset[d].abs());
            }
        }
        let mut site_nodes: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
        let top: Vec<usize> = split(region, interior, &range, &mut site_nodes).into_iter().collect();

        // Independent blocks of orbitals: connected through the device or the leads.
        let n = h.n;
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut [usize], mut i: usize) -> usize {
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        let join = |p: &mut Vec<usize>, a: usize, b: usize| {
            let (ra, rb) = (find(p, a), find(p, b));
            if ra != rb {
                p[ra.max(rb)] = ra.min(rb);
            }
        };
        for r in 0..n {
            for (c, v) in h.row(r) {
                if v != c64::new(0.0, 0.0) {
                    join(&mut parent, r, c);
                }
            }
        }
        for (lead, orbs) in [(&left, &left_orbitals), (&right, &right_orbitals)] {
            for m in [&lead.h00, &lead.h01] {
                for a in 0..m.nrows() {
                    for b in 0..m.ncols() {
                        if m[(a, b)] != c64::new(0.0, 0.0) {
                            join(&mut parent, orbs[a], orbs[b]);
                        }
                    }
                }
            }
        }
        let mut roots: Vec<usize> = (0..n).map(|i| find(&mut parent, i)).collect();
        roots.iter_mut().for_each(|r| *r = find(&mut parent, *r));
        let mut labels: Vec<usize> = roots.clone();
        labels.sort_unstable();
        labels.dedup();

        let blocks = labels
            .iter()
            .map(|&label| {
                let mine = |o: usize| roots[o] == label;
                let pick =
                    |orbs: &[usize]| -> Vec<usize> { (0..orbs.len()).filter(|&k| mine(orbs[k])).collect() };
                let (kl, kr) = (pick(&left_orbitals), pick(&right_orbitals));
                let restrict =
                    |m: &CMat, k: &[usize]| CMat::from_fn(k.len(), k.len(), |a, b| m[(k[a], k[b])]);
                let block_orbitals = |sites: &[usize]| -> Vec<usize> {
                    let mut v: Vec<usize> = sites
                        .iter()
                        .flat_map(|&i| region.offset(i)..region.offset(i) + region.site_orbitals(i))
                        .filter(|&o| mine(o))
                        .collect();
                    v.sort_unstable();
                    v
                };
                let mut nodes: Vec<Node> = site_nodes
                    .iter()
                    .map(|(sites, children)| Node {
                        eliminate: block_orbitals(sites),
                        boundary: Vec::new(),
                        children: children.clone(),
                    })
                    .collect();
                symbolic(&h, &mut nodes);
                Block {
                    nodes,
                    top: top.clone(),
                    left: Lead { h00: restrict(&left.h00, &kl), h01: restrict(&left.h01, &kl) },
                    right: Lead { h00: restrict(&right.h00, &kr), h01: restrict(&right.h01, &kr) },
                    left_orbitals: kl.iter().map(|&k| left_orbitals[k]).collect(),
                    right_orbitals: kr.iter().map(|&k| right_orbitals[k]).collect(),
                }
            })
            .filter(|b: &Block| !b.left_orbitals.is_empty() && !b.right_orbitals.is_empty())
            .collect();
        Self { h, blocks }
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
                let ml = block.left.modes_at(z)?;
                let mr = if same_lead(&block.left, &block.right) {
                    ml.clone()
                } else {
                    block.right.modes_at(z)?
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
                let t = broadening(&s.sigma_l) * &g_lr * broadening(&s.sigma_r) * g_lr.adjoint();
                Ok(trace(&t).re)
            })
            .collect();
        parts.into_iter().sum()
    }

    /// `G_LR`, the block of `(z − H − Σ)⁻¹` between the first and last slices.
    fn lead_green(&self, block: &Block, z: c64, sigma_l: &CMat, sigma_r: &CMat) -> Result<CMat, GreenError> {
        let kids = self.fronts(block, &block.top, z)?;
        let same = block.left_orbitals == block.right_orbitals;
        let mut b: Vec<usize> = block.left_orbitals.clone();
        if !same {
            b.extend_from_slice(&block.right_orbitals);
        }
        let in_b = Positions::new(&b);
        // Variables delayed up to the root, eliminated before the lead slices.
        let mut delayed: Vec<usize> =
            kids.iter().flat_map(|(v, _)| v.iter().copied()).filter(|&o| in_b.get(o).is_none()).collect();
        delayed.sort_unstable();
        delayed.dedup();
        let vars: Vec<usize> = delayed.iter().chain(&b).copied().collect();
        let pos = Positions::new(&vars);
        let nd = delayed.len();
        let mut f = CMat::zeros(vars.len(), vars.len());
        for &i in &b {
            let pi = pos.get(i).unwrap();
            f[(pi, pi)] += z;
            for (j, hij) in self.h.row(i) {
                if in_b.get(j).is_some() {
                    f[(pi, pos.get(j).unwrap())] -= hij;
                }
            }
        }
        let (nl, nr) = (block.left_orbitals.len(), block.right_orbitals.len());
        let r0 = if same { nd } else { nd + nl };
        for a in 0..nl {
            for c in 0..nl {
                f[(nd + a, nd + c)] -= sigma_l[(a, c)];
            }
        }
        for a in 0..nr {
            for c in 0..nr {
                f[(r0 + a, r0 + c)] -= sigma_r[(a, c)];
            }
        }
        extend_add(&mut f, &pos, &kids);
        let s = if nd > 0 { schur(&f, nd).ok_or(GreenError::Singular(z.re))? } else { f };
        // Columns of the right slice of S⁻¹, rows of the left slice.
        let rhs = CMat::from_fn(s.nrows(), nr, |i, j| {
            if i == r0 - nd + j { c64::new(1.0, 0.0) } else { c64::new(0.0, 0.0) }
        });
        let x = Lu::new(s.as_ref()).solve(rhs.as_ref());
        if !x.norm_max().is_finite() {
            return Err(GreenError::Singular(z.re));
        }
        Ok(CMat::from_fn(nl, nr, |i, j| x[(i, j)]))
    }

    /// Fronts of the given nodes, in parallel: `(variables, Schur complement)` of each.
    fn fronts(&self, block: &Block, nodes: &[usize], z: c64) -> Result<Vec<(Vec<usize>, CMat)>, GreenError> {
        match nodes {
            [] => Ok(Vec::new()),
            [a] => Ok(vec![self.front(block, *a, z)?]),
            [a, b] => {
                let (x, y) = rayon::join(|| self.front(block, *a, z), || self.front(block, *b, z));
                Ok(vec![x?, y?])
            }
            _ => nodes.iter().map(|&k| self.front(block, k, z)).collect(),
        }
    }

    /// Eliminates node `k` of a block (after its children) and returns the variables left in its
    /// front and their Schur complement.
    fn front(&self, block: &Block, k: usize, z: c64) -> Result<(Vec<usize>, CMat), GreenError> {
        let node = &block.nodes[k];
        let kids = self.fronts(block, &node.children, z)?;
        let in_boundary = |o: usize| node.boundary.binary_search(&o).is_ok();
        // Own separator, plus variables the children could not eliminate.
        let mut elim: Vec<usize> = node.eliminate.clone();
        for (vars, _) in &kids {
            elim.extend(vars.iter().copied().filter(|&o| !in_boundary(o)));
        }
        elim.sort_unstable();
        elim.dedup();
        let vars: Vec<usize> = elim.iter().chain(&node.boundary).copied().collect();
        let pos = Positions::new(&vars);
        let mut f = CMat::zeros(vars.len(), vars.len());
        // Entries of the own separator with every later variable (the earlier ones were
        // assembled where those were eliminated).
        for &i in &node.eliminate {
            let pi = pos.get(i).unwrap();
            f[(pi, pi)] += z;
            for (j, hij) in self.h.row(i) {
                if j == i {
                    f[(pi, pi)] -= hij;
                } else if let Some(pj) = pos.get(j) {
                    // Later variables only: the rest of the own separator after i, and the
                    // boundary. A child's delayed variables already carry these entries.
                    let later = if node.eliminate.binary_search(&j).is_ok() { j > i } else { in_boundary(j) };
                    if later {
                        f[(pi, pj)] -= hij;
                        f[(pj, pi)] -= hij.conj();
                    }
                }
            }
        }
        extend_add(&mut f, &pos, &kids);
        let ne = elim.len();
        if ne == 0 {
            return Ok((vars, f));
        }
        match schur(&f, ne) {
            Some(s) => Ok((node.boundary.clone(), s)),
            // Singular separator block: hand the whole front to the parent.
            None => Ok((vars, f)),
        }
    }
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

/// Adds the children's Schur complements into the front `f` (variables at `pos`).
fn extend_add(f: &mut CMat, pos: &Positions, kids: &[(Vec<usize>, CMat)]) {
    for (vars, u) in kids {
        let map: Vec<usize> =
            vars.iter().map(|&o| pos.get(o).expect("child variable in the parent front")).collect();
        for (b, &pb) in map.iter().enumerate() {
            for (a, &pa) in map.iter().enumerate() {
                f[(pa, pb)] += u[(a, b)];
            }
        }
    }
}

/// Schur complement of the leading `ne × ne` block of `f`, or `None` if that block is
/// numerically singular.
fn schur(f: &CMat, ne: usize) -> Option<CMat> {
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
        return Some(CMat::zeros(0, 0));
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
    Some(s)
}
