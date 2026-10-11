//! Lattices, finite regions and transport slices.
//!
//! A [`Lattice`] is a Bravais lattice (1, 2 or 3 primitive vectors) with a basis of
//! [`Sublattice`]s, each carrying a number of orbitals. A [`Region`] is a finite, ordered set of
//! [`Site`]s of a lattice; it fixes the orbital ordering of every matrix built on it.

use std::collections::HashMap;

/// Cartesian vector.
pub type Vec3 = [f64; 3];

/// Integer coordinates of a unit cell along the primitive vectors (unused axes are 0).
pub type Cell = [i64; 3];

/// One site of the basis of a lattice.
#[derive(Clone, Debug, PartialEq)]
pub struct Sublattice {
    pub name: String,
    /// Position inside the unit cell (Cartesian).
    pub offset: Vec3,
    /// Number of orbitals (including spin) carried by each site of this sublattice.
    pub orbitals: usize,
}

/// A Bravais lattice with a basis.
#[derive(Clone, Debug, PartialEq)]
pub struct Lattice {
    vectors: Vec<Vec3>,
    sublattices: Vec<Sublattice>,
}

impl Lattice {
    /// Lattice with the given primitive vectors (1 to 3 of them) and no sublattice yet.
    pub fn new(vectors: &[Vec3]) -> Self {
        assert!((1..=3).contains(&vectors.len()), "a lattice has 1, 2 or 3 primitive vectors");
        Self { vectors: vectors.to_vec(), sublattices: Vec::new() }
    }

    /// Adds a sublattice and returns the lattice (builder style).
    pub fn with_sublattice(mut self, name: &str, offset: Vec3, orbitals: usize) -> Self {
        assert!(orbitals > 0, "a sublattice carries at least one orbital");
        self.sublattices.push(Sublattice { name: name.to_owned(), offset, orbitals });
        self
    }

    /// Square lattice of constant `a` with one sublattice of `orbitals` orbitals.
    pub fn square(a: f64, orbitals: usize) -> Self {
        Self::new(&[[a, 0.0, 0.0], [0.0, a, 0.0]]).with_sublattice("A", [0.0; 3], orbitals)
    }

    /// Simple cubic lattice of constant `a` with one sublattice of `orbitals` orbitals.
    pub fn cubic(a: f64, orbitals: usize) -> Self {
        Self::new(&[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]).with_sublattice("A", [0.0; 3], orbitals)
    }

    /// Honeycomb lattice with nearest-neighbour distance `a` (sublattices A and B).
    pub fn honeycomb(a: f64, orbitals: usize) -> Self {
        let s3 = 3f64.sqrt();
        Self::new(&[[s3 * a, 0.0, 0.0], [0.5 * s3 * a, 1.5 * a, 0.0]])
            .with_sublattice("A", [0.0; 3], orbitals)
            .with_sublattice("B", [0.0, a, 0.0], orbitals)
    }

    /// Number of primitive vectors.
    pub fn dim(&self) -> usize {
        self.vectors.len()
    }

    pub fn vectors(&self) -> &[Vec3] {
        &self.vectors
    }

    pub fn sublattices(&self) -> &[Sublattice] {
        &self.sublattices
    }

    /// Cartesian position of a site.
    pub fn position(&self, site: Site) -> Vec3 {
        let mut r = self.sublattices[site.sub].offset;
        for (n, v) in site.cell.iter().zip(&self.vectors) {
            for (ri, vi) in r.iter_mut().zip(v) {
                *ri += *n as f64 * vi;
            }
        }
        r
    }
}

/// A site: a unit cell and a sublattice index.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Site {
    pub cell: Cell,
    pub sub: usize,
}

/// A finite, ordered set of sites. The order fixes the orbital layout of matrices: site `i`
/// occupies orbitals `offset(i) .. offset(i) + orbitals(i)`.
#[derive(Clone, Debug)]
pub struct Region {
    lattice: Lattice,
    sites: Vec<Site>,
    index: HashMap<Site, usize>,
    offsets: Vec<usize>,
    orbitals: usize,
}

impl Region {
    /// Region made of the given sites, in this order (duplicates are an error).
    pub fn from_sites(lattice: &Lattice, sites: Vec<Site>) -> Self {
        let mut index = HashMap::with_capacity(sites.len());
        let mut offsets = Vec::with_capacity(sites.len());
        let mut orbitals = 0;
        for (i, s) in sites.iter().enumerate() {
            assert!(s.sub < lattice.sublattices.len(), "unknown sublattice {}", s.sub);
            assert!(index.insert(*s, i).is_none(), "duplicate site {s:?}");
            offsets.push(orbitals);
            orbitals += lattice.sublattices[s.sub].orbitals;
        }
        Self { lattice: lattice.clone(), sites, index, offsets, orbitals }
    }

    /// All sites of the cells `0..extent[a]` along each primitive vector `a` (extents beyond
    /// the lattice dimension are ignored). Sites are ordered with the first axis varying
    /// slowest, then the sublattice fastest: for a ribbon along x this groups the sites by
    /// column, which is the slice order used for transport.
    pub fn cuboid(lattice: &Lattice, extent: [usize; 3]) -> Self {
        let dim = lattice.dim();
        let n = |a: usize| if a < dim { extent[a] as i64 } else { 1 };
        let mut sites = Vec::new();
        for i in 0..n(0) {
            for j in 0..n(1) {
                for k in 0..n(2) {
                    for sub in 0..lattice.sublattices.len() {
                        sites.push(Site { cell: [i, j, k], sub });
                    }
                }
            }
        }
        Self::from_sites(lattice, sites)
    }

    /// Sites of the cuboid `0..extent` whose position satisfies `inside`.
    pub fn shaped(lattice: &Lattice, extent: [usize; 3], inside: impl Fn(Vec3) -> bool) -> Self {
        let all = Self::cuboid(lattice, extent);
        let sites = all.sites.into_iter().filter(|s| inside(lattice.position(*s))).collect();
        Self::from_sites(lattice, sites)
    }

    pub fn lattice(&self) -> &Lattice {
        &self.lattice
    }

    pub fn sites(&self) -> &[Site] {
        &self.sites
    }

    pub fn len(&self) -> usize {
        self.sites.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }

    /// Index of a site in the region, if it belongs to it.
    pub fn index_of(&self, site: &Site) -> Option<usize> {
        self.index.get(site).copied()
    }

    /// First orbital of site `i`.
    pub fn offset(&self, i: usize) -> usize {
        self.offsets[i]
    }

    /// Number of orbitals of site `i`.
    pub fn site_orbitals(&self, i: usize) -> usize {
        self.lattice.sublattices[self.sites[i].sub].orbitals
    }

    /// Total number of orbitals (size of the matrices built on the region).
    pub fn orbitals(&self) -> usize {
        self.orbitals
    }

    /// Splits the region into slices along primitive vector `axis`: slice `n` holds the sites
    /// whose cell coordinate along `axis` is the `n`-th smallest value present. Each slice keeps
    /// the relative order of the region.
    pub fn slices(&self, axis: usize) -> Vec<Region> {
        let mut by_coord: std::collections::BTreeMap<i64, Vec<Site>> = Default::default();
        for s in &self.sites {
            by_coord.entry(s.cell[axis]).or_default().push(*s);
        }
        by_coord.into_values().map(|sites| Self::from_sites(&self.lattice, sites)).collect()
    }

    /// The same region translated by `cells` unit cells.
    pub fn translated(&self, cells: Cell) -> Region {
        let sites = self
            .sites
            .iter()
            .map(|s| Site {
                cell: [s.cell[0] + cells[0], s.cell[1] + cells[1], s.cell[2] + cells[2]],
                sub: s.sub,
            })
            .collect();
        Self::from_sites(&self.lattice, sites)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuboid_orders_by_first_axis() {
        let lat = Lattice::square(1.0, 4);
        let r = Region::cuboid(&lat, [3, 2, 0]);
        assert_eq!(r.len(), 6);
        assert_eq!(r.orbitals(), 24);
        assert_eq!(r.sites()[1], Site { cell: [0, 1, 0], sub: 0 });
        assert_eq!(r.offset(2), 8);
        let slices = r.slices(0);
        assert_eq!(slices.len(), 3);
        assert!(slices.iter().all(|s| s.len() == 2));
    }

    #[test]
    fn honeycomb_positions() {
        let lat = Lattice::honeycomb(1.0, 2);
        let b = lat.position(Site { cell: [0, 0, 0], sub: 1 });
        assert!((b[1] - 1.0).abs() < 1e-15);
        let r = Region::shaped(&lat, [4, 4, 0], |p| p[0] < 3.0);
        assert!(r.len() < 32);
    }
}
