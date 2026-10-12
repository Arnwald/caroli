//! Graphs whose edges carry the spins, with an order on the edges at each node.
//!
//! The walker lives on the half-edges: node `x` has one half-edge per incident edge, and the
//! position of a half-edge among those of its node is its color. Half-edges are numbered node
//! by node, so the half-edges of a node are consecutive.

use std::ops::Range;

/// A graph with numbered edges (edge `e` carries spin `e`) and colored half-edges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Graph {
    nodes: usize,
    /// Half-edges of node `x`: `offsets[x]..offsets[x + 1]`, in color order.
    offsets: Vec<usize>,
    /// Node and edge of each half-edge.
    half_edges: Vec<(usize, usize)>,
    /// The two half-edges of each edge, the first at the edge's first node.
    halves: Vec<(usize, usize)>,
}

impl Graph {
    /// From the colored neighbourhoods: `colors[x][c]` is the edge of color `c` at node `x`, and
    /// `ends[e]` the two nodes of edge `e` (they may coincide only for distinct edges, as in a
    /// two-node cycle). Every edge must appear exactly once at each of its ends.
    pub fn new(ends: &[(usize, usize)], colors: &[Vec<usize>]) -> Self {
        let nodes = colors.len();
        let mut offsets = vec![0];
        let mut half_edges = Vec::new();
        for (x, edges) in colors.iter().enumerate() {
            for &e in edges {
                assert!(e < ends.len(), "node {x}: edge {e} does not exist");
                half_edges.push((x, e));
            }
            offsets.push(half_edges.len());
        }
        let mut halves = vec![(usize::MAX, usize::MAX); ends.len()];
        for (e, &(x, y)) in ends.iter().enumerate() {
            assert!(x < nodes && y < nodes, "edge {e} joins a missing node");
            assert!(x != y, "edge {e} is a loop");
            let at = |node: usize| {
                let found: Vec<usize> =
                    (offsets[node]..offsets[node + 1]).filter(|&h| half_edges[h].1 == e).collect();
                assert!(found.len() == 1, "edge {e} must appear once at node {node}");
                found[0]
            };
            halves[e] = (at(x), at(y));
        }
        assert!(half_edges.len() == 2 * ends.len(), "every half-edge must belong to an edge of the list");
        Self { nodes, offsets, half_edges, halves }
    }

    /// Edges in the order given; the colors of each node follow the order in which its edges
    /// appear in the list.
    pub fn from_edges(nodes: usize, ends: &[(usize, usize)]) -> Self {
        let mut colors = vec![Vec::new(); nodes];
        for (e, &(x, y)) in ends.iter().enumerate() {
            colors[x].push(e);
            colors[y].push(e);
        }
        Self::new(ends, &colors)
    }

    /// Ring of `n ≥ 2` nodes, as in Sellapillay & Verga: edge `x` joins `x` and `x + 1`, color 0
    /// of node `x` points to `x + 1` and color 1 to `x − 1`.
    pub fn cycle(n: usize) -> Self {
        assert!(n >= 2, "a cycle needs two nodes");
        let ends: Vec<(usize, usize)> = (0..n).map(|x| (x, (x + 1) % n)).collect();
        let colors: Vec<Vec<usize>> = (0..n).map(|x| vec![x, (x + n - 1) % n]).collect();
        Self::new(&ends, &colors)
    }

    /// Chain of `n ≥ 2` nodes: edge `x` joins `x` and `x + 1`; color 0 points to `x + 1` and
    /// color 1 to `x − 1` (the end nodes have one color).
    pub fn path(n: usize) -> Self {
        assert!(n >= 2, "a path needs two nodes");
        let ends: Vec<(usize, usize)> = (0..n - 1).map(|x| (x, x + 1)).collect();
        let colors: Vec<Vec<usize>> = (0..n)
            .map(|x| {
                let mut c = Vec::new();
                if x + 1 < n {
                    c.push(x);
                }
                if x > 0 {
                    c.push(x - 1);
                }
                c
            })
            .collect();
        Self::new(&ends, &colors)
    }

    pub fn nodes(&self) -> usize {
        self.nodes
    }

    pub fn edges(&self) -> usize {
        self.halves.len()
    }

    /// Number of half-edges, `2|E|`.
    pub fn half_edges(&self) -> usize {
        self.half_edges.len()
    }

    /// Degree of node `x`.
    pub fn degree(&self, x: usize) -> usize {
        self.offsets[x + 1] - self.offsets[x]
    }

    /// Largest degree.
    pub fn max_degree(&self) -> usize {
        (0..self.nodes).map(|x| self.degree(x)).max().unwrap_or(0)
    }

    /// Half-edges of node `x`, in color order.
    pub fn half_edges_of(&self, x: usize) -> Range<usize> {
        self.offsets[x]..self.offsets[x + 1]
    }

    /// The half-edge of color `c` at node `x`.
    pub fn half_edge(&self, x: usize, c: usize) -> usize {
        assert!(c < self.degree(x), "node {x} has no color {c}");
        self.offsets[x] + c
    }

    /// Node and color of half-edge `h`.
    pub fn node_and_color(&self, h: usize) -> (usize, usize) {
        let x = self.half_edges[h].0;
        (x, h - self.offsets[x])
    }

    /// Edge of half-edge `h`.
    pub fn edge_of(&self, h: usize) -> usize {
        self.half_edges[h].1
    }

    /// The two half-edges of edge `e`.
    pub fn halves(&self, e: usize) -> (usize, usize) {
        self.halves[e]
    }

    /// The two nodes of edge `e`.
    pub fn ends(&self, e: usize) -> (usize, usize) {
        let (a, b) = self.halves[e];
        (self.half_edges[a].0, self.half_edges[b].0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cycle_colors() {
        let g = Graph::cycle(5);
        assert_eq!((g.nodes(), g.edges(), g.half_edges()), (5, 5, 10));
        // Edge 2 joins color 0 of node 2 and color 1 of node 3.
        assert_eq!(g.halves(2), (g.half_edge(2, 0), g.half_edge(3, 1)));
        assert_eq!(g.ends(4), (4, 0));
        assert_eq!(g.node_and_color(g.half_edge(4, 1)), (4, 1));
        assert_eq!(g.edge_of(g.half_edge(0, 1)), 4);
        // Two nodes: two distinct edges between them.
        let g = Graph::cycle(2);
        assert_eq!(g.halves(0), (0, 3));
        assert_eq!(g.halves(1), (2, 1));
    }

    #[test]
    fn path_and_edge_list() {
        let g = Graph::path(4);
        assert_eq!((g.edges(), g.degree(0), g.degree(1), g.degree(3)), (3, 1, 2, 1));
        assert_eq!(g.halves(0), (g.half_edge(0, 0), g.half_edge(1, 1)));
        let star = Graph::from_edges(4, &[(0, 1), (0, 2), (0, 3)]);
        assert_eq!((star.degree(0), star.max_degree()), (3, 3));
        assert_eq!(star.halves(2), (star.half_edge(0, 2), star.half_edge(3, 0)));
    }
}
