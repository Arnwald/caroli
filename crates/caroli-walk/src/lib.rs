//! Discrete-time quantum walks with spins: a walker jumps between the nodes of a graph and
//! interacts with spins sitting on its edges (K. Sellapillay and A. D. Verga, "Quantum walk on a
//! graph of spins: magnetism and entanglement", Phys. Rev. E 103, 032123 (2021)).
//!
//! The basis is `|x c s⟩`: node `x`, color `c` (one per edge at `x`: the half-edges of the
//! node), spin configuration `s`. One step is `U = V(J) M C` ([`SpinWalk`]). The Hilbert space
//! has dimension `2|E| · 2^|E|`, about 2·10⁷ for the 19-node ring.
//!
//! The state is stored half-edge by half-edge with the spin configuration fastest, so each part
//! of a step is one sweep over memory: the coin pairs the rows of a node, the motion and the
//! interaction pair the two rows of an edge and, inside them, the configurations with that
//! edge's spin up and down, like a one-qubit gate in a state-vector simulator. The CPU version
//! runs these sweeps in parallel; [`gpu`] runs them on the GPU, checked against the CPU.
//!
//! ```
//! use caroli_walk::{PLUS, SpinWalk};
//!
//! // Ring of 6 nodes, coin angle π/8, coupling 0.2, spins along +x, walker on node 0.
//! let walk = SpinWalk::ring(6, std::f64::consts::PI / 8.0, 0.2);
//! let mut psi = walk.product_state(&walk.localized(0, 0), &[PLUS; 6]);
//! walk.steps(&mut psi, 100);
//! let total: f64 = walk.node_probabilities(&psi).iter().sum();
//! assert!((total - 1.0).abs() < 1e-12);
//! ```

pub mod gpu;
pub mod graph;
pub mod observables;
pub mod walk;

pub use caroli_gpu::Precision;
pub use gpu::{Device, Options, Report, evolve};
pub use graph::Graph;
pub use walk::{PLUS, Real, SpinWalk, UP, fourier, grover, rotation, to_double, to_single};
