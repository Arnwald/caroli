//! caroli — a toolbox for electronic transport in materials.
//!
//! Build a tight-binding system ([`lattice`], [`model`]), cut it into slices between leads and
//! compute steady-state transport with Green's functions ([`green`]).
//!
//! ```
//! use caroli::prelude::*;
//!
//! // Clean BHZ ribbon (HgTe quantum well), 16 × 10 sites: two helical edge channels in the gap.
//! let model = bhz(BhzParams::default());
//! let region = Region::cuboid(model.lattice(), [16, 10, 0]);
//! let device = TwoTerminal::from_model(&model, &model, &region, 0);
//! let t = device.transmission(0.0).unwrap();
//! assert!((t - 2.0).abs() < 1e-8);
//! ```

pub use caroli_born as born;
pub use caroli_disorder as disorder;
pub use caroli_dynamics as dynamics;
pub use caroli_gpu as gpu;
pub use caroli_green as green;
pub use caroli_keldysh as keldysh;
pub use caroli_kpm as kpm;
pub use caroli_lattice as lattice;
pub use caroli_model as model;
pub use caroli_torus as torus;

/// The types needed for most calculations.
pub mod prelude {
    pub use caroli_disorder::rng::Xoshiro256;
    pub use caroli_disorder::{Anderson, MagneticImpurities, Moments};
    pub use caroli_green::{
        BondCurrent, Interface, Lead, LeadSelfEnergies, MultiTerminal, NestedMultiTerminal,
        NestedTwoTerminal, ScatteringStates, SelfEnergyMethod, TwoTerminal, lead_voltages,
    };
    pub use caroli_keldysh::{Lesser, fermi};
    pub use caroli_lattice::{Lattice, Region, Site};
    pub use caroli_model::continuum::Continuum;
    pub use caroli_model::pauli::{kron, s0, sx, sy, sz};
    pub use caroli_model::presets::{BhzParams, GFactors, bhz, bhz_exchange, bhz_spin};
    pub use caroli_model::{CMat, Model, SparseMatrix, c64, combine, gauge};
}
