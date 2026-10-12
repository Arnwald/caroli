//! GPU backends for the Chebyshev kernels of caroli: kernel-polynomial moments and time
//! propagation with a sparse Hamiltonian ([`caroli_model::SparseMatrix`]) rescaled so that its
//! spectrum lies in (−1, 1).
//!
//! The work is a chain of fused steps `T_{n+1} = 2 H̃ T_n − T_{n−1}`. Each step reduces
//! `⟨r|T_{n+1}⟩` (KPM) or accumulates `c_{n+1} T_{n+1}` (propagation) as it goes, so a whole
//! expansion runs on the GPU and only the moments or the final state come back. Several vectors
//! go through each step together (stochastic traces, several wave packets).
//!
//! - **Metal** (macOS). Apple GPUs have no f64:
//!   - [`Precision::Single`] runs in f32, about 1e-6 relative;
//!   - [`Precision::DoubleSingle`] stores and computes every number as an unevaluated sum of two
//!     f32, about 1e-14 relative, for twice the memory traffic of f32.
//! - **CUDA** (feature `cuda`): f64.

use caroli_model::c64;

#[cfg(target_os = "macos")]
pub mod metal;

#[cfg(target_os = "macos")]
mod metal_walk;

#[cfg(target_os = "macos")]
pub use metal::{MetalDevice, MetalOperator};

#[cfg(target_os = "macos")]
pub use metal_walk::{MAX_DEGREE, MetalWalk, WalkTables};

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::{CudaDevice, CudaOperator};

/// Floating-point format of a Metal computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// f32.
    Single,
    /// Double-single: `hi + lo` with two f32 (48-bit significand).
    DoubleSingle,
}

#[derive(thiserror::Error, Debug)]
pub enum GpuError {
    #[error("no GPU: {0}")]
    Unavailable(String),
    #[error("GPU: {0}")]
    Backend(String),
}

/// `μ_n = kernel_n · mean_r ⟨r| T_n |r⟩` from the block sums `s1[n] = Σ_r ⟨T_n|T_n⟩` and
/// `s2[n] = Σ_r ⟨T_n|T_{n+1}⟩` (`μ_2n = 2 s1[n] − μ_0`, `μ_{2n+1} = 2 s2[n] − μ_1`).
pub(crate) fn doubled_moments(s1: &[c64], s2: &[c64], vectors: usize, kernel: &[f64]) -> Vec<c64> {
    let (mu0, mu1) = (s1[0], s2[0]);
    (0..kernel.len())
        .map(|k| {
            let raw = match k {
                0 => mu0,
                1 => mu1,
                _ if k % 2 == 0 => 2.0 * s1[k / 2] - mu0,
                _ => 2.0 * s2[k / 2] - mu1,
            };
            raw * (kernel[k] / vectors as f64)
        })
        .collect()
}
