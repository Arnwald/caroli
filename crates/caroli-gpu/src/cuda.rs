//! CUDA backend (`chebyshev.cu`, f64), compiled at run time by NVRTC. The CUDA libraries are
//! loaded when the device is opened, so building needs no CUDA toolkit; running needs an
//! NVIDIA driver and NVRTC.

use crate::{GpuError, doubled_moments};
use caroli_model::{SparseMatrix, c64};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

const SOURCE: &str = include_str!("chebyshev.cu");

/// Threads per block (`GROUP` in the kernels).
const GROUP: usize = 256;

fn backend(context: &str, error: impl std::fmt::Debug) -> GpuError {
    GpuError::Backend(format!("{context}: {error:?}"))
}

fn flatten(z: &[c64]) -> Vec<f64> {
    z.iter().flat_map(|v| [v.re, v.im]).collect()
}

fn unflatten(x: &[f64]) -> Vec<c64> {
    x.chunks_exact(2).map(|v| c64::new(v[0], v[1])).collect()
}

fn config(total: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (total.div_ceil(GROUP) as u32, 1, 1),
        block_dim: (GROUP as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// A CUDA device, its stream and the compiled kernels.
pub struct CudaDevice {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kpm_step: CudaFunction,
    propagate_step: CudaFunction,
    propagate_init: CudaFunction,
}

impl CudaDevice {
    /// Device `ordinal` with the kernels compiled.
    pub fn new(ordinal: usize) -> Result<Self, GpuError> {
        // cudarc panics when it cannot load a library: check first.
        // SAFETY: these only try to open the shared libraries.
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Err(GpuError::Unavailable("CUDA driver library not found".into()));
        }
        if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
            return Err(GpuError::Unavailable("NVRTC library not found".into()));
        }
        let context = CudaContext::new(ordinal).map_err(|e| GpuError::Unavailable(format!("{e:?}")))?;
        let ptx = cudarc::nvrtc::compile_ptx(SOURCE).map_err(|e| backend("compiling the CUDA kernels", e))?;
        let module = context.load_module(ptx).map_err(|e| backend("loading the CUDA kernels", e))?;
        let function = |name: &str| module.load_function(name).map_err(|e| backend(name, e));
        Ok(Self {
            stream: context.default_stream(),
            kpm_step: function("kpm_step")?,
            propagate_step: function("propagate_step")?,
            propagate_init: function("propagate_init")?,
            context,
        })
    }

    /// The GPU's name.
    pub fn name(&self) -> String {
        self.context.name().unwrap_or_else(|_| "CUDA device".into())
    }

    /// Uploads a rescaled Hamiltonian `H̃` (spectrum in (−1, 1)).
    pub fn upload(&self, h: &SparseMatrix) -> Result<CudaOperator<'_>, GpuError> {
        let row_ptr = h
            .row_ptr
            .iter()
            .map(|&x| {
                u32::try_from(x).map_err(|_| GpuError::Backend("matrix too large for u32 indices".into()))
            })
            .collect::<Result<Vec<u32>, _>>()?;
        let s = &self.stream;
        Ok(CudaOperator {
            gpu: self,
            n: h.n,
            row_ptr: s.clone_htod(&row_ptr).map_err(|e| backend("upload", e))?,
            cols: s.clone_htod(&h.cols).map_err(|e| backend("upload", e))?,
            vals: s.clone_htod(&flatten(&h.vals)).map_err(|e| backend("upload", e))?,
        })
    }
}

/// A sparse Hamiltonian resident on a CUDA device.
pub struct CudaOperator<'a> {
    gpu: &'a CudaDevice,
    n: usize,
    row_ptr: CudaSlice<u32>,
    cols: CudaSlice<u32>,
    vals: CudaSlice<f64>,
}

impl CudaOperator<'_> {
    fn zeros(&self, entries: usize) -> Result<CudaSlice<f64>, GpuError> {
        self.gpu.stream.alloc_zeros::<f64>(2 * entries.max(1)).map_err(|e| backend("allocation", e))
    }

    /// KPM moments `μ_n = kernel_n · mean_r ⟨r| T_n(H̃) |r⟩` for a block of start vectors, with
    /// the doubling relations (as [`caroli_kpm::block_moments`]).
    pub fn kpm_moments(&self, starts: &[Vec<c64>], kernel: &[f64]) -> Result<Vec<c64>, GpuError> {
        let (n, vectors) = (self.n, starts.len());
        assert!(starts.iter().all(|s| s.len() == n), "start vectors must have {n} entries");
        assert!(kernel.len() >= 2, "at least two moments");
        let total = n * vectors;
        let total32 = u32::try_from(total).map_err(|_| GpuError::Backend("too many entries".into()))?;
        let block: Vec<c64> = (0..total).map(|i| starts[i % vectors][i / vectors]).collect();
        let s = &self.gpu.stream;
        let mut prev = self.zeros(total)?;
        let mut cur = s.clone_htod(&flatten(&block)).map_err(|e| backend("upload", e))?;
        let mut next = self.zeros(total)?;
        let groups = total.div_ceil(GROUP);
        let steps = kernel.len().div_ceil(2);
        let mut partials = self.zeros(2 * steps * groups)?;
        let vectors32 = vectors as u32;
        for step in 0..steps {
            let (first, slot) = ((step == 0) as u32, step as u32);
            let mut b = s.launch_builder(&self.gpu.kpm_step);
            b.arg(&self.row_ptr).arg(&self.cols).arg(&self.vals).arg(&prev).arg(&cur).arg(&mut next);
            b.arg(&mut partials).arg(&vectors32).arg(&total32).arg(&first).arg(&slot);
            // SAFETY: the arguments match `kpm_step`; every index stays below `total` or the
            // partial sums allocated for `steps × groups` blocks.
            unsafe { b.launch(config(total)) }.map_err(|e| backend("kpm_step", e))?;
            // (prev, cur, next) ← (T_n, T_{n+1}, free).
            std::mem::swap(&mut prev, &mut cur);
            std::mem::swap(&mut cur, &mut next);
        }
        let parts = unflatten(&s.clone_dtoh(&partials).map_err(|e| backend("download", e))?);
        let (mut s1, mut s2) = (Vec::with_capacity(steps), Vec::with_capacity(steps));
        for step in parts.chunks_exact(2 * groups).take(steps) {
            s1.push(step.iter().step_by(2).sum());
            s2.push(step.iter().skip(1).step_by(2).sum());
        }
        Ok(doubled_moments(&s1, &s2, vectors, kernel))
    }

    /// `steps` propagation steps `ψ ← Σ_n c_n T_n(H̃) ψ` for a block of `vectors` wave
    /// functions stored interleaved (`ψ[row·vectors + v]`).
    pub fn propagate(
        &self,
        coefficients: &[c64],
        psi: &mut [c64],
        vectors: usize,
        steps: usize,
    ) -> Result<(), GpuError> {
        let n = self.n;
        assert!(
            vectors > 0 && psi.len() == n * vectors,
            "ψ must hold {vectors} wave functions of {n} entries"
        );
        assert!(coefficients.len() >= 2, "at least two coefficients");
        let total = psi.len();
        let total32 = u32::try_from(total).map_err(|_| GpuError::Backend("too many entries".into()))?;
        let vectors32 = vectors as u32;
        let s = &self.gpu.stream;
        // `cur` holds T_0 = ψ at the start of each step; `acc` accumulates Σ c_k T_k.
        let mut cur = s.clone_htod(&flatten(psi)).map_err(|e| backend("upload", e))?;
        let (mut prev, mut next, mut acc) = (self.zeros(total)?, self.zeros(total)?, self.zeros(total)?);
        for _ in 0..steps {
            let c0 = coefficients[0];
            let mut l = s.launch_builder(&self.gpu.propagate_init);
            l.arg(&cur).arg(&mut acc).arg(&c0.re).arg(&c0.im).arg(&total32);
            // SAFETY: the arguments match `propagate_init`.
            unsafe { l.launch(config(total)) }.map_err(|e| backend("propagate_init", e))?;
            for (k, c) in coefficients.iter().enumerate().skip(1) {
                let first = (k == 1) as u32;
                let mut l = s.launch_builder(&self.gpu.propagate_step);
                l.arg(&self.row_ptr).arg(&self.cols).arg(&self.vals).arg(&prev).arg(&cur).arg(&mut next);
                l.arg(&mut acc).arg(&c.re).arg(&c.im).arg(&vectors32).arg(&total32).arg(&first);
                // SAFETY: the arguments match `propagate_step`.
                unsafe { l.launch(config(total)) }.map_err(|e| backend("propagate_step", e))?;
                // (prev, cur, next) ← (T_{k−1}, T_k, free).
                std::mem::swap(&mut prev, &mut cur);
                std::mem::swap(&mut cur, &mut next);
            }
            // ψ(t + dt) is T_0 of the next step.
            std::mem::swap(&mut cur, &mut acc);
        }
        let out = unflatten(&s.clone_dtoh(&cur).map_err(|e| backend("download", e))?);
        psi.copy_from_slice(&out[..total]);
        Ok(())
    }
}
