//! Metal backend (`common.metal`, `chebyshev.metal`, `walk.metal`). The kernels are compiled from source when the device is
//! opened, without fast math (the double-single arithmetic needs exact IEEE rounding). Buffers
//! live in shared memory: on Apple Silicon the CPU reads the results without a copy.

use crate::{GpuError, Precision, doubled_moments};
use caroli_model::{SparseMatrix, c64};
use objc2::rc::Retained;
use objc2::runtime::{NSObjectProtocol, ProtocolObject};
use objc2::sel;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLMathMode, MTLResourceOptions, MTLSize,
};
use std::ffi::c_void;
use std::ptr::NonNull;

const SOURCE: &str =
    concat!(include_str!("common.metal"), include_str!("chebyshev.metal"), include_str!("walk.metal"));

/// Threads per threadgroup (`GROUP` in the kernels).
pub(crate) const GROUP: usize = 256;

/// Dispatches encoded per command buffer.
const BATCH: usize = 256;

pub(crate) type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub(crate) type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub(crate) type Encoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;
type Commands = Retained<ProtocolObject<dyn MTLCommandBuffer>>;

/// `Params` of the kernels.
#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    n: u32,
    vectors: u32,
    total: u32,
    first: u32,
    slot: u32,
}

pub(crate) struct Kernels {
    kpm_step: Pipeline,
    reduce_partials: Pipeline,
    propagate_step: Pipeline,
    propagate_init: Pipeline,
    pub(crate) walk_coin: Pipeline,
    pub(crate) walk_edge: Pipeline,
}

/// The GPU, its command queue and the compiled kernels.
pub struct MetalDevice {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn objc2_metal::MTLCommandQueue>>,
    single: Kernels,
    double_single: Kernels,
    /// Row probabilities of an f32 walk state, added to an accumulator.
    pub(crate) walk_accumulate: Pipeline,
}

pub(crate) fn backend(context: &str, error: &objc2_foundation::NSError) -> GpuError {
    GpuError::Backend(format!("{context}: {}", error.localizedDescription()))
}

/// Floats per complex number.
pub(crate) fn width(precision: Precision) -> usize {
    match precision {
        Precision::Single => 2,
        Precision::DoubleSingle => 4,
    }
}

/// `x = hi + lo` with two f32.
fn split(x: f64) -> [f32; 2] {
    let hi = x as f32;
    [hi, (x - hi as f64) as f32]
}

pub(crate) fn encode(z: &[c64], precision: Precision) -> Vec<f32> {
    match precision {
        Precision::Single => z.iter().flat_map(|v| [v.re as f32, v.im as f32]).collect(),
        Precision::DoubleSingle => z
            .iter()
            .flat_map(|v| {
                let (re, im) = (split(v.re), split(v.im));
                [re[0], re[1], im[0], im[1]]
            })
            .collect(),
    }
}

pub(crate) fn decode(data: &[f32], precision: Precision) -> Vec<c64> {
    match precision {
        Precision::Single => data.chunks_exact(2).map(|v| c64::new(v[0] as f64, v[1] as f64)).collect(),
        Precision::DoubleSingle => data
            .chunks_exact(4)
            .map(|v| c64::new(v[0] as f64 + v[1] as f64, v[2] as f64 + v[3] as f64))
            .collect(),
    }
}

pub(crate) fn threadgroups(total: usize) -> MTLSize {
    MTLSize { width: total.div_ceil(GROUP), height: 1, depth: 1 }
}

pub(crate) const GROUP_SIZE: MTLSize = MTLSize { width: GROUP, height: 1, depth: 1 };

impl MetalDevice {
    /// The default GPU with the kernels compiled. Fails where there is no GPU (virtual
    /// machines, some CI runners).
    pub fn new() -> Result<Self, GpuError> {
        let device =
            MTLCreateSystemDefaultDevice().ok_or_else(|| GpuError::Unavailable("no Metal device".into()))?;
        let options = MTLCompileOptions::new();
        if options.respondsToSelector(sel!(setMathMode:)) {
            options.setMathMode(MTLMathMode::Safe);
        } else {
            #[allow(deprecated)]
            options.setFastMathEnabled(false);
        }
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), Some(&options))
            .map_err(|e| backend("compiling the Metal kernels", &e))?;
        let pipeline = |name: &str| -> Result<Pipeline, GpuError> {
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| GpuError::Backend(format!("kernel `{name}` not found")))?;
            let p = device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| backend(&format!("building the {name} pipeline"), &e))?;
            if p.maxTotalThreadsPerThreadgroup() < GROUP {
                return Err(GpuError::Backend(format!(
                    "{name}: {} threads per group, {GROUP} needed",
                    p.maxTotalThreadsPerThreadgroup()
                )));
            }
            Ok(p)
        };
        let kernels = |suffix: &str| -> Result<Kernels, GpuError> {
            Ok(Kernels {
                kpm_step: pipeline(&format!("kpm_step_{suffix}"))?,
                reduce_partials: pipeline(&format!("reduce_partials_{suffix}"))?,
                propagate_step: pipeline(&format!("propagate_step_{suffix}"))?,
                propagate_init: pipeline(&format!("propagate_init_{suffix}"))?,
                walk_coin: pipeline(&format!("walk_coin_{suffix}"))?,
                walk_edge: pipeline(&format!("walk_edge_{suffix}"))?,
            })
        };
        let (single, double_single) = (kernels("f32")?, kernels("df64")?);
        let walk_accumulate = pipeline("walk_accumulate_f32")?;
        let queue = device.newCommandQueue().ok_or_else(|| GpuError::Backend("no command queue".into()))?;
        Ok(Self { device, queue, single, double_single, walk_accumulate })
    }

    /// The GPU's name, e.g. "Apple M5".
    pub fn name(&self) -> String {
        self.device.name().to_string()
    }

    /// Largest buffer the device can allocate, in bytes.
    pub fn max_buffer_length(&self) -> usize {
        self.device.maxBufferLength()
    }

    /// Runs `dispatches` encoded by `encode_one(encoder, index)`, `BATCH` per command buffer,
    /// and waits for the last. The dispatches of an encoder run one after the other.
    pub(crate) fn run(
        &self,
        context: &str,
        dispatches: usize,
        mut encode_one: impl FnMut(&Encoder, usize),
    ) -> Result<(), GpuError> {
        let mut submitted: Vec<Commands> = Vec::new();
        let mut index = 0;
        while index < dispatches {
            let commands =
                self.queue.commandBuffer().ok_or_else(|| GpuError::Backend("no command buffer".into()))?;
            let encoder = commands
                .computeCommandEncoder()
                .ok_or_else(|| GpuError::Backend("no compute encoder".into()))?;
            for i in index..(index + BATCH).min(dispatches) {
                encode_one(&encoder, i);
            }
            encoder.endEncoding();
            commands.commit();
            index += BATCH;
            submitted.push(commands);
        }
        // Command buffers of one queue complete in order: wait for the last, then check all.
        if let Some(last) = submitted.last() {
            last.waitUntilCompleted();
        }
        for commands in &submitted {
            if let Some(error) = commands.error() {
                return Err(backend(context, &error));
            }
        }
        Ok(())
    }

    pub(crate) fn kernels(&self, precision: Precision) -> &Kernels {
        match precision {
            Precision::Single => &self.single,
            Precision::DoubleSingle => &self.double_single,
        }
    }

    pub(crate) fn buffer_from<T: Copy>(&self, data: &[T]) -> Result<Buffer, GpuError> {
        let bytes = std::mem::size_of_val(data);
        if bytes == 0 {
            return self.buffer(4);
        }
        let pointer = NonNull::new(data.as_ptr() as *mut c_void).expect("slice pointer");
        // SAFETY: `pointer` is valid for `bytes` bytes, which Metal copies before returning.
        unsafe {
            self.device.newBufferWithBytes_length_options(
                pointer,
                bytes,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| GpuError::Backend(format!("allocating a {bytes}-byte buffer")))
    }

    pub(crate) fn buffer(&self, bytes: usize) -> Result<Buffer, GpuError> {
        self.device
            .newBufferWithLength_options(bytes.max(4), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| GpuError::Backend(format!("allocating a {bytes}-byte buffer")))
    }

    /// Uploads a rescaled Hamiltonian `H̃` (spectrum in (−1, 1)).
    pub fn upload(&self, h: &SparseMatrix, precision: Precision) -> Result<MetalOperator<'_>, GpuError> {
        let to_u32 = |x: usize| {
            u32::try_from(x).map_err(|_| GpuError::Backend("matrix too large for u32 indices".into()))
        };
        let row_ptr = h.row_ptr.iter().map(|&x| to_u32(x)).collect::<Result<Vec<u32>, _>>()?;
        Ok(MetalOperator {
            gpu: self,
            precision,
            n: h.n,
            row_ptr: self.buffer_from(&row_ptr)?,
            cols: self.buffer_from(&h.cols)?,
            vals: self.buffer_from(&encode(&h.vals, precision))?,
        })
    }
}

/// A sparse Hamiltonian resident on the GPU.
pub struct MetalOperator<'a> {
    gpu: &'a MetalDevice,
    precision: Precision,
    n: usize,
    row_ptr: Buffer,
    cols: Buffer,
    vals: Buffer,
}

/// Binds buffers from index 0 and `bytes` after them.
///
/// # Safety
///
/// The arguments must be those the kernel expects, and the buffers must outlive the command
/// buffer.
pub(crate) unsafe fn bind(encoder: &Encoder, buffers: &[&Buffer], bytes: &[(NonNull<c_void>, usize)]) {
    for (i, b) in buffers.iter().enumerate() {
        // SAFETY: per the caller.
        unsafe { encoder.setBuffer_offset_atIndex(Some(&***b), 0, i) };
    }
    for (j, (pointer, length)) in bytes.iter().enumerate() {
        // SAFETY: per the caller; Metal copies the bytes when encoding.
        unsafe { encoder.setBytes_length_atIndex(*pointer, *length, buffers.len() + j) };
    }
}

pub(crate) fn as_bytes<T>(value: &T) -> (NonNull<c_void>, usize) {
    (NonNull::from(value).cast(), std::mem::size_of::<T>())
}

impl MetalOperator<'_> {
    fn bytes(&self, entries: usize) -> usize {
        entries * width(self.precision) * std::mem::size_of::<f32>()
    }

    fn run(&self, dispatches: usize, encode_one: impl FnMut(&Encoder, usize)) -> Result<(), GpuError> {
        self.gpu.run("running the Chebyshev kernels", dispatches, encode_one)
    }

    fn read(&self, buffer: &Buffer, entries: usize) -> Vec<c64> {
        let floats = entries * width(self.precision);
        // SAFETY: the buffer holds at least `floats` f32 written by completed commands.
        let data = unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr() as *const f32, floats) };
        decode(data, self.precision)
    }

    /// KPM moments `μ_n = kernel_n · mean_r ⟨r| T_n(H̃) |r⟩`, `n = 0..kernel.len()`, all start
    /// vectors at once, with the doubling relations (as [`caroli_kpm::block_moments`]).
    pub fn kpm_moments(&self, starts: &[Vec<c64>], kernel: &[f64]) -> Result<Vec<c64>, GpuError> {
        let (n, vectors) = (self.n, starts.len());
        assert!(vectors > 0, "no start vectors");
        assert!(starts.iter().all(|s| s.len() == n), "start vectors must have {n} entries");
        assert!(kernel.len() >= 2, "at least two moments");
        let total = n * vectors;
        let total32 = u32::try_from(total).map_err(|_| GpuError::Backend("too many entries".into()))?;
        let block: Vec<c64> = (0..total).map(|i| starts[i % vectors][i / vectors]).collect();
        let t = [
            self.gpu.buffer_from(&encode(&block, self.precision))?,
            self.gpu.buffer(self.bytes(total))?,
            self.gpu.buffer(self.bytes(total))?,
        ];
        let groups = total.div_ceil(GROUP);
        let steps = kernel.len().div_ceil(2);
        // Per-threadgroup shares of the current step, then the reduced sums of every step.
        let partials = self.gpu.buffer(self.bytes(2 * groups))?;
        // The sums are always double-single (4 floats per complex).
        let sums = self.gpu.buffer(2 * steps * 4 * std::mem::size_of::<f32>())?;
        let kernels = self.gpu.kernels(self.precision);
        self.run(2 * steps, |encoder, i| {
            let s = i / 2;
            if i % 2 == 0 {
                let params = Params {
                    n: n as u32,
                    vectors: vectors as u32,
                    total: total32,
                    first: (s == 0) as u32,
                    slot: s as u32,
                };
                let (prev, cur, next) = (&t[(s + 2) % 3], &t[s % 3], &t[(s + 1) % 3]);
                encoder.setComputePipelineState(&kernels.kpm_step);
                // SAFETY: `kpm_step(row_ptr, cols, vals, prev, cur, next, partials, params)`; the
                // buffers live until `run` returns.
                unsafe {
                    bind(
                        encoder,
                        &[&self.row_ptr, &self.cols, &self.vals, prev, cur, next, &partials],
                        &[as_bytes(&params)],
                    )
                };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups(total), GROUP_SIZE);
            } else {
                let params = Params { n: groups as u32, vectors: 0, total: 0, first: 0, slot: s as u32 };
                encoder.setComputePipelineState(&kernels.reduce_partials);
                // SAFETY: `reduce_partials(partials, sums, params)`.
                unsafe { bind(encoder, &[&partials, &sums], &[as_bytes(&params)]) };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups(1), GROUP_SIZE);
            }
        })?;
        // SAFETY: `sums` holds 2·steps double-single complex numbers written by the kernels.
        let raw =
            unsafe { std::slice::from_raw_parts(sums.contents().as_ptr() as *const f32, 2 * steps * 4) };
        let all = decode(raw, Precision::DoubleSingle);
        let s1: Vec<c64> = all.iter().step_by(2).copied().collect();
        let s2: Vec<c64> = all.iter().skip(1).step_by(2).copied().collect();
        Ok(doubled_moments(&s1, &s2, vectors, kernel))
    }

    /// `steps` propagation steps `ψ ← Σ_n c_n T_n(H̃) ψ` (coefficients from
    /// [`caroli_dynamics::Chebyshev`]) for a block of `vectors` wave functions stored
    /// interleaved (`ψ[row·vectors + v]`, as `SparseMatrix::apply_block`).
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
        // Buffers: three Chebyshev terms and the accumulator; T_0 and the accumulator swap
        // roles after each step.
        let mut b = vec![self.gpu.buffer_from(&encode(psi, self.precision))?];
        for _ in 0..3 {
            b.push(self.gpu.buffer(self.bytes(total))?);
        }
        let coefficients = encode(coefficients, self.precision);
        let w = width(self.precision);
        let kernels = self.gpu.kernels(self.precision);
        let terms = coefficients.len() / w;
        let (mut t0, mut acc) = (0usize, 3usize);
        let mut roles = Vec::with_capacity(steps);
        for _ in 0..steps {
            roles.push((t0, acc));
            std::mem::swap(&mut t0, &mut acc);
        }
        // Per step: one init dispatch and `terms − 1` Chebyshev dispatches.
        self.run(steps * terms, |encoder, i| {
            let (step, k) = (i / terms, i % terms);
            let (t0, acc) = roles[step];
            let others: Vec<usize> = (0..4).filter(|&j| j != t0 && j != acc).collect();
            let t = [t0, others[0], others[1]];
            let coefficient = &coefficients[k * w..(k + 1) * w];
            let c = (NonNull::from(coefficient).cast(), std::mem::size_of_val(coefficient));
            if k == 0 {
                let params =
                    Params { n: n as u32, vectors: vectors as u32, total: total32, first: 0, slot: 0 };
                encoder.setComputePipelineState(&kernels.propagate_init);
                // SAFETY: `propagate_init(cur, psi, coefficient, params)`.
                unsafe { bind(encoder, &[&b[t0], &b[acc]], &[c, as_bytes(&params)]) };
            } else {
                let s = k - 1;
                let params = Params {
                    n: n as u32,
                    vectors: vectors as u32,
                    total: total32,
                    first: (s == 0) as u32,
                    slot: 0,
                };
                let (prev, cur, next) = (&b[t[(s + 2) % 3]], &b[t[s % 3]], &b[t[(s + 1) % 3]]);
                encoder.setComputePipelineState(&kernels.propagate_step);
                // SAFETY: `propagate_step(row_ptr, cols, vals, prev, cur, next, psi, coefficient,
                // params)`.
                unsafe {
                    bind(
                        encoder,
                        &[&self.row_ptr, &self.cols, &self.vals, prev, cur, next, &b[acc]],
                        &[c, as_bytes(&params)],
                    )
                };
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups(total), GROUP_SIZE);
        })?;
        psi.copy_from_slice(&self.read(&b[t0], total));
        Ok(())
    }
}
