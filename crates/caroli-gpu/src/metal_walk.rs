//! Quantum walk on a graph of spins on the GPU (`walk.metal`): the state `ψ[h · 2^bits + s]` of
//! `caroli_walk::SpinWalk` lives in one shared buffer, and a step is two dispatches (coin, then
//! motion and interaction on every edge).

use crate::metal::{Buffer, GROUP_SIZE, MetalDevice, as_bytes, bind, encode, threadgroups, width};
use crate::{GpuError, Precision};
use caroli_model::c64;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder};
use rayon::prelude::*;

/// Largest node degree the kernels take (`MAX_DEGREE` in `walk.metal`).
pub const MAX_DEGREE: usize = 8;

/// `WalkParams` of the kernels.
#[repr(C)]
#[derive(Clone, Copy)]
struct WalkParams {
    bits: u32,
    nodes: u32,
    edges: u32,
    batch: u32,
    coin_len: u32,
}

/// What the kernels need to know about a walk.
pub struct WalkTables<'a> {
    /// Number of spins (edges): a row has `2^bits` entries.
    pub bits: usize,
    /// Rows (half-edges) of node `x`: `node_rows[x]..node_rows[x + 1]`.
    pub node_rows: &'a [usize],
    /// Coin of each node, `d × d` row-major.
    pub coins: &'a [Vec<c64>],
    /// The two rows (half-edges) of each edge.
    pub halves: &'a [(usize, usize)],
    /// `(p, α, β)` of the edge update `a0' = p b0`, `a1' = α b1 + β a0`, `b0' = β b1 + α a0`,
    /// `b1' = p a1`.
    pub exchange: [c64; 3],
}

/// A walk state resident on the GPU, or the states of a batch of walks of one graph.
pub struct MetalWalk<'a> {
    gpu: &'a MetalDevice,
    precision: Precision,
    params: WalkParams,
    entries: usize,
    psi: Buffer,
    node_rows: Buffer,
    coin_at: Buffer,
    coins: Buffer,
    halves: Buffer,
    exchange: Buffer,
    /// Accumulated row probabilities (f32), one per half-edge of each walk.
    acc: Buffer,
    /// Rows of all the walks.
    rows: usize,
}

fn to_u32(x: usize, what: &str) -> Result<u32, GpuError> {
    u32::try_from(x)
        .map_err(|_| GpuError::Backend(format!("{what} does not fit the kernels' 32-bit indices")))
}

impl MetalDevice {
    /// Uploads a walk and its state `psi` in `precision`.
    pub fn walk(
        &self,
        tables: &WalkTables,
        psi: &[c64],
        precision: Precision,
    ) -> Result<MetalWalk<'_>, GpuError> {
        self.walks(std::slice::from_ref(tables), psi, precision)
    }

    /// Uploads a batch of walks of one graph (their coins and couplings may differ) and their
    /// states, one after the other in `psi`, in `precision`. They step together.
    pub fn walks(
        &self,
        batch: &[WalkTables],
        psi: &[c64],
        precision: Precision,
    ) -> Result<MetalWalk<'_>, GpuError> {
        let tables = batch.first().ok_or_else(|| GpuError::Backend("an empty batch".into()))?;
        let nodes = tables.node_rows.len() - 1;
        let edges = tables.halves.len();
        let rows = tables.node_rows[nodes];
        let bits = tables.bits;
        if bits == 0 || bits != edges {
            return Err(GpuError::Backend("one spin per edge, and at least one edge".into()));
        }
        if batch
            .iter()
            .any(|t| t.bits != bits || t.node_rows != tables.node_rows || t.halves != tables.halves)
        {
            return Err(GpuError::Backend("the walks of a batch must share their graph".into()));
        }
        if (0..nodes).any(|x| tables.node_rows[x + 1] - tables.node_rows[x] > MAX_DEGREE) {
            return Err(GpuError::Backend(format!("node degrees above {MAX_DEGREE}")));
        }
        let entries = batch.len() * (rows << bits);
        if psi.len() != entries {
            return Err(GpuError::Backend(format!("state of {} entries, {entries} expected", psi.len())));
        }
        let bytes = entries * width(precision) * std::mem::size_of::<f32>();
        if bytes > self.max_buffer_length() {
            return Err(GpuError::Backend(format!(
                "a state of {bytes} bytes exceeds the largest Metal buffer ({} bytes)",
                self.max_buffer_length()
            )));
        }
        to_u32(batch.len() * (nodes << bits), "the coin grid")?;
        to_u32(entries, "the batch")?;
        let node_rows: Vec<u32> =
            tables.node_rows.iter().map(|&r| to_u32(r, "a row index")).collect::<Result<_, _>>()?;
        let mut coin_at = Vec::with_capacity(nodes);
        let mut coin_len = 0;
        for x in 0..nodes {
            let d = tables.node_rows[x + 1] - tables.node_rows[x];
            coin_at.push(to_u32(coin_len, "the coin table")?);
            coin_len += d * d;
        }
        let mut coins = Vec::with_capacity(batch.len() * coin_len);
        for t in batch {
            for (x, coin) in t.coins.iter().enumerate() {
                let d = tables.node_rows[x + 1] - tables.node_rows[x];
                if coin.len() != d * d {
                    return Err(GpuError::Backend(format!("node {x}: a {d} × {d} coin expected")));
                }
                coins.extend_from_slice(coin);
            }
        }
        let exchange: Vec<c64> = batch.iter().flat_map(|t| t.exchange).collect();
        let halves: Vec<[u32; 2]> = tables
            .halves
            .iter()
            .map(|&(a, b)| Ok([to_u32(a, "a row index")?, to_u32(b, "a row index")?]))
            .collect::<Result<_, GpuError>>()?;
        let mut walk = MetalWalk {
            gpu: self,
            precision,
            params: WalkParams {
                bits: bits as u32,
                nodes: nodes as u32,
                edges: edges as u32,
                batch: batch.len() as u32,
                coin_len: to_u32(coin_len, "the coin table")?,
            },
            entries,
            psi: self.buffer(bytes)?,
            node_rows: self.buffer_from(&node_rows)?,
            coin_at: self.buffer_from(&coin_at)?,
            coins: self.buffer_from(&encode(&coins, precision))?,
            halves: self.buffer_from(&halves)?,
            exchange: self.buffer_from(&encode(&exchange, precision))?,
            acc: self.buffer(batch.len() * rows * std::mem::size_of::<f32>())?,
            rows: batch.len() * rows,
        };
        walk.set_state(psi);
        walk.reset_accumulated();
        Ok(walk)
    }
}

impl MetalWalk<'_> {
    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// The state buffer as floats.
    fn floats(&self) -> &[f32] {
        // SAFETY: the buffer holds `entries · width` floats, written by completed commands.
        unsafe {
            std::slice::from_raw_parts(
                self.psi.contents().as_ptr() as *const f32,
                self.entries * width(self.precision),
            )
        }
    }

    /// Replaces the state.
    pub fn set_state(&mut self, psi: &[c64]) {
        assert_eq!(psi.len(), self.entries, "state of the wrong dimension");
        let w = width(self.precision);
        // SAFETY: the buffer holds `entries · w` floats and no command is running (`&mut self`).
        let data = unsafe {
            std::slice::from_raw_parts_mut(self.psi.contents().as_ptr() as *mut f32, self.entries * w)
        };
        let chunk = 1 << 16;
        data.par_chunks_mut(chunk * w).zip(psi.par_chunks(chunk)).for_each(|(out, z)| {
            out.copy_from_slice(&encode(z, self.precision));
        });
    }

    /// The state, decoded to f64.
    pub fn state(&self) -> Vec<c64> {
        let mut psi = vec![c64::new(0.0, 0.0); self.entries];
        self.read_into(&mut psi);
        psi
    }

    /// Decodes the state into `psi`.
    pub fn read_into(&self, psi: &mut [c64]) {
        assert_eq!(psi.len(), self.entries, "state of the wrong dimension");
        let (w, chunk) = (width(self.precision), 1 << 16);
        let precision = self.precision;
        psi.par_chunks_mut(chunk).zip(self.floats().par_chunks(chunk * w)).for_each(
            |(z, f)| match precision {
                Precision::Single => {
                    for (v, p) in z.iter_mut().zip(f.chunks_exact(2)) {
                        *v = c64::new(p[0] as f64, p[1] as f64);
                    }
                }
                Precision::DoubleSingle => {
                    for (v, p) in z.iter_mut().zip(f.chunks_exact(4)) {
                        *v = c64::new(p[0] as f64 + p[1] as f64, p[2] as f64 + p[3] as f64);
                    }
                }
            },
        );
    }

    /// `‖ψ‖²`, summed in f64 by the CPU.
    pub fn norm_sqr(&self) -> f64 {
        let w = width(self.precision);
        let precision = self.precision;
        self.floats()
            .par_chunks((1 << 16) * w)
            .map(|f| match precision {
                Precision::Single => f.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>(),
                Precision::DoubleSingle => f
                    .chunks_exact(2)
                    .map(|p| {
                        let x = p[0] as f64 + p[1] as f64;
                        x * x
                    })
                    .sum(),
            })
            .sum()
    }

    /// `Σ_s |ψ(h, s)|²` for each of the `rows` rows, summed in f64 by the CPU.
    pub fn row_probabilities(&self, rows: usize) -> Vec<f64> {
        assert!(rows > 0 && self.entries % rows == 0, "rows must divide the state");
        let w = width(self.precision);
        let precision = self.precision;
        self.floats()
            .par_chunks(self.entries / rows * w)
            .map(|f| match precision {
                Precision::Single => f.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>(),
                Precision::DoubleSingle => f
                    .chunks_exact(2)
                    .map(|p| {
                        let x = p[0] as f64 + p[1] as f64;
                        x * x
                    })
                    .sum(),
            })
            .collect()
    }

    /// `‖ψ_GPU − psi‖`, summed in f64 by the CPU.
    pub fn distance(&self, psi: &[c64]) -> f64 {
        assert_eq!(psi.len(), self.entries, "state of the wrong dimension");
        let (w, chunk) = (width(self.precision), 1 << 16);
        let precision = self.precision;
        self.floats()
            .par_chunks(chunk * w)
            .zip(psi.par_chunks(chunk))
            .map(|(f, z)| match precision {
                Precision::Single => f
                    .chunks_exact(2)
                    .zip(z)
                    .map(|(p, v)| (c64::new(p[0] as f64, p[1] as f64) - v).norm_sqr())
                    .sum::<f64>(),
                Precision::DoubleSingle => f
                    .chunks_exact(4)
                    .zip(z)
                    .map(|(p, v)| {
                        (c64::new(p[0] as f64 + p[1] as f64, p[2] as f64 + p[3] as f64) - v).norm_sqr()
                    })
                    .sum(),
            })
            .sum::<f64>()
            .sqrt()
    }

    /// `n` steps (f32 states only), adding after each step the probability of every half-edge
    /// to the accumulators ([`Self::accumulated`]): for time averages, without reading the
    /// state back.
    pub fn steps_accumulate(&mut self, n: usize) -> Result<(), GpuError> {
        if self.precision != Precision::Single {
            return Err(GpuError::Backend("the accumulation takes f32 states".into()));
        }
        let kernels = self.gpu.kernels(self.precision);
        let p = self.params;
        let coin_threads = p.batch as usize * ((p.nodes as usize) << p.bits);
        let edge_threads = p.batch as usize * ((p.edges as usize) << (p.bits - 1));
        let entries = self.entries;
        self.gpu.run("running the walk kernels", 3 * n, |encoder, i| match i % 3 {
            0 => {
                encoder.setComputePipelineState(&kernels.walk_coin);
                // SAFETY: `walk_coin(psi, node_rows, coin_at, coins, params)`.
                unsafe {
                    bind(encoder, &[&self.psi, &self.node_rows, &self.coin_at, &self.coins], &[as_bytes(&p)])
                };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups(coin_threads), GROUP_SIZE);
            }
            1 => {
                encoder.setComputePipelineState(&kernels.walk_edge);
                // SAFETY: `walk_edge(psi, halves, exchange, params)`.
                unsafe { bind(encoder, &[&self.psi, &self.halves, &self.exchange], &[as_bytes(&p)]) };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups(edge_threads), GROUP_SIZE);
            }
            _ => {
                encoder.setComputePipelineState(&self.gpu.walk_accumulate);
                // SAFETY: `walk_accumulate_f32(psi, acc, params)`.
                unsafe { bind(encoder, &[&self.psi, &self.acc], &[as_bytes(&p)]) };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups(entries), GROUP_SIZE);
            }
        })
    }

    /// The accumulated probability of every half-edge.
    pub fn accumulated(&self) -> Vec<f64> {
        // SAFETY: the buffer holds `rows` f32 written by completed commands.
        let acc =
            unsafe { std::slice::from_raw_parts(self.acc.contents().as_ptr() as *const f32, self.rows) };
        acc.iter().map(|&v| v as f64).collect()
    }

    /// Sets the accumulators to zero.
    pub fn reset_accumulated(&mut self) {
        // SAFETY: the buffer holds `rows` f32 and no command is running (`&mut self`).
        let acc =
            unsafe { std::slice::from_raw_parts_mut(self.acc.contents().as_ptr() as *mut f32, self.rows) };
        acc.fill(0.0);
    }

    /// `n` steps.
    pub fn steps(&mut self, n: usize) -> Result<(), GpuError> {
        let kernels = self.gpu.kernels(self.precision);
        let p = self.params;
        let coin_threads = p.batch as usize * ((p.nodes as usize) << p.bits);
        let edge_threads = p.batch as usize * ((p.edges as usize) << (p.bits - 1));
        self.gpu.run("running the walk kernels", 2 * n, |encoder, i| {
            if i % 2 == 0 {
                encoder.setComputePipelineState(&kernels.walk_coin);
                // SAFETY: `walk_coin(psi, node_rows, coin_at, coins, params)`; the buffers live
                // until `run` returns.
                unsafe {
                    bind(encoder, &[&self.psi, &self.node_rows, &self.coin_at, &self.coins], &[as_bytes(&p)])
                };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups(coin_threads), GROUP_SIZE);
            } else {
                encoder.setComputePipelineState(&kernels.walk_edge);
                // SAFETY: `walk_edge(psi, halves, exchange, params)`.
                unsafe { bind(encoder, &[&self.psi, &self.halves, &self.exchange], &[as_bytes(&p)]) };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups(edge_threads), GROUP_SIZE);
            }
        })
    }
}
