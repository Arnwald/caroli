// Chebyshev steps with a sparse (CSR) Hamiltonian, for the kernel polynomial method and for
// time propagation, templated over the real type (`float` or `df64`, see common.metal).
//
// A block of `vectors` vectors is stored interleaved (`x[row·vectors + v]`); a grid of
// `n·vectors` threads (padded to whole threadgroups) processes one entry each. The threads of a
// row are neighbours, so they read the row of the matrix together.

struct Params {
    uint n;        // rows
    uint vectors;  // vectors in the block
    uint total;    // n · vectors
    uint first;    // 1: T_1 = H̃ T_0 (no previous term)
    uint slot;     // KPM: index of this step's partial sums
};

/// `2 H̃ cur − prev` (or `H̃ cur` on the first step) for the entry `gid`.
template <typename T>
inline cplx<T> chebyshev_entry(device const uint* row_ptr, device const uint* cols,
                               device const cplx<T>* vals, device const cplx<T>* prev,
                               device const cplx<T>* cur, constant Params& p, uint gid) {
    uint row = gid / p.vectors;
    uint v = gid - row * p.vectors;
    cplx<T> acc = czero<T>();
    for (uint k = row_ptr[row]; k < row_ptr[row + 1]; ++k) {
        acc = acc + vals[k] * cur[cols[k] * p.vectors + v];
    }
    return p.first ? acc : (acc + acc) - prev[gid];
}

/// KPM step: `next = 2 H̃ cur − prev`, and this threadgroup's shares of `⟨cur|cur⟩` and
/// `⟨cur|next⟩` in `partials[2 group + {0, 1}]` (the moments follow from
/// `μ_2n = 2⟨T_n|T_n⟩ − μ_0` and `μ_{2n+1} = 2⟨T_n|T_{n+1}⟩ − μ_1`); `reduce_partials` then
/// adds them up.
template <typename T>
kernel void kpm_step(device const uint* row_ptr [[buffer(0)]],
                     device const uint* cols [[buffer(1)]],
                     device const cplx<T>* vals [[buffer(2)]],
                     device const cplx<T>* prev [[buffer(3)]],
                     device const cplx<T>* cur [[buffer(4)]],
                     device cplx<T>* next [[buffer(5)]],
                     device cplx<T>* partials [[buffer(6)]],
                     constant Params& p [[buffer(7)]],
                     uint gid [[thread_position_in_grid]],
                     uint lid [[thread_position_in_threadgroup]],
                     uint group [[threadgroup_position_in_grid]]) {
    threadgroup cplx<T> shared_nn[GROUP];
    threadgroup cplx<T> shared_np[GROUP];
    cplx<T> nn = czero<T>();
    cplx<T> np = czero<T>();
    if (gid < p.total) {
        cplx<T> t = chebyshev_entry(row_ptr, cols, vals, prev, cur, p, gid);
        next[gid] = t;
        cplx<T> c = cur[gid];
        nn = conj_mul(c, c);
        np = conj_mul(c, t);
    }
    shared_nn[lid] = nn;
    shared_np[lid] = np;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = GROUP / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared_nn[lid] = shared_nn[lid] + shared_nn[lid + s];
            shared_np[lid] = shared_np[lid] + shared_np[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        partials[2 * group] = shared_nn[0];
        partials[2 * group + 1] = shared_np[0];
    }
}


/// Sums the `p.n` pairs of `partials` (in a fixed order) into `sums[2 slot + {0, 1}]`, in
/// double-single whatever the precision of the partial sums; one threadgroup.
template <typename T>
kernel void reduce_partials(device const cplx<T>* partials [[buffer(0)]],
                            device cplx<df64>* sums [[buffer(1)]],
                            constant Params& p [[buffer(2)]],
                            uint lid [[thread_position_in_threadgroup]]) {
    threadgroup cplx<df64> shared_nn[GROUP];
    threadgroup cplx<df64> shared_np[GROUP];
    cplx<df64> nn = czero<df64>();
    cplx<df64> np = czero<df64>();
    for (uint g = lid; g < p.n; g += GROUP) {
        cplx<T> a = partials[2 * g];
        cplx<T> b = partials[2 * g + 1];
        nn = nn + cplx<df64>{widen(a.re), widen(a.im)};
        np = np + cplx<df64>{widen(b.re), widen(b.im)};
    }
    shared_nn[lid] = nn;
    shared_np[lid] = np;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = GROUP / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared_nn[lid] = shared_nn[lid] + shared_nn[lid + s];
            shared_np[lid] = shared_np[lid] + shared_np[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        sums[2 * p.slot] = shared_nn[0];
        sums[2 * p.slot + 1] = shared_np[0];
    }
}

/// Propagation step: `next = 2 H̃ cur − prev` and `psi += coefficient · next`.
template <typename T>
kernel void propagate_step(device const uint* row_ptr [[buffer(0)]],
                           device const uint* cols [[buffer(1)]],
                           device const cplx<T>* vals [[buffer(2)]],
                           device const cplx<T>* prev [[buffer(3)]],
                           device const cplx<T>* cur [[buffer(4)]],
                           device cplx<T>* next [[buffer(5)]],
                           device cplx<T>* psi [[buffer(6)]],
                           constant cplx<T>& coefficient [[buffer(7)]],
                           constant Params& p [[buffer(8)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= p.total) {
        return;
    }
    cplx<T> t = chebyshev_entry(row_ptr, cols, vals, prev, cur, p, gid);
    next[gid] = t;
    psi[gid] = psi[gid] + coefficient * t;
}

/// Start of a propagation step: `psi = coefficient · cur`.
template <typename T>
kernel void propagate_init(device const cplx<T>* cur [[buffer(0)]],
                           device cplx<T>* psi [[buffer(1)]],
                           constant cplx<T>& coefficient [[buffer(2)]],
                           constant Params& p [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid < p.total) {
        psi[gid] = coefficient * cur[gid];
    }
}

INSTANTIATE("kpm_step_f32", kpm_step, float)
INSTANTIATE("kpm_step_df64", kpm_step, df64)
INSTANTIATE("reduce_partials_f32", reduce_partials, float)
INSTANTIATE("reduce_partials_df64", reduce_partials, df64)
INSTANTIATE("propagate_step_f32", propagate_step, float)
INSTANTIATE("propagate_step_df64", propagate_step, df64)
INSTANTIATE("propagate_init_f32", propagate_init, float)
INSTANTIATE("propagate_init_df64", propagate_init, df64)
