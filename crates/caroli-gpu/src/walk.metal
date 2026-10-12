
// Quantum walks on a graph of spins (caroli-walk), in batches. Walk b of a batch holds the state
// ψ_b[h · 2^bits + s] (half-edge h, spin configuration s) from entry b · rows · 2^bits on, its
// own coins and exchange constants, and the graph of the others. A step is a coin dispatch then
// an edge dispatch over the whole batch: many small walks fill the GPU together.

constant uint MAX_DEGREE = 8;

struct WalkParams {
    uint bits;      // spins: rows of 2^bits entries
    uint nodes;
    uint edges;     // a walk has 2 · edges rows (half-edges)
    uint batch;     // walks
    uint coin_len;  // coin entries of one walk
};

/// First entry of walk `b`.
inline ulong walk_base(constant WalkParams& p, uint b) {
    return ((ulong)b * (2u * p.edges)) << p.bits;
}

/// Coin: one thread per (walk, node, s) applies the node's d × d coin (row-major at
/// `coins[b · coin_len + coin_at[x]]`) to its rows `node_rows[x]..node_rows[x + 1]`.
template <typename T>
kernel void walk_coin(device cplx<T>* psi [[buffer(0)]],
                      device const uint* node_rows [[buffer(1)]],
                      device const uint* coin_at [[buffer(2)]],
                      device const cplx<T>* coins [[buffer(3)]],
                      constant WalkParams& p [[buffer(4)]],
                      uint gid [[thread_position_in_grid]]) {
    uint per_walk = p.nodes << p.bits;
    uint b = gid / per_walk;
    if (b >= p.batch) {
        return;
    }
    uint local = gid - b * per_walk;
    uint x = local >> p.bits;
    ulong s = walk_base(p, b) + (local & ((1u << p.bits) - 1u));
    uint r0 = node_rows[x];
    uint d = node_rows[x + 1] - r0;
    device const cplx<T>* c = coins + b * p.coin_len + coin_at[x];
    cplx<T> v[MAX_DEGREE];
    for (uint k = 0; k < d; ++k) {
        v[k] = psi[((ulong)(r0 + k) << p.bits) + s];
    }
    for (uint i = 0; i < d; ++i) {
        cplx<T> acc = czero<T>();
        for (uint k = 0; k < d; ++k) {
            acc = acc + c[i * d + k] * v[k];
        }
        psi[((ulong)(r0 + i) << p.bits) + s] = acc;
    }
}

/// Motion and interaction `V(J) M`: one thread per (walk, edge e, configuration s with spin e
/// up) updates the four amplitudes of the edge's half-edges at s and at s with spin e down:
/// `a0' = p b0`, `a1' = α b1 + β a0`, `b0' = β b1 + α a0`, `b1' = p a1` (`exchange[3 b..]` =
/// p, α, β).
template <typename T>
kernel void walk_edge(device cplx<T>* psi [[buffer(0)]],
                      device const uint2* halves [[buffer(1)]],
                      device const cplx<T>* exchange [[buffer(2)]],
                      constant WalkParams& p [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    uint half_bits = p.bits - 1u;
    uint per_walk = p.edges << half_bits;
    uint b = gid / per_walk;
    if (b >= p.batch) {
        return;
    }
    uint local = gid - b * per_walk;
    uint e = local >> half_bits;
    uint c = local & ((1u << half_bits) - 1u);
    uint bit = 1u << e;
    ulong s = walk_base(p, b) + (((c >> e) << (e + 1u)) | (c & (bit - 1u)));
    uint2 h = halves[e];
    ulong ia = ((ulong)h.x << p.bits) + s;
    ulong ib = ((ulong)h.y << p.bits) + s;
    cplx<T> a0 = psi[ia];
    cplx<T> a1 = psi[ia + bit];
    cplx<T> b0 = psi[ib];
    cplx<T> b1 = psi[ib + bit];
    cplx<T> pp = exchange[3 * b];
    cplx<T> alpha = exchange[3 * b + 1];
    cplx<T> beta = exchange[3 * b + 2];
    psi[ia] = pp * b0;
    psi[ia + bit] = alpha * b1 + beta * a0;
    psi[ib] = beta * b1 + alpha * a0;
    psi[ib + bit] = pp * a1;
}

INSTANTIATE("walk_coin_f32", walk_coin, float)
INSTANTIATE("walk_coin_df64", walk_coin, df64)
INSTANTIATE("walk_edge_f32", walk_edge, float)
INSTANTIATE("walk_edge_df64", walk_edge, df64)

/// Adds `Σ_s |ψ(h, s)|²` of every row of every walk to `acc[b · rows + h]` (f32 states): one
/// thread per entry, a reduction per threadgroup, then one atomic addition per group when the
/// rows hold at least a threadgroup, one per entry otherwise.
kernel void walk_accumulate_f32(device const cplx<float>* psi [[buffer(0)]],
                                device atomic_float* acc [[buffer(1)]],
                                constant WalkParams& p [[buffer(2)]],
                                uint gid [[thread_position_in_grid]],
                                uint lid [[thread_position_in_threadgroup]]) {
    threadgroup float shared[GROUP];
    uint total = (p.batch * 2u * p.edges) << p.bits;
    float v = 0.0f;
    if (gid < total) {
        cplx<float> z = psi[gid];
        v = z.re * z.re + z.im * z.im;
    }
    if ((1u << p.bits) < GROUP) {
        if (gid < total) {
            atomic_fetch_add_explicit(&acc[gid >> p.bits], v, memory_order_relaxed);
        }
        return;
    }
    shared[lid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = GROUP / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] += shared[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0 && gid < total) {
        atomic_fetch_add_explicit(&acc[gid >> p.bits], shared[0], memory_order_relaxed);
    }
}
