
// Quantum walk on a graph of spins (caroli-walk): the state is ψ[h · 2^bits + s], half-edge h,
// spin configuration s. One step is a coin dispatch then an edge dispatch.

constant uint MAX_DEGREE = 8;

struct WalkParams {
    uint bits;   // spins: rows of 2^bits entries
    uint nodes;
    uint edges;
};

/// Coin: one thread per (node, s) applies the node's d × d coin (row-major at
/// `coins[coin_at[x]]`) to its rows `node_rows[x]..node_rows[x + 1]`.
template <typename T>
kernel void walk_coin(device cplx<T>* psi [[buffer(0)]],
                      device const uint* node_rows [[buffer(1)]],
                      device const uint* coin_at [[buffer(2)]],
                      device const cplx<T>* coins [[buffer(3)]],
                      constant WalkParams& p [[buffer(4)]],
                      uint gid [[thread_position_in_grid]]) {
    uint x = gid >> p.bits;
    if (x >= p.nodes) {
        return;
    }
    ulong s = gid & ((1u << p.bits) - 1u);
    uint r0 = node_rows[x];
    uint d = node_rows[x + 1] - r0;
    device const cplx<T>* c = coins + coin_at[x];
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

/// Motion and interaction `V(J) M`: one thread per (edge e, configuration s with spin e up)
/// updates the four amplitudes of the edge's half-edges at s and at s with spin e down:
/// `a0' = p b0`, `a1' = α b1 + β a0`, `b0' = β b1 + α a0`, `b1' = p a1` (`exchange` = p, α, β).
template <typename T>
kernel void walk_edge(device cplx<T>* psi [[buffer(0)]],
                      device const uint2* halves [[buffer(1)]],
                      device const cplx<T>* exchange [[buffer(2)]],
                      constant WalkParams& p [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    uint e = gid >> (p.bits - 1u);
    if (e >= p.edges) {
        return;
    }
    uint c = gid & ((1u << (p.bits - 1u)) - 1u);
    uint bit = 1u << e;
    ulong s = ((c >> e) << (e + 1u)) | (c & (bit - 1u));
    uint2 h = halves[e];
    ulong ia = ((ulong)h.x << p.bits) + s;
    ulong ib = ((ulong)h.y << p.bits) + s;
    cplx<T> a0 = psi[ia];
    cplx<T> a1 = psi[ia + bit];
    cplx<T> b0 = psi[ib];
    cplx<T> b1 = psi[ib + bit];
    cplx<T> pp = exchange[0];
    cplx<T> alpha = exchange[1];
    cplx<T> beta = exchange[2];
    psi[ia] = pp * b0;
    psi[ia + bit] = alpha * b1 + beta * a0;
    psi[ib] = beta * b1 + alpha * a0;
    psi[ib + bit] = pp * a1;
}

INSTANTIATE("walk_coin_f32", walk_coin, float)
INSTANTIATE("walk_coin_df64", walk_coin, df64)
INSTANTIATE("walk_edge_f32", walk_edge, float)
INSTANTIATE("walk_edge_df64", walk_edge, df64)
