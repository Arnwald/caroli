// Chebyshev steps with a sparse (CSR) Hamiltonian in double precision (CUDA, compiled at run
// time by NVRTC, no headers needed). Same algorithms and layout as `chebyshev.metal`: a block of
// `vectors` vectors interleaved (`x[row·vectors + v]`), one thread per entry, 256 threads per
// block.

#define GROUP 256

struct alignas(16) cplx {
    double re;
    double im;
};

__device__ inline cplx cadd(cplx a, cplx b) { return cplx{a.re + b.re, a.im + b.im}; }

__device__ inline cplx cmul(cplx a, cplx b) {
    return cplx{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

// conj(a) · b
__device__ inline cplx conj_mul(cplx a, cplx b) {
    return cplx{a.re * b.re + a.im * b.im, a.re * b.im - a.im * b.re};
}

__device__ inline cplx chebyshev_entry(const unsigned* row_ptr, const unsigned* cols,
                                       const cplx* vals, const cplx* prev, const cplx* cur,
                                       unsigned vectors, unsigned first, unsigned gid) {
    unsigned row = gid / vectors;
    unsigned v = gid - row * vectors;
    cplx acc = cplx{0.0, 0.0};
    for (unsigned k = row_ptr[row]; k < row_ptr[row + 1]; ++k) {
        acc = cadd(acc, cmul(vals[k], cur[cols[k] * vectors + v]));
    }
    if (first) {
        return acc;
    }
    cplx q = prev[gid];
    return cplx{2.0 * acc.re - q.re, 2.0 * acc.im - q.im};
}

// KPM step: next = 2 H̃ cur − prev, and this block's shares of ⟨cur|cur⟩ and ⟨cur|next⟩ in
// partials[2 (slot · gridDim.x + blockIdx.x) + {0, 1}].
extern "C" __global__ void kpm_step(const unsigned* row_ptr, const unsigned* cols,
                                    const cplx* vals, const cplx* prev, const cplx* cur,
                                    cplx* next, cplx* partials, unsigned vectors, unsigned total,
                                    unsigned first, unsigned slot) {
    __shared__ cplx s_nn[GROUP];
    __shared__ cplx s_np[GROUP];
    unsigned lid = threadIdx.x;
    unsigned gid = blockIdx.x * GROUP + lid;
    cplx nn = cplx{0.0, 0.0};
    cplx np = cplx{0.0, 0.0};
    if (gid < total) {
        cplx t = chebyshev_entry(row_ptr, cols, vals, prev, cur, vectors, first, gid);
        next[gid] = t;
        cplx c = cur[gid];
        nn = conj_mul(c, c);
        np = conj_mul(c, t);
    }
    s_nn[lid] = nn;
    s_np[lid] = np;
    __syncthreads();
    for (unsigned s = GROUP / 2; s > 0; s >>= 1) {
        if (lid < s) {
            s_nn[lid] = cadd(s_nn[lid], s_nn[lid + s]);
            s_np[lid] = cadd(s_np[lid], s_np[lid + s]);
        }
        __syncthreads();
    }
    if (lid == 0) {
        partials[2 * (slot * gridDim.x + blockIdx.x)] = s_nn[0];
        partials[2 * (slot * gridDim.x + blockIdx.x) + 1] = s_np[0];
    }
}

// Propagation step: next = 2 H̃ cur − prev and psi += coefficient · next.
extern "C" __global__ void propagate_step(const unsigned* row_ptr, const unsigned* cols,
                                          const cplx* vals, const cplx* prev, const cplx* cur,
                                          cplx* next, cplx* psi, double re, double im,
                                          unsigned vectors, unsigned total, unsigned first) {
    unsigned gid = blockIdx.x * GROUP + threadIdx.x;
    if (gid >= total) {
        return;
    }
    cplx t = chebyshev_entry(row_ptr, cols, vals, prev, cur, vectors, first, gid);
    next[gid] = t;
    psi[gid] = cadd(psi[gid], cmul(cplx{re, im}, t));
}

// Start of a propagation step: psi = coefficient · cur.
extern "C" __global__ void propagate_init(const cplx* cur, cplx* psi, double re, double im,
                                          unsigned total) {
    unsigned gid = blockIdx.x * GROUP + threadIdx.x;
    if (gid < total) {
        psi[gid] = cmul(cplx{re, im}, cur[gid]);
    }
}
