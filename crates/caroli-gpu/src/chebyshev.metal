// Chebyshev steps with a sparse (CSR) Hamiltonian, for the kernel polynomial method and for
// time propagation. Templated over the real type: `float`, or `df64` (double-single: an
// unevaluated sum hi + lo of two floats, ~48-bit significand). The double-single arithmetic
// relies on exact IEEE rounding and fma: the library is compiled without fast math.
//
// A block of `vectors` vectors is stored interleaved (`x[row·vectors + v]`); a grid of
// `n·vectors` threads (padded to whole threadgroups) processes one entry each. The threads of a
// row are neighbours, so they read the row of the matrix together.

#include <metal_stdlib>
using namespace metal;

constant uint GROUP = 256;

struct df64 {
    float hi;
    float lo;
};

// Error-free transforms.
inline df64 two_sum(float a, float b) {
    float s = a + b;
    float v = s - a;
    float e = (a - (s - v)) + (b - v);
    return df64{s, e};
}

inline df64 quick_two_sum(float a, float b) {
    float s = a + b;
    float e = b - (s - a);
    return df64{s, e};
}

inline df64 operator+(df64 a, df64 b) {
    df64 s = two_sum(a.hi, b.hi);
    df64 t = two_sum(a.lo, b.lo);
    s.lo += t.hi;
    s = quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    return quick_two_sum(s.hi, s.lo);
}

inline df64 operator-(df64 a) { return df64{-a.hi, -a.lo}; }
inline df64 operator-(df64 a, df64 b) { return a + (-b); }

inline df64 operator*(df64 a, df64 b) {
    float p = a.hi * b.hi;
    float e = fma(a.hi, b.hi, -p);
    e = fma(a.hi, b.lo, e);
    e = fma(a.lo, b.hi, e);
    return quick_two_sum(p, e);
}

template <typename T> inline T zero();
template <> inline float zero<float>() { return 0.0f; }
template <> inline df64 zero<df64>() { return df64{0.0f, 0.0f}; }

template <typename T> struct cplx {
    T re;
    T im;
};

template <typename T> inline cplx<T> czero() { return cplx<T>{zero<T>(), zero<T>()}; }

template <typename T> inline cplx<T> operator+(cplx<T> a, cplx<T> b) {
    return cplx<T>{a.re + b.re, a.im + b.im};
}

template <typename T> inline cplx<T> operator-(cplx<T> a, cplx<T> b) {
    return cplx<T>{a.re - b.re, a.im - b.im};
}

template <typename T> inline cplx<T> operator*(cplx<T> a, cplx<T> b) {
    return cplx<T>{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

/// `conj(a) · b`.
template <typename T> inline cplx<T> conj_mul(cplx<T> a, cplx<T> b) {
    return cplx<T>{a.re * b.re + a.im * b.im, a.re * b.im - a.im * b.re};
}

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

/// Sums the `p.n` pairs of `partials` (in a fixed order) into `sums[2 slot + {0, 1}]`; one
/// threadgroup.
template <typename T>
kernel void reduce_partials(device const cplx<T>* partials [[buffer(0)]],
                            device cplx<T>* sums [[buffer(1)]],
                            constant Params& p [[buffer(2)]],
                            uint lid [[thread_position_in_threadgroup]]) {
    threadgroup cplx<T> shared_nn[GROUP];
    threadgroup cplx<T> shared_np[GROUP];
    cplx<T> nn = czero<T>();
    cplx<T> np = czero<T>();
    for (uint g = lid; g < p.n; g += GROUP) {
        nn = nn + partials[2 * g];
        np = np + partials[2 * g + 1];
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

#define INSTANTIATE(name, func, T) \
    template [[host_name(name)]] [[kernel]] decltype(func<T>) func<T>;

INSTANTIATE("kpm_step_f32", kpm_step, float)
INSTANTIATE("kpm_step_df64", kpm_step, df64)
INSTANTIATE("reduce_partials_f32", reduce_partials, float)
INSTANTIATE("reduce_partials_df64", reduce_partials, df64)
INSTANTIATE("propagate_step_f32", propagate_step, float)
INSTANTIATE("propagate_step_df64", propagate_step, df64)
INSTANTIATE("propagate_init_f32", propagate_init, float)
INSTANTIATE("propagate_init_df64", propagate_init, df64)
