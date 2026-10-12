// Arithmetic shared by the kernels: complex numbers over `float` or `df64` (double-single: an
// unevaluated sum hi + lo of two floats, ~48-bit significand). The double-single arithmetic
// relies on exact IEEE rounding and fma: the library is compiled without fast math.

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

inline df64 widen(float x) { return df64{x, 0.0f}; }
inline df64 widen(df64 x) { return x; }

#define INSTANTIATE(name, func, T) \
    template [[host_name(name)]] [[kernel]] decltype(func<T>) func<T>;
