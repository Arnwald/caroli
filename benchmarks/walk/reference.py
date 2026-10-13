"""Python versions of the quantum walk on a graph of spins (K. Sellapillay and A. D. Verga,
Phys. Rev. E 103, 032123 (2021)), on the ring of n nodes, written for the benchmark of
caroli-walk (they are not the authors' code):

- sparse: the one-step operator U = V(J) M R(theta) built as a scipy.sparse CSR matrix, then
  psi <- U psi at every step;
- numpy:  the same step as vectorised NumPy operations on psi[h, s], without a matrix.

The state is psi[h * 2^n + s] as in caroli-walk: half-edge h = 2x + c (color 0 points to x + 1,
color 1 to x - 1), spin configuration s (bit e is the spin of edge e = (e, e + 1), 0 for up).
Initial state: walker on color 0 of node 0, every spin along +x.

Usage: python reference.py METHOD N STEPS [THETA J]
       python reference.py phase N STEPS THETA J SPINS   (SPINS: x or z)
Prints one JSON line: seconds per step, the build time (sparse), and p(x) after the steps; or,
for `phase`, the time of a whole run of the phase-space scan (Fig. 21 of the paper): the
time-averaged p(x) and its Kolmogorov-Smirnov distance to the uniform distribution.
"""

import json
import sys
import time

import numpy as np
import scipy.sparse as sp


def exchange(J):
    q = np.exp(-1j * J / 4)
    return np.exp(1j * J / 4), q * np.cos(J / 2), q * 1j * np.sin(J / 2)


def initial_state(n):
    L = 1 << n
    psi = np.zeros(2 * n * L, dtype=np.complex128)
    psi[0:L] = 2.0 ** (-n / 2)  # half-edge 0 (node 0, color 0), spins |+>^n
    return psi


def build_u(n, theta, J):
    L = 1 << n
    dim = 2 * n * L
    c, s = np.cos(theta), np.sin(theta)
    coin = sp.kron(sp.identity(n, format="csr"), sp.kron(sp.csr_matrix([[c, -s], [s, c]]), sp.identity(L, format="csr")), format="csr")
    p, alpha, beta = exchange(J)
    configs = np.arange(L, dtype=np.int64)
    rows, cols, vals = [], [], []
    for e in range(n):
        ha, hb = 2 * e, 2 * ((e + 1) % n) + 1
        bit = 1 << e
        up = configs[(configs & bit) == 0]
        a0, a1, b0, b1 = ha * L + up, ha * L + (up | bit), hb * L + up, hb * L + (up | bit)
        # a0' = p b0, a1' = alpha b1 + beta a0, b0' = beta b1 + alpha a0, b1' = p a1
        for r, col, v in ((a0, b0, p), (a1, b1, alpha), (a1, a0, beta), (b0, b1, beta), (b0, a0, alpha), (b1, a1, p)):
            rows.append(r)
            cols.append(col)
            vals.append(np.full(r.size, v, dtype=np.complex128))
    vm = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(dim, dim))
    return (vm @ coin).tocsr()


def numpy_step(P, n, c, s, p, alpha, beta):
    """One step on P = psi.reshape(2n, 2^n), in place."""
    a = P[0::2].copy()
    b = P[1::2].copy()
    P[0::2] = c * a - s * b
    P[1::2] = s * a + c * b
    for e in range(n):
        ha, hb = 2 * e, 2 * ((e + 1) % n) + 1
        bit = 1 << e
        A = P[ha].reshape(-1, 2, bit)
        B = P[hb].reshape(-1, 2, bit)
        a0, a1, b0, b1 = A[:, 0].copy(), A[:, 1].copy(), B[:, 0].copy(), B[:, 1].copy()
        A[:, 0] = p * b0
        A[:, 1] = alpha * b1 + beta * a0
        B[:, 0] = beta * b1 + alpha * a0
        B[:, 1] = p * a1


def initial_at(n, x0, spins):
    """Walker on color 0 of node x0; spins all up ('z') or all along +x ('x')."""
    L = 1 << n
    psi = np.zeros(2 * n * L, dtype=np.complex128)
    row = 2 * x0 * L
    if spins == "z":
        psi[row] = 1.0
    else:
        psi[row:row + L] = 2.0 ** (-n / 2)
    return psi


def phase_point(n, steps, theta, J, spins):
    """Time-averaged position distribution over `steps` steps and its Kolmogorov-Smirnov
    distance to the uniform one (Sellapillay & Verga, Eq. 31), the walker starting at the
    middle node."""
    L = 1 << n
    t0 = time.perf_counter()
    u = build_u(n, theta, J)
    psi = initial_at(n, (n - 1) // 2, spins)
    acc = np.zeros(n)
    for _ in range(steps):
        psi = u @ psi
        prob = (np.abs(psi.reshape(2 * n, L)) ** 2).sum(axis=1)
        acc += prob[0::2] + prob[1::2]
    elapsed = time.perf_counter() - t0
    p = acc / acc.sum()
    ks = np.max(np.abs(np.cumsum(p) - np.arange(1, n + 1) / n))
    return elapsed, ks, p


def main():
    if sys.argv[1] == "phase":
        n, steps, theta, J, spins = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
        elapsed, ks, p = phase_point(n, steps, theta, J, spins)
        print(json.dumps({"method": "phase", "n": n, "steps": steps, "theta": theta, "J": J, "spins": spins,
                          "seconds": elapsed, "ks": ks, "p": p.tolist()}))
        return
    method, n, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    theta = float(sys.argv[4]) if len(sys.argv) > 4 else np.pi / 8
    J = float(sys.argv[5]) if len(sys.argv) > 5 else 0.2
    L = 1 << n
    psi = initial_state(n)
    build = 0.0
    if method == "sparse":
        t0 = time.perf_counter()
        u = build_u(n, theta, J)
        build = time.perf_counter() - t0
        psi = u @ psi  # warm-up step
        t0 = time.perf_counter()
        for _ in range(steps):
            psi = u @ psi
        elapsed = time.perf_counter() - t0
    elif method == "numpy":
        P = psi.reshape(2 * n, L)
        c, s = np.cos(theta), np.sin(theta)
        p, alpha, beta = exchange(J)
        numpy_step(P, n, c, s, p, alpha, beta)  # warm-up step
        t0 = time.perf_counter()
        for _ in range(steps):
            numpy_step(P, n, c, s, p, alpha, beta)
        elapsed = time.perf_counter() - t0
        psi = P.reshape(-1)
    else:
        raise SystemExit(f"unknown method {method}")
    prob = (np.abs(psi.reshape(2 * n, L)) ** 2).sum(axis=1)
    px = (prob[0::2] + prob[1::2]).tolist()
    print(json.dumps({"method": method, "n": n, "steps": steps, "s_per_step": elapsed / steps, "build_s": build,
                      "numpy": np.__version__, "scipy": sp.__name__ and __import__("scipy").__version__, "p": px}))


if __name__ == "__main__":
    main()
