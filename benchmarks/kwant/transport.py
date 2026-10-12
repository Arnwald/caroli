"""Kwant runner for the shared transport problems of `benchmarks/transport.toml`.

The system is rebuilt from the files written by `caroli-bench prepare`: the lattice (vectors
and sublattice positions), the on-site and hopping matrices of the model, and the on-site
disorder. Every code therefore solves the same Hamiltonian. caroli's hopping matrix is
<(R, from)|H|(R + offset, to)>, which is the value of the Kwant hopping kind
(-offset, from, to).

Kwant's fastest path is used: the MUMPS solver (with its default METIS ordering; the other
orderings and `nrhs` gave no gain), only the lead 0 -> lead 1 block of the scattering matrix
(the whole matrix for a cross, which needs every T_pq), and no Hermiticity check. With the
`kwant-accelerate` environment, MUMPS and numpy run on Apple's Accelerate (5-10 % faster than
OpenBLAS). The script prints one JSON line.

    python transport.py DATA_DIR PROBLEM WIDTH
"""

import json
import os
import sys
import time

import numpy as np
import kwant
import kwant.solvers.default


def matrix(m):
    a = np.array(m, dtype=float)
    return a[..., 0] + 1j * a[..., 1]


def build(data, name, width):
    """The finalized system and whether it is a cross."""
    with open(f"{data}/{name}/problem.json") as f:
        spec = json.load(f)
    n = spec["orbitals"]
    length = spec["length"] or width
    subs = spec["sublattices"]
    onsites = [matrix(s["onsite"]) for s in subs]
    hoppings = [(h["from"], h["to"], tuple(-k for k in h["offset"]), matrix(h["matrix"])) for h in spec["hoppings"]]
    cross = spec.get("geometry") == "cross"
    arm = width // 4
    # Bounding box of the sample and its corner cell.
    (nx, ny), (x0, y0) = ((width + 2 * arm,) * 2, (-arm, -arm)) if cross else ((length, width), (0, 0))
    pot = None
    if spec["anderson"]:
        pot = np.fromfile(f"{data}/{name}/disorder_W{width}.f64", dtype="<f8").reshape(nx, ny, len(subs))
    eye = np.eye(n)

    lat = kwant.lattice.general(spec["vectors"], [s["offset"] for s in subs], norbs=n)
    fams = lat.sublattices
    syst = kwant.Builder()
    for x in range(x0, x0 + nx):
        for y in range(y0, y0 + ny):
            if cross and not (0 <= x < width or 0 <= y < width):
                continue
            for k, fam in enumerate(fams):
                v = onsites[k] if pot is None else onsites[k] + pot[x - x0, y - y0, k] * eye
                syst[fam(x, y)] = v
    for a, b, delta, m in hoppings:
        syst[kwant.builder.HoppingKind(delta, fams[a], fams[b])] = m

    def lead(direction, cells):
        lead = kwant.Builder(kwant.TranslationalSymmetry(lat.vec(direction)))
        for x, y in cells:
            for k, fam in enumerate(fams):
                lead[fam(x, y)] = onsites[k]
        for a, b, delta, m in hoppings:
            lead[kwant.builder.HoppingKind(delta, fams[a], fams[b])] = m
        return lead

    along_x = lead((-1, 0), [(0, y) for y in range(width)])
    syst.attach_lead(along_x)
    syst.attach_lead(along_x.reversed())
    if cross:
        # Leads in the order of caroli's interfaces: left, right, bottom, top.
        along_y = lead((0, -1), [(x, 0) for x in range(width)])
        syst.attach_lead(along_y)
        syst.attach_lead(along_y.reversed())
    return syst.finalized(), cross


def solve(fsys, cross, e):
    """The transmission at `e`, or for a cross every T_pq (p ≠ q, p-major)."""
    if cross:
        # Every T_pq: the full scattering matrix.
        s = kwant.smatrix(fsys, e, check_hermiticity=False)
        return [float(s.transmission(p, q)) for p in range(4) for q in range(4) if p != q]
    s = kwant.smatrix(fsys, e, in_leads=[0], out_leads=[1], check_hermiticity=False)
    return [float(s.transmission(1, 0))]


def main():
    data, name, width = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with open(f"{data}/{name}/problem.json") as f:
        spec = json.load(f)
    t0 = time.perf_counter()
    fsys, cross = build(data, name, width)
    build_s = time.perf_counter() - t0

    times, ts = [], []
    for e in spec["energies"]:
        t = time.perf_counter()
        ts += solve(fsys, cross, e)
        times.append(time.perf_counter() - t)

    print(json.dumps({
        "code": "kwant", "version": kwant.__version__,
        "solver": kwant.solvers.default.smodule.__name__,
        "env": os.path.basename(sys.prefix),
        "problem": name, "width": width, "length": spec["length"] or width, "orbitals": spec["orbitals"],
        "threads": 1, "build_s": build_s, "times_s": times, "energies": spec["energies"], "transmissions": ts,
    }))


if __name__ == "__main__":
    main()
