"""Kwant runner for the shared transport problems of `benchmarks/transport.toml`.

The system is rebuilt from the files written by `caroli-bench prepare`: the lattice (vectors
and sublattice positions), the on-site and hopping matrices of the model, and the on-site
disorder. Every code therefore solves the same Hamiltonian. caroli's hopping matrix is
<(R, from)|H|(R + offset, to)>, which is the value of the Kwant hopping kind
(-offset, from, to).

Kwant's fastest path is used: the MUMPS solver, only the lead 0 -> lead 1 block of the
scattering matrix, and no Hermiticity check. The script prints one JSON line.

    python transport.py DATA_DIR PROBLEM WIDTH
"""

import json
import sys
import time

import numpy as np
import kwant
import kwant.solvers.default


def matrix(m):
    a = np.array(m, dtype=float)
    return a[..., 0] + 1j * a[..., 1]


def main():
    data, name, width = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with open(f"{data}/{name}/problem.json") as f:
        spec = json.load(f)
    n = spec["orbitals"]
    length = spec["length"] or width
    subs = spec["sublattices"]
    onsites = [matrix(s["onsite"]) for s in subs]
    hoppings = [(h["from"], h["to"], tuple(-k for k in h["offset"]), matrix(h["matrix"])) for h in spec["hoppings"]]
    pot = None
    if spec["anderson"]:
        pot = np.fromfile(f"{data}/{name}/disorder_W{width}.f64", dtype="<f8").reshape(length, width, len(subs))
    eye = np.eye(n)

    t0 = time.perf_counter()
    lat = kwant.lattice.general(spec["vectors"], [s["offset"] for s in subs], norbs=n)
    fams = lat.sublattices
    syst = kwant.Builder()
    for x in range(length):
        for y in range(width):
            for k, fam in enumerate(fams):
                syst[fam(x, y)] = onsites[k] if pot is None else onsites[k] + pot[x, y, k] * eye
    for a, b, delta, m in hoppings:
        syst[kwant.builder.HoppingKind(delta, fams[a], fams[b])] = m
    lead = kwant.Builder(kwant.TranslationalSymmetry(lat.vec((-1, 0))))
    for y in range(width):
        for k, fam in enumerate(fams):
            lead[fam(0, y)] = onsites[k]
    for a, b, delta, m in hoppings:
        lead[kwant.builder.HoppingKind(delta, fams[a], fams[b])] = m
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    fsys = syst.finalized()
    build = time.perf_counter() - t0

    times, ts = [], []
    for e in spec["energies"]:
        t = time.perf_counter()
        s = kwant.smatrix(fsys, e, in_leads=[0], out_leads=[1], check_hermiticity=False)
        ts.append(float(s.transmission(1, 0)))
        times.append(time.perf_counter() - t)

    print(json.dumps({
        "code": "kwant", "version": kwant.__version__,
        "solver": kwant.solvers.default.smodule.__name__,
        "problem": name, "width": width, "length": length, "orbitals": n, "threads": 1,
        "build_s": build, "times_s": times, "energies": spec["energies"], "transmissions": ts,
    }))


if __name__ == "__main__":
    main()
