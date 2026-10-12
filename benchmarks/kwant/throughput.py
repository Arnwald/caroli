"""Kwant on all the cores: COUNT energies over PROCS worker processes.

Kwant (MUMPS) runs one energy on one core; a pool of processes is how Kwant users run energy
sweeps in parallel. Each worker builds the system once, from the same files as
`transport.py`, then takes energies from the queue. The energies are the problem's, shifted by
1e-3 per round to make COUNT different ones (as `caroli-bench solve ... --count`). The wall time
includes the builds, which run in parallel.

    python throughput.py DATA_DIR PROBLEM WIDTH PROCS COUNT
"""

import json
import multiprocessing as mp
import os
import sys
import time

import kwant

import transport

_system = None


def _init(data, name, width):
    global _system
    _system = transport.build(data, name, width)


def _solve(e):
    fsys, cross = _system
    return transport.solve(fsys, cross, e)


def main():
    data, name, width, procs, count = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    with open(f"{data}/{name}/problem.json") as f:
        spec = json.load(f)
    base = spec["energies"]
    energies = [base[k % len(base)] + 1e-3 * (k // len(base)) for k in range(count)]
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(procs, initializer=_init, initargs=(data, name, width)) as pool:
        ts = pool.map(_solve, energies, chunksize=1)
    wall = time.perf_counter() - t0
    print(json.dumps({
        "code": "kwant", "version": kwant.__version__, "solver": "kwant.solvers.mumps, process pool",
        "problem": name, "width": width, "length": spec["length"] or width, "orbitals": spec["orbitals"],
        "env": os.path.basename(sys.prefix), "threads": procs, "mode": "throughput", "total_s": wall, "energies": energies, "transmissions": [t for row in ts for t in row],
    }))


if __name__ == "__main__":
    main()
