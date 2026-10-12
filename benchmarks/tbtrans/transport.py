"""TBtrans runner for the shared transport problems of `benchmarks/transport.toml`.

The device and electrode Hamiltonians are rebuilt with sisl from the files written by
`caroli-bench prepare` (the same Hamiltonian as every other code), written to NetCDF, and
TBtrans (SIESTA's transport code) computes the transmission at the problem's energies.

TBtrans's scattering region holds, besides the sample, one electrode cell at each end: the
first cell of each semi-infinite lead. Its electrode self-energies come from the Lopez Sancho
iteration at `E + iη` (`TBT.Elecs.Eta`, 1 meV by default); the device stays at real energy.
TBtrans needs a real Hamiltonian in its HS file, so only real models run here.

η = 1e-6 eV here: the transmissions then agree with the exact ones to about 1e-5, for 20 %
more time than the default η. The output is written in double precision. TBtrans runs
single-threaded (OMP_NUM_THREADS=1 and one BLAS thread). The conda-forge build with OpenBLAS is
used: the Accelerate variant was not faster, and the new-LAPACK one crashes.

Its time is the wall time of the tbtrans process; per-energy times are estimated from two runs
(all energies, and the first energy alone): `(T_all − T_one) / (n − 1)`.

    python transport.py DATA_DIR PROBLEM WIDTH [WORK_DIR] [ETA_EV]
"""

import glob
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import scipy.sparse as sp
import sisl


def matrix(m):
    a = np.array(m, dtype=float)
    return a[..., 0] + 1j * a[..., 1]


def build(spec, data, name, width):
    """Device and electrode Hamiltonians (sisl), the device including one electrode cell at each end."""
    n = spec["orbitals"]
    length = spec["length"] or width
    subs = spec["sublattices"]
    onsites = [matrix(s["onsite"]) for s in subs]
    hoppings = [(h["from"], h["to"], tuple(h["offset"]), matrix(h["matrix"])) for h in spec["hoppings"]]
    for m in onsites + [h[3] for h in hoppings]:
        if np.abs(m.imag).max() > 0:
            raise SystemExit("TBtrans needs a real Hamiltonian")
    pot = None
    if spec["anderson"]:
        pot = np.fromfile(f"{data}/{name}/disorder_W{width}.f64", dtype="<f8").reshape(length, width, len(subs))
    a = np.array(spec["vectors"], dtype=float)
    a1 = np.array([a[0][0], a[0][1], 0.0])
    a2 = np.array([a[1][0], a[1][1], 0.0])
    offsets = [np.array([s["offset"][0], s["offset"][1], 0.0]) for s in subs]
    nsub = len(subs)
    # Atoms ordered by column x (slowest), then y, then sublattice; columns -1 and `length` are
    # the electrode cells.
    xs = range(-1, length + 1)

    def atom(x, y, s):
        return ((x + 1) * width + y) * nsub + s

    xyz = np.array([x * a1 + y * a2 + offsets[s] for x in xs for y in range(width) for s in range(nsub)])
    orb = sisl.Atom(1, R=[1.5] * n)
    big = 10.0 * (np.abs(xyz).max() + 10.0)
    lattice = sisl.Lattice([[big, 0, 0], [0, big, 0], [0, 0, big]], nsc=[1, 1, 1])
    geom = sisl.Geometry(xyz, orb, lattice=lattice)
    na = len(xyz)
    no = na * n
    rows, cols, vals = [], [], []

    def block(i, j, m):
        # Orbital block <atom i| H |atom j> = m.
        r = np.repeat(np.arange(n), n) + i * n
        c = np.tile(np.arange(n), n) + j * n
        rows.append(r)
        cols.append(c)
        vals.append(m.reshape(-1).real)

    eye = np.eye(n)
    for x in xs:
        for y in range(width):
            for s in range(nsub):
                v = onsites[s]
                if pot is not None and 0 <= x < length:
                    v = v + pot[x, y, s] * eye
                block(atom(x, y, s), atom(x, y, s), v)
    for f, t, (dx, dy), m in hoppings:
        for x in xs:
            for y in range(width):
                x2, y2 = x + dx, y + dy
                if -1 <= x2 <= length and 0 <= y2 < width:
                    i, j = atom(x, y, f), atom(x2, y2, t)
                    block(i, j, m)
                    block(j, i, m.conj().T)
    h = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(no, no))
    h.sum_duplicates()
    device = sisl.Hamiltonian.fromsp(geom, h)

    # Electrode: one column, periodic along a1.
    exyz = np.array([y * a2 + offsets[s] for y in range(width) for s in range(nsub)])
    elattice = sisl.Lattice([a1, [0, big, 0], [0, 0, big]], nsc=[3, 1, 1])
    egeom = sisl.Geometry(exyz, orb, lattice=elattice)
    eno = len(exyz) * n
    rows, cols, vals = [], [], []
    for y in range(width):
        for s in range(nsub):
            i = y * nsub + s
            block(i, i, onsites[s])
    for f, t, (dx, dy), m in hoppings:
        for y in range(width):
            y2 = y + dy
            if not 0 <= y2 < width:
                continue
            i, j = y * nsub + f, y2 * nsub + t
            # <(0, i)| H |(dx, j)> = m, and <(0, j)| H |(-dx, i)> = m†.
            isc = egeom.lattice.sc_index([dx, 0, 0])
            isc_back = egeom.lattice.sc_index([-dx, 0, 0])
            r = np.repeat(np.arange(n), n)
            c = np.tile(np.arange(n), n)
            rows.append(r + i * n)
            cols.append(c + j * n + isc * eno)
            vals.append(m.reshape(-1).real)
            rows.append(r + j * n)
            cols.append(c + i * n + isc_back * eno)
            vals.append(m.conj().T.reshape(-1).real)
    eh = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(eno, eno * egeom.n_s)
    )
    eh.sum_duplicates()
    electrode = sisl.Hamiltonian.fromsp(egeom, eh)
    return device, electrode, length


FDF = """SystemLabel device
TBT.HS device.nc
TBT.CDF.Precision double
TBT.Elecs.Eta {eta} eV
TBT.Contours.Eta 0. eV
%block TBT.Contours
  points
%endblock
%block TBT.Contour.points
  from {emin} eV to {emax} eV
    file energies.txt
%endblock
%block TBT.Elecs
  Left
  Right
%endblock
%block TBT.Elec.Left
  HS electrode.nc
  semi-inf-direction -A1
  electrode-position 1
%endblock
%block TBT.Elec.Right
  HS electrode.nc
  semi-inf-direction +A1
  electrode-position end -1
%endblock
"""


def tbtrans(work, energies, eta):
    with open(f"{work}/energies.txt", "w") as f:
        for e in energies:
            f.write(f"{e:.12f} 1.0\n")
    with open(f"{work}/RUN.fdf", "w") as f:
        f.write(FDF.format(eta=eta, emin=min(energies) - 1.0, emax=max(energies) + 1.0))
    for old in os.listdir(work):
        if old.startswith("device.") and old not in ("device.nc",):
            os.remove(f"{work}/{old}")
    t = time.perf_counter()
    exe = os.path.join(sys.prefix, "bin", "tbtrans")
    out = subprocess.run([exe, "RUN.fdf"], cwd=work, capture_output=True, text=True)
    wall = time.perf_counter() - t
    if out.returncode != 0:
        sys.stderr.write(out.stdout[-3000:] + out.stderr[-3000:])
        raise SystemExit("tbtrans failed")
    tbt = sisl.get_sile(f"{work}/device.TBT.nc")
    e = np.array(tbt.E)
    t = np.array(tbt.transmission("Left", "Right"))
    return wall, e, t, out.stdout


def main():
    data, name, width = sys.argv[1], sys.argv[2], int(sys.argv[3])
    work = sys.argv[4] if len(sys.argv) > 4 else tempfile.mkdtemp(prefix=f"tbtrans-{name}-{width}-")
    eta = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-6
    os.makedirs(work, exist_ok=True)
    with open(f"{data}/{name}/problem.json") as f:
        spec = json.load(f)
    if spec.get("geometry") == "cross":
        raise SystemExit("cross geometry not set up for TBtrans")
    t0 = time.perf_counter()
    device, electrode, length = build(spec, data, name, width)
    device.write(f"{work}/device.nc")
    electrode.write(f"{work}/electrode.nc")
    build_s = time.perf_counter() - t0
    energies = spec["energies"]
    wall_all, e_out, ts, _ = tbtrans(work, energies, eta)
    wall_one, _, _, _ = tbtrans(work, energies[:1], eta)
    per = (wall_all - wall_one) / max(len(energies) - 1, 1)
    # TBtrans lists the energies in its own order: map back to the problem's.
    tmap = {round(float(e), 9): float(t) for e, t in zip(e_out, ts)}
    trans = [tmap[round(e, 9)] for e in energies]
    meta = glob.glob(f"{sys.prefix}/conda-meta/siesta-*.json")
    version = os.path.basename(meta[0]).split("-")[1] if meta else "?"
    # Peak memory of the tbtrans processes (bytes on macOS).
    children_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1048576.0
    print(json.dumps({
        "code": "tbtrans", "version": f"SIESTA {version}", "solver": f"TBtrans (BTD), electrodes at eta = {eta} eV",
        "problem": name, "width": width, "length": length, "orbitals": spec["orbitals"], "threads": 1,
        "build_s": build_s, "times_s": [per] * len(energies), "setup_s": wall_one - per,
        "total_s": wall_all, "energies": energies, "transmissions": trans, "children_peak_mb": children_mb,
    }))
    if len(sys.argv) <= 4:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
