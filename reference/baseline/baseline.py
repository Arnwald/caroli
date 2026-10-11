#!/usr/bin/env python3
"""Build the thesis reference codes and record or check their golden outputs.

    python3 baseline.py deps                     # serial MPI stub + LAPACK95
    python3 baseline.py build [--check] [PROGRAM ...]  # compile (--check: bounds checking)
    python3 baseline.py run [--check] [CASE ...]  # run cases, compare with golden/
    python3 baseline.py run --update [CASE ...]  # run cases, (re)record golden/
    python3 baseline.py show FILE ...            # list the records of a Fortran output file
    python3 baseline.py list                     # programs and cases

The sources in reference/thesis are compiled as they are (never edited). Build products go to
target/reference/. Environment overrides: FC, CC, LAPACK95_SRC, FFTW_PREFIX, LAPACK_LIBS.
"""
import concurrent.futures
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPO = BASE.parent.parent
THESIS = REPO / "reference" / "thesis"
TARGET = REPO / "target" / "reference"
CASES = BASE / "cases"
GOLDEN = BASE / "golden"

FC = os.environ.get("FC", "gfortran")
CC = os.environ.get("CC", "cc")
LAPACK95_SRC = Path(os.path.expanduser(os.environ.get(
    "LAPACK95_SRC", "~/Projects/electronic-transport/TI/Fortran/librairies/LAPACK95/SRC")))
FFLAGS = ["-O2", "-g", "-fbacktrace", "-ffree-line-length-none", "-fallow-argument-mismatch",
          "-std=legacy", "-w"]
CHECK_FFLAGS = ["-O0", "-fcheck=bounds,do,mem,pointer"]  # `--check`: hunt for undefined behaviour

# Program id -> source directory under reference/thesis. Directories with several main
# programs produce one executable per main, named <id>-<main file stem>.
PROGRAMS = {
    "conductance-2t": "ti/conductance/conductance-2t/src",
    "conductance-2t-opposite-g": "ti/conductance/conductance-2t/variants/paper-antiferro",
    "conductance-4t": "ti/conductance/conductance-4t/src",
    "conductance-ldos-current": "ti/conductance/conductance-ldos-current/src",
    "local-currents-2t": "ti/local-currents/local-currents-2t/src",
    "scba-anderson": "ti/born/scba-anderson/src",
    "scba-magnetic-matrix": "ti/born/scba-magnetic-matrix/src",
    "scba-magnetic-analytic": "ti/born/scba-magnetic-analytic/src",
    "dos-kpm-magnetic": "ti/dos/dos-kpm-magnetic/src",
    "dos-kpm-magnetic-opposite-g": "ti/dos/dos-kpm-magnetic/variants/gap-antiferro",
    "dos-spin-lesser": "ti/dos/dos-spin-lesser/src",
    "ti-diag": "ti/dos/ti-diag/src",
    "ribbon-bands": "ti/bands/ribbon-bands/src",
    "ribbon-bands-fourier-y": "ti/bands/ribbon-bands-fourier-y/src",
    "graphene-current": "graphene/current-pair-creation/src",
    "graphene-width": "graphene/wavepacket-width/src",
    "graphene-density": "graphene/wavepacket-density/src",
    "graphene-dos-kpm": "graphene/dos-kpm/src",
}


# Minimal build-time fixes, applied to a copy of the sources so that a modern gfortran accepts
# them; reference/thesis is never edited. Each patch must not change the computed results.
PATCHES = [
    # filename.f90: bumpchar (contained in module FileName) reuses the module variable `i`,
    # which is also the DO index of its caller filen. The trip count of a DO loop is fixed on
    # entry, so the original binaries behaved the same; gfortran >= 14 rejects it as an error.
    ("bumpchar-local-index",
     re.compile(r"(subroutine\s+bumpchar\s*\(\s*fname\s*\)[ \t]*\n\s*implicit\s+none)", re.I),
     r"\1\n   integer :: i  ! baseline patch: local index, see reference/baseline/README.md"),
    # graphene/wavepacket-width g_s.f90: the time tables are allocated (NTDIAG) but indexed up
    # to NTDIAG+1 (and reduced over NTDIAG+1 values); the out-of-bounds write crashes at random.
    # One more element keeps every in-bounds value and makes the last one well defined.
    # ti/local-currents green.f90 (2015): the y-current loop runs over iy = 2:Ny without
    # shifting its indices, so iy = Ny reads h_0 beyond its last column (garbage, run to run).
    # That bond (Ny -> Ny+1) does not exist in a ribbon: stopping at Ny-1 is the same as reading
    # a zero hopping and leaves every other value unchanged (bond 1 -> 2 stays missing).
    ("y-current-loop-in-bounds",
     re.compile(r"forall \(iy = 2:Ny, ii = 1:dim, jj = 1:dim\)"),
     "forall (iy = 2:Ny-1, ii = 1:dim, jj = 1:dim)"),
    ("time-tables-ntdiag-plus-one",
     re.compile(r"allocate\(cur_tab\(NTDIAG\),[^\n]*w_tab_red\(NTDIAG\)\)", re.I),
     lambda m: m.group(0).replace("(NTDIAG)", "(NTDIAG+1)")),
    # graphene/wavepacket-density g_s.f90: same pattern (NT/NDIAG + 1 diagnostics written into
    # tables of NTDIAG values); only the first NTDIAG values are written to the output files.
    ("density-time-tables-ntdiag-plus-one",
     re.compile(r"allocate\(cur_tab\(NTDIAG\), n_p_tab\(NTDIAG\)\)", re.I),
     "allocate(cur_tab(NTDIAG+1), n_p_tab(NTDIAG+1))"),
    # graphene utilities.f90 xy_mesh: loops over -N/2..N/2 (N+1 points) fill N x N arrays; the
    # extra column lands beyond the array (possibly in the next one). Loop over N points.
    ("xy-mesh-in-bounds",
     re.compile(r"do j = -N/2, N/2\n(\s*)do i = -N/2, N/2\n"),
     r"do j = -N/2, N/2-1\n\1do i = -N/2, N/2-1\n"),
    # graphene/current-pair-creation g_s.f90: the disorder seed adds the wall-clock date and
    # time, so every run draws a new realisation. The baseline needs a fixed seed.
    ("fixed-seed-no-clock",
     re.compile(r"call random_set\(1237\+taskid\+\s*&\s*\n\s*date_time\(1\)\+date_time\(2\)\+"
                r"date_time\(3\)\+date_time\(5\)\+date_time\(6\)\+date_time\(7\)\)"),
     "call random_set(1237+taskid)"),
]


def run(cmd, cwd=None, log=None):
    """Run a command; on failure print it with its output and raise."""
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if log is not None:
        log.write(" ".join(map(str, cmd)) + "\n" + r.stdout + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"failed: {' '.join(map(str, cmd))}\n{r.stdout}{r.stderr}")
    return r


def fftw_prefix():
    if "FFTW_PREFIX" in os.environ:
        return Path(os.environ["FFTW_PREFIX"])
    if shutil.which("brew"):
        r = subprocess.run(["brew", "--prefix", "fftw"], capture_output=True, text=True)
        if r.returncode == 0:
            return Path(r.stdout.strip())
    return Path("/usr")


def lapack_libs():
    if "LAPACK_LIBS" in os.environ:
        return os.environ["LAPACK_LIBS"].split()
    return ["-framework", "Accelerate"] if platform.system() == "Darwin" else ["-llapack", "-lblas"]


# --------------------------------------------------------------------------------------------
# Dependencies


def build_deps():
    shims = TARGET / "shims"
    shims.mkdir(parents=True, exist_ok=True)
    run([CC, "-O2", "-c", BASE / "shims" / "mpistub.c", "-o", shims / "mpistub.o"])
    run([CC, "-O2", "-I", fftw_prefix() / "include", "-c", BASE / "shims" / "fftw_threads_f77.c",
         "-o", shims / "fftw_threads_f77.o"])
    run(["ar", "rcs", shims / "libshims.a", shims / "mpistub.o", shims / "fftw_threads_f77.o"])
    shutil.copy(BASE / "shims" / "mpif.h", shims / "mpif.h")
    print(f"shims     -> {shims} (serial MPI, FFTW threads Fortran wrappers)")

    out = TARGET / "lapack95"
    src, mod, obj = out / "src", out / "mod", out / "obj"
    for d in (src, mod, obj):
        d.mkdir(parents=True, exist_ok=True)
    head = ["la_auxmod.f90", "f77_lapack_double_dcomplex.f90", "f95_lapack_double_dcomplex.f90"]
    rest = ["la_erinfo.f90", "la_ws_gels.f90", "la_ws_gelss.f90"] + sorted(
        p.name for p in LAPACK95_SRC.glob("la_[dz]*.f90"))
    # Copy the sources: gfortran would otherwise pick up stale .mod files next to them.
    for f in head + rest:
        shutil.copy(LAPACK95_SRC / f, src / f)
    flags = ["-O2", "-ffree-form", "-fallow-argument-mismatch", "-std=legacy", "-w", "-J", mod, "-I", mod]
    for f in head:  # module files first, in dependency order
        run([FC, *flags, "-c", src / f, "-o", obj / (Path(f).stem + ".o")])

    def compile_one(f):
        try:
            run([FC, *flags, "-c", src / f, "-o", obj / (Path(f).stem + ".o")])
            return None
        except RuntimeError:
            return f

    with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as pool:
        failed = [f for f in pool.map(compile_one, rest) if f]
    lib = out / "liblapack95.a"
    lib.unlink(missing_ok=True)
    run(["ar", "rcs", lib, *sorted(obj.glob("*.o"))])
    print(f"lapack95  -> {lib} ({len(head) + len(rest) - len(failed)} files"
          + (f", skipped: {' '.join(failed)}" if failed else "") + ")")


# --------------------------------------------------------------------------------------------
# Reference programs


def scan(path):
    """Return (modules defined, modules used, is_main) for a Fortran source."""
    code = "\n".join(line.split("!")[0] for line in path.read_text(errors="replace").splitlines())
    defined = {m.lower() for m in re.findall(r"^\s*module\s+(?!procedure\b)(\w+)", code, re.I | re.M)}
    used = {u.lower() for u in re.findall(r"^\s*use\s*(?:,\s*\w+\s*::)?\s*(\w+)", code, re.I | re.M)}
    is_main = re.search(r"^\s*program\s+\w+", code, re.I | re.M) is not None
    return defined, used, is_main


def build_order(files, info, roots):
    """Topological order of the files needed by `roots` (modules before their users)."""
    owner = {m: f for f in files for m in info[f][0]}
    order, seen = [], set()

    def visit(f):
        if f in seen:
            return
        seen.add(f)
        for m in sorted(info[f][1]):
            if m in owner and owner[m] != f:
                visit(owner[m])
        order.append(f)

    for r in roots:
        visit(r)
    return order


def patched_copy(src, dst):
    """Copy the Fortran sources of `src` into `dst`, applying PATCHES. Returns applied patches."""
    dst.mkdir(parents=True, exist_ok=True)
    applied = []
    for f in sorted(src.glob("*.f90")) + sorted(src.glob("*.f")):
        text = f.read_text(errors="replace")
        for name, pattern, repl in PATCHES:
            text, n = pattern.subn(repl, text)
            if n:
                applied.append(f"{name} ({f.name})")
        (dst / f.name).write_text(text)
    return applied


def build_program(pid, check=False):
    stub, fftw = TARGET / "shims", fftw_prefix()
    lapack95 = TARGET / "lapack95"
    link = ["-L", lapack95, "-llapack95", *lapack_libs(), "-L", stub, "-lshims", "-L", fftw / "lib",
            "-lfftw3_threads", "-lfftw3"]
    tag = "-check" if check else ""
    src = TARGET / f"src{tag}" / pid
    shutil.rmtree(src, ignore_errors=True)
    patches = patched_copy(THESIS / PROGRAMS[pid], src)
    files = sorted(src.glob("*.f90")) + sorted(src.glob("*.f"))
    info = {f: scan(f) for f in files}
    mains = [f for f in files if info[f][2]]
    libs = [f for f in files if not info[f][2]]
    built = []
    for main in mains:
        name = pid if len(mains) == 1 else f"{pid}-{main.stem}"
        bdir = TARGET / f"build{tag}" / name
        shutil.rmtree(bdir, ignore_errors=True)
        bdir.mkdir(parents=True)
        (bdir / "patches.txt").write_text("\n".join(patches) + "\n")
        exe = TARGET / f"bin{tag}" / name
        exe.parent.mkdir(parents=True, exist_ok=True)
        flags = [*FFLAGS, *(CHECK_FFLAGS if check else []), "-J", bdir, "-I", bdir, "-I", stub, "-I", fftw / "include",
                 "-I", lapack95 / "mod"]
        with open(bdir / "build.log", "w") as log:
            # Files reachable through USE first; if the link misses external procedures,
            # retry with every non-main file of the directory.
            for attempt in (build_order(files, info, [main]), build_order(files, info, libs + [main])):
                try:
                    objs = []
                    for f in attempt:
                        o = bdir / (f.stem + ".o")
                        run([FC, *flags, "-c", f, "-o", o], cwd=bdir, log=log)
                        objs.append(o)
                    run([FC, *objs, "-o", exe, *link], cwd=bdir, log=log)
                    break
                except RuntimeError as e:
                    err = str(e)
            else:
                print(f"FAIL  {name}: {err.splitlines()[0][:160]}\n      log: {bdir / 'build.log'}")
                continue
        built.append(name)
        print(f"ok    {name:34s} <- {PROGRAMS[pid]}")
    return built


# --------------------------------------------------------------------------------------------
# Fortran unformatted output and comparison


def records(data):
    """Split gfortran sequential unformatted data (int32 markers) into records, or None."""
    out, i = [], 0
    while i < len(data):
        if i + 4 > len(data):
            return None
        (n,) = struct.unpack_from("<i", data, i)
        if n < 0 or i + 8 + n > len(data) or struct.unpack_from("<i", data, i + 4 + n)[0] != n:
            return None
        out.append(data[i + 4:i + 4 + n])
        i += 8 + n
    return out


# Lines of text outputs that change from run to run (timings written by the programs).
VOLATILE = re.compile(r"\bCPU\b|CPU time|elapsed", re.I)


def close(a, b, rtol, atol):
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def compare_file(new, ref, rtol, atol):
    """None if equivalent, else a short description of the first difference."""
    a, b = new.read_bytes(), ref.read_bytes()
    if a == b:
        return None
    ra, rb = records(a), records(b)
    if ra is not None and rb is not None:
        if len(ra) != len(rb):
            return f"{len(ra)} records vs {len(rb)}"
        for k, (x, y) in enumerate(zip(ra, rb)):
            if len(x) != len(y):
                return f"record {k}: {len(x)} bytes vs {len(y)}"
            if len(x) % 8:
                if x != y:
                    return f"record {k}: bytes differ"
                continue
            xs = struct.unpack(f"<{len(x) // 8}d", x)
            ys = struct.unpack(f"<{len(y) // 8}d", y)
            for j, (u, v) in enumerate(zip(xs, ys)):
                if not close(u, v, rtol, atol):
                    return f"record {k}, value {j}: {u!r} vs {v!r}"
        return None
    # Text: compare token by token, numbers with tolerance, skipping run-time measurements.
    def tokens(data):
        lines = data.decode(errors="replace").splitlines()
        return [tok for line in lines if not VOLATILE.search(line) for tok in line.split()]

    ta, tb = tokens(a), tokens(b)
    if len(ta) != len(tb):
        return f"{len(ta)} tokens vs {len(tb)}"
    for u, v in zip(ta, tb):
        if u == v:
            continue
        try:
            fu, fv = (float(s.rstrip(",").replace("D", "E").replace("d", "e")) for s in (u, v))
        except ValueError:
            return f"token {u!r} vs {v!r}"
        if not close(fu, fv, rtol, atol):
            return f"value {u} vs {v}"
    return None


def show(path):
    data = Path(path).read_bytes()
    recs = records(data)
    if recs is None:
        print(f"{path}: text or not a sequential unformatted file ({len(data)} bytes)")
        return
    print(f"{path}: {len(recs)} record(s)")
    for k, r in enumerate(recs):
        desc = f"  [{k}] {len(r)} bytes"
        if len(r) % 8 == 0 and r:
            vals = struct.unpack(f"<{len(r) // 8}d", r)
            desc += f" = {len(vals)} float64, first {', '.join(f'{v:.6g}' for v in vals[:4])}"
        print(desc)


# --------------------------------------------------------------------------------------------
# Cases


def load_case(name):
    meta = tomllib.loads((CASES / name / "case.toml").read_text())
    meta.setdefault("rtol", 1e-8)
    meta.setdefault("atol", 1e-12)
    meta.setdefault("timeout", 300)
    return meta


def run_case(name, update, check=False):
    meta = load_case(name)
    exe = TARGET / ("bin-check" if check else "bin") / meta["program"]
    if not exe.exists():
        return f"missing executable {exe.name} (run: baseline.py build)"
    with tempfile.TemporaryDirectory(prefix=f"baseline-{name}-") as tmp:
        inputs = [p for p in (CASES / name).iterdir() if p.name != "case.toml"]
        for p in inputs:
            shutil.copy(p, tmp)
        env = dict(os.environ, OMP_NUM_THREADS="1")
        t0 = time.time()
        try:
            r = subprocess.run([exe], cwd=tmp, capture_output=True, text=True, env=env,
                               stdin=subprocess.DEVNULL, timeout=meta["timeout"])
        except subprocess.TimeoutExpired:
            return f"timeout after {meta['timeout']} s"
        dt = time.time() - t0
        if r.returncode != 0:
            tail = (r.stdout + r.stderr).strip().splitlines()[-6:]
            return f"exit code {r.returncode} after {dt:.1f} s\n      " + "\n      ".join(tail)
        produced = sorted(p for p in Path(tmp).iterdir() if p.name not in {i.name for i in inputs})
        gdir = GOLDEN / name
        if update:
            shutil.rmtree(gdir, ignore_errors=True)
            gdir.mkdir(parents=True)
            for p in produced:
                shutil.copy(p, gdir / p.name)
            size = sum(p.stat().st_size for p in produced)
            print(f"rec   {name:34s} {len(produced)} files, {size / 1024:.0f} KiB, {dt:.1f} s")
            return None
        want = sorted(p.name for p in gdir.iterdir()) if gdir.exists() else []
        got = [p.name for p in produced]
        if want != got:
            return f"files differ: produced {got}, golden {want}"
        for p in produced:
            diff = compare_file(p, gdir / p.name, meta["rtol"], meta["atol"])
            if diff:
                return f"{p.name}: {diff}"
        print(f"ok    {name:34s} {len(produced)} files, {dt:.1f} s")
        return None


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, args = argv[0], argv[1:]
    if cmd == "deps":
        build_deps()
        return 0
    if cmd == "build":
        check = "--check" in args
        pids = [a for a in args if a != "--check"] or list(PROGRAMS)
        for pid in pids:
            if pid not in PROGRAMS:
                print(f"unknown program {pid}")
                return 2
        ok = [build_program(pid, check) for pid in pids]
        return 0 if all(ok) else 1
    if cmd == "run":
        update, check = "--update" in args, "--check" in args
        if update and check:
            print("--update and --check are exclusive: golden outputs come from the optimised build")
            return 2
        names = [a for a in args if not a.startswith("--")] or sorted(
            p.name for p in CASES.iterdir() if p.is_dir())
        failures = {}
        for n in names:
            err = run_case(n, update, check)
            if err:
                failures[n] = err
                print(f"FAIL  {n}: {err}")
        print(f"{len(names) - len(failures)}/{len(names)} cases passed")
        return 1 if failures else 0
    if cmd == "show":
        for f in args:
            show(f)
        return 0
    if cmd == "list":
        for pid, d in PROGRAMS.items():
            print(f"program {pid:34s} {d}")
        for c in sorted(p.name for p in CASES.iterdir() if p.is_dir()) if CASES.exists() else []:
            meta = load_case(c)
            print(f"case    {c:34s} {meta['program']}: {meta.get('description', '')}")
        return 0
    print(f"unknown command {cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
