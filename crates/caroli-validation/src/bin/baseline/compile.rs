//! Building the shims, LAPACK95 and the thesis programs.

use regex::{Captures, Regex};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::LazyLock;

/// Program id → source directory under reference/thesis. Directories with several main
/// programs produce one executable per main, named `<id>-<main file stem>`.
pub const PROGRAMS: &[(&str, &str)] = &[
    ("conductance-2t", "ti/conductance/conductance-2t/src"),
    ("conductance-2t-opposite-g", "ti/conductance/conductance-2t/variants/paper-antiferro"),
    ("conductance-4t", "ti/conductance/conductance-4t/src"),
    ("conductance-ldos-current", "ti/conductance/conductance-ldos-current/src"),
    ("local-currents-2t", "ti/local-currents/local-currents-2t/src"),
    ("scba-anderson", "ti/born/scba-anderson/src"),
    ("scba-magnetic-matrix", "ti/born/scba-magnetic-matrix/src"),
    ("scba-magnetic-analytic", "ti/born/scba-magnetic-analytic/src"),
    ("dos-kpm-magnetic", "ti/dos/dos-kpm-magnetic/src"),
    ("dos-kpm-magnetic-opposite-g", "ti/dos/dos-kpm-magnetic/variants/gap-antiferro"),
    ("dos-spin-lesser", "ti/dos/dos-spin-lesser/src"),
    ("ti-diag", "ti/dos/ti-diag/src"),
    ("ribbon-bands", "ti/bands/ribbon-bands/src"),
    ("ribbon-bands-fourier-y", "ti/bands/ribbon-bands-fourier-y/src"),
    ("graphene-current", "graphene/current-pair-creation/src"),
    ("graphene-width", "graphene/wavepacket-width/src"),
    ("graphene-density", "graphene/wavepacket-density/src"),
    ("graphene-dos-kpm", "graphene/dos-kpm/src"),
];

const FFLAGS: &[&str] = &[
    "-O2",
    "-g",
    "-fbacktrace",
    "-ffree-line-length-none",
    "-fallow-argument-mismatch",
    "-std=legacy",
    "-w",
];
/// `--check`: hunt for undefined behaviour.
const CHECK_FFLAGS: &[&str] = &["-O0", "-fcheck=bounds,do,mem,pointer"];

/// A minimal build-time fix, applied to a copy of the sources so that a modern gfortran accepts
/// them or so that runs are reproducible; reference/thesis is never edited. A patch must not
/// change any value the original program computed correctly.
struct Patch {
    name: &'static str,
    pattern: &'static str,
    replace: fn(&Captures) -> String,
}

const PATCHES: &[Patch] = &[
    // filename.f90: bumpchar (contained in module FileName) reuses the module variable `i`,
    // which is also the DO index of its caller filen. The trip count of a DO loop is fixed on
    // entry, so the original binaries behaved the same; gfortran >= 14 rejects it as an error.
    Patch {
        name: "bumpchar-local-index",
        pattern: r"(?i)(subroutine\s+bumpchar\s*\(\s*fname\s*\)[ \t]*\n\s*implicit\s+none)",
        replace: |c| {
            format!(
                "{}\n   integer :: i  ! baseline patch: local index, see reference/baseline/README.md",
                &c[1]
            )
        },
    },
    // ti/local-currents green.f90 (2015): the y-current loop runs over iy = 2:Ny without shifting
    // its indices, so iy = Ny reads h_0 beyond its last column (garbage, run to run). That bond
    // (Ny -> Ny+1) does not exist in a ribbon: stopping at Ny-1 is the same as reading a zero
    // hopping and leaves every other value unchanged (bond 1 -> 2 stays missing).
    Patch {
        name: "y-current-loop-in-bounds",
        pattern: r"forall \(iy = 2:Ny, ii = 1:dim, jj = 1:dim\)",
        replace: |_| "forall (iy = 2:Ny-1, ii = 1:dim, jj = 1:dim)".to_owned(),
    },
    // graphene/wavepacket-width g_s.f90: the time tables are allocated (NTDIAG) but indexed up to
    // NTDIAG+1 (and reduced over NTDIAG+1 values); the out-of-bounds write crashes at random.
    Patch {
        name: "time-tables-ntdiag-plus-one",
        pattern: r"(?i)allocate\(cur_tab\(NTDIAG\),[^\n]*w_tab_red\(NTDIAG\)\)",
        replace: |c| c[0].replace("(NTDIAG)", "(NTDIAG+1)"),
    },
    // graphene/wavepacket-density g_s.f90: same pattern; only NTDIAG values are written out.
    Patch {
        name: "density-time-tables-ntdiag-plus-one",
        pattern: r"(?i)allocate\(cur_tab\(NTDIAG\), n_p_tab\(NTDIAG\)\)",
        replace: |_| "allocate(cur_tab(NTDIAG+1), n_p_tab(NTDIAG+1))".to_owned(),
    },
    // graphene utilities.f90 xy_mesh: loops over -N/2..N/2 (N+1 points) fill N x N arrays; the
    // extra column lands beyond the array (possibly in the next one). Loop over N points.
    Patch {
        name: "xy-mesh-in-bounds",
        pattern: r"do j = -N/2, N/2\n(\s*)do i = -N/2, N/2\n",
        replace: |c| format!("do j = -N/2, N/2-1\n{}do i = -N/2, N/2-1\n", &c[1]),
    },
    // graphene/current-pair-creation g_s.f90: the disorder seed adds the wall-clock date and
    // time, so every run draws a new realisation. The baseline needs a fixed seed.
    Patch {
        name: "fixed-seed-no-clock",
        pattern: r"call random_set\(1237\+taskid\+\s*&\s*\n\s*date_time\(1\)\+date_time\(2\)\+date_time\(3\)\+date_time\(5\)\+date_time\(6\)\+date_time\(7\)\)",
        replace: |_| "call random_set(1237+taskid)".to_owned(),
    },
];

pub fn repo() -> PathBuf {
    caroli_validation::repo_root()
}

pub fn target() -> PathBuf {
    repo().join("target/reference")
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_owned())
}

fn lapack95_src() -> PathBuf {
    let raw = env_or("LAPACK95_SRC", "~/Projects/electronic-transport/TI/Fortran/librairies/LAPACK95/SRC");
    match raw.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(rest),
        None => PathBuf::from(raw),
    }
}

fn fftw_prefix() -> PathBuf {
    if let Ok(p) = std::env::var("FFTW_PREFIX") {
        return p.into();
    }
    if let Ok(out) = Command::new("brew").args(["--prefix", "fftw"]).output()
        && out.status.success()
    {
        return String::from_utf8_lossy(&out.stdout).trim().into();
    }
    "/usr".into()
}

fn lapack_libs() -> Vec<String> {
    if let Ok(v) = std::env::var("LAPACK_LIBS") {
        return v.split_whitespace().map(str::to_owned).collect();
    }
    if cfg!(target_os = "macos") {
        vec!["-framework".into(), "Accelerate".into()]
    } else {
        vec!["-llapack".into(), "-lblas".into()]
    }
}

/// Runs a command; on failure returns its command line and output.
fn exec(cmd: &mut Command, log: Option<&mut String>) -> Result<(), String> {
    let out = cmd.output().map_err(|e| format!("{cmd:?}: {e}"))?;
    let text = format!(
        "{:?}\n{}{}",
        cmd,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    if let Some(log) = log {
        log.push_str(&text);
    }
    if out.status.success() { Ok(()) } else { Err(text) }
}

/// Builds the shims (serial MPI, FFTW threads Fortran wrappers) and LAPACK95.
pub fn deps() -> bool {
    match try_deps() {
        Ok(()) => true,
        Err(e) => {
            eprintln!("FAIL  {e}");
            false
        }
    }
}

fn try_deps() -> Result<(), String> {
    let base = repo().join("reference/baseline/shims");
    let shims = target().join("shims");
    std::fs::create_dir_all(&shims).map_err(|e| e.to_string())?;
    let cc = env_or("CC", "cc");
    exec(
        Command::new(&cc)
            .args(["-O2", "-c"])
            .arg(base.join("mpistub.c"))
            .arg("-o")
            .arg(shims.join("mpistub.o")),
        None,
    )?;
    exec(
        Command::new(&cc)
            .args(["-O2", "-I"])
            .arg(fftw_prefix().join("include"))
            .arg("-c")
            .arg(base.join("fftw_threads_f77.c"))
            .arg("-o")
            .arg(shims.join("fftw_threads_f77.o")),
        None,
    )?;
    let _ = std::fs::remove_file(shims.join("libshims.a"));
    exec(
        Command::new("ar")
            .arg("rcs")
            .arg(shims.join("libshims.a"))
            .arg(shims.join("mpistub.o"))
            .arg(shims.join("fftw_threads_f77.o")),
        None,
    )?;
    std::fs::copy(base.join("mpif.h"), shims.join("mpif.h")).map_err(|e| e.to_string())?;
    println!("shims     -> {} (serial MPI, FFTW threads Fortran wrappers)", shims.display());

    let out = target().join("lapack95");
    let (src, mods, objs) = (out.join("src"), out.join("mod"), out.join("obj"));
    for d in [&src, &mods, &objs] {
        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    let lsrc = lapack95_src();
    let head = ["la_auxmod.f90", "f77_lapack_double_dcomplex.f90", "f95_lapack_double_dcomplex.f90"];
    let mut rest: Vec<String> =
        ["la_erinfo.f90", "la_ws_gels.f90", "la_ws_gelss.f90"].map(String::from).to_vec();
    let mut routines: Vec<String> = std::fs::read_dir(&lsrc)
        .map_err(|e| format!("LAPACK95 sources ({}): {e}", lsrc.display()))?
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| (n.starts_with("la_d") || n.starts_with("la_z")) && n.ends_with(".f90"))
        .collect();
    routines.sort();
    rest.extend(routines);
    // Copy the sources: gfortran would otherwise pick up stale .mod files next to them.
    for f in head.iter().map(|s| s.to_string()).chain(rest.iter().cloned()) {
        std::fs::copy(lsrc.join(&f), src.join(&f)).map_err(|e| format!("{f}: {e}"))?;
    }
    let fc = env_or("FC", "gfortran");
    let compile = |f: &str| -> Result<(), String> {
        exec(
            Command::new(&fc)
                .args(["-O2", "-ffree-form", "-fallow-argument-mismatch", "-std=legacy", "-w", "-J"])
                .arg(&mods)
                .arg("-I")
                .arg(&mods)
                .arg("-c")
                .arg(src.join(f))
                .arg("-o")
                .arg(objs.join(f.replace(".f90", ".o"))),
            None,
        )
    };
    for f in head {
        compile(f)?; // module files first, in dependency order
    }
    let failed = parallel_map(&rest, |f| compile(f).err().map(|_| f.clone()));
    let failed: Vec<String> = failed.into_iter().flatten().collect();
    let lib = out.join("liblapack95.a");
    let _ = std::fs::remove_file(&lib);
    let mut objects: Vec<PathBuf> = std::fs::read_dir(&objs)
        .map_err(|e| e.to_string())?
        .filter_map(|e| Some(e.ok()?.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "o"))
        .collect();
    objects.sort();
    exec(Command::new("ar").arg("rcs").arg(&lib).args(&objects), None)?;
    println!(
        "lapack95  -> {} ({} files{})",
        lib.display(),
        head.len() + rest.len() - failed.len(),
        if failed.is_empty() { String::new() } else { format!(", skipped: {}", failed.join(" ")) }
    );
    Ok(())
}

/// Maps `f` over `items` on all cores, keeping the order.
fn parallel_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = items.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        let handles: Vec<_> =
            items.chunks(chunk).map(|c| s.spawn(|| c.iter().map(&f).collect::<Vec<R>>())).collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

/// Fortran source: modules defined, modules used, contains a main program.
struct SourceInfo {
    defined: BTreeSet<String>,
    used: BTreeSet<String>,
    main: bool,
}

static MODULE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?im)^\s*module\s+(\w+)").unwrap());
static USE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?im)^\s*use\s*(?:,\s*\w+\s*::)?\s*(\w+)").unwrap());
static PROGRAM: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?im)^\s*program\s+\w+").unwrap());

fn scan(text: &str) -> SourceInfo {
    let code: String = text.lines().map(|l| l.split('!').next().unwrap_or("")).collect::<Vec<_>>().join("\n");
    let names =
        |re: &Regex| -> BTreeSet<String> { re.captures_iter(&code).map(|c| c[1].to_lowercase()).collect() };
    let mut defined = names(&MODULE);
    defined.remove("procedure");
    SourceInfo { defined, used: names(&USE), main: PROGRAM.is_match(&code) }
}

/// Topological order of the files needed by `roots` (modules before their users).
fn build_order(files: &[PathBuf], info: &BTreeMap<PathBuf, SourceInfo>, roots: &[PathBuf]) -> Vec<PathBuf> {
    let owner: BTreeMap<&String, &PathBuf> =
        files.iter().flat_map(|f| info[f].defined.iter().map(move |m| (m, f))).collect();
    let mut order = Vec::new();
    let mut seen = BTreeSet::new();
    fn visit<'a>(
        f: &'a PathBuf,
        info: &'a BTreeMap<PathBuf, SourceInfo>,
        owner: &BTreeMap<&String, &'a PathBuf>,
        seen: &mut BTreeSet<&'a PathBuf>,
        order: &mut Vec<PathBuf>,
    ) {
        if !seen.insert(f) {
            return;
        }
        for m in &info[f].used {
            if let Some(o) = owner.get(m)
                && *o != f
            {
                visit(o, info, owner, seen, order);
            }
        }
        order.push(f.clone());
    }
    for r in roots {
        visit(r, info, &owner, &mut seen, &mut order);
    }
    order
}

/// The original thesis tree (`THESIS_ORIG`, by default `~/Projects/electronic-transport`).
fn original_tree() -> PathBuf {
    let raw = env_or("THESIS_ORIG", "~/Projects/electronic-transport");
    match raw.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(rest),
        None => PathBuf::from(raw),
    }
}

/// Third-party sources of `dir` (a directory of reference/thesis) that the repository leaves
/// out for want of a license (`reference/thesis/THIRD_PARTY.tsv`), taken from the original tree
/// and checked against their sha1.
fn third_party_sources(dir: &str) -> Result<Vec<PathBuf>, String> {
    let list = std::fs::read_to_string(repo().join("reference/thesis/THIRD_PARTY.tsv")).unwrap_or_default();
    let mut out = Vec::new();
    for line in list.lines().filter(|l| !l.starts_with('#')) {
        let cols: Vec<&str> = line.split('\t').collect();
        let Some(file) = cols.first().map(Path::new) else { continue };
        let is_source = file.extension().is_some_and(|x| x == "f90" || x == "f");
        if !is_source || file.parent() != Some(Path::new(dir)) || cols.len() < 3 {
            continue;
        }
        // The first origin when the file was copied from several places.
        let origin = original_tree().join(cols[2].split(" | ").next().unwrap_or_default());
        if !origin.is_file() || crate::coverage::sha1(&origin) != cols[1] {
            return Err(format!(
                "{} is third-party code kept out of the repository: expected at {} (set THESIS_ORIG)",
                file.display(),
                origin.display()
            ));
        }
        out.push(origin);
    }
    Ok(out)
}

/// Copies the Fortran sources of `src` (the directory `dir` of reference/thesis) into `dst`,
/// with the third-party ones from the original tree, applying the patches; returns those
/// applied.
fn patched_copy(src: &Path, dir: &str, dst: &Path) -> Result<Vec<String>, String> {
    let io = |e: std::io::Error| e.to_string();
    std::fs::create_dir_all(dst).map_err(io)?;
    let mut applied = Vec::new();
    let mut files: Vec<PathBuf> = std::fs::read_dir(src)
        .map_err(io)?
        .filter_map(|e| Some(e.ok()?.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "f90" || x == "f"))
        .collect();
    files.extend(third_party_sources(dir)?);
    files.sort_by_key(|f| f.file_name().map(|n| n.to_owned()));
    for f in files {
        let mut text = String::from_utf8_lossy(&std::fs::read(&f).map_err(io)?).into_owned();
        for p in PATCHES {
            let re = Regex::new(p.pattern).expect("patch pattern");
            if re.is_match(&text) {
                text = re.replace_all(&text, p.replace).into_owned();
                applied.push(format!("{} ({})", p.name, f.file_name().unwrap().to_string_lossy()));
            }
        }
        std::fs::write(dst.join(f.file_name().unwrap()), text).map_err(io)?;
    }
    Ok(applied)
}

/// Compiles the given programs (all if empty); returns false if one failed.
pub fn programs(ids: &[&str], check: bool) -> bool {
    let selected: Vec<&(&str, &str)> = if ids.is_empty() {
        PROGRAMS.iter().collect()
    } else {
        let mut out = Vec::new();
        for id in ids {
            match PROGRAMS.iter().find(|(p, _)| p == id) {
                Some(p) => out.push(p),
                None => {
                    eprintln!("unknown program {id}");
                    return false;
                }
            }
        }
        out
    };
    let mut ok = true;
    for (id, dir) in selected {
        ok &= program(id, dir, check);
    }
    ok
}

fn program(id: &str, dir: &str, check: bool) -> bool {
    let tag = if check { "-check" } else { "" };
    let (t, fftw) = (target(), fftw_prefix());
    let (shims, lapack95) = (t.join("shims"), t.join("lapack95"));
    let src = t.join(format!("src{tag}")).join(id);
    let _ = std::fs::remove_dir_all(&src);
    let patches = match patched_copy(&repo().join("reference/thesis").join(dir), dir, &src) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("FAIL  {id}: {e}");
            return false;
        }
    };
    let mut files: Vec<PathBuf> = std::fs::read_dir(&src)
        .unwrap()
        .filter_map(|e| Some(e.ok()?.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "f90" || x == "f"))
        .collect();
    files.sort();
    let info: BTreeMap<PathBuf, SourceInfo> = files
        .iter()
        .map(|f| (f.clone(), scan(&String::from_utf8_lossy(&std::fs::read(f).unwrap()))))
        .collect();
    let mains: Vec<PathBuf> = files.iter().filter(|f| info[*f].main).cloned().collect();
    let libs: Vec<PathBuf> = files.iter().filter(|f| !info[*f].main).cloned().collect();
    let fc = env_or("FC", "gfortran");
    let mut ok = true;
    for main in &mains {
        let stem = main.file_stem().unwrap().to_string_lossy();
        let name = if mains.len() == 1 { id.to_owned() } else { format!("{id}-{stem}") };
        let bdir = t.join(format!("build{tag}")).join(&name);
        let _ = std::fs::remove_dir_all(&bdir);
        std::fs::create_dir_all(&bdir).unwrap();
        std::fs::write(bdir.join("patches.txt"), patches.join("\n") + "\n").unwrap();
        let exe = t.join(format!("bin{tag}")).join(&name);
        std::fs::create_dir_all(exe.parent().unwrap()).unwrap();
        let mut log = String::new();
        // Files reachable through USE first; if the link misses external procedures, retry
        // with every non-main file of the directory.
        let attempts = [
            build_order(&files, &info, std::slice::from_ref(main)),
            build_order(&files, &info, &[libs.clone(), vec![main.clone()]].concat()),
        ];
        let mut err = String::new();
        let built = attempts.iter().any(|order| {
            let result = (|| -> Result<(), String> {
                let mut objs = Vec::new();
                for f in order {
                    let o = bdir.join(format!("{}.o", f.file_stem().unwrap().to_string_lossy()));
                    let mut cmd = Command::new(&fc);
                    cmd.args(FFLAGS)
                        .args(if check { CHECK_FFLAGS } else { &[] })
                        .arg("-J")
                        .arg(&bdir)
                        .arg("-I")
                        .arg(&bdir)
                        .arg("-I")
                        .arg(&shims)
                        .arg("-I")
                        .arg(fftw.join("include"))
                        .arg("-I")
                        .arg(lapack95.join("mod"))
                        .arg("-c")
                        .arg(f)
                        .arg("-o")
                        .arg(&o)
                        .current_dir(&bdir);
                    exec(&mut cmd, Some(&mut log))?;
                    objs.push(o);
                }
                let mut cmd = Command::new(&fc);
                cmd.args(&objs)
                    .arg("-o")
                    .arg(&exe)
                    .arg("-L")
                    .arg(&lapack95)
                    .arg("-llapack95")
                    .args(lapack_libs())
                    .arg("-L")
                    .arg(&shims)
                    .arg("-lshims")
                    .arg("-L")
                    .arg(fftw.join("lib"))
                    .args(["-lfftw3_threads", "-lfftw3"])
                    .current_dir(&bdir);
                exec(&mut cmd, Some(&mut log))
            })();
            result.map_err(|e| err = e).is_ok()
        });
        std::fs::write(bdir.join("build.log"), &log).unwrap();
        if built {
            println!("ok    {name:34} <- {dir}");
        } else {
            let first = err.lines().find(|l| l.contains("Error") || l.contains("Undefined")).unwrap_or("");
            eprintln!("FAIL  {name}: {first}\n      log: {}", bdir.join("build.log").display());
            ok = false;
        }
    }
    ok
}
