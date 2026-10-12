//! Benchmarks of caroli against other codes on shared problems (`benchmarks/`).
//!
//! - `caroli-bench prepare`: writes, for every problem, the model (on-site and hopping
//!   matrices) and the disorder of every width to `benchmarks/data/`. The other codes build
//!   their systems from these files, so all of them solve the same Hamiltonian.
//! - `caroli-bench run [--code caroli|kwant|all] [--problem NAME] [--width W] [--threads N]`:
//!   runs each (problem, width, code) in its own process under `/usr/bin/time -l`, and appends
//!   one JSON line per run to `benchmarks/results/<machine>.jsonl`. The line includes the peak
//!   memory, and flags a GitHub runner job that was busy during the run.
//! - `caroli-bench solve PROBLEM WIDTH [nd|rgf]`: the caroli runner for one case (one JSON
//!   line), by nested dissection (`caroli`, default) or recursive Green's function
//!   (`caroli-rgf`).
//! - `caroli-bench report [FILE]`: agreement between the codes, and timing tables.
//!
//! Times are wall-clock seconds. `times_s` holds one entry per energy, computed one after the
//! other (single-energy latency). `parallel_s`, for caroli, is the time for all the energies
//! at once on the threads given.

use caroli_disorder::rng::Xoshiro256;
use caroli_green::{Interface, NestedMultiTerminal, NestedTwoTerminal, TwoTerminal};
use caroli_lattice::{Lattice, Region, Site};
use caroli_model::presets::{BhzParams, bhz};
use caroli_model::{CMat, Model, c64};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[derive(Deserialize)]
struct Problems {
    problem: Vec<Problem>,
}

#[derive(Deserialize, Serialize, Clone)]
struct Problem {
    name: String,
    model: String,
    widths: Vec<usize>,
    length: Option<usize>,
    /// "cross": a width × width square with four arms of length width / 4, one lead per arm.
    geometry: Option<String>,
    anderson: Option<f64>,
    seed: Option<u64>,
    energies: Vec<f64>,
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("repository root")
}

fn bench_dir() -> PathBuf {
    root().join("benchmarks")
}

fn problems() -> Vec<Problem> {
    let text = std::fs::read_to_string(bench_dir().join("transport.toml")).expect("transport.toml");
    toml::from_str::<Problems>(&text).expect("transport.toml").problem
}

fn problem(name: &str) -> Problem {
    problems().into_iter().find(|p| p.name == name).unwrap_or_else(|| panic!("unknown problem {name}"))
}

fn scalar(x: f64, n: usize) -> CMat {
    CMat::from_fn(n, n, |i, j| if i == j { c64::new(x, 0.0) } else { c64::new(0.0, 0.0) })
}

/// The clean model of a problem (also the leads).
fn clean_model(name: &str) -> Model {
    match name {
        "square" => Model::new(Lattice::square(1.0, 1)).hopping(0, 0, [1, 0, 0], scalar(-1.0, 1)).hopping(
            0,
            0,
            [0, 1, 0],
            scalar(-1.0, 1),
        ),
        "bhz" => bhz(BhzParams::default()),
        // Honeycomb lattice, t = 1: the transport axis is a zigzag direction, and the coupling
        // between slices has half the rank of the slice.
        "graphene" => Model::new(Lattice::honeycomb(1.0, 1))
            .hopping(0, 1, [0, 0, 0], scalar(-1.0, 1))
            .hopping(0, 1, [0, -1, 0], scalar(-1.0, 1))
            .hopping(0, 1, [1, -1, 0], scalar(-1.0, 1)),
        _ => panic!("unknown model {name}"),
    }
}

fn data_dir(p: &Problem) -> PathBuf {
    bench_dir().join("data").join(&p.name)
}

fn length(p: &Problem, w: usize) -> usize {
    p.length.unwrap_or(w)
}

fn is_cross(p: &Problem) -> bool {
    p.geometry.as_deref() == Some("cross")
}

/// Bounding box of the sample in cells, `(nx, ny)`, and the cell at its corner.
fn sample_box(p: &Problem, w: usize) -> ((usize, usize), (i64, i64)) {
    if is_cross(p) {
        let arm = w / 4;
        ((w + 2 * arm, w + 2 * arm), (-(arm as i64), -(arm as i64)))
    } else {
        ((length(p, w), w), (0, 0))
    }
}

/// On-site disorder of a (problem, width): one value per site of the bounding box, index
/// `(x · ny + y) · sublattices + sublattice` from its corner.
fn disorder(p: &Problem, w: usize) -> Option<Vec<f64>> {
    let strength = p.anderson?;
    let subs = clean_model(&p.model).lattice().sublattices().len();
    let ((nx, ny), _) = sample_box(p, w);
    let mut rng = Xoshiro256::seed_from_u64(p.seed.unwrap_or(0) * 1_000_003 + w as u64);
    Some(rng.uniforms(nx * ny * subs).into_iter().map(|u| strength * (u - 0.5)).collect())
}

/// A `w × w` square with arms of length `w / 4` on its four edges, as wide as the edges, and
/// the interfaces of their leads: left, right, bottom, top.
fn cross(lattice: &Lattice, w: usize) -> (Region, Vec<Interface>) {
    let (w, arm) = (w as i64, (w / 4) as i64);
    let subs = lattice.sublattices().len();
    let sites = |cells: Vec<(i64, i64)>| -> Vec<Site> {
        cells
            .into_iter()
            .flat_map(|(x, y)| (0..subs).map(move |s| Site { cell: [x, y, 0], sub: s }))
            .collect()
    };
    let cells: Vec<(i64, i64)> = (-arm..w + arm)
        .flat_map(|x| (-arm..w + arm).map(move |y| (x, y)))
        .filter(|&(x, y)| (0..w).contains(&x) || (0..w).contains(&y))
        .collect();
    let column = |x: i64| Region::from_sites(lattice, sites((0..w).map(|y| (x, y)).collect()));
    let row = |y: i64| Region::from_sites(lattice, sites((0..w).map(|x| (x, y)).collect()));
    let interfaces = vec![
        Interface { slice: column(-arm), outward: [-1, 0, 0] },
        Interface { slice: column(w + arm - 1), outward: [1, 0, 0] },
        Interface { slice: row(-arm), outward: [0, -1, 0] },
        Interface { slice: row(w + arm - 1), outward: [0, 1, 0] },
    ];
    (Region::from_sites(lattice, sites(cells)), interfaces)
}

fn matrix_json(m: &CMat) -> Value {
    json!(
        (0..m.nrows())
            .map(|i| (0..m.ncols()).map(|j| [m[(i, j)].re, m[(i, j)].im]).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    )
}

/// Writes `problem.json` (model, energies) and the disorder files of every problem.
fn prepare() {
    for p in problems() {
        let dir = data_dir(&p);
        std::fs::create_dir_all(&dir).expect("data directory");
        let model = clean_model(&p.model);
        let lattice = model.lattice();
        let subs = lattice.sublattices();
        let orbitals = subs[0].orbitals;
        assert!(subs.iter().all(|s| s.orbitals == orbitals), "the export takes equal orbital counts");
        let sublattices: Vec<Value> = subs
            .iter()
            .enumerate()
            .map(|(k, s)| {
                let onsite = model.onsite_matrix(&Site { cell: [0, 0, 0], sub: k });
                json!({"offset": [s.offset[0], s.offset[1]], "onsite": matrix_json(&onsite)})
            })
            .collect();
        let vectors: Vec<Value> = lattice.vectors().iter().map(|v| json!([v[0], v[1]])).collect();
        let hoppings: Vec<Value> = model
            .hoppings()
            .iter()
            .map(|h| {
                json!({"from": h.from, "to": h.to, "offset": [h.offset[0], h.offset[1]],
                       "matrix": matrix_json(&h.matrix)})
            })
            .collect();
        let spec = json!({
            "name": p.name, "model": p.model, "orbitals": orbitals, "length": p.length, "geometry": p.geometry,
            "anderson": p.anderson, "energies": p.energies,
            "vectors": vectors, "sublattices": sublattices, "hoppings": hoppings,
            "convention": "hopping matrix = <(R, from)|H|(R + offset, to)>; disorder index (x W + y) S + s",
        });
        std::fs::write(dir.join("problem.json"), serde_json::to_string_pretty(&spec).unwrap()).unwrap();
        for &w in &p.widths {
            if let Some(v) = disorder(&p, w) {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(dir.join(format!("disorder_W{w}.f64")), bytes).unwrap();
            }
        }
        println!("{}: {} × {orbitals} orbitals per cell, widths {:?}", p.name, subs.len(), p.widths);
    }
}

/// caroli on one (problem, width): one JSON line on stdout.
fn solve(name: &str, w: usize, solver: &str) {
    let p = problem(name);
    let l = length(&p, w);
    let clean = clean_model(&p.model);
    let n = clean.lattice().sublattices()[0].orbitals;
    let pot = disorder(&p, w);
    let t0 = Instant::now();
    let subs = clean.lattice().sublattices().len();
    let ((_, ny), (x0, y0)) = sample_box(&p, w);
    let device = clean_model(&p.model).with_onsite_term(move |site: &Site| {
        let (x, y) = ((site.cell[0] - x0) as usize, (site.cell[1] - y0) as usize);
        pot.as_ref().map(|v| scalar(v[(x * ny + y) * subs + site.sub], n))
    });
    // Each energy gives the transmission, or for a cross the T_pq (p ≠ q, p-major).
    // Independent blocks (e.g. the spin blocks of BHZ) are solved on their own by every solver.
    type Solver = Box<dyn Fn(f64) -> Vec<f64> + Sync>;
    let (transmission, block_count): (Solver, usize) = if is_cross(&p) {
        let (region, interfaces) = cross(clean.lattice(), w);
        let sys = NestedMultiTerminal::from_model(&device, &clean, &region, &interfaces);
        let count = sys.block_count();
        let off_diagonal = move |e: f64| -> Vec<f64> {
            let t = sys.transmissions(e).expect("transmissions");
            let n = t.len();
            (0..n)
                .flat_map(|p| (0..n).filter(move |&q| q != p).map(move |q| (p, q)))
                .map(|(p, q)| t[p][q])
                .collect()
        };
        (Box::new(off_diagonal), count)
    } else {
        let region = Region::cuboid(clean.lattice(), [l, w, 0]);
        match solver {
            "rgf" => {
                let sys = TwoTerminal::from_model(&device, &clean, &region, 0);
                let blocks = sys.blocks();
                let parts: Vec<TwoTerminal> = if blocks.len() > 1 {
                    blocks.iter().map(|b| sys.restricted(b)).collect()
                } else {
                    vec![sys]
                };
                let count = parts.len();
                (
                    Box::new(move |e| {
                        vec![parts.iter().map(|s| s.transmission(e).expect("transmission")).sum()]
                    }),
                    count,
                )
            }
            _ => {
                let sys = NestedTwoTerminal::from_model(&device, &clean, &region, 0);
                let count = sys.block_count();
                (Box::new(move |e| vec![sys.transmission(e).expect("transmission")]), count)
            }
        }
    };
    let build = t0.elapsed().as_secs_f64();
    let (mut times, mut ts) = (Vec::new(), Vec::new());
    for &e in &p.energies {
        let t = Instant::now();
        ts.extend(transmission(e));
        times.push(t.elapsed().as_secs_f64());
    }
    // All the energies at once; on one thread this is the sum of the single-energy times.
    let parallel = if rayon::current_num_threads() > 1 {
        let t = Instant::now();
        let _: Vec<Vec<f64>> = p.energies.par_iter().map(|&e| transmission(e)).collect();
        t.elapsed().as_secs_f64()
    } else {
        times.iter().sum()
    };
    let line = json!({
        "code": if solver == "rgf" { "caroli-rgf" } else { "caroli" }, "version": env!("CARGO_PKG_VERSION"),
        "problem": p.name, "width": w, "length": l, "orbitals": n, "threads": rayon::current_num_threads(),
        "blocks": block_count,
        "build_s": build, "times_s": times, "parallel_s": parallel, "energies": p.energies,
        "transmissions": ts,
    });
    println!("{line}");
}

fn shell(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd)
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

/// A GitHub runner job in progress (it would perturb the timings).
fn runner_busy() -> bool {
    !shell("pgrep", &["-f", "Runner.Worker"]).is_empty()
}

/// The RGF costs O(L W³): beyond this width it takes minutes per energy, and the smaller widths
/// already show its scaling.
const RGF_MAX_WIDTH: usize = 256;

fn run(args: &[String]) {
    let opt = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
    let codes: Vec<String> = match opt("--code").as_deref() {
        None | Some("all") => vec!["caroli".into(), "caroli-rgf".into(), "kwant".into()],
        Some(c) => vec![c.to_string()],
    };
    let threads = opt("--threads").unwrap_or_else(|| "1".into());
    let machine = shell("hostname", &["-s"]);
    let cpu = shell("sysctl", &["-n", "machdep.cpu.brand_string"]);
    let commit = shell("git", &["-C", root().to_str().unwrap(), "rev-parse", "--short", "HEAD"]);
    let out = bench_dir().join("results");
    std::fs::create_dir_all(&out).unwrap();
    let file = out.join(format!("{machine}.jsonl"));
    let exe = std::env::current_exe().unwrap();
    let python = bench_dir().join("envs/kwant/bin/python");
    let script = bench_dir().join("kwant/transport.py");
    for p in problems() {
        if opt("--problem").is_some_and(|n| n != p.name) {
            continue;
        }
        for &w in &p.widths {
            if opt("--width").is_some_and(|x| x.parse::<usize>().unwrap() != w) {
                continue;
            }
            for code in &codes {
                if code == "caroli-rgf" && (w > RGF_MAX_WIDTH || is_cross(&p)) {
                    continue;
                }
                for attempt in 0..3 {
                    // Wait for the GitHub runner to be idle (a CI job would perturb the timings).
                    let mut waited = false;
                    while runner_busy() {
                        if !waited {
                            eprintln!("waiting for the GitHub runner job to finish…");
                            waited = true;
                        }
                        std::thread::sleep(std::time::Duration::from_secs(15));
                    }
                    let busy_before = false;
                    let mut cmd = Command::new("/usr/bin/time");
                    cmd.arg("-l");
                    match code.as_str() {
                        "caroli" => cmd.arg(&exe).args(["solve", &p.name, &w.to_string(), "nd"]),
                        "caroli-rgf" => cmd.arg(&exe).args(["solve", &p.name, &w.to_string(), "rgf"]),
                        "kwant" => cmd.arg(&python).arg(&script).args([
                            data_dir(&p).parent().unwrap().to_str().unwrap(),
                            &p.name,
                            &w.to_string(),
                        ]),
                        other => panic!("unknown code {other}"),
                    };
                    for var in [
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS",
                    ] {
                        cmd.env(var, "1");
                    }
                    cmd.env("RAYON_NUM_THREADS", &threads);
                    let output = cmd.output().expect("runner");
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let Some(json_line) = stdout.lines().rev().find(|l| l.starts_with('{')) else {
                        eprintln!("{code} {} W={w} failed:\n{stderr}", p.name);
                        continue;
                    };
                    let mut v: Value = serde_json::from_str(json_line).expect("runner JSON");
                    let rss = stderr
                        .lines()
                        .find(|l| l.contains("maximum resident set size"))
                        .and_then(|l| l.split_whitespace().next())
                        .and_then(|x| x.parse::<f64>().ok())
                        .unwrap_or(f64::NAN);
                    let busy = busy_before || runner_busy();
                    if busy && attempt < 2 {
                        eprintln!(
                            "{code} {} W={w}: a runner job started during the run, measuring again",
                            p.name
                        );
                        continue;
                    }
                    let obj = v.as_object_mut().unwrap();
                    obj.insert("peak_rss_mb".into(), json!(rss / 1048576.0));
                    obj.insert("runner_busy".into(), json!(busy));
                    obj.insert("machine".into(), json!(machine));
                    obj.insert("cpu".into(), json!(cpu));
                    obj.insert("caroli_commit".into(), json!(commit));
                    obj.insert(
                        "unix_time".into(),
                        json!(
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs()
                        ),
                    );
                    let line = serde_json::to_string(&v).unwrap();
                    use std::io::Write;
                    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&file).unwrap();
                    writeln!(f, "{line}").unwrap();
                    let times: Vec<f64> =
                        v["times_s"].as_array().unwrap().iter().filter_map(|x| x.as_f64()).collect();
                    println!(
                        "{code:7} {:16} W={w:<5} build {:8.3} s, median per energy {:9.4} s, peak {:8.1} MB",
                        p.name,
                        v["build_s"].as_f64().unwrap_or(f64::NAN),
                        median(&times),
                        rss / 1048576.0
                    );
                    break;
                }
            }
        }
    }
    println!("results appended to {}", file.display());
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if s.is_empty() { f64::NAN } else { s[s.len() / 2] }
}

/// Latest run of every (problem, width, code, threads) in a results file: a summary of caroli
/// against Kwant (median time per energy, speed-up, memory, agreement), then every run.
fn report(args: &[String]) {
    let file = args
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| bench_dir().join("results").join(format!("{}.jsonl", shell("hostname", &["-s"]))));
    let text = std::fs::read_to_string(&file).expect("results file");
    let mut latest: BTreeMap<(String, u64, String, u64), Value> = BTreeMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line).unwrap();
        let key = (
            v["problem"].as_str().unwrap().to_string(),
            v["width"].as_u64().unwrap(),
            v["code"].as_str().unwrap().to_string(),
            v["threads"].as_u64().unwrap_or(1),
        );
        // Runs before the nested-dissection solver were the recursive one.
        let key = if key.2 == "caroli"
            && v["blocks"].is_u64()
            && v.get("solver").is_none()
            && v["caroli_commit"] == "1807a24"
        {
            (key.0, key.1, "caroli-rgf".to_string(), key.3)
        } else {
            key
        };
        latest.insert(key, v);
    }
    let numbers = |v: &Value, field: &str| -> Vec<f64> {
        v[field].as_array().map(|a| a.iter().filter_map(|x| x.as_f64()).collect()).unwrap_or_default()
    };
    let per_energy = |v: &Value| median(&numbers(v, "times_s"));
    let peak = |v: &Value| v["peak_rss_mb"].as_f64().unwrap_or(f64::NAN);
    let busy = |v: &Value| if v["runner_busy"].as_bool() == Some(true) { " (busy)" } else { "" };

    // Header: machine and commits.
    let first = latest.values().next();
    let mut commits: Vec<String> = latest
        .iter()
        .filter(|((_, _, code, _), _)| code.starts_with("caroli"))
        .filter_map(|(_, v)| v["caroli_commit"].as_str().map(str::to_string))
        .collect();
    commits.sort();
    commits.dedup();
    if let Some(v) = first {
        let text = |field: &str| v[field].as_str().unwrap_or("?").to_string();
        println!("Machine: {} ({}). caroli commits: {}.", text("machine"), text("cpu"), commits.join(", "));
        println!();
    }
    let kwant_version = latest.iter().find(|((_, _, code, _), _)| code == "kwant").map(|(_, v)| {
        format!("{} ({})", v["version"].as_str().unwrap_or("?"), v["solver"].as_str().unwrap_or("?"))
    });
    if let Some(k) = kwant_version {
        println!("Kwant {k}, one thread. Times are medians over the energies of each problem.");
        println!();
    }

    // Summary: one row per (problem, width).
    // (problem, width) → (code, threads) → run.
    type Runs<'a> = BTreeMap<(String, u64), &'a Value>;
    let mut cases: BTreeMap<(String, u64), Runs> = BTreeMap::new();
    for ((p, w, code, threads), v) in &latest {
        cases.entry((p.clone(), *w)).or_default().insert((code.clone(), *threads), v);
    }
    println!(
        "| problem | W | caroli (s) | Kwant (s) | speed-up | caroli RGF (s) | memory caroli / Kwant (MB) | caroli, all energies, most threads (s) | max ΔT |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    for ((p, w), runs) in &cases {
        let get = |code: &str| runs.get(&(code.to_string(), 1)).copied();
        let (caroli, kwant, rgf) = (get("caroli"), get("kwant"), get("caroli-rgf"));
        let parallel = runs
            .iter()
            .filter(|((code, threads), _)| code == "caroli" && *threads > 1)
            .max_by_key(|((_, threads), _)| *threads)
            .map(|((_, threads), v)| {
                format!("{:.3} ({threads} threads)", v["parallel_s"].as_f64().unwrap_or(f64::NAN))
            })
            .unwrap_or_else(|| "—".into());
        let time = |v: Option<&Value>| v.map_or("—".into(), |v| format!("{:.4}{}", per_energy(v), busy(v)));
        let speedup = match (caroli, kwant) {
            (Some(c), Some(k)) => format!("{:.1}×", per_energy(k) / per_energy(c)),
            _ => "—".into(),
        };
        let memory = match (caroli, kwant) {
            (Some(c), Some(k)) => format!("{:.0} / {:.0}", peak(c), peak(k)),
            (Some(c), None) => format!("{:.0} / —", peak(c)),
            _ => "—".into(),
        };
        let agreement = caroli.map(|c| numbers(c, "transmissions")).map_or("—".into(), |reference| {
            let worst = runs
                .values()
                .map(|v| numbers(v, "transmissions"))
                .filter(|t| t.len() == reference.len())
                .map(|t| t.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max))
                .fold(0.0, f64::max);
            format!("{worst:.0e}")
        });
        println!(
            "| {p} | {w} | {} | {} | {speedup} | {} | {memory} | {parallel} | {agreement} |",
            time(caroli),
            time(kwant),
            time(rgf)
        );
    }

    // Every run.
    println!();
    println!("Every run:");
    println!();
    println!(
        "| problem | W | orbitals | code | threads | build (s) | per energy (s) | all energies (s) | peak (MB) | commit |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for ((p, w), runs) in &cases {
        for ((code, threads), v) in runs {
            let times = numbers(v, "times_s");
            let all = v["parallel_s"].as_f64().unwrap_or(times.iter().sum());
            println!(
                "| {p} | {w} | {} | {code}{} | {threads} | {:.3} | {:.4} | {:.3} | {:.0} | {} |",
                v["orbitals"],
                busy(v),
                v["build_s"].as_f64().unwrap_or(f64::NAN),
                median(&times),
                all,
                peak(v),
                v["caroli_commit"].as_str().unwrap_or("—"),
            );
        }
    }
}

/// Throughput of the nested-dissection transmission against the number of threads:
/// `count` energies (the problem's, shifted slightly to make them all different) at once on
/// pools of each size.
fn scaling(name: &str, w: usize, count: usize, threads: &[usize]) {
    let p = problem(name);
    let l = length(&p, w);
    let clean = clean_model(&p.model);
    let n = clean.lattice().sublattices()[0].orbitals;
    let subs = clean.lattice().sublattices().len();
    let pot = disorder(&p, w);
    let ((_, ny), (x0, y0)) = sample_box(&p, w);
    let device = clean_model(&p.model).with_onsite_term(move |site: &Site| {
        let (x, y) = ((site.cell[0] - x0) as usize, (site.cell[1] - y0) as usize);
        pot.as_ref().map(|v| scalar(v[(x * ny + y) * subs + site.sub], n))
    });
    let region = Region::cuboid(clean.lattice(), [l, w, 0]);
    let sys = NestedTwoTerminal::from_model(&device, &clean, &region, 0);
    let energies: Vec<f64> =
        (0..count).map(|k| p.energies[k % p.energies.len()] + 1e-3 * (k / p.energies.len()) as f64).collect();
    let mut base = None;
    for &t in threads {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(t).build().expect("thread pool");
        let start = Instant::now();
        let ts = pool.install(|| sys.transmissions(&energies));
        let wall = start.elapsed().as_secs_f64();
        assert!(ts.iter().all(|t| t.is_ok()));
        let rate = count as f64 / wall;
        let speedup = base.map_or(1.0, |b: f64| rate / b);
        base.get_or_insert(rate);
        println!(
            "{name} W={w}: {t:2} threads, {count} energies in {wall:.3} s, {rate:.2} energies/s, ×{speedup:.2}"
        );
    }
}

/// Time of the lead modes against the whole nested-dissection transmission, at one energy.
fn profile(name: &str, w: usize) {
    let p = problem(name);
    let l = length(&p, w);
    let clean = clean_model(&p.model);
    let region = Region::cuboid(clean.lattice(), [l, w, 0]);
    let first = &region.slices(0)[0];
    let lead = caroli_green::Lead {
        h00: clean.hamiltonian(first),
        h01: clean.coupling(first, &first.translated([1, 0, 0])),
    };
    let e = p.energies[p.energies.len() / 2];
    let t = Instant::now();
    let _ = lead.modes(e).expect("modes");
    let modes = t.elapsed().as_secs_f64();
    let sys = NestedTwoTerminal::from_model(&clean, &clean, &region, 0);
    let t = Instant::now();
    let _ = sys.transmission(e).expect("transmission");
    let total = t.elapsed().as_secs_f64();
    println!(
        "{name} W={w} E={e}: lead modes of the whole cell {modes:.3} s, nested transmission {total:.3} s ({} blocks)",
        sys.block_count()
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("profile") => profile(&args[2], args[3].parse().expect("width")),
        Some("scaling") => {
            let count = args[4].parse().expect("energies");
            let threads: Vec<usize> = args[5..].iter().map(|t| t.parse().expect("threads")).collect();
            scaling(&args[2], args[3].parse().expect("width"), count, &threads)
        }
        Some("prepare") => prepare(),
        Some("solve") => {
            solve(&args[2], args[3].parse().expect("width"), args.get(4).map_or("nd", String::as_str))
        }
        Some("run") => run(&args[2..]),
        Some("report") => report(&args[2..]),
        _ => eprintln!(
            "usage: caroli-bench prepare | run [--code C] [--problem P] [--width W] [--threads N] | solve P W | report [FILE]"
        ),
    }
}
