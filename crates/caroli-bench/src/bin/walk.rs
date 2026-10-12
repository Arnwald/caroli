//! Benchmark of the quantum walk on a graph of spins (Sellapillay & Verga, Phys. Rev. E 103,
//! 032123): the ring of `n` nodes, `θ = π/8`, `J = 0.2`, walker on color 0 of node 0, spins along
//! +x. The authors' code is not public: the competitors are the method their paper describes
//! written in Python (`benchmarks/walk/reference.py`: the one-step operator as a SciPy sparse
//! matrix, and the same step in vectorised NumPy).
//!
//! ```text
//! walk run [--sizes 13,15,17,19] [--steps 10] [--python PATH] [--python-max-n 19]
//! walk report [FILE]
//! ```
//!
//! `run` times every method for `steps` steps after one warm-up step, checks that each one gives
//! the same position distribution `p(x)` as caroli on the CPU (f64), and appends one JSON line
//! per method to `benchmarks/results/walk-<machine>.jsonl`. It also runs the checked GPU mode
//! (`caroli_walk::evolve` in f32, tolerance 10⁻³) for 50 × `steps` steps.

use caroli_model::c64;
use caroli_walk::{Options, PLUS, Precision, SpinWalk, evolve};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("repository root")
}

fn shell(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd)
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

fn walk(n: usize) -> SpinWalk {
    SpinWalk::ring(n, std::f64::consts::PI / 8.0, 0.2)
}

/// The initial state, rebuilt for each method so that only one state lives at a time.
fn initial(w: &SpinWalk) -> Vec<c64> {
    w.product_state(&w.localized(0, 0), &vec![PLUS; w.graph().nodes()])
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// A GitHub runner job in progress (it would perturb the timings).
fn runner_busy() -> bool {
    !shell("pgrep", &["-f", "Runner.Worker"]).is_empty()
}

fn run(args: &[String]) {
    let opt = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
    let sizes: Vec<usize> = opt("--sizes")
        .map(|s| s.split(',').map(|x| x.parse().expect("size")).collect())
        .unwrap_or_else(|| vec![13, 15, 17, 19]);
    let steps: usize = opt("--steps").map_or(10, |s| s.parse().expect("steps"));
    let python_max: usize = opt("--python-max-n").map_or(19, |s| s.parse().expect("n"));
    let python = opt("--python").map(PathBuf::from).unwrap_or_else(|| {
        let env = root().join("benchmarks/envs/kwant-accelerate/bin/python");
        if env.exists() { env } else { PathBuf::from("python3") }
    });
    let machine = shell("hostname", &["-s"]);
    let commit = shell("git", &["-C", root().to_str().unwrap(), "rev-parse", "--short", "HEAD"]);
    let date = shell("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]);
    let file = root().join("benchmarks/results").join(format!("walk-{machine}.jsonl"));
    let threads = rayon::current_num_threads();
    for &n in &sizes {
        let w = walk(n);
        let dim = w.dimension();
        let busy_before = runner_busy();
        let mut records: Vec<Value> = Vec::new();
        let mut record = |method: &str, used: usize, seconds: f64, extra: Value| {
            let mut r = json!({
                "date": date, "machine": machine, "commit": commit, "n": n, "dimension": dim,
                "method": method, "threads": used, "steps": steps, "s_per_step": seconds / steps as f64,
            });
            if let (Some(r), Some(e)) = (r.as_object_mut(), extra.as_object()) {
                r.extend(e.clone());
            }
            println!("{r}");
            records.push(r);
        };
        // caroli, CPU, all threads: the reference p(x) after 1 + steps steps.
        let mut psi = initial(&w);
        w.step(&mut psi);
        let start = Instant::now();
        w.steps(&mut psi, steps);
        record("caroli-cpu", threads, start.elapsed().as_secs_f64(), json!({}));
        let p_ref = w.node_probabilities(&psi);
        drop(psi);
        // caroli, CPU, one thread.
        if n <= 21 {
            let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().expect("pool");
            let seconds = one.install(|| {
                let mut psi = initial(&w);
                w.step(&mut psi);
                let start = Instant::now();
                w.steps(&mut psi, steps);
                start.elapsed().as_secs_f64()
            });
            record("caroli-cpu", 1, seconds, json!({}));
        }
        // caroli, the one-step operator as a sparse matrix (CSR, all threads).
        if n <= 19 {
            let built = Instant::now();
            let u = w.matrix();
            let build = built.elapsed().as_secs_f64();
            let (mut a, mut b) = (initial(&w), vec![c64::new(0.0, 0.0); dim]);
            u.apply(&a, &mut b);
            std::mem::swap(&mut a, &mut b);
            let start = Instant::now();
            for _ in 0..steps {
                u.apply(&a, &mut b);
                std::mem::swap(&mut a, &mut b);
            }
            let seconds = start.elapsed().as_secs_f64();
            let dp = max_diff(&w.node_probabilities(&a), &p_ref);
            record("caroli-csr", threads, seconds, json!({"build_s": build, "max_dp": dp}));
        }
        #[cfg(target_os = "macos")]
        gpu_runs(&w, steps, &p_ref, &mut record);
        // Python references.
        if n <= python_max {
            for method in ["sparse", "numpy"] {
                if method == "sparse" && n > 19 {
                    continue;
                }
                let script = root().join("benchmarks/walk/reference.py");
                let output = Command::new(&python)
                    .args([script.to_str().unwrap(), method, &n.to_string(), &steps.to_string()])
                    .env("OMP_NUM_THREADS", "1")
                    .env("OPENBLAS_NUM_THREADS", "1")
                    .env("VECLIB_MAXIMUM_THREADS", "1")
                    .output()
                    .expect("python");
                let text = String::from_utf8_lossy(&output.stdout);
                let Some(line) = text.lines().last().and_then(|l| serde_json::from_str::<Value>(l).ok())
                else {
                    eprintln!("python {method} n = {n}: {}", String::from_utf8_lossy(&output.stderr));
                    continue;
                };
                let p: Vec<f64> = line["p"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
                let seconds = line["s_per_step"].as_f64().unwrap() * steps as f64;
                record(
                    &format!("python-{method}"),
                    1,
                    seconds,
                    json!({"build_s": line["build_s"], "max_dp": max_diff(&p, &p_ref),
                           "versions": format!("numpy {}, scipy {}", line["numpy"].as_str().unwrap_or("?"), line["scipy"].as_str().unwrap_or("?"))}),
                );
            }
        }
        let busy = busy_before || runner_busy();
        let mut out =
            std::fs::OpenOptions::new().create(true).append(true).open(&file).expect("results file");
        for mut r in records {
            use std::io::Write;
            if busy {
                r["runner_busy"] = json!(true);
            }
            writeln!(out, "{r}").expect("write");
        }
    }
    eprintln!("appended to {}", file.display());
}

#[cfg(target_os = "macos")]
fn gpu_runs(w: &SpinWalk, steps: usize, p_ref: &[f64], record: &mut impl FnMut(&str, usize, f64, Value)) {
    use caroli_gpu::{MetalDevice, WalkTables};
    let Ok(device) = MetalDevice::new() else { return };
    let graph = w.graph();
    let node_rows: Vec<usize> =
        (0..graph.nodes()).map(|x| graph.half_edges_of(x).start).chain([graph.half_edges()]).collect();
    let coins: Vec<Vec<c64>> = w
        .coins()
        .iter()
        .map(|c| (0..c.nrows()).flat_map(|i| (0..c.ncols()).map(move |j| c[(i, j)])).collect())
        .collect();
    let halves: Vec<(usize, usize)> = (0..graph.edges()).map(|e| graph.halves(e)).collect();
    let (p, a, b) = w.exchange();
    let tables = WalkTables {
        bits: graph.edges(),
        node_rows: &node_rows,
        coins: &coins,
        halves: &halves,
        exchange: [p, a, b],
    };
    let px = |rows: &[f64]| -> Vec<f64> { rows.chunks(2).map(|r| r[0] + r[1]).collect() };
    for (name, precision) in
        [("caroli-gpu-f32", Precision::Single), ("caroli-gpu-df64", Precision::DoubleSingle)]
    {
        // A double-single state is as large as the f64 one: with the f64 copy needed to upload
        // it, the largest rings would not fit in memory.
        if precision == Precision::DoubleSingle && w.dimension() > 1 << 30 {
            continue;
        }
        let Ok(mut gw) = device.walk(&tables, &initial(w), precision) else { continue };
        gw.steps(1).expect("warm-up");
        let start = Instant::now();
        gw.steps(steps).expect("steps");
        let seconds = start.elapsed().as_secs_f64();
        let dp = max_diff(&px(&gw.row_probabilities(graph.half_edges())), p_ref);
        record(name, 0, seconds, json!({"device": device.name(), "max_dp": dp}));
    }
    // The checked mode: calibration on the CPU, then f32 on the GPU, norm checked every 100 steps.
    let long = 50 * steps;
    let mut psi = initial(w);
    let start = Instant::now();
    let report = evolve(w, &mut psi, long, &Options::exploration(100), |_, _| {});
    let seconds = start.elapsed().as_secs_f64() * steps as f64 / long as f64;
    record(
        "caroli-gpu-f32-checked",
        0,
        seconds,
        json!({"device": device.name(), "run_steps": long, "ran_on": report.ran_on,
               "predicted_error": report.predicted_error, "norm_drift": report.norm_drift,
               "cpu_steps": report.cpu_steps, "gpu_steps": report.gpu_steps}),
    );
}

/// Markdown table of the latest record of each (n, method, threads).
fn report(args: &[String]) {
    let file = args.first().map(PathBuf::from).unwrap_or_else(|| {
        root().join("benchmarks/results").join(format!("walk-{}.jsonl", shell("hostname", &["-s"])))
    });
    let text = std::fs::read_to_string(&file).expect("results file");
    let mut latest: BTreeMap<(u64, String, u64), Value> = BTreeMap::new();
    for line in text.lines() {
        let v: Value = serde_json::from_str(line).expect("JSON line");
        let key = (
            v["n"].as_u64().unwrap(),
            v["method"].as_str().unwrap().to_string(),
            v["threads"].as_u64().unwrap(),
        );
        latest.insert(key, v);
    }
    let sizes: Vec<u64> =
        latest.keys().map(|k| k.0).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    let label = |m: &str, t: u64| match (m, t) {
        ("caroli-cpu", 1) => "caroli CPU, 1 thread".to_string(),
        ("caroli-cpu", t) => format!("caroli CPU, {t} threads"),
        ("caroli-csr", _) => "caroli, U as a CSR matrix".into(),
        ("caroli-gpu-f32", _) => "caroli GPU f32".into(),
        ("caroli-gpu-df64", _) => "caroli GPU double-single".into(),
        ("caroli-gpu-f32-checked", _) => "caroli GPU f32, checked".into(),
        ("python-sparse", _) => "Python, U as a SciPy sparse matrix".into(),
        ("python-numpy", _) => "Python, vectorised NumPy".into(),
        (m, _) => m.into(),
    };
    println!("Time per step (ms); in brackets, the speed-up over Python with U as a SciPy sparse matrix.\n");
    print!("| method |");
    for n in &sizes {
        print!(" n = {n} |");
    }
    println!();
    println!("|---|{}", "---|".repeat(sizes.len()));
    let mut methods: Vec<(String, u64)> = latest.keys().map(|k| (k.1.clone(), k.2)).collect();
    methods.sort();
    methods.dedup();
    for (m, t) in methods {
        print!("| {} |", label(&m, t));
        for &n in &sizes {
            match latest.get(&(n, m.clone(), t)) {
                Some(v) => {
                    let ms = v["s_per_step"].as_f64().unwrap() * 1e3;
                    let base = latest
                        .iter()
                        .find(|(k, _)| k.0 == n && k.1 == "python-sparse")
                        .map(|(_, b)| b["s_per_step"].as_f64().unwrap() * 1e3);
                    match base {
                        Some(b) if m != "python-sparse" => print!(" {ms:.3} ({:.0}×) |", b / ms),
                        _ => print!(" {ms:.3} |"),
                    }
                }
                None => print!(" — |"),
            }
        }
        println!();
    }
    let dims: Vec<String> = sizes
        .iter()
        .filter_map(|n| {
            latest.iter().find(|(k, _)| k.0 == *n).map(|(_, v)| format!("n = {n}: {}", v["dimension"]))
        })
        .collect();
    println!("\nDimensions: {}.", dims.join(", "));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("run") => run(&args[2..]),
        Some("report") => report(&args[2..]),
        _ => eprintln!(
            "usage: walk run [--sizes 13,15,17,19] [--steps 10] [--python PATH] | walk report [FILE]"
        ),
    }
}
