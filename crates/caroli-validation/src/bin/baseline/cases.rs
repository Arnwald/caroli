//! Running the baseline cases and comparing their outputs with the golden ones.

use crate::compile::{PROGRAMS, repo, target};
use caroli_validation::fortran::{as_f64, compare, records};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// `cases/<name>/case.toml`.
#[derive(Deserialize)]
struct Case {
    program: String,
    #[serde(default)]
    description: String,
    #[serde(default = "default_rtol")]
    rtol: f64,
    #[serde(default = "default_atol")]
    atol: f64,
    #[serde(default = "default_timeout")]
    timeout: u64,
}

fn default_rtol() -> f64 {
    1e-8
}
fn default_atol() -> f64 {
    1e-12
}
fn default_timeout() -> u64 {
    300
}

fn cases_dir() -> PathBuf {
    repo().join("reference/baseline/cases")
}

fn golden_dir() -> PathBuf {
    repo().join("reference/baseline/golden")
}

fn all_cases() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(cases_dir())
        .map(|d| {
            d.filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .filter_map(|e| e.file_name().into_string().ok())
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names
}

fn load(name: &str) -> Result<Case, String> {
    let path = cases_dir().join(name).join("case.toml");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn sorted_files(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|d| d.filter_map(|e| Some(e.ok()?.path())).filter(|p| p.is_file()).collect())
        .unwrap_or_default();
    v.sort();
    v
}

fn name_of(p: &Path) -> String {
    p.file_name().unwrap().to_string_lossy().into_owned()
}

/// Runs the cases (all if `names` is empty); records the golden outputs with `update`.
pub fn run(names: &[&str], update: bool, check: bool) -> bool {
    let names: Vec<String> =
        if names.is_empty() { all_cases() } else { names.iter().map(|s| s.to_string()).collect() };
    let mut failures = 0;
    for name in &names {
        if let Err(e) = run_case(name, update, check) {
            println!("FAIL  {name}: {e}");
            failures += 1;
        }
    }
    println!("{}/{} cases passed", names.len() - failures, names.len());
    failures == 0
}

fn run_case(name: &str, update: bool, check: bool) -> Result<(), String> {
    let case = load(name)?;
    let exe = target().join(if check { "bin-check" } else { "bin" }).join(&case.program);
    if !exe.exists() {
        return Err(format!("missing executable {} (run: baseline build)", case.program));
    }
    let work = std::env::temp_dir().join(format!("caroli-baseline-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).map_err(|e| e.to_string())?;
    let inputs: Vec<PathBuf> =
        sorted_files(&cases_dir().join(name)).into_iter().filter(|p| name_of(p) != "case.toml").collect();
    for p in &inputs {
        std::fs::copy(p, work.join(name_of(p))).map_err(|e| e.to_string())?;
    }
    let start = Instant::now();
    let mut child = Command::new(&exe)
        .current_dir(&work)
        .env("OMP_NUM_THREADS", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;
    let status = loop {
        if let Some(s) = child.try_wait().map_err(|e| e.to_string())? {
            break s;
        }
        if start.elapsed() > Duration::from_secs(case.timeout) {
            let _ = child.kill();
            return Err(format!("timeout after {} s", case.timeout));
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let dt = start.elapsed().as_secs_f64();
    let output = child.wait_with_output().map_err(|e| e.to_string())?;
    if !status.success() {
        let text =
            String::from_utf8_lossy(&output.stdout).into_owned() + &String::from_utf8_lossy(&output.stderr);
        let tail: Vec<&str> = text.lines().rev().take(6).collect::<Vec<_>>().into_iter().rev().collect();
        return Err(format!("{status} after {dt:.1} s\n      {}", tail.join("\n      ")));
    }
    let input_names: Vec<String> = inputs.iter().map(|p| name_of(p)).collect();
    let produced: Vec<PathBuf> =
        sorted_files(&work).into_iter().filter(|p| !input_names.contains(&name_of(p))).collect();
    let gdir = golden_dir().join(name);
    let result = if update {
        let _ = std::fs::remove_dir_all(&gdir);
        std::fs::create_dir_all(&gdir).map_err(|e| e.to_string())?;
        let mut size = 0;
        for p in &produced {
            size += std::fs::copy(p, gdir.join(name_of(p))).map_err(|e| e.to_string())?;
        }
        println!("rec   {name:34} {} files, {} KiB, {dt:.1} s", produced.len(), size / 1024);
        Ok(())
    } else {
        let want: Vec<String> = sorted_files(&gdir).iter().map(|p| name_of(p)).collect();
        let got: Vec<String> = produced.iter().map(|p| name_of(p)).collect();
        if want != got {
            Err(format!("files differ: produced {got:?}, golden {want:?}"))
        } else {
            produced
                .iter()
                .find_map(|p| {
                    let (a, b) = (std::fs::read(p).ok()?, std::fs::read(gdir.join(name_of(p))).ok()?);
                    compare(&a, &b, case.rtol, case.atol).map(|d| format!("{}: {d}", name_of(p)))
                })
                .map_or(Ok(()), Err)
                .inspect(|_| println!("ok    {name:34} {} files, {dt:.1} s", produced.len()))
        }
    };
    let _ = std::fs::remove_dir_all(&work);
    result
}

/// Prints the records of a Fortran output file.
pub fn show(path: &str) {
    let Ok(data) = std::fs::read(path) else {
        println!("{path}: cannot read");
        return;
    };
    match records(&data) {
        None => println!("{path}: text or not a sequential unformatted file ({} bytes)", data.len()),
        Some(recs) => {
            println!("{path}: {} record(s)", recs.len());
            for (k, r) in recs.iter().enumerate() {
                if r.len() % 8 == 0 && !r.is_empty() {
                    let v = as_f64(r);
                    let first: Vec<String> = v.iter().take(4).map(|x| format!("{x:.6}")).collect();
                    println!("  [{k}] {} bytes = {} float64, first {}", r.len(), v.len(), first.join(", "));
                } else {
                    println!("  [{k}] {} bytes", r.len());
                }
            }
        }
    }
}

/// Lists the programs and cases.
pub fn list() {
    for (id, dir) in PROGRAMS {
        println!("program {id:34} {dir}");
    }
    for name in all_cases() {
        match load(&name) {
            Ok(c) => println!("case    {name:34} {}: {}", c.program, c.description),
            Err(e) => println!("case    {name:34} {e}"),
        }
    }
}
