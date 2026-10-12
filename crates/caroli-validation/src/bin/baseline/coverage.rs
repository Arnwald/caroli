//! Checks that the thesis reference (`THESIS_SRC`) covers all the thesis code of the original
//! tree, and rewrites its MANIFEST.tsv.
//!
//! A code file of the original tree (`~/Projects/electronic-transport`, excluding the
//! `thesis-code` link and LAPACK95) is covered if its content (sha1) exists in
//! the thesis reference, if it is listed in EXCLUDED.tsv (per-run copy or off-topic), or if it is
//! third-party code kept out of the repository for want of a license (THIRD_PARTY.tsv).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const CODE_EXT: &[&str] = &[
    "f90", "f", "F90", "c", "py", "sh", "nb", "ipynb", "m", "ma", "jl", "asy", "gp", "plt", "pbs", "slurm",
    "job",
];
/// Documentation files of the thesis reference itself.
const OWN_FILES: &[&str] = &["README.md", "CLAUDE.md", "MANIFEST.tsv", "EXCLUDED.tsv", "THIRD_PARTY.tsv"];

pub(crate) fn sha1(path: &Path) -> String {
    sha1_smol::Sha1::from(std::fs::read(path).unwrap_or_default()).digest().to_string()
}

fn is_code(name: &str) -> bool {
    name.to_lowercase().starts_with("makefile")
        || Path::new(name).extension().and_then(|e| e.to_str()).is_some_and(|e| CODE_EXT.contains(&e))
}

/// Files under `root`, skipping directories for which `skip` is true (symlinks not followed).
fn walk(root: &Path, skip: &dyn Fn(&Path) -> bool, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else { return };
    for e in entries.flatten() {
        let (path, Ok(kind)) = (e.path(), e.file_type()) else { continue };
        if kind.is_dir() {
            if !skip(&path) {
                walk(&path, skip, out);
            }
        } else if kind.is_file() {
            out.push(path);
        }
    }
}

pub fn run(orig: Option<&str>, manifest: bool) -> bool {
    let here = crate::compile::thesis_root();
    let orig = PathBuf::from(match orig {
        Some(p) => p.to_owned(),
        None => std::env::var("THESIS_ORIG").unwrap_or_else(|_| {
            format!("{}/Projects/electronic-transport", std::env::var("HOME").unwrap_or_default())
        }),
    });
    if !orig.join("TI").is_dir() {
        println!("original tree not found at {}: pass --orig PATH or set THESIS_ORIG", orig.display());
        return false;
    }

    // Original code: sha1 -> paths relative to the original tree.
    let mut files = Vec::new();
    let skip_orig = |p: &Path| {
        let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        matches!(n, ".git" | ".ipynb_checkpoints" | "LAPACK95")
    };
    walk(&orig, &skip_orig, &mut files);
    let mut by_hash: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for f in files.iter().filter(|f| is_code(&f.file_name().unwrap().to_string_lossy())) {
        let rel = format!("./{}", f.strip_prefix(&orig).unwrap().display());
        by_hash.entry(sha1(f)).or_default().push(rel);
    }

    // The reference's contents (without its own documentation and tooling).
    let mut own = Vec::new();
    let skip_here = |p: &Path| {
        let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        matches!(n, ".manifest" | "tools" | "__pycache__")
    };
    walk(&here, &skip_here, &mut own);
    let mut present: BTreeMap<String, String> = BTreeMap::new(); // rel path -> sha1
    for f in own {
        let rel = f.strip_prefix(&here).unwrap().display().to_string();
        let name = f.file_name().unwrap().to_string_lossy().into_owned();
        if OWN_FILES.contains(&rel.as_str()) || name == "README.md" {
            continue;
        }
        present.insert(rel, sha1(&f));
    }
    let have: BTreeSet<&String> = present.values().collect();
    let excluded: BTreeSet<String> = std::fs::read_to_string(here.join("EXCLUDED.tsv"))
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| l.split('\t').next().map(str::to_owned))
        .collect();

    let third_party: BTreeSet<String> = std::fs::read_to_string(here.join("THIRD_PARTY.tsv"))
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| l.split('\t').nth(1).map(str::to_owned))
        .collect();

    let missing: Vec<&Vec<String>> = by_hash
        .iter()
        .filter(|(h, paths)| {
            !have.contains(h) && !third_party.contains(*h) && !paths.iter().any(|p| excluded.contains(p))
        })
        .map(|(_, p)| p)
        .collect();
    println!(
        "original code : {} files, {} distinct contents",
        by_hash.values().map(Vec::len).sum::<usize>(),
        by_hash.len()
    );
    println!(
        "thesis reference: {} files; documented exclusions: {}; third-party contents kept out: {}",
        present.len(),
        excluded.len(),
        third_party.len()
    );
    println!("not covered   : {}", missing.len());
    for paths in &missing {
        let first = paths.iter().min().unwrap();
        if paths.len() > 1 {
            println!("    {first} (+{} copies)", paths.len() - 1);
        } else {
            println!("    {first}");
        }
    }

    if manifest {
        // Raw provenance written during the sorting: .manifest/<domain>.tsv (dest \t source).
        let mut sources: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut manifests: Vec<PathBuf> = std::fs::read_dir(here.join(".manifest"))
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .collect();
        manifests.sort();
        for m in manifests {
            for line in std::fs::read_to_string(&m).unwrap_or_default().lines() {
                if let Some((d, s)) = line.split_once('\t') {
                    let v = sources.entry(d.to_owned()).or_default();
                    if !v.iter().any(|x| x == s) {
                        v.push(s.to_owned());
                    }
                }
            }
        }
        let mut out = String::from(
            "# file in the thesis reference\tsha1\torigin (copied from)\tidentical copies elsewhere in electronic-transport\n",
        );
        for (rel, h) in &present {
            let srcs = sources.get(rel).cloned().unwrap_or_default();
            let mut same: Vec<&String> =
                by_hash.get(h).map(|v| v.iter().filter(|p| !srcs.contains(p)).collect()).unwrap_or_default();
            same.sort();
            let same: Vec<&str> = same.iter().map(|s| s.as_str()).collect();
            out.push_str(&format!("{rel}\t{h}\t{}\t{}\n", srcs.join(" | "), same.join(" | ")));
        }
        std::fs::write(here.join("MANIFEST.tsv"), out).expect("write MANIFEST.tsv");
        println!("MANIFEST.tsv rewritten");
    }
    missing.is_empty()
}
