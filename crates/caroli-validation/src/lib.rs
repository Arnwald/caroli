//! Validation of caroli against the thesis reference codes.
//!
//! - The `baseline` binary builds the thesis programs (`reference/thesis`) and records or checks
//!   their golden outputs (`reference/baseline/golden`); see `reference/baseline/README.md`.
//! - The integration tests (`tests/`) compare caroli with those golden outputs.

pub mod fortran;

pub use fortran::{as_f64, read_records};
use std::path::{Path, PathBuf};

/// Root of the repository.
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("repository root")
}

/// Directory of the golden outputs of a baseline case.
pub fn golden_dir(case: &str) -> PathBuf {
    repo_root().join("reference/baseline/golden").join(case)
}

/// All `f64` records of a golden file of a case.
pub fn golden_f64(case: &str, file: &str) -> Vec<Vec<f64>> {
    let path = golden_dir(case).join(file);
    read_records(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
        .iter()
        .map(|r| as_f64(r))
        .collect()
}
pub mod thesis;
