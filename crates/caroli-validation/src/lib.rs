//! Validation of caroli against the thesis reference codes.
//!
//! The golden outputs live in `reference/baseline/golden/<case>/`; they are the files written by
//! the thesis programs (see `reference/baseline/README.md`). This crate reads them; the
//! comparisons are its integration tests (`tests/`).

use std::path::{Path, PathBuf};

/// Directory of the golden outputs of a baseline case.
pub fn golden_dir(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference/baseline/golden").join(case)
}

/// Records of a gfortran sequential unformatted file (int32 length markers around each record).
pub fn read_records(path: impl AsRef<Path>) -> std::io::Result<Vec<Vec<u8>>> {
    let data = std::fs::read(path.as_ref())?;
    let bad = || {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{}: not a sequential unformatted file", path.as_ref().display()),
        )
    };
    let mut out = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let len = u32::from_le_bytes(data.get(i..i + 4).ok_or_else(bad)?.try_into().unwrap()) as usize;
        let payload = data.get(i + 4..i + 4 + len).ok_or_else(bad)?;
        let end = u32::from_le_bytes(data.get(i + 4 + len..i + 8 + len).ok_or_else(bad)?.try_into().unwrap())
            as usize;
        if end != len {
            return Err(bad());
        }
        out.push(payload.to_vec());
        i += 8 + len;
    }
    Ok(out)
}

/// A record read as little-endian `f64` values.
pub fn as_f64(record: &[u8]) -> Vec<f64> {
    record.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
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
