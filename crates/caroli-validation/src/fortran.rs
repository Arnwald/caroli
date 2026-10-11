//! Fortran sequential unformatted files and tolerant comparison of program outputs.

use regex::Regex;
use std::path::Path;
use std::sync::LazyLock;

/// Records of gfortran sequential unformatted data (int32 length markers around each record),
/// or `None` if the bytes do not have that structure.
pub fn records(data: &[u8]) -> Option<Vec<&[u8]>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let len = i32::from_le_bytes(data.get(i..i + 4)?.try_into().ok()?);
        let len = usize::try_from(len).ok()?;
        let payload = data.get(i + 4..i + 4 + len)?;
        let end = i32::from_le_bytes(data.get(i + 4 + len..i + 8 + len)?.try_into().ok()?);
        if usize::try_from(end).ok()? != len {
            return None;
        }
        out.push(payload);
        i += 8 + len;
    }
    Some(out)
}

/// Records of a sequential unformatted file.
pub fn read_records(path: impl AsRef<Path>) -> std::io::Result<Vec<Vec<u8>>> {
    let path = path.as_ref();
    let data = std::fs::read(path)?;
    records(&data).map(|r| r.into_iter().map(<[u8]>::to_vec).collect()).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{}: not a sequential unformatted file", path.display()),
        )
    })
}

/// A record read as little-endian `f64` values.
pub fn as_f64(record: &[u8]) -> Vec<f64> {
    record.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}

/// Lines of text outputs that change from run to run (timings written by the programs).
static VOLATILE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\bCPU\b|CPU time|elapsed").unwrap());

fn close(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan();
    }
    (a - b).abs() <= atol + rtol * a.abs().max(b.abs())
}

/// `None` if two program outputs are equivalent, else a short description of the first
/// difference. Unformatted files are compared record by record as `f64` within tolerance; text
/// files token by token (numbers within tolerance), skipping timing lines.
pub fn compare(new: &[u8], reference: &[u8], rtol: f64, atol: f64) -> Option<String> {
    if new == reference {
        return None;
    }
    if let (Some(ra), Some(rb)) = (records(new), records(reference)) {
        if ra.len() != rb.len() {
            return Some(format!("{} records vs {}", ra.len(), rb.len()));
        }
        for (k, (x, y)) in ra.iter().zip(&rb).enumerate() {
            if x.len() != y.len() {
                return Some(format!("record {k}: {} bytes vs {}", x.len(), y.len()));
            }
            if x.len() % 8 != 0 {
                if x != y {
                    return Some(format!("record {k}: bytes differ"));
                }
                continue;
            }
            for (j, (u, v)) in as_f64(x).into_iter().zip(as_f64(y)).enumerate() {
                if !close(u, v, rtol, atol) {
                    return Some(format!("record {k}, value {j}: {u:e} vs {v:e}"));
                }
            }
        }
        return None;
    }
    let tokens = |data: &[u8]| -> Vec<String> {
        String::from_utf8_lossy(data)
            .lines()
            .filter(|l| !VOLATILE.is_match(l))
            .flat_map(|l| l.split_whitespace().map(str::to_owned).collect::<Vec<_>>())
            .collect()
    };
    let (ta, tb) = (tokens(new), tokens(reference));
    if ta.len() != tb.len() {
        return Some(format!("{} tokens vs {}", ta.len(), tb.len()));
    }
    let number = |s: &str| s.trim_end_matches(',').replace(['D', 'd'], "e").parse::<f64>().ok();
    for (u, v) in ta.iter().zip(&tb) {
        if u == v {
            continue;
        }
        match (number(u), number(v)) {
            (Some(a), Some(b)) if close(a, b, rtol, atol) => {}
            (Some(_), Some(_)) => return Some(format!("value {u} vs {v}")),
            _ => return Some(format!("token {u:?} vs {v:?}")),
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(values: &[f64]) -> Vec<u8> {
        let n = (8 * values.len()) as i32;
        let mut out = n.to_le_bytes().to_vec();
        values.iter().for_each(|v| out.extend(v.to_le_bytes()));
        out.extend(n.to_le_bytes());
        out
    }

    #[test]
    fn records_and_tolerance() {
        let a = record(&[1.0, 2.0]);
        let b = record(&[1.0, 2.0 + 1e-12]);
        assert_eq!(records(&a).unwrap().len(), 1);
        assert!(compare(&a, &b, 1e-8, 1e-12).is_none());
        assert!(compare(&a, &record(&[1.0, 2.1]), 1e-8, 1e-12).is_some());
        assert!(records(&a[..10]).is_none());
    }

    #[test]
    fn text_skips_timings() {
        let a = b"&PARAM N= 16, /\nN_imp = 25\n  CPU :    5.0E-03\n";
        let b = b"&PARAM N= 16, /\nN_imp = 25\n  CPU :    1.0E-03\n";
        assert!(compare(a, b, 1e-8, 1e-12).is_none());
        assert!(compare(a, b"&PARAM N= 17, /\nN_imp = 25\n", 1e-8, 1e-12).is_some());
    }
}
