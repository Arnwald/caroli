//! The `caroli` command: a system and its simulations described in a TOML file.
//!
//! ```toml
//! title = "Clean BHZ ribbon"
//!
//! [parameters]
//! M = -0.137
//!
//! [model]
//! preset = "bhz"
//! mass = "M"
//!
//! [system]
//! size = [40, 20]
//!
//! [[run]]
//! type = "transmission"
//! energies = { from = -0.3, to = 0.3, count = 61 }
//! ```
//!
//! `caroli run input.toml` writes one CSV table per `[[run]]` in an output directory. The input
//! format is described in `crates/caroli-cli/README.md`, with commented examples in
//! `crates/caroli-cli/inputs/`.

pub mod build;
pub mod input;
pub mod run;

use std::collections::BTreeMap;
use std::path::Path;

pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// Parses an input file.
pub fn parse(text: &str) -> Result<input::Input, Error> {
    let input: input::Input = toml::from_str(text)?;
    if input.runs.is_empty() {
        return Err("no [[run]] in the input".into());
    }
    Ok(input)
}

/// A short description of the system: lattice, sample, leads, runs.
pub fn describe(input: &input::Input, overrides: &BTreeMap<String, f64>) -> Result<String, Error> {
    let p = build::Params::resolve(&input.parameters, overrides)?;
    let model = build::model(&input.model, &p)?;
    let lattice = model.lattice();
    let mut out = String::new();
    if let Some(t) = &input.title {
        out += &format!("{t}\n");
    }
    out += &format!(
        "lattice: dimension {}, {} sublattice(s) with {:?} orbitals\n",
        lattice.dim(),
        lattice.sublattices().len(),
        lattice.sublattices().iter().map(|s| s.orbitals).collect::<Vec<_>>()
    );
    out += &format!("parameters: {:?}\n", p.0);
    if let Some(size) = &input.system.size {
        let region = build::region(&model, &input.system)?;
        // Builds the disorder too, to report its errors.
        build::disordered(input, &p, &region, 0)?;
        out += &format!(
            "sample: {:?} cells, {} sites, {} orbitals{}\n",
            size,
            region.len(),
            region.orbitals(),
            if input.system.periodic { ", periodic" } else { "" }
        );
    }
    out += &format!("disorder: {} term(s)\n", input.disorder.len());
    for (k, r) in input.runs.iter().enumerate() {
        out += &format!("run {k}: {}\n", output_name(k, r));
    }
    Ok(out)
}

/// Name of the table of run `k`.
pub fn output_name(k: usize, r: &input::RunInput) -> String {
    use input::RunInput::*;
    let (kind, output) = match r {
        Transmission(x) => ("transmission", &x.output),
        Local(x) => ("local", &x.output),
        Bands(x) => ("bands", &x.output),
        Spectrum(x) => ("spectrum", &x.output),
        Dos(x) => ("dos", &x.output),
        Conductivity(x) => ("conductivity", &x.output),
        Scba(x) => ("scba", &x.output),
        Evolution(x) => ("evolution", &x.output),
    };
    output.clone().unwrap_or_else(|| format!("{k:02}-{kind}.csv"))
}

/// Runs the calculations of `input` (all, or those of `only`), writing their tables in
/// `out_dir`. Returns a summary line per calculation.
pub fn run_all(
    input: &input::Input,
    out_dir: &Path,
    overrides: &BTreeMap<String, f64>,
    only: Option<&[usize]>,
) -> Result<Vec<String>, Error> {
    std::fs::create_dir_all(out_dir)?;
    let mut lines = Vec::new();
    for (k, r) in input.runs.iter().enumerate() {
        if only.is_some_and(|o| !o.contains(&k)) {
            continue;
        }
        let t = std::time::Instant::now();
        let table = run::run(input, r, overrides).map_err(|e| format!("run {k}: {e}"))?;
        let path = out_dir.join(output_name(k, r));
        std::fs::write(&path, &table.csv)?;
        lines.push(format!("{} → {} ({:.1?})", table.summary, path.display(), t.elapsed()));
    }
    Ok(lines)
}
