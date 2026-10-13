//! `caroli run INPUT.toml [--out DIR] [--set NAME=VALUE]... [--only K,...]`
//! `caroli check INPUT.toml [--set NAME=VALUE]...`

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "\
caroli — tight-binding systems and their simulations from a TOML file

usage:
  caroli run INPUT.toml [--out DIR] [--set NAME=VALUE]... [--only K,...]
      runs the [[run]] tables of INPUT.toml and writes one CSV per run in DIR
      (default: INPUT next to the file, without the .toml extension)
  caroli check INPUT.toml [--set NAME=VALUE]...
      reads the input, builds the model and the sample, and describes them

  --set NAME=VALUE   overrides a parameter of [parameters]
  --only K,...       runs only these runs (0 is the first)
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match main_inner(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn main_inner(args: &[String]) -> Result<(), caroli_cli::Error> {
    let command = args.first().map(String::as_str);
    if matches!(command, None | Some("help" | "--help" | "-h")) {
        print!("{USAGE}");
        return Ok(());
    }
    if command == Some("--version") {
        println!("caroli {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    let file = PathBuf::from(args.get(1).ok_or("missing INPUT.toml")?);
    let (mut out, mut overrides, mut only) = (None, BTreeMap::new(), None);
    let mut rest = args[2..].iter();
    while let Some(a) = rest.next() {
        match a.as_str() {
            "--out" => out = Some(PathBuf::from(rest.next().ok_or("--out DIR")?)),
            "--set" => {
                let s = rest.next().ok_or("--set NAME=VALUE")?;
                let (name, value) = s.split_once('=').ok_or("--set NAME=VALUE")?;
                let v: f64 = value.parse().map_err(|_| format!("--set {s}: not a number"))?;
                overrides.insert(name.to_string(), v);
            }
            "--only" => {
                let list = rest.next().ok_or("--only K,...")?;
                let ks: Vec<usize> = list.split(',').map(|k| k.trim().parse()).collect::<Result<_, _>>()?;
                only = Some(ks);
            }
            other => return Err(format!("unknown option `{other}`\n\n{USAGE}").into()),
        }
    }
    let text = std::fs::read_to_string(&file).map_err(|e| format!("{}: {e}", file.display()))?;
    let input = caroli_cli::parse(&text).map_err(|e| format!("{}: {e}", file.display()))?;
    match command {
        Some("check") => print!("{}", caroli_cli::describe(&input, &overrides)?),
        Some("run") => {
            let out = out.unwrap_or_else(|| file.with_extension(""));
            for line in caroli_cli::run_all(&input, &out, &overrides, only.as_deref())? {
                println!("{line}");
            }
        }
        Some(other) => return Err(format!("unknown command `{other}`\n\n{USAGE}").into()),
        None => unreachable!(),
    }
    Ok(())
}
