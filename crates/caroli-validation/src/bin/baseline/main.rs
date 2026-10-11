//! Builds the thesis reference codes and records or checks their golden outputs.
//!
//! ```text
//! baseline deps                         serial MPI and FFTW shims, LAPACK95
//! baseline build [--check] [PROGRAM…]   compile programs (--check: -O0 + bounds checking)
//! baseline run [--update|--check] [CASE…]
//!                                       run cases and compare with golden/ (--update: record)
//! baseline show FILE…                   records of a Fortran output file
//! baseline list                         programs and cases
//! baseline coverage [--orig PATH] [--manifest]
//!                                       check that reference/thesis covers the original tree
//! ```
//!
//! Run it with `cargo run -p caroli-validation --bin baseline -- <command>`. The sources in
//! reference/thesis are compiled as they are, with a few build-time patches (see `compile.rs`).
//! Products go to target/reference/. Environment overrides: FC, CC, LAPACK95_SRC, FFTW_PREFIX,
//! LAPACK_LIBS.

mod cases;
mod compile;
mod coverage;

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else {
        eprintln!("usage: baseline <deps|build|run|show|list|coverage> [args]");
        return ExitCode::from(2);
    };
    let rest: Vec<&str> = args[1..].iter().map(String::as_str).collect();
    let flag = |f: &str| rest.contains(&f);
    let names: Vec<&str> = rest.iter().copied().filter(|a| !a.starts_with("--")).collect();
    let ok = match cmd.as_str() {
        "deps" => compile::deps(),
        "build" => compile::programs(&names, flag("--check")),
        "run" => {
            if flag("--update") && flag("--check") {
                eprintln!("--update and --check are exclusive: golden outputs come from the optimised build");
                return ExitCode::from(2);
            }
            cases::run(&names, flag("--update"), flag("--check"))
        }
        "show" => {
            names.iter().for_each(|f| cases::show(f));
            true
        }
        "list" => {
            cases::list();
            true
        }
        "coverage" => {
            let orig = rest.iter().position(|a| *a == "--orig").and_then(|i| rest.get(i + 1)).copied();
            coverage::run(orig, flag("--manifest"))
        }
        other => {
            eprintln!("unknown command {other}");
            return ExitCode::from(2);
        }
    };
    if ok { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}
