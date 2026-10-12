//! Time per step of the walk on rings of `n` nodes (dimension `n · 2^(n+1)`): the structured CPU
//! sweeps (f64, all cores), the one-step operator as a sparse matrix (CSR, f64, all cores), and
//! the GPU in f32 and double-single. One JSON line per size and method.
//!
//! ```text
//! cargo run --release -p caroli-walk --example timing -- [n...] [--steps S] [--no-csr] [--no-gpu]
//! ```

use caroli_model::c64;
use caroli_walk::{PLUS, Precision, SpinWalk};
use std::time::Instant;

fn main() {
    let mut sizes = Vec::new();
    let (mut steps, mut csr, mut gpu) = (20usize, true, true);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--steps" => steps = args.next().and_then(|s| s.parse().ok()).expect("--steps S"),
            "--no-csr" => csr = false,
            "--no-gpu" => gpu = false,
            n => sizes.push(n.parse::<usize>().expect("ring size")),
        }
    }
    if sizes.is_empty() {
        sizes = vec![13, 15, 17, 19];
    }
    let threads = rayon::current_num_threads();
    for n in sizes {
        let walk = SpinWalk::ring(n, std::f64::consts::PI / 8.0, 0.2);
        let dim = walk.dimension();
        let line = |method: &str, seconds: f64, extra: &str| {
            println!(
                "{{\"n\": {n}, \"dimension\": {dim}, \"method\": \"{method}\", \"threads\": {threads}, \
                 \"steps\": {steps}, \"s_per_step\": {:.6e}{extra}}}",
                seconds / steps as f64
            );
        };
        let mut psi = walk.product_state(&walk.localized(0, 0), &vec![PLUS; n]);
        walk.steps(&mut psi, 2); // warm-up
        let start = Instant::now();
        walk.steps(&mut psi, steps);
        line("cpu", start.elapsed().as_secs_f64(), "");
        if csr && dim <= 1 << 26 {
            let built = Instant::now();
            let u = walk.matrix();
            let build = built.elapsed().as_secs_f64();
            let mut next = vec![c64::new(0.0, 0.0); dim];
            let start = Instant::now();
            for _ in 0..steps {
                u.apply(&psi, &mut next);
                std::mem::swap(&mut psi, &mut next);
            }
            line("csr", start.elapsed().as_secs_f64(), &format!(", \"build_s\": {build:.3}"));
        }
        #[cfg(target_os = "macos")]
        if gpu {
            run_gpu(&walk, &psi, steps, &line);
        }
    }
}

#[cfg(target_os = "macos")]
fn run_gpu(walk: &SpinWalk, psi: &[c64], steps: usize, line: &dyn Fn(&str, f64, &str)) {
    use caroli_gpu::{MetalDevice, WalkTables};
    let device = MetalDevice::new().expect("Metal device");
    let graph = walk.graph();
    let node_rows: Vec<usize> =
        (0..graph.nodes()).map(|x| graph.half_edges_of(x).start).chain([graph.half_edges()]).collect();
    let coins: Vec<Vec<c64>> = walk
        .coins()
        .iter()
        .map(|c| (0..c.nrows()).flat_map(|i| (0..c.ncols()).map(move |j| c[(i, j)])).collect())
        .collect();
    let halves: Vec<(usize, usize)> = (0..graph.edges()).map(|e| graph.halves(e)).collect();
    let (p, a, b) = walk.exchange();
    let tables = WalkTables {
        bits: graph.edges(),
        node_rows: &node_rows,
        coins: &coins,
        halves: &halves,
        exchange: [p, a, b],
    };
    for (name, precision) in [("gpu_f32", Precision::Single), ("gpu_df64", Precision::DoubleSingle)] {
        match device.walk(&tables, psi, precision) {
            Ok(mut w) => {
                w.steps(2).expect("warm-up");
                let start = Instant::now();
                w.steps(steps).expect("steps");
                line(name, start.elapsed().as_secs_f64(), &format!(", \"device\": \"{}\"", device.name()));
            }
            Err(e) => eprintln!("{name}: {e}"),
        }
    }
}
