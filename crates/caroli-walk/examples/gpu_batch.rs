//! Throughput of the GPU on batches of small walks (ring of n nodes, f32, with the half-edge
//! probabilities accumulated at every step), against the batch size.
//!
//! ```text
//! cargo run --release -p caroli-walk --example gpu_batch -- [n] [steps]
//! ```

fn main() {
    #[cfg(target_os = "macos")]
    {
        use caroli_gpu::{MetalDevice, Precision, WalkTables};
        use caroli_model::c64;
        use caroli_walk::{PLUS, SpinWalk};
        let mut args = std::env::args().skip(1);
        let n: usize = args.next().map_or(13, |s| s.parse().expect("n"));
        let steps: usize = args.next().map_or(500, |s| s.parse().expect("steps"));
        let device = MetalDevice::new().expect("Metal");
        for batch in [1usize, 2, 4, 8, 16, 32] {
            let walks: Vec<SpinWalk> =
                (0..batch).map(|k| SpinWalk::ring(n, 0.2 + 0.01 * k as f64, 0.7)).collect();
            let graph = walks[0].graph();
            let node_rows: Vec<usize> = (0..graph.nodes())
                .map(|x| graph.half_edges_of(x).start)
                .chain([graph.half_edges()])
                .collect();
            let halves: Vec<(usize, usize)> = (0..graph.edges()).map(|e| graph.halves(e)).collect();
            let coins: Vec<Vec<Vec<c64>>> = walks
                .iter()
                .map(|w| {
                    w.coins()
                        .iter()
                        .map(|c| {
                            (0..c.nrows()).flat_map(|i| (0..c.ncols()).map(move |j| c[(i, j)])).collect()
                        })
                        .collect()
                })
                .collect();
            let tables: Vec<WalkTables> = walks
                .iter()
                .zip(&coins)
                .map(|(w, c)| {
                    let (p, a, b) = w.exchange();
                    WalkTables {
                        bits: graph.edges(),
                        node_rows: &node_rows,
                        coins: c,
                        halves: &halves,
                        exchange: [p, a, b],
                    }
                })
                .collect();
            let states: Vec<c64> =
                walks.iter().flat_map(|w| w.product_state(&w.localized(0, 0), &vec![PLUS; n])).collect();
            let mut gw = device.walks(&tables, &states, Precision::Single).expect("upload");
            gw.steps_accumulate(steps / 10).expect("warm-up");
            let start = std::time::Instant::now();
            gw.steps_accumulate(steps).expect("steps");
            let s = start.elapsed().as_secs_f64();
            println!(
                "n = {n}, batch {batch:>2}: {:.3} ms per batched step, {:.1} µs per walk-step, {:.0} walk-steps/s",
                s * 1e3 / steps as f64,
                s * 1e6 / (steps * batch) as f64,
                (steps * batch) as f64 / s
            );
        }
    }
}
