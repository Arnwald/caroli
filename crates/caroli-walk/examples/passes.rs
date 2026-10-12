//! Time per step on the CPU in f64 and f32: one walk on all the threads, and many walks side
//! by side on one thread each (as in a scan of parameters).
//!
//! ```text
//! cargo run --release -p caroli-walk --example passes -- [n...]
//! ```

use caroli_walk::{PLUS, SpinWalk, to_single};
use rayon::prelude::*;
use std::time::Instant;

fn main() {
    let sizes: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().expect("n")).collect();
    let sizes = if sizes.is_empty() { vec![13, 19] } else { sizes };
    let threads = rayon::current_num_threads();
    for n in sizes {
        let w = SpinWalk::ring(n, 0.3, 0.2);
        let psi = w.product_state(&w.localized(0, 0), &vec![PLUS; n]);
        let d = w.dimension() as f64;
        // Enough steps for about half a second.
        let reps = ((5e8 / d) as usize).clamp(4, 2000);
        let mut double = psi.clone();
        let mut single = to_single(&psi);
        w.steps(&mut double, reps / 4 + 1);
        let t = Instant::now();
        w.steps(&mut double, reps);
        let f64_all = t.elapsed().as_secs_f64() / reps as f64;
        w.steps(&mut single, reps / 4 + 1);
        let t = Instant::now();
        w.steps(&mut single, reps);
        let f32_all = t.elapsed().as_secs_f64() / reps as f64;
        // One walk per thread, side by side.
        let side = |single: bool| {
            let t = Instant::now();
            (0..threads).into_par_iter().for_each(|_| {
                if single {
                    let mut s = to_single(&psi);
                    w.steps_serial(&mut s, reps);
                } else {
                    let mut s = psi.clone();
                    w.steps_serial(&mut s, reps);
                }
            });
            t.elapsed().as_secs_f64() / reps as f64
        };
        let (f64_side, f32_side) = if n <= 17 { (side(false), side(true)) } else { (f64::NAN, f64::NAN) };
        println!(
            "n = {n} ({threads} threads): one walk {:.3} ms (f64), {:.3} ms (f32); {threads} walks side by side {:.3} ms (f64), {:.3} ms (f32) per step, {:.2} / {:.2} ns per amplitude-step",
            f64_all * 1e3,
            f32_all * 1e3,
            f64_side * 1e3,
            f32_side * 1e3,
            f64_side * 1e9 / (d * threads as f64),
            f32_side * 1e9 / (d * threads as f64),
        );
    }
}
