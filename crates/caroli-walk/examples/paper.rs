//! Figures 2 and 3 of Sellapillay & Verga (Phys. Rev. E 103, 032123): the ring of 13 nodes, the
//! walker on color 0 of node 0, every spin along +x (state 'x'), for (θ, J) = (π/8, 0.2) and
//! (π/2, 0.2). Prints the mean spin and the entropies `S_x`, `S_c`, `S_s` every 100 steps up to
//! 2000, and the time per step.
//!
//! ```text
//! cargo run --release -p caroli-walk --example paper
//! ```

use caroli_walk::{PLUS, SpinWalk};
use std::f64::consts::PI;

fn main() {
    let n = 13;
    for (label, theta) in [("π/8", PI / 8.0), ("π/2", PI / 2.0)] {
        let walk = SpinWalk::ring(n, theta, 0.2);
        let mut psi = walk.product_state(&walk.localized(0, 0), &vec![PLUS; n]);
        println!("(θ, J) = ({label}, 0.2), dimension {}", walk.dimension());
        println!("{:>5} {:>8} {:>8} {:>8} {:>7} {:>7} {:>7}", "t", "s_x", "s_y", "s_z", "S_x", "S_c", "S_s");
        let start = std::time::Instant::now();
        let mut stepping = std::time::Duration::ZERO;
        for t in 0..=2000 {
            if t % 100 == 0 {
                let spins = walk.spins(&psi);
                let mean = |k: usize| spins.iter().map(|s| s[k]).sum::<f64>() / n as f64;
                let [sx, sc, ss] = walk.entropies(&psi);
                println!(
                    "{t:>5} {:>8.4} {:>8.4} {:>8.4} {sx:>7.3} {sc:>7.3} {ss:>7.3}",
                    mean(0),
                    mean(1),
                    mean(2)
                );
            }
            let before = std::time::Instant::now();
            walk.step(&mut psi);
            stepping += before.elapsed();
        }
        println!(
            "2000 steps: {:.2} s in all, {:.1} µs per step\n",
            start.elapsed().as_secs_f64(),
            stepping.as_secs_f64() * 1e6 / 2001.0
        );
    }
}
