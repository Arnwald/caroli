//! Impact of the thesis' Re(Γ) truncation on its four-terminal production runs (`conductance-4t`,
//! `run/ti.ini`: 20×20 square, arms of 10, Anderson disorder on a log grid U0 ∈ [0.55, 5.5],
//! E ∈ [−0.27, 1.1], spin-up block, seed 12347).
//!
//! On the 50 × 50 (U0, E) grid (or every `step`-th point), compares the thesis formula
//! `Tr[Re Γ_p G Re Γ_q G†]` with the Caroli formula from the same solution, for T13 (`sigma_xx`)
//! and T24 (`sigma_xy`), separately on the plateaus (exact T within 0.05 of an integer) and
//! elsewhere.
//!
//! `cargo run --release -p caroli-validation --example gamma_impact_4t -- [step]`

use caroli_green::{MultiTerminal, SelfEnergyMethod};
use caroli_model::presets::{BhzParams, Spin, bhz_block};
use caroli_validation::thesis::{FourTerminal, thesis_transmission_pq};

fn main() {
    let p = FourTerminal { nx: 20, ny: 20, ntp: 10, seed: 12347 };
    let (region, interfaces) = p.geometry();
    let clean = bhz_block(BhzParams::default(), Spin::Up);
    let (nu, ne) = (50, 50);
    let u0 = |k: usize| 0.55 * (5.5f64 / 0.55).powf(k as f64 / (nu - 1) as f64);
    let e = |k: usize| -0.27 + (1.1 + 0.27) * k as f64 / (ne - 1) as f64;
    println!("{} device orbitals", region.orbitals());
    let step: usize = std::env::args().nth(1).map_or(1, |a| a.parse().expect("step"));
    // [T13, T24] × [plateau, elsewhere]: (max, sum, count).
    let mut acc = [[(0.0f64, 0.0f64, 0usize); 2]; 2];
    let mut largest = (0.0, 0.0, 0.0, 0.0, 0.0);
    for iu in (0..nu).step_by(step) {
        let device = bhz_block(BhzParams::default(), Spin::Up).with_onsite_term(p.anderson(u0(iu)));
        let sys = MultiTerminal::from_model(&device, &clean, &region, &interfaces);
        for ie in (0..ne).step_by(step) {
            let sol = sys.solve(e(ie), SelfEnergyMethod::ModesBroadened { eta: 1e-5 }).unwrap();
            for (k, (a, b)) in [(0, 2), (1, 3)].into_iter().enumerate() {
                let thesis = thesis_transmission_pq(&sys, &sol, a, b);
                let exact = sol.transmission(&sys, a, b);
                let d = (thesis - exact).abs();
                let slot = &mut acc[k][((exact - exact.round()).abs() >= 0.05) as usize];
                *slot = (slot.0.max(d), slot.1 + d, slot.2 + 1);
                if d > largest.0 {
                    largest = (d, u0(iu), e(ie), thesis, exact);
                }
            }
        }
    }
    println!("Per spin block (the thesis doubles them for both spins), |thesis − exact|:");
    for (k, name) in ["T13 (sigma_xx)", "T24 (sigma_xy)"].iter().enumerate() {
        for (w, place) in ["plateaus", "elsewhere"].iter().enumerate() {
            let (m, s, c) = acc[k][w];
            println!("  {name} {place:9}: {c:4} points, max {m:.3}, mean {:.4}", s / c.max(1) as f64);
        }
    }
    let (d, u, en, t, x) = largest;
    println!("  largest at U0 = {u:.2}, E = {en:.3}: thesis {t:.3}, exact {x:.3} (Δ = {d:.3})");
}
