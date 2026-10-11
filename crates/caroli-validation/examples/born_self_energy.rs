//! Exact renormalised masses of the g2g1 systems, to test the Born lines without the Born
//! approximation.
//!
//! The disorder-averaged Green's function at k = 0 defines the exact self-energy
//! `Σ(E) = E − H₀(k = 0) − ⟨G⟩⁻¹`. In each spin block, `H₀(0) = m τ_z`, and the barred
//! parameters of the paper follow:
//! - `M̄ = m + Re(Σ_E − Σ_H)/2`;
//! - `μ̄ = −Re(Σ_E + Σ_H)/2`, so the gap is centred on −μ̄;
//! - `Γ = −Im(Σ_E + Σ_H)/2`, the disorder broadening (it stays near zero when no state lies
//!   at E_F).
//!
//! `⟨G⟩(k = 0)` comes from one large periodic sample (L × L sites, conc·L² impurities with
//! M_z = 1), averaged over the k = 0 plane waves of each orbital. The plane-wave elements are
//! expanded in Chebyshev polynomials at `E + iη`. With s = √(z̃² − 1) and t = z̃ − s, |t| < 1:
//!
//! `1/(z̃ − x) = (1 + 2 Σ_{n≥1} tⁿ T_n(x)) / s`.
//!
//! At weak disorder, Σ is second order in J_I and the Born approximation is exact, so the sign
//! of the mass correction can be read directly. In the 'F' case, the mean field does not touch
//! the mass at all.
//!
//! Writes `target/born_lines/self_energy.csv` (exact, per sample) and `born_masses.csv` (Born
//! lines, published and corrected) for `plot_born_masses.py`.
//!
//! `cargo run --release -p caroli-validation --example born_self_energy`

use caroli_disorder::rng::Xoshiro256;
use caroli_disorder::{MagneticImpurities, Moments};
use caroli_kpm::interleave;
use caroli_lattice::Region;
use caroli_model::presets::{BhzParams, GFactors, bhz, bhz_spin};
use caroli_model::{SparseMatrix, c64};
use caroli_validation::thesis::{ScbaAnalytic, impurity_region};
use rayon::prelude::*;

const L: usize = 160;
const ETA: f64 = 6e-3;
const MASS: f64 = -0.137;
const SEEDS: [i32; 2] = [2015, 4031];

/// (title, opposite-g, conc, E_F, J_I values).
type Case = (&'static str, i32, f64, f64, &'static [f64]);
const CASES: [Case; 2] = [
    ("'F' (same-g), n_I = 0.2, M_z = 1", 0, 0.2, 0.0, &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    ("'AF' (opposite-g), n_I = 0.4, M_z = 1", 1, 0.4, 0.02, &[0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]),
];

/// Per-vector moments `⟨φ_v| T_n(H̃) |φ_v⟩` with the doubling `μ_{2n} = 2⟨T_n|T_n⟩ − μ_0`,
/// `μ_{2n+1} = 2⟨T_n|T_{n+1}⟩ − μ_1` (the block is read once per product).
fn moments(h: &SparseMatrix, starts: &[Vec<c64>], count: usize) -> Vec<Vec<c64>> {
    let m = starts.len();
    let zero = c64::new(0.0, 0.0);
    let dots = |a: &[c64], b: &[c64]| -> Vec<c64> {
        a.par_chunks(4096 * m)
            .zip(b.par_chunks(4096 * m))
            .map(|(x, y)| {
                let mut s = vec![zero; m];
                for (i, (p, q)) in x.iter().zip(y).enumerate() {
                    s[i % m] += p.conj() * q;
                }
                s
            })
            .reduce(
                || vec![zero; m],
                |mut a, b| {
                    a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                    a
                },
            )
    };
    let mut prev = interleave(starts);
    let mut cur = vec![zero; prev.len()];
    h.apply_block(&prev, &mut cur, m);
    let (mut s1, mut s2) = (vec![dots(&prev, &prev)], vec![dots(&prev, &cur)]);
    let mut next = vec![zero; prev.len()];
    for _ in 1..count.div_ceil(2) {
        h.apply_block(&cur, &mut next, m);
        next.par_iter_mut().zip(&prev).for_each(|(x, p)| *x = 2.0 * *x - p);
        s1.push(dots(&cur, &cur));
        s2.push(dots(&cur, &next));
        std::mem::swap(&mut prev, &mut cur);
        std::mem::swap(&mut cur, &mut next);
    }
    (0..m)
        .map(|v| {
            (0..count)
                .map(|k| match k {
                    0 => s1[0][v],
                    1 => s2[0][v],
                    _ if k % 2 == 0 => 2.0 * s1[k / 2][v] - s1[0][v],
                    _ => 2.0 * s2[k / 2][v] - s2[0][v],
                })
                .collect()
        })
        .collect()
}

/// `⟨φ| (z − H)⁻¹ |φ⟩` from the moments of `H̃ = (H − centre)/half`.
fn resolvent(mu: &[c64], z: c64, centre: f64, half: f64) -> c64 {
    let zt = (z - centre) / half;
    let mut s = (zt * zt - 1.0).sqrt();
    if (zt - s).norm() > 1.0 {
        s = -s;
    }
    let t = zt - s;
    let (mut acc, mut tn) = (mu[0], c64::new(1.0, 0.0));
    for &m in &mu[1..] {
        tn *= t;
        acc += 2.0 * tn * m;
    }
    acc / s / half
}

/// Exact `(M̄↑, μ̄↑, M̄↓, μ̄↓, Γ↑, Γ↓)` at `E_F + iη` for one sample.
fn exact(anti: i32, conc: f64, j: f64, e: f64, seed: i32) -> [f64; 6] {
    let n_imp = (conc * (L * L) as f64) as usize;
    let imp = MagneticImpurities::generate(
        &mut Xoshiro256::gfortran(seed),
        &impurity_region(0, L, L),
        n_imp,
        Moments::Cone { theta0: 0.0 },
    );
    let coupling = if anti == 1 { GFactors::Opposite } else { GFactors::Same };
    let model = bhz(BhzParams::default()).with_onsite_term(imp.exchange(j, &bhz_spin(coupling)));
    let region = Region::cuboid(model.lattice(), [L, L, 0]);
    let period = L as i64;
    let h = model.sparse_bloch_hamiltonian(&region, &[[period, 0, 0], [0, period, 0]], &[0.0, 0.0]);
    let (lo, hi) = h.spectral_bounds();
    let (centre, half) = ((lo + hi) / 2.0, (hi - lo) / 2.0 * 1.01);
    let ht = h.rescaled(centre, half);
    // Enough terms for |t|^n < 1e-10 at the worst point of the window.
    let z = c64::new(e, ETA);
    let zt = (z - centre) / half;
    let t = {
        let s = (zt * zt - 1.0).sqrt();
        if (zt - s).norm() > 1.0 { zt + s } else { zt - s }
    };
    let count = (-23.0 / t.norm().ln()).ceil() as usize;
    // k = 0 plane wave of each orbital (E↑, H↑, E↓, H↓), normalised.
    let starts: Vec<Vec<c64>> = (0..4)
        .map(|a| {
            (0..4 * L * L)
                .map(|i| if i % 4 == a { c64::new(1.0 / L as f64, 0.0) } else { c64::new(0.0, 0.0) })
                .collect()
        })
        .collect();
    let mu = moments(&ht, &starts, count);
    let g: Vec<c64> = mu.iter().map(|m| resolvent(m, z, centre, half)).collect();
    let block = |s: usize| {
        let se = z - MASS - 1.0 / g[2 * s];
        let sh = z + MASS - 1.0 / g[2 * s + 1];
        (MASS + ((se - sh) * 0.5).re, -((se + sh) * 0.5).re, -((se + sh) * 0.5).im)
    };
    let (up, dn) = (block(0), block(1));
    [up.0, up.1, dn.0, dn.1, up.2, dn.2]
}

fn main() {
    let jobs: Vec<(usize, f64, i32)> = CASES
        .iter()
        .enumerate()
        .flat_map(|(c, case)| case.4.iter().flat_map(move |&j| SEEDS.iter().map(move |&s| (c, j, s))))
        .collect();
    let results: Vec<[f64; 6]> = jobs
        .par_iter()
        .map(|&(c, j, s)| {
            let (_, anti, conc, e, _) = CASES[c];
            exact(anti, conc, j, e, s)
        })
        .collect();
    use std::io::Write;
    let dir = caroli_validation::repo_root().join("target/born_lines");
    std::fs::create_dir_all(&dir).expect("output directory");
    let mut exact_csv = std::fs::File::create(dir.join("self_energy.csv")).expect("self_energy.csv");
    writeln!(exact_csv, "case,J_I,E_F,seed,M_up,mu_up,M_dn,mu_dn,gamma_up,gamma_dn").unwrap();
    for (&(c, j, s), r) in jobs.iter().zip(&results) {
        let e = CASES[c].3;
        writeln!(exact_csv, "{c},{j},{e},{s},{},{},{},{},{},{}", r[0], r[1], r[2], r[3], r[4], r[5]).unwrap();
    }
    let mut born_csv = std::fs::File::create(dir.join("born_masses.csv")).expect("born_masses.csv");
    writeln!(born_csv, "case,J_I,E_F,version,M_up,mu_up,M_dn,mu_dn").unwrap();
    println!("Exact renormalised parameters at k = 0 ({L} × {L} torus, η = {ETA}, {} samples)", SEEDS.len());
    println!("against the Born program as published and corrected. Topological: M < 0 (B < 0).");
    for (c, &(title, anti, conc, e, js)) in CASES.iter().enumerate() {
        let born = ScbaAnalytic {
            m: 256,
            n: 64,
            conc,
            mz: 1.0,
            anti,
            points: 64,
            j_range: (0.0, 1.5),
            ef: (-0.16, 0.16),
            eta: 1e-3,
        };
        let jmax = js.iter().copied().fold(0.0, f64::max);
        for i in 0..=60 {
            let j = jmax * i as f64 / 60.0;
            for (version, r) in [("published", born.point(j, e)), ("corrected", born.point_corrected(j, e))] {
                writeln!(born_csv, "{c},{j},{e},{version},{},{},{},{}", r[0], r[1], r[2], r[3]).unwrap();
            }
        }
        println!("\n{title}, E_F = {e:+.3}");
        println!("  J_I  | M̄↑ exact (samples)       published corrected | M̄↓ exact (samples)       published corrected | μ̄↑ exact  pub.    corr.   | Γ↑ exact Γ↓ exact");
        for &j in js {
            let runs: Vec<&[f64; 6]> = jobs
                .iter()
                .zip(&results)
                .filter(|((cc, jj, _), _)| *cc == c && *jj == j)
                .map(|(_, r)| r)
                .collect();
            let mean = |k: usize| runs.iter().map(|r| r[k]).sum::<f64>() / runs.len() as f64;
            let spread = |k: usize| {
                let v: Vec<String> = runs.iter().map(|r| format!("{:+.4}", r[k])).collect();
                v.join(" ")
            };
            let (p, q) = (born.point(j, e), born.point_corrected(j, e));
            println!(
                "  {j:.2} | {:+.4} ({})  {:+.4}   {:+.4}  | {:+.4} ({})  {:+.4}   {:+.4}  | {:+.4}  {:+.4}  {:+.4} | {:.4}   {:.4}",
                mean(0),
                spread(0),
                p[0],
                q[0],
                mean(2),
                spread(2),
                p[2],
                q[2],
                mean(1),
                p[1],
                q[1],
                mean(4),
                mean(5)
            );
        }
    }
}
