//! Impact of the thesis' bond-current formula on the local-current maps of the g2g1 paper
//! (Fig. 2: `anomalous/Results/{Antiferro,Ferro}/*_0.2_1.0/Lcur`, runs 00–02).
//!
//! The thesis sums the element-wise products `−2 Re[t_ab G^<_ab]` over the orbitals. For the x
//! bonds this equals the current `2 Re Tr[V G^<_{n+1,n}]` because `t_x` is symmetric. For the
//! y bonds it computes `−2 Re Tr[t_yᵀ G^<_{y+1,y}]`, which flips the sign of the symmetric part
//! `DΓ0 + BΓ55` of `t_y`; the bond between rows 1 and 2 is not stored either. For each run,
//! the same lesser blocks give both versions, compared as the figure draws them (site vectors
//! `(j_x, j_y)` summed over the orbitals).
//!
//! `cargo run --release -p caroli-validation --example lcur_impact`

use caroli_green::SelfEnergyMethod;
use caroli_keldysh::{Lesser, fermi};
use caroli_validation::thesis::{LocalCurrents, thesis_local_currents};

/// (figure, couplage, J_I = U0, E_F, seed), from the runs' `param_0N.dat`.
const RUNS: [(&str, i32, f64, f64, i32); 6] = [
    ("A, G = 2 (f_A.21_02)", -1, 0.2, 0.053333, 57151),
    ("A, G = 1 (f_A.21_00)", -1, 0.666, 0.053333, 57151),
    ("A, G = 0 (f_A.21_01)", -1, 1.6, 0.053333, 57151),
    ("F, G = 2 (f_F.21_01)", 1, 0.2, -0.08889, 13151),
    ("F, G = 1 (f_F.21_00)", 1, 1.0714, -0.08889, 13151),
    ("F, G = 0 (f_F.21_02)", 1, 1.4, 0.1, 13151),
];

fn main() {
    let (nx, ny, d) = (400usize, 64usize, 4usize);
    println!(
        "run                     |Δj_x|/max|j_x|   ‖Δj_y‖/‖j_y‖   arrows turned > 30° (of sites with |j| > 10 % max)"
    );
    for &(name, couplage, u0, e_f, seed) in &RUNS {
        let p = LocalCurrents {
            nx,
            ny,
            nx_imp: 256,
            ne: 10,
            conc: 0.2,
            mz: 1.0,
            couplage,
            ev_l: 0.001,
            ev_r: -0.001,
            e_f,
            emin: -0.0009,
            emax: 0.0009,
            u0,
            eta: 1e-5,
            seed,
        };
        let imp = p.impurities();
        let sys = p.device(&imp);
        let deltae = (p.emax - p.emin) / (p.ne - 1) as f64;
        // Site maps [ix][iy] of the thesis formula and of the correct currents.
        let (mut tx, mut ty) = (vec![0.0; nx * ny], vec![0.0; nx * ny]);
        let (mut cx, mut cy) = (vec![0.0; nx * ny], vec![0.0; nx * ny]);
        for e in p.energies() {
            let (fl, fr) = (fermi(e, p.e_f + p.ev_l, 0.0), fermi(e, p.e_f + p.ev_r, 0.0));
            let l =
                Lesser::compute(&sys, e, SelfEnergyMethod::ModesBroadened { eta: p.eta }, fl, fr).unwrap();
            let (lcur, _) = thesis_local_currents(&sys, std::slice::from_ref(&l), nx, ny, deltae);
            let idx = |c: usize, a: usize, b: usize, iy: usize, ix: usize| {
                c + 2 * (a + d * (b + d * (iy + ny * ix)))
            };
            for ix in 0..nx {
                for iy in 0..ny {
                    for a in 0..d {
                        for b in 0..d {
                            tx[ix * ny + iy] += lcur[idx(0, a, b, iy, ix)];
                            ty[ix * ny + iy] += lcur[idx(1, a, b, iy, ix)];
                        }
                    }
                }
            }
            // Correct: x bond (m, m+1) row iy: 2 Re Σ_ab V_ab G^<_{m+1,m;ba} = −2 Re Σ conj(V_ab) G^<_{m,m+1;ab};
            // y bond (iy, iy+1) in slice m: 2 Re Σ_ab t_ab G^<_{mm;(iy+1,b),(iy,a)}.
            for m in 1..=nx {
                let (v, h, gn, gd) = (&sys.couplings[m], &sys.slices[m], l.next(m), l.diag(m));
                for iy in 0..ny {
                    let mut jx = 0.0;
                    for a in 0..d {
                        for b in 0..d {
                            jx -=
                                2.0 * (v[(iy * d + a, iy * d + b)].conj() * gn[(iy * d + a, iy * d + b)]).re;
                        }
                    }
                    cx[(m - 1) * ny + iy] += jx * deltae;
                    if iy + 1 < ny {
                        let mut jy = 0.0;
                        for a in 0..d {
                            for b in 0..d {
                                jy += 2.0
                                    * (h[(iy * d + a, (iy + 1) * d + b)]
                                        * gd[((iy + 1) * d + b, iy * d + a)])
                                        .re;
                            }
                        }
                        cy[(m - 1) * ny + iy] += jy * deltae;
                    }
                }
            }
        }
        let max_x = cx.iter().fold(0f64, |m, v| m.max(v.abs()));
        let dx = tx.iter().zip(&cx).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / max_x;
        let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let dy = norm(&ty.iter().zip(&cy).map(|(a, b)| a - b).collect::<Vec<_>>()) / norm(&cy);
        let mag: Vec<f64> = cx.iter().zip(&cy).map(|(x, y)| x.hypot(*y)).collect();
        let big = mag.iter().fold(0f64, |m, v| m.max(*v)) * 0.1;
        let (mut sig, mut turned) = (0, 0);
        for k in 0..nx * ny {
            if mag[k] > big {
                sig += 1;
                let cos = (tx[k] * cx[k] + ty[k] * cy[k]) / (tx[k].hypot(ty[k]) * mag[k]);
                turned += (cos < 30f64.to_radians().cos()) as usize;
            }
        }
        println!(
            "{name:24}  {dx:13.1e}   {dy:12.3}   {turned:5} / {sig:5} ({:.1} %)",
            100.0 * turned as f64 / sig.max(1) as f64
        );
    }
}
