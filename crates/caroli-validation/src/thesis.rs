//! The thesis setups, rebuilt with caroli: same parameters, same random configurations (through
//! the gfortran-compatible stream), and the thesis' own quirks where a comparison needs them.

use caroli_disorder::rng::Xoshiro256;
use caroli_disorder::{MagneticImpurities, Moments};
use caroli_green::EndToEnd;
use caroli_lattice::Site;
use caroli_model::{CMat, c64};

/// Sites of the impurity region of a thesis ribbon, in the order of `impurities_set`
/// (x fastest): `x0 .. x0 + nx_imp` × `0 .. ny`.
pub fn impurity_region(x0: i64, nx_imp: usize, ny: usize) -> Vec<Site> {
    (0..ny as i64)
        .flat_map(|y| (0..nx_imp as i64).map(move |x| Site { cell: [x0 + x, y, 0], sub: 0 }))
        .collect()
}

/// Disorder parameters of `conductance-2t` (`ti.ini`).
#[derive(Clone, Copy, Debug)]
pub struct Conductance2t {
    pub nx: usize,
    pub ny: usize,
    pub nx_imp: usize,
    pub conc: f64,
    /// 0: isotropic moments; otherwise in a cone of half-angle `theta0_deg` around z.
    pub spin: i32,
    pub theta0_deg: f64,
    /// `graine`.
    pub seed: i32,
}

impl Conductance2t {
    /// Magnetic impurities of `potential`: `int(conc·Nx_imp·Ny)` impurities in a centred region
    /// of length `nx_imp`. `x_first` is the cell index of the first sample slice.
    pub fn impurities(&self, x_first: i64) -> MagneticImpurities {
        let n_imp = (self.conc * self.nx_imp as f64 * self.ny as f64) as usize;
        let x0 = x_first + ((self.nx - self.nx_imp) / 2) as i64;
        let moments = if self.spin == 0 {
            Moments::Isotropic
        } else {
            Moments::Cone { theta0: self.theta0_deg * std::f64::consts::PI / 180.0 }
        };
        let mut rng = Xoshiro256::gfortran(self.seed);
        MagneticImpurities::generate(&mut rng, &impurity_region(x0, self.nx_imp, self.ny), n_imp, moments)
    }
}

/// Real part of a matrix, element by element.
fn real_part(m: &CMat) -> CMat {
    CMat::from_fn(m.nrows(), m.ncols(), |i, j| c64::new(m[(i, j)].re, 0.0))
}

/// The thesis transmission (`green.f90`): `Tr[Re(Γ_L) G Re(Γ_R) G†]`, with the broadenings
/// truncated to their real part (`gamma_p = real(IM*(Σ − Σ†))`). Evaluated on the slices that
/// carry the lead self-energies, as in the thesis (one clean lead slice at each end).
pub fn thesis_transmission(sol: &EndToEnd) -> f64 {
    let gl = real_part(&caroli_green::broadening(&sol.sigma_l));
    let gr = real_part(&caroli_green::broadening(&sol.sigma_r));
    caroli_green::trace(&(gl * &sol.g1n * gr * sol.g1n.adjoint())).re
}

/// Parameters of `local-currents-2t` (`ti_Lcur.ini`).
#[derive(Clone, Copy, Debug)]
pub struct LocalCurrents {
    pub nx: usize,
    pub ny: usize,
    pub nx_imp: usize,
    pub ne: usize,
    pub conc: f64,
    pub mz: f64,
    /// > 0: same-g, < 0: opposite-g, 0: Anderson (on-site `U0 n_z`, `n_z` uniform in [−1, 1]).
    pub couplage: i32,
    pub ev_l: f64,
    pub ev_r: f64,
    pub e_f: f64,
    pub emin: f64,
    pub emax: f64,
    pub u0: f64,
    pub eta: f64,
    pub seed: i32,
}

impl LocalCurrents {
    /// Impurities: `int(conc·(8 Nx / 10)·Ny)` (integer division) in the region of length
    /// `nx_imp`, moments `axisymetricmoment(Mz)`. Device slices: 0 and `nx + 1` are lead slices.
    pub fn impurities(&self) -> MagneticImpurities {
        let n_imp = (self.conc * ((self.nx * 8) / 10) as f64 * self.ny as f64) as usize;
        let x0 = 1 + ((self.nx - self.nx_imp) / 2) as i64;
        let mut rng = Xoshiro256::gfortran(self.seed);
        MagneticImpurities::generate(
            &mut rng,
            &impurity_region(x0, self.nx_imp, self.ny),
            n_imp,
            Moments::Polarized { mz: self.mz },
        )
    }

    /// Exchange matrices `(J_x, J_y, J_z)` of `couplage`.
    pub fn coupling(&self) -> [CMat; 3] {
        use caroli_model::presets::{GFactors, bhz_spin};
        match self.couplage.signum() {
            1 => bhz_spin(GFactors::Same),
            -1 => bhz_spin(GFactors::Opposite),
            _ => [CMat::zeros(4, 4), CMat::zeros(4, 4), caroli_model::gamma::gamma0()],
        }
    }

    /// Energies `emin + E_f + k·Δe`.
    pub fn energies(&self) -> Vec<f64> {
        let de = (self.emax - self.emin) / (self.ne - 1) as f64;
        (0..self.ne).map(|k| de * k as f64 + self.emin + self.e_f).collect()
    }

    /// The device of `green_recur`: lead slice, `nx` sample slices with the linear bias drop
    /// and the impurities, lead slice; leads shifted by `ev_l`, `ev_r`.
    pub fn device(&self, imp: &MagneticImpurities) -> caroli_green::TwoTerminal {
        use caroli_lattice::Region;
        use caroli_model::presets::{BhzParams, bhz};
        let (nx, ev_l, ev_r) = (self.nx, self.ev_l, self.ev_r);
        let bias = move |site: &Site| {
            let x = site.cell[0];
            let v = if x == 0 {
                ev_l
            } else if x == nx as i64 + 1 {
                ev_r
            } else {
                ev_l + x as f64 * (ev_r - ev_l) / (nx + 1) as f64
            };
            Some(CMat::from_fn(4, 4, |a, b| if a == b { c64::new(v, 0.0) } else { c64::new(0.0, 0.0) }))
        };
        let clean = bhz(BhzParams::default());
        let model = bhz(BhzParams::default())
            .with_onsite_term(bias)
            .with_onsite_term(imp.exchange(self.u0, &self.coupling()));
        let region = Region::cuboid(clean.lattice(), [nx + 2, self.ny, 0]);
        caroli_green::TwoTerminal::from_model(&model, &clean, &region, 0).with_lead_potentials(ev_l, ev_r)
    }
}

/// The thesis local-current outputs from caroli's lesser blocks, with the thesis' conventions:
/// `Lcur(2, 4, 4, Ny, Nx)` (orbital-resolved, element-wise products `−2 Re[t†_ab G^<_ab]` for x
/// and `−2 Re[t_ab G^<_{y+1,y; ab}]` for y with `iy = 2..Ny−1`, right-injected x part shifted by
/// one slice), summed over energies times `Δe`; and `Ldos(Ny, 4, Nx)` per energy,
/// `Im G^<_aa / π`. Both are column-major vectors.
pub fn thesis_local_currents(
    sys: &caroli_green::TwoTerminal,
    lessers: &[caroli_keldysh::Lesser],
    nx: usize,
    ny: usize,
    deltae: f64,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let d = 4;
    let idx = |c: usize, a: usize, b: usize, iy: usize, ix: usize| c + 2 * (a + d * (b + d * (iy + ny * ix)));
    let mut lcur = vec![0.0; 2 * d * d * ny * nx];
    let mut ldos_all = Vec::new();
    for l in lessers {
        for m in 1..=nx {
            let tx = &sys.couplings[m]; // block-diagonal in y: t_x on each row
            let h = &sys.slices[m];
            for iy in 0..ny {
                for a in 0..d {
                    for b in 0..d {
                        let ta = tx[(iy * d + b, iy * d + a)].conj(); // (t†)_ab
                        let gl = l.left.next[m][(iy * d + a, iy * d + b)];
                        let gr = l.right.next[m - 1][(iy * d + a, iy * d + b)];
                        lcur[idx(0, a, b, iy, m - 1)] += -2.0 * (ta * gl).re - 2.0 * (ta * gr).re;
                    }
                }
            }
            for iy in 1..ny.saturating_sub(1) {
                for a in 0..d {
                    for b in 0..d {
                        let ty = h[(iy * d + a, (iy + 1) * d + b)];
                        let g = l.left.diag[m][((iy + 1) * d + a, iy * d + b)]
                            + l.right.diag[m][((iy + 1) * d + a, iy * d + b)];
                        lcur[idx(1, a, b, iy, m - 1)] += -2.0 * (ty * g).re;
                    }
                }
            }
        }
        let mut ldos = vec![0.0; ny * d * nx];
        for m in 1..=nx {
            let g = l.diag(m);
            for iy in 0..ny {
                for a in 0..d {
                    ldos[iy + ny * (a + d * (m - 1))] = g[(iy * d + a, iy * d + a)].im / std::f64::consts::PI;
                }
            }
        }
        ldos_all.push(ldos);
    }
    lcur.iter_mut().for_each(|x| *x *= deltae);
    (lcur, ldos_all)
}

/// BHZ Bloch Hamiltonians on the thesis k grid (`k_FBZ(M, N)`).
pub fn bhz_hk(params: caroli_model::presets::BhzParams, m: usize, n: usize) -> Vec<CMat> {
    use caroli_lattice::Region;
    let model = caroli_model::presets::bhz(params);
    let cell = Region::cuboid(model.lattice(), [1, 1, 0]);
    caroli_born::k_grid(m, n)
        .into_iter()
        .map(|[kx, ky]| model.bloch_hamiltonian(&cell, &[[1, 0, 0], [0, 1, 0]], &[kx, ky]))
        .collect()
}

fn diag_shift(m: &CMat, v: c64) -> CMat {
    CMat::from_fn(m.nrows(), m.ncols(), |i, j| if i == j { m[(i, j)] + v } else { m[(i, j)] })
}

/// `scba-anderson` (`born.ini`).
#[derive(Clone, Copy, Debug)]
pub struct ScbaAnderson {
    pub params: caroli_model::presets::BhzParams,
    pub m: usize,
    pub n: usize,
    pub u0: (f64, f64),
    pub ef: (f64, f64),
    pub eta: f64,
    pub points: usize,
}

impl ScbaAnderson {
    /// `born.f90`: Σ carried from one E_F to the next with −iη added each time, mixing 0.5, stop
    /// at max|ΔΣ| ≤ 1e-4 or 400 steps. Returns `(mass + Re Σ55, E_F − Re Σ0, U0 axis, E_F
    /// axis)`, the maps column-major with U0 fastest.
    pub fn run(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let Self { params, m, n, u0, ef, eta, points } = *self;
        let hk = bhz_hk(params, m, n);
        let us: Vec<f64> = (0..points)
            .map(|i| (i as f64 * (u0.1.ln() - u0.0.ln()) / (points - 1) as f64 + u0.0.ln()).exp())
            .collect();
        let es: Vec<f64> =
            (0..points).map(|i| i as f64 * (ef.1 - ef.0) / (points - 1) as f64 + ef.0).collect();
        let (mut mass, mut mu) = (vec![0.0; points * points], vec![0.0; points * points]);
        for (iu, &u) in us.iter().enumerate() {
            let (mf, vertices) = caroli_born::anderson(u, 4);
            let scba = caroli_born::Scba { hk: hk.clone(), mean_field: mf, vertices };
            let mut sigma = CMat::zeros(4, 4);
            for (ie, &e) in es.iter().enumerate() {
                sigma = diag_shift(&sigma, c64::new(0.0, -eta));
                let (mut diff, mut kk) = (1.0, 0);
                while diff > 1e-4 && kk < 400 {
                    let next = scba.update(c64::new(e, 0.0), &sigma);
                    diff = caroli_born::max_abs_diff(&next, &sigma);
                    sigma = caroli_born::mix(&next, &sigma, 0.5);
                    kk += 1;
                }
                let s55 = (sigma[(0, 0)] - sigma[(1, 1)] + sigma[(2, 2)] - sigma[(3, 3)]) / 4.0;
                let s0 = (sigma[(0, 0)] + sigma[(1, 1)] + sigma[(2, 2)] + sigma[(3, 3)]) / 4.0;
                mass[iu + points * ie] = params.mass + s55.re;
                mu[iu + points * ie] = e - s0.re;
            }
        }
        (mass, mu, us, es)
    }
}

/// Largest `max_ij(|a_ij − b_ij| − prec |a_ij|)` (the thesis convergence measure).
fn relative_change(next: &CMat, sigma: &CMat, prec: f64) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for j in 0..4 {
        for i in 0..4 {
            m = m.max((next[(i, j)] - sigma[(i, j)]).norm() - prec * next[(i, j)].norm());
        }
    }
    m
}

/// `scba-magnetic-matrix` (`born.ini`).
#[derive(Clone, Copy, Debug)]
pub struct ScbaMatrix {
    pub params: caroli_model::presets::BhzParams,
    pub m: usize,
    pub n: usize,
    pub conc: f64,
    pub j_range: (f64, f64),
    pub ef: (f64, f64),
    pub eta: f64,
    pub points: usize,
    pub gegh: i32,
}

impl ScbaMatrix {
    /// `born.f90`: impurities along z with vertex Γ12 (`gegh ≥ 0`) or Γ12p (`gegh < 0`), mean
    /// field `conc J Γ` in the Green's function; two stages (mixing 0.05 to a relative change
    /// 0.1, then 1.0 to 1e-5), mean field added to Σ at the end and Σ carried to the next E_F.
    /// Returns `(mass_u, mu_u, mass_d, mu_d)` maps, J fastest.
    pub fn run(&self) -> [Vec<f64>; 4] {
        let Self { params, m, n, conc, j_range, ef, eta, points, gegh } = *self;
        let hk = bhz_hk(params, m, n);
        let gamma = if gegh < 0 { caroli_model::gamma::gamma12p() } else { caroli_model::gamma::gamma12() };
        let scale = |a: &CMat, s: f64| CMat::from_fn(4, 4, |i, j| c64::new(s, 0.0) * a[(i, j)]);
        let mut out = [
            vec![0.0; points * points],
            vec![0.0; points * points],
            vec![0.0; points * points],
            vec![0.0; points * points],
        ];
        for iu in 0..points {
            let ji = iu as f64 * (j_range.1 - j_range.0) / (points - 1) as f64 + j_range.0;
            let scba = caroli_born::Scba {
                hk: hk.clone(),
                mean_field: scale(&gamma, conc * ji),
                vertices: vec![(conc * ji * ji, gamma.clone())],
            };
            let mut sigma = CMat::zeros(4, 4);
            for ie in 0..points {
                let e = ie as f64 * (ef.1 - ef.0) / (points - 1) as f64 + ef.0;
                sigma = diag_shift(&sigma, c64::new(0.0, -eta));
                let mut diff = 1.0;
                for (p, prec) in [(0.05, 0.1), (1.0, 1e-5)] {
                    let mut kk = 0;
                    while diff > prec && kk < 400 {
                        let next = scba.update(c64::new(e, 0.0), &sigma);
                        diff = relative_change(&next, &sigma, prec);
                        sigma = caroli_born::mix(&next, &sigma, p);
                        kk += 1;
                    }
                }
                sigma += scale(&gamma, conc * ji);
                let k = iu + points * ie;
                out[0][k] = params.mass + ((sigma[(0, 0)] - sigma[(1, 1)]) / 2.0).re;
                out[1][k] = e - ((sigma[(0, 0)] + sigma[(1, 1)]) / 2.0).re;
                out[2][k] = params.mass + ((sigma[(2, 2)] - sigma[(3, 3)]) / 2.0).re;
                out[3][k] = e - ((sigma[(2, 2)] + sigma[(3, 3)]) / 2.0).re;
            }
        }
        out
    }
}

/// `scba-magnetic-analytic` (`born.ini`; mass, B and D are hard-coded in the program).
#[derive(Clone, Copy, Debug)]
pub struct ScbaAnalytic {
    pub m: usize,
    pub n: usize,
    pub conc: f64,
    pub mz: f64,
    pub anti: i32,
    pub points: usize,
    pub j_range: (f64, f64),
    pub ef: (f64, f64),
    pub eta: f64,
}

impl ScbaAnalytic {
    /// `self_born.f90`: the SCBA reduced to four complex scalars `(M↑, M↓, μ↑, μ↓)` for the BHZ
    /// blocks (λ = C = 0, mass −0.137), moments in a cone of polarisation `mz`, at `E_F + iη`.
    /// Weights as in the thesis: `v_z = (1 − conc²) mz² + (1 − mz)²/3`,
    /// `v_⊥ = (2/3)(1 − mz)(1 + 2 mz)` with 2/3 in single precision. Two stages (mixing 0.15 to
    /// 1e-4, then 0.8 to 1e-6). Returns `(M↑, μ↑, M↓, μ↓)` maps (real parts), J fastest.
    pub fn run(&self) -> [Vec<f64>; 4] {
        let Self { m, n, conc, mz, anti, points, j_range, ef, eta } = *self;
        let (mass, b, d) = (-0.137, -0.376406, -0.280933);
        let grid = caroli_born::k_grid(m, n);
        let kc: Vec<f64> = grid.iter().map(|k| 4.0 - 2.0 * k[0].cos() - 2.0 * k[1].cos()).collect();
        let ks: Vec<f64> = grid.iter().map(|k| k[0].sin().powi(2) + k[1].sin().powi(2)).collect();
        let vz = (1.0 - conc * conc) * mz * mz + (1.0 - mz) * (1.0 - mz) / 3.0;
        let vperp = (2.0f32 / 3.0f32) as f64 * (1.0 - mz) * (1.0 + 2.0 * mz);
        let anti = anti as f64;
        let mut out = [
            vec![0.0; points * points],
            vec![0.0; points * points],
            vec![0.0; points * points],
            vec![0.0; points * points],
        ];
        for iu in 0..points {
            let ji = iu as f64 * (j_range.1 - j_range.0) / (points - 1) as f64 + j_range.0;
            let norm = conc * ji * ji / (n * m) as f64;
            for ie in 0..points {
                let e = ie as f64 * (ef.1 - ef.0) / (points - 1) as f64 + ef.0;
                let z = c64::new(e, eta);
                let mu0 = c64::new(mass + anti * ji * conc * mz, 0.0);
                let md0 = c64::new(mass - anti * ji * conc * mz, 0.0);
                let muu0 = c64::new(-(1.0 - anti) * ji * conc * mz, 0.0);
                let mud0 = c64::new((1.0 - anti) * ji * conc * mz, 0.0);
                let (mut mu, mut md, mut muu, mut mud) = (mu0, md0, muu0, mud0);
                for (p, prec) in [(0.15, 1e-4), (0.8, 1e-6)] {
                    let (mut diff, mut kk) = (1.0, 0);
                    while diff > prec && kk < 500 {
                        let (mut dmu, mut dmd, mut dmuu, mut dmud) =
                            (c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0));
                        for (c, s) in kc.iter().zip(&ks) {
                            let eu = z + muu + d * c;
                            let ed = z + mud + d * c;
                            let (mu_k, md_k) = (mu - b * c, md - b * c);
                            let detu = eu * eu - mu_k * mu_k - s;
                            let detd = ed * ed - md_k * md_k - s;
                            let (vzu, vzd, vpu, vpd) = (vz / detu, vz / detd, vperp / detu, vperp / detd);
                            dmu += vzu * mu_k + vpd * md_k;
                            dmd += vzd * md_k + vpu * mu_k;
                            dmuu += vzu * eu + vpd * ed;
                            dmud += vzd * ed + vpu * eu;
                        }
                        let (mut_, mdt, muut, mudt) =
                            (mu0 - dmu * norm, md0 - dmd * norm, muu0 - dmuu * norm, mud0 - dmud * norm);
                        let ch = |t: c64, x: c64| ((t - x).norm() - prec * t.norm()).abs();
                        diff = p * (ch(mut_, mu) + ch(mdt, md) + ch(muut, muu) + ch(mudt, mud));
                        let mixp = |t: c64, x: c64| t * p + x * (1.0 - p);
                        (mu, md, muu, mud) =
                            (mixp(mut_, mu), mixp(mdt, md), mixp(muut, muu), mixp(mudt, mud));
                        kk += 1;
                    }
                }
                let k = iu + points * ie;
                out[0][k] = mu.re;
                out[1][k] = muu.re;
                out[2][k] = md.re;
                out[3][k] = mud.re;
            }
        }
        out
    }
}

/// Parameters of `conductance-4t` (`ti.ini`): an `nx × ny` square with four arms of length
/// `ntp` (one per edge, as wide as the edge) ending in semi-infinite leads, Anderson disorder on
/// the square only. The code works with the spin-up BHZ block (`dim = 2`, see
/// [`caroli_model::presets::bhz_block`]); its σ_xx and σ_xy are transmissions of that block.
#[derive(Clone, Copy, Debug)]
pub struct FourTerminal {
    pub nx: usize,
    pub ny: usize,
    pub ntp: usize,
    pub seed: i32,
}

impl FourTerminal {
    /// Device sites (square and arms) and the four lead interfaces in the thesis order
    /// (1: x = 1 − ntp, 2: y = ny + ntp, 3: x = nx + ntp, 4: y = 1 − ntp), thesis coordinates
    /// (1-based square).
    pub fn geometry(&self) -> (caroli_lattice::Region, Vec<caroli_green::Interface>) {
        use caroli_lattice::{Lattice, Region};
        let (nx, ny, ntp) = (self.nx as i64, self.ny as i64, self.ntp as i64);
        let lat = Lattice::square(1.0, 2);
        let site = |x: i64, y: i64| Site { cell: [x, y, 0], sub: 0 };
        let mut sites = Vec::new();
        for x in 1 - ntp..=nx + ntp {
            for y in 1 - ntp..=ny + ntp {
                let in_square = (1..=nx).contains(&x) && (1..=ny).contains(&y);
                let in_x_arm = (1..=ny).contains(&y);
                let in_y_arm = (1..=nx).contains(&x);
                if in_square || in_x_arm || in_y_arm {
                    sites.push(site(x, y));
                }
            }
        }
        let region = Region::from_sites(&lat, sites);
        let column = |x: i64| Region::from_sites(&lat, (1..=ny).map(|y| site(x, y)).collect());
        let row = |y: i64| Region::from_sites(&lat, (1..=nx).map(|x| site(x, y)).collect());
        let interfaces = vec![
            caroli_green::Interface { slice: column(1 - ntp), outward: [-1, 0, 0] },
            caroli_green::Interface { slice: row(ny + ntp), outward: [0, 1, 0] },
            caroli_green::Interface { slice: column(nx + ntp), outward: [1, 0, 0] },
            caroli_green::Interface { slice: row(1 - ntp), outward: [0, -1, 0] },
        ];
        (region, interfaces)
    }

    /// `anderson`: `random_set(graine)`, `random_number(U(Nx, Ny))` (x fastest),
    /// `U0·u − U0/2` on both orbitals of the square sites (the code uses one spin block,
    /// `dim = 2`).
    pub fn anderson(&self, u0: f64) -> caroli_disorder::OnSiteMap {
        let mut rng = Xoshiro256::gfortran(self.seed);
        let mut map = caroli_disorder::OnSiteMap::default();
        for y in 1..=self.ny as i64 {
            for x in 1..=self.nx as i64 {
                let v = u0 * rng.uniform() - u0 * 0.5;
                map.add(
                    Site { cell: [x, y, 0], sub: 0 },
                    CMat::from_fn(2, 2, |a, b| if a == b { c64::new(v, 0.0) } else { c64::new(0.0, 0.0) }),
                );
            }
        }
        map
    }
}

/// The thesis multi-terminal transmission `Tr[Re Γ_p G_pq Re Γ_q G_pq†]`.
pub fn thesis_transmission_pq(
    sys: &caroli_green::MultiTerminal,
    sol: &caroli_green::MultiSolution,
    p: usize,
    q: usize,
) -> f64 {
    let g = sol.block(sys, p, q);
    let gp = real_part(&caroli_green::broadening(&sol.sigmas[p]));
    let gq = real_part(&caroli_green::broadening(&sol.sigmas[q]));
    caroli_green::trace(&(gp * &g * gq * g.adjoint())).re
}

/// The thesis KPM programs (`dos_ti.f90`, graphene `dos.f90`) on rank 0 of the serial MPI
/// stub: only the spin-up states are traced (`choice = 1`, no LDOS).
#[derive(Clone, Copy, Debug)]
pub struct Kpm {
    pub n: usize,
    pub m: usize,
    pub n_state: usize,
    pub imp_conc: f64,
    pub v_exc: f64,
    /// 0: isotropic moments; otherwise along z (graphene: A +z, B −z).
    pub choice_spin: i32,
    /// TI only: opposite-g exchange (`variants/gap-antiferro`).
    pub opposite: bool,
}

/// Thesis `random_state`: four `random_number(phi(N, N))`, `exp(2πiφ)`, then `N · ifft`.
fn kpm_random_state(op: &caroli_torus::TorusOperator, rng: &mut Xoshiro256) -> Vec<c64> {
    let mut s: Vec<c64> =
        (0..op.len()).map(|_| c64::from_polar(1.0, 2.0 * std::f64::consts::PI * rng.uniform())).collect();
    op.to_x(&mut s);
    s.iter_mut().for_each(|z| *z *= op.n as f64);
    s
}

/// Moments `(n_x, n_y, n_z)` drawn as the thesis `potential` (uniform then phi).
fn kpm_moments(rng: &mut Xoshiro256, count: usize, isotropic: bool) -> Vec<[f64; 3]> {
    let u: Vec<f64> = rng.uniforms(count).into_iter().map(|u| 2.0 * u - 1.0).collect();
    let phi: Vec<f64> = rng.uniforms(count).into_iter().map(|p| 2.0 * std::f64::consts::PI * p).collect();
    u.iter()
        .zip(&phi)
        .map(|(&z, &p)| {
            if isotropic {
                [(1.0 - z * z).sqrt() * p.cos(), (1.0 - z * z).sqrt() * p.sin(), z]
            } else {
                [0.0, 0.0, 1.0]
            }
        })
        .collect()
}

/// Exchange matrix on components `(a, b)` of a 4-component spinor: `n_z` on (a, a), `−n_z` on
/// (b, b), `n_x ∓ i n_y` off-diagonal; `sign` multiplies everything (opposite-g H orbitals).
fn put_exchange(v: &mut CMat, a: usize, b: usize, n: [f64; 3], sign: f64) {
    v[(a, a)] += c64::new(sign * n[2], 0.0);
    v[(a, b)] += c64::new(sign * n[0], -sign * n[1]);
    v[(b, a)] += c64::new(sign * n[0], sign * n[1]);
    v[(b, b)] += c64::new(-sign * n[2], 0.0);
}

impl Kpm {
    /// Sites in `impurities_set` order (x fastest) as torus indices.
    fn draw_sites(&self, rng: &mut Xoshiro256, count: usize) -> Vec<usize> {
        let n = self.n;
        let candidates: Vec<Site> =
            (0..n * n).map(|k| Site { cell: [(k % n) as i64, (k / n) as i64, 0], sub: 0 }).collect();
        caroli_disorder::choose_sites(rng, &candidates, count)
            .iter()
            .map(|s| s.cell[0] as usize + n * s.cell[1] as usize)
            .collect()
    }

    fn run(
        &self,
        kinetic: Vec<CMat>,
        local: Vec<(usize, CMat)>,
        e_k: &[f64],
        de: f64,
        rng: &mut Xoshiro256,
    ) -> Vec<f64> {
        let op = caroli_torus::TorusOperator::new(self.n, 4, kinetic, local);
        let kernel = caroli_kpm::jackson(self.m);
        let nn = (self.n * self.n) as f64;
        let mut dos = vec![0.0; 2 * e_k.len()];
        for _ in 0..self.n_state {
            let mut state = kpm_random_state(&op, rng);
            let nn2 = self.n * self.n;
            state[2 * nn2..].iter_mut().for_each(|z| *z = c64::new(0.0, 0.0));
            let mu = caroli_kpm::moments(|v, out| op.apply(v, out), &state, &kernel);
            let series = caroli_kpm::chebyshev_series(&mu, e_k);
            for (k, (s, e)) in series.iter().zip(e_k).enumerate() {
                let norm = 1.0 / (std::f64::consts::PI * (1.0 - e * e).sqrt() * self.n_state as f64 * nn);
                dos[k] += norm * s.re;
            }
        }
        // Output record: dE·e_k + e0, then dos(:, 1:2)/dE (spin down left at zero on rank 0).
        let mut out: Vec<f64> = e_k.iter().map(|e| de * e).collect();
        out.extend(dos.iter().map(|d| d / de));
        out
    }

    /// `dos_ti.f90` (gap version): BHZ (mass −0.137) with the thesis k-space signs
    /// (`d33 = −sin kx`, `d44 = −sin ky`), `dE = (6 + |V_exc|)·1.01`, energies `i/(10M)`.
    pub fn ti(&self) -> Vec<f64> {
        let (mass, b, c, d) = (-0.137, -0.376406, 0.0, -0.280933);
        let n = self.n;
        let de = (6.0 + self.v_exc.abs()) * 1.01;
        let mut rng = Xoshiro256::gfortran(1237);
        let kinetic = (0..n * n)
            .map(|k| {
                let (kx, ky) = (caroli_torus::k_value(k % n, n), caroli_torus::k_value(k / n, n));
                let (d33, d44) = (-kx.sin(), -ky.sin());
                let f = 4.0 - 2.0 * kx.cos() - 2.0 * ky.cos();
                let (d55, h0) = (mass - b * f, c - d * f);
                let i = c64::new(0.0, 1.0);
                let r = |x: f64| c64::new(x, 0.0);
                let z = r(0.0);
                let rows = [
                    [r(d55 + h0), r(d33) - i * d44, z, z],
                    [r(d33) + i * d44, r(-d55 + h0), z, z],
                    [z, z, r(d55 + h0), r(-d33) - i * d44],
                    [z, z, r(-d33) + i * d44, r(-d55 + h0)],
                ];
                CMat::from_fn(4, 4, |a, bb| rows[a][bb] / de)
            })
            .collect();
        let n_imp = (self.imp_conc * (n * n) as f64) as usize;
        let sites = self.draw_sites(&mut rng, n_imp);
        let moments = kpm_moments(&mut rng, n_imp, self.choice_spin == 0);
        let local = sites
            .iter()
            .zip(&moments)
            .map(|(&s, &nm)| {
                let mut v = CMat::zeros(4, 4);
                put_exchange(&mut v, 0, 2, nm, 1.0);
                put_exchange(&mut v, 1, 3, nm, if self.opposite { -1.0 } else { 1.0 });
                (s, CMat::from_fn(4, 4, |a, bb| v[(a, bb)] * (self.v_exc / de)))
            })
            .collect();
        let m = self.m as i64;
        let e_k: Vec<f64> = (-m / 2..m / 2).map(|i| i as f64 / (10 * m) as f64).collect();
        self.run(kinetic, local, &e_k, de, &mut rng)
    }

    /// Graphene `dos.f90`: `hk = −e^{ikx} − e^{−i(kx+√3ky)/2} − e^{−i(kx−√3ky)/2}` on the grid
    /// `k0x = 4π/(3N)`, `k0y = 4π/(√3 N)`, components (A↑, B↑, A↓, B↓), impurities drawn on A
    /// then on B, `dE = (3 + 1.6)·1.01` (1.6 in single precision), energies at the 2M Chebyshev
    /// nodes.
    pub fn graphene(&self) -> Vec<f64> {
        let n = self.n;
        // `(THREE + 1.6)*1.01_db`: the literal 1.6 is single precision in the thesis code.
        let de = (3.0 + 1.6f32 as f64) * 1.01;
        let mut rng = Xoshiro256::gfortran(1237);
        let s3 = 3f64.sqrt();
        let kv = |i: usize, k0: f64| (if i < n / 2 { i as f64 } else { i as f64 - n as f64 }) * k0;
        let (k0x, k0y) = (
            2.0 * 2.0 * std::f64::consts::PI / (3 * n) as f64,
            2.0 * 2.0 * std::f64::consts::PI / (n as f64 * s3),
        );
        let kinetic = (0..n * n)
            .map(|k| {
                let (kx, ky) = (kv(k % n, k0x), kv(k / n, k0y));
                let i = c64::new(0.0, 1.0);
                let hk = -(i * kx).exp()
                    - (-(i * 0.5) * (kx + s3 * ky)).exp()
                    - (-(i * 0.5) * (kx - s3 * ky)).exp();
                let mut h = CMat::zeros(4, 4);
                h[(0, 1)] = hk / de;
                h[(1, 0)] = hk.conj() / de;
                h[(2, 3)] = hk / de;
                h[(3, 2)] = hk.conj() / de;
                h
            })
            .collect();
        let n_imp = (self.imp_conc * (n * n) as f64) as usize;
        let mut local: Vec<(usize, CMat)> = Vec::new();
        for (sub, sign_z) in [(0usize, 1.0), (1usize, -1.0)] {
            let sites = self.draw_sites(&mut rng, n_imp);
            let mut moments = kpm_moments(&mut rng, n_imp, self.choice_spin == 0);
            if self.choice_spin != 0 {
                moments.iter_mut().for_each(|m| m[2] = sign_z);
            }
            for (&s, &nm) in sites.iter().zip(&moments) {
                let mut v = CMat::zeros(4, 4);
                put_exchange(&mut v, sub, sub + 2, nm, 1.0);
                local.push((s, CMat::from_fn(4, 4, |a, b| v[(a, b)] * (self.v_exc / de))));
            }
        }
        let k = 2 * self.m;
        let e_k: Vec<f64> =
            (0..k).map(|i| (std::f64::consts::PI * (i as f64 + 0.5) / k as f64).cos()).collect();
        self.run(kinetic, local, &e_k, de, &mut rng)
    }
}
