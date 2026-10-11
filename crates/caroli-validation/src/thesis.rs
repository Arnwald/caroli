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
        self.grid(false)
    }

    fn grid(&self, corrected: bool) -> [Vec<f64>; 4] {
        let Self { m, n, points, j_range, ef, .. } = *self;
        let grid = caroli_born::k_grid(m, n);
        let k = self.k_terms(&grid);
        let mut out = [
            vec![0.0; points * points],
            vec![0.0; points * points],
            vec![0.0; points * points],
            vec![0.0; points * points],
        ];
        for iu in 0..points {
            let ji = iu as f64 * (j_range.1 - j_range.0) / (points - 1) as f64 + j_range.0;
            for ie in 0..points {
                let e = ie as f64 * (ef.1 - ef.0) / (points - 1) as f64 + ef.0;
                let r = self.solve_point(&k, ji, e, corrected);
                for (o, v) in out.iter_mut().zip(r) {
                    o[iu + points * ie] = v;
                }
            }
        }
        out
    }

    /// `(M↑, μ↑, M↓, μ↓)` at one `(J_I, E_F)` (the program's iteration for one grid point).
    pub fn point(&self, ji: f64, e: f64) -> [f64; 4] {
        self.solve_point(&self.k_terms(&caroli_born::k_grid(self.m, self.n)), ji, e, false)
    }

    /// The same program with its two errors corrected, everything else unchanged (equations,
    /// start from the mean field, two-stage mixing): the mass self-energy with the sign of the
    /// general SCBA, and the z weight from the second cumulant, `(1 − conc) mz² + (1 − mz)²/3`
    /// (and 2/3 in double precision). Equivalent to `caroli_born` with `magnetic` vertices.
    pub fn point_corrected(&self, ji: f64, e: f64) -> [f64; 4] {
        self.solve_point(&self.k_terms(&caroli_born::k_grid(self.m, self.n)), ji, e, true)
    }

    /// Corrected maps (see [`Self::point_corrected`]), laid out as [`Self::run`].
    pub fn run_corrected(&self) -> [Vec<f64>; 4] {
        self.grid(true)
    }

    /// `(k_c, k_s)` on the k grid.
    fn k_terms(&self, grid: &[[f64; 2]]) -> Vec<(f64, f64)> {
        grid.iter()
            .map(|k| (4.0 - 2.0 * k[0].cos() - 2.0 * k[1].cos(), k[0].sin().powi(2) + k[1].sin().powi(2)))
            .collect()
    }

    fn solve_point(&self, k: &[(f64, f64)], ji: f64, e: f64, corrected: bool) -> [f64; 4] {
        let Self { m, n, conc, mz, anti, eta, .. } = *self;
        let (mass, b, d) = (-0.137, -0.376406, -0.280933);
        let (vz, vperp, mass_sign) = if corrected {
            (
                (1.0 - conc) * mz * mz + (1.0 - mz) * (1.0 - mz) / 3.0,
                2.0 / 3.0 * (1.0 - mz) * (1.0 + 2.0 * mz),
                1.0,
            )
        } else {
            (
                (1.0 - conc * conc) * mz * mz + (1.0 - mz) * (1.0 - mz) / 3.0,
                (2.0f32 / 3.0f32) as f64 * (1.0 - mz) * (1.0 + 2.0 * mz),
                -1.0,
            )
        };
        let anti = anti as f64;
        let norm = conc * ji * ji / (n * m) as f64;
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
                for &(c, s) in k {
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
                let (mut_, mdt, muut, mudt) = (
                    mu0 + dmu * norm * mass_sign,
                    md0 + dmd * norm * mass_sign,
                    muu0 - dmuu * norm,
                    mud0 - dmud * norm,
                );
                let ch = |t: c64, x: c64| ((t - x).norm() - prec * t.norm()).abs();
                diff = p * (ch(mut_, mu) + ch(mdt, md) + ch(muut, muu) + ch(mudt, mud));
                let mixp = |t: c64, x: c64| t * p + x * (1.0 - p);
                (mu, md, muu, mud) = (mixp(mut_, mu), mixp(mdt, md), mixp(muut, muu), mixp(mudt, mud));
                kk += 1;
            }
        }
        [mu.re, muu.re, md.re, mud.re]
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

/// k point of FFT index `idx` on the thesis graphene grid `k_FBZ(N, N/2, N√3/2)`: steps
/// `4π/N` and `4π/(√3 N)`.
pub fn graphene_k(n: usize, idx: usize) -> (f64, f64) {
    let s3 = 3f64.sqrt();
    let kv = |i: usize, k0: f64| (if i < n / 2 { i as f64 } else { i as f64 - n as f64 }) * k0;
    let k0x = 2.0 * std::f64::consts::PI / (n as f64 * 0.5);
    let k0y = 2.0 * std::f64::consts::PI / (n as f64 * s3 * 0.5);
    (kv(idx % n, k0x), kv(idx / n, k0y))
}

/// `h(k, t) = −e^{i(ky − E t)/√3} − 2 cos(kx/2) e^{−i(ky − E t)/(2√3)}` of the thesis graphene codes.
///
/// Where `h` is real, gfortran's complex arithmetic leaves an imaginary part −0 (so `atan2`
/// gives −π); the replica does the same, because that gauge choice of the band states matters
/// once disorder mixes the k points.
pub fn graphene_h(n: usize, idx: usize, elec: f64, t: f64) -> c64 {
    let s3 = 3f64.sqrt();
    let (kx, ky) = graphene_k(n, idx);
    let i = c64::new(0.0, 1.0);
    let h =
        -(i * (ky - elec * t) / s3).exp() - 2.0 * (0.5 * kx).cos() * (-i * (ky - elec * t) * 0.5 / s3).exp();
    if h.im == 0.0 { c64::new(h.re, -0.0) } else { h }
}

/// Parameters of graphene `current-pair-creation` (`g_s.ini`, baseline-patched fixed seed).
#[derive(Clone, Copy, Debug)]
pub struct GrapheneCurrent {
    pub n: usize,
    pub vimp: f64,
    pub imp_conc: f64,
    pub elec_max: f64,
    pub nelec: usize,
    pub t_total: f64,
    pub samples: usize,
    pub dt: f64,
    pub ndiag: usize,
    /// 0: paramagnetic (isotropic); otherwise AF (A: +z, B: −z).
    pub choice_spin: i32,
}

impl GrapheneCurrent {
    fn k(&self, idx: usize) -> (f64, f64) {
        graphene_k(self.n, idx)
    }

    fn h(&self, idx: usize, elec: f64, t: f64) -> c64 {
        graphene_h(self.n, idx, elec, t)
    }

    /// Current and pair creation, one `(curr, np)` series of `NT/NDIAG + 1` values per field.
    pub fn run(&self) -> Vec<(Vec<f64>, Vec<f64>)> {
        let (n, nn) = (self.n, self.n * self.n);
        let nt = (self.t_total / self.dt) as usize;
        let ntdiag = nt / self.ndiag;
        let n_imp = (self.imp_conc * nn as f64) as usize;
        let s3 = 3f64.sqrt();
        let i = c64::new(0.0, 1.0);
        let fft = caroli_torus::Fft2::new(n);
        let to_k = |psi: &mut [c64]| psi.chunks_exact_mut(nn).for_each(|c| fft.forward(c));
        let to_x = |psi: &mut [c64]| {
            psi.chunks_exact_mut(nn).for_each(|c| {
                fft.backward(c);
                c.iter_mut().for_each(|z| *z /= nn as f64);
            })
        };
        let mut out = Vec::new();
        for ee in 1..=self.nelec {
            let elec = ee as f64 * (self.elec_max / self.nelec as f64);
            let mut rng = Xoshiro256::gfortran(1237);
            let (mut cur_tab, mut np_tab) = (vec![0.0; ntdiag + 1], vec![0.0; ntdiag + 1]);
            // Band state ±(e^{iφ/2}, e^{−iφ/2}) on the spin-up components.
            let band = |t: f64, sign: f64| -> Vec<c64> {
                let mut s = vec![c64::new(0.0, 0.0); 4 * nn];
                for k in 0..nn {
                    let phi = self.h(k, elec, t).arg();
                    s[k] = sign * (i * 0.5 * phi).exp();
                    s[k + nn] = (-i * 0.5 * phi).exp();
                }
                s
            };
            let current = |psi_k: &[c64], psi_0: &[c64], t: f64| -> (f64, f64) {
                if t == 0.0 {
                    return (0.0, 0.0);
                }
                let (mut cur, mut np) = (0.0, 0.0);
                for k in 0..nn {
                    let (kx, ky) = self.k(k);
                    let dh = -i * (i * (ky - elec * t) / s3).exp() / s3
                        + i * (0.5 * kx).cos() * (-i * (ky - elec * t) * 0.5 / s3).exp() / s3;
                    cur += (psi_k[k].conj() * dh * psi_k[k + nn]
                        + psi_k[k + 2 * nn].conj() * dh * psi_k[k + 3 * nn])
                        .re;
                    let p: c64 = (0..4).map(|c| psi_k[k + c * nn].conj() * psi_0[k + c * nn]).sum();
                    np += p.norm_sqr();
                }
                (-4.0 * cur, 2.0 * np)
            };
            for _ in 0..self.samples {
                let mut psi_k = band(0.0, -1.0);
                let norm = psi_k.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
                psi_k.iter_mut().for_each(|z| *z /= norm);
                let mut uv: Vec<(usize, CMat)> = Vec::new();
                let mut psi_x = Vec::new();
                if self.vimp != 0.0 {
                    let cand: Vec<Site> =
                        (0..nn).map(|k| Site { cell: [(k % n) as i64, (k / n) as i64, 0], sub: 0 }).collect();
                    let sites: Vec<usize> = caroli_disorder::choose_sites(&mut rng, &cand, 2 * n_imp)
                        .iter()
                        .map(|s| s.cell[0] as usize + n * s.cell[1] as usize)
                        .collect();
                    let u: Vec<f64> = rng.uniforms(2 * n_imp).into_iter().map(|u| 2.0 * u - 1.0).collect();
                    let ph: Vec<f64> =
                        rng.uniforms(2 * n_imp).into_iter().map(|p| 2.0 * std::f64::consts::PI * p).collect();
                    let (c, s) = ((0.5 * self.dt * self.vimp).cos(), (0.5 * self.dt * self.vimp).sin());
                    for (k, &site) in sites.iter().enumerate() {
                        let iso = self.choice_spin == 0;
                        let (nx, ny, mut nz) = if iso {
                            (
                                (1.0 - u[k] * u[k]).sqrt() * ph[k].cos(),
                                (1.0 - u[k] * u[k]).sqrt() * ph[k].sin(),
                                u[k],
                            )
                        } else {
                            (0.0, 0.0, 1.0)
                        };
                        let (a, b) = if k < n_imp { (0, 2) } else { (1, 3) };
                        if k >= n_imp && !iso {
                            nz = -1.0;
                        }
                        let mut m =
                            CMat::from_fn(
                                4,
                                4,
                                |p, q| if p == q { c64::new(1.0, 0.0) } else { c64::new(0.0, 0.0) },
                            );
                        m[(a, a)] = c - i * nz * s;
                        m[(a, b)] = (-i * nx - ny) * s;
                        m[(b, a)] = (-i * nx + ny) * s;
                        m[(b, b)] = c + i * nz * s;
                        uv.push((site, m));
                    }
                    psi_x = psi_k.clone();
                    to_x(&mut psi_x);
                }
                let mut is = 1;
                for it in 1..=nt {
                    let t = it as f64 * self.dt;
                    if self.vimp != 0.0 {
                        caroli_dynamics::apply_local(&mut psi_x, nn, 4, &uv);
                        psi_k = psi_x.clone();
                        to_k(&mut psi_k);
                    }
                    let uh: Vec<CMat> = (0..nn)
                        .map(|k| {
                            let phi = self.h(k, elec, t).arg();
                            let eps = ((self.h(k, elec, t) + self.h(k, elec, t + self.dt)) * 0.5).norm();
                            let (cs, sn) = ((self.dt * eps).cos(), (self.dt * eps).sin());
                            let mut m = CMat::zeros(4, 4);
                            for o in [0, 2] {
                                m[(o, o)] = c64::new(cs, 0.0);
                                m[(o, o + 1)] = -i * (i * phi).exp() * sn;
                                m[(o + 1, o)] = -i * (-i * phi).exp() * sn;
                                m[(o + 1, o + 1)] = c64::new(cs, 0.0);
                            }
                            m
                        })
                        .collect();
                    caroli_dynamics::apply_k(&mut psi_k, 4, &uh);
                    if self.vimp != 0.0 {
                        psi_x = psi_k.clone();
                        to_x(&mut psi_x);
                        caroli_dynamics::apply_local(&mut psi_x, nn, 4, &uv);
                    }
                    if it % self.ndiag == 0 {
                        if self.vimp != 0.0 {
                            psi_k = psi_x.clone();
                            to_k(&mut psi_k);
                        }
                        let psi_0 = band(t, 1.0);
                        let (cur, np) = current(&psi_k, &psi_0, t);
                        cur_tab[is] += cur;
                        np_tab[is] += np;
                        is += 1;
                    }
                }
                if self.vimp == 0.0 {
                    break;
                }
            }
            if self.vimp != 0.0 {
                cur_tab.iter_mut().for_each(|x| *x /= self.samples as f64);
                np_tab.iter_mut().for_each(|x| *x /= self.samples as f64);
            }
            out.push((cur_tab, np_tab));
        }
        out
    }
}

/// Graphene impurity half-step blocks `exp(−i Vimp dt/2 n·σ)` of the thesis `potentialevol`:
/// `2·n_imp` distinct sites (A then B), moments isotropic or AF (A +z, B −z).
fn graphene_half_steps(
    rng: &mut Xoshiro256,
    n: usize,
    n_imp: usize,
    vimp: f64,
    dt: f64,
    choice_spin: i32,
) -> Vec<(usize, CMat)> {
    let i = c64::new(0.0, 1.0);
    let cand: Vec<Site> =
        (0..n * n).map(|k| Site { cell: [(k % n) as i64, (k / n) as i64, 0], sub: 0 }).collect();
    let sites: Vec<usize> = caroli_disorder::choose_sites(rng, &cand, 2 * n_imp)
        .iter()
        .map(|s| s.cell[0] as usize + n * s.cell[1] as usize)
        .collect();
    let u: Vec<f64> = rng.uniforms(2 * n_imp).into_iter().map(|u| 2.0 * u - 1.0).collect();
    let ph: Vec<f64> = rng.uniforms(2 * n_imp).into_iter().map(|p| 2.0 * std::f64::consts::PI * p).collect();
    let (c, s) = ((0.5 * dt * vimp).cos(), (0.5 * dt * vimp).sin());
    sites
        .iter()
        .enumerate()
        .map(|(k, &site)| {
            let iso = choice_spin == 0;
            let (nx, ny, mut nz) = if iso {
                ((1.0 - u[k] * u[k]).sqrt() * ph[k].cos(), (1.0 - u[k] * u[k]).sqrt() * ph[k].sin(), u[k])
            } else {
                (0.0, 0.0, 1.0)
            };
            let (a, b) = if k < n_imp { (0, 2) } else { (1, 3) };
            if k >= n_imp && !iso {
                nz = -1.0;
            }
            let mut m =
                CMat::from_fn(4, 4, |p, q| if p == q { c64::new(1.0, 0.0) } else { c64::new(0.0, 0.0) });
            m[(a, a)] = c - i * nz * s;
            m[(a, b)] = (-i * nx - ny) * s;
            m[(b, a)] = (-i * nx + ny) * s;
            m[(b, b)] = c + i * nz * s;
            (site, m)
        })
        .collect()
}

/// Parameters of graphene `wavepacket-width` (`g_s.ini`; `imp_conc` in percent).
#[derive(Clone, Copy, Debug)]
pub struct GrapheneWidth {
    pub n: usize,
    pub vimp: f64,
    pub imp_conc: f64,
    pub elec: f64,
    pub t_total: f64,
    pub samples: usize,
    pub dt: f64,
    pub ndiag: usize,
    pub choice_spin: i32,
}

impl GrapheneWidth {
    /// Band → orbital basis at time `t` (thesis `ugauge`), per spin block.
    fn gauge(&self, band: &[c64], t: f64, inverse: bool) -> Vec<c64> {
        let nn = self.n * self.n;
        let i = c64::new(0.0, 1.0);
        let s2 = 2f64.sqrt();
        let mut out = vec![c64::new(0.0, 0.0); 4 * nn];
        for k in 0..nn {
            let phi = graphene_h(self.n, k, self.elec, t).arg();
            for o in [0, 2] {
                let (p, q) = (band[k + o * nn], band[k + (o + 1) * nn]);
                if inverse {
                    // ugaugedag: (e^{−iφ} p + q, −e^{−iφ} p + q)/√2
                    out[k + o * nn] = ((-i * phi).exp() * p + q) / s2;
                    out[k + (o + 1) * nn] = (-(-i * phi).exp() * p + q) / s2;
                } else {
                    // ugauge: (e^{iφ} p − e^{iφ} q, p + q)/√2
                    out[k + o * nn] = ((i * phi).exp() * p - (i * phi).exp() * q) / s2;
                    out[k + (o + 1) * nn] = (p + q) / s2;
                }
            }
        }
        out
    }

    /// Current, pair creation and wave-packet width, `NT/NDIAG + 1` values each.
    pub fn run(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (n, nn) = (self.n, self.n * self.n);
        let nt = (self.t_total / self.dt) as usize;
        let ntdiag = nt / self.ndiag;
        let n_imp = (self.imp_conc * nn as f64 / 100.0) as usize;
        let (s3, i, elec, dt) = (3f64.sqrt(), c64::new(0.0, 1.0), self.elec, self.dt);
        let fft = caroli_torus::Fft2::new(n);
        let to_k = |psi: &mut [c64]| psi.chunks_exact_mut(nn).for_each(|c| fft.forward(c));
        let to_x = |psi: &mut [c64]| {
            psi.chunks_exact_mut(nn).for_each(|c| {
                fft.backward(c);
                c.iter_mut().for_each(|z| *z /= nn as f64);
            })
        };
        let dh_t = |k: usize, t: f64| {
            let (kx, ky) = graphene_k(n, k);
            i * elec * (i * (ky - elec * t) / s3).exp() / s3
                - i * elec * (0.5 * kx).cos() * (-i * (ky - elec * t) * 0.5 / s3).exp() / s3
        };
        let mut rng = Xoshiro256::gfortran(1237);
        let (mut cur_tab, mut np_tab, mut w_tab) =
            (vec![0.0; ntdiag + 1], vec![0.0; ntdiag + 1], vec![0.0; ntdiag + 1]);
        for _ in 0..self.samples {
            let mut psi_k = vec![c64::new(0.0, 0.0); 4 * nn];
            psi_k[nn..2 * nn].iter_mut().for_each(|z| *z = c64::new(1.0 / (nn as f64).sqrt(), 0.0));
            let uv = if self.vimp != 0.0 {
                graphene_half_steps(&mut rng, n, n_imp, self.vimp, dt, self.choice_spin)
            } else {
                Vec::new()
            };
            let mut psi_x = self.gauge(&psi_k, 0.0, false);
            to_x(&mut psi_x);
            let mut iss = 0;
            for it in 0..=nt {
                let t = it as f64 * dt;
                caroli_dynamics::apply_local(&mut psi_x, nn, 4, &uv);
                let mut k1 = psi_x.clone();
                to_k(&mut k1);
                psi_k = self.gauge(&k1, t, true);
                let uh: Vec<CMat> = (0..nn)
                    .map(|k| {
                        let (h0, h1) = (graphene_h(n, k, elec, t), graphene_h(n, k, elec, t + dt));
                        let eps = (h0.norm() + h1.norm()) * 0.5;
                        let vdphi = -0.5 * ((dh_t(k, t) / h0).im + (dh_t(k, t + dt) / h1).im) * 0.5;
                        let sq = (eps * eps + vdphi * vdphi).sqrt();
                        let (cs, sn) = ((dt * sq).cos(), (dt * sq).sin());
                        let mut m = CMat::zeros(4, 4);
                        for o in [0, 2] {
                            m[(o, o)] = cs + i * eps * sn / sq;
                            m[(o, o + 1)] = -i * vdphi * sn / sq;
                            m[(o + 1, o)] = -i * vdphi * sn / sq;
                            m[(o + 1, o + 1)] = cs - i * eps * sn / sq;
                        }
                        m
                    })
                    .collect();
                caroli_dynamics::apply_k(&mut psi_k, 4, &uh);
                psi_x = self.gauge(&psi_k, t, false);
                to_x(&mut psi_x);
                caroli_dynamics::apply_local(&mut psi_x, nn, 4, &uv);
                if it % self.ndiag == 0 {
                    let mut k1 = psi_x.clone();
                    to_k(&mut k1);
                    psi_k = self.gauge(&k1, t, true);
                    let orb = self.gauge(&psi_k, t, false);
                    let mut cur = 0.0;
                    let mut np = 0.0;
                    for k in 0..nn {
                        let (kx, ky) = graphene_k(n, k);
                        let dh = -i * (i * (ky - elec * t) / s3).exp() / s3
                            + i * (0.5 * kx).cos() * (-i * (ky - elec * t) * 0.5 / s3).exp() / s3;
                        cur += (orb[k].conj() * dh * orb[k + nn]
                            + orb[k + 2 * nn].conj() * dh * orb[k + 3 * nn])
                            .re;
                        np += psi_k[k].norm_sqr();
                    }
                    let cur = if it == 0 { 0.0 } else { -4.0 * cur };
                    cur_tab[iss] += cur;
                    np_tab[iss] += 2.0 * np;
                    w_tab[iss] += graphene_width(n, &psi_x);
                    iss += 1;
                }
            }
            if self.vimp == 0.0 {
                break;
            }
        }
        if self.vimp != 0.0 {
            for v in [&mut cur_tab, &mut np_tab, &mut w_tab] {
                v.iter_mut().for_each(|x| *x /= self.samples as f64);
            }
        }
        (cur_tab, np_tab, w_tab)
    }
}

/// Thesis `width_t` (with the baseline's in-bounds `xy_mesh`): sublattice densities, quadrant
/// swap, `sqrt(⟨x²⟩ − ⟨x⟩² + ⟨y²⟩ − ⟨y⟩²)` over the A (x1, y1) and B (x2, y2) positions.
pub fn graphene_width(n: usize, psi_x: &[c64]) -> f64 {
    let nn = n * n;
    let s3 = 3f64.sqrt();
    let (h, n2) = (0.5, n / 2);
    let idx = |a: usize, b: usize| a + n * b; // (first index x, second y), 0-based
    let (mut x1, mut y1, mut x2, mut y2) = (vec![0.0; nn], vec![0.0; nn], vec![0.0; nn], vec![0.0; nn]);
    for j in -(n2 as i64)..(n2 as i64) {
        for i in -(n2 as i64)..(n2 as i64) {
            let p = idx((i + n2 as i64) as usize, (j + n2 as i64) as usize);
            x1[p] = (i + 1) as f64 * h + (j + 1) as f64 + h;
            y1[p] = (i + 1) as f64 * s3 * h + h / s3;
            x2[p] = (i + 1) as f64 * h + (j + 1) as f64;
            y2[p] = (i + 1) as f64 * s3 * h;
        }
    }
    let dens = |a: usize, b: usize| -> Vec<f64> {
        (0..nn).map(|p| psi_x[p + a * nn].norm_sqr() + psi_x[p + b * nn].norm_sqr()).collect()
    };
    let swap = |d: &[f64]| -> Vec<f64> {
        let mut out = d.to_vec();
        for jj in 0..n2 {
            for kk in 0..n2 {
                out[idx(jj, kk)] = d[idx(jj + n2, kk + n2)];
                out[idx(jj + n2, kk + n2)] = d[idx(jj, kk)];
                out[idx(jj + n2, kk)] = d[idx(jj, kk + n2)];
                out[idx(jj, kk + n2)] = d[idx(jj + n2, kk)];
            }
        }
        out
    };
    let (d1, d2) = (swap(&dens(0, 2)), swap(&dens(1, 3)));
    let moment = |c1: &[f64], c2: &[f64], pw: i32| -> f64 {
        (0..nn).map(|p| c1[p].powi(pw) * d1[p] + c2[p].powi(pw) * d2[p]).sum()
    };
    let wx2 = moment(&x1, &x2, 2) - moment(&x1, &x2, 1).powi(2);
    let wy2 = moment(&y1, &y2, 2) - moment(&y1, &y2, 1).powi(2);
    (wx2 + wy2).sqrt()
}

/// Parameters of graphene `wavepacket-density` (`g_s.ini`; `imp_conc` in percent; one
/// realisation).
#[derive(Clone, Copy, Debug)]
pub struct GrapheneDensity {
    pub n: usize,
    pub vimp: f64,
    pub imp_conc: f64,
    pub elec: f64,
    pub t_total: f64,
    pub dt: f64,
    pub ndiag: usize,
    pub choice_spin: i32,
}

/// Output of [`GrapheneDensity::run`].
pub struct DensityRun {
    pub current: Vec<f64>,
    pub pairs: Vec<f64>,
    /// Sublattice densities (A, then B) of the conduction-band projection, one `n × n` map per
    /// diagnostic, quadrant-swapped.
    pub density_a: Vec<Vec<f64>>,
    pub density_b: Vec<Vec<f64>>,
}

impl GrapheneDensity {
    pub fn run(&self) -> DensityRun {
        let (n, nn) = (self.n, self.n * self.n);
        let nt = (self.t_total / self.dt) as usize;
        let ntdiag = nt / self.ndiag;
        let n_imp = (self.imp_conc * nn as f64 / 100.0) as usize;
        let (s3, i, elec, dt) = (3f64.sqrt(), c64::new(0.0, 1.0), self.elec, self.dt);
        let fft = caroli_torus::Fft2::new(n);
        let to_k = |psi: &mut [c64]| psi.chunks_exact_mut(nn).for_each(|c| fft.forward(c));
        let to_x = |psi: &mut [c64]| {
            psi.chunks_exact_mut(nn).for_each(|c| {
                fft.backward(c);
                c.iter_mut().for_each(|z| *z /= nn as f64);
            })
        };
        let band = |t: f64, sign: f64| -> Vec<c64> {
            let mut s = vec![c64::new(0.0, 0.0); 4 * nn];
            for k in 0..nn {
                let phi = graphene_h(n, k, elec, t).arg();
                s[k] = sign * (i * 0.5 * phi).exp();
                s[k + nn] = (-i * 0.5 * phi).exp();
            }
            s
        };
        let mut rng = Xoshiro256::gfortran(1237);
        let mut psi_k = band(0.0, -1.0);
        let norm = (2.0 * nn as f64).sqrt();
        psi_k.iter_mut().for_each(|z| *z /= norm);
        let uv = if self.vimp != 0.0 {
            graphene_half_steps(&mut rng, n, n_imp, self.vimp, dt, self.choice_spin)
        } else {
            Vec::new()
        };
        let mut psi_x = psi_k.clone();
        to_x(&mut psi_x);
        let (mut cur_tab, mut np_tab) = (vec![0.0; ntdiag + 1], vec![0.0; ntdiag + 1]);
        let (mut da, mut db) = (Vec::new(), Vec::new());
        let mut iss = 0;
        for it in 0..=nt {
            let t = it as f64 * dt;
            caroli_dynamics::apply_local(&mut psi_x, nn, 4, &uv);
            psi_k = psi_x.clone();
            to_k(&mut psi_k);
            let uh: Vec<CMat> = (0..nn)
                .map(|k| {
                    let phi = graphene_h(n, k, elec, t).arg();
                    let eps = ((graphene_h(n, k, elec, t) + graphene_h(n, k, elec, t + dt)) * 0.5).norm();
                    let (cs, sn) = ((dt * eps).cos(), (dt * eps).sin());
                    let mut m = CMat::zeros(4, 4);
                    for o in [0, 2] {
                        m[(o, o)] = c64::new(cs, 0.0);
                        m[(o, o + 1)] = -i * (i * phi).exp() * sn;
                        m[(o + 1, o)] = -i * (-i * phi).exp() * sn;
                        m[(o + 1, o + 1)] = c64::new(cs, 0.0);
                    }
                    m
                })
                .collect();
            caroli_dynamics::apply_k(&mut psi_k, 4, &uh);
            psi_x = psi_k.clone();
            to_x(&mut psi_x);
            caroli_dynamics::apply_local(&mut psi_x, nn, 4, &uv);
            if it % self.ndiag == 0 {
                psi_k = psi_x.clone();
                to_k(&mut psi_k);
                // Conduction-band projection back to orbitals, densities per sublattice.
                let mut p2 = vec![c64::new(0.0, 0.0); 4 * nn];
                for k in 0..nn {
                    let phi = graphene_h(n, k, elec, t).arg();
                    for o in [0, 2] {
                        let p1 = (-i * phi).exp() * psi_k[k + o * nn] + psi_k[k + (o + 1) * nn];
                        p2[k + o * nn] = (i * phi).exp() * p1;
                        p2[k + (o + 1) * nn] = p1;
                    }
                }
                to_x(&mut p2);
                let swap = |d: Vec<f64>| -> Vec<f64> {
                    let (n2, idx) = (n / 2, |a: usize, b: usize| a + n * b);
                    let mut out = d.clone();
                    for jj in 0..n2 {
                        for kk in 0..n2 {
                            out[idx(jj, kk)] = d[idx(jj + n2, kk + n2)];
                            out[idx(jj + n2, kk + n2)] = d[idx(jj, kk)];
                            out[idx(jj + n2, kk)] = d[idx(jj, kk + n2)];
                            out[idx(jj, kk + n2)] = d[idx(jj + n2, kk)];
                        }
                    }
                    out
                };
                da.push(swap((0..nn).map(|p| p2[p].norm_sqr() + p2[p + 2 * nn].norm_sqr()).collect()));
                db.push(swap((0..nn).map(|p| p2[p + nn].norm_sqr() + p2[p + 3 * nn].norm_sqr()).collect()));
                let psi_0 = band(t, 1.0);
                let (mut cur, mut np) = (0.0, 0.0);
                for k in 0..nn {
                    let (kx, ky) = graphene_k(n, k);
                    let dh = -i * (i * (ky - elec * t) / s3).exp() / s3
                        + i * (0.5 * kx).cos() * (-i * (ky - elec * t) * 0.5 / s3).exp() / s3;
                    cur += (psi_k[k].conj() * dh * psi_k[k + nn]
                        + psi_k[k + 2 * nn].conj() * dh * psi_k[k + 3 * nn])
                        .re;
                    let p: c64 = (0..4).map(|c| psi_k[k + c * nn].conj() * psi_0[k + c * nn]).sum();
                    np += p.norm_sqr();
                }
                let (cur, np) = if t == 0.0 { (0.0, 0.0) } else { (-4.0 * cur, 2.0 * np) };
                cur_tab[iss] += cur;
                np_tab[iss] += np;
                iss += 1;
            }
        }
        // The program writes the first NTDIAG entries only.
        cur_tab.truncate(ntdiag);
        np_tab.truncate(ntdiag);
        DensityRun { current: cur_tab, pairs: np_tab, density_a: da, density_b: db }
    }
}

/// The 2013 `dos-spin-lesser` program (first recursive code), as it actually computes.
///
/// - Its `green_lesser` never forms G^<: the Fermi functions and Σ^< are computed but unused.
/// - The returned matrix is the last one of a recursion that counts the left self-energy
///   twice on slice 0 and restarts at slice `m` from an uninverted matrix.
/// - The "up"/"down" sums group the slice indices `y + Ny·a` by their value mod 4, i.e. by
///   position across the ribbon, not by spin (equal in a clean mirror-symmetric sample).
///
/// The 2013 Γ matrices (Γ44 = −τ_y) with the `+i/2 Γ33` of its `t_x` amount to the BHZ model
/// with `A → −A` (the transverse-field terms vanish at `Ez = 0`).
#[derive(Clone, Copy, Debug)]
pub struct DosSpinLesser2013 {
    pub nx: usize,
    pub ny: usize,
    pub ne: usize,
    pub params: caroli_model::presets::BhzParams,
}

impl DosSpinLesser2013 {
    /// `(energies, dos_u, dos_d)` on the program's hard-coded window [−2, 6].
    pub fn run(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        use caroli_green::{SelfEnergyMethod, TwoTerminal};
        use faer::linalg::solvers::DenseSolveCore;
        let eta = 1e-10;
        let model = caroli_model::presets::bhz(caroli_model::presets::BhzParams { a: -1.0, ..self.params });
        let region = caroli_lattice::Region::cuboid(model.lattice(), [self.nx, self.ny, 0]);
        let sys = TwoTerminal::from_model(&model, &model, &region, 0);
        let (h, t) = (&sys.slices[0], &sys.couplings[0]);
        let n = h.nrows();
        let inv = |m: CMat| -> CMat { m.partial_piv_lu().inverse() };
        let resolvent = |z: c64, sigma: &CMat| {
            CMat::from_fn(n, n, |i, j| if i == j { z } else { c64::new(0.0, 0.0) }) - h - sigma
        };
        let de = 8.0 / (self.ne - 1) as f64;
        let energies: Vec<f64> = (0..self.ne).map(|i| de * i as f64 - 2.0).collect();
        let (mut up, mut down) = (vec![0.0; self.ne], vec![0.0; self.ne]);
        for (ie, &e) in energies.iter().enumerate() {
            let (sl, sr) = sys.self_energies(e, SelfEnergyMethod::ModesBroadened { eta }).unwrap();
            let z = c64::new(e, eta);
            let g0l = inv(resolvent(c64::new(e, 0.0), &sl));
            for m in 1..=self.nx {
                let mut a = g0l.clone();
                for ix in 0..=m {
                    let s = t.adjoint() * &a * t;
                    a = inv(if ix == 0 { resolvent(z, &(s + &sl)) } else { resolvent(z, &s) });
                }
                let mut c = resolvent(z, &(t.adjoint() * &a * t));
                for ix in m..=self.nx + 1 {
                    let s = t.adjoint() * &c * t;
                    c = inv(if ix == self.nx + 1 { resolvent(z, &(s + &sr)) } else { resolvent(z, &s) });
                }
                // caroli slice index 4y + a ↔ Fortran index y + Ny·a.
                for k in 0..n {
                    let f = k / 4 + self.ny * (k % 4);
                    let target = if f % 4 < 2 { &mut up[ie] } else { &mut down[ie] };
                    *target -= c[(k, k)].im;
                }
            }
        }
        (energies, up, down)
    }
}

/// Parameters of the 2014 `conductance-ldos-current` program (`Manuscrit/codes/ti`): BHZ
/// ribbon between two leads at potentials `ev_l`, `ev_r` (linear drop across the sample),
/// Anderson disorder `[−u0/2, u0/2]` on slices `2nx/10 ..= nx − 2nx/10`, `ne` energies.
///
/// Its hoppings are `t_x = DΓ0 + BΓ55 + (i/2)Γ33` and `t_y = DΓ0 + BΓ55 + (i/2)Γ44`: the sign of
/// the A term along x is opposite to the BHZ preset (the two are related by the spin flip s_x).
#[derive(Clone, Copy, Debug)]
pub struct CondLdosCurrent2014 {
    pub nx: usize,
    pub ny: usize,
    pub ne: usize,
    pub ev_l: f64,
    pub ev_r: f64,
    pub mass: f64,
    pub emin: f64,
    pub emax: f64,
    pub u0: f64,
    pub eta: f64,
    pub seed: i32,
}

/// Output of [`CondLdosCurrent2014::run`], column-major as the Fortran arrays.
pub struct CondLdosRun {
    pub energies: Vec<f64>,
    /// `T(E)·(f_R − f_L)` (the program's `cond_e` when `ev_r ≠ ev_l`).
    pub conductance: Vec<f64>,
    /// `Ldos(Ny, 4, Nx, Ne) = −Im G^r_aa`.
    pub ldos: Vec<f64>,
    /// `Lcur(Ny, 4, Nx, 4, 2)`, summed over energies times `−Δe/2π`.
    pub lcur: Vec<f64>,
}

impl CondLdosCurrent2014 {
    fn model(&self) -> caroli_model::Model {
        use caroli_model::gamma::{gamma0, gamma33, gamma44, gamma55};
        use caroli_model::presets::BhzParams;
        let p = BhzParams { mass: self.mass, ..BhzParams::default() };
        let (r, i) = (|x: f64| c64::new(x, 0.0), |x: f64| c64::new(0.0, x));
        let onsite =
            caroli_model::combine(&[(r(p.c - 4.0 * p.d), &gamma0()), (r(p.mass - 4.0 * p.b), &gamma55())]);
        let tx = caroli_model::combine(&[(r(p.d), &gamma0()), (r(p.b), &gamma55()), (i(0.5), &gamma33())]);
        let ty = caroli_model::combine(&[(r(p.d), &gamma0()), (r(p.b), &gamma55()), (i(0.5), &gamma44())]);
        caroli_model::Model::new(caroli_lattice::Lattice::square(1.0, 4))
            .onsite(0, onsite)
            .hopping(0, 0, [1, 0, 0], tx)
            .hopping(0, 0, [0, 1, 0], ty)
    }

    /// Lead slice, `nx` sample slices (bias drop and disorder), lead slice.
    pub fn device(&self) -> caroli_green::TwoTerminal {
        let (nx, ny, ev_l, ev_r) = (self.nx, self.ny, self.ev_l, self.ev_r);
        let mut rng = Xoshiro256::gfortran(self.seed);
        let u: Vec<f64> = rng.uniforms(ny * nx).iter().map(|x| self.u0 * x - self.u0 * 0.5).collect();
        let (lo, hi) = (2 * nx / 10, nx - 2 * nx / 10);
        let onsite = move |site: &Site| {
            let (x, y) = (site.cell[0], site.cell[1] as usize);
            let mut v = ev_l + x as f64 * (ev_r - ev_l) / (nx + 1) as f64;
            if x >= lo as i64 && x <= hi as i64 {
                v += u[y + ny * (x as usize - 1)];
            }
            Some(CMat::from_fn(4, 4, |a, b| if a == b { c64::new(v, 0.0) } else { c64::new(0.0, 0.0) }))
        };
        let region = caroli_lattice::Region::cuboid(self.model().lattice(), [nx + 2, ny, 0]);
        // `self_contact` builds `h00 − (E + iη) − ev`: the leads sit at −ev while the boundary
        // slices carry +ev.
        caroli_green::TwoTerminal::from_model(
            &self.model().with_onsite_term(onsite),
            &self.model(),
            &region,
            0,
        )
        .with_lead_potentials(-ev_l, -ev_r)
    }

    pub fn run(&self) -> CondLdosRun {
        use caroli_green::SelfEnergyMethod;
        use caroli_keldysh::{Lesser, fermi};
        let (nx, ny, d) = (self.nx, self.ny, 4);
        let sys = self.device();
        let method = SelfEnergyMethod::ModesBroadened { eta: self.eta };
        let de = (self.emax - self.emin) / (self.ne - 1) as f64;
        let energies: Vec<f64> = (0..self.ne).map(|i| de * i as f64 + self.emin).collect();
        let (mut conductance, mut ldos) = (Vec::new(), Vec::new());
        let mut lcur = vec![0.0; ny * d * nx * d * 2];
        let li =
            |iy: usize, a: usize, ix: usize, b: usize, c: usize| iy + ny * (a + d * (ix + nx * (b + d * c)));
        let temp = 5e-3;
        for &e in &energies {
            let (fl, fr) = (fermi(e, self.ev_l, temp), fermi(e, self.ev_r, temp));
            let l = Lesser::compute(&sys, e, method, fl, fr).unwrap();
            let bias = self.ev_r - self.ev_l != 0.0;
            conductance.push(if bias { l.transmission * (fr - fl) } else { l.transmission });
            let sw = sys.sweeps(e, method).unwrap();
            for m in 1..=nx {
                let g = sw.diagonal(&sys, m);
                let (gx, gy) = (l.next(m), l.diag(m));
                let (tx, h) = (&sys.couplings[m], &sys.slices[m]);
                for a in 0..d {
                    for iy in 0..ny {
                        ldos.push(0.0);
                        let k = ldos.len() - 1;
                        debug_assert_eq!(k, iy + ny * (a + d * ((m - 1) + nx * (conductance.len() - 1))));
                        ldos[k] = -g[(iy * d + a, iy * d + a)].im;
                    }
                }
                for iy in 0..ny {
                    for a in 0..d {
                        for b in 0..d {
                            let t_dag = tx[(iy * d + b, iy * d + a)].conj();
                            let jx = 2.0 * (t_dag * gx[(iy * d + a, iy * d + b)]).re;
                            lcur[li(iy, a, m - 1, b, 0)] -= jx * de / (2.0 * std::f64::consts::PI);
                            if iy + 1 < ny {
                                let hy = h[((iy + 1) * d + b, iy * d + a)];
                                let jy = 2.0 * (hy * gy[(iy * d + a, (iy + 1) * d + b)]).re;
                                lcur[li(iy, a, m - 1, b, 1)] -= jy * de / (2.0 * std::f64::consts::PI);
                            }
                        }
                    }
                }
            }
        }
        CondLdosRun { energies, conductance, ldos, lcur }
    }
}
