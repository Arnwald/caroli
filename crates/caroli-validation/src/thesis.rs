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
