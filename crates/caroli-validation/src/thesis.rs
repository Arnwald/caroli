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
