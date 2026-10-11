//! 4×4 matrices of the BHZ basis `(E1↑, H1↑, E1↓, H1↓)`, named as in the thesis codes
//! (`numbers.f90`, 2014 and later). With `s` the spin and `τ` the E/H orbital Pauli matrices:
//!
//! | matrix | Kronecker form | role |
//! |---|---|---|
//! | `gamma0` | `1` | identity |
//! | `gamma55` | `τ_z` | mass term |
//! | `gamma33` | `s_z τ_x` | `A sin k_x` |
//! | `gamma44` | `τ_y` | `A sin k_y` |
//! | `gamma13` | `−s_y` | inter-block spin-orbit `λ` (x) |
//! | `gamma14` | `s_x τ_z` | inter-block spin-orbit `λ` (y) |
//! | `gamma23`, `gamma31`, `gamma12` | `s_x`, `s_y`, `s_z` | exchange, same-g (`g_E = g_H`) |
//! | `gamma23p`, `gamma31p`, `gamma12p` | `s_x τ_z`, `s_y τ_z`, `s_z τ_z` | exchange, opposite-g (`g_E = −g_H`) |

use crate::{CMat, c64};

const O: c64 = c64::new(0.0, 0.0);
const L: c64 = c64::new(1.0, 0.0);
const I: c64 = c64::new(0.0, 1.0);

fn from_rows(rows: [[c64; 4]; 4]) -> CMat {
    CMat::from_fn(4, 4, |i, j| rows[i][j])
}

pub fn gamma0() -> CMat {
    from_rows([[L, O, O, O], [O, L, O, O], [O, O, L, O], [O, O, O, L]])
}

pub fn gamma12() -> CMat {
    from_rows([[L, O, O, O], [O, L, O, O], [O, O, -L, O], [O, O, O, -L]])
}

pub fn gamma12p() -> CMat {
    from_rows([[L, O, O, O], [O, -L, O, O], [O, O, -L, O], [O, O, O, L]])
}

pub fn gamma13() -> CMat {
    from_rows([[O, O, I, O], [O, O, O, I], [-I, O, O, O], [O, -I, O, O]])
}

pub fn gamma14() -> CMat {
    from_rows([[O, O, L, O], [O, O, O, -L], [L, O, O, O], [O, -L, O, O]])
}

pub fn gamma23() -> CMat {
    from_rows([[O, O, L, O], [O, O, O, L], [L, O, O, O], [O, L, O, O]])
}

pub fn gamma23p() -> CMat {
    from_rows([[O, O, L, O], [O, O, O, -L], [L, O, O, O], [O, -L, O, O]])
}

pub fn gamma31() -> CMat {
    from_rows([[O, O, -I, O], [O, O, O, -I], [I, O, O, O], [O, I, O, O]])
}

pub fn gamma31p() -> CMat {
    from_rows([[O, O, -I, O], [O, O, O, I], [I, O, O, O], [O, -I, O, O]])
}

pub fn gamma33() -> CMat {
    from_rows([[O, L, O, O], [L, O, O, O], [O, O, O, -L], [O, O, -L, O]])
}

pub fn gamma44() -> CMat {
    from_rows([[O, -I, O, O], [I, O, O, O], [O, O, O, -I], [O, O, I, O]])
}

pub fn gamma55() -> CMat {
    from_rows([[L, O, O, O], [O, -L, O, O], [O, O, L, O], [O, O, O, -L]])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mul(a: &CMat, b: &CMat) -> CMat {
        a * b
    }

    fn close(a: &CMat, b: &CMat) -> bool {
        (0..4).all(|i| (0..4).all(|j| (a[(i, j)] - b[(i, j)]).norm() < 1e-15))
    }

    #[test]
    fn hermitian_and_involutive() {
        for g in [
            gamma12(),
            gamma12p(),
            gamma13(),
            gamma14(),
            gamma23(),
            gamma23p(),
            gamma31(),
            gamma31p(),
            gamma33(),
            gamma44(),
            gamma55(),
        ] {
            assert!(close(&g, &g.adjoint().to_owned()));
            assert!(close(&mul(&g, &g), &gamma0()));
        }
    }

    #[test]
    fn spin_algebra() {
        // s_x s_y = i s_z, for both couplings.
        let i = c64::new(0.0, 1.0);
        assert!(close(&mul(&gamma23(), &gamma31()), &(&gamma12() * faer::Scale(i))));
        // opposite-g: (s_x τ_z)(s_y τ_z) = i s_z
        assert!(close(&mul(&gamma23p(), &gamma31p()), &(&gamma12() * faer::Scale(i))));
        // gamma13 = -gamma31
        assert!(close(&gamma13(), &(&gamma31() * faer::Scale(c64::new(-1.0, 0.0)))));
    }
}
