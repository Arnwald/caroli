//! Reproducible random streams.
//!
//! caroli draws all its randomness from xoshiro256** (Blackman & Vigna, 2018), a small, fast
//! generator with a fixed definition, so that a seed gives the same disorder on every platform
//! and version. [`Xoshiro256::gfortran`] reproduces the stream of gfortran ≥ 9 after
//! `random_seed(put = [seed, seed, …])`, the seeding used by the thesis codes: their disorder
//! configurations can be rebuilt exactly.

/// xoshiro256** generator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Xoshiro256 {
    s: [u64; 4],
}

/// Keys gfortran XORs into a user seed (`scramble_seed`, libgfortran/intrinsics/random.c).
const GFORTRAN_KEYS: [u64; 4] =
    [0xbd0c5b6e50c2df49, 0xd46061cd46e1df38, 0xbb4f4d4ed6103544, 0x114a583d0756ad39];

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

impl Xoshiro256 {
    /// Generator seeded from a 64-bit value (state filled by splitmix64, as recommended by the
    /// authors of xoshiro).
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut sm = seed;
        Self { s: [splitmix64(&mut sm), splitmix64(&mut sm), splitmix64(&mut sm), splitmix64(&mut sm)] }
    }

    /// Generator in the state gfortran ≥ 9 reaches after `random_seed(put = put)` (`put` has
    /// 8 elements; the thesis codes fill it with one value, see [`Self::gfortran`]).
    pub fn gfortran_put(put: [i32; 8]) -> Self {
        // gfortran copies put(8), put(7), …, put(1) into consecutive 32-bit words of the state,
        // then XORs the four 64-bit words with fixed keys.
        let w = |k: usize| put[k] as u32 as u64;
        let raw = [w(7) | w(6) << 32, w(5) | w(4) << 32, w(3) | w(2) << 32, w(1) | w(0) << 32];
        Self { s: std::array::from_fn(|i| raw[i] ^ GFORTRAN_KEYS[i]) }
    }

    /// Generator in the state of the thesis routine `random_set(seed)`, i.e. gfortran's
    /// `random_seed(put = [seed, seed, …])`.
    pub fn gfortran(seed: i32) -> Self {
        Self::gfortran_put([seed; 8])
    }

    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform number in `[0, 1)` with 53 random bits (identical to gfortran's
    /// `random_number` for `real(8)`).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// `n` uniform numbers (like `random_number` on an array of size `n`).
    pub fn uniforms(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.uniform()).collect()
    }

    /// Integer uniform in `lo..=hi`, drawn as `lo + ⌊(hi − lo + 1) u⌋` (thesis
    /// `random_integer`).
    pub fn integer(&mut self, lo: usize, hi: usize) -> usize {
        lo + ((hi - lo + 1) as f64 * self.uniform()) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values printed by gfortran 16 (`random_seed(put = …)` then `random_number`).
    #[test]
    #[allow(clippy::excessive_precision)]
    fn reproduces_gfortran() {
        let mut r = Xoshiro256::gfortran(12345);
        let expected = [
            4.65636743897908767e-01,
            1.00833111555351462e-01,
            8.41914420882911130e-01,
            7.73831493534653414e-01,
            1.80372332904081167e-01,
            8.44873831312145440e-01,
        ];
        for e in expected {
            assert_eq!(r.uniform(), e);
        }
        assert_eq!(Xoshiro256::gfortran(1237).uniform(), 4.72208155982870581e-01);
        assert_eq!(Xoshiro256::gfortran_put([1, 2, 3, 4, 5, 6, 7, 8]).uniform(), 4.71070899360821249e-01);
    }

    #[test]
    fn seeds_differ() {
        let a = Xoshiro256::seed_from_u64(1).uniforms(4);
        let b = Xoshiro256::seed_from_u64(2).uniforms(4);
        assert_ne!(a, b);
        assert!(a.iter().all(|x| (0.0..1.0).contains(x)));
    }
}
