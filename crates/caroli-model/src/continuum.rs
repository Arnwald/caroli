//! Continuum (k·p) Hamiltonians and their discretisation on a lattice, as Kwant's
//! `continuum.discretize`.
//!
//! An expression such as `"C - D*(k_x^2 + k_y^2) + (M - B*(k_x^2 + k_y^2))*kron(s0, sz)"` is a
//! polynomial in `(k_x, k_y, k_z)` with matrix coefficients. On a square (cubic) lattice of
//! constant `a`, `k → −i∂` and the derivatives become central finite differences:
//! `∂ → (T − T†)/2a`, `∂² → (T − 2 + T†)/a²`, higher powers by products of these, and mixed terms
//! by products over the directions.
//!
//! Syntax: numbers, the named parameters, `k_x k_y k_z` (or `kx ky kz`), the Pauli matrices
//! `s0 sx sy sz` (or `sigma_0`...), the imaginary unit `i`, `kron(a, b)` (index of `b` fastest),
//! `+ − * /` (division by numbers), integer powers `^` or `**`, parentheses. A number stands for
//! that multiple of the identity.

use crate::pauli::{s0, sx, sy, sz};
use crate::{CMat, Model, c64};
use caroli_lattice::Lattice;
use std::collections::BTreeMap;

/// Errors of [`Continuum::parse`].
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ContinuumError {
    #[error("unexpected `{0}` at position {1}")]
    Unexpected(String, usize),
    #[error("unknown name `{0}`")]
    Unknown(String),
    #[error("matrices of sizes {0} and {1} do not match")]
    Size(usize, usize),
    #[error("{0}")]
    Invalid(String),
}

/// A coefficient: a number (a multiple of the identity) or a matrix.
#[derive(Clone, Debug)]
enum Value {
    Number(c64),
    Matrix(CMat),
}

impl Value {
    fn size(&self) -> Option<usize> {
        match self {
            Value::Number(_) => None,
            Value::Matrix(m) => Some(m.nrows()),
        }
    }

    fn matrix(&self, n: usize) -> CMat {
        match self {
            Value::Number(x) => CMat::from_fn(n, n, |i, j| if i == j { *x } else { c64::new(0.0, 0.0) }),
            Value::Matrix(m) => m.clone(),
        }
    }

    fn add(&self, other: &Value) -> Result<Value, ContinuumError> {
        Ok(match (self, other) {
            (Value::Number(a), Value::Number(b)) => Value::Number(a + b),
            _ => {
                let n = size_of(self, other)?;
                Value::Matrix(&self.matrix(n) + &other.matrix(n))
            }
        })
    }

    fn mul(&self, other: &Value) -> Result<Value, ContinuumError> {
        Ok(match (self, other) {
            (Value::Number(a), Value::Number(b)) => Value::Number(a * b),
            (Value::Number(a), Value::Matrix(m)) | (Value::Matrix(m), Value::Number(a)) => {
                Value::Matrix(CMat::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)] * a))
            }
            (Value::Matrix(a), Value::Matrix(b)) => {
                size_of(self, other)?;
                Value::Matrix(a * b)
            }
        })
    }
}

fn size_of(a: &Value, b: &Value) -> Result<usize, ContinuumError> {
    match (a.size(), b.size()) {
        (Some(x), Some(y)) if x != y => Err(ContinuumError::Size(x, y)),
        (Some(x), _) | (_, Some(x)) => Ok(x),
        (None, None) => Ok(1),
    }
}

/// A polynomial in `(k_x, k_y, k_z)`: powers → coefficient.
type Poly = BTreeMap<[u32; 3], Value>;

fn add(a: &Poly, b: &Poly) -> Result<Poly, ContinuumError> {
    let mut out = a.clone();
    for (k, v) in b {
        let sum = match out.get(k) {
            Some(x) => x.add(v)?,
            None => v.clone(),
        };
        out.insert(*k, sum);
    }
    Ok(out)
}

fn mul(a: &Poly, b: &Poly) -> Result<Poly, ContinuumError> {
    let mut out = Poly::new();
    for (ka, va) in a {
        for (kb, vb) in b {
            let k = [ka[0] + kb[0], ka[1] + kb[1], ka[2] + kb[2]];
            let term = va.mul(vb)?;
            let sum = match out.get(&k) {
                Some(x) => x.add(&term)?,
                None => term,
            };
            out.insert(k, sum);
        }
    }
    Ok(out)
}

fn constant(v: Value) -> Poly {
    Poly::from([([0, 0, 0], v)])
}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Number(f64),
    Name(String),
    Symbol(char),
    Power,
}

fn tokens(text: &str) -> Result<Vec<(Token, usize)>, ContinuumError> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut k = 0;
    while k < chars.len() {
        let c = chars[k];
        if c.is_whitespace() {
            k += 1;
        } else if c.is_ascii_digit() || c == '.' {
            let start = k;
            while k < chars.len() && (chars[k].is_ascii_digit() || chars[k] == '.') {
                k += 1;
            }
            if k < chars.len() && (chars[k] == 'e' || chars[k] == 'E') {
                k += 1;
                if k < chars.len() && (chars[k] == '+' || chars[k] == '-') {
                    k += 1;
                }
                while k < chars.len() && chars[k].is_ascii_digit() {
                    k += 1;
                }
            }
            let s: String = chars[start..k].iter().collect();
            let x = s.parse().map_err(|_| ContinuumError::Unexpected(s.clone(), start))?;
            out.push((Token::Number(x), start));
        } else if c.is_alphabetic() || c == '_' {
            let start = k;
            while k < chars.len() && (chars[k].is_alphanumeric() || chars[k] == '_') {
                k += 1;
            }
            out.push((Token::Name(chars[start..k].iter().collect()), start));
        } else if c == '*' && chars.get(k + 1) == Some(&'*') {
            out.push((Token::Power, k));
            k += 2;
        } else if c == '^' {
            out.push((Token::Power, k));
            k += 1;
        } else if "+-*/(),".contains(c) {
            out.push((Token::Symbol(c), k));
            k += 1;
        } else {
            return Err(ContinuumError::Unexpected(c.to_string(), k));
        }
    }
    Ok(out)
}

struct Parser<'a> {
    tokens: Vec<(Token, usize)>,
    at: usize,
    params: &'a [(&'a str, f64)],
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.at).map(|t| &t.0)
    }

    fn unexpected(&self) -> ContinuumError {
        match self.tokens.get(self.at) {
            Some((t, pos)) => ContinuumError::Unexpected(format!("{t:?}"), *pos),
            None => ContinuumError::Invalid("unexpected end of the expression".into()),
        }
    }

    fn expect(&mut self, c: char) -> Result<(), ContinuumError> {
        if self.peek() == Some(&Token::Symbol(c)) {
            self.at += 1;
            Ok(())
        } else {
            Err(self.unexpected())
        }
    }

    /// expr := term (('+' | '−') term)*
    fn expr(&mut self) -> Result<Poly, ContinuumError> {
        let mut acc = self.term()?;
        while let Some(Token::Symbol(c @ ('+' | '-'))) = self.peek().cloned() {
            self.at += 1;
            let mut rhs = self.term()?;
            if c == '-' {
                rhs = mul(&rhs, &constant(Value::Number(c64::new(-1.0, 0.0))))?;
            }
            acc = add(&acc, &rhs)?;
        }
        Ok(acc)
    }

    /// term := factor (('*' | '/') factor)*
    fn term(&mut self) -> Result<Poly, ContinuumError> {
        let mut acc = self.factor()?;
        while let Some(Token::Symbol(c @ ('*' | '/'))) = self.peek().cloned() {
            self.at += 1;
            let rhs = self.factor()?;
            acc = if c == '*' {
                mul(&acc, &rhs)?
            } else {
                let divisor = match (rhs.len(), rhs.get(&[0, 0, 0])) {
                    (1, Some(Value::Number(x))) => *x,
                    _ => {
                        return Err(ContinuumError::Invalid(
                            "division by something else than a number".into(),
                        ));
                    }
                };
                mul(&acc, &constant(Value::Number(c64::new(1.0, 0.0) / divisor)))?
            };
        }
        Ok(acc)
    }

    /// factor := ('+' | '−') factor | atom ('^' integer)?
    fn factor(&mut self) -> Result<Poly, ContinuumError> {
        match self.peek() {
            Some(Token::Symbol('-')) => {
                self.at += 1;
                let f = self.factor()?;
                return mul(&f, &constant(Value::Number(c64::new(-1.0, 0.0))));
            }
            Some(Token::Symbol('+')) => {
                self.at += 1;
                return self.factor();
            }
            _ => {}
        }
        let base = self.atom()?;
        if self.peek() == Some(&Token::Power) {
            self.at += 1;
            let power = match self.peek() {
                Some(Token::Number(x)) if x.fract() == 0.0 && *x >= 0.0 => *x as u32,
                _ => return Err(ContinuumError::Invalid("powers must be non-negative integers".into())),
            };
            self.at += 1;
            let mut acc = constant(Value::Number(c64::new(1.0, 0.0)));
            for _ in 0..power {
                acc = mul(&acc, &base)?;
            }
            return Ok(acc);
        }
        Ok(base)
    }

    /// atom := number | name | name '(' expr ',' expr ')' | '(' expr ')'
    fn atom(&mut self) -> Result<Poly, ContinuumError> {
        match self.peek().cloned() {
            Some(Token::Number(x)) => {
                self.at += 1;
                Ok(constant(Value::Number(c64::new(x, 0.0))))
            }
            Some(Token::Symbol('(')) => {
                self.at += 1;
                let e = self.expr()?;
                self.expect(')')?;
                Ok(e)
            }
            Some(Token::Name(name)) => {
                self.at += 1;
                if name == "kron" {
                    self.expect('(')?;
                    let a = self.expr()?;
                    self.expect(',')?;
                    let b = self.expr()?;
                    self.expect(')')?;
                    return kron(&a, &b);
                }
                let k = |d: usize| {
                    let mut p = [0u32; 3];
                    p[d] = 1;
                    Ok(Poly::from([(p, Value::Number(c64::new(1.0, 0.0)))]))
                };
                match name.as_str() {
                    "k_x" | "kx" => k(0),
                    "k_y" | "ky" => k(1),
                    "k_z" | "kz" => k(2),
                    "i" => Ok(constant(Value::Number(c64::new(0.0, 1.0)))),
                    "s0" | "sigma_0" => Ok(constant(Value::Matrix(s0()))),
                    "sx" | "sigma_x" => Ok(constant(Value::Matrix(sx()))),
                    "sy" | "sigma_y" => Ok(constant(Value::Matrix(sy()))),
                    "sz" | "sigma_z" => Ok(constant(Value::Matrix(sz()))),
                    _ => match self.params.iter().find(|(p, _)| *p == name) {
                        Some((_, x)) => Ok(constant(Value::Number(c64::new(*x, 0.0)))),
                        None => Err(ContinuumError::Unknown(name)),
                    },
                }
            }
            _ => Err(self.unexpected()),
        }
    }
}

/// `kron(a, b)` of two constant matrix expressions.
fn kron(a: &Poly, b: &Poly) -> Result<Poly, ContinuumError> {
    let only = |p: &Poly| -> Result<CMat, ContinuumError> {
        match (p.len(), p.get(&[0, 0, 0])) {
            (1, Some(Value::Matrix(m))) => Ok(m.clone()),
            (1, Some(Value::Number(x))) => Ok(CMat::from_fn(1, 1, |_, _| *x)),
            _ => Err(ContinuumError::Invalid("kron takes constant matrices".into())),
        }
    };
    Ok(constant(Value::Matrix(crate::pauli::kron(&only(a)?, &only(b)?))))
}

/// A continuum (k·p) Hamiltonian: a polynomial in `(k_x, k_y, k_z)` with matrix coefficients.
#[derive(Clone, Debug)]
pub struct Continuum {
    orbitals: usize,
    terms: BTreeMap<[u32; 3], CMat>,
}

impl Continuum {
    /// Parses `expression` with the named numeric `params` (see the module documentation).
    pub fn parse(expression: &str, params: &[(&str, f64)]) -> Result<Self, ContinuumError> {
        let mut parser = Parser { tokens: tokens(expression)?, at: 0, params };
        let poly = parser.expr()?;
        if parser.at != parser.tokens.len() {
            return Err(parser.unexpected());
        }
        let mut orbitals: Option<usize> = None;
        for v in poly.values() {
            if let Some(n) = v.size() {
                match orbitals {
                    Some(m) if m != n => return Err(ContinuumError::Size(m, n)),
                    _ => orbitals = Some(n),
                }
            }
        }
        let orbitals = orbitals.unwrap_or(1);
        Ok(Self { orbitals, terms: poly.iter().map(|(k, v)| (*k, v.matrix(orbitals))).collect() })
    }

    /// Number of orbitals per site (the size of the matrices).
    pub fn orbitals(&self) -> usize {
        self.orbitals
    }

    /// Number of directions in which `k` appears (1, 2 or 3).
    pub fn dim(&self) -> usize {
        self.terms.keys().map(|p| (0..3).rev().find(|&d| p[d] > 0).map_or(1, |d| d + 1)).max().unwrap_or(1)
    }

    /// Tight-binding model on a lattice of constant `a`: a chain, square or cubic lattice of
    /// [`Self::dim`] dimensions.
    pub fn discretize(&self, a: f64) -> Model {
        self.discretize_in(a, self.dim())
    }

    /// Tight-binding model on the `dim`-dimensional hypercubic lattice of constant `a`.
    ///
    /// Panics if the Hamiltonian is not Hermitian (an on-site matrix that is not Hermitian, or
    /// opposite hoppings that are not conjugate).
    pub fn discretize_in(&self, a: f64, dim: usize) -> Model {
        assert!(
            (1..=3).contains(&dim) && dim >= self.dim(),
            "a lattice of {dim} dimensions cannot hold k in {} directions",
            self.dim()
        );
        let n = self.orbitals;
        // 1D stencils of ∂^p: offset → weight.
        let stencil = |p: u32| -> BTreeMap<i64, f64> {
            let d1 = BTreeMap::from([(-1, -0.5 / a), (1, 0.5 / a)]);
            let d2 = BTreeMap::from([(-1, 1.0 / (a * a)), (0, -2.0 / (a * a)), (1, 1.0 / (a * a))]);
            let convolve = |x: &BTreeMap<i64, f64>, y: &BTreeMap<i64, f64>| {
                let mut out = BTreeMap::new();
                for (i, u) in x {
                    for (j, v) in y {
                        *out.entry(i + j).or_insert(0.0) += u * v;
                    }
                }
                out
            };
            let mut s = if p % 2 == 1 { d1 } else { BTreeMap::from([(0, 1.0)]) };
            for _ in 0..p / 2 {
                s = convolve(&s, &d2);
            }
            s
        };
        let mut blocks: BTreeMap<[i64; 3], CMat> = BTreeMap::new();
        for (power, coefficient) in &self.terms {
            // k → −i∂.
            let total: u32 = power.iter().sum();
            let phase = (0..total).fold(c64::new(1.0, 0.0), |acc, _| acc * c64::new(0.0, -1.0));
            let mut offsets: Vec<([i64; 3], f64)> = vec![([0, 0, 0], 1.0)];
            for (d, &p) in power.iter().enumerate() {
                let s = stencil(p);
                offsets = offsets
                    .iter()
                    .flat_map(|(o, w)| {
                        s.iter().map(move |(&step, &ws)| {
                            let mut o2 = *o;
                            o2[d] += step;
                            (o2, w * ws)
                        })
                    })
                    .collect();
            }
            for (o, w) in offsets {
                let block = blocks.entry(o).or_insert_with(|| CMat::zeros(n, n));
                *block += CMat::from_fn(n, n, |i, j| coefficient[(i, j)] * phase * w);
            }
        }
        let vectors: Vec<[f64; 3]> =
            (0..dim).map(|d| std::array::from_fn(|c| if c == d { a } else { 0.0 })).collect();
        let lattice = Lattice::new(&vectors).with_sublattice("A", [0.0; 3], n);
        let zero = CMat::zeros(n, n);
        let onsite = blocks.get(&[0, 0, 0]).cloned().unwrap_or_else(|| zero.clone());
        let scale = blocks.values().map(|m| m.norm_max()).fold(0.0, f64::max).max(f64::MIN_POSITIVE);
        assert!(
            (&onsite - onsite.adjoint()).norm_max() <= 1e-12 * scale,
            "the on-site matrix is not Hermitian"
        );
        let mut model = Model::new(lattice).onsite(0, onsite);
        for (o, m) in &blocks {
            let positive = o.iter().find(|&&c| c != 0).is_some_and(|&c| c > 0);
            if !positive || m.norm_max() == 0.0 {
                continue;
            }
            let back = blocks.get(&[-o[0], -o[1], -o[2]]).unwrap_or(&zero);
            assert!(
                (back - m.adjoint()).norm_max() <= 1e-12 * scale,
                "the hoppings {o:?} and their opposites are not conjugate: the Hamiltonian is not Hermitian"
            );
            model = model.hopping(0, 0, *o, m.clone());
        }
        model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets::{BhzParams, bhz};
    use caroli_lattice::Region;

    /// The thesis BHZ preset is the central-difference discretisation of the continuum BHZ
    /// Hamiltonian `C − D k² + (M − B k²) τ_z + A (k_x s_z τ_x − k_y τ_y)` on a lattice of
    /// constant 1.
    #[test]
    fn bhz_from_its_continuum_form() {
        let p = BhzParams::default();
        let params = [("A", p.a), ("B", p.b), ("C", p.c), ("D", p.d), ("M", p.mass)];
        let h = Continuum::parse(
            "C - D*(k_x^2 + k_y^2) + (M - B*(k_x**2 + k_y**2))*kron(s0, sz) + A*(k_x*kron(sz, sx) - k_y*kron(s0, sy))",
            &params,
        )
        .unwrap();
        assert_eq!((h.orbitals(), h.dim()), (4, 2));
        let model = h.discretize(1.0);
        let region = Region::cuboid(model.lattice(), [4, 3, 0]);
        let reference = bhz(p);
        let diff = (model.hamiltonian(&region)
            - reference.hamiltonian(&Region::cuboid(reference.lattice(), [4, 3, 0])))
        .norm_max();
        assert!(diff < 1e-14, "BHZ: {diff:e}");
    }

    /// A free particle: `k²` gives the on-site 4/a² and the hoppings −1/a²; its band
    /// `(4 − 2 cos k_x a − 2 cos k_y a)/a²` follows `k²` at small `k`.
    #[test]
    fn free_particle_and_errors() {
        let a = 0.5;
        let model = Continuum::parse("k_x^2 + k_y^2", &[]).unwrap().discretize(a);
        assert!(
            (model.onsite_matrix(&caroli_lattice::Site { cell: [0, 0, 0], sub: 0 })[(0, 0)].re
                - 4.0 / (a * a))
                .abs()
                < 1e-14
        );
        assert_eq!(model.hoppings().len(), 2);
        assert!(model.hoppings().iter().all(|h| (h.matrix[(0, 0)].re + 1.0 / (a * a)).abs() < 1e-14));
        // Third powers, mixed terms and a 3D lattice.
        let h = Continuum::parse("k_x^3 + 2*k_x*k_y + k_z", &[]).unwrap();
        assert_eq!(h.dim(), 3);
        let m = h.discretize(1.0);
        assert!(
            m.hoppings().iter().any(|hop| hop.offset == [2, 0, 0])
                && m.hoppings().iter().any(|hop| hop.offset == [1, 1, 0])
        );
        assert!(matches!(Continuum::parse("k_x + q", &[]), Err(ContinuumError::Unknown(_))));
        assert!(matches!(Continuum::parse("sx + kron(sx, sz)", &[]), Err(ContinuumError::Size(2, 4))));
        assert!(Continuum::parse("(k_x", &[]).is_err());
    }

    #[test]
    #[should_panic(expected = "not Hermitian")]
    fn non_hermitian_terms() {
        let _ = Continuum::parse("i*k_x^2", &[]).unwrap().discretize(1.0);
    }
}
