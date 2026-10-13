//! From the input to caroli's objects: parameters, model, disorder, sample and leads.

use crate::Error;
use crate::input::*;
use caroli::disorder::OnSiteMap;
use caroli::model::continuum::{self, Continuum};
use caroli::model::presets::{BhzParams, GFactors, Spin, bhz, bhz_block, bhz_spin};
use caroli::prelude::*;
use std::collections::BTreeMap;

/// Resolved parameters.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Params(pub BTreeMap<String, f64>);

impl Params {
    /// The parameters of the file with `overrides`: expressions may use the other parameters, in
    /// any order (no cycle).
    pub fn resolve(raw: &BTreeMap<String, Num>, overrides: &BTreeMap<String, f64>) -> Result<Self, Error> {
        for name in overrides.keys() {
            if !raw.contains_key(name) {
                return Err(format!("unknown parameter `{name}` (not in [parameters])").into());
            }
        }
        let mut done: BTreeMap<String, f64> = overrides.clone();
        let mut pending: Vec<(&String, &Num)> = raw.iter().filter(|(k, _)| !done.contains_key(*k)).collect();
        while !pending.is_empty() {
            let before = pending.len();
            let mut last_error = None;
            pending.retain(|(name, value)| match Params(done.clone()).num(value) {
                Ok(x) => {
                    done.insert((*name).clone(), x);
                    false
                }
                Err(e) => {
                    last_error = Some(format!("parameter `{name}`: {e}"));
                    true
                }
            });
            if pending.len() == before {
                return Err(last_error.unwrap_or_default().into());
            }
        }
        Ok(Self(done))
    }

    fn list(&self) -> Vec<(&str, f64)> {
        self.0.iter().map(|(k, v)| (k.as_str(), *v)).collect()
    }

    pub fn num(&self, n: &Num) -> Result<f64, Error> {
        match n {
            Num::Value(x) => Ok(*x),
            Num::Expr(e) => continuum::number(e, &self.list()).map_err(|err| format!("`{e}`: {err}").into()),
        }
    }

    pub fn opt(&self, n: &Option<Num>, default: f64) -> Result<f64, Error> {
        n.as_ref().map_or(Ok(default), |n| self.num(n))
    }

    pub fn matrix(&self, e: &str) -> Result<CMat, Error> {
        continuum::evaluate(e, &self.list()).map_err(|err| format!("`{e}`: {err}").into())
    }

    /// The values of a range, a list or a number.
    pub fn values(&self, v: &Values) -> Result<Vec<f64>, Error> {
        match v {
            Values::One(n) => Ok(vec![self.num(n)?]),
            Values::List(l) => l.iter().map(|n| self.num(n)).collect(),
            Values::Range(r) => {
                let (from, to) = (self.num(&r.from)?, self.num(&r.to)?);
                match (r.count, &r.step) {
                    (Some(n), None) => Ok((0..n)
                        .map(|k| if n == 1 { from } else { from + (to - from) * k as f64 / (n - 1) as f64 })
                        .collect()),
                    (None, Some(step)) => {
                        let step = self.num(step)?;
                        if step.is_nan() || step <= 0.0 {
                            return Err("a range step must be positive".into());
                        }
                        let n = ((to - from) / step + 1e-9).floor() as usize + 1;
                        Ok((0..n).map(|k| from + step * k as f64).collect())
                    }
                    _ => Err("a range needs either `count` or `step`".into()),
                }
            }
        }
    }
}

/// The points of a scan: every combination of the scanned parameters, with the parameters of the
/// file resolved for each (expressions follow the scanned values).
pub fn scan_points(
    raw: &BTreeMap<String, Num>,
    base: &BTreeMap<String, f64>,
    scan: &BTreeMap<String, Values>,
) -> Result<Vec<(Vec<f64>, Params)>, Error> {
    let p0 = Params::resolve(raw, base)?;
    let axes: Vec<(String, Vec<f64>)> =
        scan.iter().map(|(k, v)| Ok((k.clone(), p0.values(v)?))).collect::<Result<_, Error>>()?;
    let mut points = vec![Vec::new()];
    for (_, values) in &axes {
        points = points
            .into_iter()
            .flat_map(|p| values.iter().map(move |v| [p.clone(), vec![*v]].concat()))
            .collect();
    }
    points
        .into_iter()
        .map(|values| {
            let mut overrides = base.clone();
            for ((name, _), v) in axes.iter().zip(&values) {
                overrides.insert(name.clone(), *v);
            }
            Ok((values, Params::resolve(raw, &overrides)?))
        })
        .collect()
}

fn vec3(v: &[f64]) -> [f64; 3] {
    std::array::from_fn(|i| v.get(i).copied().unwrap_or(0.0))
}

fn cell3(v: &[i64]) -> [i64; 3] {
    std::array::from_fn(|i| v.get(i).copied().unwrap_or(0))
}

/// The clean model (without disorder).
pub fn model(m: &ModelInput, p: &Params) -> Result<Model, Error> {
    let sources =
        [m.preset.is_some(), m.continuum.is_some(), m.lattice.is_some()].iter().filter(|x| **x).count();
    if sources != 1 {
        return Err("[model] needs exactly one of `preset`, `continuum` and `lattice`".into());
    }
    let spacing = p.opt(&m.spacing, 1.0)?;
    let t = p.opt(&m.t, 1.0)?;
    let spin = |x: CMat| if m.spin { kron(&x, &s0()) } else { x };
    let scalar = |x: f64| spin(CMat::from_fn(1, 1, |_, _| c64::new(x, 0.0)));
    let preset_only = |name: &str, given: bool| -> Result<(), Error> {
        if given { Err(format!("`{name}` does not apply to this model").into()) } else { Ok(()) }
    };
    let bhz_options = [&m.a, &m.b, &m.c, &m.d, &m.mass, &m.lambda].iter().any(|x| x.is_some());
    let mut model = if let Some(preset) = &m.preset {
        let bhz_params = || -> Result<BhzParams, Error> {
            let d = BhzParams::default();
            Ok(BhzParams {
                a: p.opt(&m.a, d.a)?,
                b: p.opt(&m.b, d.b)?,
                c: p.opt(&m.c, d.c)?,
                d: p.opt(&m.d, d.d)?,
                mass: p.opt(&m.mass, d.mass)?,
                lambda: p.opt(&m.lambda, d.lambda)?,
            })
        };
        if !preset.starts_with("bhz") {
            preset_only("a, b, c, d, mass, lambda", bhz_options)?;
        } else {
            preset_only("t", m.t.is_some())?;
            preset_only("spin", m.spin)?;
        }
        if preset != "qwz" {
            preset_only("m", m.m.is_some())?;
        }
        match preset.as_str() {
            "bhz" => bhz(bhz_params()?),
            "bhz-up" => bhz_block(bhz_params()?, Spin::Up),
            "bhz-down" => bhz_block(bhz_params()?, Spin::Down),
            "graphene" => {
                let h = scalar(-t);
                Model::new(Lattice::honeycomb(spacing, if m.spin { 2 } else { 1 }))
                    .hopping(0, 1, [0, 0, 0], &h)
                    .hopping(0, 1, [0, -1, 0], &h)
                    .hopping(0, 1, [1, -1, 0], &h)
            }
            "chain" | "square" | "cubic" => {
                let orbitals = if m.spin { 2 } else { 1 };
                let lattice = match preset.as_str() {
                    "chain" => Lattice::new(&[[spacing, 0.0, 0.0]]).with_sublattice("a", [0.0; 3], orbitals),
                    "square" => Lattice::square(spacing, orbitals),
                    _ => Lattice::cubic(spacing, orbitals),
                };
                let dim = lattice.dim();
                (0..dim).fold(Model::new(lattice), |model, a| {
                    model.hopping(0, 0, std::array::from_fn(|c| (c == a) as i64), scalar(-t))
                })
            }
            "qwz" => {
                preset_only("spin", m.spin)?;
                let mass = p.opt(&m.m, 1.0)?;
                let i = c64::new(0.0, 1.0);
                let half = c64::new(0.5, 0.0);
                Model::new(Lattice::square(spacing, 2))
                    .onsite(0, combine(&[(mass, &sz())]))
                    .hopping(0, 0, [1, 0, 0], combine(&[(half, &sz()), (-i * 0.5, &sx())]))
                    .hopping(0, 0, [0, 1, 0], combine(&[(half, &sz()), (-i * 0.5, &sy())]))
            }
            other => return Err(format!("unknown preset `{other}`").into()),
        }
    } else if let Some(expr) = &m.continuum {
        preset_only("a, b, c, d, mass, lambda", bhz_options)?;
        Continuum::parse(expr, &p.list()).map_err(|e| format!("continuum `{expr}`: {e}"))?.discretize(spacing)
    } else {
        let l = m.lattice.as_ref().expect("lattice");
        if l.vectors.is_empty() || l.vectors.len() > 3 {
            return Err("a lattice needs 1 to 3 vectors".into());
        }
        let vectors: Vec<[f64; 3]> = l.vectors.iter().map(|v| vec3(v)).collect();
        let lattice = l.sublattices.iter().enumerate().fold(Lattice::new(&vectors), |lat, (k, s)| {
            lat.with_sublattice(s.name.as_deref().unwrap_or(&format!("s{k}")), vec3(&s.position), s.orbitals)
        });
        Model::new(lattice)
    };
    let subs = model.lattice().sublattices().len();
    for o in &m.onsite {
        if o.sublattice >= subs {
            return Err(format!("on-site term on sublattice {} of {subs}", o.sublattice).into());
        }
        let x = p.matrix(&o.matrix)?;
        let n = model.lattice().sublattices()[o.sublattice].orbitals;
        if x.nrows() != n {
            return Err(format!("on-site `{}` of size {} for {n} orbitals", o.matrix, x.nrows()).into());
        }
        if m.lattice.is_some() {
            model = model.onsite(o.sublattice, x);
        } else {
            let sub = o.sublattice;
            model = model.with_onsite_term(move |s: &Site| (s.sub == sub).then(|| x.clone()));
        }
    }
    let dim = model.lattice().dim();
    for h in &m.hoppings {
        if h.from >= subs || h.to >= subs || h.offset.len() != dim {
            return Err(
                format!("hopping {h:?}: sublattices below {subs}, an offset of {dim} integers").into()
            );
        }
        let x = p.matrix(&h.matrix)?;
        let orb = |s: usize| model.lattice().sublattices()[s].orbitals;
        if (x.nrows(), x.ncols()) != (orb(h.from), orb(h.to)) {
            return Err(format!("hopping `{}` of size {}×{}", h.matrix, x.nrows(), x.ncols()).into());
        }
        model = model.hopping(h.from, h.to, cell3(&h.offset), x);
    }
    if let Some(b) = &m.field {
        model = model.with_field(p.num(b)?);
    }
    Ok(model)
}

/// The model with the disorder of configuration `config` on the sites of `region`.
pub fn disordered(input: &Input, p: &Params, region: &Region, config: usize) -> Result<Model, Error> {
    let mut model = model(&input.model, p)?;
    for (k, d) in input.disorder.iter().enumerate() {
        let map = disorder_map(d, p, region, config, k).map_err(|e| format!("[[disorder]] {k}: {e}"))?;
        model = model.with_onsite_term(map);
    }
    Ok(model)
}

fn disorder_map(
    d: &DisorderInput,
    p: &Params,
    region: &Region,
    config: usize,
    index: usize,
) -> Result<OnSiteMap, Error> {
    let lattice = region.lattice();
    let inside = |c: i64, r: &Option<[i64; 2]>| r.is_none_or(|[a, b]| (a..b).contains(&c));
    let cells = d.cells.clone().unwrap_or_default();
    let sites: Vec<Site> = region
        .sites()
        .iter()
        .copied()
        .filter(|s| {
            inside(s.cell[0], &cells.x)
                && inside(s.cell[1], &cells.y)
                && inside(s.cell[2], &cells.z)
                && d.sublattices.as_ref().is_none_or(|l| l.contains(&s.sub))
        })
        .collect();
    let seed = d.seed.unwrap_or(1 + 1000 * index as u64) + config as u64;
    let mut rng = Xoshiro256::seed_from_u64(seed);
    let mut map = OnSiteMap::default();
    match d.kind.as_str() {
        "anderson" => {
            let w = p.num(d.strength.as_ref().ok_or("anderson disorder needs `strength`")?)?;
            let a = Anderson::generate(&mut rng, &sites, w);
            for (s, e) in a.sites.iter().zip(&a.energies) {
                let n = lattice.sublattices()[s.sub].orbitals;
                map.add(
                    *s,
                    CMat::from_fn(n, n, |i, j| if i == j { c64::new(*e, 0.0) } else { c64::new(0.0, 0.0) }),
                );
            }
        }
        "magnetic" => {
            let conc = p.num(d.concentration.as_ref().ok_or("magnetic disorder needs `concentration`")?)?;
            let j = p.num(d.exchange.as_ref().ok_or("magnetic disorder needs `exchange`")?)?;
            let moments = moments(d, p)?;
            let imp =
                MagneticImpurities::generate(&mut rng, &sites, (conc * sites.len() as f64) as usize, moments);
            for (s, n) in imp.sites.iter().zip(&imp.moments) {
                let spin = spin_matrices(d, p, lattice.sublattices()[s.sub].orbitals)?;
                map.add(*s, combine(&[(j * n[0], &spin[0]), (j * n[1], &spin[1]), (j * n[2], &spin[2])]));
            }
        }
        other => return Err(format!("unknown disorder `{other}` (anderson, magnetic)").into()),
    }
    Ok(map)
}

pub fn moments(d: &DisorderInput, p: &Params) -> Result<Moments, Error> {
    Ok(match d.moments.as_deref().unwrap_or("isotropic") {
        "isotropic" => Moments::Isotropic,
        "polarized" | "polarised" => Moments::Polarized { mz: p.opt(&d.mz, 1.0)? },
        "cone" => Moments::Cone {
            theta0: p.num(d.theta0.as_ref().ok_or("cone moments need `theta0`")?)?.to_radians(),
        },
        other => return Err(format!("unknown moments `{other}` (isotropic, polarized, cone)").into()),
    })
}

/// The spin matrices of a magnetic disorder for sites of `orbitals` orbitals.
pub fn spin_matrices(d: &DisorderInput, p: &Params, orbitals: usize) -> Result<[CMat; 3], Error> {
    let s = match &d.spin {
        None if orbitals == 2 => [sx(), sy(), sz()],
        None => return Err(format!("give the `spin` matrices of sites with {orbitals} orbitals").into()),
        Some(SpinInput::Name(n)) => match n.as_str() {
            "pauli" => [sx(), sy(), sz()],
            "same-g" => bhz_spin(GFactors::Same),
            "opposite-g" => bhz_spin(GFactors::Opposite),
            other => {
                return Err(
                    format!("unknown spin `{other}` (pauli, same-g, opposite-g, or three matrices)").into()
                );
            }
        },
        Some(SpinInput::Matrices(m)) if m.len() == 3 => {
            [p.matrix(&m[0])?, p.matrix(&m[1])?, p.matrix(&m[2])?]
        }
        Some(SpinInput::Matrices(_)) => return Err("`spin` takes three matrices".into()),
    };
    if s.iter().any(|m| m.nrows() != orbitals) {
        return Err(format!("spin matrices of size {} for sites of {orbitals} orbitals", s[0].nrows()).into());
    }
    Ok(s)
}

/// The sample: `size` cells along each lattice direction.
pub fn region(model: &Model, system: &SystemInput) -> Result<Region, Error> {
    let dim = model.lattice().dim();
    let size = system.size.as_ref().ok_or("[system] needs `size`")?;
    if size.len() != dim || size.contains(&0) {
        return Err(format!("`size` needs {dim} positive numbers of cells").into());
    }
    Ok(Region::cuboid(model.lattice(), std::array::from_fn(|i| size.get(i).copied().unwrap_or(1))))
}

/// The periods of a torus of the sample.
pub fn periods(system: &SystemInput) -> Vec<[i64; 3]> {
    let size = system.size.clone().unwrap_or_default();
    (0..size.len()).map(|a| std::array::from_fn(|c| if c == a { size[a] as i64 } else { 0 })).collect()
}

/// Leads: two along one axis (the fast two-terminal solver), or any faces.
pub enum Leads {
    Axis(usize),
    Faces(Vec<Interface>),
}

pub fn leads(system: &SystemInput, region: &Region) -> Result<Leads, Error> {
    let names = system.leads.clone().unwrap_or_else(|| vec!["-x".into(), "+x".into()]);
    let dim = region.lattice().dim();
    let parsed: Vec<(i64, usize)> = names
        .iter()
        .map(|n| {
            let sign = match n.chars().next() {
                Some('-') => -1,
                Some('+') => 1,
                _ => return Err(format!("lead `{n}`: a face such as -x or +y")),
            };
            let axis = match &n[1..] {
                "x" => 0,
                "y" => 1,
                "z" => 2,
                _ => return Err(format!("lead `{n}`: a face such as -x or +y")),
            };
            if axis >= dim {
                return Err(format!("lead `{n}` on a lattice of dimension {dim}"));
            }
            Ok((sign, axis))
        })
        .collect::<Result<_, String>>()?;
    if parsed.len() < 2 {
        return Err("transport needs at least two leads".into());
    }
    if parsed.len() == 2 && parsed[0] == (-1, parsed[0].1) && parsed[1] == (1, parsed[0].1) {
        return Ok(Leads::Axis(parsed[0].1));
    }
    let interfaces = parsed
        .iter()
        .map(|&(sign, axis)| {
            let coords = region.sites().iter().map(|s| s.cell[axis]);
            let edge = if sign < 0 { coords.min() } else { coords.max() }.expect("sites");
            let sites: Vec<Site> = region.sites().iter().copied().filter(|s| s.cell[axis] == edge).collect();
            Interface {
                slice: Region::from_sites(region.lattice(), sites),
                outward: std::array::from_fn(|c| if c == axis { sign } else { 0 }),
            }
        })
        .collect();
    Ok(Leads::Faces(interfaces))
}
