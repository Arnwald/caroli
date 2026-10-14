//! The calculations: each writes one CSV table, a row per scan point, configuration and value.

use crate::Error;
use crate::build::{self, Leads, Params};
use crate::input::*;
use caroli::born::{Disorder, Scba};
use caroli::dynamics::{Evolution, Torus};
use caroli::kpm::Spectrum;
use caroli::prelude::*;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// A finished calculation: its table and a one-line summary.
pub struct Table {
    pub csv: String,
    pub summary: String,
}

/// A scan point: the values of the scanned parameters and every parameter.
type Point = (Vec<f64>, Params);

/// Header columns of the scanned parameters.
fn scan_header(scan: &BTreeMap<String, Values>) -> String {
    scan.keys().map(|k| format!("{k},")).collect()
}

fn scan_prefix(values: &[f64]) -> String {
    values.iter().map(|v| format!("{v},")).collect()
}

fn fmt(x: f64) -> String {
    format!("{x:e}")
}

/// Runs `job` on every pair (scan point, configuration) in parallel; the rows keep their order.
fn over_points<F>(points: &[Point], configs: usize, job: F) -> Result<String, Error>
where
    F: Fn(&Point, usize) -> Result<String, Error> + Sync,
{
    let jobs: Vec<(usize, usize)> =
        (0..points.len()).flat_map(|p| (0..configs).map(move |c| (p, c))).collect();
    let rows: Vec<String> =
        jobs.par_iter().map(|&(p, c)| job(&points[p], c)).collect::<Result<_, Error>>()?;
    Ok(rows.concat())
}

/// The calculation `run` of `input` with the parameters `base`.
pub fn run(input: &Input, run: &RunInput, base: &BTreeMap<String, f64>) -> Result<Table, Error> {
    match run {
        RunInput::Transmission(r) => transmission(input, r, base),
        RunInput::Local(r) => local(input, r, base),
        RunInput::Bands(r) => bands(input, r, base),
        RunInput::Spectrum(r) => spectrum(input, r, base),
        RunInput::Dos(r) => dos(input, r, base),
        RunInput::Conductivity(r) => conductivity(input, r, base),
        RunInput::Scba(r) => scba(input, r, base),
        RunInput::Evolution(r) => evolution(input, r, base),
    }
}

/// The clean model, the sample, and the disordered model of configuration `c`.
fn system(input: &Input, p: &Params, c: usize) -> Result<(Model, Region, Model), Error> {
    let clean = build::model(&input.model, p)?;
    let region = build::region(&clean, &input.system)?;
    let device = build::disordered(input, p, &region, c)?;
    Ok((clean, region, device))
}

fn transmission(input: &Input, r: &TransmissionRun, base: &BTreeMap<String, f64>) -> Result<Table, Error> {
    let points = build::scan_points(&input.parameters, base, &r.scan)?;
    let configs = r.configurations.unwrap_or(1);
    let (_, region, _) = system(input, &points[0].1, 0)?;
    let leads = build::leads(&input.system, &region)?;
    let two = matches!(leads, Leads::Axis(_)) && r.sectors.is_none();
    let count = match &leads {
        Leads::Axis(_) => 2,
        Leads::Faces(f) => f.len(),
    };
    let mut header = format!("{}config,energy", scan_header(&r.scan));
    if two {
        header += ",T";
    } else if let Some(s) = &r.sectors {
        let sectors: Vec<usize> =
            s.iter().copied().collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        for p in 0..count {
            for &a in &sectors {
                for q in 0..count {
                    for &b in &sectors {
                        if (p, a) != (q, b) {
                            write!(header, ",T_{p}.{a}_{q}.{b}").unwrap();
                        }
                    }
                }
            }
        }
    } else {
        for p in 0..count {
            for q in 0..count {
                if p != q {
                    write!(header, ",T_{p}_{q}").unwrap();
                }
            }
        }
    }
    header.push('\n');
    let mut csv = header;
    for point in &points {
        let energies = point.1.values(&r.energies)?;
        let (clean, region, first) = system(input, &point.1, 0)?;
        let leads = build::leads(&input.system, &region)?;
        let rows = match leads {
            Leads::Axis(axis) if r.sectors.is_none() => {
                // The leads are clean: one set of self-energies per energy serves every configuration.
                let reference = NestedTwoTerminal::from_model(&first, &clean, &region, axis);
                let sigma: Vec<_> = energies.par_iter().map(|&e| reference.self_energies(e)).collect();
                drop(reference);
                over_points(std::slice::from_ref(point), configs, |(values, p), c| {
                    let (clean, region, device) = system(input, p, c)?;
                    let sys = NestedTwoTerminal::from_model(&device, &clean, &region, axis);
                    let mut out = String::new();
                    for (e, s) in energies.iter().zip(&sigma) {
                        let t =
                            s.as_ref().ok().and_then(|s| sys.transmission_with(s).ok()).unwrap_or(f64::NAN);
                        writeln!(out, "{}{c},{e},{}", scan_prefix(values), fmt(t)).unwrap();
                    }
                    Ok(out)
                })?
            }
            leads => {
                let interfaces = match leads {
                    Leads::Faces(f) => f,
                    Leads::Axis(axis) => face_pair(&region, axis),
                };
                over_points(std::slice::from_ref(point), configs, |(values, p), c| {
                    let (clean, region, device) = system(input, p, c)?;
                    let sys = NestedMultiTerminal::from_model(&device, &clean, &region, &interfaces);
                    let mut out = String::new();
                    for &e in &energies {
                        let cells: Vec<f64> = match &r.sectors {
                            None => match sys.transmissions(e) {
                                Ok(t) => (0..count)
                                    .flat_map(|p| (0..count).filter(move |q| *q != p).map(move |q| (p, q)))
                                    .map(|(p, q)| t[p][q])
                                    .collect(),
                                Err(_) => vec![f64::NAN; count * (count - 1)],
                            },
                            Some(s) => {
                                let labels = vec![Some(s.clone()); count];
                                let labs: Vec<usize> = s
                                    .iter()
                                    .copied()
                                    .collect::<std::collections::BTreeSet<_>>()
                                    .into_iter()
                                    .collect();
                                let n = labs.len();
                                match sys.sector_transmissions(e, &labels) {
                                    Ok(st) => {
                                        let at = |l: (usize, usize)| st.labels.iter().position(|x| *x == l);
                                        let mut v = Vec::new();
                                        for p in 0..count {
                                            for &a in &labs {
                                                for q in 0..count {
                                                    for &b in &labs {
                                                        if (p, a) != (q, b) {
                                                            v.push(match (at((p, a)), at((q, b))) {
                                                                (Some(k), Some(l)) => st.t[k][l],
                                                                _ => 0.0,
                                                            });
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        v
                                    }
                                    Err(_) => vec![f64::NAN; (count * n) * (count * n - 1)],
                                }
                            }
                        };
                        let cells: Vec<String> = cells.into_iter().map(fmt).collect();
                        writeln!(out, "{}{c},{e},{}", scan_prefix(values), cells.join(",")).unwrap();
                    }
                    Ok(out)
                })?
            }
        };
        csv.push_str(&rows);
    }
    let summary =
        format!("transmission: {} scan points × {configs} configurations, {count} leads", points.len());
    Ok(Table { csv, summary })
}

/// The two end faces of `axis`.
fn face_pair(region: &Region, axis: usize) -> Vec<Interface> {
    [-1i64, 1]
        .iter()
        .map(|&sign| {
            let coords = region.sites().iter().map(|s| s.cell[axis]);
            let edge = if sign < 0 { coords.min() } else { coords.max() }.expect("sites");
            let sites: Vec<Site> = region.sites().iter().copied().filter(|s| s.cell[axis] == edge).collect();
            Interface {
                slice: Region::from_sites(region.lattice(), sites),
                outward: std::array::from_fn(|c| if c == axis { sign } else { 0 }),
            }
        })
        .collect()
}

/// Scattering states of a two-terminal or multi-terminal device.
enum Device {
    Two(NestedTwoTerminal),
    Multi(NestedMultiTerminal),
}

impl Device {
    fn states(&self, e: f64) -> Result<ScatteringStates, Error> {
        Ok(match self {
            Device::Two(d) => d.scattering_states(e)?,
            Device::Multi(d) => d.scattering_states(e)?,
        })
    }
    fn bonds(&self, s: &ScatteringStates, f: &[f64], op: Option<&CMat>) -> Vec<BondCurrent> {
        match (self, op) {
            (Device::Two(d), None) => d.bond_currents(s, f),
            (Device::Two(d), Some(o)) => d.operator_currents(s, f, o),
            (Device::Multi(d), None) => d.bond_currents(s, f),
            (Device::Multi(d), Some(o)) => d.operator_currents(s, f, o),
        }
    }
    fn densities(&self, s: &ScatteringStates, f: &[f64], op: Option<&CMat>) -> Vec<f64> {
        match self {
            Device::Two(d) => d.site_densities(s, f, op),
            Device::Multi(d) => d.site_densities(s, f, op),
        }
    }
}

fn local(input: &Input, r: &LocalRun, base: &BTreeMap<String, f64>) -> Result<Table, Error> {
    let points = build::scan_points(&input.parameters, base, &r.scan)?;
    let configs = r.configurations.unwrap_or(1);
    let ops: Vec<(String, CMat)> = r
        .operators
        .iter()
        .map(|(k, v)| Ok((k.clone(), points[0].1.matrix(v)?)))
        .collect::<Result<_, Error>>()?;
    let mut header = format!("{}config,x,y,z,jx,jy,jz,density,ldos", scan_header(&r.scan));
    for (name, _) in &ops {
        write!(header, ",{name},j{name}_x,j{name}_y,j{name}_z").unwrap();
    }
    header.push('\n');
    let rows = over_points(&points, configs, |(values, p), c| {
        let (clean, region, device) = system(input, p, c)?;
        let lattice = region.lattice().clone();
        let (dev, leads) = match build::leads(&input.system, &region)? {
            Leads::Axis(axis) => {
                (Device::Two(NestedTwoTerminal::from_model(&device, &clean, &region, axis)), 2)
            }
            Leads::Faces(f) => {
                let n = f.len();
                (Device::Multi(NestedMultiTerminal::from_model(&device, &clean, &region, &f)), n)
            }
        };
        let f: Vec<f64> = match &r.occupations {
            Some(o) if o.len() == leads => o.clone(),
            Some(o) => return Err(format!("{} occupations for {leads} leads", o.len()).into()),
            None => (0..leads).map(|k| (k == 0) as u8 as f64).collect(),
        };
        let ones = vec![1.0; leads];
        let energies = p.values(&r.energies)?;
        let n = region.len();
        let width = 5 + 4 * ops.len();
        // Per site: current (3), density, ldos, then per operator density and current (3).
        let mut acc = vec![vec![0.0; width]; n];
        let index: std::collections::HashMap<Site, usize> =
            region.sites().iter().enumerate().map(|(i, s)| (*s, i)).collect();
        let add = |acc: &mut Vec<Vec<f64>>, bonds: Vec<BondCurrent>, slot: usize| {
            for b in bonds {
                let (ra, rb) = (lattice.position(b.from), lattice.position(b.to));
                let d = [rb[0] - ra[0], rb[1] - ra[1], rb[2] - ra[2]];
                let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                for s in [b.from, b.to] {
                    if let Some(&i) = index.get(&s) {
                        for k in 0..3 {
                            acc[i][slot + k] += 0.5 * b.current * d[k] / norm;
                        }
                    }
                }
            }
        };
        for &e in &energies {
            let states = dev.states(e)?;
            add(&mut acc, dev.bonds(&states, &f, None), 0);
            for (i, (d, l)) in
                dev.densities(&states, &f, None).iter().zip(dev.densities(&states, &ones, None)).enumerate()
            {
                acc[i][3] += d;
                acc[i][4] += l;
            }
            for (k, (_, op)) in ops.iter().enumerate() {
                let slot = 5 + 4 * k;
                for (i, d) in dev.densities(&states, &f, Some(op)).iter().enumerate() {
                    acc[i][slot] += d;
                }
                add(&mut acc, dev.bonds(&states, &f, Some(op)), slot + 1);
            }
        }
        let m = energies.len() as f64;
        let mut out = String::new();
        for (site, a) in region.sites().iter().zip(&acc) {
            let r = lattice.position(*site);
            let v: Vec<String> = a.iter().map(|x| fmt(x / m)).collect();
            writeln!(out, "{}{c},{},{},{},{}", scan_prefix(values), r[0], r[1], r[2], v.join(",")).unwrap();
        }
        Ok(out)
    })?;
    let summary = format!("local: {} scan points × {configs} configurations", points.len());
    Ok(Table { csv: header + &rows, summary })
}

fn bands(input: &Input, r: &BandsRun, base: &BTreeMap<String, f64>) -> Result<Table, Error> {
    if !input.disorder.is_empty() {
        eprintln!("bands: the disorder is ignored (bands of the clean model)");
    }
    let points = build::scan_points(&input.parameters, base, &r.scan)?;
    let eig = |h: CMat| -> Vec<f64> { h.self_adjoint_eigenvalues(faer::Side::Lower).expect("eigenvalues") };
    let (header, rows) = if let Some(path) = &r.path {
        let header = format!("{}s,band,energy\n", scan_header(&r.scan));
        let rows = over_points(&points, 1, |(values, p), _| {
            let model = build::model(&input.model, p)?;
            let dim = model.lattice().dim();
            let corners: Vec<Vec<f64>> = path
                .iter()
                .map(|c| {
                    if c.len() != dim {
                        return Err(Error::from(format!("path corners need {dim} Bloch phases")));
                    }
                    c.iter().map(|x| p.num(x)).collect()
                })
                .collect::<Result<_, Error>>()?;
            let cell = Region::cuboid(model.lattice(), [1, 1, 1]);
            let periods: Vec<[i64; 3]> = (0..dim).map(|a| std::array::from_fn(|c| (c == a) as i64)).collect();
            let per = r.points.unwrap_or(50);
            let mut out = String::new();
            let mut s = 0.0;
            for w in corners.windows(2) {
                let length = w[0].iter().zip(&w[1]).map(|(a, b)| (b - a).powi(2)).sum::<f64>().sqrt();
                for k in 0..per {
                    let f = k as f64 / per as f64;
                    let phase: Vec<f64> = w[0].iter().zip(&w[1]).map(|(a, b)| a + f * (b - a)).collect();
                    for (b, e) in eig(model.bloch_hamiltonian(&cell, &periods, &phase)).iter().enumerate() {
                        writeln!(out, "{}{},{b},{e}", scan_prefix(values), s + f * length).unwrap();
                    }
                }
                s += length;
            }
            let last = corners.last().expect("corner");
            for (b, e) in eig(model.bloch_hamiltonian(&cell, &periods, last)).iter().enumerate() {
                writeln!(out, "{}{s},{b},{e}", scan_prefix(values)).unwrap();
            }
            Ok(out)
        })?;
        (header, rows)
    } else {
        let header = format!("{}k,band,energy\n", scan_header(&r.scan));
        let rows = over_points(&points, 1, |(values, p), _| {
            let model = build::model(&input.model, p)?;
            let region = build::region(&model, &input.system)?;
            let axis = match build::leads(&input.system, &region)? {
                Leads::Axis(a) => a,
                Leads::Faces(f) => (0..3).find(|&c| f[0].outward[c] != 0).expect("axis"),
            };
            let slice = &region.slices(axis)[0];
            let period: [i64; 3] = std::array::from_fn(|c| (c == axis) as i64);
            let ks = match &r.k {
                Some(v) => p.values(v)?,
                None => (0..201)
                    .map(|k| -std::f64::consts::PI + 2.0 * std::f64::consts::PI * k as f64 / 200.0)
                    .collect(),
            };
            let mut out = String::new();
            for k in ks {
                for (b, e) in eig(model.bloch_hamiltonian(slice, &[period], &[k])).iter().enumerate() {
                    writeln!(out, "{}{k},{b},{e}", scan_prefix(values)).unwrap();
                }
            }
            Ok(out)
        })?;
        (header, rows)
    };
    Ok(Table { csv: header + &rows, summary: format!("bands: {} scan points", points.len()) })
}

/// Largest dense problem of `spectrum`.
const DENSE_LIMIT: usize = 8000;

fn spectrum(input: &Input, r: &SpectrumRun, base: &BTreeMap<String, f64>) -> Result<Table, Error> {
    let points = build::scan_points(&input.parameters, base, &r.scan)?;
    let configs = r.configurations.unwrap_or(1);
    let rows = over_points(&points, configs, |(values, p), c| {
        let (_, region, device) = system(input, p, c)?;
        if region.orbitals() > DENSE_LIMIT {
            return Err(
                format!("{} orbitals: too many for a dense spectrum (use `dos`)", region.orbitals()).into()
            );
        }
        let h = if input.system.periodic {
            let periods = build::periods(&input.system);
            device.bloch_hamiltonian(&region, &periods, &vec![0.0; periods.len()])
        } else {
            device.hamiltonian(&region)
        };
        let e = h.self_adjoint_eigenvalues(faer::Side::Lower).map_err(|e| format!("{e:?}"))?;
        Ok(e.iter().enumerate().map(|(k, x)| format!("{}{c},{k},{x}\n", scan_prefix(values))).collect())
    })?;
    let header = format!("{}config,index,energy\n", scan_header(&r.scan));
    Ok(Table {
        csv: header + &rows,
        summary: format!("spectrum: {} scan points × {configs} configurations", points.len()),
    })
}

/// The KPM spectrum of a configuration.
fn kpm<'a>(
    device: &'a Model,
    region: &'a Region,
    periods: &[[i64; 3]],
    moments: Option<usize>,
    vectors: Option<usize>,
    seed: Option<u64>,
    bounds: Option<[f64; 2]>,
) -> Spectrum<'a> {
    let mut s = Spectrum::periodic(device, region, periods)
        .moments(moments.unwrap_or(512))
        .vectors(vectors.unwrap_or(10));
    if let Some(seed) = seed {
        s = s.seed(seed);
    }
    if let Some([lo, hi]) = bounds {
        s = s.bounds(lo, hi);
    }
    s
}

fn dos(input: &Input, r: &DosRun, base: &BTreeMap<String, f64>) -> Result<Table, Error> {
    let points = build::scan_points(&input.parameters, base, &r.scan)?;
    let configs = r.configurations.unwrap_or(1);
    let periods = if input.system.periodic { build::periods(&input.system) } else { Vec::new() };
    let mut header = format!("{}config,energy,dos", scan_header(&r.scan));
    r.projections.keys().for_each(|k| write!(header, ",{k}").unwrap());
    header.push('\n');
    let rows = over_points(&points, configs, |(values, p), c| {
        let (_, region, device) = system(input, p, c)?;
        let s = kpm(&device, &region, &periods, r.moments, r.vectors, r.seed.map(|x| x + c as u64), r.bounds);
        let energies = p.values(&r.energies)?;
        let mut columns = vec![s.dos(&energies)];
        for e in r.projections.values() {
            columns.push(s.projected_dos(&p.matrix(e)?, &energies));
        }
        let mut out = String::new();
        for (k, e) in energies.iter().enumerate() {
            let v: Vec<String> = columns.iter().map(|col| fmt(col[k])).collect();
            writeln!(out, "{}{c},{e},{}", scan_prefix(values), v.join(",")).unwrap();
        }
        Ok(out)
    })?;
    Ok(Table {
        csv: header + &rows,
        summary: format!("dos: {} scan points × {configs} configurations", points.len()),
    })
}

fn direction(axis: char) -> Result<[f64; 3], Error> {
    Ok(match axis {
        'x' => [1.0, 0.0, 0.0],
        'y' => [0.0, 1.0, 0.0],
        'z' => [0.0, 0.0, 1.0],
        other => return Err(format!("component axis `{other}`").into()),
    })
}

fn conductivity(input: &Input, r: &ConductivityRun, base: &BTreeMap<String, f64>) -> Result<Table, Error> {
    if !input.system.periodic {
        return Err("the conductivity needs a periodic sample ([system] periodic = true)".into());
    }
    let points = build::scan_points(&input.parameters, base, &r.scan)?;
    let configs = r.configurations.unwrap_or(1);
    let components = r.components.clone().unwrap_or_else(|| vec!["xx".into(), "xy".into()]);
    let pairs: Vec<([f64; 3], [f64; 3])> = components
        .iter()
        .map(|c| {
            let ch: Vec<char> = c.chars().collect();
            if ch.len() != 2 {
                return Err(Error::from(format!("component `{c}`: two axes such as xy")));
            }
            Ok((direction(ch[0])?, direction(ch[1])?))
        })
        .collect::<Result<_, Error>>()?;
    let periods = build::periods(&input.system);
    let mut header = format!("{}config,energy", scan_header(&r.scan));
    components.iter().for_each(|k| write!(header, ",sigma_{k}").unwrap());
    header.push('\n');
    let rows = over_points(&points, configs, |(values, p), c| {
        let (_, region, device) = system(input, p, c)?;
        let s = kpm(
            &device,
            &region,
            &periods,
            r.moments.or(Some(256)),
            r.vectors,
            r.seed.map(|x| x + c as u64),
            r.bounds,
        );
        let energies = p.values(&r.energies)?;
        let temperature = p.opt(&r.temperature, 0.0)?;
        let columns: Vec<Vec<f64>> =
            pairs.iter().map(|(a, b)| s.conductivity(*a, *b, &energies, temperature)).collect();
        let mut out = String::new();
        for (k, e) in energies.iter().enumerate() {
            let v: Vec<String> = columns.iter().map(|col| fmt(col[k])).collect();
            writeln!(out, "{}{c},{e},{}", scan_prefix(values), v.join(",")).unwrap();
        }
        Ok(out)
    })?;
    Ok(Table {
        csv: header + &rows,
        summary: format!("conductivity (e²/h): {} scan points × {configs} configurations", points.len()),
    })
}

/// The disorder of the input as SCBA vertices: independent on every site, so one set of vertices
/// per sublattice of the cell. The `cells` windows do not apply (bulk).
fn born_disorder(input: &Input, p: &Params, model: &Model) -> Result<Disorder, Error> {
    let cell = Region::cuboid(model.lattice(), [1, 1, 1]);
    let n = cell.orbitals();
    let embed = |m: &CMat, o: usize| {
        let mut out = CMat::zeros(n, n);
        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                out[(o + i, o + j)] = m[(i, j)];
            }
        }
        out
    };
    let mut total = Disorder { mean_field: CMat::zeros(n, n), vertices: Vec::new() };
    for d in &input.disorder {
        for (i, site) in cell.sites().iter().enumerate() {
            if d.sublattices.as_ref().is_some_and(|l| !l.contains(&site.sub)) {
                continue;
            }
            let (o, orb) = (cell.offset(i), cell.site_orbitals(i));
            let part = match d.kind.as_str() {
                "anderson" => {
                    let w = p.num(d.strength.as_ref().ok_or("anderson disorder needs `strength`")?)?;
                    let id =
                        CMat::from_fn(
                            orb,
                            orb,
                            |a, b| if a == b { c64::new(1.0, 0.0) } else { c64::new(0.0, 0.0) },
                        );
                    Disorder { mean_field: CMat::zeros(n, n), vertices: vec![(w * w / 12.0, embed(&id, o))] }
                }
                "magnetic" => {
                    let conc =
                        p.num(d.concentration.as_ref().ok_or("magnetic disorder needs `concentration`")?)?;
                    let j = p.num(d.exchange.as_ref().ok_or("magnetic disorder needs `exchange`")?)?;
                    let s = build::spin_matrices(d, p, orb)?;
                    Disorder::magnetic(
                        conc,
                        j,
                        build::moments(d, p)?,
                        &[embed(&s[0], o), embed(&s[1], o), embed(&s[2], o)],
                    )
                }
                other => return Err(format!("unknown disorder `{other}`").into()),
            };
            total = total.and(part);
        }
    }
    Ok(total)
}

fn scba(input: &Input, r: &ScbaRun, base: &BTreeMap<String, f64>) -> Result<Table, Error> {
    let points = build::scan_points(&input.parameters, base, &r.scan)?;
    let model0 = build::model(&input.model, &points[0].1)?;
    let n = Region::cuboid(model0.lattice(), [1, 1, 1]).orbitals();
    let dim = model0.lattice().dim();
    drop(model0);
    let grid = r.k_grid.clone().unwrap_or_else(|| vec![64; dim]);
    if grid.len() != dim {
        return Err(format!("`k_grid` needs {dim} sizes").into());
    }
    let mut header = format!("{}energy,dos", scan_header(&r.scan));
    (0..n).for_each(|k| write!(header, ",sigma_re_{k},sigma_im_{k}").unwrap());
    header += ",residual\n";
    let rows = over_points(&points, 1, |(values, p), _| {
        let model = build::model(&input.model, p)?;
        let scba = Scba::from_model(&model, &grid, born_disorder(input, p, &model)?);
        let energies = p.values(&r.energies)?;
        let eta = p.opt(&r.eta, 1e-3)?;
        let mut out = String::new();
        for (e, sol) in energies.iter().zip(scba.sweep(&energies, eta)) {
            let total = scba.total_self_energy(&sol.sigma);
            let sigma: Vec<String> =
                (0..n).map(|k| format!("{},{}", fmt(total[(k, k)].re), fmt(total[(k, k)].im))).collect();
            let dos = scba.dos(*e, eta, &sol.sigma);
            writeln!(
                out,
                "{}{e},{},{},{}",
                scan_prefix(values),
                fmt(dos),
                sigma.join(","),
                fmt(sol.residual)
            )
            .unwrap();
        }
        Ok(out)
    })?;
    Ok(Table { csv: header + &rows, summary: format!("scba: {} scan points, k grid {grid:?}", points.len()) })
}

fn evolution(input: &Input, r: &EvolutionRun, base: &BTreeMap<String, f64>) -> Result<Table, Error> {
    let points = build::scan_points(&input.parameters, base, &r.scan)?;
    let configs = r.configurations.unwrap_or(1);
    let size = input.system.size.clone().unwrap_or_default();
    if size.len() != 2 || size[0] != size[1] {
        return Err("the evolution runs on an n × n torus: [system] size = [n, n]".into());
    }
    let n = size[0];
    let rows = over_points(&points, configs, |(values, p), c| {
        let (_, _, device) = system(input, p, c)?;
        let force = match &r.force {
            Some(f) => {
                let v: Vec<f64> = f.iter().map(|x| p.num(x)).collect::<Result<_, Error>>()?;
                std::array::from_fn(|i| v.get(i).copied().unwrap_or(0.0))
            }
            None => [0.0; 3],
        };
        if !device.has_uniform_hoppings() {
            return Err("the evolution needs hoppings without field or site-dependent terms".into());
        }
        let mut torus = Torus::new(&device, n).with_force(force);
        if let InitialInput::Band { orbitals: Some(o), .. } = &r.initial {
            let comps = torus.components();
            if o.iter().any(|&a| a >= comps) {
                return Err(format!("`orbitals` of the band: a cell has {comps} orbitals").into());
            }
            if let Some(g) = torus
                .groups()
                .iter()
                .find(|g| g.iter().any(|a| o.contains(a)) && !g.iter().all(|a| o.contains(a)))
            {
                return Err(format!(
                    "`orbitals` of the band must include all of {g:?}, coupled by the hoppings"
                )
                .into());
            }
            torus = torus.with_sector(o);
        }
        let starts = match &r.initial {
            InitialInput::FermiSea { fermi, vectors, seed, moments } => {
                let (mu, count, seed) =
                    (p.opt(fermi, 0.0)?, vectors.unwrap_or(1), seed.unwrap_or(1) + c as u64);
                if input.disorder.is_empty() {
                    torus.filled(mu, count, seed)
                } else {
                    torus.fermi_sea(mu, count, seed, moments.unwrap_or(1024))
                }
            }
            InitialInput::Band { fermi, vectors, seed, .. } => {
                torus.filled(p.opt(fermi, 0.0)?, vectors.unwrap_or(1), seed.unwrap_or(1) + c as u64)
            }
            InitialInput::Localized { cell, orbital, window } => {
                if cell.len() != 2 || cell[0] >= n || cell[1] >= n {
                    return Err("`cell` of the localized state: two indices inside the torus".into());
                }
                let o = orbital.unwrap_or(0);
                if o >= torus.components() {
                    return Err(format!("orbital {o} of a cell with {} orbitals", torus.components()).into());
                }
                vec![torus.localized([cell[0], cell[1]], o, window.map(|[a, b]| (a, b)))]
            }
        };
        let ev = Evolution {
            dt: p.num(&r.dt)?,
            steps: r.steps,
            every: r.every.unwrap_or(1),
            fermi: p.opt(&r.fermi, 0.0)?,
            densities: false,
        };
        let rec = torus.evolve(&starts, &ev);
        let mut out = String::new();
        for k in 0..rec.times.len() {
            writeln!(
                out,
                "{}{c},{},{},{},{},{},{}",
                scan_prefix(values),
                rec.times[k],
                fmt(rec.norm[k]),
                fmt(rec.current[k][0]),
                fmt(rec.current[k][1]),
                fmt(rec.excited[k]),
                fmt(rec.width[k])
            )
            .unwrap();
        }
        Ok(out)
    })?;
    let header = format!("{}config,t,norm,jx,jy,excited,width\n", scan_header(&r.scan));
    Ok(Table {
        csv: header + &rows,
        summary: format!("evolution: {} scan points × {configs} configurations", points.len()),
    })
}
