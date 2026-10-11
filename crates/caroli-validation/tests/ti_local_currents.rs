//! Keldysh local currents and occupied densities against `local-currents-2t`
//! (`ti-lcur-same-g`, `-opposite-g`, `-anderson`), with the thesis' random configurations.

use caroli_green::SelfEnergyMethod;
use caroli_keldysh::{Lesser, fermi};
use caroli_validation::thesis::{LocalCurrents, thesis_local_currents};
use caroli_validation::{golden_dir, golden_f64, read_records};

fn params(couplage: i32) -> LocalCurrents {
    let (conc, mz, nx_imp) = if couplage == 0 { (1.0, 0.0, 16) } else { (0.4, 1.0, 8) };
    LocalCurrents {
        nx: 20,
        ny: 8,
        nx_imp,
        ne: 3,
        conc,
        mz,
        couplage,
        ev_l: 0.001,
        ev_r: -0.001,
        e_f: 0.035,
        emin: -0.0009,
        emax: 0.0009,
        u0: 0.33,
        eta: 1e-5,
        seed: 12300,
    }
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn check(case: &str, couplage: i32) {
    let p = params(couplage);
    let imp = p.impurities();

    // Impurity positions written by the thesis program: x_rand(N_imp), y_rand(N_imp) (int32).
    let rec = &read_records(golden_dir(case).join("imp_00.dat")).unwrap()[0];
    let ints: Vec<i32> = rec.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    let n = imp.sites.len();
    assert_eq!(ints.len(), 2 * n, "{case}: number of impurities");
    for (k, s) in imp.sites.iter().enumerate() {
        assert_eq!((s.cell[0], s.cell[1] + 1), (ints[k] as i64, ints[n + k] as i64), "{case}: impurity {k}");
    }

    let sys = p.device(&imp);
    let energies = p.energies();
    assert!(max_diff(&energies, &golden_f64(case, "axes_00.dat")[0]) < 1e-15);
    let lessers: Vec<Lesser> = energies
        .iter()
        .map(|&e| {
            let (fl, fr) = (fermi(e, p.e_f + p.ev_l, 0.0), fermi(e, p.e_f + p.ev_r, 0.0));
            Lesser::compute(&sys, e, SelfEnergyMethod::ModesBroadened { eta: p.eta }, fl, fr).unwrap()
        })
        .collect();
    let deltae = (p.emax - p.emin) / (p.ne - 1) as f64;
    let (lcur, ldos) = thesis_local_currents(&sys, &lessers, p.nx, p.ny, deltae);

    let cur_ref = &golden_f64(case, "cur_00.dat")[0];
    let dos_ref = &golden_f64(case, "dos_00.dat")[0];
    let scale = cur_ref.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let dc = max_diff(&lcur, cur_ref);
    let dd = max_diff(&ldos.concat(), dos_ref);
    println!(
        "{case}: {n} impurities, max |Lcur − golden| = {dc:.1e} (max |Lcur| = {scale:.1e}), max |Ldos − golden| = {dd:.1e}"
    );
    assert!(dc < 1e-9 * scale.max(1.0), "{case}: currents off by {dc:e}");
    assert!(dd < 1e-9, "{case}: densities off by {dd:e}");

    // Exact physics on the same device: every interface carries T (f_L − f_R).
    for l in &lessers {
        for n in 0..sys.couplings.len() {
            let j = l.interface_current(&sys, n);
            let expected = l.transmission * (l.occupations.0 - l.occupations.1);
            assert!((j - expected).abs() < 1e-8, "{case}: interface {n}: {j} vs {expected}");
        }
    }
}

#[test]
fn same_g() {
    check("ti-lcur-same-g", 1);
}

#[test]
fn opposite_g() {
    check("ti-lcur-opposite-g", -1);
}

#[test]
fn anderson() {
    check("ti-lcur-anderson", 0);
}

/// `ti-cond-ldos-current`: the 2014 manuscript code (Anderson-disordered 12×8 ribbon, small
/// bias window): conductance, LDOS and orbital-resolved bond currents.
#[test]
fn conductance_ldos_current_2014() {
    use caroli_validation::thesis::CondLdosCurrent2014;
    let p = CondLdosCurrent2014 {
        nx: 12,
        ny: 8,
        ne: 3,
        ev_l: 0.001,
        ev_r: -0.001,
        mass: -0.13,
        emin: -0.05,
        emax: 0.05,
        u0: 1.0,
        eta: 1e-3,
        seed: 12347,
    };
    let out = p.run();
    let g = |f: &str| golden_f64("ti-cond-ldos-current", f).remove(0);
    let maxd = |a: &[f64], b: &[f64]| {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    };
    let scale = |v: &[f64]| v.iter().fold(0f64, |m, x| m.max(x.abs()));
    let (ge, gc, gd, gl) = (g("axes_00.dat"), g("cond_00.dat"), g("dos_00.dat"), g("cur_00.dat"));
    let de = maxd(&out.energies, &ge);
    let dc = maxd(&out.conductance, &gc) / scale(&gc);
    let dd = maxd(&out.ldos, &gd) / scale(&gd);
    let dl = maxd(&out.lcur, &gl) / scale(&gl);
    println!(
        "cond-ldos-current: |ΔE| ≤ {de:.1e}, relative |Δ G| ≤ {dc:.1e}, |Δ LDOS| ≤ {dd:.1e}, |Δ Lcur| ≤ {dl:.1e}"
    );
    assert!(de < 1e-15 && dc < 1e-10 && dd < 1e-10 && dl < 1e-10);
}
