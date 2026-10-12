//! Gauge changes `ψ → exp(iχ(r)) ψ` for leads in a uniform magnetic field.
//!
//! A model built with [`Model::with_field`] uses the Landau gauge invariant along x. A lead along
//! another direction `d` needs the Landau gauge invariant along `d`, which differs by a gradient
//! `∇χ` ([`caroli_model::gauge::landau_x_to_along`]). The transport constructors express each
//! lead in its own gauge, and the device-lead couplings in the mixed gauge: the device keeps its
//! gauge, so its Green's function and the transmissions do not change.

use caroli_lattice::{Cell, Region};
use caroli_model::{CMat, Model, SparseMatrix, c64};

/// The gauge invariant along a lattice direction, for a model with a uniform field (the
/// identity otherwise).
pub(crate) struct Gauge<'a> {
    model: &'a Model,
    chi: Option<Box<dyn Fn([f64; 3]) -> f64 + 'a>>,
}

impl<'a> Gauge<'a> {
    /// The gauge invariant along `step` (cells) for `model`.
    pub(crate) fn along(model: &'a Model, step: Cell) -> Self {
        let chi = model.field().map(|b| {
            let v = model.lattice().vectors();
            let d = (0..3).fold([0.0; 3], |mut d, k| {
                if let Some(vk) = v.get(k) {
                    for (dc, vc) in d.iter_mut().zip(vk) {
                        *dc += step[k] as f64 * vc;
                    }
                }
                d
            });
            Box::new(caroli_model::gauge::landau_x_to_along(b, d)) as Box<dyn Fn([f64; 3]) -> f64>
        });
        Self { model, chi }
    }

    /// `exp(iχ(r))` for every orbital of `region`, or `None` for the identity.
    pub(crate) fn factors(&self, region: &Region) -> Option<Vec<c64>> {
        let chi = self.chi.as_ref()?;
        let lattice = self.model.lattice();
        let mut u = Vec::with_capacity(region.orbitals());
        for (i, site) in region.sites().iter().enumerate() {
            let f = c64::from_polar(1.0, chi(lattice.position(*site)));
            u.extend(std::iter::repeat_n(f, region.site_orbitals(i)));
        }
        Some(u)
    }

    /// `diag(u_a) m diag(u_b)†` for a matrix with rows on `a` and columns on `b`.
    pub(crate) fn transform(&self, m: CMat, a: &Region, b: &Region) -> CMat {
        match (self.factors(a), self.factors(b)) {
            (Some(ua), Some(ub)) => {
                CMat::from_fn(m.nrows(), m.ncols(), |i, j| ua[i] * m[(i, j)] * ub[j].conj())
            }
            _ => m,
        }
    }

    /// Columns only: `m diag(u_b)†`, for couplings from a region kept in the model's gauge.
    pub(crate) fn transform_columns(&self, m: CMat, b: &Region) -> CMat {
        match self.factors(b) {
            Some(ub) => CMat::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)] * ub[j].conj()),
            None => m,
        }
    }

    /// The sparse Hamiltonian of `region` in this gauge, in place.
    pub(crate) fn transform_sparse(&self, h: &mut SparseMatrix, region: &Region) {
        let Some(u) = self.factors(region) else { return };
        for r in 0..h.n {
            for k in h.row_ptr[r]..h.row_ptr[r + 1] {
                h.vals[k] *= u[r] * u[h.cols[k] as usize].conj();
            }
        }
    }

    /// `⟨cell| H |cell⟩` and `⟨cell| H |cell + step⟩` of a lead of `model` in this gauge.
    /// Panics unless they are the same from cell to cell along `along` (the direction of the
    /// lead cells from `cell`): a vector potential or a potential that changes along the lead
    /// would make its modes wrong.
    pub(crate) fn lead(&self, cell: &Region, step: Cell, along: Cell) -> (CMat, CMat) {
        let pair = |c: &Region| {
            let next = c.translated(step);
            (
                self.transform(self.model.hamiltonian(c), c, c),
                self.transform(self.model.coupling(c, &next), c, &next),
            )
        };
        let (h00, h01) = pair(cell);
        let (h00_next, h01_next) = pair(&cell.translated(along));
        let tol = 1e-12 * h00.norm_max().max(h01.norm_max());
        assert!(
            (&h00 - &h00_next).norm_max() <= tol && (&h01 - &h01_next).norm_max() <= tol,
            "the lead model changes along the lead: use Model::with_field for a uniform field, or a \
             gauge invariant along the lead, and no potential that varies along it"
        );
        (h00, h01)
    }
}

/// Panics unless the device and lead models carry the same uniform field (or none).
pub(crate) fn assert_same_field(device: &Model, lead: &Model) {
    assert_eq!(device.field(), lead.field(), "the device and the leads must carry the same field");
}
