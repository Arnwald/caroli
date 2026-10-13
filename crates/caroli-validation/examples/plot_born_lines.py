"""Born phase diagrams of the g2g1 paper (Fig. 4): published lines (thesis code `self_born`)
against the corrected SCBA, drawn as in the paper, in the colours of the thesis (`born_style`).

Reads the CSV files written by
    cargo run --release -p caroli-validation --example born_lines
and writes into the same directory:
    born_fig4.png           the paper figure, published (left) and corrected (right);
    born_fig4_numerics.png  the same lines over the exact numerical conductance maps;
    born_all_panels.png     the eight runs 00-07, published and corrected.

    python3 crates/caroli-validation/examples/plot_born_lines.py [csv directory]
"""

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from born_style import LINES, MU_LINES, PLATEAU_1, born_fields, draw_lines, g_colorbar, g_map, line_handles, save

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "..", "target", "born_lines")

# The lines of the paper's figure: band edges, inversions and renormalised chemical potentials.
KEYS = tuple(LINES) + tuple(MU_LINES)

# Paper Fig. 4: run, title, schematic G = 1 region (quadrilateral of the paper), numerical map.
FIG4 = [
    (
        2,
        r"'AF', $n_I=0.4$, $M_z=1$",
        [(0.32, 0.01), (0.60, 0.02), (0.60, 0.10), (0.355, 0.052)],
        "numerical_Antiferro_anti_400_256_x64_0.4_1.0_32Jx64e.csv",
    ),
    (
        1,
        r"'F', $n_I=0.2$, $M_z=1$",
        [(0.73, -0.041), (1.00, -0.050), (1.00, -0.140), (0.75, -0.150)],
        "numerical_Ferro_ferro_400_256_x64_0.2_1.0_64Jx64e.csv",
    ),
]

TITLES = [
    r"'AF', $n_I=0.2$, $M_z=1$",
    r"'F', $n_I=0.2$, $M_z=1$",
    r"'AF', $n_I=0.4$, $M_z=1$",
    r"'F', $n_I=0.4$, $M_z=1$",
    r"'AF', $n_I=0.2$, $M_z=0.8$",
    r"'F', $n_I=0.2$, $M_z=0.8$",
    r"'AF', $n_I=0.4$, $M_z=0.8$",
    r"'F', $n_I=0.4$, $M_z=0.8$",
]


def read_csv(name):
    with open(os.path.join(DIR, name)) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if not k.startswith("G_born")}


def panel(run):
    """Grids (E index, J index) of the published and corrected Born fields."""
    d = read_csv(f"panel_{run:02d}.csv")
    n = int(round(np.sqrt(len(d["J_I"]))))
    grid = lambda k: d[k].reshape(n, n)
    j, e = grid("J_I"), grid("E_F")
    out = {
        version: born_fields(grid("M_up" + s), grid("mu_up" + s), grid("M_dn" + s), grid("mu_dn" + s), e)
        for version, s in (("published", "_thesis"), ("corrected", ""))
    }
    return j, e, out


def numerical(name):
    """The published conductance map on its grid (E index, J index)."""
    d = read_csv(name)
    js, es = np.unique(d["J_I"]), np.unique(d["E_F"])
    g = d["G_numerical"].reshape(len(es), len(js))
    jj, ee = np.meshgrid(js, es)
    return jj, ee, g


def draw(ax, j, e, fields, title, polygon=None, background=None):
    mesh = g_map(ax, *background) if background is not None else None
    if polygon is not None:
        ax.add_patch(Polygon(polygon, closed=True, color=PLATEAU_1, lw=0))
    draw_lines(ax, j, e, fields, KEYS)
    ax.set_xlim(0.0, 1.5)
    ax.set_ylim(-0.16, 0.16)
    ax.set_title(title, fontsize=11)
    return mesh


def legend(fig, extra=()):
    fig.legend(handles=line_handles(KEYS) + list(extra), loc="lower center", ncol=5, fontsize=9, frameon=False)


def fig4(background):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), sharex=True, sharey=True)
    mesh = None
    for row, (run, title, polygon, num) in enumerate(FIG4):
        j, e, versions = panel(run)
        bg = numerical(num) if background else None
        for col, version in enumerate(("published", "corrected")):
            head = "published (thesis code)" if version == "published" else "corrected SCBA"
            m = draw(
                axes[row, col],
                j,
                e,
                versions[version],
                f"{title} — {head}",
                polygon=None if background else polygon,
                background=bg,
            )
            mesh = m or mesh
    for ax in axes[1]:
        ax.set_xlabel(r"disorder $J_I$ $(\varepsilon_0)$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"energy $E_F$ $(\varepsilon_0)$")
    extra = []
    if not background:
        extra.append(Polygon([(0, 0)], color=PLATEAU_1, label="G = 1 (schematic, paper)"))
    legend(fig, extra)
    fig.subplots_adjust(bottom=0.14, top=0.95, hspace=0.18, wspace=0.08)
    if mesh is not None:
        g_colorbar(fig, mesh, r"exact $\langle G \rangle$ ($e^2/h$), thesis conductance maps", ax=axes, shrink=0.6, pad=0.02)
    save(fig, DIR, "born_fig4_numerics" if background else "born_fig4")


def all_panels():
    fig, axes = plt.subplots(8, 2, figsize=(10, 34), sharex=True, sharey=True)
    for run in range(8):
        j, e, versions = panel(run)
        for col, version in enumerate(("published", "corrected")):
            head = "published" if version == "published" else "corrected"
            draw(axes[run, col], j, e, versions[version], f"run {run:02d}: {TITLES[run]} — {head}")
    for ax in axes[-1]:
        ax.set_xlabel(r"$J_I$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$E_F$")
    legend(fig)
    fig.subplots_adjust(bottom=0.04, top=0.99, hspace=0.25, wspace=0.08)
    save(fig, DIR, "born_all_panels", dpi=110)


if __name__ == "__main__":
    fig4(background=False)
    fig4(background=True)
    all_panels()
