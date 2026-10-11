"""Born phase diagrams of the g2g1 paper (Fig. 4): published lines (thesis code `self_born`)
against the corrected SCBA, drawn as in the paper.

Reads the CSV files written by
    cargo run --release -p caroli-validation --example born_lines
and writes into the same directory:
    born_fig4.png/pdf           the paper figure, published (left) and corrected (right);
    born_fig4_numerics.png/pdf  the same lines over the exact numerical conductance maps;
    born_all_panels.png         the eight runs 00-07, published and corrected.

    python3 crates/caroli-validation/examples/plot_born_lines.py [csv directory]
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "..", "target", "born_lines")

# (contour field, colour, line style, width, label) as in the paper.
LINES = [
    ("ep_up", "blue", "-", 1.2, r"E$\uparrow$ band edge"),
    ("hp_up", "red", "-", 1.2, r"H$\uparrow$ band edge"),
    ("ep_dn", "green", "--", 2.0, r"E$\downarrow$ band edge"),
    ("hp_dn", "cyan", "--", 2.0, r"H$\downarrow$ band edge"),
    ("m_up", "black", ":", 2.0, r"$m_\uparrow = 0$"),
    ("m_dn", "gold", ":", 2.0, r"$m_\downarrow = 0$"),
    ("mu_up", "black", "-.", 2.0, r"$\mu_\uparrow$"),
    ("mu_dn", "gold", "-.", 2.0, r"$\mu_\downarrow$"),
]

# Paper Fig. 4: run, title, schematic G = 1 region (blue quadrilateral of the paper), numerical map.
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
    """Grids (E index, J index) of the published and corrected renormalised parameters."""
    d = read_csv(f"panel_{run:02d}.csv")
    n = int(round(np.sqrt(len(d["J_I"]))))
    grid = lambda k: d[k].reshape(n, n)
    j, e = grid("J_I"), grid("E_F")
    out = {}
    for version, suffix in (("published", "_thesis"), ("corrected", "")):
        mu, muu = grid("M_up" + suffix), grid("mu_up" + suffix)
        md, mud = grid("M_dn" + suffix), grid("mu_dn" + suffix)
        out[version] = {
            "ep_up": mu - muu - e,
            "hp_up": mu + muu + e,
            "ep_dn": md - mud - e,
            "hp_dn": md + mud + e,
            "m_up": mu,
            "m_dn": md,
            "mu_up": muu + e,
            "mu_dn": mud + e,
        }
    return j, e, out


def numerical(name):
    d = read_csv(name)
    js, es = np.unique(d["J_I"]), np.unique(d["E_F"])
    g = d["G_numerical"].reshape(len(es), len(js))
    return js, es, g


def draw(ax, j, e, fields, title, polygon=None, background=None):
    if background is not None:
        js, es, g = background
        mesh = ax.pcolormesh(js, es, g, cmap="Blues", vmin=0.0, vmax=2.6, shading="nearest", alpha=0.75)
    else:
        mesh = None
    if polygon is not None:
        ax.add_patch(Polygon(polygon, closed=True, color="#b9bdf2", alpha=0.85, lw=0))
    for key, colour, style, width, _ in LINES:
        ax.contour(j, e, fields[key], [0.0], colors=colour, linestyles=style, linewidths=width)
    ax.set_xlim(0.0, 1.5)
    ax.set_ylim(-0.16, 0.16)
    ax.set_title(title, fontsize=11)
    return mesh


def legend(fig, extra=()):
    handles = [Line2D([], [], color=c, ls=s, lw=w, label=l) for _, c, s, w, l in LINES] + list(extra)
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, frameon=False)


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
        extra.append(Polygon([(0, 0)], color="#b9bdf2", label="G = 1 (schematic, paper)"))
    legend(fig, extra)
    fig.subplots_adjust(bottom=0.14, top=0.95, hspace=0.18, wspace=0.08)
    if mesh is not None:
        bar = fig.colorbar(mesh, ax=axes, shrink=0.6, pad=0.02)
        bar.set_label(r"exact $\langle G \rangle$ ($e^2/h$), thesis conductance maps")
    name = "born_fig4_numerics" if background else "born_fig4"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(DIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    fig.savefig(os.path.join(DIR, "born_all_panels.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig4(background=False)
    fig4(background=True)
    all_panels()
    print("figures written to", os.path.abspath(DIR))
