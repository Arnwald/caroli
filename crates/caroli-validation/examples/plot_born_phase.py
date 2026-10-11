"""Born lines over the numerical conductance phase diagrams, in the colours of the thesis
(`anomalous/ipy/plot_psb.py`: grey scale for G in [0, 2.4], light blue around G = 1, blue
around G = 2), published lines (left) and corrected SCBA (right).

Needs the CSV files of
    cargo run --release -p caroli-validation --example born_lines
and the averaged maps of the thesis archive (`anomalous/ipy/data/{A,F}/*.npy`).

    python3 crates/caroli-validation/examples/plot_born_phase.py [thesis archive] [csv directory]
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Projects/electronic-transport")
DIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "..", "..", "target", "born_lines")
DATA = os.path.join(ARCHIVE, "anomalous", "ipy", "data")

# Born lines drawn in the thesis phase diagrams (plot_psb.py): band edges and m = 0.
LINES = [
    ("ep_up", "blue", "-", 1.2, r"E$\uparrow$ band edge"),
    ("hp_up", "red", "-", 1.2, r"H$\uparrow$ band edge"),
    ("ep_dn", "green", "--", 2.0, r"E$\downarrow$ band edge"),
    ("hp_dn", "cyan", "--", 2.0, r"H$\downarrow$ band edge"),
    ("m_up", "black", ":", 2.0, r"$m_\uparrow = 0$"),
    ("m_dn", "yellow", ":", 2.0, r"$m_\downarrow = 0$"),
]

# (Born run, numerical map, title, J range of the thesis figure)
CASES = [
    (2, "A/G_A.256x64.NI04.Mz1", r"'AF', $n_I=0.4$, $M_z=1$", 0.75),
    (1, "F/G_F.256x64.NI02.Mz1", r"'F', $n_I=0.2$, $M_z=1$", 1.5),
]


def g_colors(nc=64, vmin=0.0, vmax=2.4, d=1):
    """The thesis colour map: grey scale, light blue around G = 1, blue around G = 2."""
    u1 = int(1.0 * nc / (vmax - vmin))
    u2 = int(2.0 * nc / (vmax - vmin))
    c = np.zeros((nc, 3))
    for i in range(nc):
        c[i, :] = i / (nc - 1.0)
    c[u1 - d : u1 + d + 1] = [213.0 / 255.0, 237.0 / 255.0, 1.0]
    c[u2 - d : u2 + d + 1] = [60.0 / 255.0, 171.0 / 255.0, 1.0]
    return ListedColormap(c)


def born(run):
    with open(os.path.join(DIR, f"panel_{run:02d}.csv")) as f:
        rows = list(csv.DictReader(f))
    n = int(round(np.sqrt(len(rows))))
    col = lambda k: np.array([float(r[k]) for r in rows]).reshape(n, n)
    e = col("E_F")
    out = {}
    for version, suffix in (("published", "_thesis"), ("corrected", "")):
        mu, muu = col("M_up" + suffix), col("mu_up" + suffix)
        md, mud = col("M_dn" + suffix), col("mu_dn" + suffix)
        out[version] = {
            "ep_up": mu - muu - e,
            "hp_up": mu + muu + e,
            "ep_dn": md - mud - e,
            "hp_dn": md + mud + e,
            "m_up": mu,
            "m_dn": md,
        }
    return col("J_I"), e, out


def main():
    cmap = g_colors()
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=True)
    for row, (run, stem, title, jmax) in enumerate(CASES):
        g = np.load(os.path.join(DATA, stem + ".npy"))
        jn = np.load(os.path.join(DATA, stem.replace("/G_", "/JI_") + ".npy"))
        en = np.load(os.path.join(DATA, stem.replace("/G_", "/EF_") + ".npy"))
        j, e, versions = born(run)
        for col, version in enumerate(("published", "corrected")):
            ax = axes[row, col]
            cf = ax.contourf(jn, en, g, np.linspace(0.0, 2.4, 49), vmin=0.0, vmax=2.4, cmap=cmap, extend="max")
            for key, colour, style, width, _ in LINES:
                ax.contour(j, e, versions[version][key], [0.0], colors=colour, linestyles=style, linewidths=width)
            ax.set_xlim(0.0, jmax)
            ax.set_ylim(-0.16, 0.16)
            head = "published Born lines" if version == "published" else "corrected SCBA"
            ax.set_title(f"{title} — {head}", fontsize=11)
            ax.set_xlabel(r"disorder $J_I$ $(\varepsilon_0)$")
        axes[row, 0].set_ylabel(r"energy $E_F$ $(\varepsilon_0)$")
    handles = [Line2D([], [], color=c, ls=s, lw=w, label=l) for _, c, s, w, l in LINES]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9, frameon=False)
    fig.subplots_adjust(bottom=0.11, top=0.95, hspace=0.3, wspace=0.08, right=0.86)
    for row in range(2):
        pos = axes[row, 1].get_position()
        cax = fig.add_axes([0.88, pos.y0, 0.015, pos.height])
        bar = fig.colorbar(cf, cax=cax, ticks=[0, 0.5, 1, 1.5, 2, 2.4])
        bar.set_label(r"$\langle G \rangle$ ($e^2/h$), exact numerics")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(DIR, f"born_phase.{ext}"), dpi=150, bbox_inches="tight")
    print("written", os.path.abspath(os.path.join(DIR, "born_phase.png")))


if __name__ == "__main__":
    main()
