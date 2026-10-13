"""Born lines over the published conductance maps, in the colours of the thesis (`born_style`:
grey scale for G in [0, 2.4], light blue around G = 1, blue around G = 2), published lines
(left) and corrected SCBA (right).

The maps are those of the paper's Fig. 1, averaged over the realisations of
`anomalous/Results/Antiferro/anti_400(256)x64_0.4_1.0/32Jx64e` (101) and
`anomalous/Results/Ferro/ferro_400(256)x64_0.2_1.0/64Jx64e` (50). `born_lines` writes them,
with both versions of the Born parameters on the same grid, to `numerical_*.csv`:
    cargo run --release -p caroli-validation --example born_lines

    python3 crates/caroli-validation/examples/plot_born_phase.py [csv directory]
"""

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from born_style import born_fields, draw_lines, g_colorbar, g_map, line_handles, save

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "..", "target", "born_lines")

# (published map with its Born parameters, title, J range of the paper's figure)
CASES = [
    ("numerical_Antiferro_anti_400_256_x64_0.4_1.0_32Jx64e.csv", r"'AF', $n_I=0.4$, $M_z=1$", 0.75),
    ("numerical_Ferro_ferro_400_256_x64_0.2_1.0_64Jx64e.csv", r"'F', $n_I=0.2$, $M_z=1$", 1.5),
]


def load(name):
    """Grids (E index, J index) of the map and of the published and corrected Born fields."""
    with open(os.path.join(DIR, name)) as f:
        rows = list(csv.DictReader(f))
    d = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    js, es = np.unique(d["J_I"]), np.unique(d["E_F"])
    assert d["J_I"][1] != d["J_I"][0] and d["E_F"][1] == d["E_F"][0], "J must run fastest"
    grid = lambda k: d[k].reshape(len(es), len(js))
    e = grid("E_F")
    fields = {
        version: born_fields(grid("M_up" + s), grid("mu_up" + s), grid("M_dn" + s), grid("mu_dn" + s), e)
        for version, s in (("published", "_thesis"), ("corrected", ""))
    }
    return grid("J_I"), e, grid("G_numerical"), fields


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=True)
    for row, (name, title, jmax) in enumerate(CASES):
        j, e, g, versions = load(name)
        for col, version in enumerate(("published", "corrected")):
            ax = axes[row, col]
            cf = g_map(ax, j, e, g)
            draw_lines(ax, j, e, versions[version])
            ax.set_xlim(0.0, jmax)
            ax.set_ylim(-0.16, 0.16)
            head = "published Born lines" if version == "published" else "corrected SCBA"
            ax.set_title(f"{title} — {head}", fontsize=11)
            ax.set_xlabel(r"disorder $J_I$ $(\varepsilon_0)$")
        axes[row, 0].set_ylabel(r"energy $E_F$ $(\varepsilon_0)$")
    fig.legend(handles=line_handles(), loc="lower center", ncol=6, fontsize=9, frameon=False)
    fig.subplots_adjust(bottom=0.11, top=0.95, hspace=0.3, wspace=0.08, right=0.86)
    for row in range(2):
        pos = axes[row, 1].get_position()
        cax = fig.add_axes([0.88, pos.y0, 0.015, pos.height])
        g_colorbar(fig, cf, r"$\langle G \rangle$ ($e^2/h$), published maps", cax=cax)
    save(fig, DIR, "born_phase")


if __name__ == "__main__":
    main()
