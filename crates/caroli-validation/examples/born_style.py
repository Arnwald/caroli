"""Shared style of the Born addendum figures, imported by the `plot_born_*.py` scripts.

- Maps of ⟨G⟩, or of the transmission of one spin block, all in the colours of the thesis
  (`anomalous/src/plot_psb.py`): a grey scale over [0, 2.4] (e²/h), light blue on the G = 1
  plateau and blue on G = 2, drawn as filled contours with 49 levels, as there.
- Born lines (band edges and inversions of each spin block) in the thesis colours.
- PNG output at 300 dpi: PDF viewers draw hairlines between the cells of a map.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

G_MAX = 2.4
G_LEVELS = np.linspace(0.0, G_MAX, int(G_MAX * 20) + 1)
G_TICKS = [0.0, 0.5, 1.0, 1.5, 2.0, G_MAX]


def g_colors(nc=64, vmin=0.0, vmax=G_MAX, d=1):
    """The thesis colour map: grey scale, light blue around G = 1, blue around G = 2."""
    u1, u2 = int(1.0 * nc / (vmax - vmin)), int(2.0 * nc / (vmax - vmin))
    c = np.array([[i / (nc - 1.0)] * 3 for i in range(nc)])
    c[u1 - d : u1 + d + 1] = [213.0 / 255.0, 237.0 / 255.0, 1.0]
    c[u2 - d : u2 + d + 1] = [60.0 / 255.0, 171.0 / 255.0, 1.0]
    return ListedColormap(c)


CMAP = g_colors()
# The colour of the G = 1 plateau, for the schematic plateau of the paper's Fig. 4.
PLATEAU_1 = CMAP(1.0 / G_MAX)


def g_map(ax, j, e, g):
    """⟨G⟩ or a block's ⟨T⟩ (e²/h) on the (J_I, E_F) grid (E index, J index)."""
    return ax.contourf(j, e, g, G_LEVELS, vmin=0.0, vmax=G_MAX, cmap=CMAP, extend="max")


def g_colorbar(fig, mappable, label, **kwargs):
    bar = fig.colorbar(mappable, ticks=G_TICKS, format="%1.1f", **kwargs)
    bar.set_label(label)
    return bar


# Born lines, zeros of each field: (colour, line style, width, label), as in the thesis.
LINES = {
    "ep_up": ("blue", "-", 1.5, r"E$\uparrow$ band edge"),
    "hp_up": ("red", "-", 1.5, r"H$\uparrow$ band edge"),
    "ep_dn": ("green", "--", 2.0, r"E$\downarrow$ band edge"),
    "hp_dn": ("cyan", "--", 2.0, r"H$\downarrow$ band edge"),
    "m_up": ("black", ":", 2.0, r"$\bar m_\uparrow = 0$"),
    "m_dn": ("gold", ":", 2.0, r"$\bar m_\downarrow = 0$"),
}
# The renormalised chemical potentials, drawn only on the panels of every run.
MU_LINES = {
    "mu_up": ("black", "-.", 2.0, r"$\mu_\uparrow$"),
    "mu_dn": ("gold", "-.", 2.0, r"$\mu_\downarrow$"),
}
BLOCKS = {"up": ("ep_up", "hp_up", "m_up"), "dn": ("ep_dn", "hp_dn", "m_dn")}


def born_fields(m_up, mu_up, m_dn, mu_dn, e):
    """The fields whose zeros are the Born lines, from the renormalised masses and chemical
    potentials of each block on the (J_I, E_F) grid."""
    return {
        "ep_up": m_up - mu_up - e,
        "hp_up": m_up + mu_up + e,
        "ep_dn": m_dn - mu_dn - e,
        "hp_dn": m_dn + mu_dn + e,
        "m_up": m_up,
        "m_dn": m_dn,
        "mu_up": mu_up + e,
        "mu_dn": mu_dn + e,
    }


def draw_lines(ax, j, e, fields, keys=tuple(LINES)):
    styles = {**LINES, **MU_LINES}
    for key in keys:
        colour, style, width, _ = styles[key]
        ax.contour(j, e, fields[key], [0.0], colors=colour, linestyles=style, linewidths=width)


def line_handles(keys=tuple(LINES)):
    styles = {**LINES, **MU_LINES}
    return [Line2D([], [], color=c, ls=s, lw=w, label=l) for c, s, w, l in (styles[k] for k in keys)]


def save(fig, directory, name, dpi=300):
    path = os.path.join(directory, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("written", os.path.abspath(path))
