"""Phase diagrams of the g2g1 systems on wider ribbons, spin-resolved.

Needs the CSV files of
    cargo run --release -p caroli-validation --example born_lines
    cargo run --release -p caroli-validation --example born_wide_maps -- <F|AF> <W> <configs> <map|E_F>
and writes into the same directory:
    born_wide_phase.png/pdf  ⟨G⟩ maps: published (W = 64), caroli W = 64 and W = 128, corrected lines;
    born_spin_maps.png/pdf   ⟨T↑⟩ and ⟨T↓⟩ at W = 128 with the corrected band edges of each block;
    born_width.png/pdf       cuts at fixed E_F for W = 64, 128, 256.

    python3 crates/caroli-validation/examples/plot_born_wide.py [csv directory]
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
DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "..", "target", "born_lines")
B = -0.376406
PUBLISHED = {
    "AF": "numerical_Antiferro_anti_400_256_x64_0.4_1.0_32Jx64e.csv",
    "F": "numerical_Ferro_ferro_400_256_x64_0.2_1.0_64Jx64e.csv",
}
TITLES = {"AF": r"'AF' (opposite-$g$), $n_I=0.4$", "F": r"'F' (same-$g$), $n_I=0.2$"}
JMAX = {"AF": 0.75, "F": 1.5}
# Band edges and inversions of each block, coloured as in the thesis.
EDGES = {
    "up": [("ep", "blue", "-", r"E$\uparrow$ edge"), ("hp", "red", "-", r"H$\uparrow$ edge"), ("m", "black", ":", r"$\bar m_\uparrow=0$")],
    "dn": [("ep", "green", "--", r"E$\downarrow$ edge"), ("hp", "cyan", "--", r"H$\downarrow$ edge"), ("m", "gold", ":", r"$\bar m_\downarrow=0$")],
}


def g_colors(nc=64, vmin=0.0, vmax=2.4, d=1):
    """The thesis colour map: grey scale, light blue around G = 1, blue around G = 2."""
    u1, u2 = int(1.0 * nc / (vmax - vmin)), int(2.0 * nc / (vmax - vmin))
    c = np.array([[i / (nc - 1.0)] * 3 for i in range(nc)])
    c[u1 - d : u1 + d + 1] = [213.0 / 255.0, 237.0 / 255.0, 1.0]
    c[u2 - d : u2 + d + 1] = [60.0 / 255.0, 171.0 / 255.0, 1.0]
    return ListedColormap(c)


def read(name):
    path = os.path.join(DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def published(case, suffix=""):
    """Published map and Born fields on its grid (E index, J index): corrected SCBA, or the
    published lines with suffix "_thesis"."""
    d = read(PUBLISHED[case])
    js, es = np.unique(d["J_I"]), np.unique(d["E_F"])
    grid = lambda k: d[k].reshape(len(es), len(js))
    j, e = grid("J_I"), grid("E_F")
    lines = {}
    for s in ("up", "dn"):
        m, mu = grid("M_" + s + suffix), grid("mu_" + s + suffix)
        lines[s] = {"ep": m - mu - e, "hp": m + mu + e, "m": m}
    return j, e, grid("G_numerical"), lines


def crossings(case, energy, block, key, suffix):
    """J_I values where a Born line crosses the energy row closest to `energy`."""
    j, e, _, lines = published(case, suffix)
    ie = int(np.argmin(abs(e[:, 0] - energy)))
    f = lines[block][key][ie]
    out = []
    for k in np.where(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]:
        out.append(j[ie, k] - f[k] * (j[ie, k + 1] - j[ie, k]) / (f[k + 1] - f[k]))
    return out


def wide(case, w, tag):
    """Configuration averages (E index, J index) of T↑ and T↓, or None."""
    d = read(f"wide_{case}_W{w}_{tag}.csv")
    if d is None:
        return None
    js, es = np.unique(d["J_I"]), np.unique(d["E_F"])
    out = {}
    for key in ("T_up", "T_dn"):
        acc = np.zeros((len(es), len(js)))
        cnt = np.zeros((len(es), len(js)))
        for j, e, t in zip(d["J_I"], d["E_F"], d[key]):
            ie, ij = np.searchsorted(es, e), np.searchsorted(js, j)
            acc[ie, ij] += t
            cnt[ie, ij] += 1
        out[key] = acc / cnt
    out["n"] = int(cnt.max())
    return js, es, out


def draw_lines(ax, j, e, lines, blocks):
    for s in blocks:
        for key, colour, style, _ in EDGES[s]:
            ax.contour(j, e, lines[s][key], [0.0], colors=colour, linestyles=style, linewidths=1.8)


def phase_figure():
    cmap = g_colors()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    for row, case in enumerate(("AF", "F")):
        j, e, g, lines = published(case)
        panels = [("published, W = 64", j, e, g)]
        for w in (64, 128):
            r = wide(case, w, "map")
            if r is not None:
                js, es, t = r
                jj, ee = np.meshgrid(js, es)
                panels.append((f"caroli, W = {w} ({t['n']} configurations)", jj, ee, t["T_up"] + t["T_dn"]))
        for col, (head, pj, pe, pg) in enumerate(panels):
            ax = axes[row, col]
            cf = ax.pcolormesh(pj, pe, pg, cmap=cmap, vmin=0.0, vmax=2.4, shading="nearest")
            draw_lines(ax, j, e, lines, ("up", "dn"))
            ax.set_xlim(0.0, JMAX[case])
            ax.set_ylim(-0.16, 0.16)
            ax.set_title(head, fontsize=10)
            ax.set_xlabel(r"disorder $J_I$ $(\varepsilon_0)$")
        axes[row, 0].set_ylabel(f"{TITLES[case]}\n" + r"energy $E_F$ $(\varepsilon_0)$")
    handles = [Line2D([], [], color=c, ls=s, lw=1.8, label=l) for b in ("up", "dn") for _, c, s, l in EDGES[b]]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9, frameon=False, title="corrected SCBA")
    fig.subplots_adjust(bottom=0.13, top=0.95, hspace=0.3, wspace=0.06, right=0.9)
    cax = fig.add_axes([0.915, 0.13, 0.012, 0.82])
    fig.colorbar(cf, cax=cax, ticks=[0, 0.5, 1, 1.5, 2, 2.4]).set_label(r"$\langle G\rangle$ ($e^2/h$)")
    save(fig, "born_wide_phase")


def spin_figure(w=128):
    if all(wide(case, w, "map") is None for case in ("AF", "F")):
        print(f"no W = {w} maps yet: born_spin_maps skipped")
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=True)
    for row, case in enumerate(("AF", "F")):
        j, e, _, lines = published(case)
        r = wide(case, w, "map")
        if r is None:
            continue
        js, es, t = r
        for col, (key, block, name) in enumerate((("T_up", "up", r"$\langle T_\uparrow\rangle$"), ("T_dn", "dn", r"$\langle T_\downarrow\rangle$"))):
            ax = axes[row, col]
            mesh = ax.pcolormesh(js, es, t[key], cmap="Greys", vmin=0.0, vmax=1.0, shading="nearest")
            draw_lines(ax, j, e, lines, (block,))
            ax.set_xlim(0.0, JMAX[case])
            ax.set_ylim(-0.16, 0.16)
            ax.set_title(f"{TITLES[case]}, W = {w} — {name}", fontsize=10)
            ax.set_xlabel(r"disorder $J_I$ $(\varepsilon_0)$")
        axes[row, 0].set_ylabel(r"energy $E_F$ $(\varepsilon_0)$")
    handles = [Line2D([], [], color=c, ls=s, lw=1.8, label=l) for b in ("up", "dn") for _, c, s, l in EDGES[b]]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9, frameon=False, title="corrected SCBA")
    fig.subplots_adjust(bottom=0.13, top=0.95, hspace=0.3, wspace=0.06, right=0.88)
    cax = fig.add_axes([0.9, 0.13, 0.015, 0.82])
    fig.colorbar(mesh, cax=cax).set_label("transmission of the block")
    save(fig, "born_spin_maps")


def width_figure():
    # Born lines marked on each cut: (block, field, version suffix, label, colour, style).
    marks = {
        "AF": [("up", "m", "_thesis", r"published $\bar m_\uparrow=0$", "black", ":"),
               ("up", "m", "", r"corrected $\bar m_\uparrow=0$", "black", "-.")],
        "F": [("up", "ep", "", r"corrected E$\uparrow$ edge", "blue", "-."),
              ("up", "m", "_thesis", r"published $\bar m_\uparrow=0$", "black", ":"),
              ("dn", "m", "_thesis", r"published $\bar m_\downarrow=0$", "goldenrod", ":")],
    }
    cuts = [("AF", 0.02, "0.02"), ("F", -0.06, "-0.06")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    colours = {64: "tab:blue", 128: "tab:orange", 256: "tab:green"}
    for ax, (case, energy, tag) in zip(axes, cuts):
        for w in (64, 128, 256):
            r = wide(case, w, "map" if w < 256 else tag)
            if r is None:
                continue
            js, es, t = r
            ie = int(np.argmin(abs(es - energy)))
            if abs(es[ie] - energy) > 1e-9:
                continue
            ax.plot(js, t["T_up"][ie], "-o", color=colours[w], ms=3.5, label=rf"$T_\uparrow$, W = {w}")
            ax.plot(js, t["T_dn"][ie], "--s", color=colours[w], ms=3.0, mfc="white", label=rf"$T_\downarrow$, W = {w}")
        for block, key, suffix, label, colour, style in marks[case]:
            for k, x in enumerate(crossings(case, energy, block, key, suffix)):
                ax.axvline(x, color=colour, ls=style, lw=1.3, label=label if k == 0 else None)
        ax.set_xlim(0.0, JMAX[case])
        ax.set_ylim(-0.03, 1.12)
        ax.set_xlabel(r"disorder $J_I$ $(\varepsilon_0)$")
        ax.set_title(f"{TITLES[case]}, $E_F = {energy:+.2f}$", fontsize=10)
        ax.legend(fontsize=8, ncol=2, loc="lower left" if case == "AF" else "center right")
    axes[0].set_ylabel("transmission of the block (configuration mean)")
    fig.subplots_adjust(wspace=0.15)
    save(fig, "born_width")


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(DIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("written", os.path.abspath(os.path.join(DIR, f"{name}.png")))


if __name__ == "__main__":
    phase_figure()
    spin_figure(int(os.environ.get("SPIN_W", "128")))
    width_figure()
