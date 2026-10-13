"""Phase diagrams of the g2g1 systems on wider ribbons, spin-resolved, in the colours of the
thesis (`born_style`).

Needs the CSV files of
    cargo run --release -p caroli-validation --example born_lines
    cargo run --release -p caroli-validation --example born_wide_maps -- <F|AF> <W> <configs> <map|E_F>
and writes into the same directory:
    born_wide_phase.png  ⟨G⟩ maps: published (W = 64), caroli W = 64 and W = 128, corrected lines;
    born_spin_maps.png   ⟨T↑⟩ and ⟨T↓⟩ at W = 128 with the corrected band edges of each block;
    born_width.png       cuts at fixed E_F for W = 64, 128, 256.

    python3 crates/caroli-validation/examples/plot_born_wide.py [csv directory]
"""

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from born_style import BLOCKS, LINES, born_fields, draw_lines, g_colorbar, g_map, line_handles, save

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "..", "target", "born_lines")
PUBLISHED = {
    "AF": "numerical_Antiferro_anti_400_256_x64_0.4_1.0_32Jx64e.csv",
    "F": "numerical_Ferro_ferro_400_256_x64_0.2_1.0_64Jx64e.csv",
}
TITLES = {"AF": r"'AF' (opposite-$g$), $n_I=0.4$", "F": r"'F' (same-$g$), $n_I=0.2$"}
JMAX = {"AF": 0.75, "F": 1.5}


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
    fields = born_fields(grid("M_up" + suffix), grid("mu_up" + suffix), grid("M_dn" + suffix), grid("mu_dn" + suffix), e)
    return j, e, grid("G_numerical"), fields


def crossings(case, energy, key, suffix):
    """J_I values where a Born line crosses the energy row closest to `energy`."""
    j, e, _, fields = published(case, suffix)
    ie = int(np.argmin(abs(e[:, 0] - energy)))
    f = fields[key][ie]
    out = []
    for k in np.where(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]:
        out.append(j[ie, k] - f[k] * (j[ie, k + 1] - j[ie, k]) / (f[k + 1] - f[k]))
    return out


def wide(case, w, tag):
    """Configuration averages (E index, J index) of T↑ and T↓ on the grid (J_I, E_F), or None."""
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
    jj, ee = np.meshgrid(js, es)
    return jj, ee, out


def phase_figure():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    for row, case in enumerate(("AF", "F")):
        j, e, g, fields = published(case)
        panels = [("published, W = 64", j, e, g)]
        for w in (64, 128):
            r = wide(case, w, "map")
            if r is not None:
                jj, ee, t = r
                panels.append((f"caroli, W = {w} ({t['n']} configurations)", jj, ee, t["T_up"] + t["T_dn"]))
        for col, (head, pj, pe, pg) in enumerate(panels):
            ax = axes[row, col]
            cf = g_map(ax, pj, pe, pg)
            draw_lines(ax, j, e, fields)
            ax.set_xlim(0.0, JMAX[case])
            ax.set_ylim(-0.16, 0.16)
            ax.set_title(head, fontsize=10)
            ax.set_xlabel(r"disorder $J_I$ $(\varepsilon_0)$")
        axes[row, 0].set_ylabel(f"{TITLES[case]}\n" + r"energy $E_F$ $(\varepsilon_0)$")
    fig.legend(handles=line_handles(), loc="lower center", ncol=6, fontsize=9, frameon=False, title="corrected SCBA")
    fig.subplots_adjust(bottom=0.13, top=0.95, hspace=0.3, wspace=0.06, right=0.9)
    cax = fig.add_axes([0.915, 0.13, 0.012, 0.82])
    g_colorbar(fig, cf, r"$\langle G\rangle$ ($e^2/h$)", cax=cax)
    save(fig, DIR, "born_wide_phase")


def spin_figure(w=128):
    if all(wide(case, w, "map") is None for case in ("AF", "F")):
        print(f"no W = {w} maps yet: born_spin_maps skipped")
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=True)
    for row, case in enumerate(("AF", "F")):
        j, e, _, fields = published(case)
        r = wide(case, w, "map")
        if r is None:
            continue
        jj, ee, t = r
        for col, (key, block, name) in enumerate((("T_up", "up", r"$\langle T_\uparrow\rangle$"), ("T_dn", "dn", r"$\langle T_\downarrow\rangle$"))):
            ax = axes[row, col]
            cf = g_map(ax, jj, ee, t[key])
            draw_lines(ax, j, e, fields, BLOCKS[block])
            ax.set_xlim(0.0, JMAX[case])
            ax.set_ylim(-0.16, 0.16)
            ax.set_title(f"{TITLES[case]}, W = {w} — {name}", fontsize=10)
            ax.set_xlabel(r"disorder $J_I$ $(\varepsilon_0)$")
        axes[row, 0].set_ylabel(r"energy $E_F$ $(\varepsilon_0)$")
    fig.legend(handles=line_handles(), loc="lower center", ncol=6, fontsize=9, frameon=False, title="corrected SCBA")
    fig.subplots_adjust(bottom=0.13, top=0.95, hspace=0.3, wspace=0.06, right=0.88)
    cax = fig.add_axes([0.9, 0.13, 0.015, 0.82])
    g_colorbar(fig, cf, r"transmission of the block ($e^2/h$)", cax=cax)
    save(fig, DIR, "born_spin_maps")


def width_figure():
    # Born lines marked on each cut, in their colours: (field, version suffix, label, style).
    marks = {
        "AF": [("m_up", "_thesis", r"published $\bar m_\uparrow=0$", ":"),
               ("m_up", "", r"corrected $\bar m_\uparrow=0$", "-.")],
        "F": [("ep_up", "", r"corrected E$\uparrow$ band edge", "-."),
              ("m_up", "_thesis", r"published $\bar m_\uparrow=0$", ":"),
              ("m_dn", "_thesis", r"published $\bar m_\downarrow=0$", ":")],
    }
    cuts = [("AF", 0.02, "0.02"), ("F", -0.06, "-0.06")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    colours = {64: "tab:blue", 128: "tab:orange", 256: "tab:green"}
    for ax, (case, energy, tag) in zip(axes, cuts):
        for w in (64, 128, 256):
            r = wide(case, w, "map" if w < 256 else tag)
            if r is None:
                continue
            jj, ee, t = r
            js, es = jj[0], ee[:, 0]
            ie = int(np.argmin(abs(es - energy)))
            if abs(es[ie] - energy) > 1e-9:
                continue
            ax.plot(js, t["T_up"][ie], "-o", color=colours[w], ms=3.5, label=rf"$T_\uparrow$, W = {w}")
            ax.plot(js, t["T_dn"][ie], "--s", color=colours[w], ms=3.0, mfc="white", label=rf"$T_\downarrow$, W = {w}")
        for key, suffix, label, style in marks[case]:
            for k, x in enumerate(crossings(case, energy, key, suffix)):
                ax.axvline(x, color=LINES[key][0], ls=style, lw=1.5, label=label if k == 0 else None)
        ax.set_xlim(0.0, JMAX[case])
        ax.set_ylim(-0.03, 1.12)
        ax.set_xlabel(r"disorder $J_I$ $(\varepsilon_0)$")
        ax.set_title(f"{TITLES[case]}, $E_F = {energy:+.2f}$", fontsize=10)
    axes[0].set_ylabel(r"transmission of the block $\langle T\rangle$ ($e^2/h$)")
    # One legend for both cuts, under them, clear of the curves.
    keys = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            keys.setdefault(label, handle)
    fig.legend(keys.values(), keys.keys(), loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.subplots_adjust(bottom=0.25, wspace=0.15)
    save(fig, DIR, "born_width")


if __name__ == "__main__":
    phase_figure()
    spin_figure(int(os.environ.get("SPIN_W", "128")))
    width_figure()
