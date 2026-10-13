"""Figures of the topological-insulator study (`ti_magnetic.rs`), from the CSV files it writes.

    cargo run --release -p caroli --example ti_magnetic -- all OUT_DIR
    python3 crates/caroli/examples/plot_ti_magnetic.py OUT_DIR

- `maps.png`: ⟨G⟩, ⟨T↑⟩ and ⟨T↓⟩ against (J_I, E_F) for both couplings, in the colours of the
  thesis (grey scale, light blue on G = 1, blue on G = 2), with the Born lines when `born.csv`
  is there: band edges of each spin block (solid: spin up, dashed: spin down) and mass
  inversions (dotted).
- `local.png`: the current injected by the left lead (arrows, averaged over 4 × 4 sites) over
  the spin-up fraction of the injected density, in the six regimes.
- `ldos.png`: the local densities of states of the two spins in the six regimes.

Needs numpy and matplotlib.
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

G_MAX = 2.4
G_LEVELS = np.linspace(0.0, G_MAX, int(G_MAX * 20) + 1)
G_TICKS = [0.0, 0.5, 1.0, 1.5, 2.0, G_MAX]
B = -0.376406  # BHZ parameter B: a block is topological where M/B > 0.


def thesis_colors(nc=64, d=1):
    """Grey scale over [0, 2.4], light blue around G = 1, blue around G = 2."""
    u1, u2 = int(1.0 * nc / G_MAX), int(2.0 * nc / G_MAX)
    c = np.array([[i / (nc - 1.0)] * 3 for i in range(nc)])
    c[u1 - d : u1 + d + 1] = [213.0 / 255.0, 237.0 / 255.0, 1.0]
    c[u2 - d : u2 + d + 1] = [60.0 / 255.0, 171.0 / 255.0, 1.0]
    return ListedColormap(c)


CMAP = thesis_colors()
# Born lines: zeros of each field, (colour, style, width, label).
LINES = {
    "ep_up": ("blue", "-", 1.5, r"E$\uparrow$ band edge"),
    "hp_up": ("red", "-", 1.5, r"H$\uparrow$ band edge"),
    "ep_dn": ("green", "--", 2.0, r"E$\downarrow$ band edge"),
    "hp_dn": ("cyan", "--", 2.0, r"H$\downarrow$ band edge"),
    "m_up": ("black", ":", 2.0, r"$\bar M_\uparrow = 0$"),
    "m_dn": ("gold", ":", 2.0, r"$\bar M_\downarrow = 0$"),
}


def read(directory, name):
    path = os.path.join(directory, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def column(rows, key, kind=float):
    return np.array([kind(r[key]) for r in rows])


def grid(rows, key):
    """Mean of `key` over the configurations on the (E_F, J_I) grid of one coupling."""
    j, e, v = column(rows, "j_i"), column(rows, "e_f"), column(rows, key)
    js, es = np.unique(j), np.unique(e)
    total, count = np.zeros((len(es), len(js))), np.zeros((len(es), len(js)))
    np.add.at(total, (np.searchsorted(es, e), np.searchsorted(js, j)), v)
    np.add.at(count, (np.searchsorted(es, e), np.searchsorted(js, j)), 1)
    jj, ee = np.meshgrid(js, es)
    return jj, ee, total / count


def maps(directory):
    rows = read(directory, "maps.csv")
    if rows is None:
        return
    born = read(directory, "born.csv")
    couplings = ["same-g", "opposite-g"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=True, layout="constrained")
    for r, coupling in enumerate(couplings):
        sub = [x for x in rows if x["coupling"] == coupling]
        conc = sub[0]["conc"]
        configs = len(np.unique(column(sub, "config")))
        for x in sub:
            x["g"] = str(float(x["t_up"]) + float(x["t_dn"]))
        for c, (key, label) in enumerate([("g", r"$\langle G\rangle$"), ("t_up", r"$\langle T_\uparrow\rangle$"), ("t_dn", r"$\langle T_\downarrow\rangle$")]):
            ax = axes[r, c]
            j, e, v = grid(sub, key)
            cf = ax.contourf(j, e, v, G_LEVELS, vmin=0.0, vmax=G_MAX, cmap=CMAP, extend="max")
            if born is not None:
                b = [x for x in born if x["coupling"] == coupling]
                bj, be, m_up = grid(b, "m_up")
                fields = {
                    "ep_up": m_up - grid(b, "mu_up")[2] - be,
                    "hp_up": m_up + grid(b, "mu_up")[2] + be,
                    "ep_dn": grid(b, "m_dn")[2] - grid(b, "mu_dn")[2] - be,
                    "hp_dn": grid(b, "m_dn")[2] + grid(b, "mu_dn")[2] + be,
                    "m_up": m_up,
                    "m_dn": grid(b, "m_dn")[2],
                }
                keys = {"g": LINES, "t_up": ("ep_up", "hp_up", "m_up"), "t_dn": ("ep_dn", "hp_dn", "m_dn")}[key]
                for k in keys:
                    colour, style, width, _ = LINES[k]
                    if np.nanmin(fields[k]) < 0.0 < np.nanmax(fields[k]):
                        ax.contour(bj, be, fields[k], [0.0], colors=colour, linestyles=style, linewidths=width)
            ax.set_title(f"{coupling}, $n_I = {conc}$: {label} ({configs} configurations)", fontsize=10)
            ax.set_xlabel(r"$J_I$")
            if c == 0:
                ax.set_ylabel(r"$E_F$")
    bar = fig.colorbar(cf, ax=axes, ticks=G_TICKS, format="%1.1f", shrink=0.8)
    bar.set_label(r"$e^2/h$")
    if born is not None:
        handles = [Line2D([], [], color=c, ls=s, lw=w, label=l) for c, s, w, l in LINES.values()]
        fig.legend(handles=handles, loc="outside lower center", ncol=6, fontsize=9)
    save(fig, directory, "maps")


def site_grid(rows, key):
    """A per-site quantity on the (y, x) grid of the ribbon."""
    x, y, v = column(rows, "x"), column(rows, "y"), column(rows, key)
    xs, ys = np.unique(x), np.unique(y)
    out = np.full((len(ys), len(xs)), np.nan)
    out[np.searchsorted(ys, y), np.searchsorted(xs, x)] = v
    return xs, ys, out


def blocks(a, n):
    """Means over n × n blocks."""
    h, w = (a.shape[0] // n) * n, (a.shape[1] // n) * n
    return a[:h, :w].reshape(h // n, n, w // n, n).mean(axis=(1, 3))


def regimes(rows):
    out = {}
    for r in rows:
        out.setdefault(int(r["regime"]), []).append(r)
    return [out[k] for k in sorted(out)]


def title(rows, totals):
    r = rows[0]
    g = next((float(t["g"]) for t in totals if t["regime"] == r["regime"]), float("nan")) if totals else float("nan")
    return rf"{r['coupling']}: $J_I = {float(r['j_i']):g}$, $E_F = {float(r['e_f']):g}$, $G = {g:.2f}$"


def local(directory):
    rows = read(directory, "local.csv")
    if rows is None:
        return
    totals = read(directory, "local_totals.csv")
    imp = read(directory, "impurities.csv")
    groups = regimes(rows)
    fig, axes = plt.subplots(len(groups), 1, figsize=(14, 2.6 * len(groups)), sharex=True, layout="constrained")
    for ax, g in zip(np.atleast_1d(axes), groups):
        xs, ys, up = site_grid(g, "rho_up")
        _, _, dn = site_grid(g, "rho_dn")
        frac = up / np.where(up + dn > 0, up + dn, np.nan)
        im = ax.pcolormesh(xs, ys, frac, cmap="coolwarm", vmin=0.0, vmax=1.0, shading="nearest")
        n = 4
        jx, jy = blocks(site_grid(g, "jx")[2], n), blocks(site_grid(g, "jy")[2], n)
        bx, by = blocks(np.tile(xs, (len(ys), 1)), n), blocks(np.tile(ys[:, None], (1, len(xs))), n)
        norm = np.nanmax(np.hypot(jx, jy))
        ax.quiver(bx, by, jx / norm, jy / norm, scale=25, width=0.0012, color="k")
        if imp is not None:
            ax.plot(column(imp, "x"), column(imp, "y"), ",", color="0.5", alpha=0.4)
        ax.set_aspect("equal")
        ax.set_title(title(g, totals), fontsize=10)
        ax.set_ylabel("y")
    np.atleast_1d(axes)[-1].set_xlabel("x")
    bar = fig.colorbar(im, ax=axes, shrink=0.6)
    bar.set_label(r"$\rho_\uparrow / (\rho_\uparrow + \rho_\downarrow)$ of the injected density")
    save(fig, directory, "local")


def ldos(directory):
    rows = read(directory, "local.csv")
    if rows is None:
        return
    totals = read(directory, "local_totals.csv")
    groups = regimes(rows)
    fig, axes = plt.subplots(len(groups), 2, figsize=(16, 2.2 * len(groups)), sharex=True, sharey=True, layout="constrained")
    axes = np.atleast_2d(axes)
    for row, g in zip(axes, groups):
        for ax, key, spin in zip(row, ("ldos_up", "ldos_dn"), (r"\uparrow", r"\downarrow")):
            xs, ys, v = site_grid(g, key)
            im = ax.pcolormesh(xs, ys, v, cmap="magma", shading="nearest", vmin=0.0, vmax=np.nanpercentile(v, 99.5))
            ax.set_aspect("equal")
            ax.set_title(title(g, totals) + rf", LDOS$_{spin}$", fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.7)
    save(fig, directory, "ldos")


def save(fig, directory, name):
    path = os.path.join(directory, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("written", os.path.abspath(path))


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    maps(directory)
    local(directory)
    ldos(directory)
