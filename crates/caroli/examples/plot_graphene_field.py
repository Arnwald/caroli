"""Figures of the graphene study (`graphene_field.rs`) in the layout of A. Demion and A. D. Verga,
Phys. Rev. B 90, 085412 (2014), from the CSV files the example writes.

    cargo run --release -p caroli --example graphene_field -- all OUT_DIR
    python3 crates/caroli/examples/plot_graphene_field.py OUT_DIR

- `fig3_clean.png`: current j(t) and pairs N(t) of the clean sheet, and j/E, N/E against t√E
  with the slopes of Eq. (18) per unit cell (the currents and pairs are per cell, both spins).
- `fig4_dos.png`: densities of states per cell, paramagnetic (a) and magnetic (b).
- `fig6_spreading.png`: width w(t) of a wave packet (a) and the exponent β against J_I (b).
- `fig7_disorder.png`: current and pairs with disorder, as the paper's Fig. 7 (a–f).
- `fig8_ldos.png`: distribution of ln(ρ_i/ρ) at ε = 0.2 (a), and maps of ρ↑(+ε)/ρ↑(−ε) (b) and
  ρ↑/ρ↓ (c) at ε = 0.1, J_I = 1, with the impurities (spin up blue, spin down red). The total
  local density of states is the same at ±ε, and ρ↑(−ε) = ρ↓(+ε): the electron–hole map of one
  spin is the spin map (the script prints both symmetries as measured).

Needs numpy and matplotlib.
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SIGMA0 = 2 / np.pi**2  # 4e²/πh with e = ħ = 1
VF = np.sqrt(3) / 2
CELL = np.sqrt(3) / 2  # area of the unit cell: the currents and pairs are per cell


def read(directory, name):
    path = os.path.join(directory, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def column(rows, key):
    return np.array([float(r[key]) for r in rows])


def colours(count):
    return plt.cm.jet(np.linspace(0.0, 1.0, max(count, 2)))


def save(fig, directory, name):
    path = os.path.join(directory, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("written", os.path.abspath(path))


def series(axes, rows, key, scaled):
    """Curves of `key` against t (left) and `key`/E against t√E (right), one per field."""
    fields = sorted({float(r["field"]) for r in rows})
    for c, f in zip(colours(len(fields)), fields):
        sub = [r for r in rows if float(r["field"]) == f]
        t, v = column(sub, "t"), column(sub, key)
        axes[0].plot(t, 100 * v, color=c, lw=1)
        keep = t * np.sqrt(f) <= 7
        axes[1].plot(t[keep] * np.sqrt(f), v[keep] / f, color=c, lw=1)
    axes[0].set_xlabel("$t$")
    axes[1].set_xlabel(r"$t\sqrt{E}$")
    if scaled is not None:
        x = np.linspace(0, 7, 2)
        axes[1].plot(x, scaled * x, "k:", lw=1, label=f"slope {scaled:.3f} (Eq. 18, per cell)")
        axes[1].legend(fontsize=8, loc="upper left")


def fig3(directory):
    rows = read(directory, "clean.csv")
    if rows is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), layout="constrained")
    series(axes[0], rows, "j", CELL * SIGMA0 * np.sqrt(VF))
    series(axes[1], rows, "pairs", CELL * SIGMA0 * np.sqrt(VF) / (2 * VF))
    axes[0, 0].set_ylabel(r"$j \times 100$")
    axes[0, 1].set_ylabel("$j/E$")
    axes[1, 0].set_ylabel(r"$N \times 100$")
    axes[1, 1].set_ylabel("$N/E$")
    fig.suptitle("Clean graphene (Fig. 3): E = 0.002 … 0.02, blue to red", fontsize=10)
    save(fig, directory, "fig3_clean")


def fig4(directory):
    rows = read(directory, "dos.csv")
    if rows is None:
        return
    fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True, layout="constrained")
    for ax, phase, title in zip(axes, ("para", "magnetic"), ("(a) paramagnetic", "(b) magnetic")):
        for c, j in zip(("k", "tab:blue", "tab:green", "tab:red"), (0.1, 0.5, 1.0, 1.5)):
            sub = [r for r in rows if r["phase"] == phase and float(r["j_i"]) == j]
            ax.plot(column(sub, "energy"), column(sub, "dos"), color=c, lw=1, label=f"$J_I = {j}$")
        ax.set_xlim(-4, 4)
        ax.set_ylim(0, 2)
        ax.set_ylabel(r"$\rho(\epsilon)$")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel(r"$\epsilon$")
    save(fig, directory, "fig4_dos")


def fig6(directory):
    rows, fits = read(directory, "spreading.csv"), read(directory, "spreading_beta.csv")
    if rows is None or fits is None:
        return
    fig, axes = plt.subplots(2, 1, figsize=(6, 9), layout="constrained")
    js = sorted({float(r["j_i"]) for r in rows})
    for j, c in zip(js, colours(len(js))):
        sub = [r for r in rows if r["phase"] == "magnetic" and float(r["j_i"]) == j]
        t, w = column(sub, "t"), column(sub, "width")
        axes[0].loglog(t[t > 0], w[t > 0], color=c, lw=1, label=f"$J_I = {j:g}$")
    axes[0].set_xlabel("$t$")
    axes[0].set_ylabel("$w(t)$ (units of $a$)")
    axes[0].set_title("(a) magnetic case, every $J_I$", fontsize=10)
    axes[0].legend(fontsize=6, ncol=3)
    phases = {"para": ("Paramagnetic", "tab:blue", "o"), "magnetic": ("Antiferromagnetic", "tab:red", "s")}
    for phase, (label, colour, marker) in phases.items():
        sub = [r for r in fits if r["phase"] == phase]
        axes[1].plot(column(sub, "j_i"), column(sub, "beta"), marker=marker, color=colour, label=label)
    axes[1].axhline(0.5, color="0.6", lw=0.8, ls=":")
    axes[1].set_xlabel("$J_I$")
    axes[1].set_ylabel(r"$\beta$")
    axes[1].set_title(r"(b) $w \sim t^\beta$", fontsize=10)
    axes[1].legend(fontsize=8)
    save(fig, directory, "fig6_spreading")


def fig7(directory):
    rows = read(directory, "disorder.csv")
    if rows is None:
        return
    fig, axes = plt.subplots(2, 5, figsize=(20, 7), layout="constrained")
    for col, phase in ((0, "para"), (2, "magnetic")):
        sub = [r for r in rows if r["phase"] == phase and float(r["j_i"]) == 0.2]
        series(axes[0, col : col + 2], sub, "j", None)
        series(axes[1, col : col + 2], sub, "pairs", None)
        name = "paramagnetic" if phase == "para" else "magnetic"
        axes[0, col].set_title(f"({'a' if phase == 'para' else 'c'}) {name}, $J_I = 0.2$", fontsize=10)
        axes[1, col].set_title(f"({'b' if phase == 'para' else 'd'})", fontsize=10)
        axes[0, col].set_ylabel(r"$j \times 100$")
        axes[1, col].set_ylabel(r"$N \times 100$")
        axes[0, col + 1].set_ylabel("$j/E$")
        axes[1, col + 1].set_ylabel("$N/E$")
    sub = [r for r in rows if r["phase"] == "magnetic" and float(r["field"]) == 0.02]
    js = sorted({float(r["j_i"]) for r in sub})
    for c, j in zip(colours(len(js)), js):
        s = [r for r in sub if float(r["j_i"]) == j]
        axes[0, 4].plot(column(s, "t"), column(s, "j") / 0.02, color=c, lw=1)
        axes[1, 4].plot(column(s, "t"), column(s, "pairs") / 0.02, color=c, lw=1)
    axes[0, 4].axhline(CELL * SIGMA0 / 2, color="k", ls="--", lw=1)
    axes[0, 4].set_ylabel("$j/E$")
    axes[1, 4].set_ylabel("$N/E$")
    axes[0, 4].set_title(r"(e) magnetic, $E = 0.02$, $J_I$ = 0 (blue) … 1 (red); dashed: $\sigma_0/2$ per cell", fontsize=9)
    axes[1, 4].set_title("(f)", fontsize=10)
    for ax in axes[:, 4]:
        ax.set_xlabel("$t$")
    save(fig, directory, "fig7_disorder")


def fig8(directory):
    hist, maps = read(directory, "ldos_histogram.csv"), read(directory, "ldos_map.csv")
    imp = read(directory, "ldos_impurities.csv")
    if hist is None or maps is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained")
    for c, j in zip(("k", "tab:blue", "tab:green", "tab:red"), (0.1, 0.5, 1.0, 1.5)):
        rho = column([r for r in hist if float(r["j_i"]) == j], "ldos")
        r = np.log(rho[rho > 0] / rho.mean())
        edges = np.arange(np.floor(r.min() / 0.2) * 0.2, r.max() + 0.2, 0.2)
        counts, edges = np.histogram(r, bins=edges)
        f = counts / (len(r) * 0.2)
        centres = 0.5 * (edges[1:] + edges[:-1])
        axes[0].semilogy(centres[f > 0], f[f > 0], "o", ms=3, color=c, label=f"$J_I = {j}$")
    axes[0].set_xlim(-5, 3)
    axes[0].set_xlabel(r"$R = \ln[\rho(i, \epsilon)/\rho(\epsilon)]$")
    axes[0].set_ylabel(r"$f_\epsilon[R]$")
    axes[0].set_title(r"(a) $\epsilon = 0.2$, magnetic", fontsize=10)
    axes[0].legend(fontsize=8)
    x, y = column(maps, "x"), column(maps, "y")
    up, down = column(maps, "up_plus"), column(maps, "down_plus")
    up_minus, down_minus = column(maps, "up_minus"), column(maps, "down_minus")
    # The symmetries of the model (see graphene_field.rs), to the precision of the moments.
    charge = np.abs(np.log((up + down) / (up_minus + down_minus))).max()
    mirror = np.abs(np.log(up / down_minus)).max()
    print(f"fig8: max |ln rho(+e)/rho(-e)| = {charge:.1e}, max |ln rho_up(e)/rho_down(-e)| = {mirror:.1e}")
    panels = (
        (axes[1], np.log(up / up_minus), r"(b) spin up: $\ln \rho_{\uparrow}(+\epsilon)/\rho_{\uparrow}(-\epsilon)$", "BrBG"),
        (axes[2], np.log(up / down), r"(c) $\ln \rho_\uparrow/\rho_\downarrow$ at $+\epsilon$", "RdBu"),
    )
    for ax, v, title, cmap in panels:
        lim = np.percentile(np.abs(v), 98)
        cs = ax.tricontourf(x, y, np.clip(v, -lim, lim), levels=np.linspace(-lim, lim, 25), cmap=cmap)
        if imp is not None:
            for spin, colour in (("up", "blue"), ("down", "red")):
                s = [r for r in imp if r["spin"] == spin]
                ax.scatter(column(s, "x"), column(s, "y"), s=6, facecolors=colour, edgecolors="k", linewidths=0.3)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title + r", $\epsilon = 0.1$, $J_I = 1$", fontsize=10)
        fig.colorbar(cs, ax=ax, shrink=0.8)
    save(fig, directory, "fig8_ldos")


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    fig3(directory)
    fig4(directory)
    fig6(directory)
    fig7(directory)
    fig8(directory)
