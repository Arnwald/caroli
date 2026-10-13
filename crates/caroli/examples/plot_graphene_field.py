"""Figures of the graphene study (`graphene_field.rs`), from the CSV files it writes.

    cargo run --release -p caroli --example graphene_field -- all OUT_DIR
    python3 crates/caroli/examples/plot_graphene_field.py OUT_DIR

- `current.png`: current density (minus that of the same random vectors without field) and
  density of pairs created since t = 0 against time, per field, for clean, paramagnetic and
  antiferromagnetic graphene.
- `spreading.png`: width of a wave packet against time (log-log), with the ballistic slope.
- `dos.png`: densities of states per unit cell.
- `conductivity.png`: σ_xx(μ) in units of e²/h, with the minimal conductivity 4/π of clean
  graphene (spin included).

Needs numpy and matplotlib.
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PHASES = {"clean": ("clean", "k"), "para": ("paramagnetic", "tab:blue"), "af": ("antiferromagnetic", "tab:red")}


def read(directory, name):
    path = os.path.join(directory, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def column(rows, key):
    return np.array([float(r[key]) for r in rows])


def save(fig, directory, name):
    path = os.path.join(directory, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("written", os.path.abspath(path))


def current(directory):
    rows = read(directory, "current.csv")
    if rows is None:
        return
    # F = 0: the same random vectors without field, whose current is subtracted from the others
    # (it removes the noise of the stochastic trace at early times, before the evolutions
    # decorrelate).
    fields = sorted({float(r["field"]) for r in rows} - {0.0})
    fig, axes = plt.subplots(2, len(fields), figsize=(4.5 * len(fields), 7), sharex=True, layout="constrained")
    axes = np.array(axes).reshape(2, len(fields))
    for c, f in enumerate(fields):
        for phase, (label, colour) in PHASES.items():
            sub = [r for r in rows if r["phase"] == phase and float(r["field"]) == f]
            if not sub:
                continue
            t, j = column(sub, "t"), column(sub, "j")
            reference = [r for r in rows if r["phase"] == phase and float(r["field"]) == 0.0]
            if reference:
                j = j - column(reference, "j")
            axes[0, c].plot(t, j, color=colour, label=label)
            pairs = column(sub, "pairs")
            axes[1, c].plot(t, pairs - pairs[0], color=colour, label=label)
        axes[0, c].set_title(f"$F = {f:g}$")
        axes[1, c].set_xlabel("$t$")
    axes[0, 0].set_ylabel("current density $j_y$")
    axes[1, 0].set_ylabel("pairs created per unit area")
    axes[0, 0].legend()
    save(fig, directory, "current")


def spreading(directory):
    rows = read(directory, "spreading.csv")
    if rows is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
    for phase, (label, colour) in PHASES.items():
        sub = [r for r in rows if r["phase"] == phase]
        t, w = column(sub, "t"), column(sub, "width")
        keep = t > 0
        ax.loglog(t[keep], w[keep], color=colour, label=label)
    t = column(rows, "t")
    t = t[t > 0]
    ax.loglog(t, 0.6 * t, ":", color="0.5", label=r"$\propto t$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("width $w(t)$")
    ax.legend()
    save(fig, directory, "spreading")


def dos(directory):
    rows = read(directory, "dos.csv")
    if rows is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
    for phase, (label, colour) in PHASES.items():
        sub = [r for r in rows if r["phase"] == phase]
        ax.plot(column(sub, "energy"), column(sub, "dos"), color=colour, label=label)
    ax.set_xlabel("$E$")
    ax.set_ylabel("density of states per cell")
    ax.legend()
    save(fig, directory, "dos")


def conductivity(directory):
    rows = read(directory, "conductivity.csv")
    if rows is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
    for phase, (label, colour) in PHASES.items():
        sub = [r for r in rows if r["phase"] == phase]
        ax.plot(column(sub, "mu"), column(sub, "sigma_xx"), color=colour, label=label)
    ax.axhline(4 / np.pi, ls=":", color="0.5", label=r"$4e^2/\pi h$")
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\sigma_{xx}$ ($e^2/h$)")
    ax.legend()
    save(fig, directory, "conductivity")


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    current(directory)
    spreading(directory)
    dos(directory)
    conductivity(directory)
