"""Renormalised masses of the g2g1 systems: exact (disorder-averaged Green's function at k = 0)
against the Born program as published and corrected, with the mean field for reference.

Needs the CSV files of
    cargo run --release -p caroli-validation --example born_self_energy

    python3 crates/caroli-validation/examples/plot_born_masses.py [csv directory] [tag] [note]

`tag` selects the files of a run with other settings (e.g. `_L256_k256`), and `note` is printed
under the legend (the sizes of the run).
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "..", "target", "born_lines")
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
NOTE = sys.argv[3] if len(sys.argv) > 3 else ""
MASS = -0.137
# (case index, title, conc, mean-field mass shift of spin up per unit J (opposite-g only))
CASES = [
    (0, r"'F' (same-g), $n_I=0.2$, $M_z=1$, $E_F=0$", 0.2, 0.0),
    (1, r"'AF' (opposite-g), $n_I=0.4$, $M_z=1$, $E_F=0.02$", 0.4, 1.0),
]
SPINS = [("M_up", r"$\bar m_\uparrow$", "tab:blue"), ("M_dn", r"$\bar m_\downarrow$", "tab:red")]


def rows(name):
    with open(os.path.join(DIR, name)) as f:
        return list(csv.DictReader(f))


def main():
    exact, born = rows(f"self_energy{TAG}.csv"), rows(f"born_masses{TAG}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (case, title, conc, opposite) in zip(axes, CASES):
        for key, label, colour in SPINS:
            for version, style in (("published", "--"), ("corrected", "-")):
                pts = [r for r in born if int(r["case"]) == case and r["version"] == version]
                j = np.array([float(r["J_I"]) for r in pts])
                ax.plot(j, [float(r[key]) for r in pts], style, color=colour, lw=1.4,
                        label=f"{label} Born, {version}")
            sign = 1.0 if key == "M_up" else -1.0
            ax.plot(j, MASS + sign * opposite * conc * j, ":", color=colour, lw=1.0,
                    label=f"{label} mean field")
            samples = defaultdict(list)
            for r in exact:
                if int(r["case"]) == case:
                    samples[float(r["J_I"])].append(float(r[key]))
            js = sorted(samples)
            ax.plot(js, [np.mean(samples[x]) for x in js], "o", color=colour, ms=6, mfc="white", mew=1.6,
                    label=f"{label} exact")
        ax.axhline(0.0, color="black", lw=0.8)
        ax.text(0.01, 0.003, "trivial", transform=ax.get_yaxis_transform(), fontsize=9, va="bottom")
        ax.text(0.01, -0.003, "topological", transform=ax.get_yaxis_transform(), fontsize=9, va="top")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"disorder $J_I$ $(\varepsilon_0)$")
        ax.set_xlim(0.0, None)
    axes[0].set_ylabel(r"renormalised mass $\bar m$ $(\varepsilon_0)$")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
    if NOTE:
        fig.suptitle(NOTE, fontsize=9, style="italic", y=0.995)
    fig.subplots_adjust(bottom=0.27, top=0.86 if NOTE else 0.92, wspace=0.18)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(DIR, f"born_masses{TAG}.{ext}"), dpi=150, bbox_inches="tight")
    print("written", os.path.abspath(os.path.join(DIR, f"born_masses{TAG}.png")))


if __name__ == "__main__":
    main()
