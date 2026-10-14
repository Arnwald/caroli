# Reproducing the graphene study on one machine

The example `graphene_field` recomputes the figures 3, 4, 6, 7 and 8 of A. Demion and A. D. Verga,
Phys. Rev. B 90, 085412 (2014). With `--paper` it uses every field and disorder strength of the
figures and statistics comparable to those of the paper. This page runs that campaign on a
workstation (here a Mac; Linux is the same without the first step).

## Setup

```bash
xcode-select --install                                # compilers (macOS)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
python3 -m pip install --user numpy matplotlib        # for the plots only
git clone https://github.com/Arnwald/caroli && cd caroli
```

## How long

```bash
cargo run --release -p caroli --example graphene_field -- estimate --paper
```

times the dynamics and the kernel polynomial method on this machine (about a minute) and prints
the duration of each figure. On four cores of a 2.8 GHz Xeon:

| figure | default | `--paper` |
|---|---|---|
| clean (Fig. 3) | 1 min | 3 min |
| dos (Fig. 4) | < 1 min | < 1 min |
| ldos (Fig. 8) | 3 min | 3.0 h |
| disorder (Fig. 7) | 6 min | 12.7 h |
| spreading (Fig. 6) | 11 min | 12.1 h |
| all | 20 min | 28 h |

The time is inversely proportional to the throughput of the machine: a computer with more and
faster cores divides it accordingly.

## Run

```bash
nohup cargo run --release -p caroli --example graphene_field -- all ~/graphene-paper --paper \
    > ~/graphene-paper.log 2>&1 &
tail -f ~/graphene-paper.log          # progress
cat ~/graphene-paper/campaign.log     # the figures done and their durations
```

`all` computes every figure, then draws `fig3_clean.png` … `fig8_ldos.png` next to the CSV tables.
On macOS it keeps the machine awake while it runs (`caffeinate`). If it stops (power cut,
restart), run the same command again: the finished figures, and the finished jobs of the long
ones (`parts/`), are kept. The directory remembers its options (`options.txt`) and refuses to mix
results computed with others.

## The statistics

The thesis averaged Fig. 7 over 8192 disorder configurations, each with one hole started at the
origin. The example traces over the whole band with random-phase vectors instead, each of which
averages over every starting site of the torus. `noise` compares the two (`noise_summary.csv`):
one vector on a 128 × 128 torus is worth 12 to 140 thesis configurations for the current (about
40 typically) and 7 to 3000 for the pairs, and in proportion to the area on a larger torus.
`--paper` (a 256 × 256 torus, 64 vectors) is thus worth 3000 to 36000 configurations.

Redoing the thesis protocol itself, 8192 configurations for each of the 40 cases of Fig. 7, would
take 830 h on the four cores above (`estimate` prints it for the machine at hand), Fig. 6 not
included. The thesis's grid carried, besides the sheet of 32768 cells it simulated, a second,
empty copy (the two parity classes of its N × N grid): twice that work.

Lighter or heavier runs: `--vectors R` (the error of the current goes as 1/√R), `--n N` (the torus,
the error as 1/N), `--configurations C` (per point of Fig. 6), `--ldos-sites S`,
`--ldos-configurations C` (the histogram of Fig. 8). The list is in the example's documentation.
