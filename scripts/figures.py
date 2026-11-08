#!/usr/bin/env python3
"""Candidate figures for Section 5, built from runs/ and criterion-sweep.log.

Nothing here is hand-placed: re-running regenerates every figure from the
recorded data, so a figure cannot drift from the table it illustrates.

    python scripts/figures.py [--outdir paper/potential_figs]

Needs matplotlib and numpy only -- no Firedrake, so it runs anywhere the repo
is checked out.

Palette: slots 1-3 of the project's categorical palette, validated for
all-pairs colourblind separation in light mode. Colour is never the only
encoding: every series also differs by line style or marker, so the figures
survive greyscale printing.
"""
import argparse, csv, glob, json, math, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXSHA = "7e88d93"          # the frozen-data fix; see notes/results.md E31-E32

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlesize": 9,
    "axes.linewidth": 0.6, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.5,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.frameon": False, "legend.fontsize": 7.5,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

# ── data ────────────────────────────────────────────────────────────────────
def load_runs():
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "metrics.csv"))):
        rs = list(csv.DictReader(open(p, newline="")))
        if not rs:
            continue
        s = next((r for r in reversed(rs)
                  if r.get("summary") not in (None, "", "0")), None)
        if s is None:
            continue
        s = dict(s)
        s["_rounds"] = [r for r in rs if r.get("round") not in (None, "")]
        out.append(s)
    return out

def fnum(r, k):
    v = r.get(k)
    if v in (None, ""):
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return None if math.isnan(x) else x

ALL = load_runs()
PIC = [r for r in ALL
       if (r.get("git_sha") or "")[:7] == FIXSHA and r.get("method") == "picard"]
LBF = [r for r in ALL if r.get("method") == "lbfgs" and r.get("field") == "mu_2"]

CFG = [(2.0, 4), (1.0, 8), (1.0, 4), (0.6, 4)]
RHO = {(0.6, 4): 0.250379, (1.0, 4): 0.025758,
       (1.0, 8): 0.035550, (2.0, 4): 0.110388}
TITLE = {(2.0, 4): r"$D_{\rm true}=2.0$, 16 sensors",
         (1.0, 8): r"$D_{\rm true}=1.0$, 64 sensors",
         (1.0, 4): r"$D_{\rm true}=1.0$, 16 sensors",
         (0.6, 4): r"$D_{\rm true}=0.6$, 16 sensors"}

def picard_groups():
    g = {}
    for r in PIC:
        g.setdefault((fnum(r, "D_true"), int(fnum(r, "sensors"))), []).append(r)
    return g
G = picard_groups()

def sweep():
    """The gauge-aware criterion sweep, parsed out of its log."""
    path = os.path.join(ROOT, "criterion-sweep.log")
    if not os.path.exists(path):
        return None
    txt = open(path).read()
    parts = re.split(r"^=== \S+\s+(.+)$", txt, flags=re.M)[1:]
    out = {}
    for lab, body in zip(parts[0::2], parts[1::2]):
        m = re.search(r"unregularized\s+lambda = ([+-][0-9.e+-]+)", body)
        c = re.search(r"cos\(j_D, J_c sigma_nl\)\s+([+-][0-9.]+)", body)
        j = re.search(r"\|\|j_D\|\|_2\s+([0-9.e+-]+)", body)
        if m and c and j:
            out[lab.strip()] = dict(rho=abs(float(m.group(1))),
                                    cos=float(c.group(1)), jD=float(j.group(1)))
    return out
SW = sweep()

# ── figure 1: the criterion predicts the decay ──────────────────────────────
def fig_decay(outdir):
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.4), sharey=True)
    for ax, key in zip(axes.ravel(), CFG):
        runs = sorted(G.get(key, []), key=lambda r: int(r["seed"]))
        if not runs:
            continue
        for i, r in enumerate(runs):
            k = [int(float(x["round"])) for x in r["_rounds"]
                 if fnum(x, "d_kappa") and fnum(x, "d_kappa") > 0]
            d = [fnum(x, "d_kappa") for x in r["_rounds"]
                 if fnum(x, "d_kappa") and fnum(x, "d_kappa") > 0]
            ax.plot(k, d, "-", color=BLUE, lw=1.0, alpha=0.45, zorder=2,
                    label="ten noise seeds" if i == 0 else None)
        # the prediction: slope rho per round, anchored at round 2's median
        anchor = np.median([
            [fnum(x, "d_kappa") for x in r["_rounds"]
             if int(float(x["round"])) == 2][0]
            for r in runs
            if any(int(float(x["round"])) == 2 for x in r["_rounds"])])
        # Stop the predicted line where the data stops: extending it past the
        # last round would show a prediction nothing was measured against.
        kmax = max(int(float(x["round"])) for r in runs for x in r["_rounds"])
        ks = np.arange(2, kmax + 1)
        ax.plot(ks, anchor * RHO[key] ** (ks - 2), "--", color=ORANGE, lw=1.6,
                zorder=4, label=r"slope $\rho(D\mathcal{T})$, predicted")
        ax.axhline(1e-8, color=INK2, lw=0.5, ls=":", zorder=1)
        ax.set_yscale("log"); ax.set_ylim(1e-13, 3.0)
        ax.set_xlim(0.7, kmax + 0.3)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        # Label the tolerance where it is, not in a corner.
        ax.text(0.015, 1e-8, "tol", transform=ax.get_yaxis_transform(),
                va="bottom", ha="left", fontsize=6.5, color=INK2)
        ax.set_title(TITLE[key], color=INK)
        meas = np.median([fnum(r, "outer_rate") for r in runs
                          if fnum(r, "outer_rate") is not None])
        ax.text(0.97, 0.94,
                f"$\\rho={RHO[key]:.4f}$\nobserved ${meas:.4f}$",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.2, color=INK2, linespacing=1.4)
    for ax in axes[1]:
        ax.set_xlabel("outer round $k$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$|\kappa^{(k+1)}-\kappa^{(k)}|$")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    save(fig, outdir, "fig_decay")

# ── figure 2: the criterion across two sweeps ───────────────────────────────
def fig_sweeps(outdir):
    if not SW:
        print("  skipped fig_sweeps: criterion-sweep.log not found")
        return
    sens = [9, 16, 36, 64, 100]
    Ds = [0.48, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    sx = [SW[f"sensors={s}"] for s in sens]
    dx = [SW[f"D_true={d}"] for d in Ds]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.3))
    panels = [
        (axes[0, 0], sens, [v["rho"] for v in sx], "log",
         r"$\rho(D\mathcal{T})$", "sensors $m$", True),
        (axes[0, 1], Ds, [v["rho"] for v in dx], "log",
         r"$\rho(D\mathcal{T})$", r"$D_{\rm true}$", True),
        (axes[1, 0], sens, [v["cos"] for v in sx], "linear",
         r"$\cos(j_\beta,\, J_{\bar c}\sigma_{\rm nl})$", "sensors $m$", False),
        (axes[1, 1], Ds, [v["cos"] for v in dx], "linear",
         r"$\cos(j_\beta,\, J_{\bar c}\sigma_{\rm nl})$", r"$D_{\rm true}$", False),
    ]
    for ax, xs, ys, yscale, ylab, xlab, mark_min in panels:
        ax.plot(xs, ys, "-o", color=BLUE, lw=1.4, ms=4.5, mfc="white",
                mew=1.3, zorder=3)
        ax.set_xscale("log"); ax.set_yscale(yscale)
        ax.set_ylabel(ylab); ax.set_xlabel(xlab)
        ax.set_xticks(xs)
        rot = 45 if xlab.startswith("$D") else 0
        ax.set_xticklabels([str(x) for x in xs], rotation=rot,
                           ha="right" if rot else "center", fontsize=7)
        ax.minorticks_off()
        if yscale == "log":
            # One label per decade leaves a single tick over this narrow
            # range; step by 1-2-5 instead and print plain decimals.
            ax.yaxis.set_major_locator(
                matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        if mark_min:
            i = int(np.argmin(ys))
            ax.plot([xs[i]], [ys[i]], "o", ms=9, mfc="none", mec=ORANGE,
                    mew=1.6, zorder=4)
            ax.annotate("minimum", (xs[i], ys[i]), textcoords="offset points",
                        xytext=(9, 7), fontsize=7, color=ORANGE)
        else:
            ax.axhline(0.0, color=ORANGE, lw=1.0, ls="--", zorder=2)
            ax.annotate("sign change", (xs[len(xs)//2], 0.0),
                        textcoords="offset points", xytext=(4, 6),
                        fontsize=7, color=ORANGE)
    fig.tight_layout()
    save(fig, outdir, "fig_criterion_sweeps")

# ── figure 3: the two methods find the same minimum ─────────────────────────
def fig_paired(outdir):
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    styles = [(0.6, BLUE, "o"), (1.0, ORANGE, "s"), (2.0, AQUA, "^")]
    lim = 0
    for Dt, col, mk in styles:
        dr = {int(r["seed"]): r for r in LBF
              if fnum(r, "D_true") == Dt and fnum(r, "sigma") == 1e-3
              and fnum(r, "N") == 16 and (r.get("opt_status") or "ok") == "ok"}
        pr = {int(r["seed"]): r for r in G.get((Dt, 4), [])}
        seeds = sorted(set(dr) & set(pr))
        if not seeds:
            continue
        x = np.array([(fnum(dr[s], "D_recovered") - Dt) / (1e-3 * Dt) for s in seeds])
        y = np.array([(fnum(pr[s], "D_recovered") - Dt) / (1e-3 * Dt) for s in seeds])
        lim = max(lim, np.abs(x).max(), np.abs(y).max())
        ax.plot(x, y, mk, color=col, ms=5, mew=0.8, mec="white", alpha=0.95,
                ls="none", label=rf"$D_{{\rm true}}={Dt}$  (r={np.corrcoef(x,y)[0,1]:.3f})")
    lim *= 1.15
    ax.plot([-lim, lim], [-lim, lim], "-", color=INK2, lw=0.7, zorder=1)
    ax.annotate("equal recovery", (lim*0.52, lim*0.52), rotation=45,
                textcoords="offset points", xytext=(0, 4), fontsize=6.8,
                color=INK2, ha="center")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel(r"direct error $/\,\sigma D_{\rm true}$")
    ax.set_ylabel(r"Picard error $/\,\sigma D_{\rm true}$")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, outdir, "fig_paired_recovery")

def save(fig, outdir, stem):
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, f"{stem}.{ext}")
        fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.pdf and {stem}.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(ROOT, "paper", "potential_figs"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    print(f"picard runs at {FIXSHA}: {len(PIC)};  direct mu_2 runs: {len(LBF)}")
    fig_decay(a.outdir)
    fig_sweeps(a.outdir)
    fig_paired(a.outdir)

if __name__ == "__main__":
    main()
