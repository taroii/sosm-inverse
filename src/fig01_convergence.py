"""
Figure 1 -- VALIDATION GATE.

Mesh convergence of our ported SOSM forward solver against the manufactured
solution of Baier-Reinio & Farrell section 5.1.

This is the regression test for the port in src/sosm.py. Our port changes one
thing relative to the original: point BCs + the Woodbury identity are replaced
by real-space constants plus three integral constraints (see the module
docstring in src/sosm.py). If that reformulation is correct, the observed
convergence rates here must match the original's. If it is wrong, the rates
will degrade or the solve will diverge -- which is exactly what this script is
for. Nothing downstream should be trusted until this passes.

Ground truth is available: run the original for the same configuration and
compare tables directly.

    python multicomponent_code/manufactured_solution.py 2 4 tet False 8 4

Expected: rate ~= k for the flux, velocity and potential errors.

Usage:
    python src/fig01_convergence.py --d 2 --k 4 --N0 8 --loops 4
"""

import argparse

import matplotlib
matplotlib.use("Agg")          # headless server -- no display
import matplotlib.pyplot as plt
import numpy as np

from firedrake.petsc import PETSc
from sosm import SOSMProblem, solve_forward
from runlog import Run

# Fields to plot. The rest are still recorded in metrics.csv.
PLOT_FIELDS = ["mm_1", "mm_2", "v", "mu_1", "p", "x_1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=2, choices=(2, 3))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--mesh-type", default="tet", choices=("tet", "hex"))
    ap.add_argument("--N0", type=int, default=8, help="initial cells per direction")
    ap.add_argument("--loops", type=int, default=4, help="number of refinements")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    config = vars(args).copy()
    config.pop("allow_dirty")

    with Run("convergence", config, allow_dirty=args.allow_dirty) as run:
        history = []

        for i in range(args.loops):
            N = args.N0 * (2 ** i)
            PETSc.Sys.Print(f"\n=== refinement {i}, N_mesh = {N} ===", flush=True)

            problem = SOSMProblem(d=args.d, k=args.k,
                                  mesh_type=args.mesh_type, N_mesh=N)
            sln = solve_forward(problem)

            errs = problem.errors(sln)
            ndofs = problem.Z.dim()

            row = {"refinement": i, "N_mesh": N, "h": 1.0 / N, "ndofs": ndofs}
            row.update({k: float(v) for k, v in errs.items()})
            run.record(**row)
            history.append(row)

            PETSc.Sys.Print(f"    ndofs = {ndofs}", flush=True)
            for name in PLOT_FIELDS:
                PETSc.Sys.Print(f"    {name:>10s} err = {row[name]:.6e}", flush=True)

        # -- Observed rates. ------------------------------------------------
        PETSc.Sys.Print("\n=== observed convergence rates ===", flush=True)
        for name in PLOT_FIELDS:
            e = np.array([r[name] for r in history])
            h = np.array([r["h"] for r in history])
            rates = np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:])
            pretty = ", ".join(f"{r:.2f}" for r in rates)
            PETSc.Sys.Print(f"  {name:>10s}: {pretty}", flush=True)

        # -- Plot. ----------------------------------------------------------
        h = np.array([r["h"] for r in history])
        fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
        for name in PLOT_FIELDS:
            e = np.array([r[name] for r in history])
            ax.loglog(h, e, "o-", label=name)

        ref = h ** args.k
        ax.loglog(h, ref * (history[0][PLOT_FIELDS[0]] / ref[0]),
                  "k--", lw=1, label=f"$h^{{{args.k}}}$")

        ax.set_xlabel("$h$")
        ax.set_ylabel(r"$L^2$ error")
        ax.set_title(f"SOSM manufactured solution, $d={args.d}$, $k={args.k}$")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, which="both", alpha=0.3)

        out = run.dir / "fig01_convergence.pdf"
        fig.savefig(out)
        PETSc.Sys.Print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
