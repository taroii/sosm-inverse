"""
Figure 2 -- adjoint gradient verification against centered finite differences.

Tests the second unknown in the port: whether `firedrake.adjoint` can tape the
SOSM residual at all. The original implementation cannot be taped, because it
enforces its integral constraints inside a custom SNES convergence-test
callback doing raw PETSc vector surgery. Our reformulation puts those
constraints in the variational form precisely so that this script can work.

README.md section IV requires the adjoint gradient to be verified against
centered differences "at every new configuration, as opposed to once at the
outset". This script is that check, and it is meant to be re-run whenever the
forward model, the observation operator, or the discretization changes.

The expected signature of a correct gradient is a V in the log-log plot: the
relative error falls as O(eps^2) from truncation, then rises as eps drops
further and round-off takes over. A flat or monotone curve means the gradient
is wrong.

Run fig01_convergence.py first. A gradient check on a broken forward solve
tells you nothing.

Usage:
    python src/fig02_gradient_check.py --N 8 --k 3
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from firedrake import *
from firedrake.adjoint import *
from firedrake.petsc import PETSc

from sosm import SOSMProblem, solve_forward
from runlog import Run


def misfit(problem, sln, target):
    """Full-field L^2 misfit in the mole fraction of species 1.

    Deliberately the simplest tapeable functional, because this script is
    testing the gradient machinery and not the observation model. The actual
    inverse problem uses sparse point observations; those are taped via
    `VertexOnlyMesh` interpolation, which is the supported route for
    differentiable point evaluation. `Function.at()` is NOT tapeable and must
    not be used inside the objective.
    """
    x_1 = split(sln)[6]
    return assemble(0.5 * inner(x_1 - target, x_1 - target) * problem.dx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=2, choices=(2, 3))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--D-true", type=float, default=1.0,
                    help="D_12 used to generate the target field")
    ap.add_argument("--D-eval", type=float, default=1.4,
                    help="D_12 at which the gradient is checked")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    config = vars(args).copy()
    config.pop("allow_dirty")

    with Run("gradient-check", config, allow_dirty=args.allow_dirty) as run:

        # -- Everything that must NOT be taped. -----------------------------
        # ONE problem, used for both solves. Two SOSMProblem instances would
        # build two separate UnitSquareMesh objects, and a form mixing a
        # coefficient from one with an argument from the other raises
        #     MismatchingDomainError: Argument or Coefficient domain not found
        # even though the meshes are numerically identical.
        #
        # Annotation stays off here because SOSMProblem.__init__ calls
        # _fix_compatibility, which projects and then assigns to T_1/T_2 --
        # Constants that the flux Dirichlet BCs close over, so taping them would
        # mutate boundary data underneath a recorded solve.
        pause_annotation()

        problem = SOSMProblem(d=args.d, k=args.k, N_mesh=args.N)

        sln_true = solve_forward(problem, D_12=args.D_true)
        target = Function(problem.spaces["X_1"])
        target.assign(sln_true.subfunctions[6])

        problem.D_12.assign(args.D_eval)

        # Build the initial guess ONCE, here, outside the tape. Left to
        # solve_forward it would be rebuilt inside the annotated region, and
        # initial_guess() performs nine L^2 projections, each an LU solve with
        # MUMPS. Those would then be replayed on every functional evaluation --
        # roughly 30 of them below -- for a result that never changes, since the
        # guess depends only on the manufactured solution and not on D_12.
        guess = problem.initial_guess()

        continue_annotation()

        # -- Taped from here: only the solve and the functional. ------------
        # sln is reset from `guess` inside the tape, so every replay starts from
        # the same state rather than warm-starting from the previous solve. The
        # assign is annotated and cheap; the projections it replaces are not.
        control = Control(problem.D_12)

        sln = Function(problem.Z)
        sln.assign(guess)

        # check=False: check_constraints performs three assembles, and inside the
        # annotated region those are taped and replayed too. The constraints are
        # verified after the tape is closed instead.
        sln = solve_forward(problem, sln=sln, check=False)
        J = misfit(problem, sln, target)

        # derivative() returns a Cofunction in the dual of the control space,
        # not a float; for an "R" control that is a single dof.
        Jhat = ReducedFunctional(J, control)
        g_adj = float(Jhat.derivative().dat.data_ro[0])
        J0 = float(J)

        PETSc.Sys.Print(f"\nJ(D_eval)      = {J0:.8e}", flush=True)
        PETSc.Sys.Print(f"adjoint dJ/dD  = {g_adj:.8e}\n", flush=True)

        # -- Centered finite differences, tape off. -------------------------
        # The control is an "R"-space Function, so Jhat must be re-evaluated at
        # an "R"-space Function too -- passing a Constant does not match the
        # control type.
        def at(value):
            f = Function(problem.R0)
            f.assign(value)
            return f

        pause_annotation()

        # Constraint residuals and multipliers at the taped solution, now that
        # annotation is off and these assembles will not be recorded.
        for name, val in problem.check_constraints(sln).items():
            PETSc.Sys.Print(f"  {name:>10s} = {val:+.6e}", flush=True)
        PETSc.Sys.Print("", flush=True)

        steps = np.logspace(-1, -8, 15)
        rows = []
        for eps in steps:
            jp = float(Jhat(at(args.D_eval + eps)))
            jm = float(Jhat(at(args.D_eval - eps)))
            g_fd = (jp - jm) / (2.0 * eps)
            rel = abs(g_fd - g_adj) / max(abs(g_adj), 1e-300)
            rows.append({"eps": eps, "g_fd": g_fd, "g_adj": g_adj, "rel_err": rel})
            run.record(**rows[-1])
            PETSc.Sys.Print(f"  eps={eps:.2e}  fd={g_fd:.8e}  rel_err={rel:.3e}",
                            flush=True)
        continue_annotation()

        best = min(rows, key=lambda r: r["rel_err"])
        PETSc.Sys.Print(f"\nbest relative error {best['rel_err']:.3e} "
                        f"at eps={best['eps']:.2e}", flush=True)

        # -- Also run Firedrake's own Taylor test, which is the stricter check.
        # The FD loop above left the control at the last perturbed value; reset
        # it so the Taylor test expands about D_eval, not D_eval - 1e-8.
        pause_annotation()
        h = Function(problem.R0)
        h.assign(1.0)
        rate = taylor_test(Jhat, at(args.D_eval), h)
        PETSc.Sys.Print(f"taylor_test convergence rate: {rate:.3f}", flush=True)
        continue_annotation()

        # -- Plot. ----------------------------------------------------------
        eps = np.array([r["eps"] for r in rows])
        rel = np.array([r["rel_err"] for r in rows])

        fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
        ax.loglog(eps, rel, "o-", label="centered FD vs adjoint")
        ax.loglog(eps, eps ** 2 * (rel[0] / eps[0] ** 2), "k--", lw=1,
                  label=r"$\varepsilon^2$")
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel("relative error in $dJ/dD_{12}$")
        ax.set_title(f"Adjoint gradient verification, $d={args.d}$, "
                     f"$k={args.k}$, $N={args.N}$")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

        out = run.dir / "fig02_gradient_check.pdf"
        fig.savefig(out)
        PETSc.Sys.Print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
