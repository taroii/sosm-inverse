"""
Diagnostic: can pyadjoint differentiate through a matfree/fieldsplit solve on a
mixed space containing an "R" block?

This is the central computational premise of the whole project, and it does not
depend on SOSM at all. Our approach requires four things to work together:

    1. an "R"-space Lagrange multiplier inside a mixed space
    2. mat_type "matfree" (forced -- R blocks cannot be assembled monolithically)
    3. a Schur fieldsplit with the PDE block assembled into MUMPS
    4. firedrake.adjoint taping the whole thing, with an R-space Control

If that combination does not work, no amount of fixing the SOSM gauge question
saves the approach. Better to find out on a thirty-line Poisson problem than
after the SOSM formulation is settled.

The test problem is the minimal analogue of our structure: Neumann Poisson,

    -D laplacian(u) = f   in Omega,    du/dn = 0,    int u = 0

whose constant nullspace is removed by an R-space multiplier in exactly the
saddle-point form proposed as the fix for the SOSM gauge problem
(see open_problems.md item 1):

    res += lam * v * dx      # multiplier in u's own equation
    res += u * t * dx        # the constraint

So a pass here is evidence for two separate things: that the toolchain supports
what we need, and that this particular way of attaching multipliers is sound.

We invert for the scalar diffusivity D, which is the same shape of inverse
problem as recovering D_12.

Usage:
    python src/diag_adjoint_rspace.py
"""

import argparse

import numpy as np

from firedrake import *
from firedrake.adjoint import *
from firedrake.petsc import PETSc


def build(N, degree=2):
    mesh = UnitSquareMesh(N, N)
    V = FunctionSpace(mesh, "CG", degree)
    R = FunctionSpace(mesh, "R", 0)
    Z = V * R

    x = SpatialCoordinate(mesh)
    # Zero mean, so the pure-Neumann problem is solvable.
    f = cos(pi * x[0]) * cos(pi * x[1])
    return mesh, Z, R, f


def solver_parameters():
    """Same structure as SOSMProblem.solver_parameters: matfree + Schur."""
    return {"snes_type": "newtonls",
            "snes_linesearch_type": "basic",
            "snes_atol": 1e-12,
            "snes_rtol": 0.0,
            "snes_max_it": 10,
            "mat_type": "matfree",
            "ksp_type": "fgmres",
            "ksp_atol": 1e-13,
            "ksp_rtol": 1e-13,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "full",
            "pc_fieldsplit_0_fields": "0",
            "pc_fieldsplit_1_fields": "1",
            "fieldsplit_0": {
                "ksp_type": "preonly",
                "pc_type": "python",
                "pc_python_type": "firedrake.AssembledPC",
                "assembled": {
                    "ksp_type": "gmres",
                    "ksp_max_it": 3,
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",
                },
            },
            "fieldsplit_1": {
                "ksp_type": "gmres",
                "ksp_max_it": 1,
            }}


def solve_poisson(Z, R, f, D):
    z = Function(Z)
    u, lam = split(z)
    v, t = TestFunctions(Z)

    res = D * inner(grad(u), grad(v)) * dx - f * v * dx
    res += lam * v * dx          # multiplier enters u's own equation
    res += u * t * dx            # the constraint, int u = 0

    problem = NonlinearVariationalProblem(res, z)
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters())
    solver.solve()
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--degree", type=int, default=2)
    ap.add_argument("--D-true", type=float, default=1.0)
    ap.add_argument("--D-eval", type=float, default=1.7)
    args = ap.parse_args()

    mesh, Z, R, f = build(args.N, args.degree)

    def at(value):
        g = Function(R)
        g.assign(value)
        return g

    # -- Target, generated at the true D, tape off. --------------------------
    pause_annotation()
    z_true = solve_poisson(Z, R, f, at(args.D_true))
    target = Function(Z.sub(0))
    target.assign(z_true.subfunctions[0])
    continue_annotation()

    # -- Taped solve at D_eval. ----------------------------------------------
    D = at(args.D_eval)
    control = Control(D)

    z = solve_poisson(Z, R, f, D)
    u = split(z)[0]
    J = assemble(0.5 * inner(u - target, u - target) * dx)

    Jhat = ReducedFunctional(J, control)
    g_adj = float(Jhat.derivative())

    PETSc.Sys.Print(f"\nJ(D_eval)     = {float(J):.8e}")
    PETSc.Sys.Print(f"adjoint dJ/dD = {g_adj:.8e}")

    if abs(g_adj) < 1e-14:
        PETSc.Sys.Print("\nFAIL: gradient is zero. The tape is empty or the "
                        "control is not connected to the residual.")
        return

    # -- Centered differences. -----------------------------------------------
    pause_annotation()
    PETSc.Sys.Print("")
    best = np.inf
    for eps in np.logspace(-2, -8, 13):
        jp = float(Jhat(at(args.D_eval + eps)))
        jm = float(Jhat(at(args.D_eval - eps)))
        g_fd = (jp - jm) / (2.0 * eps)
        rel = abs(g_fd - g_adj) / abs(g_adj)
        best = min(best, rel)
        PETSc.Sys.Print(f"  eps={eps:.1e}  fd={g_fd:.8e}  rel_err={rel:.3e}")

    rate = taylor_test(Jhat, at(args.D_eval), at(1.0))
    continue_annotation()

    PETSc.Sys.Print(f"\nbest FD relative error : {best:.3e}   (want < 1e-6)")
    PETSc.Sys.Print(f"taylor_test rate       : {rate:.3f}   (want ~2.0)")

    ok = best < 1e-6 and rate > 1.9
    PETSc.Sys.Print("\nPASS: matfree + fieldsplit + R-space + pyadjoint works together."
                    if ok else
                    "\nFAIL: see numbers above. This blocks the whole approach, "
                    "independently of SOSM.")


if __name__ == "__main__":
    main()
