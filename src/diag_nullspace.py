"""
Diagnostic: is the residual invariant under the candidate gauge shifts?

Cheap, decisive, and needs no assembled matrix -- which matters, because the
mixed space contains "R" blocks and cannot be assembled monolithically anyway.

The question. Our mixed space carries three real-space constants l_1, l_2, l_p
alongside mu_1, mu_2, p. If the residual is invariant under

    (mu_1, l_1) -> (mu_1 + c, l_1 - c)        and likewise for mu_2, p

then the Jacobian has a nullspace of that dimension, the formulation is
singular, and Newton cannot converge to an isolated solution no matter what
the solver does.

The test. Evaluate the residual vector at the initial guess, apply each shift,
evaluate again, and compare. An unchanged residual is an exact symmetry --
there is nothing probabilistic about it.

    ||F(U + delta)|| - ||F(U)||  ==  0   ->  SINGULAR in that direction
                                !=  0   ->  symmetry broken, direction is fine

Reference: the electrolyte code breaks the pressure symmetry by using raw `p`
in the Stokes operator and `p + l_p` only in the constitutive matrix. That works
there because its flux test functions have nonzero normal trace on part of the
boundary. Our manufactured solution prescribes Dirichlet data on the whole
boundary, so int div(u_i) = 0 identically and the same trick may buy nothing.
This script settles it.

Usage:
    python src/diag_nullspace.py            # k=2 on a 2x2 mesh, a few seconds
"""

import argparse

import numpy as np

from firedrake import *
from firedrake.petsc import PETSc

from sosm import SOSMProblem

# (label, index of the raw field, index of its real-space constant)
SHIFTS = [
    ("mu_1 / l_1", 3, 9),
    ("mu_2 / l_2", 4, 10),
    ("p    / l_p", 5, 11),
]


def residual_norm(problem, sln):
    F = problem.residual(sln, problem.r_1_d, problem.r_2_d)
    vec = assemble(F, form_compiler_parameters={"quadrature_degree": problem.deg_max})
    with vec.dat.vec_ro as v:
        return v.norm()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=2, choices=(2, 3))
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--N", type=int, default=2, help="cells per direction; keep tiny")
    ap.add_argument("--c", type=float, default=0.25, help="shift magnitude")
    args = ap.parse_args()

    problem = SOSMProblem(d=args.d, k=args.k, N_mesh=args.N)
    base = problem.initial_guess()

    ref = residual_norm(problem, base)
    PETSc.Sys.Print(f"\nmixed space has {problem.Z.num_sub_spaces()} fields, "
                    f"{problem.Z.dim()} dofs")
    PETSc.Sys.Print(f"||F(U0)|| = {ref:.12e}\n")

    singular = []
    for label, i_field, i_const in SHIFTS:
        shifted = base.copy(deepcopy=True)
        subs = shifted.subfunctions

        # Add c to the raw field, subtract c from its real-space constant, so
        # the augmented combination (field + constant) is unchanged.
        subs[i_field].assign(subs[i_field] + Constant(args.c))
        subs[i_const].assign(subs[i_const] - Constant(args.c))

        val = residual_norm(problem, shifted)
        rel = abs(val - ref) / max(ref, 1e-300)

        verdict = "SINGULAR -- exact symmetry" if rel < 1e-10 else "ok, symmetry broken"
        if rel < 1e-10:
            singular.append(label)

        PETSc.Sys.Print(f"  {label}:  ||F|| = {val:.12e}   rel_change = {rel:.3e}"
                        f"   {verdict}")

    PETSc.Sys.Print("")
    if singular:
        PETSc.Sys.Print(f"RESULT: {len(singular)} singular direction(s): "
                        f"{', '.join(singular)}")
        PETSc.Sys.Print("The formulation is rank-deficient. The multipliers must be")
        PETSc.Sys.Print("attached differently -- see notes/context.md section 9.")
    else:
        PETSc.Sys.Print("RESULT: no exact symmetry found in the three shift directions.")
        PETSc.Sys.Print("Necessary, not sufficient: Newton can still fail for other")
        PETSc.Sys.Print("reasons, but the constants are at least determined.")


if __name__ == "__main__":
    main()
