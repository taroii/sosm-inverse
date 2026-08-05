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


def residual_norms(problem, sln):
    """Return (norm without BCs, norm with BCs applied).

    The second is the one that matters, and getting this wrong gives a false
    pass. Assembling without BCs leaves rows for the constrained flux and
    velocity DOFs, and those rows DO respond to a shift in mu_i. But the solve
    discards them. For the test functions that actually survive -- the
    unconstrained RT basis functions, which have zero normal trace on the whole
    boundary -- we have

        int div(u_i) dx = oint u_i . n ds = 0

    so the shift contributes nothing there. A symmetry can therefore be invisible
    in the first number and present in the second.
    """
    F = problem.residual(sln, problem.r_1_d, problem.r_2_d)
    fcp = {"quadrature_degree": problem.deg_max}

    raw = assemble(F, form_compiler_parameters=fcp)
    with raw.dat.vec_ro as v:
        n_raw = v.norm()

    bc_applied = assemble(F, bcs=problem.bcs(), form_compiler_parameters=fcp)
    with bc_applied.dat.vec_ro as v:
        n_bc = v.norm()

    return n_raw, n_bc


def constant_row_test(problem, sln):
    """Test Lemma 2.8: do the conservation rows vanish for constant test
    functions?

    This is the test that matters once the multipliers sit in the conservation
    block. If those rows are genuinely degenerate, l_1, l_2, l_p occupy them and
    the system is square. If they are not, the multipliers over-determine it.

    The residual is linear in the test function, so pairing it with a chosen
    test function is a dot product against the assembled Cofunction. We pick the
    function that is 1 in one conservation slot and 0 everywhere else.

    Both rows should vanish to ROUNDING, not to discretization error, because
    the cancellations are exact at any state:

      constant w_i:  -M_i^-1 int div(J_i) + int r_i
                   = -M_i^-1 oint J_D,i.n + int r_i,  zero by the T_i scaling
                     chosen in _fix_compatibility

      constant q:    int div(Psi(J_1+J_2)) - int div(v)
                     - oint (Psi(J_1+J_2) - v).n
                   = 0 termwise, which is what the density consistency term is for

    So a large value here indicts _fix_compatibility or the ds term specifically,
    which is far more useful than a generic "Newton diverged".
    """
    F = problem.residual(sln, problem.r_1_d, problem.r_2_d)
    R = assemble(F, form_compiler_parameters={"quadrature_degree": problem.deg_max})

    # Scale to compare against: the largest entry of the assembled residual.
    with R.dat.vec_ro as rv:
        scale = max(rv.norm(), 1e-300)

    out = {}
    for label, slot in (("w_1 (continuity 1)", 3),
                        ("w_2 (continuity 2)", 4),
                        ("q   (mass average)", 5)):
        phi = Function(problem.Z)
        phi.subfunctions[slot].assign(1.0)
        with R.dat.vec_ro as rv, phi.dat.vec_ro as pv:
            out[label] = rv.dot(pv) / scale
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=2, choices=(2, 3))
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--N", type=int, default=2, help="cells per direction; keep tiny")
    ap.add_argument("--c", type=float, default=0.25, help="shift magnitude")
    args = ap.parse_args()

    problem = SOSMProblem(d=args.d, k=args.k, N_mesh=args.N)
    base = problem.initial_guess()

    ref_raw, ref_bc = residual_norms(problem, base)
    PETSc.Sys.Print(f"\nmixed space has {problem.Z.num_sub_spaces()} fields, "
                    f"{problem.Z.dim()} dofs")
    PETSc.Sys.Print(f"||F(U0)||  no bcs = {ref_raw:.12e}")
    PETSc.Sys.Print(f"||F(U0)||  bcs    = {ref_bc:.12e}   <-- the one that matters\n")

    singular = []
    for label, i_field, i_const in SHIFTS:
        shifted = base.copy(deepcopy=True)
        subs = shifted.subfunctions

        # Add c to the raw field, subtract c from its real-space constant, so
        # the augmented combination (field + constant) is unchanged.
        subs[i_field].assign(subs[i_field] + Constant(args.c))
        subs[i_const].assign(subs[i_const] - Constant(args.c))

        val_raw, val_bc = residual_norms(problem, shifted)
        rel_raw = abs(val_raw - ref_raw) / max(ref_raw, 1e-300)
        rel_bc = abs(val_bc - ref_bc) / max(ref_bc, 1e-300)

        if rel_bc < 1e-10:
            singular.append(label)
            verdict = "SINGULAR"
        else:
            verdict = "ok"

        PETSc.Sys.Print(f"  {label}:  rel_change no bcs = {rel_raw:.3e}   "
                        f"with bcs = {rel_bc:.3e}   {verdict}")

    PETSc.Sys.Print("")
    if singular:
        PETSc.Sys.Print(f"RESULT: {len(singular)} singular direction(s) with bcs "
                        f"applied: {', '.join(singular)}")
        PETSc.Sys.Print("The formulation is rank-deficient. See open_problems.md item 1.")
    else:
        PETSc.Sys.Print("RESULT: no exact symmetry with bcs applied.")
        PETSc.Sys.Print("Necessary, not sufficient -- Newton can still fail for other")
        PETSc.Sys.Print("reasons, but the three constants are at least determined.")

    PETSc.Sys.Print("\nIf the two columns disagree, trust the second: the first")
    PETSc.Sys.Print("includes constrained rows that the solve throws away.")

    # -- Lemma 2.8. The test that matters now the multipliers moved. ---------
    PETSc.Sys.Print("\n=== constant-test rows of the conservation block ===")
    PETSc.Sys.Print("(relative to ||F||; want ~1e-12 or below -- these cancel")
    PETSc.Sys.Print(" exactly at any state, so discretization error is no excuse)")

    rows = constant_row_test(problem, base)
    bad = []
    for label, val in rows.items():
        ok = abs(val) < 1e-10
        if not ok:
            bad.append(label)
        PETSc.Sys.Print(f"  {label}: {val:+.3e}   {'degenerate' if ok else 'NOT degenerate'}")

    PETSc.Sys.Print("")
    if bad:
        PETSc.Sys.Print("RESULT: Lemma 2.8 FAILS for: " + ", ".join(bad))
        PETSc.Sys.Print("Those rows are not degenerate, so the multipliers placed")
        PETSc.Sys.Print("there over-determine the system. Suspect, in order:")
        PETSc.Sys.Print("  w_i row -> _fix_compatibility / the T_i scaling")
        PETSc.Sys.Print("  q   row -> the density consistency ds term or its sign")
    else:
        PETSc.Sys.Print("RESULT: Lemma 2.8 holds numerically. The conservation rows")
        PETSc.Sys.Print("are degenerate for constant test functions, so the three")
        PETSc.Sys.Print("multipliers have exactly the rows they need.")


if __name__ == "__main__":
    main()
