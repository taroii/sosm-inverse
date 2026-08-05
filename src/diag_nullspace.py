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

Two tests, and the second is the one that matters now that the multipliers sit
in the conservation block:

  1. shift symmetry -- decisive in ONE direction only. An unchanged norm proves
     singularity. A changed norm does not prove nonsingularity.
  2. Lemma 2.8, the constant-test rows of the conservation block. These must
     vanish to rounding at every configuration, since the cancellations are
     exact rather than asymptotic.

Both run over a sweep of (k, N_mesh, shift), because one configuration is an
observation, not a result. An exact cancellation holds everywhere; an accidental
near-zero does not, and a value that shrinks with N is discretization error
masquerading as one.

Usage:
    python src/diag_nullspace.py                    # the sweep
    python src/diag_nullspace.py --single --k 2 --N 2
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


def run_one(d, k, N, c, apply_bcs):
    """One configuration. Returns (shift results, constant-row results)."""
    problem = SOSMProblem(d=d, k=k, N_mesh=N)
    base = problem.initial_guess()
    if apply_bcs:
        for bc in problem.bcs():
            bc.apply(base)

    _, ref_bc = residual_norms(problem, base)

    shifts = {}
    for label, i_field, i_const in SHIFTS:
        shifted = base.copy(deepcopy=True)
        subs = shifted.subfunctions
        subs[i_field].assign(subs[i_field] + Constant(c))
        subs[i_const].assign(subs[i_const] - Constant(c))
        _, val_bc = residual_norms(problem, shifted)
        shifts[label] = abs(val_bc - ref_bc) / max(ref_bc, 1e-300)

    return shifts, constant_row_test(problem, base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=2, choices=(2, 3))
    ap.add_argument("--c", type=float, default=0.25, help="shift magnitude")
    ap.add_argument("--single", action="store_true",
                    help="one configuration only, instead of the sweep")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--N", type=int, default=2, help="cells per direction")
    args = ap.parse_args()

    # A single configuration is one observation. An exact cancellation must hold
    # at EVERY mesh size, polynomial degree and shift magnitude; an accidental
    # near-zero will not. The sweep is what separates the two, and it costs
    # seconds because every configuration here is tiny.
    if args.single:
        configs = [(args.k, args.N, args.c)]
    else:
        configs = [(k, N, c)
                   for k in (2, 3)
                   for N in (2, 3, 4)
                   for c in (0.25, 1.0)]

    PETSc.Sys.Print(f"\nsweeping {len(configs)} configurations "
                    f"(k, N_mesh, shift), d={args.d}\n")

    rows_all, shifts_all = [], []
    for (k, N, c) in configs:
        shifts, rows = run_one(args.d, k, N, c, apply_bcs=True)
        rows_all.append(((k, N, c), rows))
        shifts_all.append(((k, N, c), shifts))
        tag = f"k={k} N={N} c={c}"
        row_str = "  ".join(f"{lbl.split()[0]}={v:+.2e}" for lbl, v in rows.items())
        PETSc.Sys.Print(f"  {tag:16s} conservation rows: {row_str}")

    # -- Verdicts across the whole sweep. ------------------------------------
    PETSc.Sys.Print("\n=== Lemma 2.8, across all configurations ===")
    worst = {}
    for _, rows in rows_all:
        for lbl, v in rows.items():
            worst[lbl] = max(worst.get(lbl, 0.0), abs(v))

    failed = [lbl for lbl, v in worst.items() if v > 1e-10]
    for lbl, v in worst.items():
        PETSc.Sys.Print(f"  {lbl}: worst |value| = {v:.3e}   "
                        f"{'degenerate' if v <= 1e-10 else 'NOT degenerate'}")

    PETSc.Sys.Print("")
    if failed:
        PETSc.Sys.Print("RESULT: Lemma 2.8 FAILS for: " + ", ".join(failed))
        PETSc.Sys.Print("  w_i row -> _fix_compatibility / the T_i scaling")
        PETSc.Sys.Print("  q   row -> the density consistency ds term or its sign")
        PETSc.Sys.Print("A value that shrinks with N is discretization error, not an")
        PETSc.Sys.Print("exact cancellation, and still means the row is not degenerate.")
    else:
        PETSc.Sys.Print("RESULT: Lemma 2.8 holds at every configuration tested.")
        PETSc.Sys.Print("Still an observation, not a proof -- but an accidental")
        PETSc.Sys.Print("cancellation across all of these would be a strange accident.")

    PETSc.Sys.Print("\n=== shift symmetry, worst case across sweep ===")
    worst_shift = {}
    for _, shifts in shifts_all:
        for lbl, v in shifts.items():
            worst_shift[lbl] = min(worst_shift.get(lbl, np.inf), v)
    for lbl, v in worst_shift.items():
        PETSc.Sys.Print(f"  {lbl}: smallest rel_change = {v:.3e}   "
                        f"{'SINGULAR' if v < 1e-10 else 'ok'}")
    PETSc.Sys.Print("\nReminder: a changed norm is NOT proof of nonsingularity.")
    PETSc.Sys.Print("It only rules out this particular family of symmetries.")


if __name__ == "__main__":
    main()
