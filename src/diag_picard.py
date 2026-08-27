"""
Diagnostic: does the outer Picard inverse iteration contract?

Theorem 2 of `paper/template.tex` gives a locally checkable criterion for the
outer loop of section 3.1: it converges locally when

    rho(DT) < 1,   DT = P_c F_U^-1 F_beta H_bb^-1 G_c

evaluated at a self-consistent point. That criterion has never been evaluated.
This file evaluates it exactly -- not by estimating a spectral radius, but as a
single signed number -- in three linear solves.

Why three solves suffice for n = 2. DT is the Jacobian of the composition

    c_bar --(A) inner solve--> beta --(B) nonlinear solve--> c_bar

The inner problem has N_beta = C(n,2) = 1 unknown, so DT factors through a
one-dimensional space and has rank one. A rank-one matrix has exactly one
nonzero eigenvalue and it equals the trace, so

    lambda = - <j_beta, J_c sigma_nl>_W / ( <j_beta, j_beta>_W + R'' ),

    j_beta   = O d(U_lin)/d(beta)                one LINEARIZED solve
    sigma_nl = P_c dU/dbeta through the NONLINEAR solve   one NONLINEAR-Jacobian solve
    J_c s    = O d(U_lin)/d(c_bar) . sigma_nl    one LINEARIZED solve

and rho(DT) = |lambda|. Nothing here is an estimate. For n > 2 the same
reduction gives a C(n,2) x C(n,2) matrix rather than the N_c x N_c operator the
theorem names, which is the generalization this file is written to admit later.

Three properties of the reported number, each worth knowing before reading it.

  1. It does not depend on the noise level. W = sigma^-2 I cancels between
     numerator and denominator, so the UNREGULARIZED lambda is identical at
     every sigma. Only the regularized value moves with sigma, through the
     ratio of alpha to the misfit curvature.

  2. It does not depend on the parameterization. Under beta -> D -> kappa the
     numerator picks up one factor of the chain rule and the Gauss-Newton
     denominator picks up two, and DT is a map on c_bar-space either way.
     Verified below by computing in D and reporting in all three. Regularized
     values are NOT invariant, because R'' does not transform with the misfit.

  3. It is measured at the TRUE parameter with NOISELESS data, so the residual
     r* vanishes and the second-order terms T_bb, T_bc in H_bb and G_c drop.
     That is the self-consistent point Theorem 2 assumes. At a noisy optimum
     r* != 0 and both terms return, at size O(||r*||); this file reports
     ||r*|| so the size of what was dropped is visible rather than assumed.

WHAT THIS FILE DOES NOT MEASURE, and a correction to an argument recorded
elsewhere in this repository. An analysis circulated earlier reduced DT to
(I - K)^-1 applied to the frozen sensitivity, where K is the Jacobian of the
ORDINARY forward Picard iteration, and concluded that the outer loop contracts
iff spec(K) lies in {Re z < 1/2} against {|z| < 1} for forward Picard. That
reduction needs the Picard-consistency identity

    F_lin(U, beta; P_c U) == F(U, beta)

to hold with F_lin and F on the SAME space. Here they are not: the linearized
system drops the constitutive rows (y_1, y_2), the density row (r) and the
three integral constraints, so its state has 9 fields against the nonlinear
system's 12, and P_c U_lin does not exist -- the linearized model cannot
produce a new c_bar at all, which is precisely why step (B) exists. So K is
not defined for this formulation and the 1/2 threshold does not apply to it.
The rank-one criterion computed here needs no such identity and is exact.

The gauge. The linearized system inherits the same three-fold nullspace as the
nonlinear one -- constants in mu_1, mu_2 and p -- and inherits it MORE strongly,
since the constitutive rows that tied mu_i to x_i are gone. The original removes
it with point Dirichlet conditions (manufactured_solution.py:692); we remove it
the way we remove it everywhere else, with three R-space multipliers in the
(w_1, w_2, q) rows, constrained here by matching the field means to the
reference solution's. The targets are constants, so they differentiate away and
affect none of A, B or C: they change WHERE the linearized solution sits, not
how it responds. They matter only to the consistency check below.

The consistency check. At c_bar = c_bar* and D = D_true the linearized solution
must equal the nonlinear one in all six shared fields, since freezing a
coefficient at its own value cannot change the equation it sits in. That is the
one place the implementation can be wrong without any solve failing, so it is
checked and reported first. A large number there invalidates everything after
it.

Observables. The linearized state carries mm_1, mm_2, v, mu_1, mu_2, p and
NOTHING ELSE. x_1 is not a variable of the linearized model, so it cannot be
the step (A) observable at any regularization -- a structural fact, not a
sensitivity claim, and one the inverse experiments currently violate by
observing X1. Scalar observables only here (mu_1, mu_2, p); the vector fields
need a VectorFunctionSpace observer, which diag_sensitivity.py exercises.

Usage:
    python src/diag_picard.py                         # default, mu_2
    python src/diag_picard.py --field p --k 4 --N 16
    python src/diag_picard.py --field mu_1 --alpha 1e-4
"""

import argparse
import functools
import math

import numpy as np

from firedrake import *
from firedrake.petsc import PETSc

from sosm import SOSMProblem, solve_forward
from inverse import sensor_points, observer

# Field name -> (index in the 12-field nonlinear space, index in the 9-field
# linearized space). Only the six shared fields appear; only scalars are
# observable with the DG0 observer.
SHARED = {"mm_1": (0, 0), "mm_2": (1, 1), "v": (2, 2),
          "mu_1": (3, 3), "mu_2": (4, 4), "p": (5, 5)}
SCALAR = ("mu_1", "mu_2", "p")

# Frozen coefficients, as (name, index in the nonlinear space, space key).
# These are the four fields Aaron's solve_linear replaces by manufactured
# values: c_1 and c_2 enter through conc_relation(x_1, x_2, p), and rho_inv
# enters directly. Freezing the STATE variables rather than c_1, c_2 avoids a
# projection: c_1 = x_1/(x_1+x_2) * p/RT is not in any one of our spaces.
FROZEN = (("x_1", 6, "X_1"), ("x_2", 7, "X_2"),
          ("p", 5, "P"), ("rho_inv", 8, "R"))


class Linearized:
    """The Picard-linearized SOSM system, with c_bar an explicit coefficient.

    Mirrors SOSMProblem.residual term for term. Every coefficient position that
    the original's `solve_linear` fills with a manufactured value is filled
    here from `self.frozen`; everything else is a live unknown.

    Dropped relative to the nonlinear residual, matching the original:
    the constitutive rows (y_1, y_2), the density-reciprocal row (r), and the
    three integral constraints -- all three of which constrain frozen
    quantities and would be vacuous.
    """

    def __init__(self, problem, gauge_targets):
        self.p = problem
        S = problem.spaces
        self.Z = (S["W_1"] * S["W_2"] * S["V"] * S["U_1"] * S["U_2"] * S["P"]
                  * S["L"] * S["L"] * S["L"])
        self.frozen = {name: Function(S[key], name="bar_" + name)
                       for name, _, key in FROZEN}
        self.gauge = gauge_targets

    def freeze_at(self, sln):
        """Copy c_bar out of a nonlinear state."""
        for name, idx, _ in FROZEN:
            self.frozen[name].assign(sln.subfunctions[idx])

    def bcs(self, homogeneous=False):
        p = self.p
        g = ([Constant((0.0,) * p.d)] * 3 if homogeneous
             else [p.g_1, p.g_2, p.g_v])
        return [DirichletBC(self.Z.sub(i), g[i], p.bc_markers) for i in range(3)]

    def residual(self, z):
        p = self.p
        dx_, ds_ = p.dx, p.ds
        M_1, M_2, RT = p.M_1, p.M_2, p.RT
        eta, lame, gamma = p.eta, p.lame, p.gamma

        mm_1, mm_2, v, mu_1, mu_2, pr, l_1, l_2, l_p = split(z)
        u_1, u_2, u, w_1, w_2, q, t_1, t_2, t_p = TestFunctions(self.Z)

        xb_1, xb_2, pb, rb = (self.frozen["x_1"], self.frozen["x_2"],
                              self.frozen["p"], self.frozen["rho_inv"])
        _, c_1, c_2 = p.conc_relation(xb_1, xb_2, pb)
        grad_rb = grad(p.rho_inv_ms) if p.use_grad_rho_inv_exact else grad(rb)

        A_visc = 2.0 * eta * inner(sym(grad(v)), sym(grad(u))) * dx_
        A_visc += lame * inner(div(v), div(u)) * dx_

        A_osm = (RT / ((c_1 + c_2) * p.D_12)) * (
            (c_2 / (M_1 * M_1 * c_1)) * inner(mm_1, u_1)
            + (c_1 / (M_2 * M_2 * c_2)) * inner(mm_2, u_2)
            - (1.0 / (M_1 * M_2)) * (inner(mm_1, u_2) + inner(mm_2, u_1))) * dx_
        A_osm += gamma * inner(v - (rb * (mm_1 + mm_2)),
                               u - (rb * (u_1 + u_2))) * dx_

        B = (inner(pr, (rb * div(u_1 + u_2)) + dot(grad_rb, u_1 + u_2))
             - inner(pr, div(u))) * dx_
        B -= ((1.0 / M_1) * inner(mu_1, div(u_1))
              + (1.0 / M_2) * inner(mu_2, div(u_2))) * dx_

        BT = (inner(q, (rb * div(mm_1 + mm_2)) + dot(grad_rb, mm_1 + mm_2))
              - inner(q, div(v))) * dx_
        BT -= ((1.0 / M_1) * inner(w_1, div(mm_1))
               + (1.0 / M_2) * inner(w_2, div(mm_2))) * dx_

        res = A_visc + A_osm + B + BT

        # Multipliers in the degenerate conservation rows, as in the nonlinear
        # system. The gauge below replaces the three integral constraints,
        # which here constrain frozen quantities and so are vacuous.
        res += (l_1 * w_1 + l_2 * w_2 + l_p * q) * dx_
        res += (mu_1 - self.gauge[0]) * t_1 * dx_
        res += (mu_2 - self.gauge[1]) * t_2 * dx_
        res += (pr - self.gauge[2]) * t_p * dx_

        if p.density_consistency:
            res -= q * inner((rb * (mm_1 + mm_2)) - v, p.nml) * ds_

        res -= (inner(p.f * ((M_1 * c_1) + (M_2 * c_2)), u)
                - inner(w_1, p.r_1_d) - inner(w_2, p.r_2_d)) * dx_

        return res

    def solver_parameters(self, linear=True):
        """Same matfree + Schur structure as the nonlinear solve.

        The linearized system still contains R blocks -- the three multipliers
        -- so it is still not monolithically assemblable. Six PDE fields in
        split 0 instead of nine; otherwise identical.
        """
        sp = self.p.solver_parameters()
        sp["pc_fieldsplit_0_fields"] = "0,1,2,3,4,5"
        sp["pc_fieldsplit_1_fields"] = "6,7,8"
        return without_snes(sp) if linear else sp


def without_snes(params):
    """Drop the Newton options, which conflict with a linear solve.

    Firedrake routes `solve(a == L, ...)` through a LinearVariationalSolver,
    which sets `snes_type: ksponly` itself; leaving `newtonls` and its
    tolerances in the dictionary fights that.
    """
    out = dict(params)
    for key in [k for k in out if k.startswith("snes_")]:
        out.pop(key)
    return out


def solve_linear_system(a, L, space, bcs, params, fcp):
    """One linear solve on a space with R blocks."""
    out = Function(space)
    solve(a == L, out, bcs=bcs, solver_parameters=params,
          form_compiler_parameters=fcp)
    return out


def point_values(fn, spaces, field):
    """Point evaluation of one field, in input point order, as a numpy array."""
    P0, P0_input = spaces
    at_points = assemble(interpolate(split(fn)[field], P0))
    return assemble(interpolate(at_points, P0_input)).dat.data_ro.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=2, choices=(2, 3))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--N", type=int, default=16, help="cells per direction")
    ap.add_argument("--D-true", type=float, default=1.0)
    ap.add_argument("--field", default="mu_2", choices=SCALAR,
                    help="step (A) observable")
    ap.add_argument("--sensors", type=int, default=4,
                    help="sensors per dimension")
    ap.add_argument("--sigma", type=float, default=1e-3,
                    help="noise level; enters only the regularized value")
    ap.add_argument("--alpha", type=float, default=1e-4,
                    help="Tikhonov weight, as in eq. prototype-objective")
    args = ap.parse_args()

    PETSc.Sys.Print(f"\nd={args.d} k={args.k} N={args.N} "
                    f"D_true={args.D_true} field={args.field}")

    # -- The self-consistent point: nonlinear solve at the true parameter. ---
    problem = SOSMProblem(d=args.d, k=args.k, N_mesh=args.N, quiet=True)
    fcp = {"quadrature_degree": problem.deg_max}
    star = solve_forward(problem, D_12=args.D_true, check=False)
    PETSc.Sys.Print(f"nonlinear dofs   {problem.Z.dim()}")

    # Gauge targets: the reference means. Constants, so they differentiate
    # away and touch none of A, B, C -- see the module docstring.
    vol = problem.vol
    gauge = [Constant(assemble(star.subfunctions[i] * problem.dx) / vol)
             for i in (3, 4, 5)]

    lin = Linearized(problem, gauge)
    lin.freeze_at(star)
    lin_params = lin.solver_parameters()
    PETSc.Sys.Print(f"linearized dofs  {lin.Z.dim()} "
                    f"({lin.Z.dim() / problem.Z.dim():.3f} of nonlinear)")

    # -- Consistency check: freezing at the solution's own values must
    # -- reproduce the solution in the six shared fields. -------------------
    z_lin = Function(lin.Z)
    F_lin = lin.residual(z_lin)
    solve(F_lin == 0, z_lin, bcs=lin.bcs(),
          solver_parameters={**problem.solver_parameters(),
                             "pc_fieldsplit_0_fields": "0,1,2,3,4,5",
                             "pc_fieldsplit_1_fields": "6,7,8"},
          form_compiler_parameters=fcp)

    PETSc.Sys.Print("\nconsistency of the linearization "
                    "(||U_lin - U*|| / ||U*|| per shared field, frozen at c*)")
    # `.subfunctions`, not `split()`: a form mixing two DIFFERENT mixed
    # functions carries two MeshSequenceGeometry domains, which is the same
    # class of trap that produced the MismatchingDomainError in the observer.
    # Subfunctions are ordinary Functions on the base mesh.
    for name, (i_nl, i_l) in SHARED.items():
        a_, b_ = z_lin.subfunctions[i_l], star.subfunctions[i_nl]
        num = math.sqrt(assemble(inner(a_ - b_, a_ - b_) * problem.dx))
        den = math.sqrt(assemble(inner(b_, b_) * problem.dx))
        PETSc.Sys.Print(f"  {name:>6s}  {num / den:.6e}")

    # -- The three solves. ---------------------------------------------------
    # Computed in D. lambda is invariant under D -> beta -> kappa when
    # unregularized; the regularized values are converted explicitly below.
    dD = Function(problem.R0)
    dD.assign(1.0)

    a_lin = derivative(F_lin, z_lin)
    hom = lin.bcs(homogeneous=True)

    # (i) j_D = O dU_lin/dD
    s_D = solve_linear_system(a_lin, -derivative(F_lin, problem.D_12, dD),
                              lin.Z, hom, lin_params, fcp)

    # (ii) sigma_nl = P_c dU/dD through the NONLINEAR solve
    F_nl = problem.residual(star, problem.r_1_d, problem.r_2_d)
    hom_nl = [DirichletBC(problem.Z.sub(i), Constant((0.0,) * problem.d),
                          problem.bc_markers) for i in range(3)]
    sig = solve_linear_system(derivative(F_nl, star),
                              -derivative(F_nl, problem.D_12, dD),
                              problem.Z, hom_nl,
                              without_snes(problem.solver_parameters()), fcp)

    # (iii) J_c sigma_nl = O dU_lin/dc_bar . sigma_nl, summed over the four
    # frozen coefficients.
    # functools.reduce, not sum(): sum() starts from the integer 0 and UFL
    # will not add a Form to it.
    C_sig = functools.reduce(
        lambda a, b: a + b,
        [derivative(F_lin, lin.frozen[name], sig.subfunctions[idx])
         for name, idx, _ in FROZEN])
    s_c = solve_linear_system(a_lin, -C_sig, lin.Z, hom, lin_params, fcp)

    # -- Assemble the criterion. --------------------------------------------
    points = sensor_points(args.d, args.sensors)
    spaces = observer(problem.mesh, points)
    i_l = SHARED[args.field][1]

    j = point_values(s_D, spaces, i_l)
    Jc = point_values(s_c, spaces, i_l)

    jj = float(j @ j)
    jJc = float(j @ Jc)
    lam_unreg = -jJc / jj

    # r* at the true parameter with noiseless data is the consistency residual
    # above, restricted to the sensors.
    r_star = (point_values(z_lin, spaces, i_l)
              - point_values(star, spaces, SHARED[args.field][0]))

    PETSc.Sys.Print(f"\nsensors                 {len(points)} ({args.sensors}^{args.d})")
    PETSc.Sys.Print(f"||r*||_2 at sensors     {np.linalg.norm(r_star):.6e}"
                    "   (dropped second-order terms are O of this)")
    PETSc.Sys.Print(f"||j_D||_2               {np.sqrt(jj):.6e}")
    PETSc.Sys.Print(f"||J_c sigma_nl||_2      {np.linalg.norm(Jc):.6e}")
    PETSc.Sys.Print(f"cos(j_D, J_c sigma_nl)  "
                    f"{jJc / (np.sqrt(jj) * np.linalg.norm(Jc)):+.6f}")

    PETSc.Sys.Print("\nlambda = -<j,Jc.sigma>_W / (<j,j>_W + R''),  "
                    "rho(DT) = |lambda|, exact for n=2")
    PETSc.Sys.Print(f"  unregularized         lambda = {lam_unreg:+.6e}   "
                    f"rho = {abs(lam_unreg):.6e}")

    # Regularized, in each parameterization. W = 1/sigma^2 does not cancel once
    # R'' is present, so sigma enters here and only here. Chain-rule factors at
    # D = D_true: beta = 1/D so dbeta/dD = -1/D^2; kappa = log D so
    # dkappa/dD = 1/D. R'' is alpha in whichever variable R is written.
    D = args.D_true
    W = 1.0 / args.sigma ** 2
    for label, chain in (("beta = 1/D  ", -1.0 / D ** 2),
                         ("D           ", 1.0),
                         ("kappa = logD", 1.0 / D)):
        # <j,j> in that variable picks up chain^2; the numerator picks up chain.
        num = W * jJc * chain
        den = W * jj * chain ** 2 + args.alpha
        PETSc.Sys.Print(f"  R in {label}      lambda = {-num / den:+.6e}   "
                        f"rho = {abs(num / den):.6e}")

    # The alpha that would bring rho to 1, in each parameterization. Negative
    # means rho < 1 already at alpha = 0.
    PETSc.Sys.Print("\nalpha giving rho(DT) = 1  (negative: already below 1 at alpha = 0)")
    for label, chain in (("beta = 1/D  ", -1.0 / D ** 2),
                         ("D           ", 1.0),
                         ("kappa = logD", 1.0 / D)):
        a_crit = W * abs(jJc * chain) - W * jj * chain ** 2
        PETSc.Sys.Print(f"  R in {label}      {a_crit:+.6e}")


if __name__ == "__main__":
    main()
