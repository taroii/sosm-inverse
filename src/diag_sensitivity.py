"""
Diagnostic: how much does each field at the sensors respond to a change in D_12?

No adjoint, no optimizer, two forward solves. This decides which observable the
inverse problem should use, and it is cheap enough that guessing is inexcusable.

Why it is not obvious. The manufactured configuration prescribes J_i . n and v as
Dirichlet data on the WHOLE boundary, and fixes div(J_i) = M_i r_i in the
interior. A flux field with prescribed divergence and prescribed normal trace has
only a divergence-free, zero-normal-trace remainder left to vary, so the fluxes
may be nearly blind to D_12 even though the diffusivity is exactly what governs
them. The driving forces have no such constraint: the manufactured solution has
mu_i = g / D_i, which scales as 1/D directly.

If that reasoning holds, the fields that are easiest to measure in a laboratory
are the ones carrying least information here, and the reverse. Either way the
numbers settle it.

Reported per field: the relative change in the sensor values between two
diffusivities, in the same input point ordering the inverse problem uses.

This also exercises VectorFunctionSpace on a VertexOnlyMesh, which every vector
candidate depends on and which has never been run.

Usage:
    python src/diag_sensitivity.py
    python src/diag_sensitivity.py --D-a 1.0 --D-b 1.2 --k 4 --N 16
"""

import argparse

import numpy as np

from firedrake import *
from firedrake.petsc import PETSc

from sosm import SOSMProblem, solve_forward
from inverse import sensor_points

# (label, index in the mixed space, is it a vector, present in the
#  Picard-linearized six-field system?)
FIELDS = [
    ("J_1",     0, True,  True),
    ("J_2",     1, True,  True),
    ("v",       2, True,  True),
    ("mu_1",    3, False, True),
    ("mu_2",    4, False, True),
    ("p",       5, False, True),
    ("x_1",     6, False, False),
    ("x_2",     7, False, False),
    ("rho_inv", 8, False, False),
]


def sample(problem, sln, points):
    """Every field at the sensor points, in input ordering."""
    vom = VertexOnlyMesh(problem.mesh, points)
    out = {}
    for label, idx, is_vec, _ in FIELDS:
        if is_vec:
            P = VectorFunctionSpace(vom, "DG", 0)
            P_in = VectorFunctionSpace(vom.input_ordering, "DG", 0)
        else:
            P = FunctionSpace(vom, "DG", 0)
            P_in = FunctionSpace(vom.input_ordering, "DG", 0)
        at_pts = assemble(interpolate(split(sln)[idx], P))
        out[label] = assemble(interpolate(at_pts, P_in)).dat.data_ro.copy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=2, choices=(2, 3))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--sensors", type=int, default=4)
    ap.add_argument("--D-a", type=float, default=1.0)
    ap.add_argument("--D-b", type=float, default=1.2)
    args = ap.parse_args()

    points = sensor_points(args.d, args.sensors)

    problem = SOSMProblem(d=args.d, k=args.k, N_mesh=args.N, quiet=True)
    a = sample(problem, solve_forward(problem, D_12=args.D_a, check=False), points)

    problem_b = SOSMProblem(d=args.d, k=args.k, N_mesh=args.N, quiet=True)
    b = sample(problem_b, solve_forward(problem_b, D_12=args.D_b, check=False), points)

    dD = (args.D_b - args.D_a) / args.D_a

    PETSc.Sys.Print(f"\n{len(points)} sensors, d={args.d}, k={args.k}, N={args.N}")
    PETSc.Sys.Print(f"D: {args.D_a} -> {args.D_b}   (relative change {dD:+.3f})\n")
    PETSc.Sys.Print(f"{'field':>8} {'lin?':>5} {'rms(A)':>12} {'rms(B-A)':>12} "
                    f"{'rel change':>12} {'per unit dD':>12}")

    for label, _, _, in_lin in FIELDS:
        va, vb = a[label], b[label]
        rms_a = float(np.sqrt(np.mean(va ** 2)))
        rms_d = float(np.sqrt(np.mean((vb - va) ** 2)))
        rel = rms_d / rms_a if rms_a > 0 else float("nan")
        PETSc.Sys.Print(f"{label:>8} {'yes' if in_lin else 'no':>5} "
                        f"{rms_a:12.5e} {rms_d:12.5e} {rel:12.5e} "
                        f"{rel / abs(dD):12.5e}")

    PETSc.Sys.Print("\nlin? = present as an unknown in the Picard-linearized system")
    PETSc.Sys.Print("rel change = rms(B-A)/rms(A) at the sensor points")


if __name__ == "__main__":
    main()
