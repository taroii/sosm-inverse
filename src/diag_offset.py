"""Does the Picard fixed point's offset from the direct optimum match the
first-order prediction from its own residual?

THE IDENTITY UNDER TEST. Direct optimization drives the reduced gradient of the
FULL nonlinear objective to zero. The Picard fixed point does not: it is
stationary for the frozen objective, so the true gradient there is whatever the
frozen approximation left behind. One Newton step from the fixed point toward
the true optimum is

    kappa_dir - kappa_pic  ~  - grad J(kappa_pic) / H,

with H the curvature of the full reduced objective. In D, to first order,
D_dir - D_pic ~ D_pic * that. If the proposition is right, this holds SEED BY
SEED, not merely in spread.

TWO THINGS THAT HAD TO BE SETTLED BEFORE WRITING THIS.

1. The `j_D` printed by diag_picard.py is NOT the right sensitivity for this
   predictor. It is `O dU_lin/dD` -- the LINEARIZED sensitivity (see that
   file's module docstring). The inner problem is stationary for exactly that
   contraction, so

       <j_lin, W r*>  =  -R'(kappa_pic)  ~  0,

   and a predictor built on it would forecast almost no offset at all, which
   would read as the proposition failing when it is the wrong g. The quantity
   that survives is the FULL sensitivity, and the part of it that matters is
   the difference between full and linearized.

2. The predictor does not have to be assembled from g and r at all. pyadjoint
   already gives the exact reduced gradient of the full objective at any
   kappa for one nonlinear solve plus one adjoint, and that is the quantity the
   Newton step wants. So the headline number is computed directly, and the
   decomposition is reported alongside it as a check rather than used as the
   route. Since

       grad J = scale * ( <g_full, W r*> + alpha (kappa - kappa_prior) ),

   the contraction <g_full, W r*> follows from the gradient with no extra
   solves, and only <j_lin, W r*> needs a linearized solve to confirm item 1.

MEASURED, not judged. Per seed: the two recovered values, the reduced gradient
and its decomposition at the Picard fixed point, the curvature, the residual,
the predicted offset and the actual one. No tolerance is applied and nothing is
called agreement -- the comparison is the output.

Usage:
    python src/diag_offset.py --D-true 1.0 --seeds 0 1 2 3 4 5 6 7 8 9
"""
import argparse
import csv
import glob
import os

import numpy as np

from firedrake import *
from firedrake.adjoint import *
from firedrake.petsc import PETSc

from inverse import (FIELDS, Inversion, sensor_points, synthetic_data)
from runlog import Run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def recorded(method, field, D_true, sigma, k, N, sensors=None):
    """Recovered values already on record, keyed by seed.

    `sensors` is absent from metrics written before 6a52fc1, so a caller that
    needs to distinguish sensor counts must pass one and accept that older runs
    cannot match.
    """
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "metrics.csv"))):
        rows = list(csv.DictReader(open(p, newline="")))
        if not rows:
            continue
        s = next((r for r in reversed(rows)
                  if r.get("summary") not in (None, "", "0")), None)
        if s is None or s.get("method") != method:
            continue
        if (s.get("field") or "x_1") != field:
            continue
        try:
            if (abs(float(s["D_true"]) - D_true) > 1e-12
                    or abs(float(s["sigma"]) - sigma) > 1e-15
                    or int(float(s["k"])) != k or int(float(s["N"])) != N):
                continue
            if sensors is not None:
                if s.get("sensors") in (None, ""):
                    continue
                if int(float(s["sensors"])) != sensors:
                    continue
            if method == "lbfgs" and (s.get("opt_status") or "ok") != "ok":
                continue
            seed = int(s["seed"])
        except (KeyError, ValueError, TypeError):
            continue
        d = os.path.basename(os.path.dirname(p))
        if seed not in out or d > out[seed][0]:
            out[seed] = (d, s)
    return {k_: v for k_, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=2, choices=(2, 3))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--data-k", type=int, default=None)
    ap.add_argument("--data-N", type=int, default=None)
    ap.add_argument("--sensors", type=int, default=4)
    ap.add_argument("--field", default="mu_2")
    ap.add_argument("--D-true", type=float, default=1.0)
    ap.add_argument("--D-prior", type=float, default=None)
    ap.add_argument("--sigma", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(range(10)))
    ap.add_argument("--hessian-step", type=float, default=1e-2,
                    help="central-difference step for H at the fixed point")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    if args.data_k is None:
        args.data_k = 5 if args.d == 2 else 4
    if args.data_N is None:
        args.data_N = 64 if args.d == 2 else 8
    if args.D_prior is None:
        args.D_prior = (5.0e-2 / 3.5e-2) * args.D_true

    n_sens = args.sensors ** args.d
    pic = recorded("picard", args.field, args.D_true, args.sigma,
                   args.k, args.N, sensors=args.sensors)
    dir_ = recorded("lbfgs", args.field, args.D_true, args.sigma,
                    args.k, args.N)
    seeds = [s for s in args.seeds if s in pic and s in dir_]
    missing = [s for s in args.seeds if s not in seeds]

    PETSc.Sys.Print(f"\nD_true = {args.D_true}  field = {args.field}  "
                    f"sensors = {n_sens}  sigma = {args.sigma:g}", flush=True)
    PETSc.Sys.Print(f"paired seeds on record: {len(seeds)} of "
                    f"{len(args.seeds)}", flush=True)
    if missing:
        PETSc.Sys.Print(f"  no pair for seeds {missing}", flush=True)
    if not seeds:
        return

    config = vars(args).copy()
    config.pop("allow_dirty")

    points = sensor_points(args.d, args.sensors)

    with Run("offset", config, seed=None, allow_dirty=args.allow_dirty) as run:
        PETSc.Sys.Print(
            f"\n{'seed':>4} {'D_picard':>13} {'D_direct':>13} "
            f"{'grad J(k_pic)':>14} {'<g,Wr*>':>12} {'H':>11} "
            f"{'pred dD':>12} {'actual dD':>12} {'pred/actual':>11}", flush=True)
        for seed in seeds:
            D_pic = float(pic[seed][1]["D_recovered"])
            D_dir = float(dir_[seed][1]["D_recovered"])
            kap_pic = float(np.log(D_pic))

            data, _clean = synthetic_data(points, args.D_true, args.sigma,
                                          seed, d=args.d, k=args.data_k,
                                          N=args.data_N,
                                          field=FIELDS[args.field])
            inv = Inversion(points, data, args.sigma, D_pic,
                            D_prior=args.D_prior, alpha=args.alpha,
                            d=args.d, k=args.k, N=args.N,
                            field=FIELDS[args.field], quiet=True)

            # The exact reduced gradient of the FULL objective at the Picard
            # fixed point. One nonlinear solve plus one adjoint.
            inv.Jhat(inv.at_kappa(kap_pic))
            grad = float(inv.Jhat.derivative().dat.data_ro[0])

            # Decomposition, free. The regularizer is integrated over the
            # domain, not evaluated, so its derivative carries |Omega|:
            #     grad = scale * ( <g, W r*> + alpha * dkappa * |Omega| ).
            vol = float(inv.problem.vol)
            dkap_prior = kap_pic - float(np.log(args.D_prior))
            reg_term = args.alpha * dkap_prior * vol
            contraction = grad / inv.scale - reg_term

            # Curvature at the fixed point, by central differences of the same
            # adjoint gradient -- the path verified to 1e-10, not pyadjoint's
            # tangent-linear pass, which has never worked here.
            h = args.hessian_step

            def grad_at(kv):
                inv.Jhat(inv.at_kappa(kv))
                return float(inv.Jhat.derivative().dat.data_ro[0])

            H = (grad_at(kap_pic + h) - grad_at(kap_pic - h)) / (2.0 * h)

            # Residual at the sensors. `Inversion` keeps the solution local
            # to its tape builder, so rather than reach for it, invert the
            # objective, which contains the residual exactly:
            #     J/scale = (1/2) sum_j r_j^2 / sigma^2
            #               + (1/2) alpha dkappa^2 |Omega|.
            J_pic = float(inv.Jhat(inv.at_kappa(kap_pic)))
            ss = args.sigma ** 2 * (2.0 * J_pic / inv.scale
                                    - args.alpha * dkap_prior ** 2 * vol)
            r_rms = float(np.sqrt(max(ss, 0.0) / n_sens))

            d_kap_pred = -grad / H
            dD_pred = D_pic * np.expm1(d_kap_pred)
            dD_act = D_dir - D_pic

            PETSc.Sys.Print(
                f"{seed:>4} {D_pic:>13.8f} {D_dir:>13.8f} {grad:>14.6e} "
                f"{contraction:>12.4e} {H:>11.4e} {dD_pred:>12.4e} "
                f"{dD_act:>12.4e} "
                f"{(dD_pred / dD_act if dD_act else float('nan')):>11.4f}",
                flush=True)
            run.record(seed=seed, D_true=args.D_true, D_picard=D_pic,
                       D_direct=D_dir, kappa_picard=kap_pic,
                       grad_full=grad, contraction_g_W_r=contraction,
                       reg_term=reg_term, H_at_fixed_point=H,
                       H_recorded_direct=dir_[seed][1].get("hess_min"),
                       r_star_rms=r_rms, J_at_fixed_point=J_pic,
                       sigma=args.sigma,
                       d_kappa_pred=d_kap_pred, dD_pred=dD_pred,
                       dD_actual=dD_act, field=args.field, sensors=n_sens,
                       k=args.k, N=args.N, alpha=args.alpha,
                       hessian_step=h,
                       picard_run=pic[seed][0], direct_run=dir_[seed][0])
        run.record(summary=1, git_sha=run.provenance["git_sha"],
                   method="offset", D_true=args.D_true, field=args.field,
                   sensors=n_sens, sigma=args.sigma, k=args.k, N=args.N,
                   alpha=args.alpha, n_seeds=len(seeds))


if __name__ == "__main__":
    main()
