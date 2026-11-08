"""Per-seed test of the frozen-model error identity at the Picard fixed point.

At a fixed point (c_bar*, beta*) of the outer iteration, with O the observation
operator, W = sigma^-2 I, and r* = O U(beta*) - d the residual there:

    J_beta  = O dU_lin/dbeta |_c_bar    partial sensitivity, concentrations frozen
    sigma_nl= P_c dU/dbeta              nonlinear concentration sensitivity
    g       = O dU_lin/dc_bar . sigma_nl        the criterion's numerator vector
    J_full  = O dU/dbeta                total sensitivity through the nonlinear solve

The claims tested, each reported as numbers rather than judged:

    T1   grad Phi(beta*) = <g, W r*>, with Phi the direct reduced objective
    T2   <J_beta, W r*> ~ 0 (inner stationarity) while <J_full, W r*> = H Delta
    T3   Delta_s = <g, W r*_s> / H, seed by seed, no fitted parameter
    T4   J_full = J_beta + g componentwise, and a = ||g||_W/||J_beta||_W,
         c = cos_W(J_beta, g) recomputed at the fixed point rather than at D_true
    T5   discretization bias: O U_fine(beta_true) - O U_h(beta_true)

PARAMETERIZATION, since mixing them is the easiest way to get this wrong.
The linearized solves are taken in D, matching diag_picard.py. The recorded
Hessian is in kappa = log D, because inverse.py optimizes there. A sensitivity
converts as X^kappa = D X^D, so every inner product below is reported in BOTH
and the kappa column is the one that pairs with H and with Delta in kappa.

SIGNS, verified against the code rather than assumed. `solve_linear_system`
solves a.x = L, and the three right-hand sides carry explicit minus signs, so
sigma_nl = -P_c F_U^-1 F_beta and J_beta = -F_lin,U^-1 F_lin,beta -- the
convention the identity above is written in. Delta = D_picard - D_direct.

MEASURED, not judged. No tolerance is applied, nothing is called agreement, and
both sides of every claim are printed so the comparison is the output.

Usage:
    python src/diag_offset.py --D-true 1.0 --seeds 0 1 2 3 4 5 6 7 8 9
"""
import argparse
import csv
import functools
import glob
import os

import numpy as np

from firedrake import *
from firedrake.petsc import PETSc

from sosm import SOSMProblem, solve_forward
from inverse import (FIELDS, Inversion, observer, sensor_points,
                     synthetic_data)
from diag_picard import (FROZEN, SHARED, Linearized, point_values,
                         solve_linear_system, without_snes)
from runlog import Run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def recorded(method, field, D_true, sigma, k, N, sensors=None, sha=None):
    """Recovered values already on record, newest run per seed."""
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
            if sha is not None and (s.get("git_sha") or "")[:7] != sha:
                continue
            if method == "lbfgs" and (s.get("opt_status") or "ok") != "ok":
                continue
            seed = int(s["seed"])
        except (KeyError, ValueError, TypeError):
            continue
        d = os.path.basename(os.path.dirname(p))
        if seed not in out or d > out[seed][0]:
            out[seed] = (d, s)
    return out


def criterion_vectors(problem, star, spaces, field, fcp):
    """J_beta, g and lambda at the state `star`, all as sensor vectors in D.

    The same three solves as diag_picard.py, gauge-aware, at an arbitrary
    state rather than at D_true. The gauge targets take the state's own field
    means, which is what picard.py's constitutive update reduces to at a fixed
    point.
    """
    vol = problem.vol
    gauge = []
    for i in (3, 4, 5):
        f = Function(problem.R0)
        f.assign(assemble(star.subfunctions[i] * problem.dx) / vol)
        gauge.append(f)
    lin = Linearized(problem, gauge)
    lin.freeze_at(star)
    lin_params = lin.solver_parameters()

    z_lin = Function(lin.Z)
    F_lin = lin.residual(z_lin)
    solve(F_lin == 0, z_lin, bcs=lin.bcs(),
          solver_parameters={**problem.solver_parameters(),
                             "pc_fieldsplit_0_fields": "0,1,2,3,4,5",
                             "pc_fieldsplit_1_fields": "6,7,8"},
          form_compiler_parameters=fcp)

    dD = Function(problem.R0)
    dD.assign(1.0)
    a_lin = derivative(F_lin, z_lin)
    hom = lin.bcs(homogeneous=True)

    s_D = solve_linear_system(a_lin, -derivative(F_lin, problem.D_12, dD),
                              lin.Z, hom, lin_params, fcp)

    F_nl = problem.residual(star, problem.r_1_d, problem.r_2_d)
    hom_nl = [DirichletBC(problem.Z.sub(i), Constant((0.0,) * problem.d),
                          problem.bc_markers) for i in range(3)]
    sig = solve_linear_system(derivative(F_nl, star),
                              -derivative(F_nl, problem.D_12, dD),
                              problem.Z, hom_nl,
                              without_snes(problem.solver_parameters()), fcp)

    C_sig = functools.reduce(
        lambda a, b: a + b,
        [derivative(F_lin, lin.frozen[name], sig.subfunctions[idx])
         for name, idx, _ in FROZEN])

    # The gauge's own dependence on c_bar, as in diag_picard --gauge-aware.
    mu_1_cr, mu_2_cr = problem.mu_relation(lin.frozen["x_1"],
                                           lin.frozen["x_2"],
                                           lin.frozen["p"])
    deps = [(mu_1_cr, ("x_1", "p")), (mu_2_cr, ("x_2", "p")),
            (lin.frozen["p"], ("p",))]
    idx_of = {name: idx for name, idx, _ in FROZEN}
    for gi, (expr, names) in zip(gauge, deps):
        delta = 0.0
        for nm in names:
            delta += assemble(derivative(expr * problem.dx, lin.frozen[nm],
                                         sig.subfunctions[idx_of[nm]]))
        d_g = Function(problem.R0)
        d_g.assign(delta / vol)
        C_sig = C_sig + derivative(F_lin, gi, d_g)

    s_c = solve_linear_system(a_lin, -C_sig, lin.Z, hom, lin_params, fcp)

    i_l = SHARED[field][1]
    return (point_values(s_D, spaces, i_l), point_values(s_c, spaces, i_l),
            point_values(z_lin, spaces, i_l))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=2, choices=(2, 3))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--data-k", type=int, default=None)
    ap.add_argument("--data-N", type=int, default=None)
    ap.add_argument("--sensors", type=int, default=4)
    ap.add_argument("--field", default="mu_2", choices=SHARED.keys())
    ap.add_argument("--D-true", type=float, default=1.0)
    ap.add_argument("--D-prior", type=float, default=None)
    ap.add_argument("--sigma", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--picard-sha", default="7e88d93",
                    help="only pair against Picard runs at this commit")
    ap.add_argument("--fd-step", type=float, default=1e-3,
                    help="relative central-difference step for J_full")
    ap.add_argument("--newton-max-it", type=int, default=50)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    if args.data_k is None:
        args.data_k = 5 if args.d == 2 else 4
    if args.data_N is None:
        args.data_N = 64 if args.d == 2 else 8
    if args.D_prior is None:
        args.D_prior = (5.0e-2 / 3.5e-2) * args.D_true

    n_sens = args.sensors ** args.d
    i_nl = SHARED[args.field][0]
    pic = recorded("picard", args.field, args.D_true, args.sigma, args.k,
                   args.N, sensors=args.sensors, sha=args.picard_sha)
    dirc = recorded("lbfgs", args.field, args.D_true, args.sigma, args.k, args.N)
    seeds = [s for s in args.seeds if s in pic and s in dirc]

    PETSc.Sys.Print(f"\nD_true = {args.D_true}  field = {args.field}  "
                    f"sensors = {n_sens}  sigma = {args.sigma:g}  "
                    f"paired seeds = {len(seeds)}", flush=True)
    if not seeds:
        return

    points = sensor_points(args.d, args.sensors)
    problem = SOSMProblem(d=args.d, k=args.k, N_mesh=args.N, quiet=True,
                          newton_max_it=args.newton_max_it)
    spaces = observer(problem.mesh, points)
    # Same convention as diag_picard.py: the parameters belong on the form,
    # and deg_max is the problem's own choice.
    fcp = {"quadrature_degree": problem.deg_max}
    W = 1.0 / args.sigma ** 2

    config = vars(args).copy()
    config.pop("allow_dirty")

    with Run("offset", config, seed=None, allow_dirty=args.allow_dirty) as run:
        # --- T5, once per configuration: the fine-vs-coarse observation gap.
        _d, clean_fine = synthetic_data(points, args.D_true, args.sigma,
                                        args.seeds[0], d=args.d,
                                        k=args.data_k, N=args.data_N,
                                        field=FIELDS[args.field])
        star_t = solve_forward(problem, D_12=args.D_true, check=False)
        obs_h = point_values(star_t, spaces, i_nl)
        delta_disc = np.asarray(clean_fine) - obs_h
        PETSc.Sys.Print(
            f"\nT5  ||O U_fine(D_true) - O U_h(D_true)||_2 / sigma = "
            f"{np.linalg.norm(delta_disc) / args.sigma:.6e}"
            f"   (max |.|/sigma = {np.abs(delta_disc).max() / args.sigma:.6e})",
            flush=True)
        run.record(test="T5", D_true=args.D_true,
                   disc_norm_over_sigma=float(np.linalg.norm(delta_disc) / args.sigma),
                   disc_max_over_sigma=float(np.abs(delta_disc).max() / args.sigma))

        PETSc.Sys.Print(
            f"\n{'seed':>4} {'D_pic':>12} {'D_dir':>12} {'gradPhi':>12} "
            f"{'<g,Wr>_kap':>12} {'rel diff':>10} {'<Jb,Wr>_kap':>12} "
            f"{'<Jf,Wr>_kap':>12} {'H*Delta':>12} {'a':>7} {'c':>8} "
            f"{'Jf-(Jb+g)':>11}", flush=True)

        for seed in seeds:
            D_pic = float(pic[seed][1]["D_recovered"])
            D_dir = float(dirc[seed][1]["D_recovered"])
            H_kap = float(dirc[seed][1]["hess_min"])
            data, _ = synthetic_data(points, args.D_true, args.sigma, seed,
                                     d=args.d, k=args.data_k, N=args.data_N,
                                     field=FIELDS[args.field])
            data = np.asarray(data)

            star = solve_forward(problem, D_12=D_pic, check=False)
            r_star = point_values(star, spaces, i_nl) - data

            J_b_D, g_D, _ = criterion_vectors(problem, star, spaces,
                                              args.field, fcp)

            # J_full by central differences through the NONLINEAR solve.
            h = args.fd_step * D_pic
            up = point_values(solve_forward(problem, D_12=D_pic + h,
                                            sln=star.copy(deepcopy=True),
                                            check=False), spaces, i_nl)
            dn = point_values(solve_forward(problem, D_12=D_pic - h,
                                            sln=star.copy(deepcopy=True),
                                            check=False), spaces, i_nl)
            J_f_D = (up - dn) / (2.0 * h)

            # D -> kappa on every sensitivity, so they pair with H and Delta.
            J_b, g, J_f = D_pic * J_b_D, D_pic * g_D, D_pic * J_f_D

            ip = lambda u, v: float(W * np.dot(u, v))
            gWr, JbWr, JfWr = ip(g, r_star), ip(J_b, r_star), ip(J_f, r_star)
            nb, ng = np.sqrt(ip(J_b, J_b)), np.sqrt(ip(g, g))
            a = ng / nb
            c = ip(J_b, g) / (nb * ng)
            resid = J_f - (J_b + g)
            comp = float(np.abs(resid).max() / max(np.abs(J_f).max(), 1e-300))

            # The exact reduced gradient of the full objective at beta*.
            inv = Inversion(points, data, args.sigma, D_pic,
                            D_prior=args.D_prior, alpha=args.alpha, d=args.d,
                            k=args.k, N=args.N, field=FIELDS[args.field],
                            quiet=True)
            inv.Jhat(inv.at_kappa(float(np.log(D_pic))))
            grad = float(inv.Jhat.derivative().dat.data_ro[0]) / inv.scale

            d_kap = float(np.log(D_pic) - np.log(D_dir))
            rel = abs(grad - gWr) / max(abs(grad), 1e-300)

            PETSc.Sys.Print(
                f"{seed:>4} {D_pic:>12.8f} {D_dir:>12.8f} {grad:>12.4e} "
                f"{gWr:>12.4e} {rel:>10.2e} {JbWr:>12.4e} {JfWr:>12.4e} "
                f"{H_kap * d_kap:>12.4e} {a:>7.4f} {c:>+8.4f} {comp:>11.2e}",
                flush=True)
            run.record(test="T1-T4", seed=seed, D_true=args.D_true,
                       D_picard=D_pic, D_direct=D_dir,
                       grad_phi=grad, g_W_r=gWr, grad_vs_g_W_r_rel=rel,
                       Jbeta_W_r=JbWr, Jfull_W_r=JfWr,
                       H_kappa=H_kap, H_gauss_newton=float(ip(J_f, J_f)),
                       delta_kappa=d_kap, H_times_delta=H_kap * d_kap,
                       pred_delta_kappa=gWr / H_kap,
                       a_ratio=a, cos_c=c, Jfull_decomp_rel_err=comp,
                       r_star_rms=float(np.sqrt(np.mean(r_star ** 2))),
                       norm_Jbeta_W=nb, norm_g_W=ng,
                       sigma=args.sigma, field=args.field, sensors=n_sens,
                       k=args.k, N=args.N, alpha=args.alpha,
                       picard_run=pic[seed][0], direct_run=dirc[seed][0])

        run.record(summary=1, git_sha=run.provenance["git_sha"],
                   method="offset", D_true=args.D_true, field=args.field,
                   sensors=n_sens, sigma=args.sigma, k=args.k, N=args.N,
                   alpha=args.alpha, n_seeds=len(seeds))


if __name__ == "__main__":
    main()
