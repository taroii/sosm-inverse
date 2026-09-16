"""
Run ONE inversion and record it. Computes; does not plot.

Every paper figure is a query against the table this builds, so the axes a
figure needs must be flags here rather than separate scripts:

    --sigma      noise level          -> noise scaling
    --k --N      inversion mesh       -> mesh independence
    --D-init     starting guess       -> basin of attraction
    --seed       noise realisation    -> repeatability
    --method     lbfgs (picard next)  -> cost comparison
    --d          2 or 3               -> the three-dimensional demonstration

One run per process, one row per run in `runs/index.csv`, full history in the
run's `metrics.csv`.

Usage:
    python src/invert.py --check-gradient        # run this FIRST
    python src/invert.py --sigma 1e-3 --seed 0
    python src/invert.py --sigma 1e-2 --seed 3 --D-init 100.0
"""

import argparse

import numpy as np

from firedrake.petsc import PETSc

from inverse import sensor_points, synthetic_data, Inversion
from runlog import Run


def main():
    ap = argparse.ArgumentParser()

    # Truth and data generation. The data mesh is deliberately finer and of
    # higher degree than the inversion mesh; see inverse.synthetic_data.
    ap.add_argument("--D-true", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    # Defaults resolved per dimension after parsing: k=5, N=64 is an 11 GB
    # solve in 2-D (E9) but astronomically larger in 3-D, where E8 shows k=4,
    # N=8 is already 472888 dofs. Passing 2-D defaults to a 3-D run would OOM
    # before the inversion started.
    ap.add_argument("--data-k", type=int, default=None)
    ap.add_argument("--data-N", type=int, default=None)
    ap.add_argument("--sensors", type=int, default=4,
                    help="sensors per spatial direction")

    # Inversion.
    # 1.2, not 0.3. The old default predates the solvability measurement and
    # sits below the D ~ 0.45 limit, so every run using it died mid-continuation
    # after a 50-iteration Newton failure. 20 percent off the truth is a
    # reasonable starting guess and is comfortably inside the solvable range.
    ap.add_argument("--D-init", type=float, default=1.2)
    # Default is the paper's ratio, not the initial guess. template.tex section
    # 4 sets D_true = 3.5e-2, D_prior = 5.0e-2, so the prior sits 1.4286x ABOVE
    # the truth, and section 4's Interpretation attributes the residual 6.64
    # percent bias to exactly that pull. Defaulting the prior to D_init instead
    # would make the regularization gradient zero at the start and quietly change
    # what the Tikhonov term does, which is not the experiment the paper runs.
    # Expressed as a ratio so it tracks D_true rather than being a magic number.
    ap.add_argument("--D-prior", type=float, default=None,
                    help="default: 1.4286 * D_true, the paper's ratio")
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--d", type=int, default=2, choices=(2, 3))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--method", default="lbfgs", choices=("lbfgs",))
    ap.add_argument("--max-iter", type=int, default=100,
                    help="optimizer iterations")
    ap.add_argument("--gtol", type=float, default=1e-6,
                    help="projected-gradient tolerance; see Inversion.solve")
    # Search bounds. Continuation from the exact solution at D = 1 fails below
    # D ~ 0.45 in this configuration. That is a property of this solver and
    # anchor, not of the problem (see Inversion.solve), but it is where forward
    # solves currently stop existing, so the lower bound sits above it.
    ap.add_argument("--D-min", type=float, default=0.5)
    ap.add_argument("--D-max", type=float, default=10.0)
    ap.add_argument("--newton-max-it", type=int, default=50,
                    help="Newton iterations per forward solve")
    ap.add_argument("--cont-max-step", type=float, default=0.35,
                    help="largest continuation step in kappa (0.35 ~ 1.4x in D)")
    ap.add_argument("--verbose", action="store_true",
                    help="show the SNES residual history for each solve")

    ap.add_argument("--check-gradient", action="store_true",
                    help="verify the adjoint gradient instead of inverting")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    # (inversion N, data k, data N) per dimension. The data mesh must be finer
    # AND higher degree than the inversion mesh; that separation is the
    # inverse-crime avoidance.
    if args.N is None:
        args.N = 16 if args.d == 2 else 4
    if args.data_k is None:
        args.data_k = 5 if args.d == 2 else 4
    if args.data_N is None:
        args.data_N = 64 if args.d == 2 else 8

    if args.D_prior is None:
        args.D_prior = (5.0e-2 / 3.5e-2) * args.D_true

    if not (args.D_min <= args.D_init <= args.D_max):
        raise SystemExit(
            f"--D-init {args.D_init} is outside the search bounds "
            f"[{args.D_min}, {args.D_max}].\n"
            f"The continuation walk starts at D=1 and ends at D_init, so an "
            f"out-of-range start is not a hard guess -- it is a guess the "
            f"forward problem may have no solution for. Continuation from "
            f"D=1 has been observed to fail below D ~ 0.45 in this "
            f"configuration. Widen --D-min only if you have measured that the "
            f"forward solve survives there.")

    if args.data_k <= args.k or args.data_N <= args.N:
        PETSc.Sys.Print(
            f"NOTE: data mesh (k={args.data_k}, N={args.data_N}) is not finer "
            f"than the inversion mesh (k={args.k}, N={args.N}); "
            f"discretization errors will partly cancel.", flush=True)

    config = vars(args).copy()
    config.pop("allow_dirty")
    config.pop("verbose")
    slug = "gradient-check" if args.check_gradient else f"invert-{args.method}"

    with Run(slug, config, seed=args.seed, allow_dirty=args.allow_dirty) as run:

        points = sensor_points(args.d, args.sensors)
        data, clean = synthetic_data(points, args.D_true, args.sigma, args.seed,
                                     d=args.d, k=args.data_k, N=args.data_N)

        PETSc.Sys.Print(f"\nsensors        = {len(points)}", flush=True)
        PETSc.Sys.Print(f"data rms       = {np.sqrt(np.mean(clean**2)):.6e}",
                        flush=True)
        PETSc.Sys.Print(f"noise sigma    = {args.sigma:.6e}", flush=True)

        inv = Inversion(points, data, args.sigma, args.D_init,
                        D_prior=args.D_prior, alpha=args.alpha,
                        d=args.d, k=args.k, N=args.N,
                        newton_max_it=args.newton_max_it,
                        cont_max_step=args.cont_max_step,
                        quiet=not args.verbose)

        PETSc.Sys.Print(f"continuation   = {inv.n_cont} steps "
                        f"from D=1 to D={args.D_init}", flush=True)

        # Standing check, one extra solve. The misfit between this mesh's own
        # prediction at D_true and the data must sit at the noise level; if it
        # does not, the objective is not minimized near the truth and every
        # downstream number is meaningless. Cheap enough to run every time.
        rms = inv.data_check(args.D_true)
        PETSc.Sys.Print(f"data check     = {rms:.6e} rms at D_true "
                        f"(sigma = {args.sigma:.1e})", flush=True)

        if args.check_gradient:
            _check_gradient(inv, run, args)
            return

        J0 = inv.value()
        g0 = inv.gradient()
        PETSc.Sys.Print(f"J(D_init)      = {J0:.8e}", flush=True)
        PETSc.Sys.Print(f"dJ/dkappa      = {g0:.8e}\n", flush=True)

        D_rec = inv.solve(tol=args.gtol, max_iter=args.max_iter,
                          D_min=args.D_min, D_max=args.D_max)

        # Decoupled deliberately: an optional diagnostic must not destroy a
        # completed inversion. By finite differences of the adjoint gradient,
        # since pyadjoint's tangent-linear Hessian never worked here; see
        # Inversion.hessian_spectrum. Its solves are excluded from the forward
        # and adjoint counts below and reported separately.
        #
        # The full message is printed and the status recorded, so a failure
        # stays visible in the results table rather than only in scrollback.
        HESS_STEPS = (1e-2, 1e-3)
        spectrum = spectrum_coarse = None
        hess_status = "ok"
        try:
            spectrum_coarse, spectrum = inv.hessian_spectrum(steps=HESS_STEPS)
        except Exception as exc:
            hess_status = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:200]
            PETSc.Sys.Print(f"hessian eigs   = UNAVAILABLE", flush=True)
            PETSc.Sys.Print(f"  {hess_status}", flush=True)

        for row in inv.history:
            run.record(**row)

        rel_err = abs(D_rec - args.D_true) / args.D_true
        PETSc.Sys.Print(f"\nsearch bounds  = [{args.D_min}, {args.D_max}]", flush=True)
        PETSc.Sys.Print(f"D_true         = {args.D_true:.8e}", flush=True)
        PETSc.Sys.Print(f"D_recovered    = {D_rec:.8e}", flush=True)
        PETSc.Sys.Print(f"relative error = {rel_err:.6e}", flush=True)
        PETSc.Sys.Print(f"forward solves = {inv.n_forward}", flush=True)
        PETSc.Sys.Print(f"adjoint solves = {inv.n_adjoint}", flush=True)
        if spectrum is not None:
            for h, eigs in zip(HESS_STEPS, (spectrum_coarse, spectrum)):
                PETSc.Sys.Print(f"hessian eigs   = "
                                f"{', '.join(f'{e:.6e}' for e in eigs)}"
                                f"   (FD step {h:.0e} in kappa)", flush=True)
            PETSc.Sys.Print(f"hessian cost   = {inv.hessian_cost[0]} forward, "
                            f"{inv.hessian_cost[1]} adjoint (excluded above)",
                            flush=True)

        # One summary row, tagged so figures can separate it from the history.
        # git_sha on the row itself, not only in env.json: aggregating across
        # runs/ picks up everything ever run, including inversions from before a
        # bug fix. The sigma=1e-3 seed=0 cell showed three values, one of them
        # 1.59 from before the point-ordering fix, and nothing in the row said so.
        run.record(summary=1, git_sha=run.provenance["git_sha"],
                   D_true=args.D_true, D_recovered=D_rec,
                   rel_err=rel_err, sigma=args.sigma, seed=args.seed,
                   D_init=args.D_init, alpha=args.alpha, k=args.k, N=args.N,
                   D_min=args.D_min, D_max=args.D_max, gtol=args.gtol,
                   d=args.d, method=args.method,
                   n_forward=inv.n_forward, n_adjoint=inv.n_adjoint,
                   hess_status=hess_status,
                   hess_min=float(spectrum.min()) if spectrum is not None else "",
                   hess_max=float(spectrum.max()) if spectrum is not None else "",
                   hess_min_coarse=(float(spectrum_coarse.min())
                                    if spectrum is not None else ""),
                   hess_step=HESS_STEPS[1], hess_step_coarse=HESS_STEPS[0],
                   hess_n_forward=(inv.hessian_cost[0]
                                   if spectrum is not None else ""))


def _check_gradient(inv, run, args):
    """Centered differences against the adjoint, on THIS objective.

    Required by README.md section IV at every new configuration. E7 verified the
    gradient of a full-field misfit with respect to D_12; this objective is a
    different functional of a different variable, so it needs its own check.
    """
    from firedrake.adjoint import taylor_test, pause_annotation, continue_annotation

    g_adj = inv.gradient()
    kappa0 = float(np.log(args.D_init))
    PETSc.Sys.Print(f"\nadjoint dJ/dkappa = {g_adj:.8e}\n", flush=True)

    pause_annotation()
    for eps in np.logspace(-1, -8, 15):
        jp = float(inv.Jhat(inv.at_kappa(kappa0 + eps)))
        jm = float(inv.Jhat(inv.at_kappa(kappa0 - eps)))
        g_fd = (jp - jm) / (2.0 * eps)
        rel = abs(g_fd - g_adj) / max(abs(g_adj), 1e-300)
        run.record(eps=eps, g_fd=g_fd, g_adj=g_adj, rel_err=rel)
        PETSc.Sys.Print(f"  eps={eps:.2e}  fd={g_fd:.8e}  rel_err={rel:.3e}",
                        flush=True)

    rate = taylor_test(inv.Jhat, inv.at(args.D_init), inv.direction(1.0))
    continue_annotation()
    PETSc.Sys.Print(f"\ntaylor_test convergence rate: {rate:.3f}", flush=True)


if __name__ == "__main__":
    main()
