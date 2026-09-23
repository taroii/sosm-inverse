"""
The outer Picard inverse iteration of `paper/template.tex` section 3.1.

One outer round is

    c_bar^(k)  --(A)-->  beta^(k+1)  --(B)-->  U^(k+1)  -->  c_bar^(k+1)

Step (A) freezes the concentrations at c_bar^(k) and minimizes the reduced
objective of the PICARD-LINEARIZED system over the parameter. Step (B) runs the
full nonlinear SOSM solver at the updated parameter and reads off a new
concentration field. The linearized system is the one already built and checked
in `diag_picard.py`, imported rather than duplicated.

WHY THIS IS WORTH RUNNING. E14 evaluated Theorem 2's criterion, rho(DT) < 1, at
every configuration reachable by the forward solver: it holds everywhere, worst
case 0.370 at D_true = 0.48 and about 0.045 at the reference configuration once
the sensor grid is dense. rho bounds the NUMBER of outer rounds, so a value near
0.045 predicts two or three. What it does not bound is the cost of a round,
which is what this file exists to measure.

THREE THINGS THAT ARE NOT FREE CHOICES.

1. The observable cannot be a concentration. The linearized system carries
   mm_1, mm_2, v, mu_1, mu_2, p and nothing else -- x_1 and x_2 are the frozen
   data, not unknowns -- so an x_1 observation has identically zero parameter
   sensitivity in step (A) and, with any regularization, step (A) silently
   returns the prior every round. Default here is mu_2, which E14 measured as
   the most sensitive of the shared fields. Every inversion in E12-E19 observes
   x_1, so the head-to-head comparison must re-run the DIRECT method on mu_2
   too (`invert.py --field mu_2`); otherwise it compares observables, not
   methods.

2. The gauge is frozen data, not a free constant. The linearized system inherits
   the three-fold nullspace (constants in mu_1, mu_2, p) MORE strongly than the
   nonlinear one, because the constitutive rows that pin mu_i to x_i are
   dropped. `Linearized` removes it with three real-space multipliers whose
   constraints pin the field means. For a SENSITIVITY those targets are
   irrelevant -- they are constants and differentiate away, which is why
   diag_picard could take them from a reference solution. For an INVERSION they
   are not: mu_2's absolute level enters the misfit, so a wrong gauge biases the
   parameter directly.

   The targets are therefore computed from the FROZEN state through the
   constitutive relation the linearized system drops,

       mean(mu_i) = mean(RT log(x_i_bar * p_bar)),   mean(p) = mean(p_bar),

   which is what the nonlinear system would produce at that state. That puts the
   gauge on the same footing as c_bar: exact at the fixed point, carrying the
   same O(||c_bar - c_bar*||) error away from it. It is part of the Picard
   approximation rather than an extra one.

3. Step (B) is never differentiated, so it is never taped. The tape is rebuilt
   from scratch each round because c_bar and the gauge have both moved, and
   replaying a stale tape would keep differentiating through the previous
   round's frozen state.

WHAT IS MEASURED, not judged. Per round: the parameter update, the concentration
update, the distance to the true parameter, the misfit of the NONLINEAR model
against the data (the honest objective, not step (A)'s surrogate), and the cost
in both units. Those are the quantities Algorithm 1 of the paper asks for.

Cost is reported in two units because they are not interchangeable: a linearized
solve is 0.76 of the nonlinear system's degrees of freedom (E14) but drops its
cheapest blocks. Wall time is reported alongside both counts, and the comparison
in README section IV should quote all three.

Usage:
    python src/picard.py --sigma 1e-3 --seed 0 --D-init 1.2
    python src/picard.py --field mu_1 --rounds 8
    python src/picard.py --D-true 2.0 --D-init 2.4
"""

import argparse
import time

import numpy as np

from firedrake import *
from firedrake.adjoint import *
from firedrake.exceptions import ConvergenceError
from firedrake.petsc import PETSc
from pyadjoint import Tape, set_working_tape

from sosm import SOSMProblem, solve_forward
from inverse import (FIELDS, LINEARIZED_OK, sensor_points, observer, observe,
                     synthetic_data)
from diag_picard import Linearized, FROZEN
from runlog import Run


class PicardInversion:
    """The outer loop. Build, then `run`."""

    def __init__(self, points, data, sigma, D_init, D_true, D_prior=None,
                 alpha=1e-4, d=2, k=4, N=16, field="mu_2", newton_max_it=50,
                 quiet=True, sigma_ref=1e-3, cont_max_step=0.35):
        if field not in LINEARIZED_OK:
            raise SystemExit(
                f"--field {field} is not a variable of the Picard-linearized "
                f"system, so step (A) would have zero parameter sensitivity and "
                f"would return the prior every round without raising. Choose "
                f"from {LINEARIZED_OK}. See E14 in notes/results.md.")
        self.field = field
        self.field_idx = FIELDS[field]
        self.sigma = sigma
        self.alpha = alpha
        self.D_true = D_true
        self.cont_max_step = cont_max_step
        # Same objective scaling as inverse.Inversion, for the same reason: it
        # keeps the magnitude, and hence the meaning of a gradient tolerance,
        # independent of sigma. A positive constant, so the minimizer is
        # unchanged.
        self.scale = (sigma / sigma_ref) ** 2

        pause_annotation()
        self.problem = SOSMProblem(d=d, k=k, N_mesh=N, quiet=quiet,
                                   newton_max_it=newton_max_it)
        self.kappa = Function(self.problem.R0, name="kappa")
        self.kappa.assign(float(np.log(D_init)))
        self.kappa_prior = Function(self.problem.R0)
        self.kappa_prior.assign(float(np.log(
            D_prior if D_prior is not None else D_init)))

        self.P0, self.P0_input = observer(self.problem.mesh, points)
        self.data = Function(self.P0_input)
        self.data.dat.data[:] = data

        # Gauge targets, rewritten from the frozen state every round. Constants,
        # so `Linearized` keeps a live reference to them.
        self.gauge = [Constant(0.0), Constant(0.0), Constant(0.0)]
        self.lin = Linearized(self.problem, self.gauge)

        # Initial nonlinear state at D_init, by continuation from the exact
        # solution at D = 1. This is step (B) run once before the first round,
        # and it supplies c_bar^(0).
        span = abs(float(np.log(D_init)))
        self.n_cont = max(1, int(np.ceil(span / cont_max_step)))
        self.state = self.problem.initial_guess()
        # initial_guess solves the problem exactly at D_1 * D_2 = 1, kappa = 0.
        self.kappa_at_state = 0.0
        self.n_nonlinear = 0
        self._step_B(float(np.log(D_init)), n_steps=self.n_cont)
        continue_annotation()

        self.n_linearized = 0
        self.n_lin_adjoint = 0
        self.history = []

    # -- the two steps ------------------------------------------------------

    def _step_B(self, kappa, n_steps=1):
        """Full nonlinear solve at exp(kappa), warm-started, never taped.

        Walks from `self.kappa_at_state` to `kappa` in `n_steps` equal steps,
        leaving both the state and `kappa_at_state` at the target.

        A failure is retried once as a finer walk. Step (A) can propose a
        parameter far from the current state, especially on the first round, and
        a step (B) that raises would destroy the whole loop -- while the walk is
        exactly the tool that reached D_init to begin with. Retried solves are
        counted in n_nonlinear, so a recovery shows up as cost rather than being
        hidden.
        """
        start = self.kappa_at_state
        backup = self.state.copy(deepcopy=True)
        pause_annotation()
        saved = self.problem.D_12
        try:
            for j in range(1, n_steps + 1):
                kap_j = start + (j / n_steps) * (kappa - start)
                self.problem.D_12 = Constant(float(np.exp(kap_j)))
                self.state = solve_forward(self.problem, sln=self.state,
                                           check=False)
                self.n_nonlinear += 1
        except ConvergenceError:
            self.problem.D_12 = saved
            continue_annotation()
            finer = max(2, int(np.ceil(abs(kappa - start) / self.cont_max_step)))
            if n_steps >= finer:
                raise
            PETSc.Sys.Print(
                f"  step (B) Newton failed; retrying as {finer} continuation "
                f"steps from D = {np.exp(start):.6f}", flush=True)
            self.state = backup
            self.kappa_at_state = start
            return self._step_B(kappa, n_steps=finer)
        self.problem.D_12 = saved
        continue_annotation()
        self.kappa_at_state = kappa

    def _update_gauge(self):
        """Gauge targets from the frozen state, via the dropped constitutive law.

        See the module docstring: for an inversion these are NOT free constants.
        """
        xb_1, xb_2, pb = (self.lin.frozen["x_1"], self.lin.frozen["x_2"],
                          self.lin.frozen["p"])
        mu_1_cr, mu_2_cr = self.problem.mu_relation(xb_1, xb_2, pb)
        vol, dx_ = self.problem.vol, self.problem.dx
        for target, expr in zip(self.gauge, (mu_1_cr, mu_2_cr, pb)):
            target.assign(assemble(expr * dx_) / vol)

    def _inner_functional(self):
        """Tape ONE linearized solve as a function of kappa, and reduce it.

        A fresh tape per round: c_bar and the gauge have both moved, and
        replaying last round's tape would differentiate through last round's
        frozen state.
        """
        set_working_tape(Tape())
        control = Control(self.kappa)
        self.problem.D_12 = exp(self.kappa)

        z = Function(self.lin.Z)
        F = self.lin.residual(z)
        # linear=False keeps the Newton options: F is linear in z, so SNES
        # converges in one step, but pyadjoint needs a solve block to tape.
        solve(F == 0, z, bcs=self.lin.bcs(),
              solver_parameters=self.lin.solver_parameters(linear=False),
              form_compiler_parameters={"quadrature_degree":
                                        self.problem.deg_max})
        self.n_linearized += 1

        obs = observe(z, (self.P0, self.P0_input), self.field_idx)
        misfit = 0.5 * inner(obs - self.data, obs - self.data) / self.sigma ** 2
        dk = self.kappa - self.kappa_prior
        reg = 0.5 * self.alpha * inner(dk, dk)
        J = self.scale * (assemble(misfit * dx)
                          + assemble(reg * dx(self.problem.mesh)))

        def on_eval(*_):
            self.n_linearized += 1

        def on_derivative(value, derivative, controls):
            # Must RETURN the derivative: pyadjoint uses this callback's return
            # value, not only its side effect, and raises if it gets None.
            self.n_lin_adjoint += 1
            return derivative

        return ReducedFunctional(J, control, eval_cb_post=on_eval,
                                 derivative_cb_post=on_derivative)

    def _nonlinear_misfit(self):
        """Misfit RMS of the CURRENT nonlinear state against the data.

        The honest objective. Step (A) minimizes a surrogate built on the
        linearized model; this is the quantity the surrogate stands in for, and
        the two agree only at the fixed point.
        """
        pause_annotation()
        obs = observe(self.state, (self.P0, self.P0_input), self.field_idx)
        r = obs.dat.data_ro - self.data.dat.data_ro
        continue_annotation()
        return float(np.sqrt(np.mean(r ** 2)))

    def _concentrations(self):
        """The frozen-field vector, for measuring the outer update.

        Concatenates x_1, x_2, p and rho_inv -- exactly what is frozen. They
        carry different units, so the RMS of a difference is a convergence
        diagnostic, not a physical norm.
        """
        return np.concatenate([self.state.subfunctions[i].dat.data_ro.copy()
                               for _, i, _ in FROZEN])

    # -- the loop -----------------------------------------------------------

    def run(self, rounds=12, gtol=1e-6, max_inner=100,
            D_min=0.5, D_max=10.0, tol=1e-8):
        """Outer rounds until both updates fall below `tol`, or `rounds`.

        `rounds` has a ceiling for a reason: with rho about 0.045 the loop should
        settle in two or three, so a run that needs twelve is itself the
        measurement and should not be left to spin.
        """
        kappa_prev = float(self.kappa.dat.data_ro[0])
        c_prev = self._concentrations()

        for k in range(1, rounds + 1):
            t0 = time.time()
            # Bookkeeping, not modelling: keep it off the tape. It would be
            # discarded by the set_working_tape below anyway, but taping an
            # assemble per round for nothing is still waste.
            pause_annotation()
            self.lin.freeze_at(self.state)
            self._update_gauge()
            continue_annotation()

            Jhat = self._inner_functional()
            opt_status = "ok"
            try:
                opt = minimize(Jhat, method="L-BFGS-B",
                               bounds=[float(np.log(D_min)),
                                       float(np.log(D_max))],
                               options={"gtol": gtol, "maxiter": max_inner})
                kappa_new = float(opt.dat.data_ro[0])
            except Exception as exc:
                # Same trade as inverse.Inversion.solve: keep the previous
                # iterate and flag it rather than losing the whole run.
                opt_status = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:160]
                kappa_new = kappa_prev
                PETSc.Sys.Print(f"  inner solve failed: {opt_status}",
                                flush=True)
            self.kappa.assign(kappa_new)

            self._step_B(kappa_new)
            c_new = self._concentrations()

            d_kappa = abs(kappa_new - kappa_prev)
            d_c = float(np.sqrt(np.mean((c_new - c_prev) ** 2)))
            row = {"round": k,
                   "D": float(np.exp(kappa_new)),
                   "kappa": kappa_new,
                   "d_kappa": d_kappa,
                   "d_c": d_c,
                   "err_vs_true": abs(float(np.exp(kappa_new)) - self.D_true)
                                  / self.D_true,
                   "misfit_rms": self._nonlinear_misfit(),
                   "n_nonlinear": self.n_nonlinear,
                   "n_linearized": self.n_linearized,
                   "n_lin_adjoint": self.n_lin_adjoint,
                   "wall_s": time.time() - t0,
                   "opt_status": opt_status}
            self.history.append(row)
            PETSc.Sys.Print(
                f"  round {k:>2d}  D = {row['D']:.8e}  |dkappa| = {d_kappa:.3e}"
                f"  |dc| = {d_c:.3e}  misfit = {row['misfit_rms']:.4e}"
                f"  nl/lin = {self.n_nonlinear}/{self.n_linearized}", flush=True)

            kappa_prev, c_prev = kappa_new, c_new
            # `opt_status == "ok"` is load-bearing. A failed inner solve leaves
            # kappa where it was, so d_kappa is 0 and step (B) reproduces the
            # same state, making d_c 0 too -- which without this guard would be
            # indistinguishable from convergence and would report a failed run
            # as a converged one.
            if opt_status == "ok" and d_kappa < tol and d_c < tol:
                break

        return float(np.exp(kappa_prev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=2, choices=(2, 3))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--data-k", type=int, default=None)
    ap.add_argument("--data-N", type=int, default=None)
    ap.add_argument("--sensors", type=int, default=4)
    ap.add_argument("--field", default="mu_2", choices=LINEARIZED_OK)
    ap.add_argument("--D-true", type=float, default=1.0)
    ap.add_argument("--D-init", type=float, default=1.2)
    ap.add_argument("--D-prior", type=float, default=None,
                    help="default: 1.4286 * D_true, the paper's ratio")
    ap.add_argument("--sigma", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--gtol", type=float, default=1e-6)
    ap.add_argument("--max-inner", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--D-min", type=float, default=0.5)
    ap.add_argument("--D-max", type=float, default=10.0)
    ap.add_argument("--newton-max-it", type=int, default=50)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    if args.data_k is None:
        args.data_k = 5 if args.d == 2 else 4
    if args.data_N is None:
        args.data_N = 64 if args.d == 2 else 8
    if args.D_prior is None:
        args.D_prior = (5.0e-2 / 3.5e-2) * args.D_true

    config = vars(args).copy()
    config.pop("allow_dirty")
    config.pop("verbose")

    with Run("picard", config, seed=args.seed,
             allow_dirty=args.allow_dirty) as run:
        points = sensor_points(args.d, args.sensors)
        data, clean = synthetic_data(points, args.D_true, args.sigma, args.seed,
                                     d=args.d, k=args.data_k, N=args.data_N,
                                     field=FIELDS[args.field])
        PETSc.Sys.Print(f"\nfield          = {args.field}", flush=True)
        PETSc.Sys.Print(f"sensors        = {len(points)}", flush=True)
        PETSc.Sys.Print(f"data rms       = {np.sqrt(np.mean(clean**2)):.6e}",
                        flush=True)
        PETSc.Sys.Print(f"noise sigma    = {args.sigma:.6e}\n", flush=True)

        inv = PicardInversion(points, data, args.sigma, args.D_init,
                              args.D_true, D_prior=args.D_prior,
                              alpha=args.alpha, d=args.d, k=args.k, N=args.N,
                              field=args.field,
                              newton_max_it=args.newton_max_it,
                              quiet=not args.verbose)

        t0 = time.time()
        D_rec = inv.run(rounds=args.rounds, gtol=args.gtol,
                        max_inner=args.max_inner, D_min=args.D_min,
                        D_max=args.D_max, tol=args.tol)
        wall = time.time() - t0

        for row in inv.history:
            run.record(**row)

        rel_err = abs(D_rec - args.D_true) / args.D_true
        # Geometric decay fitted to the outer updates: the empirical counterpart
        # of rho(DT) from diag_picard.py. |dkappa| should fall by rho per round,
        # so the fitted slope in log space exponentiates to that factor.
        # TRANSIENT ONLY, as the paper specifies. Once the outer update reaches
        # the inner solver's own noise floor it stops decaying, and including
        # that tail drags the fitted rate toward 1 -- so fit the leading stretch
        # over which |dkappa| is still strictly falling, and report how many
        # rounds went into the fit so the number can be judged.
        dk = [r["d_kappa"] for r in inv.history]
        trans = []
        for v in dk:
            if v <= 0 or (trans and v >= trans[-1]):
                break
            trans.append(v)
        rate, n_fit = "", len(trans)
        if n_fit >= 3:
            rate = float(np.exp(np.polyfit(np.arange(n_fit),
                                           np.log(trans), 1)[0]))

        PETSc.Sys.Print(f"\nrounds            = {len(inv.history)}", flush=True)
        PETSc.Sys.Print(f"D_true            = {args.D_true:.8e}", flush=True)
        PETSc.Sys.Print(f"D_recovered       = {D_rec:.8e}", flush=True)
        PETSc.Sys.Print(f"relative error    = {rel_err:.6e}", flush=True)
        PETSc.Sys.Print(f"outer rate        = {rate}   (geometric decay of "
                        f"|dkappa| over {n_fit} transient rounds of "
                        f"{len(inv.history)}; compare rho(DT) from "
                        f"diag_picard.py)", flush=True)
        PETSc.Sys.Print(f"nonlinear solves  = {inv.n_nonlinear}", flush=True)
        PETSc.Sys.Print(f"linearized solves = {inv.n_linearized} forward, "
                        f"{inv.n_lin_adjoint} adjoint", flush=True)
        PETSc.Sys.Print(f"wall s            = {wall:.2f}", flush=True)

        run.record(summary=1, git_sha=run.provenance["git_sha"],
                   method="picard", field=args.field,
                   D_true=args.D_true, D_recovered=D_rec, rel_err=rel_err,
                   sigma=args.sigma, seed=args.seed, D_init=args.D_init,
                   D_prior=args.D_prior, alpha=args.alpha,
                   k=args.k, N=args.N, d=args.d,
                   rounds=len(inv.history), outer_rate=rate,
                   outer_rate_n_fit=n_fit,
                   n_nonlinear=inv.n_nonlinear,
                   n_linearized=inv.n_linearized,
                   n_lin_adjoint=inv.n_lin_adjoint, wall_s=wall,
                   opt_status=";".join(sorted({r["opt_status"]
                                              for r in inv.history})))


if __name__ == "__main__":
    main()
