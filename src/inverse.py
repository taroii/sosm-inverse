"""
Inverse-problem machinery for the SOSM forward model in `sosm.py`.

Library only -- no CLI, no plotting. `invert.py` drives it.

The pieces, in the order an inversion uses them:

  sensor_points   where we pretend to measure
  observe         differentiable point evaluation (VertexOnlyMesh)
  synthetic_data  generate noisy observations at the true parameters
  Inversion       reduced objective, gradient, optimizer, Hessian spectrum

Two design points worth stating up front.

**Point observations are what make the inverse crime avoidable.** Generating the
data with the same mesh used for inversion lets the discretization errors cancel
exactly, and the recovery looks perfect for the wrong reason. Here the data are
produced on a finer, higher-degree mesh, evaluated at sensor locations, and
returned as plain numbers. Numbers carry no mesh, so the inversion compares its
own interpolation against them with no cross-mesh form -- which is also what
avoids the MismatchingDomainError that two SOSMProblem instances would otherwise
produce.

**The parameter is inferred in log space.** The control is kappa with
D_12 = exp(kappa), so positivity is structural rather than a bound the optimizer
has to respect, and a multiplicative error in D becomes an additive error in
kappa -- which is the right scale for a quantity known only to an order of
magnitude.

>>> NOT YET VALIDATED. Written without running it. `invert.py --check-gradient`
>>> is the gate: README.md section IV requires the adjoint gradient verified
>>> against centered differences at every new configuration, and the objective
>>> here is a new configuration. E7 does not carry over.
"""

import hashlib
import json

import numpy as np

from firedrake import *
from firedrake.adjoint import *
from firedrake.petsc import PETSc

from sosm import SOSMProblem, solve_forward

__all__ = ["sensor_points", "observer", "observe", "synthetic_data",
           "Inversion"]

# Index of x_1 in the mixed space, i.e. the field we observe.
X1 = 6


def _scalar(control):
    """Read the scalar out of whatever pyadjoint hands a callback.

    ReducedFunctional calls `self.controls.delist(values)`, which returns a bare
    Function when there is a single control and a list when there are several.
    Indexing the bare Function hits UFL's operator and raises IndexError, so
    normalise here rather than at each call site -- this becomes a list again
    the moment n > 2 introduces several diffusivities.
    """
    if isinstance(control, (list, tuple)):
        control = control[0]
    return float(control.dat.data_ro[0])


# ---------------------------------------------------------------------------
# Observation operator
# ---------------------------------------------------------------------------

def sensor_points(d, n_per_dim, margin=0.15):
    """A regular grid of sensor locations, held off the boundary.

    The margin matters: the flux and velocity are prescribed on the whole
    boundary, so sensors placed there would measure the boundary data we
    supplied rather than the solution's response to D_12.
    """
    axis = np.linspace(margin, 1.0 - margin, n_per_dim)
    grids = np.meshgrid(*([axis] * d), indexing="ij")
    return np.column_stack([g.ravel() for g in grids])


def observer(mesh, points):
    """(P0, P0_input) on a VertexOnlyMesh at `points`.

    Returns TWO spaces, and the second is the one that matters for correctness.

    A VertexOnlyMesh does not keep the points in the order they were given: it
    orders them by which cell owns them, so the ordering depends on the mesh.
    Data generated on the fine mesh and compared against observations on the
    coarse mesh are therefore permuted differently, and a misfit built from the
    raw `.dat.data` of each compares sensor i against sensor j. It does not
    error -- it silently minimizes the wrong quantity.

    `vom.input_ordering` is a mesh whose ordering matches the input array, so
    interpolating into its P0 space puts every mesh's values in the same,
    canonical order. All comparisons happen there.

    Created ONCE per mesh and reused: two VertexOnlyMesh objects over the same
    points are still distinct domains, and a form mixing them raises
    MismatchingDomainError.

    `mesh` is passed explicitly rather than taken from a solution because for a
    mixed function space `.mesh()` returns a MeshSequenceGeometry, which
    VertexOnlyMesh does not accept.
    """
    vom = VertexOnlyMesh(mesh, points)
    return (FunctionSpace(vom, "DG", 0),
            FunctionSpace(vom.input_ordering, "DG", 0))


def observe(sln, spaces, field=X1):
    """Differentiable point evaluation of one field, in INPUT point order.

    `Function.at()` is NOT tapeable and must never appear in an objective.
    Interpolation onto a VertexOnlyMesh is the supported route, and it is what
    firedrake.adjoint records; the second interpolation restores input ordering
    and is equally tapeable.

    The returned Function lives on the input-ordering mesh, whose `dx` measure
    sums over the points, so a misfit is `assemble(... * dx)` as usual.
    """
    P0, P0_input = spaces
    at_points = assemble(interpolate(split(sln)[field], P0))
    return assemble(interpolate(at_points, P0_input))


def _cache_path(d, k, N, D_true, points, field):
    """Cache key for the clean observations, including the code version.

    The git SHA is in the key deliberately: any change to the forward model
    invalidates the cache automatically, which is what stops a stale fine-mesh
    solve from silently contaminating every later run.

    `field` is in the key too, and its absence was a latent bug: `synthetic_data`
    takes a field index but the key ignored it, so changing the observed field
    would have silently loaded observations of the PREVIOUS field. No error, and
    a misfit comparing one quantity against another -- the same failure class as
    the point-ordering bug of E11.
    """
    from runlog import provenance, repo_root
    blob = json.dumps({"d": d, "k": k, "N": N, "D": D_true, "field": field,
                       "pts": np.asarray(points).round(12).tolist(),
                       "sha": provenance()["git_sha"]}, sort_keys=True).encode()
    cache = repo_root() / "runs" / ".data_cache"
    return cache / (hashlib.sha256(blob).hexdigest()[:16] + ".npy")


def hash_sigma(sigma):
    """A stable integer key for a float noise level, for seeding.

    Via the exact bit pattern, so 1e-3 and 0.001 give the same stream and two
    distinct floats never collide -- unlike rounding or string formatting.
    """
    return int(np.float64(sigma).view(np.uint64))


def synthetic_data(points, D_true, sigma, seed, d=2, k=5, N=64, field=X1):
    """Observations at the true parameter, on a deliberately finer mesh.

    Defaults are higher degree and finer mesh than any inversion should use --
    that separation IS the inverse-crime avoidance, so do not quietly match them
    to the inversion configuration. They are also 2-D defaults; see invert.py,
    which resolves them per dimension.

    The clean solve is CACHED, because it does not depend on the seed. Without
    that, a ten-seed sweep repeats an identical fine-mesh solve ten times -- at
    k=5, N=64 in 2-D that is 11 GB and 65 s each (E9), so six concurrent jobs
    would exceed the machine's 48 GB before any inversion started.

    Returns (values, clean_values) as numpy arrays: noisy and noise-free. The
    second is only for reporting the achievable floor, never for fitting.
    """
    path = _cache_path(d, k, N, D_true, points, field)
    if path.exists():
        clean = np.load(path)
    else:
        pause_annotation()
        problem = SOSMProblem(d=d, k=k, N_mesh=N, quiet=True)
        sln = solve_forward(problem, D_12=D_true, check=False)
        clean = observe(sln, observer(problem.mesh, points), field).dat.data_ro.copy()
        continue_annotation()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, clean)

    # Seed on (seed, sigma), not seed alone. With `default_rng(seed)` the standard
    # normal draw z is IDENTICAL at every noise level and only rescaled by sigma,
    # so a noise sweep measures one realization five times instead of five. That
    # makes "error scales linearly with sigma" follow from smoothness of the
    # parameter-to-data map rather than from the data, and it forbids pooling
    # across levels. SeedSequence mixes the two entries, so (0, 1e-3) and
    # (0, 1e-2) are independent streams while (0, 1e-3) stays reproducible.
    # Changing this changes every noise-sweep number: see E13 in notes/results.md.
    rng = np.random.default_rng(np.random.SeedSequence([seed, hash_sigma(sigma)]))
    noisy = clean + rng.normal(0.0, sigma, size=clean.shape)
    return noisy, clean


# ---------------------------------------------------------------------------
# The inverse problem
# ---------------------------------------------------------------------------

class Inversion:
    """Reduced objective over log D_12, with gradient, optimizer and Hessian.

    J(kappa) = 1/(2 sigma^2) sum_j (obs_j - data_j)^2
             + alpha/2 |kappa - kappa_prior|^2

    Scaling the misfit by sigma^-2 is what makes alpha mean the same thing at
    every noise level; without it the regularization silently strengthens as the
    data get cleaner.
    """

    def __init__(self, points, data, sigma, D_init, D_prior=None, alpha=1e-4,
                 d=2, k=4, N=16, field=X1, newton_max_it=50, quiet=True,
                 cont_max_step=0.35, sigma_ref=1e-3, tape_continuation=False):
        self.points = points
        self.sigma = sigma
        # Overall scale factor. Multiplying an objective by a positive constant
        # leaves its minimizer, and the relative weight of misfit against
        # regularization, exactly unchanged -- only the magnitude the solver
        # sees changes.
        #
        # That magnitude matters. The misfit carries a 1/sigma^2 factor, so J
        # spans eight orders of magnitude across the noise sweep, and with it the
        # adjoint right-hand side. At sigma = 1e-4 the adjoint residual cannot
        # reach an absolute snes_atol and every seed exits DIVERGED_MAX_IT, while
        # sigma = 3e-4 and coarser all succeed. Scaling by (sigma/sigma_ref)^2
        # pins the magnitude to what it would be at sigma_ref, for every sigma.
        self.scale = (sigma / sigma_ref) ** 2
        self.field = field
        self.alpha = alpha
        self.n_forward = 0
        self.n_adjoint = 0
        self.history = []

        pause_annotation()
        # newton_max_it is generous here, and deliberately larger than the
        # forward model's default of 10. Every solve in E2-E8 started from the
        # projected manufactured solution AT D_12 = 1.0, so Newton had almost
        # nothing to do and converged in one to four iterations. An inversion
        # evaluates the forward model far from the true parameter -- that is the
        # entire point -- so it needs a real iteration budget.
        self.problem = SOSMProblem(d=d, k=k, N_mesh=N, quiet=quiet,
                                   newton_max_it=newton_max_it)

        # Record the continuation trace so a failed run says WHICH step failed
        # rather than only that something did.
        self.cont_trace = []

        # Control is kappa, with D_12 = exp(kappa). Overriding the attribute
        # with a UFL expression is deliberate: sosm.py only ever reads D_12
        # inside the Onsager block, so an expression works exactly as a Function
        # does. Do NOT call solve_forward(..., D_12=value) after this point --
        # that path assigns to D_12 and assumes it is still a Function.
        self.kappa = Function(self.problem.R0, name="kappa")
        self.kappa.assign(float(np.log(D_init)))
        self.problem.D_12 = exp(self.kappa)

        self.kappa_prior = Function(self.problem.R0)
        self.kappa_prior.assign(float(np.log(D_prior if D_prior is not None else D_init)))

        # Built once, outside the tape: nine LU projections that depend only on
        # the manufactured solution, and would otherwise replay on every
        # objective evaluation.
        self.guess = self.problem.initial_guess()

        # ONE observation space, shared by the data and by the taped
        # observation in _build. The values arrived from the finer data mesh as
        # plain numbers, so there is no cross-mesh coupling -- but the data and
        # the observation must live on the SAME vertex-only mesh.
        self.P0, self.P0_input = observer(self.problem.mesh, points)
        self.data = Function(self.P0_input)
        self.data.dat.data[:] = data
        continue_annotation()

        # Continuation step count, fixed once here and never varied afterwards.
        #
        # `initial_guess` is the projected manufactured solution, which solves
        # the problem exactly at D_12 = D_1 * D_2 = 1, i.e. kappa = 0. Starting
        # Newton there and jumping straight to kappa = log(0.3) stalls: the
        # residual falls from 12.7 to 9.9 and then sits at 10.4 while the line
        # search cuts the step to nothing. So we walk there instead, in steps
        # small enough that each one is the size of jump Newton handles easily
        # (a factor of about 1.4 in D, which converged in four iterations in E7).
        #
        # The count depends on D_init only, so the tape has a FIXED number of
        # solves -- which it must, since ReducedFunctional replays the recorded
        # tape rather than re-running this function.
        self.kappa_ref = 0.0
        span = abs(float(np.log(D_init)) - self.kappa_ref)
        self.n_cont = max(1, int(np.ceil(span / cont_max_step)))

        # Solves performed per objective evaluation. With the walk ON the tape
        # every replay repeats all n_cont of them; with it off, a replay is a
        # single solve and the walk is paid once, at build. This is the factor
        # the cost comparison in README.md section IV turns on.
        self.tape_continuation = tape_continuation
        self._solves_per_eval = self.n_cont if tape_continuation else 1

        self.Jhat = self._build()
        # _build performs n_cont forward solves to lay the tape (taped) or to
        # reach the starting state (untaped), and those do not pass through
        # eval_cb_post either way. Counting them matters: they are real work.
        self.n_forward = self.n_cont

    # -- Tape ---------------------------------------------------------------

    def _build(self):
        """Lay the tape. The continuation walk is on it or not, per the flag.

        WHY UNTAPING IS CORRECT, since it looks like it should change the
        gradient and does not. The taped walk makes every intermediate state a
        differentiable function of the control, so a replay repeats all n_cont
        solves and the adjoint runs backwards through all of them. But the
        discrete solution satisfies F(U, kappa) = 0, so

            dU/dkappa = -F_U^-1 F_kappa

        which depends on the EQUATION, not on the path Newton took to solve it.
        The walk only supplies a starting state. Taping it therefore costs
        n_cont times the work for a derivative that is, up to solver tolerance,
        the same one. Untaping keeps the walk as a warm start, performed once
        with annotation off, and tapes the single solve at kappa itself.

        That argument is why the flag exists rather than a silent switch: it is
        an argument, and E19 measures it both ways before the untaped path is
        trusted.

        The cost of taping is the whole reason to care. One derivative() call
        replays the tape, so at D_init = 8 (n_cont = 6) a taped evaluation is
        six nonlinear solves and six adjoint solves; untaped it is one of each.
        """
        control = Control(self.kappa)

        sln = Function(self.problem.Z)
        sln.assign(self.guess)
        kappa_end = float(self.kappa.dat.data_ro[0])

        if not self.tape_continuation:
            pause_annotation()

        # Walk from kappa_ref to kappa in n_cont equal steps, warm-starting each
        # solve from the previous. Interpolating in kappa (not D) makes the
        # steps geometric in D, which is the right spacing for a quantity
        # spanning orders of magnitude.
        n_walk = self.n_cont if self.tape_continuation else self.n_cont - 1
        for j in range(1, n_walk + 1):
            frac = j / self.n_cont
            kap_j = self.kappa_ref + frac * (kappa_end - self.kappa_ref)
            if self.tape_continuation:
                # Every intermediate is a differentiable function of the control.
                self.problem.D_12 = exp(self.kappa_ref
                                        + frac * (self.kappa - self.kappa_ref))
            else:
                # A plain number: this walk must leave nothing on the tape.
                self.problem.D_12 = Constant(float(np.exp(kap_j)))
            PETSc.Sys.Print(f"  continuation {j}/{self.n_cont}: "
                            f"D = {np.exp(kap_j):.6f}", flush=True)
            sln = solve_forward(self.problem, sln=sln, check=False)
            self.cont_trace.append(float(np.exp(kap_j)))

        if not self.tape_continuation:
            continue_annotation()
            # The one taped solve, at kappa itself, warm-started from the walk.
            self.problem.D_12 = exp(self.kappa)
            PETSc.Sys.Print(f"  taped solve at D = {np.exp(kappa_end):.6f} "
                            f"(walk of {n_walk} untaped)", flush=True)
            sln = solve_forward(self.problem, sln=sln, check=False)
            self.cont_trace.append(float(np.exp(kappa_end)))

        obs = observe(sln, (self.P0, self.P0_input), self.field)
        misfit = 0.5 * inner(obs - self.data, obs - self.data) / self.sigma ** 2

        dkappa = self.kappa - self.kappa_prior
        reg = 0.5 * self.alpha * inner(dkappa, dkappa)

        J = self.scale * (assemble(misfit * dx)
                          + assemble(reg * dx(self.problem.mesh)))

        return ReducedFunctional(J, control,
                                 eval_cb_post=self._on_eval,
                                 derivative_cb_post=self._on_derivative)

    def _on_eval(self, value, controls):
        self.n_forward += self._solves_per_eval
        kappa = _scalar(controls)
        self.history.append({"eval": self.n_forward,
                             "J": float(value),
                             "kappa": kappa,
                             "D_12": float(np.exp(kappa))})

    def _on_derivative(self, value, derivative, controls):
        # Must RETURN the derivatives: pyadjoint uses this callback's return
        # value, not just its side effect, and raises if it gets None.
        #
        # += n_cont, not += 1. One derivative() call replays the WHOLE tape, so
        # it performs n_cont adjoint solves, not one. The old accounting was
        # right only because every reported run so far had n_cont = 1; a basin
        # cell at D_init = 8 would have reported 9 while performing 54. This is
        # the headline number for the cost comparison in README.md section IV.
        self.n_adjoint += self._solves_per_eval
        return derivative

    # -- Interface ----------------------------------------------------------

    def data_check(self, D_true):
        """Misfit RMS between this mesh's prediction at D_true and the data.

        Should sit at the noise level. Anything larger means the model and the
        data disagree for a reason other than noise -- a mismatched observation
        operator, a stale cache, or points compared in different orders. The
        last of those produced a misfit of 0.06 against a data RMS of 0.67 and a
        recovered D that was 59 percent high, with no error raised anywhere.
        """
        pause_annotation()
        saved, self.problem.D_12 = self.problem.D_12, Constant(float(D_true))
        sln = Function(self.problem.Z)
        sln.assign(self.guess)
        sln = solve_forward(self.problem, sln=sln, check=False)
        obs = observe(sln, (self.P0, self.P0_input), self.field)
        diff = obs.dat.data_ro - self.data.dat.data_ro
        self.problem.D_12 = saved
        continue_annotation()
        return float(np.sqrt(np.mean(diff ** 2)))

    def at_kappa(self, kappa):
        """An R-space Function holding kappa, the shape the control expects."""
        f = Function(self.problem.R0)
        f.assign(float(kappa))
        return f

    def at(self, D):
        """Same, given D rather than kappa = log D."""
        return self.at_kappa(np.log(D))

    def direction(self, value=1.0):
        """A perturbation direction in kappa, for taylor_test."""
        return self.at_kappa(value)

    def value(self, D=None):
        return float(self.Jhat(self.at(D))) if D else float(self.Jhat.functional)

    def gradient(self):
        return float(self.Jhat.derivative().dat.data_ro[0])

    def solve(self, tol=1e-6, max_iter=100, D_min=0.5, D_max=10.0):
        """Minimize over log D_12, bounded. Returns the recovered D_12.

        The bounds are not a convenience -- they are what keeps the optimizer
        inside the region where the forward problem has a solution at all.

        `tol` is the projected-gradient tolerance. 1e-6, not 1e-8. The honest
        reason is the second one below; the first needs a conversion it does not
        get. A gradient tolerance is not a parameter accuracy -- they differ by
        the Hessian, which is about 8e4 here (E12), so tol=1e-6 corresponds to
        roughly 1e-11 in kappa, far inside a noise-set error of 3*sigma ~ 3e-3.
        The conversion happens to be comfortable, but it has to be done, not
        asserted. The decisive reason: demanding 1e-8 made L-BFGS-B exit ABNORMAL
        on 3 of 50 cells when its line search could no longer make progress.
        Compare recovered values against a 1e-8 run before trusting this -- they
        should agree to many digits, since both are far inside the noise.

        Continuation experiments put the lower solvability limit near
        D_12 = 0.45 for this configuration: Newton takes 3-4 iterations down to
        D = 0.52, 8 at D = 0.477, and fails outright at D = 0.435, with
        ten-times-finer steps moving that edge only from 0.55 to 0.48.

        WHAT THAT NUMBER IS, stated carefully, because an earlier version of
        this docstring called it "a measured property of the problem" and it is
        not. Every continuation walk in this repository starts from the SAME
        place: kappa_ref = 0.0, D_1 * D_2 = 1.0, and D_true = 1.0 all coincide,
        and `initial_guess()` is the projected EXACT solution. So the walk
        begins at the truth with an exact initial state, and 0.45 is how far
        THIS solver, from THAT anchor, with THIS step size, gets before Newton
        stops converging. It is a property of solver-plus-anchor. A different
        anchor, a warm start from the previous iterate, or a divergence fallback
        would each move it, and none of that has been tried.

        The mechanism proposed for it -- that below the edge the Onsager drag is
        strong enough that the prescribed boundary fluxes demand
        chemical-potential gradients driving a mole fraction to zero, where
        mu = RT ln(x p) is singular -- is a hypothesis consistent with the
        failure mode, not a measurement. Distinguishing it from ordinary Newton
        stagnation needs the mole fraction minimum tracked along the walk, which
        no run has done.

        Unbounded, L-BFGS takes a first step from D = 1.2 large enough to cross
        that edge, and the run dies inside a line-search trial. Bounding keeps
        the optimizer inside the region the solver currently reaches, and the
        bounds must be reported with the result for that reason. The basin
        sweep's range is derived from 0.45, so it currently measures this
        configuration rather than the method -- see notes/results.md.
        """
        opt = minimize(self.Jhat, method="L-BFGS-B",
                       bounds=[float(np.log(D_min)), float(np.log(D_max))],
                       options={"gtol": tol, "maxiter": max_iter})
        kappa_opt = float(opt.dat.data_ro[0])
        self.Jhat(self.at_kappa(kappa_opt))
        self.kappa_opt = np.array([kappa_opt])
        return float(np.exp(kappa_opt))

    def hessian_spectrum(self, steps=(1e-2, 1e-3)):
        """Eigenvalues of the Hessian in kappa at the optimum, by central
        differences of the ADJOINT GRADIENT. Call after `solve`.

        Returns one eigenvalue array per step size, in the order of `steps`.

        With one parameter this is a single number. It becomes the
        identifiability measurement once n > 2: a near-zero eigenvalue means the
        data does not constrain that combination of diffusivities and the
        reported value is coming from the regularization instead.

        Why not `Jhat.hessian`. pyadjoint reaches it through a tangent-linear
        pass that failed in every cell of E15 and E16 with
            ConvergenceError: DIVERGED_LINEAR_SOLVE  (0 iterations)
        -- untested hypothesis: the matfree fieldsplit parameters handed to a
        solve on an already-assembled matrix. The adjoint GRADIENT, by
        contrast, is verified against finite differences to 1e-10 (E7, E10),
        so differencing it inherits that verification instead of depending on a
        path that has never once worked.

        Why two step sizes. Truncation error shrinks with h and gradient noise
        is amplified as 1/h, and the crossover is not known in advance. The
        column is computed at every step in `steps` and ALL are reported, so
        agreement between them is measured rather than assumed.

        Cost: 2 m gradient evaluations per step, each a full continuation plus
        adjoint. They are NOT inversion cost, so the solve counters and the
        iteration history are restored afterwards -- otherwise E-series cost
        baselines and convergence histories would silently include them. The
        Hessian's own cost is kept on `self.hessian_cost` instead.
        """
        kappa = np.asarray(self.kappa_opt, dtype=float)
        m = kappa.size
        saved = (self.n_forward, self.n_adjoint, len(self.history))

        def grad_at(k):
            f = Function(self.problem.R0)
            f.dat.data[:] = k
            self.Jhat(f)
            return np.array(self.Jhat.derivative().dat.data_ro, dtype=float)

        spectra = []
        for h in steps:
            H = np.zeros((m, m))
            for j in range(m):
                e = np.zeros(m)
                e[j] = h
                H[:, j] = (grad_at(kappa + e) - grad_at(kappa - e)) / (2.0 * h)
            spectra.append(np.linalg.eigvalsh(0.5 * (H + H.T)))

        grad_at(kappa)      # leave the tape at the optimum, as `solve` does

        self.hessian_cost = (self.n_forward - saved[0], self.n_adjoint - saved[1])
        self.n_forward, self.n_adjoint = saved[0], saved[1]
        del self.history[saved[2]:]
        return spectra
