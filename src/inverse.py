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


def _cache_path(d, k, N, D_true, points):
    """Cache key for the clean observations, including the code version.

    The git SHA is in the key deliberately: any change to the forward model
    invalidates the cache automatically, which is what stops a stale fine-mesh
    solve from silently contaminating every later run.
    """
    from runlog import provenance, repo_root
    blob = json.dumps({"d": d, "k": k, "N": N, "D": D_true,
                       "pts": np.asarray(points).round(12).tolist(),
                       "sha": provenance()["git_sha"]}, sort_keys=True).encode()
    cache = repo_root() / "runs" / ".data_cache"
    return cache / (hashlib.sha256(blob).hexdigest()[:16] + ".npy")


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
    path = _cache_path(d, k, N, D_true, points)
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

    rng = np.random.default_rng(seed)
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
                 cont_max_step=0.35):
        self.points = points
        self.sigma = sigma
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

        self.Jhat = self._build()
        # _build performs n_cont forward solves to lay the tape, and those do
        # not pass through eval_cb_post. Counting them matters: the cost
        # comparison in README.md section IV is measured in forward solves.
        self.n_forward = self.n_cont

    # -- Tape ---------------------------------------------------------------

    def _build(self):
        control = Control(self.kappa)

        sln = Function(self.problem.Z)
        sln.assign(self.guess)

        # Walk from kappa_ref to kappa in n_cont equal steps, warm-starting each
        # solve from the previous. Interpolating in kappa (not D) makes the
        # steps geometric in D, which is the right spacing for a quantity
        # spanning orders of magnitude. Every intermediate is a differentiable
        # function of the control, so the whole walk is on the tape.
        for j in range(1, self.n_cont + 1):
            frac = j / self.n_cont
            kap_j = self.kappa_ref + frac * (float(self.kappa.dat.data_ro[0])
                                             - self.kappa_ref)
            self.problem.D_12 = exp(self.kappa_ref
                                    + frac * (self.kappa - self.kappa_ref))
            PETSc.Sys.Print(f"  continuation {j}/{self.n_cont}: "
                            f"D = {np.exp(kap_j):.6f}", flush=True)
            sln = solve_forward(self.problem, sln=sln, check=False)
            self.cont_trace.append(float(np.exp(kap_j)))

        obs = observe(sln, (self.P0, self.P0_input), self.field)
        misfit = 0.5 * inner(obs - self.data, obs - self.data) / self.sigma ** 2

        dkappa = self.kappa - self.kappa_prior
        reg = 0.5 * self.alpha * inner(dkappa, dkappa)

        J = assemble(misfit * dx) + assemble(reg * dx(self.problem.mesh))

        return ReducedFunctional(J, control,
                                 eval_cb_post=self._on_eval,
                                 derivative_cb_post=self._on_derivative)

    def _on_eval(self, value, controls):
        self.n_forward += self.n_cont
        kappa = _scalar(controls)
        self.history.append({"eval": self.n_forward,
                             "J": float(value),
                             "kappa": kappa,
                             "D_12": float(np.exp(kappa))})

    def _on_derivative(self, value, derivative, controls):
        # Must RETURN the derivatives: pyadjoint uses this callback's return
        # value, not just its side effect, and raises if it gets None.
        self.n_adjoint += 1
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

    def solve(self, tol=1e-8, max_iter=100, D_min=0.5, D_max=10.0):
        """Minimize over log D_12, bounded. Returns the recovered D_12.

        The bounds are not a convenience -- they are what keeps the optimizer
        inside the region where the forward problem has a solution at all.

        Continuation experiments put the lower solvability limit near
        D_12 = 0.45 for this configuration: Newton takes 3-4 iterations down to
        D = 0.52, 8 at D = 0.477, and fails outright at D = 0.435, with
        ten-times-finer steps moving that edge only from 0.55 to 0.48. Below it
        the Onsager drag is strong enough that the prescribed boundary fluxes
        demand chemical-potential gradients driving a mole fraction to zero,
        where mu = RT ln(x p) is singular.

        Unbounded, L-BFGS takes a first step from D = 1.2 large enough to cross
        that edge, and the run dies inside a line-search trial. Bounding is the
        honest fix: the search region is a measured property of the problem, not
        a tuning parameter, and reporting it is part of the result.
        """
        opt = minimize(self.Jhat, method="L-BFGS-B",
                       bounds=[float(np.log(D_min)), float(np.log(D_max))],
                       options={"gtol": tol, "maxiter": max_iter})
        kappa_opt = float(opt.dat.data_ro[0])
        self.Jhat(self.at_kappa(kappa_opt))
        return float(np.exp(kappa_opt))

    def hessian_spectrum(self):
        """Eigenvalues of the Hessian in kappa, one Hessian action per column.

        With one parameter this is a single number. It becomes the identifiability
        measurement once n > 2: a near-zero eigenvalue means the data does not
        constrain that combination of diffusivities and the reported value is
        coming from the regularization instead.
        """
        m = self.kappa.function_space().dim()
        H = np.zeros((m, m))
        for j in range(m):
            e = Function(self.problem.R0)
            e.dat.data[j] = 1.0
            H[:, j] = self.Jhat.hessian(e).dat.data_ro
        return np.linalg.eigvalsh(0.5 * (H + H.T))
