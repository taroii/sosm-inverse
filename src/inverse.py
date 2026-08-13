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

from sosm import SOSMProblem, solve_forward

__all__ = ["sensor_points", "observer", "observe", "synthetic_data",
           "Inversion"]

# Index of x_1 in the mixed space, i.e. the field we observe.
X1 = 6


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
    """The P0 space on a VertexOnlyMesh at `points`.

    Created ONCE per mesh and reused. Two VertexOnlyMesh objects over the same
    points are still two distinct domains, so a form comparing an observation
    built on one against data held on the other raises MismatchingDomainError.
    Everything that must appear in the same form has to share this space.

    Note `mesh` is passed explicitly rather than taken from a solution: for a
    mixed function space `.mesh()` returns a MeshSequenceGeometry, which
    VertexOnlyMesh does not accept.
    """
    return FunctionSpace(VertexOnlyMesh(mesh, points), "DG", 0)


def observe(sln, P0, field=X1):
    """Differentiable point evaluation of one field onto `P0`.

    `Function.at()` is NOT tapeable and must never appear in an objective.
    Interpolation onto a VertexOnlyMesh is the supported route, and it is what
    firedrake.adjoint records.

    The returned Function lives on the vertex-only mesh, whose `dx` measure sums
    over the points, so a misfit is `assemble(... * dx)` as usual.
    """
    return assemble(interpolate(split(sln)[field], P0))


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
                 d=2, k=4, N=16, field=X1):
        self.points = points
        self.sigma = sigma
        self.field = field
        self.alpha = alpha
        self.n_forward = 0
        self.n_adjoint = 0
        self.history = []

        pause_annotation()
        self.problem = SOSMProblem(d=d, k=k, N_mesh=N, quiet=True)

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
        self.P0 = observer(self.problem.mesh, points)
        self.data = Function(self.P0)
        self.data.dat.data[:] = data
        continue_annotation()

        self.Jhat = self._build()
        # _build performs one forward solve to lay the tape, and that solve does
        # not pass through eval_cb_post. Counting it matters: the cost
        # comparison in README.md section IV is measured in forward solves.
        self.n_forward = 1

    # -- Tape ---------------------------------------------------------------

    def _build(self):
        control = Control(self.kappa)

        sln = Function(self.problem.Z)
        sln.assign(self.guess)
        sln = solve_forward(self.problem, sln=sln, check=False)

        obs = observe(sln, self.P0, self.field)
        misfit = 0.5 * inner(obs - self.data, obs - self.data) / self.sigma ** 2

        dkappa = self.kappa - self.kappa_prior
        reg = 0.5 * self.alpha * inner(dkappa, dkappa)

        J = assemble(misfit * dx) + assemble(reg * dx(self.problem.mesh))

        return ReducedFunctional(J, control,
                                 eval_cb_post=self._on_eval,
                                 derivative_cb_post=self._on_derivative)

    def _on_eval(self, value, controls):
        self.n_forward += 1
        self.history.append({"eval": self.n_forward,
                             "J": float(value),
                             "kappa": float(controls[0].dat.data_ro[0]),
                             "D_12": float(np.exp(controls[0].dat.data_ro[0]))})

    def _on_derivative(self, value, derivative, controls):
        self.n_adjoint += 1

    # -- Interface ----------------------------------------------------------

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

    def solve(self, tol=1e-8, max_iter=100):
        """Minimize. Returns the recovered D_12.

        The control is re-evaluated at the optimum before returning, so anything
        computed afterwards -- the Hessian spectrum in particular -- is taken
        there rather than wherever the last line-search trial happened to land.
        """
        opt = minimize(self.Jhat, method="L-BFGS-B",
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
