# Open Problems

Status of `src/sosm.py`, our port of the binary SOSM forward model. Written
2026-08-04. Nothing below the "Resolved" section has been verified by running it.

Context: the port copies the physics, manufactured solution, and residual blocks
from `multicomponent_code/manufactured_solution.py` essentially verbatim. It
changes exactly one thing -- how the constants in $(\mu_1, \mu_2, p)$ are fixed --
because the original fixes them with point Dirichlet conditions plus a Woodbury
update executed inside a custom SNES convergence-test callback, and `pyadjoint`
cannot tape that. Every open problem below traces back to that one change.


## 1. BLOCKING: the constraint formulation is probably rank-deficient

We carry three real-space (`"R"`) constants $l_1, l_2, l_p$ in the mixed space and
form the augmented fields $\mu_i + l_i$ and $p + l_p$, then impose

$$\int c_1 = C_1, \qquad \int c_2 = C_2, \qquad \int (x_1 + x_2) = |\Omega|$$

as variational terms. These are the same three constraints the original asserts
after its Woodbury update.

The problem: if the augmented field is used everywhere, then

$$(\mu_i, l_i) \mapsto (\mu_i + c,\; l_i - c)$$

leaves the entire residual unchanged. That is an exact three-dimensional
symmetry, hence a singular Jacobian, and no solver configuration can fix it.

The electrolyte code avoids this by using the fields asymmetrically -- raw $p$ in
the Stokes operator (`inner(p, div(u))`) and $p + l_p$ only where an absolute
thermodynamic value is required (`mat_X(y_e_nm, p + l_p)`). We have matched that.

**Why it likely still fails for us.** That asymmetry only helps if the raw field
survives into an equation that the shift actually changes. Our manufactured
solution prescribes Dirichlet data for the fluxes and the velocity on the whole
boundary, so the test functions have zero normal trace and

$$\int_\Omega \mathrm{div}(u_i) \, dx = \oint_{\partial\Omega} u_i \cdot n \, ds = 0$$

identically. The raw-field terms enter only against $\mathrm{div}(u_i)$, so they
contribute nothing and the symmetry survives. The electrolyte Hull cell has mixed
boundary conditions and does not have this degeneracy, so the fix may simply not
transfer to our problem.

**How to check.** `python src/diag_nullspace.py` evaluates the residual before and
after each candidate shift. An unchanged norm is an exact symmetry; there is
nothing to interpret. Runs in seconds on a 2x2 mesh.

**Proposed fix, reasoned but unverified.** Drop the raw/augmented split and use
the textbook saddle point instead: put each multiplier into the equation tested by
its own field's test function,

    res += l_1 * y_1 * dx                          # multiplier
    res += (c_1 - C_1 / vol) * t_1 * dx            # constraint

Physically $l_1$ is then the reference-state offset in the chemical potential law
$\mu_1 = RT \ln(x_1 p)$, which is precisely the gauge freedom that $\int c_1 = C_1$
is there to remove. This is the piece that most needs a second pair of eyes.


## 2. UNVERIFIED: is the residual tapeable by pyadjoint?

The whole point of the reformulation is that `firedrake.adjoint` can record the
solve. That has never been tested. `src/fig02_gradient_check.py` tests it against
centered differences and `taylor_test`. Blocked behind item 1.

Known constraint: `Function.at()` is not tapeable, so sparse point observations
have to go through `VertexOnlyMesh` interpolation. Not yet written -- `fig02`
currently uses a full-field misfit, which is enough to certify the gradient
machinery but is not the observation operator the paper needs.


## 3. UNVERIFIED: is the port numerically correct?

`src/fig01_convergence.py` measures $L^2$ error against the analytic manufactured
solution and should show rates near $k$. Also blocked behind item 1. Running
`multicomponent_code/manufactured_solution.py` at the same configuration gives
error magnitudes to compare against, not just rates.


## 4. Smaller open items

- **Line-search fallback.** A forward solve that diverges at a trial $D_{12}$
  during a line search currently raises and halts the optimizer, rather than
  returning a large objective value. Needs a defined fallback before any
  optimization run.
- **Fieldsplit tuning.** The Schur configuration is copied from the electrolyte
  code (`fieldsplit_0` assembled into MUMPS, `fieldsplit_1` a 3x3 GMRES solve).
  It has not been tuned for our problem and its iteration counts are unknown.
- **Memory ceiling not measured.** Because the LU now sees only the sparse PDE
  block and never the dense constraint rows, our memory profile should be much
  better than the monolithic 1-2 TB figure quoted in the original README for 3-D.
  This is an untested claim and should be benchmarked; it may widen what is
  affordable for the 3-D capstone runs.
- **3-D hex path unexercised.** `SOSMProblem` supports `d=3, mesh_type="hex"` via
  an extruded mesh, copied from the original, but nothing has run it.


## 5. Resolved

Kept for context, since these were live problems and their fixes shape the code.

- **Monolithic assembly is impossible.** Firedrake raises
  `ValueError: Monolithic matrix assembly not supported for systems with R-space
  blocks`. An `"R"` field couples to every DOF, so its rows and columns are dense.
  This is the same difficulty the original answered with Woodbury, relocated
  rather than removed by our reformulation. Answer: `mat_type: matfree` plus a
  Schur fieldsplit, ported from the electrolyte code, so MUMPS factorizes only
  the sparse block.
- **Compatibility scaling was recomputed inside the tape.** $T_1, T_2$ and the
  projected source terms depend only on the manufactured solution, never on
  $D_{12}$, but were recomputed on every `solve_forward`. Inside an annotated
  region this tapes the projections on every replay and mutates `Constant`s that
  the flux Dirichlet BCs close over. Now computed once at construction. This
  would have silently corrupted the adjoint gradient.
- **Line search was `"basic"`.** The constitutive law evaluates $\ln(x_i p)$, so an
  overshooting Newton step gives NaN. Changed to `"l2"`, which the electrolyte
  code selects for exactly this reason.
- **`form_compiler_parameters`** belongs on `NonlinearVariationalProblem`, not on
  `NonlinearVariationalSolver`.


## 6. Research-level risks

Separate from the code. See `README.md` section V. The main one is that the outer
Picard inverse iteration may fail to contract: Baier-Reinio reports that forward
Picard iteration for SOSM converges only in tame parameter regimes and that Newton
is much more robust, so a Newton fallback should be planned from the start rather
than treated as a contingency.
