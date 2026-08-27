# Open Problems

Status of `src/sosm.py`, our port of the binary SOSM forward model. Written
2026-08-04, revised 2026-08-05. Nothing below has been verified by running it,
except where a claim is attributed to Baier-Reinio and Farrell.

Context: the port copies the physics, manufactured solution, and residual blocks
from `multicomponent_code/manufactured_solution.py` essentially verbatim. It
changes exactly one thing, namely how the constants in $(\mu_1, \mu_2, p)$ are
fixed, because the original fixes them with point Dirichlet conditions plus a
Woodbury update executed inside a custom SNES convergence-test callback, and
`pyadjoint` cannot tape that. Every open problem below traces back to that one
change.

IMPORTANT, and a correction to an earlier version of this file: the analysis of
that change is NOT written up anywhere. `paper/template.tex` has no Section 2.6 --
Section 2 runs 2.1 to 2.5 and then Section 3 -- and states no propositions, lemmas
or corollaries. (It does state three theorems, all in Section 3 and all about the
Picard inverse iteration, unrelated to what follows.) An earlier version of this
file cited numbered results from that nonexistent section throughout. Every such
citation below has been replaced by the argument itself, marked as an argument.
The arguments are supported numerically by E1b, E5 and E8 in `notes/results.md`;
none of them is written up, and writing them up is itself an open problem.


## 1. RESOLVED ANALYTICALLY, UNVERIFIED NUMERICALLY: the constraint formulation

### What was wrong

We carried three real-space (`"R"`) constants $l_1, l_2, l_p$ and formed the
augmented fields $\mu_i + l_i$ and $p + l_p$, then imposed

$$\int c_1 = C_1, \qquad \int c_2 = C_2, \qquad \int (x_1 + x_2) = |\Omega|$$

as variational terms. This is singular. The argument, unwritten, is that if the
residual depends on $(\mu_i, l_i)$ only through the sum, the Jacobian has a
three-dimensional kernel at every state, so no solver configuration recovers it.

The asymmetric raw and augmented split copied from the electrolyte code does not
save it. The argument, unwritten, is that the raw fields enter only against $\mathrm{div}$
of test functions with vanishing normal trace, and

$$\int_\Omega \mathrm{div}(u_i) \, dx = \oint_{\partial\Omega} u_i \cdot n \, ds = 0$$

identically when Dirichlet data is prescribed on the whole boundary, which the
manufactured solution does. Whether the split helps in the electrolyte
configurations depends on their boundary data, which we have not checked.

### The fix that does not work

The saddle-point attachment previously proposed here, placing each multiplier in
the equation tested by its own field's constitutive test function,

    res += l_1 * y_1 * dx      # WRONG, do not use

fails for the same reason. The argument, unwritten: the two terms in the
constitutive block shift by $t \int y_1$ and $-t \int y_1$ and cancel.

### The fix that does

Section 4 of `doi:10.1137/25M1734385` carries no separate constants. The spaces
contain all constants, the auxiliary constants are absorbed into $(p_h,
\bar\mu_h)$, and $n+1$ equations from the conservation block are eliminated
because they hold identically when the test functions are constant. The integral
constraints take their place.

That degeneracy is argued, not proved, as follows. For constant $w_i$ the only surviving term is
$-M_i^{-1} \int \mathrm{div}(J_i)$, which cancels against $\int r_i$ by the
discrete compatibility condition. For constant $q$ the volume terms convert to
boundary integrals that cancel the density consistency term exactly. Both
`_fix_compatibility` and the `ds` term are therefore load-bearing.

The multipliers belong in that block:

    res += l_1 * w_1 * dx
    res += l_2 * w_2 * dx
    res += l_p * q   * dx

with raw $\mu_1, \mu_2, p$ everywhere else and the three constraints unchanged.
The argument, unwritten, is that this breaks the symmetry and that
$l_1 = l_2 = l_p = 0$ at the solution, which `check_constraints` now asserts, and
that the resulting
Jacobian is nonsingular if and only if the original's is.

The original discards one nodal row per field rather than the constant-test row.
The argument, unwritten, is that the two eliminations agree, under the hypothesis that the
conservation block has corank exactly one per field. That hypothesis is shared
with the original, since discarding a nodal row from a block of corank two would
leave the original singular as well.

### How to check

`python src/diag_nullspace.py` evaluates the residual before and after each
candidate shift. An unchanged norm is an exact symmetry and is decisive evidence
of singularity. A changed norm is not evidence of nonsingularity, and there is no
cheap sufficient test: the relevant matrix cannot be assembled monolithically
while the mixed space carries `"R"` blocks. Run it first regardless, since it
costs seconds.


## 2. UNVERIFIED: is the residual tapeable by pyadjoint?

The whole point of the reformulation is that `firedrake.adjoint` can record the
solve. That has never been tested. `src/fig02_gradient_check.py` tests it against
centered differences and `taylor_test`.

`src/diag_adjoint_rspace.py` tests the same premise on a Neumann Poisson problem
and is not blocked by item 1. It exercises the four things that must work
together, namely an `"R"` block in a mixed space, `matfree`, a Schur fieldsplit
with MUMPS on the PDE block, and an R-space Control, on a problem small enough to
debug.

**This script needs updating before it is run.** Its docstring states that it uses
the saddle-point attachment proposed in item 1, which the argument in item 1 rejects. As
a test of the adjoint machinery it remains valid, since nothing about `matfree`
plus fieldsplit plus an R-space Control depends on where the multiplier sits. As
evidence for the SOSM formulation it is now worthless. Either fix the attachment
to match the item-1 argument or delete the claim from the docstring. Do not let a pass
here be read as support for the formulation.

The `fig02` construction of `SOSMProblem` has been moved above
`continue_annotation()`. It previously sat below, so the compatibility
projections and the `T_1`, `T_2` assignments were taped, which is the hazard the
`__init__` docstring warns about, reintroduced at the call site.

Known constraint: `Function.at()` is not tapeable, so sparse point observations
have to go through `VertexOnlyMesh` interpolation. Not yet written. `fig02`
currently uses a full-field misfit, which is enough to certify the gradient
machinery but is not the observation operator the paper needs.


## 3. UNVERIFIED: is the port numerically correct?

`src/fig01_convergence.py` measures $L^2$ error against the analytic manufactured
solution. The expected rates are not $k$ across the board. Table 2 of
`doi:10.1137/25M1734385` reports that the nonlinear problem converges
suboptimally by one order in $h$: at $d=2$, $k=4$ the observed rates are near 3
for $E_p$, $E_{\bar J}$, $E_{\bar\mu}$, $E_{\nabla v}$ and $E_{\mathrm{MA}}$, and
near 4 for $E_v$ and $E_{\bar x}$. The gate as originally written would have
failed a correct port. Per-field expectations and both reference lines are now in
the script.

The paper attributes the loss to the term $(p_h, \nabla\Psi_h \cdot K_i)$ and
states that substituting the exact $\nabla\Psi$ recovers optimal rates. The
`--exact-grad-rho-inv` flag is therefore a second experiment rather than a stray
option, and two runs differing in exactly the predicted way is stronger evidence
than one table match. Confirm the substitution is applied only to that term and
not to every use of `grad_rho_inv` across the `B` and `BT` blocks, or the
comparison tests something wider than the paper's claim.

Running `multicomponent_code/manufactured_solution.py` at the same configuration
gives error magnitudes to compare against, not just rates.


## 3b. UNMEASURED: does the outer Picard inverse iteration contract?

The paper's Theorem 2 gives the local convergence criterion
$\rho(D\mathcal{T}) < 1$ for the outer loop, with

$$D\mathcal{T} = P_c F_U^{-1} F_\beta H_{\beta\beta}^{-1} G_{\bar c}.$$

It has never been evaluated, for SOSM or for anything else. `src/diag_picard.py`
evaluates it exactly for $n = 2$, and is written but UNRUN.

**The criterion is one number, not a spectral radius estimate.** $D\mathcal{T}$
factors through the parameter space, so $\mathrm{rank}\,D\mathcal{T} \le
\binom{n}{2}$, and since $\sigma(XY)\setminus\{0\} = \sigma(YX)\setminus\{0\}$ the
criterion reduces to a $\binom{n}{2} \times \binom{n}{2}$ matrix. For $n = 2$ that
is a scalar:

$$\lambda = -\frac{\langle j_\beta,\, J_{\bar c}\sigma_{\mathrm{nl}}\rangle_W}
{\langle j_\beta, j_\beta\rangle_W + R''}, \qquad \rho(D\mathcal{T}) = |\lambda|,$$

obtained from three linear solves at the self-consistent point: one linearized
tangent solve for $j_\beta$, one solve with the nonlinear Jacobian $F_U$ for
$\sigma_{\mathrm{nl}}$, and one linearized directional solve for
$J_{\bar c}\sigma_{\mathrm{nl}}$. No outer iteration has to be run to evaluate it.

**Blocking fact found while implementing it.** The Picard-linearized state carries
$J_1, J_2, v, \mu_1, \mu_2, p$ and nothing else -- the mole fractions are the
frozen data $\bar c^{(k)}$, not unknowns. So $x_1$ cannot be the step (A)
observable: $\partial\mathcal{O}/\partial\beta = 0$ identically, and with
$\alpha > 0$ step (A) returns $\beta_{\mathrm{prior}}$ every round while nothing
raises. Our inverse experiments all observe $x_1$ (`inverse.py`, `field=X1`),
which is fine for the direct inversion they perform and unusable for a Picard
loop. Any Picard run needs a different observable; `src/diag_sensitivity.py`
ranks the candidates.

**A reduction that does NOT apply here, recorded so it is not re-derived.** An
analysis in this project reduced $D\mathcal{T}$ to $(I-K)^{-1}$ applied to the
frozen sensitivity, $K$ being the Jacobian of the ordinary FORWARD Picard
iteration, concluding that the outer loop contracts iff
$\sigma(K) \subset \{\mathrm{Re}\,z < 1/2\}$ against $\{|z| < 1\}$ for forward
Picard -- hence strictly harder in the diffusive regime. That needs the
Picard-consistency identity $F_{\mathrm{lin}}(U,\beta;P_cU) \equiv F(U,\beta)$
with both residuals on the SAME space. They are not: the linearized state has 9
fields against 12, $P_c U_{\mathrm{lin}}$ does not exist, and the linearized model
cannot produce a new $\bar c$ at all, which is why step (B) exists. $K$ is
undefined for this formulation. The rank-one criterion above needs no such
identity.

**Escape hatch, if $\rho \ge 1$.** With $R = \frac{\alpha}{2}\|\beta-\beta_0\|^2$
at a zero-residual fixed point, the singular values of
$(J^TWJ + \alpha I)^{-1}J^TW^{1/2}$ are $s_i/(s_i^2+\alpha) \le 1/(2\sqrt\alpha)$,
so $\|D\mathcal{T}_\alpha\| = O(\alpha^{-1/2})$. Contraction can always be forced
by regularizing harder, at the cost of the $C_{\mathrm{reg}}\alpha$ bias. That
makes $\alpha$ a stability dial, not only a bias dial. Caveat: as $\alpha$ grows
the fixed point moves, the residual stops vanishing, and the second-order terms
return.

**Also unmeasured.** The FORWARD Picard iteration's contraction rate. The
reference code's `picard_linearized` mode freezes at the exact manufactured
solution and performs one linear solve as a discretization study; it does not
iterate. So the rate the paper's philosophy appeals to has never been observed
for SOSM either.


## 4. Smaller open items

- **Line-search fallback.** A forward solve that diverges at a trial $D_{12}$
  during a line search currently raises and halts the optimizer, rather than
  returning a large objective value. Needs a defined fallback before any
  optimization run.
- **Fieldsplit tuning.** The Schur configuration is copied from the electrolyte
  code (`fieldsplit_0` assembled into MUMPS, `fieldsplit_1` a 3x3 GMRES solve).
  It has not been tuned for our problem and its iteration counts are unknown.
  Both 3 iterations and 300 are "working" in the sense of converging, so the
  outcome here is a number rather than a verdict.
- **Memory ceiling not measured.** Because the LU now sees only the sparse PDE
  block and never the dense constraint rows, our memory profile should be much
  better than the monolithic 1-2 TB figure quoted in the original README for 3-D.
  This is an untested claim and should be benchmarked. It may widen what is
  affordable for the 3-D capstone runs.
- **3-D hex path unexercised.** `SOSMProblem` supports `d=3, mesh_type="hex"` via
  an extruded mesh, copied from the original. Nothing has run it here, but the
  original produced the $d=3$ rows of Table 2 with the same construction, so it
  is expected to work rather than to raise. Hexahedra are the configuration the
  paper uses in three dimensions, and `scripts/validate.sh` passes `tet`
  unconditionally, so any 3-D comparison against Table 2 needs that changed.
- **`variant="equispaced"` is now vestigial.** It exists in the original so that
  a degree of freedom sits exactly on `bc_point_ref`, which `FixAtPointBC`
  requires. That machinery is gone from our port. Keep the variant anyway, since
  it keeps the discrete spaces identical to the original and every removed
  difference is one fewer explanation to rule out if an A/B comparison disagrees.


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
  the flux Dirichlet BCs close over. Now computed once at construction, with a
  double-call guard, since the scaling is correct only because $T_i$ equals one
  when the auxiliary projection is taken and a second call would compound it.
  This would have silently corrupted the adjoint gradient.
- **Line search was `"basic"`.** The constitutive law evaluates $\ln(x_i p)$, so an
  overshooting Newton step gives NaN. Changed to `"l2"`, which the electrolyte
  code selects for exactly this reason.
- **`form_compiler_parameters`** belongs on `NonlinearVariationalProblem`, not on
  `NonlinearVariationalSolver`.
- **The density reciprocal block is correct as written.** `inner(1.0 / rho_inv, r)`
  divides by an unknown, but it matches Eq. (4.9d) of the paper verbatim. An
  earlier suggestion to rewrite it as `inner(1.0 - rho_inv * rho, r)` contradicted
  the source and was withdrawn.


## 6. Research-level risks

Separate from the code. See `README.md` section V. The main one is that the outer
Picard inverse iteration may fail to contract. Baier-Reinio reports that forward
Picard iteration for SOSM converges only in tame parameter settings and that
Newton is much more robust, so a Newton fallback should be planned from the start
rather than treated as a contingency. His caution concerns the forward iteration
while ours is an outer loop on the parameters, so the situations are not
identical, but the evidence bears on the same underlying system. If the measured
spectral radius exceeds one, the available responses are damping the update,
Anderson acceleration, or abandoning the outer loop for direct reduced-space
optimization.