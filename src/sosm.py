"""
Binary SOSM forward model on the manufactured solution of Baier-Reinio & Farrell.

DERIVED FROM Aaron Baier-Reinio's code, used with permission:
  - multicomponent_code/manufactured_solution.py            (physics, residual, MMS)
  - multicomponent_electrolyte_code/unsteady_hull_cell_2d.py (constraint formulation)
Original papers: doi:10.1137/25M1734385 and arXiv:2510.14923.

WHAT IS COPIED, WHAT IS CHANGED
-------------------------------
Copied essentially verbatim from `manufactured_solution.py`: the constitutive
relations, the manufactured solution, and the four residual blocks (A_visc,
A_osm, B_blf, BT_blf) plus the constitutive/density/forcing terms.

Changed, deliberately, in one place: **how the constants in (mu_1, mu_2, p) are
fixed.** The original augments those spaces with constants, kills the resulting
singularity with point Dirichlet BCs (`FixAtPointBC`), and then restores the
true integral constraints via the Woodbury identity executed inside a custom
SNES convergence-test callback. That callback does raw PETSc vector surgery and
`pyadjoint` cannot tape it, which makes the original residual non-differentiable
for our purposes.

We instead follow the newer electrolyte code: carry the three constants as
real-space (`"R"`) fields in the mixed space, use the augmented combinations
`mu_i + l_i` and `p + l_p` throughout the residual, and append the three
integral constraints as ordinary variational terms. The result is a single
UFL residual solved by one `NonlinearVariationalSolver` -- fully tapeable.

The three constraints are exactly the ones the original asserts after its solve:
    int c_1 = c_1_integral,   int c_2 = c_2_integral,   int (x_1 + x_2) = |Omega|

>>> NOT YET VALIDATED. This module has never been executed -- Firedrake is not
>>> installed on the machine where it was written. `fig01_convergence.py`
>>> is the gate: it must reproduce the convergence rates of the original before
>>> anything downstream is trusted. The constraint reformulation above is the
>>> part most likely to be wrong, since it is the part that is not a copy.
"""

from firedrake import *
from firedrake.petsc import PETSc

__all__ = ["SOSMProblem", "solve_forward"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class SOSMProblem:
    """A binary SOSM manufactured-solution problem on the unit square/cube.

    The Stefan-Maxwell diffusivity D_12 is held as a real-space `Function` so
    that `firedrake.adjoint` can take it as a Control. Everything else is a
    fixed `Constant`.

    Note on the manufactured solution: it is built from *two* parameters
    D_1, D_2 with D_12 = D_1 * D_2, because the exact fields are
    mu_i = g / D_i and v_i = D_i grad(g). The exact solution therefore depends
    on the split of D_12 into (D_1, D_2), not on the product alone.

    For the inverse problem this is exactly what we want. The problem *data*
    -- the forcing terms r_1, r_2, f and the boundary data -- are generated
    once at the true (D_1, D_2) and then held fixed; they are the experimental
    configuration. Only D_12, appearing in the Onsager block A_osm, is varied.
    At D_12 = D_1 * D_2 the discrete solution recovers the manufactured
    solution up to discretization error, which gives a known ground truth.
    """

    def __init__(self, d=2, k=4, mesh_type="tet", N_mesh=8,
                 D_1=0.5, D_2=2.0, deg_max=15,
                 eta=1e-1, zeta=1e-1, gamma=1e1,
                 density_consistency=True, use_grad_rho_inv_exact=False,
                 newton_atol=1e-10, newton_max_it=10,
                 ksp_atol=1e-13, ksp_rtol=1e-13):
        assert d in (2, 3)
        assert mesh_type in ("tet", "hex")

        self.d = d
        self.k = k
        self.mesh_type = mesh_type
        self.N_mesh = N_mesh
        self.deg_max = deg_max
        self.density_consistency = density_consistency
        self.use_grad_rho_inv_exact = use_grad_rho_inv_exact
        self.newton_atol = newton_atol
        self.newton_max_it = newton_max_it
        self.ksp_atol = ksp_atol
        self.ksp_rtol = ksp_rtol

        self._build_mesh()

        # -- Physical parameters. -------------------------------------------
        # The manufactured solution requires M_1 = M_2 = 1 and RT = 1.
        self.M_1 = Constant(1.0)
        self.M_2 = Constant(1.0)
        self.RT = Constant(1.0)

        self.D_1 = Constant(D_1)
        self.D_2 = Constant(D_2)

        self.eta = Constant(eta)
        self.zeta = Constant(zeta)
        self.lame = self.zeta - (2.0 * self.eta / d)
        self.gamma = Constant(gamma)

        # The inference target, as an R-space Function so it can be a Control.
        self.R0 = FunctionSpace(self.mesh, "R", 0)
        self.D_12 = Function(self.R0, name="D_12")
        self.D_12.assign(D_1 * D_2)

        self._build_manufactured_solution()
        self._build_spaces()

        # Done ONCE, at construction, and cached. These depend only on the
        # manufactured solution -- never on D_12 -- so recomputing them per
        # solve is both wasteful and unsafe under pyadjoint: the projections
        # would be taped on every replay, and T_1/T_2 are Constants that the
        # flux Dirichlet BCs close over, so reassigning them mid-tape mutates
        # the boundary data underneath a recorded solve.
        self.r_1_d, self.r_2_d = self._fix_compatibility()

    # -- Mesh ---------------------------------------------------------------

    def _build_mesh(self):
        d, N, mt = self.d, self.N_mesh, self.mesh_type

        if d == 2:
            self.mesh = UnitSquareMesh(N, N, quadrilateral=(mt == "hex"))
            self.bc_markers = (1, 2, 3, 4)
            self.dx = dx(self.mesh, degree=self.deg_max)
            self.ds = ds(self.mesh, degree=self.deg_max)
        else:
            if mt == "tet":
                self.mesh = UnitCubeMesh(N, N, N)
                self.bc_markers = (1, 2, 3, 4, 5, 6)
                self.dx = dx(self.mesh, degree=self.deg_max)
                self.ds = ds(self.mesh, degree=self.deg_max)
            else:
                self.mesh = ExtrudedMesh(UnitSquareMesh(N, N, quadrilateral=True), N)
                self.bc_markers = (1, 2, 3, 4, "top", "bottom")
                self.dx = dx(self.mesh, degree=self.deg_max)
                self.ds = (ds_t(self.mesh, degree=self.deg_max)
                           + ds_b(self.mesh, degree=self.deg_max)
                           + ds_v(self.mesh, degree=self.deg_max))

        self.vol = assemble(1.0 * self.dx)
        self.nml = FacetNormal(self.mesh)
        self.Id = Identity(self.d)

    # -- Constitutive relations (verbatim from the original) ----------------

    def mu_relation(self, x_1, x_2, p):
        """Ideal gas law for the chemical potentials."""
        return (self.RT * ln(x_1 * p), self.RT * ln(x_2 * p))

    def conc_relation(self, x_1, x_2, p):
        """Volumetric equation of state."""
        x_1_nm = x_1 / (x_1 + x_2)
        x_2_nm = x_2 / (x_1 + x_2)
        c_tot = p / self.RT
        return (c_tot, x_1_nm * c_tot, x_2_nm * c_tot)

    # -- Manufactured solution (verbatim from the original) -----------------

    def _build_manufactured_solution(self):
        d = self.d
        M_1, M_2 = self.M_1, self.M_2
        D_1, D_2 = self.D_1, self.D_2

        x = SpatialCoordinate(self.mesh)
        if d == 2:
            g = sin(pi * x[0]) * sin(pi * x[1])
        else:
            g = sin(pi * x[0]) * sin(pi * x[1]) * sin(pi * x[2])
        self.g_ms = g

        self.mu_1_ms = g / D_1
        self.mu_2_ms = g / D_2

        c_1 = exp(self.mu_1_ms)
        c_2 = exp(self.mu_2_ms)
        self.c_1_ms, self.c_2_ms = c_1, c_2

        c_T = c_1 + c_2
        rho = (M_1 * c_1) + (M_2 * c_2)
        self.c_T_ms, self.rho_ms = c_T, rho
        self.rho_inv_ms = 1.0 / rho

        self.x_1_ms = c_1 / c_T
        self.x_2_ms = c_2 / c_T

        omega_1 = (M_1 * c_1) / rho
        omega_2 = (M_2 * c_2) / rho

        v_1 = D_1 * grad(g)
        v_2 = D_2 * grad(g)
        self.v_ms = (omega_1 * v_1) + (omega_2 * v_2)

        self.mm_1_ms = M_1 * c_1 * v_1
        self.mm_2_ms = M_2 * c_2 * v_2

        eps_v = sym(grad(self.v_ms))
        tau = (2.0 * self.eta * eps_v) + self.lame * tr(eps_v) * self.Id
        self.p_ms = c_T

        # -- Problem data. Generated once, at the true (D_1, D_2), and then
        # -- held fixed. This is the "experimental configuration".
        self.f = (grad(self.p_ms) - div(tau)) / rho
        self.r_1 = div(c_1 * v_1)
        self.r_2 = div(c_2 * v_2)

        self.g_v = self.v_ms

        # Scaled at solve time to satisfy the discrete compatibility condition.
        self.T_1 = Constant(1.0)
        self.T_2 = Constant(1.0)
        self.g_1 = self.T_1 * self.mm_1_ms
        self.g_2 = self.T_2 * self.mm_2_ms

        # The three integral constraints that replace point BCs + Woodbury.
        self.c_1_integral = Constant(assemble(c_1 * self.dx))
        self.c_2_integral = Constant(assemble(c_2 * self.dx))
        self.mfs_integral = Constant(self.vol)

    # -- Discrete spaces ----------------------------------------------------

    def _build_spaces(self):
        mesh, k, d, mt = self.mesh, self.k, self.d, self.mesh_type

        if mt == "tet":
            cell = triangle if d == 2 else tetrahedron
            flux_family, dg = "RT", "DG"
        else:
            cell = quadrilateral if d == 2 else hexahedron
            flux_family, dg = ("RTCF" if d == 2 else "NCF"), "DQ"

        var = "equispaced"

        W_1 = FunctionSpace(mesh, flux_family, k)                                  # species 1 mass flux
        W_2 = FunctionSpace(mesh, flux_family, k)                                  # species 2 mass flux
        V = VectorFunctionSpace(mesh, "CG", k)                                     # barycentric velocity
        U_1 = FunctionSpace(mesh, FiniteElement(dg, cell, k - 1, variant=var))     # chemical potential 1
        U_2 = FunctionSpace(mesh, FiniteElement(dg, cell, k - 1, variant=var))     # chemical potential 2
        P = FunctionSpace(mesh, "CG", k - 1)                                       # pressure
        X_1 = FunctionSpace(mesh, dg, k - 1)                                       # mole fraction 1
        X_2 = FunctionSpace(mesh, dg, k - 1)                                       # mole fraction 2
        R = FunctionSpace(mesh, "CG", k - 1)                                       # density reciprocal
        L = FunctionSpace(mesh, "R", 0)                                            # the three constants

        # Field order is fixed here and relied on by `split` below and by any
        # fieldsplit solver configuration. The three "R" fields come last.
        self.Z = W_1 * W_2 * V * U_1 * U_2 * P * X_1 * X_2 * R * L * L * L
        self.spaces = dict(W_1=W_1, W_2=W_2, V=V, U_1=U_1, U_2=U_2,
                           P=P, X_1=X_1, X_2=X_2, R=R, L=L)
        self.num_lagrange_mults = 3
        self.Sm = FunctionSpace(mesh, dg, k + 2)

    # -- Utilities ----------------------------------------------------------

    def _project(self, expr, space, bcs=None):
        params = {"ksp_type": "gmres",
                  "ksp_max_it": 3,
                  "ksp_convergence_test": "skip",
                  "pc_type": "lu",
                  "pc_factor_mat_solver_type": "mumps",
                  "mat_mumps_icntl_14": 105}
        return project(expr, space, bcs=bcs,
                       solver_parameters=params,
                       form_compiler_parameters={"quadrature_degree": self.deg_max})

    def _fix_compatibility(self):
        """Rescale the flux BCs so the compatibility condition holds discretely.

        Verbatim in effect from the original: T_i is chosen so that the
        integrated source matches the boundary flux at the discrete level.
        """
        # Correct ONLY on the first call: g_1 = T_1 * mm_1_ms, so if T_1 has
        # already been rescaled, aux_1 carries that factor and the new T_1
        # divides by the old one. Compounds silently.
        assert not getattr(self, "_compat_done", False), \
            "_fix_compatibility called twice; T_1/T_2 would compound"
        self._compat_done = True

        aux_1 = self._project(self.g_1, self.spaces["W_1"],
                              bcs=[DirichletBC(self.spaces["W_1"], self.g_1, self.bc_markers)])
        aux_2 = self._project(self.g_2, self.spaces["W_2"],
                              bcs=[DirichletBC(self.spaces["W_2"], self.g_2, self.bc_markers)])

        r_1_d = self._project(self.r_1, self.Sm)
        r_2_d = self._project(self.r_2, self.Sm)

        self.T_1.assign(float(self.M_1) * assemble(r_1_d * self.dx)
                        / assemble(inner(aux_1, self.nml) * self.ds))
        self.T_2.assign(float(self.M_2) * assemble(r_2_d * self.dx)
                        / assemble(inner(aux_2, self.nml) * self.ds))

        return r_1_d, r_2_d

    # -- The residual -------------------------------------------------------

    def residual(self, sln, r_1_d, r_2_d):
        """The full nonlinear SOSM residual F(U, D_12) as a single UFL form."""
        dx_, ds_ = self.dx, self.ds
        M_1, M_2, RT = self.M_1, self.M_2, self.RT
        eta, lame, gamma = self.eta, self.lame, self.gamma

        (mm_1, mm_2, v, mu_1, mu_2, p, x_1, x_2, rho_inv,
         l_1, l_2, l_p) = split(sln)
        (u_1, u_2, u, w_1, w_2, q, y_1, y_2, r,
         t_1, t_2, t_p) = TestFunctions(self.Z)

        # NO augmented fields. mu_1, mu_2, p are used raw everywhere, exactly
        # as in the original -- its spaces already contain the constants, and
        # it carries no separate scalar unknowns (verified: `Z_h` in
        # manufactured_solution.py:300 has nine fields and its `R_h` is the
        # CG density-reciprocal space, not a real-number space).
        #
        # An earlier version formed mu_i + l_i and p + l_p and used those
        # throughout. That makes (mu_i, l_i) -> (mu_i + c, l_i - c) an exact
        # symmetry, hence a singular Jacobian. See open_problems.md item 1.
        #
        # Instead l_1, l_2, l_p are multipliers occupying the degenerate rows
        # of the conservation block: the rows tested by constant w_1, w_2, q.
        #
        # Relation to the original, stated precisely. Verified at
        # manufactured_solution.py:403-405 and :530-535, it zeroes
        # `dof_index_in_mixed_space(Z_h, 3/4/5)` -- one NODAL row in each of the
        # U_1, U_2, P blocks -- and writes our three integral constraints there.
        # Same block, but a nodal row rather than the constant-test row, so the
        # two eliminations are NOT identical by inspection. Proposition 2.11 of
        # paper/template.tex proves they agree under the hypothesis that the
        # conservation block has corank exactly one per field; that hypothesis is
        # shared with the original, since discarding a nodal row from a block of
        # corank two would leave the original singular too.
        #
        # At the solution l_1 = l_2 = l_p = 0; check_constraints asserts this.

        grad_rho_inv = grad(self.rho_inv_ms) if self.use_grad_rho_inv_exact else grad(rho_inv)

        c_tot, c_1, c_2 = self.conc_relation(x_1, x_2, p)

        # -- Stokes viscous terms.
        A_visc = 2.0 * eta * inner(sym(grad(v)), sym(grad(u))) * dx_
        A_visc += lame * inner(div(v), div(u)) * dx_

        # -- Augmented Onsager transport terms. D_12 enters ONLY here.
        A_osm = (RT / ((c_1 + c_2) * self.D_12)) * (
            (c_2 / (M_1 * M_1 * c_1)) * inner(mm_1, u_1)
            + (c_1 / (M_2 * M_2 * c_2)) * inner(mm_2, u_2)
            - (1.0 / (M_1 * M_2)) * (inner(mm_1, u_2) + inner(mm_2, u_1))) * dx_
        A_osm += gamma * inner(v - (rho_inv * (mm_1 + mm_2)),
                               u - (rho_inv * (u_1 + u_2))) * dx_

        # -- Driving forces and the Stokes pressure term.
        B = (inner(p, (rho_inv * div(u_1 + u_2)) + dot(grad_rho_inv, u_1 + u_2))
             - inner(p, div(u))) * dx_
        B -= ((1.0 / M_1) * inner(mu_1, div(u_1))
              + (1.0 / M_2) * inner(mu_2, div(u_2))) * dx_

        # -- Mass-average constraint and continuity.
        BT = (inner(q, (rho_inv * div(mm_1 + mm_2)) + dot(grad_rho_inv, mm_1 + mm_2))
              - inner(q, div(v))) * dx_
        BT -= ((1.0 / M_1) * inner(w_1, div(mm_1))
               + (1.0 / M_2) * inner(w_2, div(mm_2))) * dx_

        res = A_visc + A_osm + B + BT

        # -- Thermodynamic constitutive relation.
        mu_1_cr, mu_2_cr = self.mu_relation(x_1, x_2, p)
        res += (inner(mu_1 - mu_1_cr, y_1) + inner(mu_2 - mu_2_cr, y_2)) * dx_

        # -- Multipliers, in the conservation-block rows that go degenerate for
        # constant test functions. NOT on y_1, y_2: the constitutive rows are
        # pointwise and not degenerate, so a multiplier there would be
        # unconstrained.
        res += (l_1 * w_1 + l_2 * w_2 + l_p * q) * dx_

        # -- Density reciprocal.
        res += inner(1.0 / rho_inv, r) * dx_
        res -= inner((M_1 * c_1) + (M_2 * c_2), r) * dx_

        # -- Density consistency.
        if self.density_consistency:
            res -= q * inner((rho_inv * (mm_1 + mm_2)) - v, self.nml) * ds_

        # -- Forcing.
        res -= (inner(self.f * ((M_1 * c_1) + (M_2 * c_2)), u)
                - inner(w_1, r_1_d) - inner(w_2, r_2_d)) * dx_

        # -- The three integral constraints, replacing point BCs + Woodbury.
        # Each reads  int (field - target/|Omega|) * t = 0  for t constant,
        # i.e. exactly  int field = target.
        res += (c_1 - self.c_1_integral / self.vol) * t_1 * dx_
        res += (c_2 - self.c_2_integral / self.vol) * t_2 * dx_
        res += ((x_1 + x_2) - self.mfs_integral / self.vol) * t_p * dx_

        return res

    # -- Boundary conditions and initial guess ------------------------------

    def bcs(self):
        return [DirichletBC(self.Z.sub(0), self.g_1, self.bc_markers),
                DirichletBC(self.Z.sub(1), self.g_2, self.bc_markers),
                DirichletBC(self.Z.sub(2), self.g_v, self.bc_markers)]

    def initial_guess(self):
        """L^2 projection of the exact solution, as in the original."""
        sln = Function(self.Z)
        mm_1, mm_2, v, mu_1, mu_2, p, x_1, x_2, rho_inv, l_1, l_2, l_p = sln.subfunctions

        S = self.spaces
        mm_1.assign(self._project(self.mm_1_ms, S["W_1"]))
        mm_2.assign(self._project(self.mm_2_ms, S["W_2"]))
        v.assign(self._project(self.v_ms, S["V"]))
        mu_1.assign(self._project(self.mu_1_ms, S["U_1"]))
        mu_2.assign(self._project(self.mu_2_ms, S["U_2"]))
        p.assign(self._project(self.p_ms, S["P"]))
        x_1.assign(self._project(self.x_1_ms, S["X_1"]))
        x_2.assign(self._project(self.x_2_ms, S["X_2"]))
        rho_inv.assign(self._project(self.rho_inv_ms, S["R"]))
        # Multipliers start at zero, which is also their value at the solution.
        # Re-checked after the raw-field change: the projections above are
        # unaffected, since mu_i and p now carry the full field directly rather
        # than a split against a constant.
        l_1.assign(0.0)
        l_2.assign(0.0)
        l_p.assign(0.0)

        return sln

    # -- Solver -------------------------------------------------------------

    def solver_parameters(self, monolithic=False):
        """Newton parameters: matfree + Schur fieldsplit.

        A monolithic LU is NOT available here, and this is structural rather
        than a tuning choice. Firedrake refuses to assemble a monolithic AIJ
        matrix for a mixed space containing `"R"` blocks:

            ValueError: Monolithic matrix assembly not supported for systems
                        with R-space blocks

        An `"R"` field couples to every degree of freedom, so its rows and
        columns are dense, and a sparse monolithic format cannot represent
        them. This is precisely the difficulty the original code met and
        answered with the Woodbury identity plus hand-written PETSc: the dense
        constraint rows were held outside the sparse matrix and folded back in
        by a rank-3 update. Our reformulation keeps the constraints in the
        form, so the same difficulty reappears here and has to be answered a
        different way -- the electrolyte code's answer, which is to never
        assemble the whole operator at all.

        The configuration below is ported from
        `multicomponent_electrolyte_code/unsteady_hull_cell_2d.py`. The matrix
        is `matfree`; a Schur fieldsplit separates the nine PDE fields (split
        0) from the three real-space constants (split 1). Split 0 is assembled
        via `firedrake.AssembledPC` and handed to MUMPS -- so MUMPS still does
        the heavy lifting, but on the sparse block only, never on the dense
        constraint rows. Split 1 is tiny (3x3) and falls to a few GMRES
        iterations.

        A useful consequence: the memory profile is the electrolyte code's, not
        the 1-2 TB monolithic one, because the LU never sees the full system.
        """
        if monolithic:
            raise NotImplementedError(
                "monolithic LU cannot assemble R-space blocks; see this docstring")

        n_fields = self.Z.num_sub_spaces()
        n_pde = n_fields - self.num_lagrange_mults

        split_0 = ",".join(str(i) for i in range(n_pde))
        split_1 = ",".join(str(i) for i in range(n_pde, n_fields))

        return {"snes_type": "newtonls",
                # "l2", not "basic". The electrolyte code chooses this with the
                # comment "Prevent the fractions from becoming negative", and
                # the same hazard is ours: the constitutive law evaluates
                # ln(x_i * p), so a Newton step that overshoots x_i or p below
                # zero produces NaN and kills the solve rather than recovering.
                # The original used "basic", but it also ran its own
                # convergence test inside the Woodbury callback.
                "snes_linesearch_type": "l2",
                "snes_monitor": "",
                "snes_converged_reason": "",
                "snes_atol": self.newton_atol,
                "snes_rtol": 0.0,
                "snes_stol": 0.0,
                "snes_max_it": self.newton_max_it,

                "mat_type": "matfree",
                "ksp_type": "fgmres",
                "ksp_gmres_cgs_refinement_type": "refine_always",
                "ksp_atol": self.ksp_atol,
                "ksp_rtol": self.ksp_rtol,
                "ksp_converged_reason": "",

                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "schur",
                "pc_fieldsplit_schur_fact_type": "full",
                "pc_fieldsplit_0_fields": split_0,
                "pc_fieldsplit_1_fields": split_1,

                # Split 0: the nine PDE fields. Assembled, then MUMPS.
                "fieldsplit_0": {
                    "ksp_type": "preonly",
                    "pc_type": "python",
                    "pc_python_type": "firedrake.AssembledPC",
                    "assembled": {
                        "ksp_type": "gmres",
                        "ksp_max_it": 3,
                        "ksp_atol": self.ksp_atol,
                        "ksp_rtol": self.ksp_rtol,
                        "pc_type": "lu",
                        "pc_factor_mat_solver_type": "mumps",
                        "mat_mumps_icntl_14": 120,
                    },
                },

                # Split 1: the three real-space constants. Tiny.
                "fieldsplit_1": {
                    "ksp_type": "gmres",
                    "ksp_max_it": self.num_lagrange_mults,
                    "ksp_atol": self.ksp_atol,
                    "ksp_rtol": self.ksp_rtol,
                }}

    def check_constraints(self, sln):
        """Return the three constraint residuals and the three multipliers.

        Returns numbers. Does not raise and does not judge them -- the caller
        decides what a given magnitude means.
        """
        _, _, _, _, _, p, x_1, x_2, _, l_1, l_2, l_p = sln.subfunctions
        _, c_1, c_2 = self.conc_relation(x_1, x_2, p)

        errs = {
            "int_c_1": abs(assemble(c_1 * self.dx) - float(self.c_1_integral)),
            "int_c_2": abs(assemble(c_2 * self.dx) - float(self.c_2_integral)),
            "int_mfs": abs(assemble((x_1 + x_2) * self.dx) - float(self.mfs_integral)),
            "l_1": float(l_1.dat.data_ro[0]),
            "l_2": float(l_2.dat.data_ro[0]),
            "l_p": float(l_p.dat.data_ro[0]),
        }
        return errs

    def errors(self, sln):
        """L^2 errors against the manufactured solution."""
        mm_1, mm_2, v, mu_1, mu_2, p, x_1, x_2, rho_inv, l_1, l_2, l_p = sln.subfunctions

        def l2(a, b):
            return sqrt(assemble(inner(a - b, a - b) * self.dx))

        out = {
            "mu_1": l2(self.mu_1_ms, mu_1 + l_1),
            "mu_2": l2(self.mu_2_ms, mu_2 + l_2),
            "grad_mu_1": l2(grad(self.mu_1_ms), grad(mu_1)),
            "grad_mu_2": l2(grad(self.mu_2_ms), grad(mu_2)),
            "p": l2(self.p_ms, p + l_p),
            "grad_p": l2(grad(self.p_ms), grad(p)),
            "mm_1": l2(self.mm_1_ms, mm_1),
            "mm_2": l2(self.mm_2_ms, mm_2),
            "div_mm_1": l2(div(self.mm_1_ms), div(mm_1)),
            "div_mm_2": l2(div(self.mm_2_ms), div(mm_2)),
            "v": l2(self.v_ms, v),
            "grad_v": l2(grad(self.v_ms), grad(v)),
            "x_1": l2(self.x_1_ms, x_1),
            "x_2": l2(self.x_2_ms, x_2),
        }

        # Combined per-species norms, as Table 2 of the paper defines them:
        #   E_J   = sqrt(sum_i ||J_i - J_h,i||^2)
        #   E_mu  = sqrt(sum_i ||mu_i - mu_h,i||^2)
        #   E_x   = sqrt(sum_i ||x_i - x_h,i||^2)
        # and the mass-average constraint error
        #   E_MA  = ||v_h - Psi_h sum_i J_h,i||
        out["E_J"] = (float(out["mm_1"]) ** 2 + float(out["mm_2"]) ** 2) ** 0.5
        out["E_mu"] = (float(out["mu_1"]) ** 2 + float(out["mu_2"]) ** 2) ** 0.5
        out["E_x"] = (float(out["x_1"]) ** 2 + float(out["x_2"]) ** 2) ** 0.5

        ma = v - (rho_inv * (mm_1 + mm_2))
        out["E_MA"] = sqrt(assemble(inner(ma, ma) * self.dx))

        return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def solve_forward(problem, D_12=None, sln=None, monolithic=False, check=True):
    """Solve the nonlinear SOSM system for a given D_12.

    Parameters
    ----------
    problem : SOSMProblem
    D_12    : float, optional. If given, overwrites `problem.D_12` first.
              Leave as None when D_12 is an active pyadjoint Control.
    sln     : Function, optional. Initial guess; defaults to the projected
              manufactured solution. Pass the previous solution to warm-start
              a continuation sweep.
    check   : compute the constraint residuals and multipliers after solving
              and store them on `problem.constraint_errors`. They are recorded,
              not judged.

    Returns the solution Function on the mixed space.
    """
    if D_12 is not None:
        problem.D_12.assign(D_12)

    # r_1_d, r_2_d, T_1, T_2 were computed once at construction -- see __init__.
    if sln is None:
        sln = problem.initial_guess()

    F = problem.residual(sln, problem.r_1_d, problem.r_2_d)
    # form_compiler_parameters belongs on the problem, not the solver.
    prob = NonlinearVariationalProblem(
        F, sln, bcs=problem.bcs(),
        form_compiler_parameters={"quadrature_degree": problem.deg_max})
    solver = NonlinearVariationalSolver(
        prob,
        solver_parameters=problem.solver_parameters(monolithic=monolithic))

    solver.solve()

    problem.constraint_errors = problem.check_constraints(sln) if check else None

    return sln
