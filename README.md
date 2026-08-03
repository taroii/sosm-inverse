# Parameter Inference for the Stokes-Onsager-Stefan-Maxwell Equations: A Finite Element Approach

This is the working repostory for the paper *Parameter Inference for the Stokes-Onsager-Stefan-Maxwell Equations: A Finite Element Approach* by BruinML. 

We propose a finite-element parameter inference approach for the SOSM equations. 

## I. Objective

1. To demonstrate that adjoint-based optimization recovers the Stefan--Maxwell diffusivities $D_{ij}$ from sparse, noisy observations of a nonlinear SOSM forward solve.

2. To establish whether the outer Picard inverse iteration contracts in practice, and if so, whether it costs fewer forward solves than direct reduced-space optimization over the full nonlinear residual.

These results must occur in a setting that has cross-diffusion and a concentration-dependent transport matrix. 


## II. Getting Started

The forward model is built on [Firedrake](https://www.firedrakeproject.org), which is not
installable with `pip install -r requirements.txt` — it builds its own PETSc. Create the
environment, then follow Firedrake's own
[install instructions](https://www.firedrakeproject.org/install.html) inside it:

```bash
conda create -n sosm-inverse python=3.11   # check the version against the install page
conda activate sosm-inverse
# ... Firedrake install, per the link above ...
```

Firedrake must be built with [Netgen](https://www.firedrakeproject.org/demos/netgen_mesh.py.html)
and [Irksome](https://www.firedrakeproject.org/Irksome/) support to run the reference
implementations; `tabulate` is also required. `pyadjoint` needs no separate install — it
ships with Firedrake as `firedrake.adjoint`.

Verify the install before running anything, in particular that MPI is not duplicated:

```bash
python -c "from firedrake import *; print('ok')"
mpiexec -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.rank)"
```

The second command must print `0` and `1`. Two zeros means two MPI stacks are linked into
the same process, and every subsequent "parallel" run will silently be serial.

(Optional) Clone the two reference implementations:

```bash
git clone https://bitbucket.org/abaierr/multicomponent_code.git
git clone https://bitbucket.org/abaierr/multicomponent_electrolyte_code.git
```

These repositories contain the original code we build off of, but are unnecessary to obtain the results of our paper. I.e. this repository is fully self-contained and doesn't require cloning any other repositories. 

## III. System Specifications for Reported Runs

For all of the experiments and results we report in our paper, we use a dedicated Linux server with the following specifications:

| **Component** | **Details** |
|--------------|-------------|
| **Machine**  | Lenovo ThinkStation P3 Tower Gen 2 |
| **CPU**      | Intel Core Ultra 9 285K (24 cores, up to 7.3 GHz) |
| **GPU**      | NVIDIA RTX 4000 Ada Generation (Lovelace) |
| **RAM**      | 48 GB |
| **Storage**  | 1.86 TB NVMe SSD + 2×10.9 TB HDD (≈ 23.7 TB total) |
| **OS**       | Ubuntu 24.04.2 LTS (Noble Numbat) |
| **Driver**   | NVIDIA 580.126.09 |


## IV. Desired Results

- Adjoint gradient verified against centered finite differences at every new configuration, as opposed to once at the outset.
- Recovery of known diffusivities from synthetic data, reported across species count $n$, mesh size $h$, and noise level $\sigma$.
- Convergence history of the outer iteration, reporting $\|\beta^{(k+1)} - \beta^{(k)}\|$, $\|\bar c^{(k+1)} - \bar c^{(k)}\|$, $\|\beta^{(k)} - \beta^\star\|$, and the data misfit.
- Numerical estimate of $\rho(D\mathcal{T}(\bar c^\star))$ from the assembled matrices, compared against the contraction hypothesis $q < 1$. 
- Mesh independence of the recovered parameter under refinement, testing the error decomposition.
- Error scaling with $\sigma$, showing that the estimate reaches the noise floor rather than a systematic bias.
- Basin of attraction, measured by sweeping initial guesses over several orders of magnitude.
- Head-to-head cost comparison against direct L-BFGS on the full reduced objective, counted in nonlinear forward solves and wall-clock computation time (reported alongside system specs of our server).
- Sensitivity spectrum of the Gauss-Newton Hessian, indicating which diffusivities the observation operator determines and which the regularization is supplying.
- Data generated on a finer mesh or higher polynomial degree than the inversion mesh, avoiding an inverse crime.
- Fixed seeds and versioned configuration files for every reported run. Ideally, high seed runs (at least 10 or something) to reduce variance of results. 

## V. Anticipated Hurdles 

- The biggest problem is that the outer Picard iteration may fail to contract. Forward Picard iteration for SOSM converges only for tame parameter values. 
- The nonlinear forward solve may diverge at trial parameter values proposed during a line search, which halts the optimizer rather than returning a large objective value. Failed solves need a defined fallback.
- Differentiating through the existing forward implementation with pyadjoint may not work without modification, particularly where the Woodbury identity and custom PETSc code handle the dense constraint rows. The electrolyte repository avoids Woodbury and is the better starting point.
- Some $D_{ij}$ may be weakly determined by any realistic observation operator. Regularization will then dominate those components and produce a biased estimate that appears converged, so the sensitivity spectrum must be reported alongside the recovered values.
- Cost becomes limiting in three dimensions, where each objective evaluation requires a full nonlinear solve.
- The Onsager matrix degenerates as any $c_i$ approaches zero, so near-trace species will make the inverse problem ill-conditioned.
- Positivity of the diffusivities and the semidefinite structure of $M$ must survive the parameter updates. The logarithmic reparameterization handles the first. The second requires attention once concentration-dependent diffusivities enter.

## VI. References

Our forward model is adapted from the following works of Baier-Reinio and coauthors,
whose reference implementations are the two repositories cloned in §II. The SOSM
discretization we invert is that of `doi:10.1137/25M1734385`; the constraint formulation
we use to make the residual differentiable is taken from `arXiv:2510.14923`.

```bibtex
@article{baier2026high,
  title   = {High-Order Finite Element Methods for Three-Dimensional
             Multicomponent Convection-Diffusion},
  author  = {Baier-Reinio, Aaron and Farrell, Patrick E.},
  journal = {SIAM Journal on Scientific Computing},
  volume  = {48},
  number  = {2},
  pages   = {A540--A567},
  year    = {2026},
  doi     = {10.1137/25M1734385}
}

@article{baier2025electroneutral,
  title   = {Finite element methods for electroneutral multicomponent
             electrolyte flows},
  author  = {Baier-Reinio, Aaron and Farrell, Patrick E. and Monroe, Charles W.},
  journal = {arXiv preprint arXiv:2510.14923},
  year    = {2025}
}
```

Archived software versions for the first paper are on Zenodo at
[`10.5281/zenodo.16416180`](https://doi.org/10.5281/zenodo.16416180); code and data for
its reported results are at <https://zenodo.org/records/16416181>.
