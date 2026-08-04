#!/usr/bin/env bash
#
# The validation gate. Run this FIRST on the server, before any experiment.
#
# The port in src/sosm.py has never been executed. It replaces the original's
# point BCs + Woodbury constraint handling with real-space constants and three
# integral constraints, so that the residual is differentiable. Two things must
# be established before anything downstream means anything:
#
#   1. the port still solves the SOSM system correctly  (fig01, convergence)
#   2. firedrake.adjoint can tape it                    (fig02, gradient)
#
# Step 0 runs Aaron's ORIGINAL code at the same configuration. That gives the
# ground-truth convergence table to compare our fig01 against. Do not skip it:
# "the rates look plausible" is not the same as "the rates match his".
#
# Usage:
#     tmux new -s validate
#     source scripts/env.sh
#     bash scripts/validate.sh 2>&1 | tee validate.log

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

D=${D:-2}
K=${K:-4}
N0=${N0:-8}
LOOPS=${LOOPS:-4}

echo "=============================================================="
echo " STEP 0 -- ground truth from the original implementation"
echo "=============================================================="
# OPTIONAL. fig01 already measures error against the analytic manufactured
# solution, so the port can be validated without this. Running the original
# adds a comparison of error magnitudes rather than just rates, and -- the real
# reason -- leaves a known-good implementation on the machine to A/B against if
# STEP 1 comes out wrong.
if [[ -f multicomponent_code/manufactured_solution.py ]]; then
    # Args: d k mesh_type picard_linearized N_mesh_initial n_loops
    /usr/bin/time -v python multicomponent_code/manufactured_solution.py \
        "$D" "$K" tet False "$N0" "$LOOPS"
else
    echo " SKIPPED -- multicomponent_code/ not present."
    echo " The gate still works: STEP 1 compares against the analytic solution."
    echo " Clone it only if STEP 1 fails and you want something to A/B against:"
    echo "   git clone https://bitbucket.org/abaierr/multicomponent_code.git"
fi

echo
echo "=============================================================="
echo " STEP 1 -- our port, same configuration (fig01)"
echo "=============================================================="
echo " Compare the rates below against STEP 0. They must match."
/usr/bin/time -v python src/fig01_convergence.py \
    --d "$D" --k "$K" --N0 "$N0" --loops "$LOOPS"

echo
echo "=============================================================="
echo " STEP 2 -- adjoint gradient (fig02)"
echo "=============================================================="
echo " Small k and N deliberately: this is a correctness check, not a"
echo " production run, and it does many forward solves."
/usr/bin/time -v python src/fig02_gradient_check.py \
    --d "$D" --k 3 --N 8

echo
echo "=============================================================="
echo " Gate passed only if:"
echo "   - STEP 1 rates are ~= k  (and match STEP 0, if it ran)"
echo "   - STEP 2 shows a V-shaped error curve and taylor_test ~= 2.0"
echo " Record the outcome in notes/todo.md before moving on."
echo "=============================================================="
