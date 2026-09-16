#!/usr/bin/env bash
#
# Drive the inversion grid. One process per run, several at a time.
#
# Concurrency is bounded by MEMORY, not cores, and it is now derived PER MESH
# rather than taken from one global JOBS. An earlier version of this header
# claimed "JOBS defaults to 6, which is safe up to N=64". It is not, and the
# mesh axis is the only one that reaches N=64, which is why that axis was the
# one that died. The 5.822 GB figure it relied on is E9's, and E9 is
# scripts/bench.sh measuring ONE FORWARD SOLVE. An inversion holds more: the
# pyadjoint tape retains every continuation state plus adjoint workspace, for
# every L-BFGS iteration. Six of them is 34.9 GB of forward solves alone
# against 48 GB, before the tape, the OS and the page cache.
#
# Every run writes its own directory, so runs are independent and the sweep can
# be interrupted and restarted. runlog skips nothing, though -- re-running
# repeats work, so narrow the axes rather than re-running the lot.
#
# Usage:
#     tmux new -s sweep
#     source scripts/env.sh
#     systemd-run --user --scope -p MemoryMax=40G bash scripts/sweep.sh mesh
#     bash scripts/sweep.sh noise
#     N=16 bash scripts/sweep.sh basin
#     ANCHORS="0.6 2.0 4.0" bash scripts/sweep.sh anchor
#
# The systemd-run wrapper is not decoration. Ubuntu 24.04 enables systemd-oomd,
# which kills an entire cgroup rather than one process, and an unwrapped sweep
# shares its cgroup with the tmux server -- so the session disappears instead of
# one job failing. Running the sweep in its own scope with a hard MemoryMax
# confines the kill to the sweep and leaves tmux, and the log, alive.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

AXIS="${1:-noise}"
JOBS="${JOBS:-6}"          # ceiling only; the memory budget usually binds first
MEM_BUDGET_GB="${MEM_BUDGET_GB:-36}"
MEM_SAFETY="${MEM_SAFETY:-2.0}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
K="${K:-4}"
N="${N:-16}"
D="${D:-2}"
D_INIT="${D_INIT:-1.2}"
# The paper's prototype puts the prior ABOVE the truth (1.4286x) and the initial
# guess further above (3.4286x), and attributes its residual bias to the prior's
# pull. We keep the prior ratio; D_INIT stays at 1.2 for the noise and mesh axes
# because the basin axis is where the initial guess is the variable under study.
# Set D_INIT=3.4286 to mirror the prototype exactly -- verify one run first, it
# needs four continuation steps per objective evaluation rather than one.
D_PRIOR="${D_PRIOR:-1.4286}"
ANCHORS="${ANCHORS:-0.6 2.0 4.0}"
LOG=sweep-$AXIS.log

run_one() {
    python src/invert.py "$@" >>"$LOG" 2>&1 \
        && echo "ok   $*" || echo "FAIL $*"
}

# Peak RSS of one k=4 FORWARD solve at this N, from E9. An inversion holds more
# and by how much has never been measured, which is what MEM_SAFETY stands in
# for. Replace both with a measurement as soon as one is available: runlog
# records peak_rss_gb in every run directory, so the first completed N=64 cell
# settles it. Until then the safety factor is a guess and is labelled as one.
footprint_gb() {
    case "$1" in
        8)  echo 0.395 ;;
        16) echo 0.632 ;;
        32) echo 1.634 ;;
        64) echo 5.822 ;;
        *)  echo 5.822 ;;   # unmeasured N: assume the largest measured
    esac
}

jobs_for() {
    awk -v b="$MEM_BUDGET_GB" -v f="$(footprint_gb "$1")" \
        -v s="$MEM_SAFETY" -v cap="$JOBS" \
        'BEGIN { n = int(b / (f * s)); if (n < 1) n = 1; if (n > cap) n = cap;
                 print n }'
}

# Run one group of cells, all at the same mesh, at a concurrency that fits the
# budget. Grouping matters for the mesh axis: a single global -P would apply
# the N=8 job count to the N=64 cells.
run_group() {
    local n="$1" j
    j=$(jobs_for "$n")
    echo "N=$n: $j concurrent (budget ${MEM_BUDGET_GB} GB, \
$(footprint_gb "$n") GB forward x $MEM_SAFETY)"
    xargs -P "$j" -I{} bash -c 'run_one {}'
}

# Warm the clean-data cache serially before fanning out. The fine-mesh solve is
# identical across seeds, but if N jobs start together they all miss the cache
# and each builds an 11 GB solve at once. One cheap serial run first, then the
# rest hit the cache and hold only their own inversion.
# Returns non-zero on failure, so a caller can decline to fan out into cold
# starts. Doubles as the per-configuration adjoint check README section IV asks
# for, since --check-gradient verifies the gradient at the configuration given.
warm_cache() {
    echo "warming data cache (one fine-mesh solve)..."
    if python src/invert.py "$@" --check-gradient >>"$LOG" 2>&1; then
        echo "cache warm"
    else
        echo "WARN: cache warm failed, see $LOG"
        return 1
    fi
}
export -f run_one
export LOG

# The log is appended, so mark where each sweep begins. Without this, grepping
# for failures mixes this run with every previous one -- an earlier diagnosis
# counted 60 solver failures that mostly belonged to a superseded sweep.
{ echo; echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ)  axis=$AXIS  K=$K N=$N D=$D \
D_INIT=$D_INIT D_PRIOR=$D_PRIOR  sha=$(git rev-parse --short HEAD) ==="; } >>"$LOG"

warm_cache --k "$K" --N "$N" --d "$D" --sigma 1e-3 --seed 0 \
           --D-init "$D_INIT" --D-prior "$D_PRIOR"

cells() {
case "$AXIS" in
    noise)
        # Recovery error against noise level. Includes the baseline cell.
        for s in 1e-4 3e-4 1e-3 3e-3 1e-2; do
            for seed in $SEEDS; do
                echo --sigma "$s" --seed "$seed" --k "$K" --N "$N" --d "$D" \
                     --D-init "$D_INIT" --D-prior "$D_PRIOR"
            done
        done
        ;;
    mesh)
        # Recovery error against inversion mesh, data mesh held fixed. Emits
        # only the N passed in $1; the driver below walks the meshes so each
        # gets its own concurrency.
        for seed in $SEEDS; do
            echo --N "$1" --k "$K" --seed "$seed" --sigma 1e-3 --d "$D" \
                 --D-init "$D_INIT" --D-prior "$D_PRIOR"
        done
        ;;
    basin)
        # Starting guesses spanning the range the SOLVER currently reaches, not
        # four orders of magnitude. Continuation fails below D ~ 0.45 here, so
        # 0.01 and 0.1 would report failures rather than a wide basin.
        #
        # Read the result accordingly: 0.45 is not a property of the problem.
        # Every continuation walk starts at kappa_ref = 0, which coincides with
        # D_true = 1, from the projected exact solution -- so the edge is where
        # this solver stops converging when walking away from the truth, and it
        # would move under a different anchor or a divergence fallback. This
        # sweep therefore measures the basin of THIS configuration. See the
        # `solve` docstring in src/inverse.py. The upper edge is unmeasured;
        # 8.0 sits inside the default bound.
        for init in 0.6 0.8 1.5 2.5 4.0 8.0; do
            for seed in $SEEDS; do
                echo --D-init "$init" --seed "$seed" --k "$K" --N "$N" \
                     --sigma 1e-3 --d "$D" --D-prior "$D_PRIOR"
            done
        done
        ;;
    anchor)
        # Recovery at a true diffusivity away from 1. Every other axis has
        # D_true = 1, which coincides with the continuation anchor
        # kappa_ref = 0 and with D_1 * D_2, so every walk starts at the truth
        # from the exact solution. Emits only the D_true passed in $1. D_init
        # and D_prior keep the same RATIOS to the truth as the other axes,
        # 1.2x and 1.4286x, so the only thing that changes is the truth.
        local d_init d_prior
        d_init=$(awk -v t="$1" 'BEGIN { printf "%.6g", 1.2 * t }')
        d_prior=$(awk -v t="$1" 'BEGIN { printf "%.6g", 1.4286 * t }')
        for seed in $SEEDS; do
            echo --D-true "$1" --D-init "$d_init" --D-prior "$d_prior" \
                 --seed "$seed" --k "$K" --N "$N" --sigma 1e-3 --d "$D"
        done
        ;;
    *)
        echo "unknown axis: $AXIS  (noise|mesh|basin|anchor)" >&2
        exit 2
        ;;
esac
}

# Smallest mesh first, so a budget that is wrong shows up on the cheap cells
# rather than after hours of work.
if [ "$AXIS" = mesh ]; then
    for n in 8 16 32 64; do
        cells "$n" | run_group "$n"
    done
elif [ "$AXIS" = anchor ]; then
    # The data cache key includes D_true, so the warm-up above (at D_true = 1)
    # covers none of these. Warm each truth serially first: fanning out cold
    # would start several 11 GB fine-mesh solves at once, which is the
    # out-of-memory crash the per-mesh budget exists to prevent. If a warm-up
    # fails there is no data for that truth, so skip it rather than fan out.
    for t in $ANCHORS; do
        d_init=$(awk -v t="$t" 'BEGIN { printf "%.6g", 1.2 * t }')
        d_prior=$(awk -v t="$t" 'BEGIN { printf "%.6g", 1.4286 * t }')
        if warm_cache --D-true "$t" --D-init "$d_init" --D-prior "$d_prior" \
                      --k "$K" --N "$N" --d "$D" --sigma 1e-3 --seed 0; then
            cells "$t" | run_group "$N"
        else
            echo "skipping D_true=$t: no data"
        fi
    done
else
    cells | run_group "$N"
fi

echo
echo "log: $LOG"
echo "results: runs/index.csv and runs/*/metrics.csv"
