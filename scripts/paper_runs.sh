#!/usr/bin/env bash
#
# The two experiments the paper's main body is missing. Sequential on purpose.
#
#   PHASE 1  Picard vs direct, paired over ten seeds on the SAME observable.
#            Table 3, the Objective 2 answer. Picard can only observe mu_2
#            (E14), and every direct inversion on record observes x_1, so the
#            existing numbers compare observables rather than methods.
#
#   PHASE 2  Measured outer rate against rho(DT) predicted beforehand, at
#            configurations spanning rho from 0.045 to 0.228. Figure 2. One
#            point (E22: 0.0570 measured vs 0.0666 predicted) cannot support
#            the claim that a quantity computed from three linear solves
#            predicts how fast the method converges; five can.
#
# SEQUENTIAL, not concurrent, deliberately. E23 left open whether the historical
# 5-6 percent optimizer failures came from running six cells at once. Serial
# execution removes that confound, so whatever failure count this batch produces
# is quotable. It also means no memory budget is needed.
#
# Usage:
#     tmux new -s paper
#     source scripts/env.sh
#     bash scripts/paper_runs.sh
#
# About 70-90 minutes. Every run is independent and writes its own directory, so
# the batch can be interrupted and restarted; completed cells will simply be
# recomputed, not skipped.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOG="paper-runs.log"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
RATE_SEEDS="${RATE_SEEDS:-0 1 2}"
SIGMA="${SIGMA:-1e-3}"
K="${K:-4}"
N="${N:-16}"

# Refuse to start rather than produce a batch with the wrong thread count or no
# Firedrake. OMP_NUM_THREADS is the one that bit us: unset, each job spawns a
# thread per core, and on a concurrent run that is 24x oversubscription. env.sh
# pins it to 1.
if ! python -c "import firedrake" >/dev/null 2>&1; then
    echo "ERROR: Firedrake not importable. Run: source scripts/env.sh" >&2
    exit 1
fi
if [ "${OMP_NUM_THREADS:-unset}" != "1" ]; then
    echo "ERROR: OMP_NUM_THREADS is ${OMP_NUM_THREADS:-unset}, expected 1." >&2
    echo "       Run: source scripts/env.sh" >&2
    exit 1
fi

OK=0
FAILED=0

# One run. Appends full output to $LOG, prints one progress line to stdout.
# Never aborts the batch: a failure is a data point, and runlog records the
# status either way.
run() {
    local label="$1"; shift
    local t0 t1
    t0=$(date +%s)
    { echo; echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ)  $label"; echo "    $*"; } >>"$LOG"
    if "$@" >>"$LOG" 2>&1; then
        t1=$(date +%s)
        OK=$((OK + 1))
        printf '  ok    %-44s %4ds\n' "$label" "$((t1 - t0))"
    else
        t1=$(date +%s)
        FAILED=$((FAILED + 1))
        printf '  FAIL  %-44s %4ds   (see %s)\n' "$label" "$((t1 - t0))" "$LOG"
    fi
}

# D_init and D_prior keep the same ratios to the truth as every other axis --
# 1.2x and 1.4286x -- so that moving D_true changes the truth and nothing else.
# picard.py and invert.py both default D_prior to 1.4286 * D_true, so only
# D_init has to be computed.
d_init_for() { awk -v t="$1" 'BEGIN { printf "%.6g", 1.2 * t }'; }

{ echo; echo "############ $(date -u +%Y-%m-%dT%H:%M:%SZ)  paper_runs  \
sha=$(git rev-parse --short HEAD)  sigma=$SIGMA k=$K N=$N ############"; } >>"$LOG"

echo "paper_runs starting $(date -u +%H:%M:%SZ)  sha=$(git rev-parse --short HEAD)"
echo "log: $LOG"

# ---------------------------------------------------------------------------
echo
echo "PHASE 1  head-to-head on mu_2, ten seeds each (Table 3)"
# ---------------------------------------------------------------------------

for s in $SEEDS; do
    run "direct mu_2 seed=$s" \
        python src/invert.py --field mu_2 --seed "$s" --sigma "$SIGMA" \
        --k "$K" --N "$N"
done

for s in $SEEDS; do
    run "picard mu_2 seed=$s" \
        python src/picard.py --seed "$s" --sigma "$SIGMA" \
        --k "$K" --N "$N" --rounds 12
done

# ---------------------------------------------------------------------------
echo
echo "PHASE 2  predicted rho(DT) against measured outer rate (Figure 2)"
# ---------------------------------------------------------------------------

# Diagnostics first: they are seconds each, and they are the PREDICTION. Run
# them before the loops so the prediction is on record independently of the
# loop's outcome rather than fitted after the fact.
for D in 0.6 2.0; do
    run "diag rho D_true=$D" \
        python src/diag_picard.py --k "$K" --N "$N" --D-true "$D"
done
run "diag rho sensors=8" \
    python src/diag_picard.py --k "$K" --N "$N" --sensors 8

# Then the loops. Three seeds each: enough to see whether the measured rate is
# stable at a configuration, not enough to call it a distribution.
for D in 0.6 2.0; do
    di=$(d_init_for "$D")
    for s in $RATE_SEEDS; do
        run "picard D_true=$D seed=$s" \
            python src/picard.py --D-true "$D" --D-init "$di" \
            --seed "$s" --sigma "$SIGMA" --k "$K" --N "$N" --rounds 12
    done
done

for s in $RATE_SEEDS; do
    run "picard sensors=8 seed=$s" \
        python src/picard.py --sensors 8 --seed "$s" --sigma "$SIGMA" \
        --k "$K" --N "$N" --rounds 12
done

# ---------------------------------------------------------------------------
echo
echo "finished $(date -u +%H:%M:%SZ):  $OK ok, $FAILED failed"
echo "results: runs/index.csv and runs/*/metrics.csv;  output: $LOG"
echo
echo "To bring it back, from the Mac:"
echo "  ssh bruinml 'cd taro/sosm-inverse && tar czf - runs/index.csv \\"
echo "      runs/*/metrics.csv paper-runs.log' | tar xzf -"
