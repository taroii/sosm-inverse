#!/usr/bin/env bash
#
# Every Picard entry in the main body at ten seeds, on one SHA.
#
# WHY, per table (numbering as in the draft):
#
#   Table 4  rate against the criterion. Three of its four rows rest on 2 or 3
#            seeds. The rate itself is reproducible to 1-3 percent across seeds
#            (E24), so the entries will barely move; this makes the seed count
#            match the other tables rather than changing the numbers.
#
#   Table 5  accuracy against direct optimization. The Picard column at
#            D_true = 0.6 and 2.0 rests on three seeds. At D_true = 2.0 those
#            three span a factor of 11 in error (0.57, 3.97, 6.47 sigma), and
#            the median of them is the "within a factor of two of direct"
#            claim and one of the three points in the D^-2.8 fit. This is the
#            batch's reason for existing.
#
#   Table 6  cost. Its linearized-solve and wall-time entries were measured
#            before stall_max capped a stalled run at two wasted rounds, so
#            they describe code that no longer exists. Re-measured here.
#
# The direct arm at D_true = 0.6 and 2.0 (E26, E28) is re-run rather than
# re-pulled: those cells were produced at 23ad86e and the Picard cells they are
# compared against will now come from HEAD. The comparison is paired by design,
# so both arms should come off the same stack.
#
# Phase A runs first and is seconds: it is the PREDICTION, and it goes on
# record before the loops rather than after them.
#
# SEQUENTIAL, as paper_runs.sh was, so the failure count is quotable and no
# memory budget is needed.
#
# Usage:
#     tmux new -s seeds
#     source scripts/env.sh
#     bash scripts/paper_seeds.sh
#
# About two hours. Every run writes its own directory, so the batch can be
# interrupted and restarted; completed cells are recomputed, not skipped.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOG="paper-seeds.log"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
SIGMA="${SIGMA:-1e-3}"
K="${K:-4}"
N="${N:-16}"
ROUNDS="${ROUNDS:-12}"

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

run() {
    local label="$1"; shift
    local t0 t1
    t0=$(date +%s)
    { echo; echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ)  $label"; echo "    $*"; } >>"$LOG"
    if "$@" >>"$LOG" 2>&1; then
        t1=$(date +%s); OK=$((OK + 1))
        printf '  ok    %-40s %4ds\n' "$label" "$((t1 - t0))"
    else
        t1=$(date +%s); FAILED=$((FAILED + 1))
        printf '  FAIL  %-40s %4ds   (see %s)\n' "$label" "$((t1 - t0))" "$LOG"
    fi
}

# D_init and D_prior hold the same ratios to the truth as every other axis,
# 1.2x and 1.4286x, so moving D_true moves the truth and nothing else.
# picard.py and invert.py both default D_prior to 1.4286 * D_true.
d_init_for() { awk -v t="$1" 'BEGIN { printf "%.6g", 1.2 * t }'; }

{ echo; echo "############ $(date -u +%Y-%m-%dT%H:%M:%SZ)  paper_seeds  \
sha=$(git rev-parse --short HEAD)  sigma=$SIGMA k=$K N=$N ############"; } >>"$LOG"

echo "paper_seeds starting $(date -u +%H:%M:%SZ)  sha=$(git rev-parse --short HEAD)"
echo "log: $LOG"

# ---------------------------------------------------------------------------
echo
echo "PHASE A  the criterion at the four Table 4 configurations (seconds)"
# ---------------------------------------------------------------------------
for D in 0.6 1.0 2.0; do
    run "diag gauge-aware D_true=$D" \
        python src/diag_picard.py --k "$K" --N "$N" --D-true "$D" --gauge-aware
done
run "diag gauge-aware sensors=8" \
    python src/diag_picard.py --k "$K" --N "$N" --sensors 8 --gauge-aware

# ---------------------------------------------------------------------------
echo
echo "PHASE B  Picard, ten seeds at each Table 4/5 configuration"
# ---------------------------------------------------------------------------
# D_true = 2.0 first: it is the thin cell that carries the weak-coupling claim.
for D in 2.0 0.6 1.0; do
    di=$(d_init_for "$D")
    for s in $SEEDS; do
        run "picard D_true=$D seed=$s" \
            python src/picard.py --D-true "$D" --D-init "$di" \
            --seed "$s" --sigma "$SIGMA" --k "$K" --N "$N" --rounds "$ROUNDS"
    done
done

for s in $SEEDS; do
    run "picard sensors=8 seed=$s" \
        python src/picard.py --sensors 8 --seed "$s" --sigma "$SIGMA" \
        --k "$K" --N "$N" --rounds "$ROUNDS"
done

# ---------------------------------------------------------------------------
echo
echo "PHASE C  direct arm of Table 5 at the two off-anchor points"
# ---------------------------------------------------------------------------
for D in 0.6 2.0; do
    di=$(d_init_for "$D")
    for s in $SEEDS; do
        run "direct mu_2 D_true=$D seed=$s" \
            python src/invert.py --field mu_2 --D-true "$D" --D-init "$di" \
            --seed "$s" --sigma "$SIGMA" --k "$K" --N "$N"
    done
done

# ---------------------------------------------------------------------------
echo
echo "finished $(date -u +%H:%M:%SZ):  $OK ok, $FAILED failed"
echo "results: runs/index.csv and runs/*/metrics.csv;  output: $LOG"
echo
echo "To bring it back, from the Mac:"
echo "  ssh bruinml 'cd taro/sosm-inverse && tar czf - runs/index.csv \\"
echo "      runs/*/metrics.csv paper-seeds.log' | tar xzf -"
