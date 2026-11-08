#!/usr/bin/env bash
#
# Table 5c: the per-seed test of the frozen-model error identity.
#
# Table 5b shows the two methods correlate at +0.97 to +0.997, which is
# consistent with the identity but does not test it -- two quantities can
# correlate because both track the same noise. This predicts each seed's offset
# from that seed's own residual and curvature, with no fitted parameter, and
# reports both sides of four separate claims (T1-T4) plus the discretization
# gap (T5).
#
# No Picard or direct runs are repeated: beta* and betahat are read from runs/,
# and the new work is the three criterion solves plus a handful of forward
# solves per seed, at that seed's own fixed point rather than at D_true.
#
# Usage:
#     source scripts/env.sh
#     bash scripts/offset_test.sh
#
# Roughly 30-60 minutes for 30 seeds.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOG="offset-test.log"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"

if ! python -c "import firedrake" >/dev/null 2>&1; then
    echo "ERROR: Firedrake not importable. Run: source scripts/env.sh" >&2
    exit 1
fi
if [ "${OMP_NUM_THREADS:-unset}" != "1" ]; then
    echo "ERROR: OMP_NUM_THREADS is ${OMP_NUM_THREADS:-unset}, expected 1." >&2
    exit 1
fi

{ echo; echo "############ $(date -u +%Y-%m-%dT%H:%M:%SZ)  offset_test  \
sha=$(git rev-parse --short HEAD) ############"; } >>"$LOG"
echo "offset_test starting $(date -u +%H:%M:%SZ)  sha=$(git rev-parse --short HEAD)"
echo "log: $LOG"

OK=0; FAILED=0
for D in 0.6 1.0 2.0; do
    t0=$(date +%s)
    { echo; echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ)  D_true=$D"; } >>"$LOG"
    if python src/diag_offset.py --D-true "$D" --seeds $SEEDS >>"$LOG" 2>&1; then
        OK=$((OK+1)); printf '  ok    D_true=%-6s %5ds\n' "$D" "$(( $(date +%s) - t0 ))"
    else
        FAILED=$((FAILED+1))
        printf '  FAIL  D_true=%-6s %5ds   (see %s)\n' "$D" "$(( $(date +%s) - t0 ))" "$LOG"
    fi
done

echo
echo "finished $(date -u +%H:%M:%SZ):  $OK ok, $FAILED failed"
echo "per-seed rows: runs/*_offset_*/metrics.csv;  output: $LOG"
echo
echo "To bring it back, from the Mac:"
echo "  ssh bruinml 'cd taro/sosm-inverse && tar czf - runs/index.csv \\"
echo "      runs/*/metrics.csv offset-test.log' | tar xzf -"
