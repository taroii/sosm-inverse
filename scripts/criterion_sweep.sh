#!/usr/bin/env bash
#
# Section 5.3's sweeps, with --gauge-aware.
#
# WHY. Every number in Section 5.3 was computed with diag_picard's DEFAULT map,
# in which the gauge targets are constants that differentiate away. E32 measured
# the outer rate against the GAUGE-AWARE criterion and got agreement to 0.5
# percent at four configurations; the default map misses by factors of 0.011 to
# 6.8. So the gauge-aware value is the criterion of the theorem as the loop
# realizes it, and Section 5.3 currently reports the other one. Left alone the
# paper would state rho = 0.0666 at D_true = 1, 16 sensors in Section 5.3 and
# 0.0258 for the same configuration in Table 4.
#
# These are diagnostics: three linear solves each, no noise, no seed, no
# optimizer. Nothing here is a median over anything -- one deterministic number
# per configuration, which is the point.
#
# Usage:
#     source scripts/env.sh
#     bash scripts/criterion_sweep.sh
#
# A few minutes. N=32 is the only slow cell.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOG="criterion-sweep.log"
K="${K:-4}"

if ! python -c "import firedrake" >/dev/null 2>&1; then
    echo "ERROR: Firedrake not importable. Run: source scripts/env.sh" >&2
    exit 1
fi

OK=0; FAILED=0
run() {
    local label="$1"; shift
    local t0 t1; t0=$(date +%s)
    { echo; echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ)  $label"; echo "    $*"; } >>"$LOG"
    if "$@" >>"$LOG" 2>&1; then
        t1=$(date +%s); OK=$((OK+1))
        printf '  ok    %-40s %4ds\n' "$label" "$((t1-t0))"
    else
        t1=$(date +%s); FAILED=$((FAILED+1))
        printf '  FAIL  %-40s %4ds   (see %s)\n' "$label" "$((t1-t0))" "$LOG"
    fi
}

{ echo; echo "############ $(date -u +%Y-%m-%dT%H:%M:%SZ)  criterion_sweep  \
sha=$(git rev-parse --short HEAD) ############"; } >>"$LOG"
echo "criterion_sweep starting $(date -u +%H:%M:%SZ)  sha=$(git rev-parse --short HEAD)"
echo "log: $LOG"

echo
echo "MESH  (paragraph 'Mesh': rho in the fourth significant figure)"
for N in 8 16 32; do
    run "mesh N=$N" python src/diag_picard.py --k "$K" --N "$N" --gauge-aware
done

echo
echo "OBSERVATION OPERATOR  (9, 16, 36, 64, 100 sensors)"
for s in 3 4 6 8 10; do
    run "sensors=$((s*s))" python src/diag_picard.py --k "$K" --N 16 \
        --sensors "$s" --gauge-aware
done

echo
echo "OBSERVED FIELD  (||J_beta|| spans seven orders; pressure is the hard case)"
for fld in mu_1 mu_2 p; do
    run "field=$fld" python src/diag_picard.py --k "$K" --N 16 \
        --field "$fld" --gauge-aware
done

echo
echo "OPERATING POINT  (the whole range the forward solver reaches)"
for D in 0.48 0.6 0.8 1.0 1.5 2.0 3.0 5.0; do
    run "D_true=$D" python src/diag_picard.py --k "$K" --N 16 \
        --D-true "$D" --gauge-aware
done

echo
echo "finished $(date -u +%H:%M:%SZ):  $OK ok, $FAILED failed"
echo "every value is printed in $LOG; grep for 'rho =' and '||j_D||'"
echo
echo "To bring it back, from the Mac:"
echo "  ssh bruinml 'cd taro/sosm-inverse && tar czf - criterion-sweep.log' | tar xzf -"
