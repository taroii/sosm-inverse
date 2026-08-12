#!/usr/bin/env bash
#
# Cost sweep: wall time and peak memory for one forward solve, over (k, N).
#
# Each configuration is a separate process running a single refinement, so the
# numbers include Firedrake startup and JIT compilation -- which is what a solve
# actually costs inside an optimizer loop, and therefore what the experiment
# budget in README.md section IV has to be planned against.
#
# Writes two files:
#   bench.csv   one row per configuration
#   bench.log   full stdout, appended
# Both are gitignored.
#
# Resumable: configurations already present in bench.csv are skipped, so an
# interrupted sweep can be restarted without repeating work.
#
# Usage:
#     tmux new -s bench
#     source scripts/env.sh
#     bash scripts/bench.sh
#
#     KS="2 3 4" NS="8 16 32" bash scripts/bench.sh    # narrower sweep
#     TIMEOUT=600 bash scripts/bench.sh                # 10 min per config

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

KS="${KS:-2 3 4 5}"
NS="${NS:-8 16 32 64 128}"
TIMEOUT="${TIMEOUT:-1800}"
D="${D:-2}"

CSV=bench.csv
LOG=bench.log

if [[ ! -f "$CSV" ]]; then
    echo "d,k,N,ndofs,wall_s,peak_rss_kb,peak_rss_gb,exit_status" > "$CSV"
fi

for k in $KS; do
    for N in $NS; do

        if cut -d, -f1-3 "$CSV" | grep -qx "$D,$k,$N"; then
            echo "skip d=$D k=$k N=$N (already in $CSV)"
            continue
        fi

        echo "=== d=$D k=$k N=$N ===" | tee -a "$LOG"
        tmp=$(mktemp)

        # %e wall seconds, %M peak RSS in kbytes. Portable across the runs we
        # care about and easier to parse than -v.
        timeout "$TIMEOUT" /usr/bin/time -f "%e %M" -o "$tmp" \
            python src/fig01_convergence.py \
                --d "$D" --k "$k" --N0 "$N" --loops 1 >>"$LOG" 2>&1
        status=$?

        ndofs=$(grep -a "ndofs" "$LOG" | tail -1 | tr -dc '0-9')
        read -r wall rss < <(tail -1 "$tmp") || { wall=; rss=; }
        rm -f "$tmp"

        if [[ -n "${rss:-}" ]]; then
            gb=$(awk -v r="$rss" 'BEGIN{printf "%.3f", r/1048576}')
        else
            gb=
        fi

        echo "$D,$k,$N,${ndofs:-},${wall:-},${rss:-},${gb:-},$status" >> "$CSV"
        echo "  wall=${wall:-?}s  peak=${gb:-?}GB  ndofs=${ndofs:-?}  exit=$status" \
            | tee -a "$LOG"

        # 124 is timeout, 137 is SIGKILL (the OOM killer). Both mean this k has
        # reached its ceiling, so stop refining and move to the next k.
        if [[ $status -eq 124 || $status -eq 137 ]]; then
            echo "  stopping N sweep for k=$k (exit $status)" | tee -a "$LOG"
            break
        fi
    done
done

echo
echo "wrote $CSV"
column -s, -t "$CSV"
