#!/usr/bin/env bash
# Train beta in {1, 4} x 3 seeds, one job at a time, then cache each model's latent statistics.
# Safe to re-run: finished runs are skipped, interrupted runs resume from their last checkpoint.
#
#   tmux new -s bvae
#   ./run.sh                    # full budget (500k iterations per run)
#   BUDGET=reduced ./run.sh     # 150k iterations per run
#   ./run.sh bench              # time training steps and project wall time, trains nothing kept
#   EXTRA="--tf32" ./run.sh     # extra flags passed to every train.py call
#
# Env overrides: BUDGET, BETAS, SEEDS, RUN_DIR, DATA_DIR, EXTRA, PYTHON, CUDA_VISIBLE_DEVICES.
# Logs: $RUN_DIR/<run>/stdout.log per run, $RUN_DIR/run_all.log for the whole sweep.
set -uo pipefail
cd "$(dirname "$0")"

BUDGET=${BUDGET:-full}
BETAS=${BETAS:-"1 4"}
SEEDS=${SEEDS:-"0 1 2"}
RUN_DIR=${RUN_DIR:-runs}
DATA_DIR=${DATA_DIR:-data}
EXTRA=${EXTRA:-}
PYTHON=${PYTHON:-python}
mkdir -p "$RUN_DIR"
ALL_LOG="$RUN_DIR/run_all.log"

stamp() { date '+[%Y-%m-%d %H:%M:%S]'; }

$PYTHON train.py --prepare-data --data-dir "$DATA_DIR" 2>&1 | tee -a "$ALL_LOG" || exit 1

if [[ "${1:-}" == "bench" ]]; then
    $PYTHON train.py --benchmark --data-dir "$DATA_DIR" $EXTRA 2>&1 | tee -a "$ALL_LOG"
    exit $?
fi

echo "$(stamp) sweep start: budget=$BUDGET betas=[$BETAS] seeds=[$SEEDS] extra=[$EXTRA]" | tee -a "$ALL_LOG"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | tee -a "$ALL_LOG"

failed=()
for beta in $BETAS; do
    for seed in $SEEDS; do
        name=$(printf "beta%g_seed%d" "$beta" "$seed")
        mkdir -p "$RUN_DIR/$name"
        echo "$(stamp) start $name" | tee -a "$ALL_LOG"
        $PYTHON -u train.py --beta "$beta" --seed "$seed" --budget "$BUDGET" \
            --run-dir "$RUN_DIR" --data-dir "$DATA_DIR" $EXTRA 2>&1 | tee -a "$RUN_DIR/$name/stdout.log"
        rc=$?
        status=$(cat "$RUN_DIR/$name/status.json" 2>/dev/null || echo '{"status": "missing"}')
        echo "$(stamp) end $name exit=$rc $status" | tee -a "$ALL_LOG"
        [[ $rc -ne 0 ]] && failed+=("$name")
    done
done

echo "$(stamp) sweep finished" | tee -a "$ALL_LOG"
for d in "$RUN_DIR"/beta*_seed*; do
    printf "  %-16s %s  analysis:%s\n" "$(basename "$d")" "$(cat "$d/status.json" 2>/dev/null)" \
        "$([[ -f $d/analysis.npz ]] && echo yes || echo no)" | tee -a "$ALL_LOG"
done
if (( ${#failed[@]} )); then
    echo "$(stamp) runs that exited with an error (re-run ./run.sh to resume): ${failed[*]}" | tee -a "$ALL_LOG"
    exit 1
fi
