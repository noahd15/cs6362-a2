#!/usr/bin/env bash
# Sequential Figure 1 sweep on one GPU. Safe to re-run: finished runs with an identical config
# are skipped by train_fig1.py; a failed run is logged and the queue continues.
#
#   ./run_sweep.sh                         # defaults below
#   DEVICE=cpu RESULTS_DIR=./r ./run_sweep.sh
#   EXTRA_ARGS="--max-epochs 150" ./run_sweep.sh   # applies to every run (and changes every cache key)
set -uo pipefail

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
RESULTS_DIR="${RESULTS_DIR:-./results_fig1}"
DATA_ROOT="${DATA_ROOT:-./data}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "$LOG_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"

# Download CIFAR-10 once before the queue starts.
python -c "import torchvision as tv; [tv.datasets.CIFAR10('$DATA_ROOT', train=t, download=True) for t in (True, False)]"

RUNS=()
# (1) Label-corruption grid for panels (b) and (c), batch 128.
#     Ordered cheapest model first so the MLP and AlexNet curves are complete early.
for model in mlp_1x512 alexnet inception; do
  for p in 0.0 0.2 0.4 0.6 0.8 1.0; do
    RUNS+=("--model $model --mode labels --p $p")
  done
done
# (2) Input randomizations for panel (a), Inception at p=0. True labels and random labels
#     are the inception p=0.0 / p=1.0 runs above. --min-steps 25000 extends each curve over
#     the full 0-25k x-axis of Fig. 1a; drop it to stop these runs at convergence instead.
for mode in shuffled_pixels random_pixels gaussian; do
  RUNS+=("--model inception --mode $mode --p 0.0 --min-steps 25000")
done
# (3) Batch-size sensitivity: Inception, random labels.
for bs in 64 256; do
  RUNS+=("--model inception --mode labels --p 1.0 --batch-size $bs")
done

n=${#RUNS[@]}; i=0; failed=0
start=$(date +%s)
for r in "${RUNS[@]}"; do
  i=$((i + 1))
  tag=$(echo "$r" | tr -d '-' | tr ' ' '_')
  echo "[$i/$n] $(date '+%F %T') $r"
  # shellcheck disable=SC2086
  if ! python train_fig1.py $r --device "$DEVICE" --results-dir "$RESULTS_DIR" \
        --data-root "$DATA_ROOT" $EXTRA_ARGS > "$LOG_DIR/$tag.log" 2>&1; then
    echo "   FAILED (see $LOG_DIR/$tag.log)"; failed=$((failed + 1))
  else
    tail -n 1 "$LOG_DIR/$tag.log" | sed 's/^/   /'
  fi
done
echo "finished $n runs ($failed failed) in $(( ($(date +%s) - start) / 60 )) min"
exit $(( failed > 0 ))
