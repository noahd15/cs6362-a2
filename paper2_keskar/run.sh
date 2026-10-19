#!/usr/bin/env bash
# Train and evaluate F1, C1, C2 one after another on one GPU.
#
#   tmux new -s keskar
#   bash run.sh                      # default: 1 trial per network
#   bash run.sh --n-trials 5         # full paper setting (any train.py flag passes through)
#   NETS="C2" bash run.sh            # a subset
#   (detach: Ctrl-b d;  reattach: tmux attach -t keskar)
#
# Safe to re-run after a crash or Ctrl-C: finished trials are loaded from disk,
# interrupted runs resume from their last progress checkpoint, finished paths are skipped.
# Extra flags must be the same on every re-run, or cached checkpoints are rejected.
set -uo pipefail
cd "$(dirname "$0")"

OUT="${OUT:-results}"
DATA="${DATA:-data}"
NETS="${NETS:-F1 C1 C2}"
PY="${PY:-python}"
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT/logs"
MASTER="$OUT/logs/run_$STAMP.log"

{
  echo "=== run $STAMP  host $(hostname)  args: $*"
  nvidia-smi || true
  "$PY" -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'torchvision', torchvision.__version__, 'gpu', torch.cuda.get_device_name() if torch.cuda.is_available() else None)"
} 2>&1 | tee -a "$MASTER"

"$PY" -u train.py --prepare-data --data-root "$DATA" 2>&1 | tee -a "$MASTER" || { echo "data download failed" | tee -a "$MASTER"; exit 1; }

failed=()
for net in $NETS; do
  echo "=== $(date '+%F %T') start $net" | tee -a "$MASTER"
  if "$PY" -u train.py --network "$net" --data-root "$DATA" --out-dir "$OUT" "$@" 2>&1 | tee -a "$MASTER"; then
    echo "=== $(date '+%F %T') done $net" | tee -a "$MASTER"
  else
    echo "=== $(date '+%F %T') FAILED $net (see $OUT/logs/$net.log)" | tee -a "$MASTER"
    failed+=("$net")
  fi
done

if ((${#failed[@]})); then
  echo "=== finished with failures: ${failed[*]}" | tee -a "$MASTER"; exit 1
fi
echo "=== all done; copy $OUT/*_results.json to wherever you run plot_results.ipynb" | tee -a "$MASTER"
