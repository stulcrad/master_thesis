#!/bin/bash
#SBATCH --partition=amd
#SBATCH --output=/home/stulcrad/master_thesis/logs/pilot_scripts/out/%x_%N_%A.out
#SBATCH --error=/home/stulcrad/master_thesis/logs/pilot_scripts/err/%x_%N_%A.err
# Pilot for the Toxic Spans constrained-generation line. Same model set and
# per-model arms as slurm/scripts/NER/submit_pilot_cg.sh -- see that file for
# the reasoning behind each one. Differences from the NER pilot:
#
#   - evaluationToxicSpans_cons_gen.py has no --dataset / --batch_size flags
#     (batch size is fixed at 1: posts are independent, so batching would only
#     test long-input robustness, not the task), so there is no DATASET/BS loop.
#   - One seed only for this timing pilot -- elapsed_minute_avg in the output
#     CSV tells us which arms fit the 16-18h target and which need trimming.
#
# Check the indices with: python -m utils.model_registry --list
set -eu
cd ~/master_thesis

# "${VAR:-default}" means: use $VAR if it's set (e.g. exported before running
# this script, like `PILOT=0 bash submit_pilot_cg.sh`), otherwise use `default`.
PILOT="${PILOT:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [ "$PILOT" -eq 1 ]; then
  SEEDS="42"
  MAX_EX_FLAG="--max-examples 25"
else
  SEEDS="42 43 44"
  MAX_EX_FLAG=""
fi

n_jobs=0


# submit <partition> <gpus> <array_idx> <max_new_tokens> <time> <shards> [extra args...]
#
# $1 $2 $3 ... are this function's positional arguments (like argv). `local`
# copies them into named variables; `shift 5` then drops the first 5 so that
# "$@" (below) is just whatever extra flags were passed after them, e.g.
submit() {
  local part=$1 gpus=$2 idx=$3 mnt=$4 wtime=$5 shards=$6; shift 6
  case "$wtime" in
    [0-9]*:[0-9][0-9]:[0-9][0-9]|[0-9]-[0-9]*:[0-9][0-9]:[0-9][0-9]) ;;
    *) echo "submit: wtime must be HH:MM:SS, got '$wtime' -- a positional arg is missing" >&2
       exit 1 ;; 
  esac
  case "$shards" in ''|*[!0-9]*)
    echo "submit: shards must be a number, got '$shards' -- a positional arg is missing" >&2
    exit 1 ;; esac
  case "$gpus" in ''|*[!0-9]*)
    echo "submit: gpus must be a number, got '$gpus' -- a positional arg is missing" >&2
    exit 1 ;; esac
  case "$idx" in ''|*[!0-9]*)
    echo "submit: idx must be a number, got '$idx' -- a positional arg is missing" >&2
    exit 1 ;; esac
  case "$mnt" in ''|*[!0-9]*)
    echo "submit: mnt must be a number, got '$mnt' -- a positional arg is missing" >&2
    exit 1 ;; esac
  echo "submit: toxic array=${idx} part=${part} time=${wtime} shards=${shards}  $*"
  n_jobs=$((n_jobs + 1))
  if [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  sbatch --partition="$part" --gres=gpu:"$gpus" --array="$idx" --time="$wtime" \
    --export=ALL,SHARDS="$shards" \
    slurm/scripts/ToxicSpans/evaluationToxicSpans_cons_gen.batch \
    $MAX_EX_FLAG --max-new-tokens "$mnt" --seeds $SEEDS "$@"
}

# Model registry indices once again, for reference:
#   idx 0  gemma-4-E2B-it    -> reasoning ON and OFF
#   idx 1  gemma-4-31B-it    -> reasoning OFF only
#   idx 2  Qwen3.8-27B       -> reasoning ON at effort=low, and OFF
#   idx 3  gpt-oss-20b       -> effort low|medium
#   idx 4  gpt-oss-120b      -> effort low|medium
#   idx 5  Qwen3-8B          -> reasoning OFF only

# -- gemma-4-E2B (idx 0): 12 GiB/proc, packs well ------------------------------
# submit h200fast   1 0 18000 4:00:00    5   --no-enable-thinking
#   -> [Claude est.] recommend --time 2:00:00 (currently 4:00:00) -- ~1.3h raw, measured 2.5s+3.6s/ex (uncon+constr), 1000ex x3 seeds
# submit h200       2 0 18000 8:00:00   12  --enable-thinking
#   -> [Claude est.] recommend --time 11:30:00 (currently 8:00:00) -- ~8.8h raw, measured 47.4s+47.0s/ex (uncon+constr), 1000ex x3 seeds

# -- gemma-4-31B (idx 1): 72 GiB/proc -> one process per H200 ------------------
# submit h200       1 1 18000 14:00:00   1  --no-enable-thinking
#   -> [Claude est.] recommend --time 4:30:00 (currently 14:00:00) -- ~3.3h raw, measured 1.9s+2.0s/ex (uncon+constr), 1000ex x3 seeds

# -- Qwen3-8B (idx 5): Semin-comparable, reasoning OFF only --------------------
# submit h200fast   1 5 18000 4:00:00    5  --no-enable-thinking
#   -> [Claude est.] recommend --time 2:30:00 (currently 4:00:00) -- ~1.6h raw, measured 3.6s+3.7s/ex (uncon+constr), 1000ex x3 seeds

# -- Qwen3.8-27B (idx 2): 65 GiB/proc ------------------------------------------
# submit h200       1 2 18000 14:00:00   2  --no-enable-thinking
#   -> [Claude est.] recommend --time 3:30:00 (currently 14:00:00) -- ~2.5h raw, measured 2.6s+2.9s/ex (uncon+constr), 1000ex x3 seeds
# submit h200       2 2 18000 24:00:00   4  --enable-thinking --reasoning-effort low
#   -> [Claude est.] recommend --time 13:30:00 (currently 24:00:00) -- ~10.4h raw, measured 23.6s+23.1s/ex (uncon+constr), 1000ex x3 seeds

# -- gpt-oss (idx 3, 4). No OFF arm -- Harmony always emits a reasoning channel. 
# submit h200fast   1 3 18000 4:00:00    5  --reasoning-effort low
#   -> [Claude est.] recommend --time 4:30:00 (currently 4:00:00) -- ~3.1h raw, measured 7.1s+7.3s/ex (uncon+constr), 1000ex x3 seeds  ** EXCEEDS 4h h200fast WALL, consider a *long partition or more shards/chunks **
submit h200       3 3 18000 12:00:00    12  --reasoning-effort medium
#   -> [Claude est.] recommend --time 7:30:00 (currently 12:00:00) -- ~5.8h raw, measured 34.5s+34.5s/ex (uncon+constr), 1000ex x3 seeds

# submit h200       2 4 18000 14:00:00   2  --reasoning-effort low
#   -> [Claude est.] recommend --time 7:30:00 (currently 14:00:00) -- ~5.7h raw, measured 6.7s+6.9s/ex (uncon+constr), 1000ex x3 seeds
# submit h200       4 4 18000 14:00:00   4  --reasoning-effort medium
#   -> [Claude est.] recommend --time 4:00:00 (currently 14:00:00) -- ~2.8h raw, measured 6.7s+6.9s/ex (uncon+constr), 1000ex x3 seeds; no exact effort match, borrowed low

echo
echo "PILOT=$PILOT  DRY_RUN=$DRY_RUN  seeds='$SEEDS'"
echo "total jobs: $n_jobs"
