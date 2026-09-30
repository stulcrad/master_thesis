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
#   -> [Claude est. v2] --time 2:00:00 (currently 4:00:00) | ~1.5h + 25% margin.
#      Measured: job 11442018_0 (1.46h) at 5 shards = 200 ex/shard.
# submit h200       2 0 18000 8:00:00   12  --enable-thinking
#   -> [Claude est. v2] --time 9:00:00 (currently 8:00:00) | ~7.1h + 25% margin.
#      Measured: job 11458384_0 (7.11h) at 12 shards = 84 ex/shard.

# -- gemma-4-31B (idx 1): 72 GiB/proc -> one process per H200 ------------------
# submit h200       1 1 18000 14:00:00   1  --no-enable-thinking
#   -> [Claude est. v2] --time 5:30:00 (currently 14:00:00) | ~4.3h + 25% margin.
#      Measured: job 11442022_1 (4.28h) at 1 shard = 1000 ex/shard.

# -- Qwen3-8B (idx 5): Semin-comparable, reasoning OFF only --------------------
# submit h200fast   1 5 18000 4:00:00    5  --no-enable-thinking
#   -> [Claude est. v2] --time 2:30:00 (currently 4:00:00) | ~1.7h + 25% margin.
#      Measured: job 11442024_5 (1.69h) at 5 shards = 200 ex/shard.

# -- Qwen3.8-27B (idx 2): 65 GiB/proc ------------------------------------------
# submit h200       1 2 18000 14:00:00   2  --no-enable-thinking
#   -> [Claude est. v2] --time 4:30:00 (currently 14:00:00) | ~3.2h + 25% margin.
#      Measured: job 11442026_2 (3.20h) at 2 shards = 500 ex/shard.
# submit h200       2 2 18000 24:00:00   4  --enable-thinking --reasoning-effort low
#   -> [Claude est. v2] --time 14:30:00 (currently 24:00:00) | ~11.2h + 25% margin.
#      Measured: job 11442028_2 (11.24h) at 4 shards = 250 ex/shard.

# -- gpt-oss (idx 3, 4). No OFF arm -- Harmony always emits a reasoning channel. 
# submit h200fast   1 3 18000 4:00:00    5  --reasoning-effort low
#   -> [Claude est. v2] --time 4:00:00 (currently 4:00:00) | ~2.9h + 25% margin.
#      Measured: job 11442030_3 (2.88h) at 5 shards = 200 ex/shard.
submit h200       3 3 18000 12:00:00    12  --reasoning-effort medium

# submit h200       2 4 18000 14:00:00   2  --reasoning-effort low
#   -> [Claude est. v2] --time 9:00:00 (currently 14:00:00) | ~6.8h + 25% margin.
#      Measured: job 11450188_4 (6.81h) at 2 shards = 500 ex/shard.
# submit h200       4 4 18000 14:00:00   4  --reasoning-effort medium
#   -> [Claude est. v2] --time 15:30:00 (currently 14:00:00) | ~10.2h + 50% margin.
#      Scaled: job 11450188_4 (6.81h at effort=low) x3 for medium at 2 shards = 500 ex/shard; this line is 4 shards = 250 ex/shard.

echo
echo "PILOT=$PILOT  DRY_RUN=$DRY_RUN  seeds='$SEEDS'"
echo "total jobs: $n_jobs"
