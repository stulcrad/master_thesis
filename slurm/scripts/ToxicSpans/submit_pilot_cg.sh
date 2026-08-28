#!/bin/bash
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
  # SEEDS="42 43 44"
  SEEDS="42"
  MAX_EX_FLAG=""
fi

n_jobs=0


# submit <partition> <gpus> <array_idx> <max_new_tokens> <time> [extra args...]
#
# $1 $2 $3 ... are this function's positional arguments (like argv). `local`
# copies them into named variables; `shift 5` then drops the first 5 so that
# "$@" (below) is just whatever extra flags were passed after them, e.g.
submit() {
  local part=$1 gpus=$2 idx=$3 mnt=$4 wtime=$5; shift 5
  case "$gpus" in ''|*[!0-9]*)
    echo "submit: gpus must be a number, got '$gpus' -- a positional arg is missing" >&2
    exit 1 ;; esac
  case "$idx" in ''|*[!0-9]*)
    echo "submit: idx must be a number, got '$idx' -- a positional arg is missing" >&2
    exit 1 ;; esac
  case "$mnt" in ''|*[!0-9]*)
    echo "submit: mnt must be a number, got '$mnt' -- a positional arg is missing" >&2
    exit 1 ;; esac
  echo "submit: toxic array=${idx} part=${part} time=${wtime}  $*"
  n_jobs=$((n_jobs + 1))
  if [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  sbatch --partition="$part" --gres=gpu:"$gpus" --array="$idx" --time="$wtime" \
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


# -- gemma-4-E2B (idx 0): 2.3B effective, cheap enough for both arms --------------
submit amdgpufast 1 0 18000 4:00:00  --no-enable-thinking
submit amdgpu     1 0 18000 16:00:00 --enable-thinking

# -- Rows with a published comparison number: reasoning OFF ----------------------
submit amdgpu     2 1 18000 6:00:00  --no-enable-thinking

# -- Qwen3-8B -> has a published comparison row, reasoning OFF only ---------------
submit amdgpufast 1 5 18000 4:00:00  --no-enable-thinking

# -- Qwen3.8-27B (idx 2): reasoning ON (low) vs OFF -------------------------------
# --reasoning-effort goes ONLY on the ON arm -- see the NER pilot's comment for why.
submit amdgpu     2 2 18000 14:00:00 --no-enable-thinking
submit amdgpulong 2 2 18000 30:00:00 --enable-thinking --reasoning-effort low

# -- gpt-oss (idx 3, 4): effort sweep --------------------------------------------
# No OFF arm -- Harmony always emits a reasoning channel (reasoning_off_supported=False).
submit h200fast   1 3 18000 4:00:00  --reasoning-effort low
submit h200       1 3 18000 14:00:00 --reasoning-effort medium

submit h200       2 4 18000 7:00:00  --reasoning-effort low
submit h200       2 4 18000 16:00:00 --reasoning-effort medium

echo
echo "PILOT=$PILOT  DRY_RUN=$DRY_RUN  seeds='$SEEDS'"
echo "total jobs: $n_jobs"
