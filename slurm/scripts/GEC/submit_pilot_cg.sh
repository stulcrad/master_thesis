#!/bin/bash
#SBATCH --partition=amd
#SBATCH --output=/home/stulcrad/master_thesis/logs/pilot_scripts/out/%x_%N_%A.out
#SBATCH --error=/home/stulcrad/master_thesis/logs/pilot_scripts/err/%x_%N_%A.err
# Submitter for the MultiGEC constrained-generation line. Same model set and
# per-model arms as slurm/scripts/NER/submit_pilot_cg.sh -- see that file for
# the reasoning behind each one. Differences:
#
#   - MultiGEC is ONE file, 504 examples, no subsets to loop over (unlike UNER's
#     18 treebanks and WMT's 24 language-domain files), so there is no chunking
#     and no macro-average step afterwards -- the CSV number is the number.
#
# 10 arms x 1 dataset = 10 jobs, comfortably under the 200-job QOS cap.
#
# PILOT=1 (default) is a cheap timing run: capped examples, one seed.
# PILOT=0 is the real run: all 504 examples. Seeds are ONE either way.
#
#     bash slurm/scripts/GEC/submit_pilot_cg.sh                  # pilot
#     PILOT=0 bash slurm/scripts/GEC/submit_pilot_cg.sh          # full run
#     DRY_RUN=1 bash slurm/scripts/GEC/submit_pilot_cg.sh        # print only
#
# Check the indices with: python -m utils.model_registry --list
set -eu
cd ~/master_thesis

PILOT="${PILOT:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [ "$PILOT" -eq 1 ]; then
  SEEDS="42"
  MAX_EX_FLAG="--max-examples 10"
else
  SEEDS="42 43 44"
  MAX_EX_FLAG=""
fi

n_jobs=0

# submit <partition> <gpus> <array_idx> <max_new_tokens> <time> [extra args...]
submit() {
  local part=$1 gpus=$2 idx=$3 mnt=$4 wtime=$5 shards=$6; shift 6
  case "$shards" in ''|*[!0-9]*)
    echo "submit: shards must be a number, got '$shards' -- a positional arg is missing" >&2
    exit 1 ;; esac
  case "$wtime" in
    [0-9]*:[0-9][0-9]:[0-9][0-9]|[0-9]-[0-9]*:[0-9][0-9]:[0-9][0-9]) ;;
    *) echo "submit: wtime must be HH:MM:SS, got '$wtime' -- a positional arg is missing" >&2
       exit 1 ;;
  esac
  case "$gpus" in ''|*[!0-9]*)
    echo "submit: gpus must be a number, got '$gpus' -- a positional arg is missing" >&2
    exit 1 ;; esac
  case "$idx" in ''|*[!0-9]*)
    echo "submit: idx must be a number, got '$idx' -- a positional arg is missing" >&2
    exit 1 ;; esac
  case "$mnt" in ''|*[!0-9]*)
    echo "submit: mnt must be a number, got '$mnt' -- a positional arg is missing" >&2
    exit 1 ;; esac
  echo "submit: multigec array=${idx}  $*"
  # Counted before the DRY_RUN bail so a dry run reports the real job total.
  n_jobs=$((n_jobs + 1))
  if [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  # $MAX_EX_FLAG is deliberately UNQUOTED: either empty (and must vanish) or two
  # words that must word-split.
  sbatch --partition="$part" --gres=gpu:"$gpus" --array="$idx" --time="$wtime" \
    --export=ALL,SHARDS="$shards" \
    slurm/scripts/GEC/evaluationMultiGEC_CG.batch \
    $MAX_EX_FLAG --max-new-tokens "$mnt" --seeds $SEEDS "$@"
}

# Model registry indices once again, for reference:
#   idx 0  gemma-4-E2B-it    -> reasoning ON and OFF
#   idx 1  gemma-4-31B-it    -> reasoning OFF only
#   idx 2  Qwen3.8-27B       -> reasoning ON at effort=low, and OFF
#   idx 3  gpt-oss-20b       -> effort low|medium
#   idx 4  gpt-oss-120b      -> effort low|medium
#   idx 5  Qwen3-8B          -> reasoning OFF only

# -- gemma-4-E2B (idx 0): 2.3B effective, cheap enough for both arms ------
# submit h200       1 0 18000  8:00:00  8  --no-enable-thinking
#   -> [Claude est.] recommend --time 4:30:00 (currently 8:00:00) -- ~3.4h raw, measured 22.7s+20.9s/ex (uncon+constr), 504ex x3 seeds
# submit h200       1 0 18000 18:00:00 10  --enable-thinking
#   -> [Claude est.] recommend --time 35:00:00 (currently 18:00:00) -- ~27.7h raw, measured 208.5s+198.1s/ex (uncon+constr), 504ex x3 seeds  ** EXCEEDS 24h h200 WALL, consider a *long partition or more shards/chunks **

# -- gemma-4-31B (idx 1): 72 GiB/proc -> one process per H200 ------------------
# submit h200       2 1 18000 12:00:00  2  --no-enable-thinking
#   -> [Claude est.] recommend --time 7:00:00 (currently 12:00:00) -- ~5.1h raw, measured 12.1s+12.2s/ex (uncon+constr), 504ex x3 seeds

# -- Qwen3-8B (idx 5): Semin-comparable, reasoning OFF only --------------------
# submit h200       1 5 18000 12:00:00  8  --no-enable-thinking   # Qwen3-8B, 1 GPU
#   -> [Claude est.] recommend --time 4:30:00 (currently 12:00:00) -- ~3.2h raw, measured 24.0s+17.7s/ex (uncon+constr), 504ex x3 seeds

# -- Qwen3.8-27B (idx 2): reasoning ON (low) vs OFF -----------------------
# submit h200       1 2 18000 24:00:00  2  --no-enable-thinking
#   -> [Claude est.] recommend --time 9:00:00 (currently 24:00:00) -- ~6.9h raw, measured 15.9s+14.9s/ex (uncon+constr), 504ex x3 seeds
# submit h200       2 2 18000 20:00:00  4  --enable-thinking --reasoning-effort low
#   -> [Claude est.] recommend --time 37:30:00 (currently 20:00:00) -- ~29.6h raw, measured 131.4s+132.3s/ex (uncon+constr), 504ex x3 seeds  ** EXCEEDS 24h h200 WALL, consider a *long partition or more shards/chunks **

# -- gpt-oss (idx 3, 4): effort sweep -------------------------------------
# submit h200       1 3 18000 10:00:00  8  --reasoning-effort low
#   -> [Claude est.] recommend --time 11:00:00 (currently 10:00:00) -- ~8.4h raw, measured 54.5s+53.8s/ex (uncon+constr), 504ex x3 seeds
# submit h200       1 3 18000 12:00:00  9  --reasoning-effort medium
#   -> [Claude est.] recommend --time 54:00:00 (currently 12:00:00) -- ~43.0h raw, measured 378.1s+215.6s/ex (uncon+constr), 504ex x3 seeds  ** EXCEEDS 24h h200 WALL, consider a *long partition or more shards/chunks **

# submit h200       2 4 18000 16:00:00   2  --reasoning-effort low
#   -> [Claude est.] recommend --time 21:00:00 (currently 16:00:00) -- ~16.5h raw, measured 39.2s+39.5s/ex (uncon+constr), 504ex x3 seeds

submit h200       5 4 18000 24:00:00  5  --reasoning-effort medium
#   -> [Claude est.] recommend --time 9:00:00 (currently 24:00:00) -- ~6.6h raw, measured 39.2s+39.5s/ex (uncon+constr), 504ex x3 seeds; no exact effort match, borrowed low

echo
echo "PILOT=$PILOT  DRY_RUN=$DRY_RUN  seeds='$SEEDS'"
echo "total jobs: $n_jobs"
