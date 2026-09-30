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
#   -> [Claude est. v2] --time 3:30:00 (currently 8:00:00) | ~2.5h + 25% margin.
#      Measured: job 11456075_0 (2.51h) at 8 shards = 63 ex/shard.
# submit h200       1 0 18000 18:00:00 10  --enable-thinking
#   -> [Claude est. v2] --time 22:00:00 (currently 18:00:00) | ~17.6h + 25% margin.
#      Measured: job 11456076_0 (17.55h) at 10 shards = 51 ex/shard.

# -- gemma-4-31B (idx 1): 72 GiB/proc -> one process per H200 ------------------
# submit h200       2 1 18000 12:00:00  2  --no-enable-thinking
#   -> [Claude est. v2] --time 7:30:00 (currently 12:00:00) | ~5.8h + 25% margin.
#      Measured: job 11456077_1 (5.81h) at 2 shards = 252 ex/shard.

# -- Qwen3-8B (idx 5): Semin-comparable, reasoning OFF only --------------------
submit h200         1 5 18000 7:00:00  8  --no-enable-thinking   # Qwen3-8B, 1 GPU -- was OOM-killed on h03
#   -> [Claude est. v2] --time 7:00:00 (currently 12:00:00) | ~2.7h + 150% margin.
#      ROUGH transfer (+/-3x on back-test): no gec run of this arm -- from job 11437687_5 (3.39h) (geometric mean of 23 transfers) on conll2003, x10.51 (gec/conll2003 ratio on the gemma-4-E2B-it OFF arm) at 1 shard = 504 ex/shard; this line is 8 shards = 63 ex/shard.

# -- Qwen3.8-27B (idx 2): reasoning ON (low) vs OFF -----------------------
# submit h200       1 2 18000 24:00:00  2  --no-enable-thinking
#   -> [Claude est. v2] --time 9:00:00 (currently 24:00:00) | ~6.8h + 25% margin.
#      Measured: job 11456079_2 (6.84h) at 2 shards = 252 ex/shard.
# submit h200       2 2 18000 20:00:00  4  --enable-thinking --reasoning-effort low
#   -> [Claude est. v2] --time 36:30:00 (currently 20:00:00) | ~14.4h + 150% margin.  ** over the 24h h200 wall: use *long, or more shards/chunks **
#      ROUGH transfer (+/-3x on back-test): no gec run of this arm -- from job 11437682_2 (6.17h) (geometric mean of 3 transfers) on conll2003, x5.87 (gec/conll2003 ratio on the OFF arm) at 1 shard = 504 ex/shard; this line is 4 shards = 126 ex/shard.

# -- gpt-oss (idx 3, 4): effort sweep -------------------------------------
# submit h200       1 3 18000 10:00:00  8  --reasoning-effort low
#   -> [Claude est. v2] --time 8:30:00 (currently 10:00:00) | ~6.5h + 25% margin.
#      Measured: job 11456081_3 (6.47h) at 8 shards = 63 ex/shard.
submit h200       3 3 18000 20:00:00  18  --reasoning-effort medium

submit h200       2 4 18000 20:00:00   2  --reasoning-effort low
#   -> [Claude est. v2] --time 22:30:00 (currently 20:00:00) | ~17.7h + 25% margin.
#      Measured: timeout 11456083_4 (16.0h = 90.5% of rows) at 2 shards = 252 ex/shard.

# submit h200       5 4 18000 24:00:00  5  --reasoning-effort medium
#   -> [Claude est. v2] --time 32:00:00 (currently 24:00:00) | ~21.3h + 50% margin.  ** over the 24h h200 wall: use *long, or more shards/chunks **
#      Scaled: timeout 11456083_4 (16.0h = 90.5% of rows) at effort=low, x3 for medium at 2 shards = 252 ex/shard; this line is 5 shards = 101 ex/shard.

echo
echo "PILOT=$PILOT  DRY_RUN=$DRY_RUN  seeds='$SEEDS'"
echo "total jobs: $n_jobs"
