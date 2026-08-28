#!/bin/bash
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
  # SEEDS="42 43 44"
  SEEDS="42"  # for now, to avoid the 3x runtime of the full run
  # --max-examples defaults to None = everything, so the flag must be omitted
  # ENTIRELY for the full run rather than set to a sentinel.
  MAX_EX_FLAG=""
fi

n_jobs=0

# submit <partition> <gpus> <array_idx> <max_new_tokens> <time> [extra args...]
submit() {
  local part=$1 gpus=$2 idx=$3 mnt=$4 wtime=$5; shift 5
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
    slurm/scripts/GEC/evaluationMultiGEC_CG.batch \
    $MAX_EX_FLAG --max-new-tokens "$mnt" --seeds $SEEDS "$@"
}

# -- gemma-4-E2B (idx 0): 2.3B effective, cheap enough for both arms ------
# submit amdgpufast 1 0 18000 4:00:00 --enable-thinking
# submit amdgpufast 1 0 18000 4:00:00 --no-enable-thinking

# -- Rows with a published comparison number: reasoning OFF ---------------
# gemma-4-31B is the direct comparison here: their MultiGEC TAG row is
# 52.9 hard F1 (thinking=False, 5 seeds, from their own results.csv).
# submit amdgpufast 2 1 18000 4:00:00 --no-enable-thinking   # gemma-4-31B, 2x A100-40
# submit amdgpufast 1 5 18000 4:00:00 --no-enable-thinking   # Qwen3-8B, 1 GPU

# -- Qwen3.8-27B (idx 2): reasoning ON (low) vs OFF -----------------------
# submit amdgpufast 2 2 18000 4:00:00 --enable-thinking --reasoning-effort low
# submit amdgpufast 2 2 18000 4:00:00 --no-enable-thinking

# -- gpt-oss (idx 3, 4): effort sweep -------------------------------------
for EFF in low medium; do
  submit h200fast 1 3 18000 4:00:00 --reasoning-effort "$EFF"
  submit h200fast 2 4 18000 4:00:00 --reasoning-effort "$EFF"
done

echo
echo "PILOT=$PILOT  DRY_RUN=$DRY_RUN  seeds='$SEEDS'"
echo "total jobs: $n_jobs"
