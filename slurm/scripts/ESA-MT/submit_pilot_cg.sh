#!/bin/bash
#SBATCH --partition=amd
#SBATCH --output=/home/stulcrad/master_thesis/logs/pilot_scripts/out/%x_%N_%A.out
#SBATCH --error=/home/stulcrad/master_thesis/logs/pilot_scripts/err/%x_%N_%A.err
# Submitter for the WMT24 ESA constrained-generation line. Same model set and per-model arms as
# slurm/scripts/NER/submit_pilot_cg.sh -- see that file for the reasoning behind each one. Differences from the NER
# pilot:
#
#   - evaluationWMT_CG.py has no --batch_size flag
#   - WMT24 ESA is 24 source-target-domain files (867 examples total).
#
# HOW THE WORK IS SPLIT (identical to the UNER half of the NER submitter)
#   SHARDS (per submit line) = parallel PROCESSES inside one job, each pinned to a GPU by evaluationWMT_CG.batch. This
#     is the main lever: decoding at batch size 1 leaves a GPU ~70% idle, so several processes interleave nearly free.
#   CHUNKS (env, default 1) = how many JOBS one arm is split across. Purely a job-count decision: the packing is global,
#     so CHUNKS=1/SHARDS=6 and CHUNKS=3/SHARDS=2 produce the SAME six groups, just spread over 1 or 3 jobs. Raise it
#     when one job would exceed the wall clock, or to get scheduled sooner in smaller pieces.
#   Total processes per arm = CHUNKS * SHARDS, capped at 24 (the subset count) -- ask for more and bin_packing returns
#     fewer, non-empty groups rather than idle processes that would each still load a full model.
#
# Groups are balanced by EXAMPLE COUNT -- the 24 files range from 12 examples (wmt-en-ja-news) to 77 (wmt-en-is-news),
# so an equal-COUNT split (e.g. 6 files per group) can still hand one process 3 "is" files (~220 examples) and another 3
# "ja"/"zh" files (~40 examples), a >5x imbalance. src/utils/bin_packing.py does the greedy longest-processing-time
# packing, once, globally, and hands each job its --chunk window of it.
#
#
# PILOT vs FULL
# -------------
# PILOT=1 (default) is a cheap timing run: capped examples, one seed.
# PILOT=0 is the real run: whole datasets. Seeds are ONE either way.
#
#     bash slurm/scripts/ESA-MT/submit_pilot_cg.sh                  # pilot
#     PILOT=0 bash slurm/scripts/ESA-MT/submit_pilot_cg.sh          # full run
#     DRY_RUN=1 bash slurm/scripts/ESA-MT/submit_pilot_cg.sh        # print, submit nothing
#     CHUNKS=3 PILOT=0 bash slurm/scripts/ESA-MT/submit_pilot_cg.sh # full, 3 jobs per arm
#
# Preview how the subsets actually split, without submitting anything:
#     python -m utils.bin_packing wmt 4              # one job, 4 processes
#     python -m utils.bin_packing wmt 4 --chunk 1/3  # job 2 of 3, its 4 processes
#
# Check the indices with: python -m utils.model_registry --list
set -eu
cd ~/master_thesis

source init_environment_python.sh

PILOT="${PILOT:-1}"      # 1 = capped timing pilot, 0 = full datasets
DRY_RUN="${DRY_RUN:-0}"  # 1 = print the sbatch lines instead of submitting
CHUNKS="${CHUNKS:-1}"    # how many JOBS to split each arm across

# CHUNKS must be a positive integer
case "$CHUNKS" in
  ''|*[!0-9]*|0) echo "CHUNKS must be a positive integer, got '$CHUNKS'" >&2; exit 1 ;;
esac

# 3 seeds to measure variance
SEEDS="42 43 44"

N_SUBSETS=24   # data/semin/wmt/wmt-*.json, the ceiling on CHUNKS * SHARDS

n_jobs=0

# submit <partition> <gpus> <array_idx> <max_new_tokens> <time> <shards> [extra args...]
#
# WMT_CHUNK_SPEC (which window of the global packing this job owns) comes from the enclosing loop.
submit() {
  local part=$1 gpus=$2 idx=$3 mnt=$4 wtime=$5 shards=$6; shift 6
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
  case "$shards" in ''|*[!0-9]*)
    echo "submit: shards must be a number, got '$shards' -- a positional arg is missing" >&2
    exit 1 ;; esac
  echo "submit: ${TAG} array=${idx} part=${part} time=${wtime} shards=${shards} $*"
  n_jobs=$((n_jobs + 1))
  if [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  # No --wmt-subset here: the eval script defaults to "all", and the batch script
  # appends this process's group AFTER these args, where argparse's last-occurrence
  # -wins rule lets it take over. $MAX_EX_FLAG is deliberately UNQUOTED: it is
  # either empty (and must vanish entirely) or two words that must word-split.
  sbatch --partition="$part" --gres=gpu:"$gpus" --array="$idx" --time="$wtime" \
    --export=ALL,SHARDS="$shards",WMT_CHUNK="$WMT_CHUNK_SPEC" \
    slurm/scripts/ESA-MT/evaluationWMT_CG.batch \
    $MAX_EX_FLAG --max-new-tokens "$mnt" --seeds $SEEDS "$@"
}

# One iteration per JOB. Each job is told which window of the global packing it
# owns via WMT_CHUNK_SPEC; bin_packing.py does the packing itself, once.
for i in $(seq 0 $((CHUNKS - 1))); do
  WMT_CHUNK_SPEC="$i/$CHUNKS"   # --chunk i/N, read by evaluationWMT_CG.batch
  TAG="wmt/chunk$((i + 1))of${CHUNKS}"

  if [ "$PILOT" -eq 1 ]; then
    # CAUTION: the cap is PER SUBSET, not per job -- 15 x 24 = 360 examples per
    # arm across the whole sweep, however it is chunked.
    MAX_EX_FLAG="--max-examples 15"
  else
    # --max-examples is an int defaulting to None = everything, so for the full
    # run the flag must be omitted ENTIRELY rather than set to a sentinel.
    MAX_EX_FLAG=""
  fi

  # -- gemma-4-E2B (idx 0): 2.3B effective, cheap enough for both arms ------
  # submit h200fast   1 0 18000  4:00:00 9  --no-enable-thinking
  #   -> [Claude est.] recommend --time 3:00:00 (currently 4:00:00) -- ~2.1h raw, measured 8.6s+8.6s/ex (uncon+constr), 867ex x3 seeds
  # submit h200       2 0 18000 16:00:00 12 --enable-thinking
  #   -> [Claude est.] no reasoning-ON data for gemma-4-E2B-it on WMT24 ESA yet -- size cautiously (its CoNLL ON/OFF ratio is ~27x, so do not assume this is close to the OFF-arm line above)

  # -- Rows with a published comparison number: reasoning OFF ---------------
  # submit h200       3 1 18000 9:00:00 3 --no-enable-thinking   # gemma-4-31B, 2x A100-40
  #   -> [Claude est.] no timing data for gemma-4-31B-it here (NO DATA); size cautiously

  # submit h200fast   1 5 18000 4:00:00 6 --no-enable-thinking   # Qwen3-8B, 1 GPU
  #   -> [Claude est.] recommend --time 4:00:00 (currently 4:00:00) -- ~2.9h raw, measured 8.7s+9.5s/ex (uncon+constr), 867ex x3 seeds

  # -- Qwen3.8-27B (idx 2): reasoning ON (low) vs OFF -----------------------
  # submit h200       2 2 18000 10:00:00 4 --no-enable-thinking
  #   -> [Claude est.] no timing data for Qwen3.8-27B here (NO DATA); size cautiously
  submit h200       3 2 18000 24:00:00  6 --enable-thinking --reasoning-effort low
  #   -> [Claude est.] no timing data for Qwen3.8-27B here (NO DATA); size cautiously

  # -- gpt-oss (idx 3, 4): effort sweep -------------------------------------
  # submit h200fast   2 3 18000 4:00:00  12   --reasoning-effort low
  #   -> [Claude est.] recommend --time 4:30:00 (currently 4:00:00) -- ~3.2h raw, measured 20.1s+19.9s/ex (uncon+constr), 867ex x3 seeds  ** EXCEEDS 4h h200fast WALL, consider a *long partition or more shards/chunks **
  # submit h200       2 3 18000 16:00:00 12   --reasoning-effort medium
  #   -> [Claude est.] recommend --time 4:30:00 (currently 16:00:00) -- ~3.2h raw, measured 20.1s+19.9s/ex (uncon+constr), 867ex x3 seeds; no exact effort match, borrowed low

  # submit h200       3 4 18000 14:00:00  3  --reasoning-effort low
  #   -> [Claude est.] no timing data for gpt-oss-120b here (NO DATA); size cautiously
  submit h200       3 4 18000 20:00:00  3  --reasoning-effort medium
  #   -> [Claude est.] no timing data for gpt-oss-120b here (NO DATA); size cautiously
done

echo
echo "PILOT=$PILOT  DRY_RUN=$DRY_RUN  CHUNKS=$CHUNKS  seeds='$SEEDS'"
echo "total jobs: $n_jobs  (10 arms x $CHUNKS chunk(s) covering $N_SUBSETS subsets)"
if [ "$n_jobs" -gt 200 ]; then
  echo "WARNING: $n_jobs jobs exceeds the 200-job QOS cap -- lower CHUNKS."
fi
