#!/bin/bash
#SBATCH --partition=amd
#SBATCH --output=/home/stulcrad/master_thesis/logs/pilot_scripts/out/%x_%N_%A.out
#SBATCH --error=/home/stulcrad/master_thesis/logs/pilot_scripts/err/%x_%N_%A.err
#
# Submitter for the NER/UNER constrained-generation line.
#
# PILOT vs FULL
#     bash slurm/scripts/NER/submit_pilot_cg.sh                  # pilot
#     PILOT=0 bash slurm/scripts/NER/submit_pilot_cg.sh          # full run
#     DRY_RUN=1 bash slurm/scripts/NER/submit_pilot_cg.sh        # print, submit nothing
#     # full run with UNER split into 4 jobs (each with SHARDS processes)
#     UNER_CHUNKS=4 PILOT=0 bash slurm/scripts/NER/submit_pilot_cg.sh
#
# DATASET INSTANCES. CoNLL-2003 is a single corpus, no --subset. UNER is 18
# treebanks, run UNER_CHUNKS at a time (--subset takes several treebanks per
# job, keeping the model resident).
#
# MODEL ARMS
#   idx 0  gemma-4-E2B-it    -> reasoning ON and OFF
#   idx 1  gemma-4-31B-it    -> reasoning OFF only
#   idx 2  Qwen3.8-27B       -> reasoning ON at effort=low, and OFF
#   idx 3  gpt-oss-20b       -> effort low|medium
#   idx 4  gpt-oss-120b      -> effort low|medium
#   idx 5  Qwen3-8B          -> reasoning OFF only
# Check the indices with: python -m utils.model_registry --list

# -e: stop the whole script immediately if any command fails,
# instead of plowing on and submitting the rest with something broken.
# -u: treat use of an unset variable as an error, instead of silently treating
# it as an empty string -- catches typos like $SUSBET_FLAG.
set -eu
cd ~/master_thesis

source init_environment_python.sh

# "${VAR:-default}" means: use $VAR if it's set (e.g. exported before running
# this script, like `PILOT=0 bash submit_pilot_cg.sh`), otherwise use `default`.
PILOT="${PILOT:-1}"
DRY_RUN="${DRY_RUN:-0}"
UNER_CHUNKS="${UNER_CHUNKS:-1}"
# Guard: UNER_CHUNKS=0 would make `seq 0 -1` emit nothing, silently submitting no
# UNER jobs at all -- no error, just a missing dataset noticed days later.
case "$UNER_CHUNKS" in
  ''|*[!0-9]*|0) echo "UNER_CHUNKS must be a positive integer, got '$UNER_CHUNKS'" >&2; exit 1 ;;
esac

if [ "$PILOT" -eq 1 ]; then
  SEEDS="42"
else
  SEEDS="42 43 44"
fi

n_jobs=0

# submit <partition> <gpus> <array_idx> <max_new_tokens> <time> [extra args...]
#
# $1 $2 $3 ... are this function's positional arguments (like argv). `local`
# copies them into named variables; `shift 6` then drops the first 6 so that
# "$@" (below) is just whatever extra flags were passed after them, e.g.
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
  echo "submit: ${TAG} bs=${BS} array=${idx} part=${part} time=${wtime} shards=${shards} $*"
  n_jobs=$((n_jobs + 1))
  if [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi

  # $SUBSET_FLAG and $MAX_EX_FLAG are used WITHOUT quotes on purpose: when one
  # is set to "" (empty), an unquoted "" vanishes entirely from the command
  # line instead of being passed as a literal empty argument; when it holds
  # several words (e.g. "--subset en_ewt de_pud"), unquoted lets it split back
  # into separate arguments the way it needs to.
  sbatch --partition="$part" --gres=gpu:"$gpus" --array="$idx" --time="$wtime" \
    --export=ALL,SHARDS="$shards",UNER_CHUNK="$UNER_CHUNK_SPEC" \
    slurm/scripts/NER/evaluationNER_cons_gen.batch \
    --dataset "$DATASET" $SUBSET_FLAG --batch_size "$BS" $MAX_EX_FLAG \
    --max-new-tokens "$mnt" --seeds $SEEDS "$@"
}

UNER_CHUNK_SPEC="0/1"   # overridden inside the UNER loop below
DATASET="conll2003"
TAG="conll2003"
SUBSET_FLAG=""

# Model registry indices once again, for reference:
#   idx 0  gemma-4-E2B-it    -> reasoning ON and OFF
#   idx 1  gemma-4-31B-it    -> reasoning OFF only
#   idx 2  Qwen3.8-27B       -> reasoning ON at effort=low, and OFF
#   idx 3  gpt-oss-20b       -> effort low|medium
#   idx 4  gpt-oss-120b      -> effort low|medium
#   idx 5  Qwen3-8B          -> reasoning OFF only
# submit <partition> <gpus> <array_idx> <max_new_tokens> <time> [extra args...]
BS=1; if [ "$PILOT" -eq 1 ]; then MAX_EX_FLAG="--max-examples 25"; else MAX_EX_FLAG="--max-examples 1152"; fi
# submit amdgpufast 1 0 18000 4:00:00  2   --no-enable-thinking

# submit h200       2 0 18000 12:00:00  10   --enable-thinking

# submit h200       1 1 18000 5:00:00 1   --no-enable-thinking

# submit h200       1 2 18000 5:00:00  2   --no-enable-thinking
# submit h200       2 2 18000 12:00:00 4   --enable-thinking --reasoning-effort low

# submit h200fast   1 3 18000 4:00:00  5   --reasoning-effort low
# submit h200       2 3 18000 10:00:00  8   --reasoning-effort medium

# submit h200       1 4 18000 15:00:00 1   --reasoning-effort low
# submit h200       4 4 18000 10:00:00 4   --reasoning-effort medium

# submit amdgpufast 1 5 18000 4:00:00  2   --no-enable-thinking


BS=64; if [ "$PILOT" -eq 1 ]; then MAX_EX_FLAG="--max-examples 128"; else MAX_EX_FLAG="--max-examples 1152"; fi
# submit amdgpufast 1 0 18000 4:00:00 1   --no-enable-thinking
# submit amdgpufast 1 0 18000 4:00:00 3   --enable-thinking

# submit h200       1 1 18000 6:00:00 1  --no-enable-thinking

# submit h200fast   1 2 18000 4:00:00 2  --no-enable-thinking
# submit h200fast   2 2 18000 4:00:00 4  --enable-thinking --reasoning-effort low

# submit h200fast   1 3 18000 4:00:00 1   --reasoning-effort low
# submit h200fast   1 3 18000 4:00:00 4   --reasoning-effort medium

# submit h200fast   1 4 18000 4:00:00 1   --reasoning-effort low
# submit h200       1 4 18000 11:00:00 1  --reasoning-effort medium

# submit amdgpufast 1 5 18000 4:00:00 1   --no-enable-thinking


# Model registry indices once again, for reference:
#   idx 0  gemma-4-E2B-it    -> reasoning ON and OFF
#   idx 1  gemma-4-31B-it    -> reasoning OFF only
#   idx 2  Qwen3.8-27B       -> reasoning ON at effort=low, and OFF
#   idx 3  gpt-oss-20b       -> effort low|medium
#   idx 4  gpt-oss-120b      -> effort low|medium
#   idx 5  Qwen3-8B          -> reasoning OFF only
DATASET="uner"
# One iteration per JOB. Each job is told which window of the global packing it owns via UNER_CHUNK_SPEC
for i in $(seq 0 $((UNER_CHUNKS - 1))); do
  SUBSET_FLAG=""
  UNER_CHUNK_SPEC="$i/$UNER_CHUNKS" # --chunk i/N is used by evaluationNER_cons_gen.batch
  TAG="uner/chunk$((i + 1))of${UNER_CHUNKS}"

  BS=1; if [ "$PILOT" -eq 1 ]; then MAX_EX_FLAG="--max-examples 25"; else MAX_EX_FLAG="--max-examples 250"; fi
  # submit h200fast     2 0 18000 4:00:00   10   --no-enable-thinking
  #   -> [Claude est. v2] recommend --time 4:30:00 (currently 4:00:00) -- v2 rec ~4.2h, cutting it close under the 4h h200fast wall
  # submit h200       3 0 18000 24:00:00  15   --enable-thinking

  # submit h200       3 1 18000 9:00:00  3   --no-enable-thinking

  # submit h200         2 2 18000 14:00:00  4   --no-enable-thinking

  # UNER_CHUNKS=2 -- run separately: UNER_CHUNKS=2 PILOT=0 bash slurm/scripts/NER/submit_pilot_cg.sh
  submit h200       3 2 18000 16:00:00  6   --enable-thinking --reasoning-effort low
  #   -> [Claude est. v2] recommend --time 20:30:00 (currently 16:00:00) -- v2 rec ~19.8h at chunks=2
  
  # submit h200         2 3 18000 12:00:00  8   --reasoning-effort low
  #   -> [Claude est. v2] recommend --time 21:00:00 (currently 12:00:00) -- v2 rec ~20.5h
  # submit h200         2 3 18000 20:00:00 12   --reasoning-effort medium
  #   -> [Claude est. v2] recommend --time 41:00:00 on h200long (currently h200/20:00:00) -- v2 rec ~40.6h, exceeds the 24h h200 wall

  # submit h200       3 4 18000 10:00:00  3  --reasoning-effort low
  # submit h200       3 4 18000 20:00:00  3  --reasoning-effort medium

  # submit h200         1 5 18000 8:00:00   6   --no-enable-thinking

  BS=64; if [ "$PILOT" -eq 1 ]; then MAX_EX_FLAG="--max-examples 128"; else MAX_EX_FLAG="--max-examples 250"; fi
  # submit h200fast     1 0 18000 4:00:00   10   --no-enable-thinking
  # submit h200         2 0 18000 10:00:00  12   --enable-thinking
  
  # submit h200       3 1 18000 8:00:00   3   --no-enable-thinking

  # submit h200         2 2 18000 14:00:00  4   --no-enable-thinking

  # UNER_CHUNKS=2 -- run separately: UNER_CHUNKS=2 PILOT=0 bash slurm/scripts/NER/submit_pilot_cg.sh
  submit h200       2 2 18000 14:00:00  4   --enable-thinking --reasoning-effort low
  #   -> [Claude est. v2] recommend --time 15:30:00 (currently 14:00:00) -- v2 rec ~14.8h at chunks=2

  # submit h200fast     1 3 18000 4:00:00   6   --reasoning-effort low
  # submit h200         2 3 18000 8:00:00  12   --reasoning-effort medium

  # submit h200fast     3 4 18000 4:00:00   3   --reasoning-effort low
  #   -> [Claude est. v2] recommend --time 8:30:00 on h200 (currently h200fast/4:00:00) -- v2 rec ~8.0h, exceeds the 4h h200fast wall

  # UNER_CHUNKS=2
  # submit h200       3 4 18000 5:00:00   3   --reasoning-effort medium

  # submit h200         1 5 18000 6:00:00   6   --no-enable-thinking
done

echo
echo "PILOT=$PILOT  DRY_RUN=$DRY_RUN  UNER_CHUNKS=$UNER_CHUNKS  seeds='$SEEDS'"
echo "total jobs: $n_jobs"
if [ "$n_jobs" -gt 200 ]; then
  echo "WARNING: $n_jobs jobs exceeds the 200-job QOS cap -- lower UNER_CHUNKS."
fi
