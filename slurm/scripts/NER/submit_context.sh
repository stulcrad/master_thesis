#!/bin/bash
#SBATCH --partition=amdfast
#SBATCH --time=00:01:00
#SBATCH --output=/home/stulcrad/master_thesis/logs/pilot_scripts/out/%x_%N_%A.out
#SBATCH --error=/home/stulcrad/master_thesis/logs/pilot_scripts/err/%x_%N_%A.err
#
# Submitter for the NER/UNER CONTEXT-BASED (prompting) BASELINE line.
#
# Same shape as submit_pilot_cg.sh next to it -- read that file for the reasoning
# behind the arms and the chunking. Two deliberate differences:
#
#   1. A100 TARGET. The baseline is the cheap arm (a short JSON list per example,
#      not a verbatim copy of the input), so it is sized for the amdgpu/amdgpufast
#      A100 partitions rather than for h200 -- shorter queue, and the h200 GPUs
#      stay free for the constrained runs. A100s here are 40 GB, which is the
#      binding constraint on which models can run and on how many processes fit:
#
#        idx 0  gemma-4-E2B-it   ~12 GiB/proc  -> 3 processes per GPU
#        idx 5  Qwen3-8B         ~19 GiB/proc  -> 2 processes per GPU
#        idx 3  gpt-oss-20b      bf16 on A100  -> 2 GPUs, SHARDS=1
#        idx 2  Qwen3.8-27B      ~65 GiB       -> 2 GPUs, SHARDS=1
#        idx 1  gemma-4-31B-it   ~72 GiB       -> 2 GPUs, SHARDS=1
#        idx 4  gpt-oss-120b     ~240 GiB      -> does NOT fit the A100 nodes at
#                                                 4 GPUs; h200 or g11/g12 only.
#
#   2. EXAMPLE CAPS ARE THE SAME AS THE CG RUNS (CoNLL 1152, UNER 250 per
#      treebank) and so are the seeds.
#
# PILOT vs FULL
#     bash slurm/scripts/NER/submit_context.sh                  # pilot
#     PILOT=0 bash slurm/scripts/NER/submit_context.sh          # full run
#     DRY_RUN=1 bash slurm/scripts/NER/submit_context.sh        # print, submit nothing
#     UNER_CHUNKS=4 PILOT=0 bash slurm/scripts/NER/submit_context.sh
#
# Check the indices with: python -m utils.model_registry --list

# -e: stop immediately if any command fails. -u: an unset variable is an error,
# which catches typos like $SUSBET_FLAG.
set -eu
cd ~/master_thesis

source init_environment_python.sh

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

# submit <partition> <gpus> <array_idx> <max_new_tokens> <time> <shards> [extra args...]
submit() {
  local part=$1 gpus=$2 idx=$3 mnt=$4 wtime=$5 shards=$6; shift 6
  case "$wtime" in
    [0-9]*:[0-9][0-9]:[0-9][0-9]|[0-9]-[0-9]*:[0-9][0-9]:[0-9][0-9]) ;;
    *) echo "submit: wtime must be HH:MM:SS, got '$wtime' -- a positional arg is missing" >&2
       exit 1 ;;
  esac
  for v in "$gpus" "$idx" "$mnt" "$shards"; do
    case "$v" in ''|*[!0-9]*)
      echo "submit: expected a number, got '$v' -- a positional arg is missing" >&2
      exit 1 ;; esac
  done
  echo "submit: ${TAG} bs=${BS} array=${idx} part=${part} time=${wtime} shards=${shards} $*"
  n_jobs=$((n_jobs + 1))
  if [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi

  # $SUBSET_FLAG and $MAX_EX_FLAG are used WITHOUT quotes on purpose: when one is
  # empty it vanishes entirely instead of being passed as a literal empty
  # argument, and when it holds several words it splits back into arguments.
  sbatch --partition="$part" --gres=gpu:"$gpus" --array="$idx" --time="$wtime" \
    --export=ALL,SHARDS="$shards",UNER_CHUNK="$UNER_CHUNK_SPEC" \
    slurm/scripts/NER/evaluationNER_HF_context.batch \
    --dataset "$DATASET" $SUBSET_FLAG --batch_size "$BS" $MAX_EX_FLAG \
    --max-new-tokens "$mnt" --seeds $SEEDS "$@"
}

UNER_CHUNK_SPEC="0/1"   # overridden inside the UNER loop below
DATASET="conll2003"
TAG="conll2003"
SUBSET_FLAG=""

# ---------------------------------------------------------------------------
# CURRENT SCOPE (pending supervisor sign-off on whether this is enough of a baseline):
# only idx0 gemma-4-E2B-it (ON+OFF), idx2 Qwen3.8-27B (OFF), idx3 gpt-oss-20b (low).
# idx1 gemma-4-31B-it, idx5 Qwen3-8B, gpt-oss-20b medium and any reasoning-ON arm for
# Qwen3.8-27B stay commented-out reference lines below -- bigger models are CG-only
# for now, decide later.

# ---------------------------------------------------------------------------
# CoNLL-2003. Cap 1152 examples, identical to the constrained runs.
# ---------------------------------------------------------------------------
#   idx 0  gemma-4-E2B-it    -> reasoning ON and OFF
#   idx 1  gemma-4-31B-it    -> reasoning OFF only
#   idx 2  Qwen3.8-27B       -> reasoning ON at effort=low, and OFF
#   idx 3  gpt-oss-20b       -> effort low|medium
#   idx 4  gpt-oss-120b      -> h200 only, does not fit the 4xA100 nodes
#   idx 5  Qwen3-8B          -> reasoning OFF only
# submit <partition> <gpus> <array_idx> <max_new_tokens> <time> <shards> [extra args...]
BS=1; if [ "$PILOT" -eq 1 ]; then MAX_EX_FLAG="--max-examples 25"; else MAX_EX_FLAG="--max-examples 1152"; fi
# submit amdgpufast 1 0 18000 4:00:00   3   --no-enable-thinking
# submit amdgpu     2 0 18000 15:00:00  6   --enable-thinking

# submit amdgpu     2 3 18000 12:00:00  1   --reasoning-effort low

# submit amdgpu     2 2 18000 8:00:00   1   --no-enable-thinking


# submit amdgpufast 1 5 18000 4:00:00   2   --no-enable-thinking


# 23.2h in one job (Qwen3.8-27B needs 2 GPUs for its single process, so no in-job packing is
# possible) -- split across 2 jobs by example index instead, each ~11.6h.
# submit amdgpu     2 2 18000 13:00:00  1   --reasoning-effort low --shard 0/2
# submit amdgpu     2 2 18000 13:00:00  1   --reasoning-effort low --shard 1/2

# 40.1h in one job -- split across 3 jobs by example index, each ~13.4h.
# submit amdgpu     2 3 18000 14:00:00  1   --reasoning-effort medium --shard 0/3
# submit amdgpu     2 3 18000 14:00:00  1   --reasoning-effort medium --shard 1/3
# submit amdgpu     2 3 18000 14:00:00  1   --reasoning-effort medium --shard 2/3


# submit amdgpu     2 1 18000 4:00:00   1   --no-enable-thinking

# submit h200       1 4 18000 18:00:00  1   --reasoning-effort low

BS=64; if [ "$PILOT" -eq 1 ]; then MAX_EX_FLAG="--max-examples 128"; else MAX_EX_FLAG="--max-examples 1152"; fi
# submit amdgpufast 1 0 18000 4:00:00   3   --no-enable-thinking
# submit amdgpu     1 0 18000 8:00:00   3   --enable-thinking
# submit amdgpu     2 3 18000 8:00:00   1   --reasoning-effort low

# submit amdgpu     2 2 18000 6:00:00   1   --no-enable-thinking


# submit amdgpufast 1 5 18000 4:00:00   2   --no-enable-thinking

# ---------------------------------------------------------------------------
# UniversalNER. Cap 250 per treebank, identical to the constrained runs.
#
# UNER_CHUNKS is ONE global knob for this whole script invocation -- every active
# submit() call below is split into that many chunk-jobs, via the SAME loop. Arms
# need DIFFERENT chunk counts (see the estimate block above), so the default
# invocation below (UNER_CHUNKS=1) only activates arms that actually fit in one
# job; the reasoning-ON arms are commented out with the UNER_CHUNKS value they
# need -- invoke the script AGAIN with that value to add them, rather than
# uncommenting them here (which would also split the cheap OFF arms needlessly).
# ---------------------------------------------------------------------------
DATASET="uner"
# One iteration per JOB. Each job is told which window of the global packing it owns via UNER_CHUNK_SPEC.
for i in $(seq 0 $((UNER_CHUNKS - 1))); do
  SUBSET_FLAG=""
  UNER_CHUNK_SPEC="$i/$UNER_CHUNKS"   # --chunk i/N is used by evaluationNER_HF_context.batch
  TAG="uner/chunk$((i + 1))of${UNER_CHUNKS}"

  BS=1; if [ "$PILOT" -eq 1 ]; then MAX_EX_FLAG="--max-examples 25"; else MAX_EX_FLAG="--max-examples 250"; fi
  # submit amdgpu     2 0 18000 4:00:00   6   --no-enable-thinking

  # Qwen3.8-27B OFF: 1 job, ~10.4h.
  # submit amdgpu     2 2 18000 12:00:00  1   --no-enable-thinking

  # Reasoning ON: needs UNER_CHUNKS=3 (~12.7h/job). Run separately:
  #   UNER_CHUNKS=3 PILOT=0 bash slurm/scripts/NER/submit_context.sh
  # (then comment out the OFF-arm lines above for that invocation, or accept they
  # split into 3 needless-but-harmless extra jobs).
  submit amdgpu     2 0 18000 14:00:00  6   --enable-thinking

  # gpt-oss-20b low: needs UNER_CHUNKS=3 (~14.9h/job).
  submit amdgpu     2 3 18000 16:00:00  1   --reasoning-effort low

  # gpt-oss-20b medium: needs UNER_CHUNKS=13 (~17.1h/job, 811 GPU-job-h total -- consider skipping).
  # submit amdgpu     2 3 18000 18:00:00  1   --reasoning-effort medium


  # Qwen3.8-27B low: needs UNER_CHUNKS=4 (~15.8h/job).
  # submit amdgpu     2 2 18000 17:00:00  1   --reasoning-effort low
  
  # gemma-4-31B-it OFF: 1 job, ~9.3h.
  # submit amdgpu     2 1 18000 11:00:00  1   --no-enable-thinking

  # submit amdgpu     2 5 18000 6:00:00   4   --no-enable-thinking
done

# ---------------------------------------------------------------------------
# UniversalNER, bs=64. One generation per 64-example batch within each treebank:
# 3702 capped examples -> 61 generations total (not 3702), which is why every one
# of the 4 target arms fits in a SINGLE job here -- no UNER_CHUNKS needed at all,
# unlike bs=1 above. SUBSET_FLAG/UNER_CHUNK_SPEC reset to "no chunking" explicitly
# since we're outside the UNER_CHUNKS loop this time.
# ---------------------------------------------------------------------------
SUBSET_FLAG=""
UNER_CHUNK_SPEC="0/1"
TAG="uner_bs64"
BS=64; if [ "$PILOT" -eq 1 ]; then MAX_EX_FLAG="--max-examples 25"; else MAX_EX_FLAG="--max-examples 250"; fi
# submit amdgpufast 1 0 18000 4:00:00   3   --no-enable-thinking
# submit amdgpu     1 0 18000 10:00:00  3   --enable-thinking

# submit amdgpu     2 2 18000 18:00:00  1   --no-enable-thinking

# submit amdgpu     2 3 18000 10:00:00  1   --reasoning-effort low

echo
echo "PILOT=$PILOT  DRY_RUN=$DRY_RUN  UNER_CHUNKS=$UNER_CHUNKS  seeds='$SEEDS'"
echo "total jobs: $n_jobs"
if [ "$n_jobs" -gt 200 ]; then
  echo "WARNING: $n_jobs jobs exceeds the 200-job QOS cap -- lower UNER_CHUNKS."
fi
