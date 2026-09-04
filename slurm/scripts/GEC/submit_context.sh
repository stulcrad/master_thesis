#!/bin/bash
#SBATCH --partition=amd
#SBATCH --output=/home/stulcrad/master_thesis/logs/pilot_scripts/out/%x_%N_%A.out
#SBATCH --error=/home/stulcrad/master_thesis/logs/pilot_scripts/err/%x_%N_%A.err
#
# Submitter for the MultiGEC CONTEXT-BASED (prompting) BASELINE line.
#
# Same shape as submit_pilot_cg.sh next to it -- read that file for the reasoning behind the arms. Two deliberate
# differences:
#
#   1. A100 TARGET. The baseline is the cheap arm (a short JSON list per example, not a verbatim copy of the input), so
#      it is sized for the amdgpu/amdgpufast A100 partitions rather than for h200 -- shorter queue, and the h200 GPUs
#      stay free for the constrained runs. A100s here are 40 GB, which is the binding constraint on which models run and
#      on how many processes fit:
#
#        idx 0  gemma-4-E2B-it   ~12 GiB/proc  -> 3 processes per GPU
#        idx 5  Qwen3-8B         ~19 GiB/proc  -> 2 processes per GPU
#        idx 3  gpt-oss-20b      bf16 on A100  -> 2 GPUs, SHARDS=1
#        idx 2  Qwen3.8-27B      ~65 GiB       -> 2 GPUs, SHARDS=1
#        idx 1  gemma-4-31B-it   ~72 GiB       -> 2 GPUs, SHARDS=1
#        idx 4  gpt-oss-120b     ~240 GiB      -> does NOT fit the 4xA100 nodes;
#                                                 h200 or g11/g12 only.
#
#   2. THE EXAMPLE CAP AND THE SEEDS MATCH THE CG RUNS.
#
#     bash slurm/scripts/GEC/submit_context.sh                  # pilot
#     PILOT=0 bash slurm/scripts/GEC/submit_context.sh          # full run
#     DRY_RUN=1 bash slurm/scripts/GEC/submit_context.sh        # print, submit nothing
#
# Check the indices with: python -m utils.model_registry --list
set -eu
cd ~/master_thesis

source init_environment_python.sh

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
  echo "submit: multigec array=${idx} part=${part} time=${wtime} shards=${shards}  $*"
  n_jobs=$((n_jobs + 1))
  if [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  # $MAX_EX_FLAG is deliberately UNQUOTED: either empty (and must vanish) or two
  # words that must word-split.
  sbatch --partition="$part" --gres=gpu:"$gpus" --array="$idx" --time="$wtime" \
    --export=ALL,SHARDS="$shards" \
    slurm/scripts/GEC/evaluationMultiGEC_HF_context.batch \
    $MAX_EX_FLAG --max-new-tokens "$mnt" --seeds $SEEDS "$@"
}

# Model registry indices, for reference:
#   idx 0  gemma-4-E2B-it    -> reasoning ON and OFF
#   idx 1  gemma-4-31B-it    -> reasoning OFF only
#   idx 2  Qwen3.8-27B       -> reasoning ON at effort=low, and OFF
#   idx 3  gpt-oss-20b       -> effort low|medium
#   idx 4  gpt-oss-120b      -> h200 only
#   idx 5  Qwen3-8B          -> reasoning OFF only

#
# CURRENT SCOPE (pending supervisor sign-off on whether this is enough of a baseline):
# only idx0 gemma-4-E2B-it (ON+OFF), idx2 Qwen3.8-27B (OFF), idx3 gpt-oss-20b (low).
# idx1 gemma-4-31B-it, idx5 Qwen3-8B, gpt-oss-20b medium and any reasoning-ON arm for
# Qwen3.8-27B stay commented-out reference lines below -- bigger models are CG-only
# for now, decide later.
#
#   idx 0 OFF   6.6h (1 job)  | idx 0 ON  12.4h/job (3 jobs)  | idx 5 OFF  8.6h (1 job)
#   idx 3 low  13.9h/job (4 jobs)  | idx 3 med 17.8h/job (21 jobs -- 374h total, do NOT run)
#   idx 2 OFF   9.1h/job (2 jobs)  | idx 2 ON(low) 15.5h/job (5 jobs)
#
# MultiGEC is the expensive dataset for THIS arm specifically: it is the only one where the
# baseline's answer is long. ~14 gold spans per essay means a ~340-token JSON array, where every
# other dataset asks for 18-107 tokens. Budget accordingly and shard it.

submit amdgpu     1 0 18000 8:00:00   3   --no-enable-thinking

submit amdgpu     2 0 18000 14:00:00  6   --enable-thinking --shard 0/3
submit amdgpu     2 0 18000 14:00:00  6   --enable-thinking --shard 1/3
submit amdgpu     2 0 18000 14:00:00  6   --enable-thinking --shard 2/3

# submit amdgpu     1 5 18000 10:00:00  2   --no-enable-thinking

# gpt-oss-20b needs 2 GPUs for its one process (SHARDS stuck at 1), so splitting the corpus
# across jobs is the ONLY lever -- 4 jobs of ~14h each, comfortably under the 24h amdgpu wall.
submit amdgpu     2 3 18000 16:00:00  1   --reasoning-effort low --shard 0/4
submit amdgpu     2 3 18000 16:00:00  1   --reasoning-effort low --shard 1/4
submit amdgpu     2 3 18000 16:00:00  1   --reasoning-effort low --shard 2/4
submit amdgpu     2 3 18000 16:00:00  1   --reasoning-effort low --shard 3/4

# 374h total across the arm even at 21-way sharding -- not worth running for a baseline row.
# submit amdgpu     2 3 18000 18:00:00  1   --reasoning-effort medium --shard i/21   # (i = 0..20)

submit amdgpu     2 2 18000 10:00:00  1   --no-enable-thinking --shard 0/2
submit amdgpu     2 2 18000 10:00:00  1   --no-enable-thinking --shard 1/2

# 5 jobs of ~15.5h each.
# submit amdgpu     2 2 18000 17:00:00  1   --enable-thinking --reasoning-effort low --shard 0/5
# submit amdgpu     2 2 18000 17:00:00  1   --enable-thinking --reasoning-effort low --shard 1/5
# submit amdgpu     2 2 18000 17:00:00  1   --enable-thinking --reasoning-effort low --shard 2/5
# submit amdgpu     2 2 18000 17:00:00  1   --enable-thinking --reasoning-effort low --shard 3/5
# submit amdgpu     2 2 18000 17:00:00  1   --enable-thinking --reasoning-effort low --shard 4/5

# submit amdgpu     2 1 18000 10:00:00  1   --no-enable-thinking

# submit h200       1 4 18000 18:00:00  1   --reasoning-effort low

echo
echo "PILOT=$PILOT  DRY_RUN=$DRY_RUN  seeds='$SEEDS'"
echo "total jobs: $n_jobs"
