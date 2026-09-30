#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --partition=amd
#SBATCH --output=/home/stulcrad/master_thesis/logs/pilot_scripts/out/%x_%N_%A.out
#SBATCH --error=/home/stulcrad/master_thesis/logs/pilot_scripts/err/%x_%N_%A.err
#
# Submitter for the Toxic Spans CONTEXT-BASED (prompting) BASELINE line.
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
#     bash slurm/scripts/ToxicSpans/submit_context.sh                  # pilot
#     PILOT=0 bash slurm/scripts/ToxicSpans/submit_context.sh          # full run
#     DRY_RUN=1 bash slurm/scripts/ToxicSpans/submit_context.sh        # print, submit nothing
#
# Check the indices with: python -m utils.model_registry --list
set -eu
cd ~/master_thesis

source init_environment_python.sh

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
  echo "submit: toxic array=${idx} part=${part} time=${wtime} shards=${shards}  $*"
  n_jobs=$((n_jobs + 1))
  if [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  # $MAX_EX_FLAG is deliberately UNQUOTED: either empty (and must vanish) or two
  # words that must word-split.
  sbatch --partition="$part" --gres=gpu:"$gpus" --array="$idx" --time="$wtime" \
    --export=ALL,SHARDS="$shards" \
    slurm/scripts/ToxicSpans/evaluationToxicSpans_HF_context.batch \
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


submit amdgpufast 1 0 18000 4:00:00   3   --no-enable-thinking
#   -> [Claude est. v2] --time 1:00:00 (currently 4:00:00) | ~0.6h + 25% margin.
#      Measured: job 11481017_0 (0.64h) at 3 shards = 334 ex/shard.
submit amdgpu     2 0 18000 14:00:00  6   --enable-thinking
#   -> [Claude est. v2] --time 6:30:00 (currently 14:00:00) | ~2.4h + 150% margin.
#      ROUGH transfer (+/-3x on back-test): no toxic run of this arm -- from job 11477957_0 (4.36h) on conll2003, x0.64 (toxic/conll2003 ratio on the OFF arm) at 1 shard = 1000 ex/shard; this line is 6 shards = 167 ex/shard.

# submit amdgpufast 1 5 18000 4:00:00   2   --no-enable-thinking
#   -> [Claude est. v2] NO TIMING DATA for Qwen3-8B OFF on toxic. This line is
#      2 shards = ~500 ex/shard; run one seed first rather than guessing.

submit amdgpu     2 3 18000  8:00:00  1   --reasoning-effort low
#   -> [Claude est. v2] --time 19:30:00 (currently 8:00:00) | ~7.6h + 150% margin.
#      ROUGH transfer (+/-3x on back-test): no toxic run of this arm -- from timeout 11477958_3 (12.0h = 88.4% of rows) on conll2003, x0.64 (toxic/conll2003 ratio on the gemma-4-E2B-it OFF arm) at 1 shard = 1000 ex/shard.

# submit amdgpu     2 3 18000 14:00:00  1   --reasoning-effort medium --shard 0/3
#   -> [Claude est. v2] NO TIMING DATA for gpt-oss-20b ON/medium on toxic. This line is
#      1 shards, --shard x/3 = ~334 ex/shard; run one seed first rather than guessing.
# submit amdgpu     2 3 18000 14:00:00  1   --reasoning-effort medium --shard 1/3
#   -> [Claude est. v2] NO TIMING DATA for gpt-oss-20b ON/medium on toxic. This line is
#      1 shards, --shard x/3 = ~334 ex/shard; run one seed first rather than guessing.
# submit amdgpu     2 3 18000 14:00:00  1   --reasoning-effort medium --shard 2/3
#   -> [Claude est. v2] NO TIMING DATA for gpt-oss-20b ON/medium on toxic. This line is
#      1 shards, --shard x/3 = ~334 ex/shard; run one seed first rather than guessing.

submit amdgpufast 2 2 18000 4:00:00   1   --no-enable-thinking
#   -> [Claude est. v2] --time 8:00:00 (currently 4:00:00) | ~3.1h + 150% margin.  ** over the 4h amdgpufast wall: use *long, or more shards/chunks **
#      ROUGH transfer (+/-3x on back-test): no toxic run of this arm -- from job 11477959_2 (5.20h) (geometric mean of 2 transfers) on conll2003, x0.64 (toxic/conll2003 ratio on the gemma-4-E2B-it OFF arm) at 1 shard = 1000 ex/shard.

# submit amdgpu     2 2 18000 14:00:00  1   --enable-thinking --reasoning-effort low --shard 0/2
#   -> [Claude est. v2] NO TIMING DATA for Qwen3.8-27B ON/low on toxic. This line is
#      1 shards, --shard x/2 = ~500 ex/shard; run one seed first rather than guessing.
# submit amdgpu     2 2 18000 14:00:00  1   --enable-thinking --reasoning-effort low --shard 1/2
#   -> [Claude est. v2] NO TIMING DATA for Qwen3.8-27B ON/low on toxic. This line is
#      1 shards, --shard x/2 = ~500 ex/shard; run one seed first rather than guessing.

# submit amdgpu     2 1 18000 4:00:00   1   --no-enable-thinking
#   -> [Claude est. v2] NO TIMING DATA for gemma-4-31B-it OFF on toxic. This line is
#      1 shards = ~1000 ex/shard; run one seed first rather than guessing.

# submit h200       1 4 18000 18:00:00  1   --reasoning-effort low
#   -> [Claude est. v2] NO TIMING DATA for gpt-oss-120b ON/low on toxic. This line is
#      1 shards = ~1000 ex/shard; run one seed first rather than guessing.

echo
echo "PILOT=$PILOT  DRY_RUN=$DRY_RUN  seeds='$SEEDS'"
echo "total jobs: $n_jobs"
