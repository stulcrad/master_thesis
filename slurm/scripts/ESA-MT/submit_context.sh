#!/bin/bash
#SBATCH --partition=amd
#SBATCH --output=/home/stulcrad/master_thesis/logs/pilot_scripts/out/%x_%N_%A.out
#SBATCH --error=/home/stulcrad/master_thesis/logs/pilot_scripts/err/%x_%N_%A.err
#
# Submitter for the WMT24 ESA CONTEXT-BASED (prompting) BASELINE line.
#
# Same shape as submit_pilot_cg.sh next to it -- read that file for the reasoning
# behind the arms and the chunking. Two deliberate differences:
#
#   1. A100 TARGET. The baseline is the cheap arm (a short JSON list per example,
#      not a verbatim copy of the input), so it is sized for the amdgpu/amdgpufast
#      A100 partitions rather than for h200 -- shorter queue, and the h200 GPUs
#      stay free for the constrained runs. A100s here are 40 GB, which is the
#      binding constraint on which models run and on how many processes fit:
#
#        idx 0  gemma-4-E2B-it   ~12 GiB/proc  -> 3 processes per GPU
#        idx 5  Qwen3-8B         ~19 GiB/proc  -> 2 processes per GPU
#        idx 3  gpt-oss-20b      bf16 on A100  -> 2 GPUs, SHARDS=1
#        idx 2  Qwen3.8-27B      ~65 GiB       -> 2 GPUs, SHARDS=1
#        idx 1  gemma-4-31B-it   ~72 GiB       -> 2 GPUs, SHARDS=1
#        idx 4  gpt-oss-120b     ~240 GiB      -> does NOT fit the 4xA100 nodes;
#                                                 h200 or g11/g12 only.
#
#   2. The seeds match the constrained generation arms, so that the two lines are comparable.
#
# SHARDS > 1 splits by SUBSET GROUP, never by example index -- the headline is a
# MACRO average over the 24 language-domain files and each file already writes
# its own CSV row, so index-splitting would force merging partial rows before
# averaging. bin_packing does the packing ONCE, globally: with WMT_CHUNKS > 1 it
# packs into CHUNKS*SHARDS groups in a single pass and hands each job its window.
#
#     bash slurm/scripts/ESA-MT/submit_context.sh                  # pilot
#     PILOT=0 bash slurm/scripts/ESA-MT/submit_context.sh          # full run
#     DRY_RUN=1 bash slurm/scripts/ESA-MT/submit_context.sh        # print only
#     WMT_CHUNKS=2 PILOT=0 bash slurm/scripts/ESA-MT/submit_context.sh
#
# Check the indices with: python -m utils.model_registry --list
set -eu
cd ~/master_thesis

source init_environment_python.sh

PILOT="${PILOT:-1}"
DRY_RUN="${DRY_RUN:-0}"
WMT_CHUNKS="${WMT_CHUNKS:-1}"
# Guard: WMT_CHUNKS=0 would make `seq 0 -1` emit nothing, silently submitting no
# jobs at all -- no error, just a missing dataset noticed days later.
case "$WMT_CHUNKS" in
  ''|*[!0-9]*|0) echo "WMT_CHUNKS must be a positive integer, got '$WMT_CHUNKS'" >&2; exit 1 ;;
esac

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
  echo "submit: ${TAG} array=${idx} part=${part} time=${wtime} shards=${shards}  $*"
  n_jobs=$((n_jobs + 1))
  if [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  # $MAX_EX_FLAG is deliberately UNQUOTED: either empty (and must vanish) or two
  # words that must word-split. No --wmt-subset is passed here on purpose: the
  # batch script derives this job's subset groups from WMT_CHUNK, and passing a
  # subset list as well would make bin_packing pack twice.
  sbatch --partition="$part" --gres=gpu:"$gpus" --array="$idx" --time="$wtime" \
    --export=ALL,SHARDS="$shards",WMT_CHUNK="$WMT_CHUNK_SPEC" \
    slurm/scripts/ESA-MT/evaluationWMT_HF_context.batch \
    $MAX_EX_FLAG --max-new-tokens "$mnt" --seeds $SEEDS "$@"
}

# Model registry indices, for reference:
#   idx 0  gemma-4-E2B-it    -> reasoning ON and OFF
#   idx 1  gemma-4-31B-it    -> reasoning OFF only
#   idx 2  Qwen3.8-27B       -> reasoning ON at effort=low, and OFF
#   idx 3  gpt-oss-20b       -> effort low|medium
#   idx 4  gpt-oss-120b      -> h200 only
#   idx 5  Qwen3-8B          -> reasoning OFF only


# CURRENT SCOPE (pending supervisor sign-off on whether this is enough of a baseline):
# only idx0 gemma-4-E2B-it (ON+OFF), idx2 Qwen3.8-27B (OFF), idx3 gpt-oss-20b (low).
# idx1 gemma-4-31B-it, idx5 Qwen3-8B, gpt-oss-20b medium and any reasoning-ON arm for
# Qwen3.8-27B stay commented-out reference lines below -- bigger models are CG-only
# for now, decide later. idx3 (gpt-oss-20b, low) IS wanted but stays commented here
# too -- it needs its own WMT_CHUNKS=2 invocation (see below), which would also
# needlessly re-split the cheap OFF arms above if folded into this default run.
#
#   idx0 OFF   3.5h (WMT_CHUNKS=1)   | idx0 ON  13.2h (WMT_CHUNKS=1)
#   idx2 OFF   6.4h (WMT_CHUNKS=1)
#   idx3 low  15.4h/job (WMT_CHUNKS=2, own invocation)
# ---------------------------------------------------------------------------
for i in $(seq 0 $((WMT_CHUNKS - 1))); do
  WMT_CHUNK_SPEC="$i/$WMT_CHUNKS"
  TAG="wmt/chunk$((i + 1))of${WMT_CHUNKS}"

  # submit amdgpufast 1 0 18000 4:00:00   3   --no-enable-thinking
  # submit amdgpu     2 0 18000 15:00:00  6   --enable-thinking

  # submit amdgpufast 1 5 18000 4:00:00   2   --no-enable-thinking

  # submit amdgpu     2 2 18000 8:00:00   1   --no-enable-thinking

  # gpt-oss-20b needs 2 GPUs for its one process, so SHARDS is stuck at 1 -- the
  # fix from Sep 3 lets --chunk still apply at SHARDS=1, so WMT_CHUNKS genuinely
  # splits it by subset group (previously every chunk-job would have silently
  # re-run the full 24-file corpus). WANTED, but needs its own invocation since
  # WMT_CHUNKS is one knob for the whole script:
  #   WMT_CHUNKS=2 PILOT=0 bash slurm/scripts/ESA-MT/submit_context.sh
  submit amdgpu     2 3 18000 16:00:00  1   --reasoning-effort low

  # gpt-oss-20b medium: not wanted for now, needs WMT_CHUNKS=3 if it is later.
  # submit amdgpu     2 3 18000 17:00:00  1   --reasoning-effort medium

  # submit amdgpu     2 2 18000 20:00:00  1   --reasoning-effort low
  # submit amdgpu     2 1 18000 7:00:00   1   --no-enable-thinking
done

echo
echo "PILOT=$PILOT  DRY_RUN=$DRY_RUN  WMT_CHUNKS=$WMT_CHUNKS  seeds='$SEEDS'"
echo "total jobs: $n_jobs"
if [ "$n_jobs" -gt 200 ]; then
  echo "WARNING: $n_jobs jobs exceeds the 200-job QOS cap -- lower WMT_CHUNKS."
fi
