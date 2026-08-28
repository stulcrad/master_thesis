#!/bin/bash
#SBATCH --output=/home/stulcrad/master_thesis/logs/pilot_scripts/out/%x_%N_%A.out
#SBATCH --error=/home/stulcrad/master_thesis/logs/pilot_scripts/err/%x_%N_%A.err
#
# Submitter for the NER/UNER constrained-generation line.
#
# PILOT vs FULL
#     bash slurm/scripts/NER/submit_pilot_cg.sh                  # pilot
#     PILOT=0 bash slurm/scripts/NER/submit_pilot_cg.sh          # full run
#     DRY_RUN=1 bash slurm/scripts/NER/submit_pilot_cg.sh        # print, submit nothing
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
#
# Each arm/batch-size runs on the partition+time tier matching its projected
# runtime: fast (4h, amdgpufast/h200fast), normal (10h, amdgpu/h200), day (24h,
# same partitions as normal), long (48h CoNLL / 30h UNER, amdgpulong/h200long).
#
# PAIRING. The subset is drawn with random.Random(seed), so seed 42 sees the SAME
# examples in every arm and on every model; analysis/significance.py intersects on
# the `key` field, so arms with different seed counts still pair on seed 42.

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

if [ "$PILOT" -eq 1 ]; then
  SEEDS="42"
else
  SEEDS="42 43 44"
fi

# UNER's 18 treebanks range from 18 examples (tl_trg) to 709 (en_ewt), so
# slicing the alphabetical list into equal-COUNT chunks can still give one
# chunk 3x the work of another. This asks Python (which already has the real
# per-treebank example counts) to split them into UNER_CHUNKS groups balanced
# by total example COUNT instead: biggest treebank first, always added to
# whichever chunk currently has the least work so far (greedy bin-packing).
# One line of output per chunk, e.g. "en_ewt de_pud hr_set".
CHUNKS_TEXT="$(python3 - "$UNER_CHUNKS" <<'PYEOF'
import sys, json
from pathlib import Path

n_chunks = int(sys.argv[1])
uner_dir = Path("data/semin/universal_ner")
subs = sorted(p.stem[len("uner_"):] for p in uner_dir.glob("uner_*.json"))
sizes = {s: len(json.loads((uner_dir / f"uner_{s}.json").read_text(encoding="utf-8")))
         for s in subs}

chunks = [[] for _ in range(n_chunks)]
loads = [0] * n_chunks
# Sort the treebanks by size, largest first, and assign each to the chunk with
# the least total size so far (greedy bin-packing).
for s in sorted(subs, key=lambda s: -sizes[s]):
    i = loads.index(min(loads))
    chunks[i].append(s)
    loads[i] += sizes[s]

for c in chunks:
    if c:  # more chunks than treebanks (UNER_CHUNKS > 18) -> fewer lines out
        print(" ".join(c))
PYEOF
)"
# mapfile reads that multi-line string into a bash array, one element per line
# (one element per chunk).
mapfile -t UNER_CHUNK_LIST <<< "$CHUNKS_TEXT"

n_jobs=0

# submit <partition> <gpus> <array_idx> <max_new_tokens> <time> [extra args...]
#
# $1 $2 $3 ... are this function's positional arguments (like argv). `local`
# copies them into named variables; `shift 6` then drops the first 6 so that
# "$@" (below) is just whatever extra flags were passed after them, e.g.
submit() {
  local part=$1 gpus=$2 idx=$3 mnt=$4 wtime=$5 shards=$6; shift 6
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
    --export=ALL,SHARDS="$shards" \
    slurm/scripts/NER/evaluationNER_cons_gen.batch \
    --dataset "$DATASET" $SUBSET_FLAG --batch_size "$BS" $MAX_EX_FLAG \
    --max-new-tokens "$mnt" --seeds $SEEDS "$@"
}

DATASET="conll2003"
TAG="conll2003"
SUBSET_FLAG=""

# Run models for CoNLL-2003, with BS=1 and BS=64.
# We had to divide it like this based on the expected runtimes of the different arms,
# so that each arm's jobs fit under the cluster's wall clock.

# Model registry indices once again, for reference:
#   idx 0  gemma-4-E2B-it    -> reasoning ON and OFF
#   idx 1  gemma-4-31B-it    -> reasoning OFF only
#   idx 2  Qwen3.8-27B       -> reasoning ON at effort=low, and OFF
#   idx 3  gpt-oss-20b       -> effort low|medium
#   idx 4  gpt-oss-120b      -> effort low|medium
#   idx 5  Qwen3-8B          -> reasoning OFF only
# submit <partition> <gpus> <array_idx> <max_new_tokens> <time> [extra args...]
BS=1; if [ "$PILOT" -eq 1 ]; then MAX_EX_FLAG="--max-examples 25"; else MAX_EX_FLAG="--max-examples 1152"; fi
submit amdgpufast 1 0 18000 4:00:00  2   --no-enable-thinking
submit h200       1 0 18000 8:00:00  8   --enable-thinking

submit h200       1 1 18000 10:00:00 1   --no-enable-thinking

submit h200       1 2 18000 5:00:00  2   --no-enable-thinking
submit h200       2 2 18000 18:00:00 4   --enable-thinking --reasoning-effort low

submit h200fast   1 3 18000 4:00:00  5   --reasoning-effort low
submit h200       1 3 18000 6:00:00  6   --reasoning-effort medium

submit h200       1 4 18000 15:00:00 1  --reasoning-effort low
submit h200       4 4 18000 15:00:00 4  --reasoning-effort medium

submit amdgpufast 1 5 18000 4:00:00  1   --no-enable-thinking


BS=64; if [ "$PILOT" -eq 1 ]; then MAX_EX_FLAG="--max-examples 128"; else MAX_EX_FLAG="--max-examples 1152"; fi
submit amdgpufast 1 0 18000 4:00:00 1   --no-enable-thinking
submit amdgpufast 1 0 18000 4:00:00 3  --enable-thinking

submit h200       1 1 18000 10:00:00 1  --no-enable-thinking

submit h200       1 2 18000 10:00:00 2  --no-enable-thinking
submit h200       2 2 18000 14:00:00 4  --enable-thinking --reasoning-effort low

submit h200fast   1 3 18000 4:00:00 1   --reasoning-effort low
submit h200fast   1 3 18000 4:00:00 1   --reasoning-effort medium

submit h200fast   1 4 18000 4:00:00 1   --reasoning-effort low
submit h200       1 4 18000 10:00:00 1  --reasoning-effort medium

submit amdgpufast 1 5 18000 4:00:00 1   --no-enable-thinking


DATASET="uner"
# "${!ARRAY[@]}" gives the array's INDICES (0, 1, 2, ...), not its values --
# looping over indices lets us also report "chunk 2 of 5" etc. below.
for i in "${!UNER_CHUNK_LIST[@]}"; do
  # ${UNER_CHUNK_LIST[$i]} is one already-space-separated line, e.g.
  # "en_ewt de_pud hr_set" -> exactly what --subset (nargs="+") expects.
  SUBSET_FLAG="--subset ${UNER_CHUNK_LIST[$i]}"
  TAG="uner/chunk$((i + 1))of${#UNER_CHUNK_LIST[@]}"

  BS=1; if [ "$PILOT" -eq 1 ]; then MAX_EX_FLAG="--max-examples 25"; else MAX_EX_FLAG="--max-examples 1000"; fi
  # submit amdgpufast 1 0 18000 4:00:00 1   --no-enable-thinking
  # submit amdgpufast 1 5 18000 4:00:00 1   --no-enable-thinking
  # submit h200       1 3 18000 9:00:00 1   --reasoning-effort low
  # submit amdgpu     2 1 18000 9:00:00 1   --no-enable-thinking
  # submit amdgpu     2 2 18000 9:00:00 1   --no-enable-thinking

  # submit h200       2 4 18000 24:00:00 1  --reasoning-effort low
  # submit h200       1 3 18000 24:00:00 1  --reasoning-effort medium
  # submit h200       2 4 18000 24:00:00 1  --reasoning-effort medium
  # submit amdgpulong 1 0 18000 30:00:00 1  --enable-thinking
  # submit amdgpulong 2 2 18000 30:00:00 1  --enable-thinking --reasoning-effort low

  BS=64; if [ "$PILOT" -eq 1 ]; then MAX_EX_FLAG="--max-examples 128"; else MAX_EX_FLAG="--max-examples 1000"; fi
  # submit amdgpufast 1 5 18000 4:00:00 1   --no-enable-thinking
  # submit h200fast   2 4 18000 4:00:00 1   --reasoning-effort low
  # submit amdgpufast 1 0 18000 4:00:00 1   --no-enable-thinking
  # submit h200fast   1 3 18000 4:00:00 1   --reasoning-effort low
  # submit h200       1 3 18000 6:00:00 1   --reasoning-effort medium
  # submit h200       2 4 18000 9:00:00 1   --reasoning-effort medium
  # submit amdgpu     1 0 18000 9:00:00 1   --enable-thinking
  # submit amdgpu     2 1 18000 9:00:00 1   --no-enable-thinking
  # submit amdgpu     2 2 18000 9:00:00 1   --no-enable-thinking

  # submit amdgpu     2 2 18000 24:00:00 1  --enable-thinking --reasoning-effort low
done

echo
echo "PILOT=$PILOT  DRY_RUN=$DRY_RUN  UNER_CHUNKS=$UNER_CHUNKS  seeds='$SEEDS'"
echo "total jobs: $n_jobs"
if [ "$n_jobs" -gt 200 ]; then
  echo "WARNING: $n_jobs jobs exceeds the 200-job QOS cap -- lower UNER_CHUNKS."
fi
