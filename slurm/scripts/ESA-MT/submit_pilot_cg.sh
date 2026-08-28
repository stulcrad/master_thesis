#!/bin/bash
#SBATCH --output=/home/stulcrad/master_thesis/logs/pilot_scripts/out/%x_%N_%A.out
#SBATCH --error=/home/stulcrad/master_thesis/logs/pilot_scripts/err/%x_%N_%A.err
# Submitter for the WMT24 ESA constrained-generation line. Same model set and
# per-model arms as slurm/scripts/NER/submit_pilot_cg.sh -- see that file for
# the reasoning behind each one. Differences from the NER pilot:
#
#   - evaluationWMT_CG.py has no --batch_size flag (batch size is fixed at 1:
#     the system prompt embeds one source text per translation, so batching
#     would ask about several source texts at once), so there is no BS loop.
#   - WMT24 ESA is 24 source-target-domain files (867 examples total). Each is
#     still evaluated and reported SEPARATELY -- Semin et al.'s ESA-MT headline
#     is the macro average over the 24 files -- but several share one job, see
#     below.
#
# WHY SUBSETS SHARE A JOB -- the 200-job cap
# ------------------------------------------
# Our QOS (`nonrci`) allows 200 jobs SUBMITTED and 200 running. SLURM counts
# array TASKS against that limit too, so `--array=0-23%4` would be rejected at
# submit time just the same: throttling concurrency does not help when the cap
# is on submission. One job per (subset, arm) needs 24 x 10 = 240 -> over.
#
# So `--wmt-subset` takes SEVERAL subsets and the script loops over them in one
# job, keeping the model resident. Worth doing on its own merits: 24 separate
# jobs per arm means loading gpt-oss-120b 24 times, which for the heavy arms
# costs more than the inference does.
#
# CHUNKS controls the trade between job count and wall clock:
#   CHUNKS=1 (default) -> 24 subsets in 1 job per arm  =  10 jobs
#   CHUNKS=4           ->  6 subsets per job           =  40 jobs
#   CHUNKS=24          ->  1 subset per job            = 240 jobs -> OVER THE CAP
# Raise CHUNKS if one job would exceed the wall clock (the *fast partitions used
# here cap at 4h). Each subset writes its own JSONL and its own CSV row either
# way, so the macro average over the 24 files does not depend on the chunking.
#
# Chunks are balanced by EXAMPLE COUNT, not file count -- the 24 files range
# from 12 examples (wmt-en-ja-news) to 77 (wmt-en-is-news), so an equal-COUNT
# split (e.g. 6 files per chunk) can still hand one job 3 "is" files (~220
# examples) and another 3 "ja"/"zh" files (~40 examples), a >5x imbalance.
# Same greedy bin-packing as slurm/scripts/NER/submit_pilot_cg.sh's UNER
# chunking: biggest subset first, always into whichever chunk currently has
# the least total work.
#
# PILOT vs FULL
# -------------
# PILOT=1 (default) is a cheap timing run: capped examples, one seed.
# PILOT=0 is the real run: whole datasets. Seeds are ONE either way.
#
#     bash slurm/scripts/ESA-MT/submit_pilot_cg.sh                  # pilot
#     PILOT=0 bash slurm/scripts/ESA-MT/submit_pilot_cg.sh          # full run
#     DRY_RUN=1 bash slurm/scripts/ESA-MT/submit_pilot_cg.sh        # print, submit nothing
#     CHUNKS=3 PILOT=0 bash slurm/scripts/ESA-MT/submit_pilot_cg.sh # full, 30 jobs
#
# Check the indices with: python -m utils.model_registry --list
set -eu
cd ~/master_thesis

source init_environment_python.sh

PILOT="${PILOT:-1}"      # 1 = capped timing pilot, 0 = full datasets
DRY_RUN="${DRY_RUN:-0}"  # 1 = print the sbatch lines instead of submitting
CHUNKS="${CHUNKS:-1}"    # how many jobs to split the 24 subsets across

# ONE seed everywhere, pilot and full alike. Deliberate: nothing is trained, so
# there is no training variance to average over, and breadth across tasks is worth
# more than error bars on a quantity the constraint already makes invariant.
# See PUBLICATION_PLAN.md "NEXT STEPS / 0. Protocol".
SEEDS="42"

SUBSETS=(en-cs-literary en-cs-news en-cs-social en-es-literary en-es-news en-es-social \
         en-hi-literary en-hi-news en-hi-social en-is-literary en-is-news en-is-social \
         en-ja-literary en-ja-news en-ja-social en-ru-literary en-ru-news en-ru-social \
         en-uk-literary en-uk-news en-uk-social en-zh-literary en-zh-news en-zh-social)
N_SUBSETS=${#SUBSETS[@]}

# Ask Python (which can read the real per-file example counts off disk) to
# greedy-bin-pack the 24 subsets into CHUNKS groups balanced by total example
# COUNT rather than file count. One line of output per chunk, e.g.
# "en-is-news en-ja-news en-zh-news" -- exactly what --wmt-subset (nargs="+")
# expects once word-split.
CHUNKS_TEXT="$(python3 - "$CHUNKS" <<'PYEOF'
import sys, json
from pathlib import Path

n_chunks = int(sys.argv[1])
wmt_dir = Path("data/semin/wmt")
subs = sorted(p.stem.removeprefix("wmt-") for p in wmt_dir.glob("wmt-*.json"))
sizes = {s: len(json.loads((wmt_dir / f"wmt-{s}.json").read_text(encoding="utf-8")))
         for s in subs}

chunks = [[] for _ in range(n_chunks)]
loads = [0] * n_chunks
# Sort the subsets by size, largest first, and assign each to the chunk with
# the least total size so far (greedy bin-packing).
for s in sorted(subs, key=lambda s: -sizes[s]):
    i = loads.index(min(loads))
    chunks[i].append(s)
    loads[i] += sizes[s]

for c in chunks:
    if c:  # more chunks than subsets (CHUNKS > 24) -> fewer lines out
        print(" ".join(c))
PYEOF
)"
# mapfile reads that multi-line string into a bash array, one element per line
# (one element per chunk).
mapfile -t WMT_CHUNK_LIST <<< "$CHUNKS_TEXT"

n_jobs=0

# submit <partition> <gpus> <array_idx> <max_new_tokens> <time> [extra args...]
#
# CHUNK_SUBSETS (the subsets this job covers) comes from the enclosing loop.
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
  echo "submit: ${TAG} array=${idx}  $*"
  if [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  # $MAX_EX_FLAG is deliberately UNQUOTED: it is either empty (and must vanish
  # entirely) or two words that must word-split. "${CHUNK_SUBSETS[@]}" IS quoted:
  # each subset must stay one argv entry, and --wmt-subset takes nargs="+".
  sbatch --partition="$part" --gres=gpu:"$gpus" --array="$idx" --time="$wtime" \
    slurm/scripts/ESA-MT/evaluationWMT_CG.batch \
    --wmt-subset "${CHUNK_SUBSETS[@]}" $MAX_EX_FLAG \
    --max-new-tokens "$mnt" --seeds $SEEDS "$@"
}

# "${!ARRAY[@]}" gives the array's INDICES (0, 1, 2, ...), not its values --
# looping over indices lets us also report "chunk 2 of 5" etc. below.
for i in "${!WMT_CHUNK_LIST[@]}"; do
  # ${WMT_CHUNK_LIST[$i]} is one already-space-separated line, e.g.
  # "en-is-news en-ja-news en-zh-news" -> word-split into a real bash array so
  # submit()'s "${CHUNK_SUBSETS[@]}" expansion keeps each subset one argv entry.
  CHUNK_SUBSETS=(${WMT_CHUNK_LIST[$i]})
  TAG="wmt/chunk$((i + 1))of${#WMT_CHUNK_LIST[@]}[${#CHUNK_SUBSETS[@]} subsets: ${CHUNK_SUBSETS[0]}..${CHUNK_SUBSETS[${#CHUNK_SUBSETS[@]}-1]}]"

  if [ "$PILOT" -eq 1 ]; then
    # CAUTION: the cap is PER SUBSET, not per job -- 15 x 24 = 360 examples per
    # arm across the whole sweep, however they are chunked.
    MAX_EX_FLAG="--max-examples 15"
  else
    # --max-examples is an int defaulting to None = everything, so for the full
    # run the flag must be omitted ENTIRELY rather than set to a sentinel.
    MAX_EX_FLAG=""
  fi

  # -- gemma-4-E2B (idx 0): 2.3B effective, cheap enough for both arms ------
  submit amdgpufast 1 0 18000 4:00:00 --enable-thinking
  submit amdgpufast 1 0 18000 4:00:00 --no-enable-thinking

  # -- Rows with a published comparison number: reasoning OFF ---------------
  submit amdgpufast 2 1 18000 4:00:00 --no-enable-thinking   # gemma-4-31B, 2x A100-40
  submit amdgpufast 1 5 18000 4:00:00 --no-enable-thinking   # Qwen3-8B, 1 GPU

  # -- Qwen3.8-27B (idx 2): reasoning ON (low) vs OFF -----------------------
  submit amdgpufast 2 2 18000 4:00:00 --enable-thinking --reasoning-effort low
  submit amdgpufast 2 2 18000 4:00:00 --no-enable-thinking

  # -- gpt-oss (idx 3, 4): effort sweep -------------------------------------
  for EFF in low medium; do
    submit h200fast 1 3 18000 4:00:00 --reasoning-effort "$EFF"
    submit h200fast 2 4 18000 4:00:00 --reasoning-effort "$EFF"
  done

  n_jobs=$((n_jobs + 10))
done

echo
echo "PILOT=$PILOT  DRY_RUN=$DRY_RUN  CHUNKS=$CHUNKS  seeds='$SEEDS'"
echo "total jobs: $n_jobs  (10 arms x ${#WMT_CHUNK_LIST[@]} chunk(s) covering $N_SUBSETS subsets)"
if [ "$n_jobs" -gt 200 ]; then
  echo "WARNING: $n_jobs jobs exceeds the 200-job QOS cap -- lower CHUNKS."
fi
