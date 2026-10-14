#!/usr/bin/env bash
# ===================================================================== Build the anonymous artifact repo for
#  the ARR submission (plan item B4).
#
#  Produces a self-contained, name-free copy of the code and the aggregate results, small enough for
#  anonymous.4open.science: that service does not serve files over 8 MB, and this repo is 4.2 GB, 3.4 GB of
#  which is per-example prediction JSONL that no reviewer will open.
#
#      ./scripts/publication/build_artifact_repo.sh [OUTDIR]
#
#  Default OUTDIR is ~/cg-artifact, deliberately OUTSIDE this repo.
#
#  Nothing here modifies the working tree: everything is copied first and scrubbed in the copy, so a running
#  job never sees a changed path.
#
#  After it runs: push OUTDIR to a NEW public GitHub repo, then paste that repo's URL into
#  https://anonymous.4open.science/ and cite the anonymous URL in the paper. Never push it into
#  stulcrad/master_thesis and never link the two accounts.
# =====================================================================
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$HOME/cg-artifact}"

echo "source : $SRC"
echo "target : $OUT"
[ -e "$OUT" ] && { echo "!! $OUT exists -- remove it first"; exit 1; }
mkdir -p "$OUT"

# ---------------------------------------------------------------------
# 1. What goes in.
#
# Datasets are deliberately NOT redistributed. CoNLL-2003 carries Reuters terms, several UNER treebanks and
# MultiGEC languages are per-language restricted, and the concurrent work's repo under data/semin has no
# LICENSE at all. The loaders stay; the data does not.
# ---------------------------------------------------------------------
copy () {  # copy <relative path> [rsync args...]
  local rel="$1"; shift
  [ -e "$SRC/$rel" ] || { echo "   skip (absent): $rel"; return 0; }
  mkdir -p "$OUT/$(dirname "$rel")"
  rsync -a --quiet "$@" "$SRC/$rel" "$OUT/$(dirname "$rel")/"
}

echo "== copying code =="
COMMON_EX=(--exclude '__pycache__' --exclude '*.pyc' --exclude '.ipynb_checkpoints' --exclude '.git')
copy src                  "${COMMON_EX[@]}"
# scripts/publication holds this builder and its scrub patterns -- it must not travel into the artifact, or
# the patterns scrub themselves.
copy scripts              "${COMMON_EX[@]}" --exclude 'playground.py' --exclude 'publication/'
copy slurm                "${COMMON_EX[@]}"
copy analysis             "${COMMON_EX[@]}"
copy Notebooks/CG_publication "${COMMON_EX[@]}"
copy requirements.txt
copy pyproject.toml
copy init_environment_python.sh

# Aggregate results only: the per-run Csv summaries the tables are built from (~27 MB across 18 directories).
# Predictions/ and Txt/ stay behind.
echo "== copying aggregate results =="
mkdir -p "$OUT/Experiment_results_publication"
# Exclude first: rsync applies rules in order, and 'Csv/**' would otherwise pull the superseded archived runs
# back in before the exclude is reached.
rsync -a --quiet \
  --exclude 'Archived_results/' \
  --include '*/' \
  --include 'Csv/**' \
  --exclude '*' \
  "$SRC/Experiment_results_publication/" "$OUT/Experiment_results_publication/"
find "$OUT/Experiment_results_publication" -type d -empty -delete

# ---------------------------------------------------------------------
# 2. Scrub. Ordered: longest and most specific pattern first, so a later rule never eats the tail of an
#    earlier one.
# ---------------------------------------------------------------------
echo "== scrubbing =="
mapfile -d '' TEXTFILES < <(find "$OUT" -type f \
  \( -name '*.py' -o -name '*.ipynb' -o -name '*.sh' -o -name '*.batch' \
     -o -name '*.md' -o -name '*.txt' -o -name '*.toml' -o -name '*.cfg' \
     -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' \) -print0)

sed -i \
  -e 's|/home/stulcrad/master_thesis|/path/to/constrained-tagging|g' \
  -e 's|/home/stulcrad|/path/to|g' \
  -e 's|Radek Štulc|John Smith|g' \
  -e 's|Radek Stulc|John Smith|g' \
  -e 's|stulcrad|anon|g' \
  -e 's|Štulc|Smith|g' \
  -e 's|Radek|John|g' \
  -e 's|stulcrad@gmail\.com|anonymous@example.com|g' \
  "${TEXTFILES[@]}"

# ---------------------------------------------------------------------
# 3. An anonymous README replaces the one that names the author and the thesis. The thesis is not a
#    prior-publication problem (ARR exempts dissertations), but naming it here would identify the author.
# ---------------------------------------------------------------------
cat > "$OUT/README.md" <<'README_EOF'
# Constrained Tagging for Span Labeling

Anonymous code artifact accompanying the submission *Constrained Tagging for
Span Labeling*.

Span labeling with decoder-only LLMs, where the output must be a **verbatim**
substring of the input and the label set is **dynamic**. The constrained
decoding scheme makes non-verbatim output impossible by construction, needs no
task-specific training, and is evaluated unchanged across six span-labeling
tasks.

## Layout

```
src/utils/          core library: byte-level token trie and the logits processors
scripts/<Dataset>/  one evaluation entry point per dataset and per approach
scripts/analysis/   table generation and experiment auditing
slurm/              the job scripts the experiments were run with
analysis/           significance testing
Notebooks/          exploratory notebooks, including the two-phase derivation
Experiment_results_publication/
                    per-run aggregate metrics (CSV), one directory per
                    dataset and approach
```

## What is not included

* **Datasets.** Not redistributed: CoNLL-2003 carries Reuters terms, and
  several UniversalNER treebanks and MultiGEC languages are individually
  restricted. The loaders in `scripts/data/` fetch them from their original
  sources.
* **Per-example predictions.** The full prediction JSONL is ~3.4 GB and
  exceeds what this hosting supports. The aggregate CSVs that every reported
  table is computed from are included in full.

## Reproducing

```bash
pip install -r requirements.txt
python scripts/NER/evaluationNER_cons_gen.py --model <hf-id> --batch-size 16
```

Paths in the SLURM scripts are written as `/path/to/...` and need pointing at
a local checkout.
README_EOF

# ---------------------------------------------------------------------
# 4. Verify. This is the part that actually protects the submission.
# ---------------------------------------------------------------------
echo
echo "================ VERIFICATION ================"
echo "size        : $(du -sh "$OUT" | cut -f1)"
echo "files       : $(find "$OUT" -type f | wc -l)"
echo
echo "-- files over 8 MB (anonymous.4open.science will not serve these) --"
find "$OUT" -type f -size +8M -printf '%s\t%p\n' | sort -rn | head -20
echo "   count: $(find "$OUT" -type f -size +8M | wc -l)"
echo
echo "-- hard identity leaks (must be zero) --"
grep -rIn -E 'stulcrad|Radek|Štulc|Stulc' "$OUT" | head -20
echo "   count: $(grep -rIl -E 'stulcrad|Radek|Štulc|Stulc' "$OUT" 2>/dev/null | wc -l) files"
echo
echo "-- soft signals (judgement call, not auto-scrubbed) --"
echo "   cluster partition names, which identify the compute site:"
grep -rIho -E 'partition=[a-z0-9]+' "$OUT" 2>/dev/null | sort | uniq -c | sort -rn | head
echo "   institution strings in prose:"
grep -rIn -E '\bCTU\b|\bcvut\b|Czech Technical' "$OUT" 2>/dev/null | head -5
echo "=============================================="
echo
echo "next: cd $OUT && git init && git add -A && git commit -m 'Anonymous artifact'"
echo "      push to a NEW public repo, then anonymize it at anonymous.4open.science"
