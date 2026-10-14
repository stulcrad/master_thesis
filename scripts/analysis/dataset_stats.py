"""
Static facts for the datasets table (Table 1), computed from the loaders.

Every number in the table comes from here. These numbers are static facts about the datasets.

The evaluation caps
--------------------------------
CoNLL-2003 and UniversalNER are subsampled (see EVAL_CAPS). The cap is applied PER DATASET/TREEBANK and BEFORE
sharding, so effective N is `sum(min(n, cap))` over subsets, not `min(total, cap)`. The caps are duplicated
from the submit scripts, so `--check-caps` re-reads what the finished runs actually recorded and complains if
the two disagree.

Sampling is uniform and RE-DRAWN PER SEED (`random.Random(seed).sample(...)` inside the seed loop), so the
three seeds score three ~33%-overlapping subsets. Arms are still exactly paired within a seed. Sec. 3 of the
paper says so.

Usage
-----
    python scripts/analysis/dataset_stats.py                # full report, all six python
    scripts/analysis/dataset_stats.py uner wmt              # just these 
    python scripts/analysis/dataset_stats.py --check-caps   # caps vs. what the runs recorded

The HF-hosted loaders (CoNLL, Toxic Spans, LegalQAEval) hit the hub. Set HF_DATASETS_OFFLINE=1 to force the
local cache when the compute node has no network.
"""
import argparse
import csv
import statistics as stats
import sys
import unicodedata
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "Experiment_results_publication"
sys.path.insert(0, str(REPO / "src"))

from utils.span_datasets import (  # noqa: E402
    bio_tags_to_char_spans, load_conll2003, load_legalqa, load_multigec,
    load_toxic_spans, load_uner, load_wmt, uner_subsets, wmt_subsets,
)

# --max-examples as the full runs pass it, mirrored from slurm/scripts/NER/submit_pilot_cg.sh and
# submit_context.sh (the PILOT=0 branch). The other four datasets run uncapped. `--check-caps` verifies these
# against the max_examples column of the result CSVs.
EVAL_CAPS = {"conll2003": 1152, "uner": 250}

# Results subdirectory -> the loader key whose cap applies to it.
CAP_RESULT_DIRS = {"CoNLL": "conll2003", "UNER": "uner"}

# Unicode script families we care about, matched against the start of a character's Unicode name
SCRIPT_PREFIXES = ("LATIN", "CYRILLIC", "CJK", "HIRAGANA", "KATAKANA", "DEVANAGARI", "GREEK", "ARABIC", "HANGUL")

# A character has to appear this often (as a share of all script-bearing characters in a subset) before the
# subset is credited with that script. Without a floor, one Latin brand name in a Chinese sentence makes zh
# "bilingual".
SCRIPT_FLOOR_PCT = 1.0

# Editorial cells: the two columns that are a judgement call. Task is the short name the caption expands; Span
# geometry names the PRIMARY granularity only -- discontiguity and other nuance live in the caption and in
# Sec. 3, because the table is already at the full text width (see check_table_widths.py).
# Measured, not assumed: gold boundaries that fall INSIDE a word are 22% of ESA-MT spans (inflectional endings),
# 0.5% of LegalQAEval, 0.3% of Toxic Spans (annotation slips) and 0% of MultiGEC -- so only ESA-MT is "sub-word".
PRESENTATION = {
    "conll2003": ("CoNLL-2003",   "NER",           "token-aligned"),
    "toxic":     ("Toxic Spans",  "toxicity",      "word-aligned"),
    "legalqa":   ("LegalQAEval",  "extractive QA", "clause--paragraph"),
    "uner":      ("UniversalNER", "NER",           "token-aligned"),
    "wmt":       ("WMT24 ESA-MT", "MT errors",     "sub-word--clause"),
    "gec":       ("MultiGEC",     "GEC",           "incl.\\ \\emph{zero-length}"),
}

# Loader, subset lister, and the language codes of the TAGGED side. WMT is en->X, and it is the translation
# that gets copied and tagged, so the source language is not counted.
DATASETS = {
    "conll2003": (load_conll2003, None,          lambda subs: ["en"]),
    "toxic":     (load_toxic_spans, None,        lambda subs: ["en"]),
    "legalqa":   (load_legalqa, None,            lambda subs: ["en"]),
    "uner":      (load_uner, uner_subsets,       lambda subs: sorted({s.split("_")[0] for s in subs})),
    "wmt":       (load_wmt, wmt_subsets,         lambda subs: sorted({s.split("-")[1] for s in subs})),
    "gec":       (load_multigec, None,           lambda subs: ["en"]),
}


def median_chars(values):
    """
    Median character count, as an int, rounded half UP.
    """
    return int(Decimal(stats.median(values)).quantize(0, rounding=ROUND_HALF_UP))


def example_text_and_spans(example):
    """
    Character text and character spans for one example, whichever family it came from.

    Token tasks carry `tokens`/`tags`, character tasks carry `text`/`gold_spans`.
    """
    if "tokens" in example:
        return " ".join(example["tokens"]), bio_tags_to_char_spans(example["tokens"], example["tags"])
    return example["text"], example["gold_spans"]


def scripts_used(texts):
    """
    {script: percent of script-bearing characters}, for the scripts clearing SCRIPT_FLOOR_PCT.

    Punctuation, digits and whitespace carry no script and are skipped, so the percentages are over letters
    only.
    """
    counts = Counter()
    for text in texts:
        for char in text:
            try:
                name = unicodedata.name(char)
            except ValueError:  # unnamed control/private-use character, no script to attribute
                continue
            for prefix in SCRIPT_PREFIXES:
                if name.startswith(prefix):
                    counts[prefix] += 1
                    break
    total = sum(counts.values())
    if not total:
        return {}
    return {s: round(100 * n / total, 1) for s, n in counts.most_common() if 100 * n / total >= SCRIPT_FLOOR_PCT}


def measure(key):
    """
    Every number the table and the caption need, for one dataset.

    `n_eval` is the count actually scored per seed: the cap is applied per subset, so a treebank smaller than
    the cap contributes its own size. That is why this cannot be computed as min(total, cap).
    """
    loader, lister, languages_of = DATASETS[key]
    subsets = lister() if lister else None
    cap = EVAL_CAPS.get(key)

    examples = loader() if subsets is None else loader("all")
    texts, spans_per_example = [], []
    for example in examples:
        text, spans = example_text_and_spans(example)
        texts.append(text)
        spans_per_example.append(spans)

    # Effective N per seed. Without subsets the cap applies to the whole split; with them, once per subset.
    if cap is None:
        n_eval = len(examples)
    elif subsets is None:
        n_eval = min(len(examples), cap)
    else:
        n_eval = sum(min(len(loader(s)), cap) for s in subsets)

    span_lengths = [s["end"] - s["start"] for spans in spans_per_example for s in spans]
    text_lengths = [len(t) for t in texts]
    counts = [len(spans) for spans in spans_per_example]
    empty_gold = sum(1 for c in counts if c == 0)

    return {
        "key": key,
        "n_total": len(examples),
        "n_eval": n_eval,
        "capped": cap is not None,
        "subsets": subsets,
        "labels": sorted({s["label"] for spans in spans_per_example for s in spans}),
        "languages": languages_of(subsets),
        "scripts": scripts_used(texts),
        "n_spans": len(span_lengths),
        "n_zero_length": sum(1 for length in span_lengths if length == 0),
        "spans_per_example": stats.mean(counts),
        "pct_empty_gold": 100 * empty_gold / len(examples),
        "span_chars_median": median_chars(span_lengths),
        "span_chars_mean": stats.mean(span_lengths),
        "span_chars_max": max(span_lengths),
        "text_chars_median": median_chars(text_lengths),
        "text_chars_mean": stats.mean(text_lengths),
        "text_chars_max": max(text_lengths),
    }


def print_report(rows):
    """
    The human-readable form: everything measured, including the facts that feed the caption but not the grid.
    """
    for row in rows:
        name = PRESENTATION[row["key"]][0]
        print(f"\n=== {name} ===")
        if row["capped"]:
            cap = EVAL_CAPS[row["key"]]
            per = " per subset" if row["subsets"] else ""
            print(f"  N                     : {row['n_eval']:,} scored per seed / {row['n_total']:,} available "
                  f"(cap {cap:,}{per})")
        else:
            print(f"  N                     : {row['n_total']:,} (uncapped)")
        if row["subsets"]:
            print(f"  subsets               : {len(row['subsets'])}")
        print(f"  labels ({len(row['labels'])})            : {', '.join(row['labels'])}")
        print(f"  languages ({len(row['languages'])})".ljust(24) + f": {', '.join(row['languages'])}")
        print(f"  scripts               : {', '.join(f'{s} {p}%' for s, p in row['scripts'].items())}")
        print(f"  gold spans            : {row['n_spans']:,}  (zero-length: {row['n_zero_length']:,})")
        print(f"  spans/example         : {row['spans_per_example']:.2f}")
        print(f"  examples w/ empty gold: {row['pct_empty_gold']:.1f}%")
        print(f"  span chars            : median {row['span_chars_median']}  "
              f"mean {row['span_chars_mean']:.1f}  max {row['span_chars_max']}")
        print(f"  input chars           : median {row['text_chars_median']}  "
              f"mean {row['text_chars_mean']:.1f}  max {row['text_chars_max']}")

    # Caption facts: the union over every dataset, which no single row shows.
    all_languages = sorted({lang for row in rows for lang in row["languages"]})
    all_scripts = sorted({s for row in rows for s in row["scripts"]})
    print(f"\n--- for the caption ---")
    print(f"  languages ({len(all_languages)}): {', '.join(all_languages)}")
    print(f"  scripts   ({len(all_scripts)}): {', '.join(all_scripts)}")


def check_caps():
    """
    Cross-check EVAL_CAPS against the max_examples the finished runs recorded.
    """
    ok = True
    for results_dir, key in CAP_RESULT_DIRS.items():
        seen = set()
        for csv_path in sorted((RESULTS / results_dir).glob("*/Csv/*.csv")):
            with csv_path.open(encoding="utf-8") as fh:
                for record in csv.DictReader(fh):
                    if record.get("max_examples"):
                        seen.add(record["max_examples"])
                    break  # constant per file
        expected = {str(EVAL_CAPS[key])}
        status = "ok" if seen == expected else "MISMATCH"
        if seen != expected:
            ok = False
        print(f"  {results_dir:8} EVAL_CAPS={EVAL_CAPS[key]:<6} recorded={sorted(seen) or '(none)'}  {status}")
    if not ok:
        raise SystemExit("\nEVAL_CAPS disagrees with the runs on disk -- fix before trusting Table 1.")
    print("\n  caps agree with the runs on disk.")


def main():
    parser = argparse.ArgumentParser("Static dataset facts for Table 1.")
    parser.add_argument("datasets", nargs="*", choices=list(DATASETS),
                        help="Which datasets to measure. Default: all six.")
    parser.add_argument("--check-caps", action="store_true",
                        help="Verify EVAL_CAPS against the max_examples recorded in the result CSVs, then exit.")
    args = parser.parse_args()

    if args.check_caps:
        print("Evaluation caps vs. what the runs recorded:")
        check_caps()
        return

    # Keep the declaration order of DATASETS: it is the order of the rows in the paper.
    keys = [k for k in DATASETS if not args.datasets or k in args.datasets]
    rows = [measure(k) for k in keys]
    print_report(rows)


if __name__ == "__main__":
    main()
