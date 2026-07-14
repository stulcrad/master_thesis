"""Paired bootstrap significance tests over per-example prediction files (JSONL).

Compares two systems A and B that were evaluated on the SAME examples
(same dataset, same seeds -> identical `key` fields). Each line of a
predictions JSONL is one generation; see the eval scripts/notebooks that
produce them (one file per experiment config).

Metrics
-------
entity_f1  Micro entity-level F1 recomputed from BIO tags. Entities are
           extracted with conlleval-style segmentation (B- starts an entity;
           I-X continues one; I-X after O or after a different type starts a
           new entity). This may differ marginally from seqeval's strict IOB2
           mode on malformed transitions, but both systems are scored with
           the identical procedure, so the paired comparison is self-consistent.
char_f1    Mean of the per-example `char_f1` field (macro char-level F1, the
           ToxicSpans/LegalQAEval metric).

Procedure
---------
Paired bootstrap: resample the N shared keys with replacement `--n-boot`
times; on each resample compute metric(A) - metric(B) over the SAME resampled
keys; report the observed delta, the percentile 95% CI, and a two-sided
bootstrap p-value (fraction of resampled deltas on the other side of zero,
doubled and clipped to 1).

Caveat: with the 250-examples x 5-seeds sampling protocol, the same source
sentence can appear under several seeds, so generations are not fully
independent units. With the full-test-set + greedy single-run design (one
generation per test example) this caveat disappears.

Usage
-----
python analysis/significance.py \
    --a Predictions/conll_gemma-3-4b-it_greedy_constrained_token_aware_bs1.jsonl \
    --b Predictions/hf_context_gemma-3-4b-it_SYSTEM_PROMPT_CONTEXT_exact_bs1.jsonl \
    --metric entity_f1 [--n-boot 10000] [--boot-seed 0] \
    [--pred-field-a pred_tags] [--pred-field-b pred_tags]

For files that store several prediction variants per line (e.g. the xgrammar
baseline stores `pred_tags_exact` and `pred_tags_fuzzy`), select the variant
with --pred-field-a / --pred-field-b.
"""

import argparse
import json
import sys

import numpy as np


def load_jsonl(path: str) -> dict:
    """Load a predictions JSONL into {key: record}."""
    records = {}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = r["key"]
            if key in records:
                sys.exit(f"{path}:{line_no}: duplicate key {key!r} — one JSONL "
                         f"file must hold exactly one config (split by config first).")
            records[key] = r
    return records


def extract_entities(tags) -> set:
    """BIO tags -> set of (label, start, end) via conlleval-style segmentation."""
    ents = set()
    start, label = None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        prefix, _, lab = tag.partition("-")
        if start is not None and (prefix in ("O", "B") or lab != label):
            ents.add((label, start, i))
            start, label = None, None
        if prefix == "B" or (prefix == "I" and start is None and lab):
            start, label = i, lab
    return ents


def entity_counts(gold_tags, pred_tags):
    """Per-generation (tp, fp, fn) entity counts."""
    g = extract_entities(gold_tags)
    p = extract_entities(pred_tags)
    tp = len(g & p)
    return tp, len(p) - tp, len(g) - tp


def micro_f1(tp: int, fp: int, fn: int) -> float:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def paired_bootstrap(stat_fn, n: int, n_boot: int, rng) -> np.ndarray:
    """Resample indices [0, n) with replacement; return array of stat_fn(idx) deltas."""
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[b] = stat_fn(idx)
    return deltas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="predictions JSONL for system A")
    ap.add_argument("--b", required=True, help="predictions JSONL for system B")
    ap.add_argument("--metric", required=True, choices=["entity_f1", "char_f1"])
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--pred-field-a", default="pred_tags")
    ap.add_argument("--pred-field-b", default="pred_tags")
    ap.add_argument("--gold-field", default="gold_tags")
    args = ap.parse_args()

    rec_a = load_jsonl(args.a)
    rec_b = load_jsonl(args.b)

    keys = sorted(rec_a.keys() & rec_b.keys())
    only_a, only_b = len(rec_a) - len(keys), len(rec_b) - len(keys)
    if not keys:
        sys.exit("No shared keys between the two files — are these the same "
                 "dataset/seeds? Pairing requires identical sampling.")
    if only_a or only_b:
        print(f"WARNING: unmatched keys dropped (only in A: {only_a}, only in B: {only_b}). "
              f"Pairing uses the {len(keys)} shared keys.", file=sys.stderr)

    rng = np.random.default_rng(args.boot_seed)
    n = len(keys)

    if args.metric == "entity_f1":
        # Precompute per-generation counts once; the bootstrap then only sums.
        counts_a = np.array([entity_counts(rec_a[k][args.gold_field], rec_a[k][args.pred_field_a]) for k in keys])
        counts_b = np.array([entity_counts(rec_b[k][args.gold_field], rec_b[k][args.pred_field_b]) for k in keys])

        def observed(counts):
            tp, fp, fn = counts.sum(axis=0)
            return micro_f1(tp, fp, fn)

        def delta_on(idx):
            ta, pa, na = counts_a[idx].sum(axis=0)
            tb, pb, nb = counts_b[idx].sum(axis=0)
            return micro_f1(ta, pa, na) - micro_f1(tb, pb, nb)

        metric_a, metric_b = observed(counts_a), observed(counts_b)

    else:  # char_f1
        vals_a = np.array([rec_a[k]["char_f1"] for k in keys], dtype=float)
        vals_b = np.array([rec_b[k]["char_f1"] for k in keys], dtype=float)

        def delta_on(idx):
            return vals_a[idx].mean() - vals_b[idx].mean()

        metric_a, metric_b = vals_a.mean(), vals_b.mean()

    observed_delta = metric_a - metric_b
    deltas = paired_bootstrap(delta_on, n, args.n_boot, rng)
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    # Two-sided bootstrap p-value: how often the resampled delta crosses zero.
    p_two_sided = min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))

    print(f"pairs (generations):     {n}")
    print(f"metric:                  {args.metric}")
    print(f"A = {args.a}")
    print(f"B = {args.b}")
    print(f"metric A:                {metric_a:.4f}")
    print(f"metric B:                {metric_b:.4f}")
    print(f"observed delta (A - B):  {observed_delta:+.4f}")
    print(f"95% bootstrap CI:        [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"two-sided bootstrap p:   {p_two_sided:.4f}   (n_boot={args.n_boot}, seed={args.boot_seed})")
    if ci_low > 0 or ci_high < 0:
        print("=> significant at the 0.05 level (CI excludes zero)")
    else:
        print("=> NOT significant at the 0.05 level (CI includes zero)")


if __name__ == "__main__":
    main()
