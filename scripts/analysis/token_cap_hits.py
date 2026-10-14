"""
How a constrained generation can fail to reproduce its input, per task, and what the metrics did with it.

Why this needs counting
-----------------------
Under the constraint the model cannot stop early in the ANSWER: end-of-sequence is only admissible once the whole
input has been copied. Reasoning is unconstrained, though, so there are exactly three legitimate ways a constrained
run ends up with `wrong_text` set:

    cap in reasoning    --max-new-tokens ran out before the reasoning-end marker appeared
    stop in reasoning   the model emitted end-of-sequence WHILE reasoning, without ever opening its answer
    cap in answer       reasoning finished (or there was none) and the copy itself ran out of budget

In the first two the answer is empty (see `_split_reasoning_answer`); in all three reconstruction fails and the example
is given NO predicted spans. Anything else is reported as `other`, and `other` should be ZERO: a constrained answer
that departs from the input for no such reason means the constraint was not applied. That column is how the missing
`reasoning_start_marker` in evaluationMultiGEC_CG.py was found (gemma-4-E2B-it skipping its thought block and then
answering unconstrained).

The part that matters for the results tables
--------------------------------------------
`compute_character_f1` scores an empty prediction against EMPTY gold as a perfect 1.0. A failed generation on an
empty-gold example is therefore counted as fully correct. Toxic Spans and LegalQAEval have ~50% empty gold, so on those
two a failure is rewarded about half the time. This script reports exactly how many such examples exist (`scored 1.0`),
so the choice (score them 0 / report F1 on non-empty gold only / leave as is and say so) is made on numbers. Semin et
al.'s pooled metric skips empty-gold examples and is unaffected; seqeval on the NER datasets has no per-example score,
so the column does not apply there.

What is counted
---------------
One JSONL row is one GENERATION. At batch size 1 that is one example; at batch size 64 (CoNLL / UNER only) it is a
whole batch, so a single failure there costs up to 64 examples. The default report is batch size 1, where generation
and example coincide; pass --batch-size 64 for the batched arms.

The cap itself is read from the `max_new_tokens` column of the result CSVs rather than assumed, and the script refuses
to continue if a dataset was run with more than one value.

Usage
-----
    python scripts/analysis/token_cap_hits.py                       # constrained arms, batch size 1
    python scripts/analysis/token_cap_hits.py --mode unconstrained  # the unconstrained arms of the same runs
    python scripts/analysis/token_cap_hits.py --by-model            # one line per (dataset, model, arm)
    python scripts/analysis/token_cap_hits.py --latex               # rows for paper/tables/tab_failures.tex

Reads ~4 GB of prediction files; expect a few minutes.
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "Experiment_results_publication"

# Results directory -> (name used in the paper, whether per-example char-F1 exists). Order is the row order of Table 1.
DATASETS = {
    "CoNLL":       ("CoNLL-2003",   False),
    "ToxicSpans":  ("Toxic Spans",  True),
    "LegalQAEval": ("LegalQAEval",  True),
    "UNER":        ("UniversalNER", False),
    "ESA-MT":      ("WMT24 ESA-MT", True),
    "GEC":         ("MultiGEC",     True),
}

# The arm (reasoning effort, or plain ON/OFF) is only recoverable from the filename: the JSONL rows carry
# `reasoning_enabled` but not the effort level.
EFFORT = re.compile(r"_effort_([a-z]+)_")


def token_cap(dataset_dir):
    """
    The --max-new-tokens this dataset's constrained runs used, from the result CSVs.

    Read rather than hardcoded because the cap is set in the submit scripts, and a hit is defined against it. More
    than one value in a dataset would make "hit the cap" ambiguous, so that is an error, not a guess.
    """
    caps = set()
    for csv_path in (RESULTS / dataset_dir / "Constrained-Gen" / "Csv").glob("*.csv"):
        with csv_path.open(encoding="utf-8") as fh:
            for record in csv.DictReader(fh):
                if record.get("max_new_tokens"):
                    caps.add(int(float(record["max_new_tokens"])))
                break  # constant per file
    if len(caps) != 1:
        raise SystemExit(f"{dataset_dir}: expected exactly one max_new_tokens value, found {sorted(caps) or 'none'}")
    return caps.pop()


def arm_of(path, row):
    """ON / OFF, or the reasoning effort when the model takes one."""
    match = EFFORT.search(path.name)
    if match:
        return match.group(1)
    return "ON" if row.get("reasoning_enabled") else "OFF"


def scan(dataset_dir, mode, batch_size):
    """
    -> {(model, arm): counts} for one dataset.

    A row hit the cap when it generated at least `cap` tokens. It is "unterminated" when the model was reasoning, did
    not skip the block, and never emitted the end marker -- by running into the cap or by stopping on its own.
    """
    cap = token_cap(dataset_dir)
    has_char_f1 = DATASETS[dataset_dir][1]
    counts = defaultdict(lambda: defaultdict(int))
    files = sorted((RESULTS / dataset_dir / "Constrained-Gen" / "Predictions").glob("*.jsonl"))
    for path in files:
        # The mode is in the filename; skipping on it avoids parsing half of the 4 GB.
        if f"_{mode}_" not in path.name:
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a truncated last line from a job killed mid-write
                if int(row.get("batch_size", 1)) != batch_size:
                    continue
                c = counts[(row["model"].split("/")[-1], arm_of(path, row))]
                c["generations"] += 1
                wrong = bool(row.get("wrong_text"))
                hit_cap = int(row.get("num_output_tokens", 0)) >= cap
                unterminated = bool(row.get("reasoning_enabled") and not row.get("reasoning_skipped")
                                    and not row.get("found_reasoning_end"))
                if unterminated:
                    c["cap_reasoning" if hit_cap else "stop_reasoning"] += 1
                elif hit_cap:
                    c["cap_answer"] += 1
                elif wrong and mode == "constrained":
                    c["other"] += 1  # should never happen: the constraint was not applied
                c["wrong_text"] += int(wrong)
                # The metric pitfall: a failed generation on empty gold is scored as perfect.
                if wrong and has_char_f1 and row.get("char_f1") == 1.0:
                    c["rewarded"] += 1
    return cap, counts


def total(counts):
    """Sum the per-(model, arm) counters of one dataset."""
    out = defaultdict(int)
    for c in counts.values():
        for key, value in c.items():
            out[key] += value
    return out


def pct(part, whole):
    return f"{100 * part / whole:.2f}" if whole else "--"


def print_report(results, by_model):
    header = (f"  {'':30}{'gens':>9}{'wrong text':>12}{'cap/reas.':>11}{'stop/reas.':>12}{'cap/answer':>12}"
              f"{'other':>8}{'scored 1.0':>12}")
    for dataset_dir, (cap, counts) in results.items():
        name, has_char_f1 = DATASETS[dataset_dir]
        print(f"\n=== {name}  (cap {cap:,} tokens) ===")
        print(header)
        rows = sorted(counts.items()) if by_model else []
        rows.append((("ALL", ""), total(counts)))
        for (model, arm), c in rows:
            rewarded = str(c["rewarded"]) if has_char_f1 else "n/a"
            flag = "   <-- CONSTRAINT NOT APPLIED" if c["other"] else ""
            print(f"  {(model + ' ' + arm).strip():30}{c['generations']:>9,}{c['wrong_text']:>12,}"
                  f"{c['cap_reasoning']:>11,}{c['stop_reasoning']:>12,}{c['cap_answer']:>12,}"
                  f"{c['other']:>8,}{rewarded:>12}{flag}")


def print_latex(results):
    """
    Data rows for tab_failures.tex, ready to paste between \\midrule and \\bottomrule.

    Column order: dataset, generations, wrong text, cap in reasoning, stop in reasoning, cap in answer, scored 1.0.
    `other` is deliberately not a column: it must be zero before the table is published, and the script says so.
    """
    leftovers = {DATASETS[d][0]: total(c)["other"] for d, (_, c) in results.items() if total(c)["other"]}
    if leftovers:
        print(f"% !! {leftovers} generation(s) are wrong-text for NO legitimate reason -- fix and re-run before using this table",
              file=sys.stderr)
    print("% rows generated by scripts/analysis/token_cap_hits.py --latex")
    cells = []
    for dataset_dir, (_, counts) in results.items():
        name, has_char_f1 = DATASETS[dataset_dir]
        c = total(counts)
        cells.append([
            name, f"{c['generations']:,}".replace(",", "{,}"), str(c["wrong_text"]), str(c["cap_reasoning"]),
            str(c["stop_reasoning"]), str(c["cap_answer"]), str(c["rewarded"]) if has_char_f1 else "---",
        ])
    widths = [max(len(cell[i]) for cell in cells) for i in range(len(cells[0]))]
    for cell in cells:
        print(" & ".join(f"{cell[i]:{'<' if i == 0 else '>'}{widths[i]}}" for i in range(len(cell))) + r" \\")


def main():
    parser = argparse.ArgumentParser("Count and classify generations that failed to reproduce their input.")
    parser.add_argument("--mode", choices=["constrained", "unconstrained"], default="constrained",
                        help="Which decoding mode of the CG runs to count.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size to count. 1 everywhere; 64 exists for CoNLL and UNER only.")
    parser.add_argument("--by-model", action="store_true", help="Break each dataset down by model and arm.")
    parser.add_argument("--latex", action="store_true", help="Emit the table rows instead of the report.")
    args = parser.parse_args()

    results = {}
    for dataset_dir in DATASETS:
        print(f"scanning {dataset_dir} ...", file=sys.stderr)
        results[dataset_dir] = scan(dataset_dir, args.mode, args.batch_size)

    print_latex(results) if args.latex else print_report(results, args.by_model)


if __name__ == "__main__":
    main()
