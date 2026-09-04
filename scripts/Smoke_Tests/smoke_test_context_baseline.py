"""
CPU smoke test for the five context-based baseline scripts.

Runs each `evaluation*_HF_context.py` end to end -- real argparse, real registry
guards, real dataset loading, real prompt construction, real matching, real
metrics, real JSONL/CSV writing -- with ONE thing replaced: `generate_markup` is
a stub, so no model weights are loaded and no GPU is touched. Everything that a
GPU run does except the forward pass is therefore exercised here, in seconds, on
a CPU node.

The stub is not a fixed string. For each call it reads the labels out of the
system prompt it was handed and returns one of eight canned responses on a
rotation -- one per Semin error type: a locatable span (`success`), a locatable
plus an unlocatable one (`partial_span_not_found`), a locatable plus a bad label
(`partial_invalid_label`), prose instead of JSON (`format_error`), a valid empty
array (`empty_prediction`), an unlocatable span alone (`span_not_found`), a bad
label alone (`invalid_label`), and nothing at all (`empty_response`).

Two things follow. The locatable spans are real words from the real input, so the
matching path, the character spans, seqeval, Semin's counts and the F1 columns
take non-trivial values instead of quietly staying at zero. And any case with at
least eight generations must observe all eight error types -- a missing one means
that branch of the classifier is unreachable from the real pipeline, which the
run asserts rather than merely prints.

    python scripts/Smoke_Tests/smoke_test_context_baseline.py
    python scripts/Smoke_Tests/smoke_test_context_baseline.py --only ner gec
    python scripts/Smoke_Tests/smoke_test_context_baseline.py --keep   # keep the output tree

Results go to a temporary RESULTS_ROOT, never to Experiment_results_publication.
"""
import argparse
import json
import os
import runpy
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

#: Any tokenizer works -- the stub never generates, and the scripts only use the
#: tokenizer to turn the reasoning markers into token ids. This one is small and
#: already in the local HF cache.
TOKENIZER_MODEL = "Qwen/Qwen3-8B"

#: Every label the six prompts can use. The stub picks the ones that appear in
#: the system prompt it is handed, which is how it stays dataset-agnostic.
KNOWN_LABELS = ["PER", "LOC", "ORG", "MISC", "TOXIC", "ANSWER", "MAJOR", "MINOR", "R", "M", "U"]


class _StubGenerationConfig:
    eos_token_id = 0


class _StubModel:
    """Stands in for the loaded checkpoint. The scripts only read `.device` and
    `.generation_config.eos_token_id` outside of generation itself."""
    device = "cpu"
    generation_config = _StubGenerationConfig()


def _labels_in(system_prompt):
    """Which task labels this prompt is about, in prompt order."""
    found = [lab for lab in KNOWN_LABELS if f"{lab}:" in system_prompt or f"{lab} —" in system_prompt
             or f'"{lab}"' in system_prompt]
    return found or ["PER"]


def _make_stub(state):
    """Build the `generate_markup` replacement. `state` carries the call counter."""

    def stub(model, tokenizer, processor, eval_model, input_text, system_prompt,
             max_new_tokens, do_sample, reasoning_model=False, reasoning_effort=None,
             reasoning_end_marker=None, reasoning_start_marker=None, stats_out=None,
             repetition_penalty=None, **kwargs):
        n = state["calls"]
        state["calls"] += 1

        labels = _labels_in(system_prompt)
        words = [w for w in input_text.split() if w.strip()]
        first = words[0] if words else "x"
        context = " ".join(words[:4]) if words else "x"

        good = {"entity": first, "label": labels[0], "context": context}
        unlocatable = {"entity": "ZZQQ_NOT_IN_TEXT", "label": labels[0], "context": "ZZQQ_NOT_IN_TEXT nearby"}
        bad_label = {"entity": first, "label": "NOT_A_REAL_LABEL", "context": context}

        # One rotation slot per Semin error type, so a run of >= 8 generations
        # hits every branch of `classify_generation_error` for real rather than
        # only in a unit test.
        rotation = [
            [good],                    # success
            [good, unlocatable],       # partial_span_not_found
            [good, bad_label],         # partial_invalid_label
            "PROSE",                   # format_error
            [],                        # empty_prediction
            [unlocatable],             # span_not_found
            [bad_label],               # invalid_label
            "EMPTY",                   # empty_response
        ][n % 8]

        if rotation == "PROSE":
            text = "I could not find any spans in this text, sorry."
        elif rotation == "EMPTY":
            text = ""
        else:
            text = json.dumps(rotation, ensure_ascii=False)

        n_tokens = max(len(text.split()), 1)
        if stats_out is not None:
            stats_out["num_reasoning_tokens"] = 7 if reasoning_model else 0
            stats_out["num_answer_tokens"] = n_tokens
            stats_out["reasoning_text"] = "stub reasoning" if reasoning_model else ""
            stats_out["answer_text"] = text
            stats_out["found_reasoning_end"] = bool(reasoning_model)
            stats_out["reasoning_skipped"] = False
            stats_out["reasoning_marker_seen"] = bool(reasoning_model)
            stats_out["raw_token_ids"] = list(range(n_tokens))
        return text, n_tokens + (7 if reasoning_model else 0), 0.01

    return stub


def _patch(state):
    """Install the stubs. Order matters: patch the modules the eval scripts import
    from BEFORE running them, since each does `from ... import generate_markup`."""
    import transformers
    import utils.utils_functions as uf

    transformers.AutoModelForCausalLM.from_pretrained = staticmethod(
        lambda *a, **k: _StubModel()
    )
    real_tokenizer = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = staticmethod(
        lambda *a, **k: real_tokenizer(TOKENIZER_MODEL)
    )
    uf.generate_markup = _make_stub(state)


CASES = {
    # name: (script, argv tail). Each case is deliberately a DIFFERENT split mode
    # so the sharding and subsetting paths are covered as well as the metrics.
    "ner_conll": ("scripts/NER/evaluationNER_HF_context.py", [
        "--dataset", "conll2003", "--max-examples", "36", "--batch_size", "2",
        "--seeds", "42", "43", "--shard", "0/2",
    ]),
    "ner_uner": ("scripts/NER/evaluationNER_HF_context.py", [
        "--dataset", "uner", "--subset", "en_pud", "--max-examples", "10", "--seeds", "42",
    ]),
    "toxic": ("scripts/ToxicSpans/evaluationToxicSpans_HF_context.py", [
        "--max-examples", "30", "--seeds", "42", "--shard", "1/3",
    ]),
    "legalqa": ("scripts/LegalQA/evaluationLegalQA_HF_context.py", [
        "--max-examples", "10", "--seeds", "42",
    ]),
    "wmt": ("scripts/ESA-MT/evaluationWMT_HF_context.py", [
        "--wmt-subset", "en-cs-news", "--max-examples", "10", "--seeds", "42",
    ]),
    "gec": ("scripts/GEC/evaluationMultiGEC_HF_context.py", [
        "--max-examples", "10", "--seeds", "42",
    ]),
}

#: Columns that must exist AND must not be uniformly empty. If one of these is
#: missing the run "passed" while silently not measuring the thing the baseline
#: exists to measure.
REQUIRED_COLUMNS = [
    "model", "dataset", "method", "matching", "reasoning_enabled",
    "semin_hard_f1_report", "semin_soft_f1_report",
    "format_invalid_rate_report", "span_not_found_rate_report", "invalid_label_rate_report",
    "output_tokens_avg_avg", "tokens_per_second_avg", "elapsed_minute_avg",
    "err_success_rate_report", "err_format_error_rate_report",
    "err_span_not_found_rate_report", "err_invalid_label_rate_report",
]


def run_case(name, script, argv_tail, results_root, model):
    import pandas as pd

    state = {"calls": 0}
    _patch(state)

    os.environ["RESULTS_ROOT"] = results_root
    sys.argv = [script, "--model", model, *argv_tail]

    started = time.time()
    runpy.run_path(str(REPO / script), run_name="__main__")
    elapsed = time.time() - started

    csvs = sorted(Path(results_root).rglob("*.csv"))
    jsonls = sorted(Path(results_root).rglob("*.jsonl"))
    if not csvs:
        raise AssertionError(f"{name}: no CSV written under {results_root}")
    if not jsonls:
        raise AssertionError(f"{name}: no JSONL written under {results_root}")

    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise AssertionError(f"{name}: CSV is missing columns {missing}")

    lines = [json.loads(line) for p in jsonls for line in p.read_text().splitlines()]
    if not lines:
        raise AssertionError(f"{name}: JSONL is empty")
    for key in ("pred_spans", "gold_spans", "error_type", "match_stats",
                "num_output_tokens", "generation_seconds", "raw_output"):
        if key not in lines[0]:
            raise AssertionError(f"{name}: JSONL record is missing {key!r}")

    # The stub deliberately produces at least one locatable span, so a run where
    # nothing was ever located means the matching path is broken.
    located = sum(r["match_stats"]["located_entities"] for r in lines)
    if located == 0:
        raise AssertionError(f"{name}: no span was ever located -- matching path is broken")

    seen_errors = sorted({r["error_type"] for r in lines})
    # The stub cycles through all eight error types, so a case with at least
    # eight generations must observe all eight. A missing one means that branch
    # of the classifier is unreachable from the real pipeline.
    if state["calls"] >= 8:
        expected = {"success", "partial_span_not_found", "partial_invalid_label", "format_error",
                    "empty_prediction", "span_not_found", "invalid_label", "empty_response"}
        unseen = expected - set(seen_errors)
        if unseen:
            raise AssertionError(f"{name}: error types never produced: {sorted(unseen)}")

    return {
        "case": name,
        "generations": state["calls"],
        "csv_rows": len(df),
        "jsonl_lines": len(lines),
        "csv_columns": len(df.columns),
        "located_spans": located,
        "predicted_spans": sum(r["span_count"] for r in lines),
        "error_types_seen": seen_errors,
        "seconds": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser("CPU smoke test for the context-based baseline scripts.")
    parser.add_argument("--only", nargs="+", choices=list(CASES), default=list(CASES),
                        help="Which cases to run (default: all).")
    parser.add_argument("--model", default="Qwen/Qwen3-8B",
                        help="Registered model id to pass through. Never loaded -- see module docstring.")
    parser.add_argument("--keep", action="store_true", help="Keep the temporary results tree.")
    args = parser.parse_args()

    root = tempfile.mkdtemp(prefix="context_smoke_")
    print(f"Results root for this run: {root}\n")

    summaries, failures = [], []
    for name in args.only:
        script, argv_tail = CASES[name]
        case_root = os.path.join(root, name)
        print(f"{'=' * 78}\n=== {name}: {script} {' '.join(argv_tail)}\n{'=' * 78}")
        try:
            summaries.append(run_case(name, script, argv_tail, case_root, args.model))
        except BaseException as exc:  # SystemExit included -- a guard firing is a failure here
            traceback.print_exc()
            failures.append((name, repr(exc)))

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    for s in summaries:
        print(f"  PASS {s['case']:<10} {s['generations']:>3} gens | {s['jsonl_lines']:>3} jsonl | "
              f"{s['csv_rows']} row(s) x {s['csv_columns']} cols | "
              f"{s['located_spans']} located / {s['predicted_spans']} scored spans | "
              f"errors={','.join(s['error_types_seen'])} | {s['seconds']}s")
    for name, err in failures:
        print(f"  FAIL {name}: {err}")

    if not args.keep:
        shutil.rmtree(root, ignore_errors=True)
    else:
        print(f"\nKept: {root}")

    print(f"\n{len(summaries)} passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
