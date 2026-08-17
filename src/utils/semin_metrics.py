"""Semin et al.'s character-overlap F1, loaded directly from their repo.

This module does NOT implement the metric. It imports their `metrics.py`
from the checkout that `scripts/data/prepare_semin_data.py` creates:

    data/semin/raw/span_labeling/span_labeling/metrics.py

so the numbers are theirs by construction and cannot drift from the paper.

Two functions, both theirs:

    compute_overlap_counts(pred_spans, gold_spans, hard_matching=True)
        -> {"overlap_chars": int, "predicted_chars": int, "gold_chars": int}
        Counts for ONE example.

    f1_from_counts(overlap_chars, predicted_chars, gold_chars)
        -> {"precision": float, "recall": float, "f1": float}

How to aggregate
----------------
Their metric is MICRO: sum the three counters over all examples, then call
`f1_from_counts` once. Averaging per-example F1 would be a different (macro)
number -- that is what our own `compute_character_f1` does, and why the two
disagree on identical predictions.

`hard_matching=True` requires labels to match; False scores span localisation
only. They report both as "hard" and "soft".

Spans are dicts with integer `start`/`end` (character offsets, end-exclusive)
and a `label` -- the same shape `parse_spans_from_tagged_output` produces.
"""

from pathlib import Path
import importlib.util

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = REPO_ROOT / "data/semin/raw/span_labeling/span_labeling/metrics.py"

_upstream = None


def _load():
    """Import their metrics.py once, by file path."""
    global _upstream
    if _upstream is None:
        if not UPSTREAM.exists():
            raise SystemExit(
                f"Semin et al.'s metrics.py not found at:\n  {UPSTREAM}\n"
                f"Run: python scripts/data/prepare_semin_data.py"
            )
        spec = importlib.util.spec_from_file_location("semin_upstream_metrics", UPSTREAM)
        _upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_upstream)
    return _upstream


def compute_overlap_counts(predicted_spans, gold_spans, hard_matching=True):
    """Their per-example counts. See module docstring."""
    return _load().compute_overlap_counts(predicted_spans, gold_spans, hard_matching)


def f1_from_counts(overlap_chars, predicted_chars, gold_chars):
    """Their P/R/F1 from pooled counts. See module docstring."""
    return _load().f1_from_counts(overlap_chars, predicted_chars, gold_chars)
