"""Dataset loaders for the span-labeling evaluation scripts.

Every loader returns a plain `list[dict]`.

Two families.

TOKEN TASKS -- CoNLL-2003, UniversalNER
    {"tokens": [...], "tags": [...], "dataset_name": str, "example_id": str}

    Gold arrives as (or converts losslessly to) BIO tags over whitespace
    tokens. The text shown to the model is `" ".join(tokens)`, which is what
    the original NER script already did, and also exactly how Semin et al.'s
    UniversalNERParser builds UNER text.

CHARACTER TASKS -- Toxic Spans, LegalQAEval
    {"text": str,
     "gold_spans": [{"start": int, "end": int, "label": str}],
     "example_id": str, "question": str (LegalQA only)}

    `text` is the ORIGINAL string, newlines and repeated spaces included --
    the model copies it verbatim, so `gold_spans` keep the dataset's own
    character offsets with no remapping.

    There is deliberately NO token view here. Gold can be sub-token (Toxic
    Spans annotates parts of words) and both metrics are character-based, so
    tokens would be decoration: the earlier version carried them only to build
    BIO tags, and rebuilding the text as `" ".join(tokens)` would have been lossy and wrong.
"""

import json
from pathlib import Path

from utils.utils_functions import build_token_char_spans, parse_position, chars_to_spans

REPO_ROOT = Path(__file__).resolve().parents[2]
SEMIN_DATA = REPO_ROOT / "data" / "semin"

#: UniversalNER is PER/ORG/LOC -- no MISC, unlike CoNLL-2003. Semin et al.'s
#: parser also maps UNER's fourth category OTH to O, so OTH entities are absent
#: from the gold. We inherit that by using their parser.
UNER_LABELS = ["PER", "ORG", "LOC"]
CONLL_LABELS = ["PER", "LOC", "ORG", "MISC"]
TOXIC_LABELS = ["TOXIC"]
LEGALQA_LABELS = ["ANSWER"]

CONLL_ID2LABEL = {
    0: "O",
    1: "B-PER", 2: "I-PER",
    3: "B-ORG", 4: "I-ORG",
    5: "B-LOC", 6: "I-LOC",
    7: "B-MISC", 8: "I-MISC",
}


def bio_tags_to_char_spans(tokens, tags, token_spans=None):
    """BIO tags -> character spans.

    `token_spans` are each token's (start, end) offsets. Defaults to the
    offsets of `" ".join(tokens)`, which is right for the token tasks.

    Segmentation is conlleval-style: `B-X` opens an entity, `I-X` continues
    one of the same type, and a stray `I-X` with nothing open starts one
    (tolerating malformed output rather than dropping it).
    """
    if token_spans is None:
        token_spans = build_token_char_spans(list(tokens))

    spans = []
    start_tok = None
    label = None
    # Add one more "O" at the end to add the ending span to the list
    for i, tag in enumerate(list(tags) + ["O"]):
        prefix, _, lab = tag.partition("-")
        if start_tok is not None and (prefix in ("O", "B") or lab != label):
            spans.append({
                "start": token_spans[start_tok][0],
                "end": token_spans[i - 1][1],
                "label": label,
            })
            start_tok = None
            label = None
        if i < len(tags) and (prefix == "B" or (prefix == "I" and start_tok is None and lab)):
            start_tok = i
            label = lab

    return spans


# ---------------------------------------------------------------------------
# Token tasks
# ---------------------------------------------------------------------------

def load_conll2003(split="test"):
    """CoNLL-2003 from the HF hub, unchanged from the original NER script."""
    from datasets import load_dataset

    raw = load_dataset("lhoestq/conll2003", split=split)
    examples = []
    for idx, row in enumerate(raw):
        examples.append({
            "tokens": list(row["tokens"]),
            "tags": [CONLL_ID2LABEL[t] for t in row["ner_tags"]],
            "dataset_name": "conll2003",
            "example_id": str(row.get("id", idx)),
        })
    return examples


def uner_subsets():
    """The 18 UniversalNER test sets on disk, sorted."""
    directory = SEMIN_DATA / "universal_ner"
    if not directory.is_dir():
        raise SystemExit(
            f"UniversalNER not found at {directory}.\n"
            f"Run: python scripts/data/prepare_semin_data.py"
        )
    return sorted(p.stem.removeprefix("uner_") for p in directory.glob("uner_*.json"))


def load_uner(subset="all"):
    """UniversalNER from the reconstructed JSON.

    Their JSON stores character spans over `" ".join(tokens)`, and every gold
    span sits on a token boundary (verified on all 7,523 examples), so the
    conversion to BIO here is lossless.
    """
    from utils.utils_functions import spans_to_bio_tags

    available = uner_subsets()
    names = available if subset == "all" else [subset]
    for name in names:
        if name not in available:
            raise SystemExit(
                f"Unknown UNER subset {name!r}. Available:\n  " + "\n  ".join(available)
            )

    valid = set(UNER_LABELS)
    examples = []
    for name in names:
        path = SEMIN_DATA / "universal_ner" / f"uner_{name}.json"
        for idx, row in enumerate(json.loads(path.read_text(encoding="utf-8"))):
            tokens = row["text"].split(" ")
            tags, _ = spans_to_bio_tags(tokens=tokens, entities=row["spans"], valid_labels=valid)
            examples.append({
                "tokens": tokens,
                "tags": tags,
                "dataset_name": name,
                "example_id": f"{name}:{idx}",
            })
    return examples


def load_ner_dataset(name, uner_subset="all"):
    """Resolve --dataset to (examples, labels, results subdirectory)."""
    if name == "conll2003":
        return load_conll2003(), CONLL_LABELS, "CoNLL"
    if name == "uner":
        return load_uner(uner_subset), UNER_LABELS, "UNER"
    raise SystemExit(f"Unknown dataset {name!r}. Choose from: conll2003, uner")


# ---------------------------------------------------------------------------
# Character tasks -- the model sees the ORIGINAL text, verbatim
# ---------------------------------------------------------------------------

def _character_example(text, gold_spans, example_id, extra=None):
    """Shared shape for Toxic Spans / LegalQAEval."""
    example = {
        "text": text or "",
        "gold_spans": gold_spans,
        "example_id": example_id,
    }
    if extra:
        example.update(extra)
    return example


def load_toxic_spans(split="test"):
    """Toxic Spans. Gold is a list of character offsets into the original post."""
    from datasets import load_dataset

    raw = load_dataset("heegyu/toxic-spans", split=split)
    examples = []
    for idx, row in enumerate(raw):
        gold_spans = [
            {"start": start, "end": end, "label": "TOXIC"}
            for start, end in chars_to_spans(sorted(set(parse_position(row["position"]))))
        ]
        examples.append(_character_example(row["text_of_post"], gold_spans, str(idx)))
    return examples


def load_legalqa(split="test"):
    """LegalQAEval. Gold answers carry character offsets into the passage."""
    from datasets import load_dataset

    raw = load_dataset("isaacus/LegalQAEval", split=split)
    examples = []
    for idx, row in enumerate(raw):
        gold_spans = [
            {"start": a["start"], "end": a["end"], "label": "ANSWER"}
            for a in row["answers"]
        ]
        examples.append(_character_example(
            row["text"], gold_spans, str(idx), extra={"question": row["question"]}
        ))
    return examples
