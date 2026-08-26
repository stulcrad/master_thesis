"""Dataset loaders for the span-labeling evaluation scripts.

Every loader returns a plain `list[dict]`.

Two families.

TOKEN TASKS -- CoNLL-2003, UniversalNER
    {"tokens": [...], "tags": [...], "dataset_name": str, "example_id": str}

    Gold arrives as (or converts losslessly to) BIO tags over whitespace
    tokens. The text shown to the model is `" ".join(tokens)`, which is what
    the original NER script already did, and also exactly how Semin et al.'s
    UniversalNERParser builds UNER text.

CHARACTER TASKS -- Toxic Spans, LegalQAEval, WMT24 ESA
    {"text": str,
     "gold_spans": [{"start": int, "end": int, "label": str}],
     "example_id": str,
     "question": str (LegalQA only),
     "source": str, "source_language": str, "target_language": str,
     "dataset_name": str (WMT only)}

    `text` is the original string, newlines and repeated spaces included --
    the model copies it verbatim. The ONE transformation applied is Unicode
    NFC normalization, with `gold_spans` offsets remapped to match (see
    `_nfc_with_offset_map`); NFC preserves canonical equivalence, so the text
    renders identically and the spans still cover the same glyphs. Everything
    else about the string is left exactly as the dataset shipped it. For WMT,
    `text` is the TRANSLATION
    (the copy/tag target) and `source` is the untagged source-language text
    shown to the model as context, the same role `question` plays for LegalQA.

    There is deliberately NO token view here. Gold can be sub-token (Toxic
    Spans annotates parts of words) and both metrics are character-based, so
    tokens would be decoration: the earlier version carried them only to build
    BIO tags, and rebuilding the text as `" ".join(tokens)` would have been lossy and wrong.
"""

import json
import unicodedata
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
#: WMT24 ESA error severity. Unlike the other character tasks, this is TWO
#: labels, so hard (label-sensitive) and soft (location-only) Semin F1 can
#: actually diverge here -- see evaluationWMT_CG.py's module docstring.
WMT_LABELS = ["MAJOR", "MINOR"]

#: MultiGEC error actions, from Semin et al.'s M2 parser (`error_type.split(":")[0]`).
#: `M` is special: it is ALWAYS a zero-length insertion point (start == end), and
#: it is the only label that ever is -- see `load_multigec`.
MULTIGEC_LABELS = ["R", "M", "U"]

WMT_LANG_NAMES = {
    "en": "English", "cs": "Czech", "es": "Spanish", "hi": "Hindi",
    "is": "Icelandic", "ja": "Japanese", "ru": "Russian", "uk": "Ukrainian",
    "zh": "Chinese",
}

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
    """
    CoNLL-2003 from the HF hub.
    """
    from datasets import load_dataset

    raw = load_dataset("lhoestq/conll2003", split=split)
    examples = []
    for idx, row in enumerate(raw):
        examples.append({
            # No character offsets on this path (gold is already BIO over tokens),
            # so tokens normalize independently -- nothing to remap.
            "tokens": [unicodedata.normalize("NFC", t) for t in row["tokens"]],
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
            # Normalize BEFORE splitting/BIO conversion: their spans are character
            # offsets into row["text"], so text and offsets must move together.
            text, spans = _normalize_text_and_spans(row["text"], row["spans"])
            tokens = text.split(" ")
            tags, _ = spans_to_bio_tags(tokens=tokens, entities=spans, valid_labels=valid)
            examples.append({
                "tokens": tokens,
                "tags": tags,
                "dataset_name": name,
                "example_id": f"{name}:{idx}",
            })
    return examples


#: (labels, results-subdirectory) per --dataset, independent of any data load --
#: lets callers resolve --subset (task 'all' -> the real list) without paying
#: for a load just to find out what labels/directory apply.
NER_DATASET_INFO = {
    "conll2003": (CONLL_LABELS, "CoNLL"),
    "uner": (UNER_LABELS, "UNER"),
}


def ner_dataset_info(name):
    """(labels, results subdirectory) for --dataset, no data load required."""
    if name not in NER_DATASET_INFO:
        raise SystemExit(f"Unknown dataset {name!r}. Choose from: {', '.join(NER_DATASET_INFO)}")
    return NER_DATASET_INFO[name]


def ner_subsets(name):
    """What `--subset all` expands to for --dataset.

    UNER always expands to its 18 real treebanks (never pooled -- required for
    the correct metric). CoNLL expands to `["all"]`, i.e. the whole test set
    in ONE pass.
    """
    if name == "conll2003":
        return ["all"]
    if name == "uner":
        return uner_subsets()
    raise SystemExit(f"Unknown dataset {name!r}. Choose from: {', '.join(NER_DATASET_INFO)}")


def load_ner_dataset(name, subset="all"):
    """Resolve --dataset (+ --subset) to (examples, labels, results subdirectory)."""
    labels, results_dir = ner_dataset_info(name)
    if name == "conll2003":
        return load_conll2003(), labels, results_dir
    return load_uner(subset), labels, results_dir


# ---------------------------------------------------------------------------
# Unicode normalization -- see Bug 3 in Notebooks/CG_publication/reasoning_aware_CG.ipynb
# ---------------------------------------------------------------------------

def _nfc_with_offset_map(text):
    """NFC-normalize `text`, returning `(normalized, index_map)`.

    `index_map[i]` is where the character originally at index `i` starts in the
    normalized string, and `index_map[len(text)]` is the normalized length, so a
    span's exclusive `end` offset maps correctly too.

    Why this exists. Some tokenizers (Qwen's, not Gemma's or GPT-OSS's) apply NFC
    inside `encode()`. The constrained processor builds its copy target by
    re-encoding the input, so on text that is not already NFC the model is forced
    to copy a DIFFERENT string from the one `validate_reconstruction` grades it
    against -- a guaranteed `wrong_text=1` no matter how well the model behaves.
    The prompt is tokenized by the same tokenizer, so the model never sees the
    un-normalized form either way; normalizing here simply makes the gold agree
    with the input the model provably receives.

    NFC preserves canonical equivalence -- the normalized string is the SAME text
    by Unicode's definition, rendered identically -- so remapped spans still cover
    the same glyphs. This is deliberately NOT NFKC, which would change meaning
    ("(1)" for U+2460, "fi" for the U+FB01 ligature).
    """
    # Normalizing per character keeps a 1:many map from old index to new. That is
    # only equal to normalizing the whole string when no composition happens ACROSS
    # a character boundary (e.g. "e" + U+0301 -> "e-acute"), which would merge two
    # source characters into one and invalidate the map. Verify, never assume.
    pieces = [unicodedata.normalize("NFC", ch) for ch in text]
    normalized = "".join(pieces)
    if normalized != unicodedata.normalize("NFC", text):
        raise ValueError(
            "NFC composed across character boundaries; the offset map would be "
            "wrong. This text needs a real alignment, not a per-character map."
        )

    index_map, pos = [], 0
    for piece in pieces:
        index_map.append(pos)
        pos += len(piece)
    index_map.append(pos)
    return normalized, index_map


def _normalize_text_and_spans(text, spans, start_key="start", end_key="end"):
    """NFC-normalize `text` and shift every span's offsets to match.

    Returns `(normalized_text, new_spans)`. Spans are copied, not mutated. A no-op
    (bar the copy) for text that is already NFC, which is all but 3 of the 8,894
    examples across WMT / UniversalNER / MultiGEC.
    """
    text = text or ""
    if unicodedata.normalize("NFC", text) == text:
        return text, [dict(s) for s in spans]

    normalized, index_map = _nfc_with_offset_map(text)
    out = []
    for span in spans:
        start, end = span[start_key], span[end_key]
        if not (0 <= start <= end <= len(text)):
            raise ValueError(
                f"span {(start, end)} out of range for text of length {len(text)}"
            )
        new = {**span, start_key: index_map[start], end_key: index_map[end]}
        # UNER's raw spans carry the entity surface form; keep it consistent with
        # the text it now indexes into (unused downstream, but a stale copy here
        # would be a trap for anyone who does start using it).
        if isinstance(new.get("text"), str):
            new["text"] = unicodedata.normalize("NFC", new["text"])
        out.append(new)
    return normalized, out


# ---------------------------------------------------------------------------
# Character tasks -- the model sees the ORIGINAL text, verbatim
# ---------------------------------------------------------------------------

def _character_example(text, gold_spans, example_id, extra=None):
    """Shared shape for Toxic Spans / LegalQAEval / WMT.

    Single choke point for NFC normalization: every character task passes through
    here, so a new dataset gets it without having to know it needs it.
    """
    text, gold_spans = _normalize_text_and_spans(text, gold_spans)
    example = {
        "text": text,
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


def load_multigec():
    """MultiGEC (W&I 2024 English dev) from Semin et al.'s reconstructed JSON.

    One file, 504 examples, 7,100 gold spans -- no subsets to concatenate.

    Two things are deliberately inherited from their `MultiGECParser` rather
    than cleaned up, so our gold is theirs and the numbers stay comparable:

    - **`correction` is dropped.** Their parser stores the replacement text, but
      `metrics.py` reads only `start`/`end`/`label`, so the correction never
      enters scoring. This is a span-labeling task, not a rewriting one.
    - **19 `M` spans sit one character past the end of `text`.** Their parser
      computes `start_char = sum(len(t) + 1 for t in tokens[:start_token])`,
      which for an insertion after the last token yields `len(text) + 1`. Those
      spans are unreachable for any method, ours and theirs alike, so they are
      kept: dropping them would inflate our recall against their published row.

    `M` is a zero-length insertion point (`start == end`), and is the only label
    that ever is -- verified on all 7,100 spans: every one of the 1,599 `M`
    spans is empty and no `R`/`U` span is. That exact correspondence is what
    lets the constrained processor gate empty spans on the label alone, via
    `allow_empty_span_labels={"M"}`.
    """
    path = SEMIN_DATA / "multigec" / "multigec_en.json"
    if not path.is_file():
        raise SystemExit(
            f"MultiGEC not found at {path}.\n"
            f"Run: python scripts/data/prepare_semin_data.py --with-multigec"
        )

    examples = []
    for idx, row in enumerate(json.loads(path.read_text(encoding="utf-8"))):
        gold_spans = [
            {"start": s["start"], "end": s["end"], "label": s["label"]}
            for s in row["spans"]
        ]
        examples.append(_character_example(row["text"], gold_spans, str(idx)))
    return examples


def wmt_subsets():
    """The 24 WMT24 ESA files on disk (source-target-domain), sorted."""
    directory = SEMIN_DATA / "wmt"
    if not directory.is_dir():
        raise SystemExit(
            f"WMT24 ESA not found at {directory}.\n"
            f"Run: python scripts/data/prepare_semin_data.py"
        )
    return sorted(p.stem.removeprefix("wmt-") for p in directory.glob("wmt-*.json"))


def load_wmt(subset="all"):
    """WMT24 ESA (error span annotation). Gold spans carry MAJOR/MINOR severity
    and character offsets into `text`, the translation -- copied verbatim by the
    model, same convention as Toxic Spans / LegalQAEval. `source` is the
    untagged source-language text, shown to the model as context only (never
    copied or tagged), the same role `question` plays for LegalQA.
    """
    available = wmt_subsets()
    names = available if subset == "all" else [subset]
    for name in names:
        if name not in available:
            raise SystemExit(
                f"Unknown WMT subset {name!r}. Available:\n  " + "\n  ".join(available)
            )

    examples = []
    for name in names:
        path = SEMIN_DATA / "wmt" / f"wmt-{name}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        for idx, row in enumerate(rows):
            gold_spans = [
                {"start": s["start"], "end": s["end"], "label": s["label"]}
                for s in row["spans"]
            ]
            examples.append(_character_example(
                row["text"], gold_spans, f"{name}:{idx}",
                extra={
                    "source": row["source"],
                    "source_language": row["source_language"],
                    "target_language": row["target_language"],
                    "dataset_name": name,
                },
            ))
    return examples
