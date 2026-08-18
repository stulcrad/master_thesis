from typing import List, Dict, Optional, Tuple
import re, ast
import json
import os
import statistics
import time
from utils.model_reasoning_utils import find_answer_start

# -------------------------
# File I/O helpers
# -------------------------
def open_jsonl_writer(path: str):
    """
    Open a per-example predictions JSONL file (one line per generation).

    Truncates any existing file: one file corresponds to one full run of one
    experiment config (all seeds; the per-line `seed` field disambiguates).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, "w", encoding="utf-8")


def log_jsonl(fh, record: dict) -> None:
    """Write one prediction record and flush immediately, so a crashed run
    keeps every example logged up to the crash."""
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    fh.flush()

# -------------------------
# Generation helpers
# -------------------------
def generate_markup(
    model,
    tokenizer,
    processor,
    eval_model: str,
    input_text: str,
    system_prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    reasoning_model: bool = False,
    reasoning_effort: str = None,
    reasoning_end_marker=None,
    reasoning_start_marker=None,
    stats_out: dict = None,
    repetition_penalty: float = None,
    temperature: float = None,
    top_p: float = None,
    top_k: int = None,
    min_p: float = None,
) -> Tuple[str, int, float]:
    """
    Generate tagged text using either constrained or unconstrained decoding.

    Returns (text, num_output_tokens, generation_seconds), where num_output_tokens
    is the TOTAL generated tokens (reasoning + answer). For a reasoning model the
    returned `text` is the ANSWER segment only (reasoning stripped), so downstream
    parsing sees just the constrained/answer output.

    Reasoning/answer split (two-phase reporting):
      - constrained: taken from processor.output_start_index (recorded at the boundary).
      - unconstrained: found by locating `reasoning_end_marker` (a list of token ids)
        in the generated ids.

    Sampling kwargs (`top_p`, `top_k`, `min_p`, `repetition_penalty`) follow a
    "None means do not pass" convention, so leaving one unset lets the model's own
    generation_config.json apply instead of being overridden with a null.

    If `stats_out` (a dict) is passed, it is filled with num_reasoning_tokens,
    num_answer_tokens, reasoning_text, answer_text, found_reasoning_end,
    reasoning_marker_seen, and raw_token_ids (the full list of generated token IDs,
    reasoning + answer -- kept because decode() with errors='replace' is destructive,
    so a corrupted decoded string alone cannot be traced back to the token that
    produced it). Callers that don't pass it get the unchanged 3-tuple behavior.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_text},
    ]

    template_kwargs = {}
    if reasoning_effort is not None:
        template_kwargs["reasoning_effort"] = reasoning_effort

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=reasoning_model,
        **template_kwargs
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    sampling_kwargs = dict(
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )

    if eval_model == "constrained":
        (full_text, total_tokens, generation_seconds, n_reason, n_answer,
         reasoning_text, answer_text, reasoning_skipped, marker_seen,
         raw_token_ids) = generate_constrained_markup(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            reasoning_model=reasoning_model,
            reasoning_end_marker=reasoning_end_marker,
            reasoning_start_marker=reasoning_start_marker,
            **sampling_kwargs,
        )
    else:
        (full_text, total_tokens, generation_seconds, n_reason, n_answer,
         reasoning_text, answer_text, reasoning_skipped, marker_seen,
         raw_token_ids) = generate_unconstrained_markup(
            model=model,
            tokenizer=tokenizer,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            reasoning_model=reasoning_model,
            reasoning_end_marker=reasoning_end_marker,
            reasoning_start_marker=reasoning_start_marker,
            **sampling_kwargs,
        )

    if reasoning_model:
        # Answer segment only (reasoning stripped via the boundary split).
        text = answer_text
    else:
        text = full_text

    if stats_out is not None:
        stats_out["num_reasoning_tokens"] = n_reason
        stats_out["num_answer_tokens"] = n_answer
        stats_out["reasoning_text"] = reasoning_text
        stats_out["answer_text"] = answer_text
        # The end marker was genuinely found: an answer exists AND the block was not
        # merely skipped. Kept distinct from `reasoning_skipped` so the termination
        # study can separate "ran out of budget" from "never started thinking".
        stats_out["found_reasoning_end"] = bool(reasoning_model and n_answer > 0
                                                and not reasoning_skipped)
        stats_out["reasoning_skipped"] = bool(reasoning_skipped)
        stats_out["reasoning_marker_seen"] = marker_seen
        stats_out["raw_token_ids"] = raw_token_ids

    return text, total_tokens, generation_seconds


def _build_gen_kwargs(max_new_tokens, do_sample, temperature,
                      repetition_penalty=None, top_p=None, top_k=None, min_p=None) -> dict:
    """Assemble model.generate() kwargs, omitting anything unset.

    Two conventions matter here:
    - `None` means "do not pass the kwarg", so the model's own generation_config.json
      applies rather than being overwritten with a null.
    - Sampling-only knobs (temperature/top_p/top_k/min_p) are dropped entirely under
      greedy decoding. HF would otherwise warn "The following generation flags are not
      valid and may be ignored" and silently discard them, which makes a run look
      configured when it is not.
    """
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}

    if do_sample:
        for name, value in (("temperature", temperature), ("top_p", top_p),
                            ("top_k", top_k), ("min_p", min_p)):
            if value is not None:
                gen_kwargs[name] = value

    # repetition_penalty is valid under both greedy and sampling.
    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = repetition_penalty

    return gen_kwargs


def _split_reasoning_answer(tokenizer, new_ids, generation_seconds, boundary, is_reasoning,
                            reasoning_opened=True):
    """Split generated ids into (reasoning, answer) at `boundary` (an index into
    new_ids, or None). Returns
    (full_text, total_tokens, generation_seconds, n_reason, n_answer, reasoning_text,
     answer_text, reasoning_skipped).

    - not a reasoning model: everything is the answer (n_reason=0).
    - reasoning model, boundary found: split there; the answer is decoded WITHOUT
      special tokens (drops </think> / <|return|> / EOS), leaving clean tagged text.
    - reasoning model, no boundary: TWO cases, distinguished by `reasoning_opened`.
        * opened but never closed -> truncated mid-reasoning. No valid answer;
          everything counts as reasoning, answer empty.
        * never opened -> the model skipped the thought block and answered directly.
          The whole output IS the answer (n_reason=0, reasoning_skipped=True).

    Gemma-4 E2B/E4B must emit `<|channel>` themselves and sometimes go straight to 
    the answer on short inputs. Qwen (template opens `<think>`) and Harmony (end marker is the answer
    opener) cannot hit it, so their specs leave reasoning_start_marker=None and
    `reasoning_opened` stays True -- for them a missing marker still means truncation.
    """
    total_tokens = int(new_ids.shape[0])
    full_text = tokenizer.decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    if not is_reasoning:
        return full_text, total_tokens, generation_seconds, 0, total_tokens, "", full_text, False

    if boundary is None:
        if reasoning_opened:
            # Truncated mid-reasoning: no answer was produced.
            return full_text, total_tokens, generation_seconds, total_tokens, 0, full_text, "", False
        # Thought block never opened: the model answered directly.
        return full_text, total_tokens, generation_seconds, 0, total_tokens, "", full_text, True

    boundary = int(boundary)
    reasoning_ids = new_ids[:boundary]
    answer_ids = new_ids[boundary:]
    reasoning_text = tokenizer.decode(
        reasoning_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    ).strip()
    answer_text = tokenizer.decode(
        answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    ).strip()
    return (full_text, total_tokens, generation_seconds, boundary, total_tokens - boundary,
            reasoning_text, answer_text, False)


def _reasoning_was_opened(new_ids, reasoning_start_marker) -> bool:
    """Did the model actually open its reasoning block?

    True when no start marker is registered: those families have the block opened
    by the chat template (Qwen) or use the end marker as the answer opener
    (Harmony), so it is open by construction and a missing end marker can only mean
    truncation. Where a start marker IS registered (Gemma-4), the model has to emit
    it, so its absence means the thought block was skipped entirely.
    """
    if not reasoning_start_marker:
        return True
    return find_answer_start(new_ids.tolist(), list(reasoning_start_marker)) is not None


def generate_unconstrained_markup(
    model,
    tokenizer,
    inputs,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    reasoning_model: bool = False,
    reasoning_end_marker=None,
    reasoning_start_marker=None,
    repetition_penalty: float = None,
    top_p: float = None,
    top_k: int = None,
    min_p: float = None,
):
    """Generate unconstrained tagged text using a HF model + tokenizer.

    Returns the 7-tuple produced by _split_reasoning_answer, plus two more:
    `reasoning_marker_seen` (see generate_markup) and the raw generated token IDs
    (`new_ids.tolist()`) -- kept for post-hoc debugging of decode-level artifacts
    (e.g. U+FFFD in multi-byte scripts) that a decoded string alone can't diagnose,
    since decode() with errors='replace' destroys the original bytes.
    """
    gen_kwargs = _build_gen_kwargs(
        max_new_tokens, do_sample, temperature, repetition_penalty, top_p, top_k, min_p,
    )

    start = time.perf_counter()
    outputs = model.generate(
        **inputs,
        **gen_kwargs,
    )
    generation_seconds = time.perf_counter() - start
    new_ids = outputs[0][inputs["input_ids"].shape[1]:]

    # Marker scan is done unconditionally (whenever a marker exists) and is independent
    # of whether we split on it, so the reasoning-OFF arm can be checked for leakage.
    marker_at = (
        find_answer_start(new_ids.tolist(), list(reasoning_end_marker))
        if reasoning_end_marker else None
    )
    # Split only when this is being treated as a reasoning run.
    boundary = marker_at if reasoning_model else None

    return (*_split_reasoning_answer(
        tokenizer, new_ids, generation_seconds, boundary, reasoning_model,
        reasoning_opened=_reasoning_was_opened(new_ids, reasoning_start_marker),
    ), marker_at is not None, new_ids.tolist())


def generate_constrained_markup(
    model,
    tokenizer,
    processor,
    inputs,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    reasoning_model: bool = False,
    reasoning_end_marker=None,
    reasoning_start_marker=None,
    repetition_penalty: float = None,
    top_p: float = None,
    top_k: int = None,
    min_p: float = None,
):
    """Generate constrained tagged text using the trie processor.

    Returns the 7-tuple produced by _split_reasoning_answer, plus two more:
    `reasoning_marker_seen` (see generate_markup) and the raw generated token IDs
    (`new_ids.tolist()`) -- kept for post-hoc debugging of decode-level artifacts
    (e.g. U+FFFD in multi-byte scripts) that a decoded string alone can't diagnose,
    since decode() with errors='replace' destroys the original bytes.
    """
    gen_kwargs = _build_gen_kwargs(
        max_new_tokens, do_sample, temperature, repetition_penalty, top_p, top_k, min_p,
    )

    start = time.perf_counter()
    outputs = model.generate(
        **inputs,
        logits_processor=[processor],
        **gen_kwargs,
    )
    generation_seconds = time.perf_counter() - start

    prompt_len = inputs["input_ids"].shape[1]
    new_ids = outputs[0][prompt_len:]

    boundary = None
    if reasoning_model:
        # output_start_index is absolute (prompt-inclusive); make it relative to new_ids.
        # getattr guards the non-trie / non-reasoning processors (e.g. xgrammar) that
        # never set it.
        osi = getattr(processor, "output_start_index", None)
        if osi is not None:
            boundary = osi - prompt_len

    # Leakage check for the reasoning-OFF arm, independent of the split above.
    marker_seen = (
        find_answer_start(new_ids.tolist(), list(reasoning_end_marker)) is not None
        if reasoning_end_marker else False
    )

    return (*_split_reasoning_answer(
        tokenizer, new_ids, generation_seconds, boundary, reasoning_model,
        reasoning_opened=_reasoning_was_opened(new_ids, reasoning_start_marker),
    ), marker_seen, new_ids.tolist())

# -------------------------
# Evaluation helpers for text
# -------------------------
def parse_spans_from_tagged_output(tagged_text: str, valid_labels: set) -> Dict:
    """
    Parse <SPAN><LABEL>..</LABEL>entity</SPAN> blocks and return entities with char offsets,
    and the reconstructed text.
    """
    cursor = 0
    plain_parts: List[str] = []
    entities: List[Dict] = []

    # Create a regex pattern to extract the labeled spans from the generated text
    LABEL_PATTERN = "|".join(valid_labels)
    SPAN_RE = re.compile(rf"<SPAN><LABEL>({LABEL_PATTERN})</LABEL>(.*?)</SPAN>", re.DOTALL)

    for match in SPAN_RE.finditer(tagged_text):
        plain_parts.append(tagged_text[cursor:match.start()])

        label = match.group(1).strip()
        entity_text = match.group(2)
        entity_start = sum(len(p) for p in plain_parts)
        entity_end = entity_start + len(entity_text)

        plain_parts.append(entity_text)
        entities.append({
            "entity": entity_text,
            "label": label,
            "start": entity_start,
            "end": entity_end,
        })
        cursor = match.end()

    plain_parts.append(tagged_text[cursor:])
    reconstructed_text = "".join(plain_parts)

    invalid_labels = [ent for ent in entities if ent["label"] not in valid_labels]

    return {
        "entities": entities,
        "reconstructed_text": reconstructed_text,
        "invalid_label_count": len(invalid_labels),
        "span_count": len(entities),
    }


def build_token_char_spans(tokens: List[str]) -> List[Tuple[int, int]]:
    """Character spans for tokens in the canonical CoNLL text: ' '.join(tokens)."""
    spans: List[Tuple[int, int]] = []
    pos = 0
    for i, tok in enumerate(tokens):
        start = pos
        end = start + len(tok)
        spans.append((start, end))
        pos = end + (1 if i < len(tokens) - 1 else 0)
    return spans


def spans_to_bio_tags(tokens: List[str], entities: List[Dict], valid_labels: set,
                      token_spans: Optional[List[Tuple[int, int]]] = None) -> Tuple[List[str], int]:
    """Convert entity char spans to token-level BIO tags for the same tokenization as input text.

    `token_spans` are the tokens' character offsets in the text the spans index
    into. Left as None it defaults to `build_token_char_spans(tokens)`, i.e. the
    offsets of `" ".join(tokens)` -- correct for CoNLL/UNER and for every caller
    that existed before. Tasks that feed the model the ORIGINAL text (Toxic
    Spans, LegalQAEval, where runs of whitespace are preserved) must pass the
    true offsets from `tokenize_with_offsets`, otherwise every offset after a
    repeated space or newline is wrong.
    """
    if token_spans is None:
        token_spans = build_token_char_spans(tokens)
    tags = ["O"] * len(tokens)
    unaligned_count = 0

    entities_sorted = sorted(entities, key=lambda x: (x["start"], x["end"]))
    for ent in entities_sorted:
        label = ent.get("label")
        if label not in valid_labels:
            continue

        e_start = int(ent.get("start", -1))
        e_end = int(ent.get("end", -1))
        if e_start < 0 or e_end <= e_start:
            continue

        covered = [
            i for i, (t_start, t_end) in enumerate(token_spans)
            if max(t_start, e_start) < min(t_end, e_end)
        ]
        if not covered:
            unaligned_count += 1
            continue

        tags[covered[0]] = f"B-{label}"
        for idx in covered[1:]:
            tags[idx] = f"I-{label}"

    return tags, unaligned_count


def validate_reconstruction(reconstructed_text: str, input_text: str) -> bool:
    """Return True only when reconstructed text exactly matches the input text."""
    return reconstructed_text == input_text


def shorten_text(text: str, max_chars: int = 220) -> str:
    """Keep diagnostics rows readable in notebook tables."""
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."

def tokenize_with_offsets(text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Tokenize text by splitting on whitespace, and return both tokens and their character offsets."""
    tokens, offsets = [] , []
    i, n = 0, len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        j = i
        while j < n and not text[j].isspace():
            j += 1
        tokens.append(text[i:j])
        offsets.append((i, j))
        i = j
    return tokens, offsets

# -------------------------
# Evaluation helpers for entity-level metrics
# -------------------------
def compute_character_f1(
        gold_chars: set,
        pred_chars: set,
) -> Tuple[float, float, float]:
    """Character-level precision, recall, and F1.

    Follows Pavlopoulos et al. (ACL 2022):
      P  = |pred ∩ gold| / |pred|
      R  = |pred ∩ gold| / |gold|
      F1 = 2·P·R / (P+R)

    Special case (Pavlopoulos et al.): if gold is empty,
      F1 = 1.0 when pred is also empty, F1 = 0.0 otherwise.

    Returns (precision, recall, f1).
    """
    if not gold_chars:
        return (1.0, 1.0, 1.0) if not pred_chars else (0.0, 0.0, 0.0)
    if not pred_chars:
        return (0.0, 0.0, 0.0)
    inter = len(gold_chars & pred_chars)
    p = inter / len(pred_chars)
    r = inter / len(gold_chars)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1

def mean_std(values):
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def to_pct(v): return v * 100.0
def format_pm(m, s): return f"{m:.2f} ± {s:.2f}"

# -------------------------
# Span parsing helpers
# -------------------------
def parse_position(raw_position: str) -> List[int]:
    """
    Return a plain Python list of int char indices.
    """
    parsed = ast.literal_eval(raw_position)
    return [int(x) for x in parsed]

def chars_to_spans(char_indices: List[int]) -> List[Tuple[int, int]]:
    """Merge sorted individual char indices into (start, end) tuples.

    Examples:
        [7, 8, 9, 10]           → [(7, 11)]
        [0,1,2,3,4,5,15,16,17]  → [(0, 6), (15, 18)]
    """
    if not char_indices:
        return []
    indices = sorted(set(char_indices))
    spans: List[Tuple[int, int]] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            spans.append((start, prev + 1))
            start = prev = idx
    spans.append((start, prev + 1))
    return spans

def example_to_tokens(text: str) -> List[str]:
    tokens = text.split() if text else []
    return tokens
