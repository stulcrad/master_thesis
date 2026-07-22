from typing import List, Dict, Tuple
import re, ast
import json
import os
import statistics
import time

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

def extract_harmony_final_channel(text: str) -> str:
    """Isolate GPT-OSS's harmony 'final' channel from its raw decoded output.

    Harmony wraps the answer as `...<|channel|>final<|message|>ANSWER<|return|>`, extract
    this answer to skip any preceding 'analysis' channel content.
    """
    marker = "<|channel|>final<|message|>"
    idx = text.rfind(marker)
    if idx == -1:
        return text
    tail = text[idx + len(marker):]
    for end_marker in ("<|return|>", "<|end|>", "<|call|>", "<|start|>"):
        end_idx = tail.find(end_marker)
        if end_idx != -1:
            tail = tail[:end_idx]
    return tail.strip()

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
    temperature: float,
    reasoning_effort: str = None,
) -> Tuple[str, int, float]:
    """
    Generate tagged text using either constrained or unconstrained decoding.

    Returns (text, num_output_tokens, generation_seconds)
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
        enable_thinking=False,
        **template_kwargs,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # GPT-OSS's harmony channels ("analysis" vs "final") are only
    # distinguishable via the special-token delimiters, so keep them in the
    # decoded text for this one model family; every other model keeps the
    # prior skip_special_tokens=True behavior untouched.
    skip_special_tokens = reasoning_effort is None

    if eval_model == "constrained":
        text, num_output_tokens, generation_seconds = generate_constrained_markup(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            skip_special_tokens=skip_special_tokens,
        )
    else:
        text, num_output_tokens, generation_seconds = generate_unconstrained_markup(
            model=model,
            tokenizer=tokenizer,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            skip_special_tokens=skip_special_tokens,
        )

    if reasoning_effort is not None:
        text = extract_harmony_final_channel(text)

    return text, num_output_tokens, generation_seconds

def generate_unconstrained_markup(
    model,
    tokenizer,
    inputs,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    skip_special_tokens: bool = True,
) -> Tuple[str, int, float]:
    """Generate unconstrained tagged text using a HF model + tokenizer."""

    start = time.perf_counter()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
    )
    generation_seconds = time.perf_counter() - start
    new_ids = outputs[0][inputs["input_ids"].shape[1]:]#.tolist()

    text = tokenizer.decode(
        new_ids,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )#.strip()
    return text, new_ids.shape[0], generation_seconds

def generate_constrained_markup(
    model,
    tokenizer,
    processor,
    inputs,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    skip_special_tokens: bool = True,
) -> Tuple[str, int, float]:
    """Generate constrained tagged text using the trie processor."""
    start = time.perf_counter()
    outputs = model.generate(
        **inputs,
        logits_processor=[processor],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
    )
    generation_seconds = time.perf_counter() - start

    new_ids = outputs[0][inputs["input_ids"].shape[1]:]#.tolist()

    text = tokenizer.decode(
        new_ids,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )#.strip()
    return text, new_ids.shape[0], generation_seconds

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


def spans_to_bio_tags(tokens: List[str], entities: List[Dict], valid_labels: set) -> Tuple[List[str], int]:
    """Convert entity char spans to token-level BIO tags for the same tokenization as input text."""
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
