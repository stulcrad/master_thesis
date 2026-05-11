from typing import List, Dict, Tuple
import re, ast
import statistics

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
) -> str:
    """Generate tagged text using either constrained or unconstrained decoding."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_text},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    if eval_model == "constrained":
        return generate_constrained_markup(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
    return generate_unconstrained_markup(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
    )

def generate_unconstrained_markup(
    model,
    tokenizer,
    inputs,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> str:
    """Generate unconstrained tagged text using a HF model + tokenizer."""

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
    )
    new_ids = outputs[0][inputs["input_ids"].shape[1]:]#.tolist()

    return tokenizer.decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )#.strip()

def generate_constrained_markup(
    model,
    tokenizer,
    processor,
    inputs,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> str:
    """Generate constrained tagged text using the trie processor."""
    outputs = model.generate(
        **inputs,
        logits_processor=[processor],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
    )

    new_ids = outputs[0][inputs["input_ids"].shape[1]:]#.tolist()

    return tokenizer.decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )#.strip()

def parse_entities_from_tagged_output(tagged_text: str, valid_labels: set) -> Dict:
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


def entities_to_bio_tags(tokens: List[str], entities: List[Dict], valid_labels: set) -> Tuple[List[str], int]:
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
