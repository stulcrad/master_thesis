from typing import List, Dict, Tuple
import re

VALID_LABELS = {"PER", "LOC", "ORG", "MISC"}
LABEL_PATTERN = "|".join(VALID_LABELS)
# Create a regex pattern to extract the labeled spans from the generated text
SPAN_RE = re.compile(rf"<SPAN><LABEL>({LABEL_PATTERN})</LABEL>(.*?)</SPAN>", re.DOTALL)

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

def parse_entities_from_tagged_output(tagged_text: str) -> Dict:
    """
    Parse <SPAN><LABEL>..</LABEL>entity</SPAN> blocks and return entities with char offsets,
    and the reconstructed text.
    """
    cursor = 0
    plain_parts: List[str] = []
    entities: List[Dict] = []

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

    invalid_labels = [ent for ent in entities if ent["label"] not in VALID_LABELS]

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


def entities_to_bio_tags(tokens: List[str], entities: List[Dict]) -> Tuple[List[str], int]:
    """Convert entity char spans to token-level BIO tags for the same tokenization as input text."""
    token_spans = build_token_char_spans(tokens)
    tags = ["O"] * len(tokens)
    unaligned_count = 0

    entities_sorted = sorted(entities, key=lambda x: (x["start"], x["end"]))
    for ent in entities_sorted:
        label = ent.get("label")
        if label not in VALID_LABELS:
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
