import json
from rapidfuzz import fuzz

def json_safe_parse(text: str) -> tuple:
    """
    Extract and parse a JSON array from model output.

    Returns:
        (parsed, ok): `parsed` is the parsed list (empty on failure, or when
        the model validly predicted zero entities). `ok` is False only when
        the output could not be parsed into a JSON list at all -- no `[...]`
        array present, malformed JSON, or the parsed value isn't a list.
        `ok=True, parsed=[]` means the model validly predicted zero entities;
        this is NOT a format failure and must not be conflated with `ok=False`.
    """
    try:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end <= start:
            return [], False
        parsed = json.loads(text[start:end])
        if not isinstance(parsed, list):
            return [], False
        return parsed, True
    except Exception:
        return [], False


def find_best_fuzzy_match(context:str, full_text:str, threshold: float = 0.7, matching_type:str = 'anchor') -> tuple:
    """
    Find the best fuzzy match of context in full_text using sliding window.
    
    Args:
        context (str): The context string to match.
        full_text (str): The full text string to search within.
        threshold (float): Minimum similarity ratio to consider a match.
        matching_type (str): Type of fuzzy matching to use: 'anchor', 'partial', 'full'.
        
    Returns:
        (start_pos, end_pos, similarity) if match found above threshold, else (None, None, 0.0).
    """
    # Normalize inputs for case-insensitive matching
    context_lower = context.lower()
    full_text_lower = full_text.lower()
    context_len = len(context_lower)

    # 1) Fast exit on exact match
    exact_start = full_text_lower.find(context_lower)
    if exact_start != -1:
        return (exact_start, exact_start + context_len, 1.0)
    
    # Fuzzy match with sliding window
    best_ratio = 0.0
    best_start = None
    best_end = None

    # 2) Fuzzy matching strategies
    if matching_type == 'anchor':   
        # ======== Partial ratio with anchor filtering ========
        # Use start of context as anchor
        anchor_len = 6 if context_len > 10 else 3
        anchor = context_lower[:anchor_len]
        # Find all positions of anchor in full_text
        anchor_starts = []
        pos = full_text_lower.find(anchor)

        if pos == -1:
            for i in range(len(full_text_lower) - anchor_len + 1):
                window = full_text_lower[i:i + anchor_len]
                ratio = fuzz.ratio(anchor, window) / 100.0
                if ratio >= threshold:
                    anchor_starts.append(i)
        else:
            while pos != -1:
                anchor_starts.append(pos)
                pos = full_text_lower.find(anchor, pos + 1)
        # If no anchors found, use regular sliding window
        if not anchor_starts:
            anchor_starts = range(0, len(full_text_lower) - context_len + 1, max(1, context_len // 3))

        for astart in anchor_starts:
            # Check +-15 characters around context
            win_start = max(0, astart - 15)
            win_end = min(len(full_text_lower), astart + context_len + 15)
            window = full_text_lower[win_start:win_end]
            # Compute partial ratio with score cutoff, to find promising windows
            ratio = fuzz.partial_ratio(context_lower, window, score_cutoff=int(threshold*100)) / 100.0
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                # Find best matching substring within window
                best_local_pos = -1
                best_local_score = 0
                # Go through all possible substrings of context length
                for i in range(0, len(window) - context_len + 1):
                    chunk = window[i:i + context_len]
                    # Find ratio with score cutoff to speed up
                    chunk_ratio = fuzz.ratio(context_lower, chunk, score_cutoff=int(best_local_score*100)) / 100.0
                    if chunk_ratio > best_local_score:
                        best_local_score = chunk_ratio
                        best_local_pos = i

                if best_local_pos != -1:
                    best_start = win_start + best_local_pos
                    best_end = best_start + context_len

    if matching_type == 'partial':
        # === Partial ratio with steps around window size ===
        # Slide in steps to reduce number of comparisons
        step = max(2, context_len // 10)

        # Allow window bigger than context to catch punctuation/spacing differences
        min_w = max(5, context_len - 5)
        max_w = context_len + 10

        for window_size in range(min_w, max_w + 1):
            for i in range(0, len(full_text_lower) - window_size + 1, step):
                window = full_text_lower[i:i + window_size]
                ratio = fuzz.partial_ratio(context_lower, window, score_cutoff=int(threshold*100)) / 100.0
                if ratio > best_ratio and ratio >= threshold:
                    best_ratio = ratio
                    best_start = i
                    best_end = i + window_size
        
    if matching_type == 'full':
        # === Full ratio with windows ===
        # Try different window sizes around the context length
        for window_size in range(max(1, context_len - 5), context_len + 10):
            for i in range(len(full_text_lower) - window_size + 1):
                window = full_text_lower[i:i + window_size]
                ratio = fuzz.ratio(context_lower, window) / 100.0
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_start = i
                    best_end = i + window_size

    # 3) Return best match if above threshold
    if best_ratio >= threshold:
        return (best_start, best_end, best_ratio)
    else:
        return (None, None, 0.0)

def assign_spans_from_context(full_text_tokens: list, entities: list, fuzzy: bool,
                                 fuzzy_threshold: float = 0.7, matching_type:str = 'anchor',
                                 json_parse_ok: bool = True, return_stats: bool = False):
    """
    Assign BIO tags to tokens based on extracted entities with context.

    Args:
        full_text_tokens (list): List of tokens from the full text.
        entities (list): List of dicts from LLM output {"entity": ..., "label": ..., "context": ...}
        fuzzy (bool): Whether to use fuzzy matching for context.
        fuzzy_threshold (float): Similarity threshold for fuzzy matching.
        matching_type (str): Type of fuzzy matching to use: 'anchor', 'partial', 'full'.
        json_parse_ok (bool): Whether the raw model output parsed into a valid
            JSON list at all (see json_safe_parse). Feeds into stats['format_invalid']
            alongside per-entity key checks, so a fully unparseable generation is
            distinguished from one that validly predicted zero entities.

    Returns:
        list: BIO tags corresponding to full_text_tokens.
        tuple(list, dict): When return_stats=True, also returns matching outcome counters.

    stats['format_invalid'] is a single per-generation 0/1 flag (1 if the top-level
    JSON failed to parse, OR at least one entity object was individually malformed),
    directly comparable across methods -- e.g. against a grammar-constrained baseline
    (xgrammar/Outlines), which should force this to 0 by construction, or against the
    constrained-generation wrong-text rate. This is distinct from stats['invalid_entity_format'],
    which counts individually malformed entity objects within an otherwise-parseable array.
    """
    tags = ['O'] * len(full_text_tokens)
    full_text_str = " ".join(full_text_tokens)

    stats = {
        'processed_entities': 0,
        'successful_entities': 0,
        'invalid_entity_format': 0,
        'format_invalid': 0,
        'context_not_in_input': 0,
        'entity_not_in_context': 0,
        'fuzzy_helped': 0,
        'exact_match': 0,
    }

    for ent in entities:
        if 'entity' not in ent or 'label' not in ent or 'context' not in ent:
            print(f"\nInvalid entity format: {ent}")
            stats['invalid_entity_format'] += 1
            continue
        stats['processed_entities'] += 1

        entity_text = ent['entity']
        entity_label = ent['label']
        context_text = ent['context']
        fuzzy_used = False

        if fuzzy:
            # Try exact match first, then fuzzy context match.
            context_start = full_text_str.lower().find(context_text.lower())
            if context_start != -1:
                context_end = context_start + len(context_text)
            else:
                context_start, context_end, _ = find_best_fuzzy_match(
                    context_text,
                    full_text_str,
                    threshold=fuzzy_threshold,
                    matching_type=matching_type,
                )
                if context_start is not None:
                    fuzzy_used = True

            if context_start is None:
                stats['context_not_in_input'] += 1
                continue

            matched_context = full_text_str[context_start:context_end]
            entity_start_in_context = matched_context.lower().find(entity_text.lower())
            if entity_start_in_context == -1:
                entity_start_in_context, _, _ = find_best_fuzzy_match(
                    entity_text,
                    matched_context,
                    threshold=fuzzy_threshold,
                    matching_type=matching_type,
                )
                if entity_start_in_context is None:
                    stats['entity_not_in_context'] += 1
                    continue
                fuzzy_used = True
        else:
            context_start = full_text_str.lower().find(context_text.lower())
            if context_start == -1:
                stats['context_not_in_input'] += 1
                continue

            context_end = context_start + len(context_text)
            matched_context = full_text_str[context_start:context_end]

            entity_start_in_context = matched_context.lower().find(entity_text.lower())
            if entity_start_in_context == -1:
                stats['entity_not_in_context'] += 1
                continue

        entity_char_start = context_start + entity_start_in_context
        entity_char_end = entity_char_start + len(entity_text)

        char_idx = 0
        for i, tok in enumerate(full_text_tokens):
            tok_start = char_idx
            tok_end = tok_start + len(tok)
            if tok_end > entity_char_start and tok_start < entity_char_end:
                tag_prefix = "B-" if i-1 < 0 or tags[i-1] == "O" or not tags[i-1].endswith(entity_label) else "I-"
                tags[i] = f"{tag_prefix}{entity_label}"
            char_idx += len(tok) + 1

        stats['successful_entities'] += 1
        if fuzzy and fuzzy_used:
            stats['fuzzy_helped'] += 1
        else:
            stats['exact_match'] += 1

    stats['format_invalid'] = 1 if (not json_parse_ok or stats['invalid_entity_format'] > 0) else 0

    if return_stats:
        return tags, stats
    return tags
