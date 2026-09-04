"""
Context-based (prompting) baseline on MultiGEC (W&I 2024 English dev).

Mirror image of `evaluationMultiGEC_CG.py` -- same model registry, reasoning handling, seeds, sampling, sharding,
dataset objects and metrics. The model is shown the ORIGINAL learner text verbatim, exactly as in the constrained arm,
so gold offsets are used as they come and the two arms are scored on identical inputs. Only the requested output format
differs: a JSON array of {entity, label, context} objects instead of inline markup over a verbatim copy.

Zero-length M spans are what make this dataset the interesting one, and they are the one thing a matching-based method
cannot express directly: a missing word has no text to name.

1. The prompt asks for the token immediately BEFORE the gap as the M item's "entity", and
   `assign_char_spans_from_context` converts that to a zero-length span at `entity_end + 1`. It is an indirection the
   constrained arm does not need.
2. Character sets are built with `spans_to_char_set`, NOT `range()`. A zero-length span covers no characters, so
   `range(start, end)` is empty and every M span silently vanishes from the score -- 22.5% of this gold.

Metrics, matching the constrained script: - `char_f1`: per-example character F1 over offsets only (label-agnostic), then
averaged -- the macro analogue of Semin
  et al.'s SOFT F1.
- `semin_hard_f1` / `semin_soft_f1`: their pooled (micro) character-overlap F1. Hard requires R/M/U to match.
- `macro_hard_f1`: per-example hard F1, then averaged -- the macro/label-sensitive corner.
- token budget and throughput, and the reasoning termination counters.

Plus the failure modes that exist only on this side of the comparison: spans the model named that are not in the text
(`span_not_found`), labels outside {R, M, U} (`invalid_label`), and responses that did not parse (`format_invalid`). The
first two are structurally impossible under constrained generation.
"""
import argparse
import os
import random
import statistics
import sys
import time
from collections import Counter

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from utils.utils_functions import (
    generate_markup, mean_std, to_pct, format_pm, compute_character_f1, spans_to_char_set,
    open_jsonl_writer, log_jsonl,
)
from utils.span_datasets import load_multigec, MULTIGEC_LABELS
from utils.semin_metrics import compute_overlap_counts, f1_from_counts
from utils.context_matching_utils import (
    json_safe_parse, assign_char_spans_from_context, classify_generation_error, ERROR_TYPES,
)
from utils.system_prompts import SYSTEM_PROMPT_CONTEXT_BASE_MULTIGEC
from utils.model_registry import get, resolve_sampling

# -------------------------
# Model configuration
# -------------------------
parser = argparse.ArgumentParser("Evaluate the context-based prompting baseline on MultiGEC.")
parser.add_argument("--model", required=True, type=str, help="Model name or ID to evaluate.")
parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True,
                    help="Whether to enable reasoning in the model's prompt.")
parser.add_argument("--reasoning-effort", choices=['low', 'medium', 'high', 'xhigh'], default=None,
                    help="Reasoning effort. The union of both families' levels: Harmony/GPT-OSS takes "
                         "low|medium|high, Qwen3.8 takes low|medium|xhigh (xhigh is its default). "
                         "Which are actually valid is checked per model against the registry.")
parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty for the model.")
parser.add_argument("--temperature", type=float, default=None, help="Temperature for sampling.")
parser.add_argument("--top-p", type=float, default=None, help="Top-p for sampling.")
parser.add_argument("--top-k", type=int, default=None, help="Top-k for sampling.")
parser.add_argument("--min-p", type=float, default=None, help="Minimum probability for sampling.")
parser.add_argument("--max-examples", type=int, default=None,
                    help="Maximum number of examples to evaluate. None means all examples.")
parser.add_argument("--max-new-tokens", type=int, default=16384, help="Maximum number of new tokens to generate.")
parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Seeds to run and average over.")
parser.add_argument("--shard", type=str, default=None,
                    help="Optional: i/N -- split the dataset into N shards and run only shard i.")

args = parser.parse_args()

# -------------------------
# Model, reasoning, eval configuration, and guards
# -------------------------
model_name = args.model
spec = get(model_name)

if args.enable_thinking and not spec.reasoning:
    raise SystemExit(f"{args.model} has no reasoning mode (no marker registered)")
if not args.enable_thinking and not spec.reasoning_off_supported:
    raise SystemExit(f"{args.model} cannot disable reasoning (reasoning_off_supported=False)")
if args.reasoning_effort and not spec.supports_reasoning_effort:
    raise SystemExit(f"{args.model} does not support reasoning effort (no levels registered)")
if args.reasoning_effort and args.reasoning_effort not in spec.reasoning_effort_levels:
    raise SystemExit(
        f"{args.model} does not accept --reasoning-effort {args.reasoning_effort!r}; "
        f"valid levels for this model: {', '.join(spec.reasoning_effort_levels)}"
    )
reasoning_model = args.enable_thinking and spec.reasoning

print(f"\nLoading model/tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
)

reasoning_end_marker = tokenizer(spec.reasoning_end_marker, add_special_tokens=False).input_ids if spec.reasoning else None
# Only set for families where the MODEL emits the opener (Gemma-4). None elsewhere, which means the block is open by
# construction -- see model_registry.ModelSpec.
reasoning_start_marker = tokenizer(spec.reasoning_start_marker, add_special_tokens=False).input_ids if spec.reasoning_start_marker else None
sampling = resolve_sampling(spec, reasoning_model,
                            temperature=args.temperature, top_p=args.top_p,
                            top_k=args.top_k, min_p=args.min_p)
DO_SAMPLE = sampling.pop("do_sample", False)

SEEDS = args.seeds
MAX_EXAMPLES = args.max_examples
MAX_NEW_TOKENS = args.max_new_tokens
repetition_penalty = args.repetition_penalty

# --shard i/N
shard_i, shard_N = None, None
if args.shard is not None:
    try:
        shard_i, shard_N = map(int, args.shard.split("/"))
    except ValueError:
        raise SystemExit(f"--shard must be of the form i/N, got {args.shard!r}")
    if not (0 <= shard_i < shard_N):
        raise SystemExit(f"--shard must satisfy 0 <= i < N, got {args.shard!r}")
    print(f"Sharding into {shard_N} shards, running shard {shard_i} (file tag: shard{shard_i + 1}of{shard_N}).")

# One text per prompt: the texts are independent essays, so concatenating them would only test long-input robustness.
BATCH_SIZE = 1
EVAL_INTERVAL = 20

# Exact matching only -- see the module docstring. A named constant rather than a flag so it lands in the CSV and the
# JSONL and cannot be silently different between two runs.
MATCHING = "exact"

# Overridable so the smoke test can write somewhere harmless; unset, it is the publication tree.
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "/home/stulcrad/master_thesis/Experiment_results_publication")

valid_labels = set(MULTIGEC_LABELS)
#: `M` is a zero-length insertion point, and the only label that ever is. See the module docstring.
INSERTION_LABELS = {"M"}

# -------------------------
# Load dataset
# -------------------------
print("Loading MultiGEC (W&I 2024 English dev)...")
raw = load_multigec()
print(f"Examples: {len(raw)}")
print(f"Max examples per iteration: {MAX_EXAMPLES}")

results = []

sampling_strategy = "sampling" if DO_SAMPLE else "greedy"


def shard_suffix():
    return f"_shard{shard_i + 1}of{shard_N}" if shard_i is not None else ""


def config_tag():
    parts = [f"think{int(args.enable_thinking)}"]
    if args.reasoning_effort: parts.append(f"effort_{args.reasoning_effort}")
    if args.repetition_penalty: parts.append(f"rep_{args.repetition_penalty}")
    if args.temperature is not None: parts.append(f"temp_{args.temperature}")
    return "_".join(parts)


exp_metrics = []
print(
    f"\nEvaluating model={model_name}, shard={shard_suffix()}, reasoning_enabled={reasoning_model}, "
    f"reasoning_effort={args.reasoning_effort}, repetition_penalty={repetition_penalty}, "
    f"sampling_strategy={sampling_strategy}, matching={MATCHING}, max_examples={MAX_EXAMPLES}, seeds={SEEDS}"
)

model_short = model_name.split("/")[-1]
pred_fh = open_jsonl_writer(
    f"{RESULTS_ROOT}/GEC/Context-Based/Predictions/"
    f"multigec{shard_suffix()}_{model_short}_think_{args.enable_thinking}_{sampling_strategy}"
    f"_context_{MATCHING}_{config_tag()}.jsonl"
)

for seed in SEEDS:
    set_seed(seed)

    # Sample BEFORE sharding, so the capped set is identical across arms with different shard counts.
    if MAX_EXAMPLES is None or MAX_EXAMPLES >= len(raw):
        sampled = raw
    else:
        sampled = random.Random(seed).sample(raw, MAX_EXAMPLES)

    if shard_i is not None:
        total_examples = len(sampled)
        shard_size = (total_examples + shard_N - 1) // shard_N
        start_idx = shard_i * shard_size
        end_idx = min(start_idx + shard_size, total_examples)
        sampled = sampled[start_idx:end_idx]
        print(f"Sharding: {total_examples} examples -> {shard_N} shards of "
              f"~{shard_size}, running shard {shard_i}: {len(sampled)} examples")

    start_time = time.time()

    # Matching / format failure counters -- the reason this arm exists.
    format_invalid_count = 0
    context_not_in_input_count = 0
    entity_not_in_context_count = 0
    invalid_label_count = 0
    located_entity_count = 0
    unlocated_entity_count = 0
    proposed_entity_count = 0
    total_predictions = 0
    error_type_counts = Counter()

    # Reasoning / token budget / throughput.
    reasoning_unterminated_count = 0
    reasoning_skipped_count = 0
    reasoning_token_counts = []
    total_output_tokens = 0
    total_reasoning_tokens = 0
    total_answer_tokens = 0
    total_generation_seconds = 0.0

    # Macro soft F1: per-example F1 averaged over examples, label-agnostic.
    char_f1_per_ex = []
    char_p_per_ex = []
    char_r_per_ex = []
    # Macro-hard F1: same aggregation, label-sensitive -- see module docstring.
    hard_f1_per_ex = []
    hard_p_per_ex = []
    hard_r_per_ex = []
    # Semin et al.'s micro counters
    hard_overlap = hard_predicted = hard_gold = 0
    soft_overlap = soft_predicted = soft_gold = 0

    for idx in tqdm(range(len(sampled)), desc=f"seed {seed}", file=sys.stdout):
        example = sampled[idx]
        # The ORIGINAL text, shown verbatim -- no whitespace rebuild, same string the constrained arm copies.
        input_text = example["text"]
        gold_spans = example["gold_spans"]
        # spans_to_char_set, not range(): zero-length M spans must count.
        gold_chars = spans_to_char_set(gold_spans)

        # Handle edge case of empty input text (all spaces or empty string)
        if not input_text.strip():
            cp, cr, cf = compute_character_f1(gold_chars, set())
            char_f1_per_ex.append(cf)
            char_p_per_ex.append(cp)
            char_r_per_ex.append(cr)
            hf1 = f1_from_counts(0, 0, len(gold_chars))
            hard_f1_per_ex.append(hf1["f1"])
            hard_p_per_ex.append(hf1["precision"])
            hard_r_per_ex.append(hf1["recall"])
            continue

        gen_stats = {}
        try:
            generated, num_output_tokens, generation_seconds = generate_markup(
                model=model,
                tokenizer=tokenizer,
                processor=None,
                eval_model="unconstrained",
                input_text=input_text,
                system_prompt=SYSTEM_PROMPT_CONTEXT_BASE_MULTIGEC,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                reasoning_model=reasoning_model,
                reasoning_effort=args.reasoning_effort,
                reasoning_end_marker=reasoning_end_marker,
                reasoning_start_marker=reasoning_start_marker,
                stats_out=gen_stats,
                repetition_penalty=repetition_penalty,
                **sampling
            )
        except Exception as e:
            # A generation failure is a data point, not a crash: scored as an empty prediction and logged.
            print(f"\n[error] example {idx}: {e}")
            generated, num_output_tokens, generation_seconds = "", 0, 0.0

        generated = generated.strip()
        pred_json, json_parse_ok = json_safe_parse(generated)

        num_reasoning_tokens = gen_stats.get("num_reasoning_tokens", 0)
        num_answer_tokens = gen_stats.get("num_answer_tokens", num_output_tokens)
        total_output_tokens += num_output_tokens
        total_reasoning_tokens += num_reasoning_tokens
        total_answer_tokens += num_answer_tokens
        total_generation_seconds += generation_seconds

        if reasoning_model:
            reasoning_token_counts.append(num_reasoning_tokens)
            if gen_stats.get("reasoning_skipped", False):
                # Answered directly without ever opening a thought block. NOT a termination failure -- the answer is
                # valid and is scored.
                reasoning_skipped_count += 1
            elif not gen_stats.get("found_reasoning_end", False):
                reasoning_unterminated_count += 1

        pred_spans, match_stats = assign_char_spans_from_context(
            input_text, pred_json, valid_labels, json_parse_ok=json_parse_ok,
            insertion_labels=INSERTION_LABELS,
        )
        error_type = classify_generation_error(
            generated, json_parse_ok, len(pred_json), match_stats,
            hit_token_cap=num_output_tokens >= MAX_NEW_TOKENS,
        )
        error_type_counts[error_type] += 1

        format_invalid_count += match_stats["format_invalid"]
        context_not_in_input_count += match_stats["context_not_in_input"]
        entity_not_in_context_count += match_stats["entity_not_in_context"]
        invalid_label_count += match_stats["invalid_label_count"]
        located_entity_count += match_stats["located_entities"]
        unlocated_entity_count += match_stats["unlocated_entities"]
        proposed_entity_count += match_stats["processed_entities"]
        total_predictions += len(pred_spans)

        pred_chars = spans_to_char_set(pred_spans)
        cp, cr, cf = compute_character_f1(gold_chars, pred_chars)
        char_f1_per_ex.append(cf)
        char_p_per_ex.append(cp)
        char_r_per_ex.append(cr)

        # Semin et al.'s micro counts, hard and soft.
        hard_counts = compute_overlap_counts(pred_spans, gold_spans, hard_matching=True)
        soft_counts = compute_overlap_counts(pred_spans, gold_spans, hard_matching=False)
        hard_overlap += hard_counts["overlap_chars"]
        hard_predicted += hard_counts["predicted_chars"]
        hard_gold += hard_counts["gold_chars"]
        soft_overlap += soft_counts["overlap_chars"]
        soft_predicted += soft_counts["predicted_chars"]
        soft_gold += soft_counts["gold_chars"]

        # Macro-hard F1: this example's own hard-matching F1, averaged over examples at the end -- unlike
        # semin_hard_f1, which pools the counts across the whole run (micro).
        ex_hard = f1_from_counts(
            hard_counts["overlap_chars"], hard_counts["predicted_chars"], hard_counts["gold_chars"]
        )
        hard_f1_per_ex.append(ex_hard["f1"])
        hard_p_per_ex.append(ex_hard["precision"])
        hard_r_per_ex.append(ex_hard["recall"])

        log_jsonl(pred_fh, {
            "key": f"{seed}:{idx}",
            "dataset": "multigec",
            "method": "context_hf",
            "matching": MATCHING,
            "model": model_name,
            "reasoning_enabled": reasoning_model,
            "reasoning_marker_seen": gen_stats.get("reasoning_marker_seen", False),
            "found_reasoning_end": gen_stats.get("found_reasoning_end", False),
            "reasoning_skipped": gen_stats.get("reasoning_skipped", False),
            "repetition_penalty": repetition_penalty,
            "sampling_strategy": sampling_strategy,
            "eval_mode": "context",
            "seed": seed,
            "example_idx": idx,
            "example_id": example["example_id"],
            "input_text": input_text,
            "gold_spans": gold_spans,
            "pred_spans": pred_spans,
            "pred_json": pred_json,
            "json_parse_ok": json_parse_ok,
            "match_stats": match_stats,
            "error_type": error_type,
            "char_precision": cp,
            "char_recall": cr,
            "char_f1": cf,
            "macro_hard_precision": ex_hard["precision"],
            "macro_hard_recall": ex_hard["recall"],
            "macro_hard_f1": ex_hard["f1"],
            "semin_hard_overlap": hard_counts["overlap_chars"],
            "semin_hard_predicted": hard_counts["predicted_chars"],
            "semin_hard_gold": hard_counts["gold_chars"],
            "semin_soft_overlap": soft_counts["overlap_chars"],
            "semin_soft_predicted": soft_counts["predicted_chars"],
            "semin_soft_gold": soft_counts["gold_chars"],
            "raw_output": generated,
            "span_count": len(pred_spans),
            "num_output_tokens": num_output_tokens,
            "num_reasoning_tokens": num_reasoning_tokens,
            "num_answer_tokens": num_answer_tokens,
            "reasoning_text": gen_stats.get("reasoning_text", ""),
            "raw_token_ids": gen_stats.get("raw_token_ids", []),
            "generation_seconds": generation_seconds,
        })

        if (idx + 1) % EVAL_INTERVAL == 0:
            partial_hard = f1_from_counts(hard_overlap, hard_predicted, hard_gold)
            elapsed = (time.time() - start_time) / 60.0
            tqdm.write(
                f"[{model_name} | {sampling_strategy} | context] seed {seed}, {idx + 1}/{len(sampled)} "
                f"charF1={statistics.mean(char_f1_per_ex):.4f} "
                f"macroHardF1={statistics.mean(hard_f1_per_ex):.4f} "
                f"hardF1={partial_hard['f1']:.4f} | "
                f"fmt_invalid={format_invalid_count} span_not_found={unlocated_entity_count} "
                f"bad_label={invalid_label_count} spans={total_predictions} | elapsed={elapsed:.1f}m"
            )

    hard = f1_from_counts(hard_overlap, hard_predicted, hard_gold)
    soft = f1_from_counts(soft_overlap, soft_predicted, soft_gold)

    n_gen = max(len(sampled), 1)
    elapsed_min = (time.time() - start_time) / 60.0
    seed_metrics = {
        "char_f1": statistics.mean(char_f1_per_ex) if char_f1_per_ex else 0.0,
        "char_precision": statistics.mean(char_p_per_ex) if char_p_per_ex else 0.0,
        "char_recall": statistics.mean(char_r_per_ex) if char_r_per_ex else 0.0,
        "macro_hard_f1": statistics.mean(hard_f1_per_ex) if hard_f1_per_ex else 0.0,
        "macro_hard_precision": statistics.mean(hard_p_per_ex) if hard_p_per_ex else 0.0,
        "macro_hard_recall": statistics.mean(hard_r_per_ex) if hard_r_per_ex else 0.0,
        "semin_hard_precision": hard["precision"],
        "semin_hard_recall": hard["recall"],
        "semin_hard_f1": hard["f1"],
        "semin_soft_precision": soft["precision"],
        "semin_soft_recall": soft["recall"],
        "semin_soft_f1": soft["f1"],
        "format_invalid_count": format_invalid_count,
        "format_invalid_rate": format_invalid_count / n_gen,
        "context_not_in_input_count": context_not_in_input_count,
        "context_not_in_input_rate": context_not_in_input_count / max(proposed_entity_count, 1),
        "entity_not_in_context_count": entity_not_in_context_count,
        "entity_not_in_context_rate": entity_not_in_context_count / max(proposed_entity_count, 1),
        "span_not_found_count": unlocated_entity_count,
        "span_not_found_rate": unlocated_entity_count / max(proposed_entity_count, 1),
        "invalid_label_count": invalid_label_count,
        "invalid_label_rate": invalid_label_count / max(located_entity_count, 1),
        "proposed_entity_count": proposed_entity_count,
        "predicted_spans": total_predictions,
        "reasoning_unterminated_count": reasoning_unterminated_count,
        "reasoning_unterminated_rate": reasoning_unterminated_count / n_gen,
        "reasoning_skipped_count": reasoning_skipped_count,
        "reasoning_skipped_rate": reasoning_skipped_count / n_gen,
        "reasoning_tokens_avg": (sum(reasoning_token_counts) / len(reasoning_token_counts))
                                if reasoning_token_counts else 0.0,
        "output_tokens_avg": total_output_tokens / n_gen,
        "answer_tokens_avg": total_answer_tokens / n_gen,
        "total_output_tokens": total_output_tokens,
        "total_reasoning_tokens": total_reasoning_tokens,
        "total_answer_tokens": total_answer_tokens,
        "generation_seconds_avg": total_generation_seconds / n_gen,
        "tokens_per_second": total_output_tokens / max(total_generation_seconds, 1e-9),
        "elapsed_minute": elapsed_min,
    }
    # One rate per Semin error type, always all of them, so the CSV columns are stable across runs even when a given
    # failure never fires.
    for name in ERROR_TYPES:
        seed_metrics[f"err_{name}_rate"] = error_type_counts[name] / n_gen
    exp_metrics.append(seed_metrics)

pred_fh.close()


def agg(key):
    return mean_std([m[key] for m in exp_metrics])


char_f1_mean, char_f1_std = agg("char_f1")
char_p_mean, char_p_std = agg("char_precision")
char_r_mean, char_r_std = agg("char_recall")
mhard_p_mean, mhard_p_std = agg("macro_hard_precision")
mhard_r_mean, mhard_r_std = agg("macro_hard_recall")
mhard_f1_mean, mhard_f1_std = agg("macro_hard_f1")
hard_p_mean, hard_p_std = agg("semin_hard_precision")
hard_r_mean, hard_r_std = agg("semin_hard_recall")
hard_f1_mean, hard_f1_std = agg("semin_hard_f1")
soft_p_mean, soft_p_std = agg("semin_soft_precision")
soft_r_mean, soft_r_std = agg("semin_soft_recall")
soft_f1_mean, soft_f1_std = agg("semin_soft_f1")
fmt_mean, fmt_std = agg("format_invalid_count")
fmt_rate_mean, fmt_rate_std = agg("format_invalid_rate")
ctx_mean, ctx_std = agg("context_not_in_input_count")
ctx_rate_mean, ctx_rate_std = agg("context_not_in_input_rate")
ent_mean, ent_std = agg("entity_not_in_context_count")
ent_rate_mean, ent_rate_std = agg("entity_not_in_context_rate")
snf_mean, snf_std = agg("span_not_found_count")
snf_rate_mean, snf_rate_std = agg("span_not_found_rate")
lab_mean, lab_std = agg("invalid_label_count")
lab_rate_mean, lab_rate_std = agg("invalid_label_rate")
prop_mean, prop_std = agg("proposed_entity_count")
ps_mean, ps_std = agg("predicted_spans")
ru_mean, ru_std = agg("reasoning_unterminated_count")
ru_rate_mean, ru_rate_std = agg("reasoning_unterminated_rate")
rs_mean, rs_std = agg("reasoning_skipped_count")
rs_rate_mean, rs_rate_std = agg("reasoning_skipped_rate")
rt_mean, rt_std = agg("reasoning_tokens_avg")
ot_mean, ot_std = agg("output_tokens_avg")
at_mean, at_std = agg("answer_tokens_avg")
gs_mean, gs_std = agg("generation_seconds_avg")
tps_mean, tps_std = agg("tokens_per_second")
elapsed_mean, elapsed_std = agg("elapsed_minute")

row = {
    "model": model_name,
    "dataset": "multigec",
    "method": "context_hf",
    "matching": MATCHING,
    "reasoning_enabled": reasoning_model,
    "reasoning_effort": args.reasoning_effort if args.reasoning_effort else "n|a",
    "repetition_penalty": repetition_penalty,
    "sampling_strategy": sampling_strategy,
    "do_sample": DO_SAMPLE,
    "eval_mode": "context",
    "batch_size": BATCH_SIZE,
    "max_examples": MAX_EXAMPLES,
    "max_new_tokens": MAX_NEW_TOKENS,
    "seeds": ",".join(str(s) for s in SEEDS),
    "n_iters": len(SEEDS),
    "char_f1_report": format_pm(to_pct(char_f1_mean), to_pct(char_f1_std)),
    "char_precision_report": format_pm(to_pct(char_p_mean), to_pct(char_p_std)),
    "char_recall_report": format_pm(to_pct(char_r_mean), to_pct(char_r_std)),
    "macro_hard_precision_report": format_pm(to_pct(mhard_p_mean), to_pct(mhard_p_std)),
    "macro_hard_recall_report": format_pm(to_pct(mhard_r_mean), to_pct(mhard_r_std)),
    "macro_hard_f1_report": format_pm(to_pct(mhard_f1_mean), to_pct(mhard_f1_std)),
    "semin_hard_precision_report": format_pm(to_pct(hard_p_mean), to_pct(hard_p_std)),
    "semin_hard_recall_report": format_pm(to_pct(hard_r_mean), to_pct(hard_r_std)),
    "semin_hard_f1_report": format_pm(to_pct(hard_f1_mean), to_pct(hard_f1_std)),
    "semin_soft_precision_report": format_pm(to_pct(soft_p_mean), to_pct(soft_p_std)),
    "semin_soft_recall_report": format_pm(to_pct(soft_r_mean), to_pct(soft_r_std)),
    "semin_soft_f1_report": format_pm(to_pct(soft_f1_mean), to_pct(soft_f1_std)),
    "format_invalid_count_avg": round(fmt_mean, 3),
    "format_invalid_count_std": round(fmt_std, 3),
    "format_invalid_rate_report": format_pm(to_pct(fmt_rate_mean), to_pct(fmt_rate_std)),
    "context_not_in_input_avg": round(ctx_mean, 3),
    "context_not_in_input_std": round(ctx_std, 3),
    "context_not_in_input_rate_report": format_pm(to_pct(ctx_rate_mean), to_pct(ctx_rate_std)),
    "entity_not_in_context_avg": round(ent_mean, 3),
    "entity_not_in_context_std": round(ent_std, 3),
    "entity_not_in_context_rate_report": format_pm(to_pct(ent_rate_mean), to_pct(ent_rate_std)),
    "span_not_found_avg": round(snf_mean, 3),
    "span_not_found_std": round(snf_std, 3),
    "span_not_found_rate_report": format_pm(to_pct(snf_rate_mean), to_pct(snf_rate_std)),
    "invalid_label_avg": round(lab_mean, 3),
    "invalid_label_std": round(lab_std, 3),
    "invalid_label_rate_report": format_pm(to_pct(lab_rate_mean), to_pct(lab_rate_std)),
    "proposed_entity_count_avg": round(prop_mean, 3),
    "proposed_entity_count_std": round(prop_std, 3),
    "predicted_spans_avg": round(ps_mean, 3),
    "predicted_spans_std": round(ps_std, 3),
    "reasoning_unterminated_count_avg": round(ru_mean, 3),
    "reasoning_unterminated_count_std": round(ru_std, 3),
    "reasoning_unterminated_rate_report": format_pm(to_pct(ru_rate_mean), to_pct(ru_rate_std)),
    "reasoning_skipped_count_avg": round(rs_mean, 3),
    "reasoning_skipped_rate_report": format_pm(to_pct(rs_rate_mean), to_pct(rs_rate_std)),
    "reasoning_tokens_avg_avg": round(rt_mean, 3),
    "reasoning_tokens_avg_std": round(rt_std, 3),
    "output_tokens_avg_avg": round(ot_mean, 3),
    "output_tokens_avg_std": round(ot_std, 3),
    "answer_tokens_avg_avg": round(at_mean, 3),
    "answer_tokens_avg_std": round(at_std, 3),
    "generation_seconds_avg_avg": round(gs_mean, 3),
    "generation_seconds_avg_std": round(gs_std, 3),
    "tokens_per_second_avg": round(tps_mean, 3),
    "tokens_per_second_std": round(tps_std, 3),
    "elapsed_minute_avg": round(elapsed_mean, 3),
    "elapsed_minute_std": round(elapsed_std, 3),
}
for name in ERROR_TYPES:
    m, s = agg(f"err_{name}_rate")
    row[f"err_{name}_rate_report"] = format_pm(to_pct(m), to_pct(s))
results.append(row)

results_df = pd.DataFrame(results)
results_path = (f"{RESULTS_ROOT}/GEC/Context-Based/Csv/"
                f"multigec{shard_suffix()}_{model_name.split('/')[-1]}_{BATCH_SIZE}_BS_{config_tag()}_{sampling_strategy}.csv")
results_txt_path = results_path.replace("Csv", "Txt").replace(".csv", ".txt")

os.makedirs(os.path.dirname(results_path), exist_ok=True)
os.makedirs(os.path.dirname(results_txt_path), exist_ok=True)

results_df.to_csv(results_path, index=False)
with open(results_txt_path, "w") as f:
    f.write(results_df.to_string(index=False))

print(f"\nResults saved to {results_path} and {results_txt_path}")
