"""Constrained generation evaluation on MultiGEC (W&I 2024 English dev).

The model is shown the ORIGINAL learner text verbatim -- newlines and repeated
spaces included -- so gold character offsets from the dataset are used exactly
as they come, with no remapping.

Zero-length spans -- what makes this task different
---------------------------------------------------
`M` ("a word is missing here") is an INSERTION POINT: `start == end`, wrapping
no characters. 1,599 of the 7,100 gold spans (22.5%) are like this, and `M` is
the only label that is ever empty (verified on all 7,100). Two consequences:

1. The constrained processor is built to forbid empty spans -- that is its
   anti-loop invariant. It is relaxed here, and ONLY here, by passing
   `allow_empty_span_labels={"M"}`, which permits an empty close for `M` while
   still requiring the cursor to advance between two consecutive empty spans.
   Every other task passes nothing and keeps the strict non-empty rule.
2. The character sets fed to `char_f1` are built with `spans_to_char_set`, NOT
   with a bare `range(start, end)`. The latter is empty for a zero-length span,
   which would silently drop 22.5% of this gold from the metric. The helper
   gives each insertion point one sentinel element instead, weighing 1.

Metrics, reported side by side
------------------------------
- `char_f1`: character-level F1 averaged over examples (the thesis metric).
  Counts insertion points, but is LABEL-AGNOSTIC, so it is the macro analogue
  of Semin's soft F1 rather than of their hard F1.
- `semin_hard_f1` / `semin_soft_f1`: Semin et al.'s pooled character-overlap F1
  from their `metrics.py` (micro: counts summed over the run, F1 once at the
  end). Hard requires the label to match, soft does not -- with three labels
  here, the two genuinely diverge.
- `macro_hard_f1`: per-example hard F1, then averaged. Same aggregation as
  `char_f1` but label-sensitive, from Semin's counts. Reading it next to
  `char_f1` separates "wrong position" from "right position, wrong label".
"""
import argparse
import os
import random
import statistics
import sys
import time

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from utils.utils_functions import (
    generate_markup, validate_reconstruction,
    parse_spans_from_tagged_output,
    mean_std, to_pct, format_pm, compute_character_f1, spans_to_char_set,
    open_jsonl_writer, log_jsonl,
)
from utils.span_datasets import load_multigec, MULTIGEC_LABELS
from utils.semin_metrics import compute_overlap_counts, f1_from_counts
from utils.TokTrie import build_toktrie_from_tokenizer
from utils.TrieSpanConstrainedProcessorTokenAware import TrieSpanConstrainedProcessorTokenAware
from utils.TrieSpanConstrainedProcessor import TrieSpanConstrainedProcessor
from utils.system_prompts import SYSTEM_PROMPT_CONSTR_GEN_MULTIGEC
from utils.model_reasoning_utils import reasoning_ended
from utils.model_registry import get, resolve_sampling

# -------------------------
# Model configuration
# -------------------------
parser = argparse.ArgumentParser("Evaluate constrained generation on MultiGEC.")
parser.add_argument("--model", required=True, type=str, help="Model name or ID to evaluate.")
parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True, help="Whether to enable reasoning in the model's prompt.")
parser.add_argument("--reasoning-effort", choices=['low', 'medium', 'high', 'xhigh'], default=None,
                    help="Reasoning effort. The union of both families' levels: Harmony/GPT-OSS takes "
                         "low|medium|high, Qwen3.8 takes low|medium|xhigh (xhigh is its default). "
                         "Which are actually valid is checked per model against the registry.")
parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty for the model.")
parser.add_argument("--temperature", type=float, default=None, help="Temperature for sampling.")
parser.add_argument("--top-p", type=float, default=None, help="Top-p for sampling.")
parser.add_argument("--top-k", type=int, default=None, help="Top-k for sampling.")
parser.add_argument("--min-p", type=float, default=None, help="Minimum probability for sampling.")
parser.add_argument("--max-examples", type=int, default=None, help="Maximum number of examples to evaluate. None means all examples.")
parser.add_argument("--max-new-tokens", type=int, default=16384, help="Maximum number of new tokens to generate.")
parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Seeds to run and average over.")
parser.add_argument("--shard", type=str, default=None,
                    help="Optional: i/N - shard the dataset into N shards and run only shard i.")

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
    torch_dtype='auto',
)

# Create a token trie from the tokenizer for constrained generation
toktrie = build_toktrie_from_tokenizer(tokenizer)

reasoning_end_marker = tokenizer(spec.reasoning_end_marker, add_special_tokens=False).input_ids if spec.reasoning else None
# Only set for families where the MODEL emits the opener (Gemma-4). None elsewhere,
# which means the block is open by construction -- see model_registry.ModelSpec.
reasoning_start_marker = tokenizer(spec.reasoning_start_marker, add_special_tokens=False).input_ids if spec.reasoning_start_marker else None
sampling = resolve_sampling(spec, reasoning_model,
                            temperature=args.temperature, top_p=args.top_p,
                            top_k=args.top_k, min_p=args.min_p)
DO_SAMPLE = sampling.pop("do_sample", False)

SEEDS = args.seeds
MAX_EXAMPLES = args.max_examples
MAX_NEW_TOKENS = args.max_new_tokens
repetition_penalty = args.repetition_penalty

# One learner text per prompt for now, may experiment with batching later.
BATCH_SIZE = 1
EVAL_INTERVAL = 20

EVAL_MODES = ["unconstrained", "constrained"]

RESULTS_ROOT = "/home/stulcrad/master_thesis/Experiment_results_publication"

PROCESSOR_CLASSES = ["token_aware"] # Use only token_aware for publication runs, since it is the better-performing one.

labels_for_constrained = MULTIGEC_LABELS
# The M label is a zero-length insertion point, so its span closes with no copied
# text. This is the ONLY task that relaxes the processor's non-empty rule.
FORCE_EMPTY_SPAN_LABELS = {"M"}

# -------------------------
# Load dataset
# -------------------------
print("Loading MultiGEC (W&I 2024 English dev)...")
raw = load_multigec()
print(f"Examples: {len(raw)}")
print(f"Max examples per iteration: {MAX_EXAMPLES}")

results = []

sampling_strategy = "sampling" if DO_SAMPLE else "greedy"

# --shard i/N
shard_i, shard_N = None, None
if args.shard is not None:
    try:
        shard_i, shard_N = map(int, args.shard.split("/"))
        if not (0 <= shard_i < shard_N):
            raise ValueError
    except ValueError:
        raise SystemExit(f"Invalid --shard value: {args.shard}. Must be in the form i/N with 0 <= i < N.")
    print(f"Sharded dataset: {len(raw)} examples (shard {shard_i}/{shard_N})")

def shard_suffix():
    return f"_shard{shard_i+1}of{shard_N}" if shard_i is not None else ""

def config_tag():
    parts = [f"think{int(args.enable_thinking)}"]
    if args.reasoning_effort: parts.append(f"effort_{args.reasoning_effort}")
    if args.repetition_penalty: parts.append(f"rep_{args.repetition_penalty}")
    if args.temperature is not None: parts.append(f"temp_{args.temperature}")
    return "_".join(parts)

for eval_mode in EVAL_MODES:
    processor_class_options = PROCESSOR_CLASSES if eval_mode == "constrained" else [None]

    for processor_class in processor_class_options:
        exp_metrics = []
        config_label = processor_class if processor_class is not None else "n|a"
        print(
            f"\nEvaluating model={model_name}, dataset=multigec, shard={shard_suffix()}, "
            f"reasoning_enabled={reasoning_model}, reasoning_effort={args.reasoning_effort}, "
            f"repetition_penalty={repetition_penalty}, sampling_strategy={sampling_strategy}, "
            f"mode={eval_mode}, processor_class={config_label}, max_examples={MAX_EXAMPLES}, seeds={SEEDS}"
        )

        model_short = model_name.split("/")[-1]
        pred_fh = open_jsonl_writer(
            f"{RESULTS_ROOT}/GEC/Constrained-Gen/Predictions/"
            f"multigec{shard_suffix()}_{model_short}_think_{args.enable_thinking}_{sampling_strategy}_{eval_mode}_{config_tag()}_{config_label}.jsonl"
        )

        for seed in SEEDS:
            set_seed(seed)

            if MAX_EXAMPLES is None or MAX_EXAMPLES >= len(raw):
                sampled = raw
            else:
                sampled = random.Random(seed).sample(raw, MAX_EXAMPLES)

            if shard_i is not None and shard_N is not None:
                total_examples = len(sampled)
                shard_size = (total_examples + shard_N - 1) // shard_N
                start_idx = shard_i * shard_size
                end_idx = min(start_idx + shard_size, total_examples)
                sampled = sampled[start_idx:end_idx]
                print(f"Sharding: {total_examples} examples -> {shard_N} shards of "
                      f"~{shard_size}, running shard {shard_i+1}/{shard_N} with {len(sampled)} examples.")

            start_time = time.time()
            wrong_text_count = 0
            reasoning_unterminated_count = 0
            reasoning_skipped_count = 0
            reasoning_token_counts = []
            total_predictions = 0
            empty_span_predictions = 0
            char_f1_per_ex = []
            char_p_per_ex = []
            char_r_per_ex = []
            # Macro-hard: per-example F1 then averaged, but from Semin's counts, so
            # unlike char_f1 it sees zero-length spans and labels.
            hard_f1_per_ex = []
            hard_p_per_ex = []
            hard_r_per_ex = []
            hard_overlap = hard_predicted = hard_gold = 0
            soft_overlap = soft_predicted = soft_gold = 0
                

            for idx in tqdm(range(len(sampled)), desc=f"seed {seed}", file=sys.stdout):
                example = sampled[idx]
                # The ORIGINAL learner text, copied verbatim -- no whitespace rebuild.
                input_text = example["text"]
                gold_spans = example["gold_spans"]
                # spans_to_char_set, not range(): zero-length M spans must count.
                gold_chars = spans_to_char_set(gold_spans)

                if not input_text.strip():
                    cp, cr, cf = compute_character_f1(gold_chars, set())
                    char_f1_per_ex.append(cf)
                    char_p_per_ex.append(cp)
                    char_r_per_ex.append(cr)
                    hf1 = f1_from_counts(0, 0, sum(max(1, s["end"] - s["start"]) for s in gold_spans))
                    hard_f1_per_ex.append(hf1["f1"])
                    hard_p_per_ex.append(hf1["precision"])
                    hard_r_per_ex.append(hf1["recall"])
                    continue

                processor = None
                if eval_mode == "constrained":
                    if processor_class == "token_aware":
                        processor = TrieSpanConstrainedProcessorTokenAware(
                            labels_for_constrained,
                            input_text,
                            tokenizer,
                            toktrie,
                            reasoning_model=reasoning_model,
                            reasoning_end_marker=reasoning_end_marker,
                            reasoning_ended=reasoning_ended,
                            model_eos_token_id=model.generation_config.eos_token_id,
                            tokenizer_eos_token_id=tokenizer.eos_token_id,
                            # The one task that needs empty spans -- see module docstring.
                            force_empty_span_labels=FORCE_EMPTY_SPAN_LABELS
                        )
                    else:
                        processor = TrieSpanConstrainedProcessor(
                            labels_for_constrained,
                            input_text,
                            tokenizer,
                            toktrie,
                            reasoning_model=reasoning_model,
                            reasoning_end_marker=reasoning_end_marker,
                            reasoning_ended=reasoning_ended,
                            model_eos_token_id=model.generation_config.eos_token_id,
                            tokenizer_eos_token_id=tokenizer.eos_token_id,
                            # The one task that needs empty spans -- see module docstring.
                            force_empty_span_labels=FORCE_EMPTY_SPAN_LABELS
                        )

                gen_stats = {}
                generated, num_output_tokens, generation_seconds = generate_markup(
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    eval_model=eval_mode,
                    input_text=input_text,
                    system_prompt=SYSTEM_PROMPT_CONSTR_GEN_MULTIGEC,
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
                # Reasoning/answer token split (0 / total for non-reasoning models).
                num_reasoning_tokens = gen_stats.get("num_reasoning_tokens", 0)
                num_answer_tokens = gen_stats.get("num_answer_tokens", num_output_tokens)

                if reasoning_model:
                    reasoning_token_counts.append(num_reasoning_tokens)
                    if gen_stats.get("reasoning_skipped", False):
                        # Answered directly without ever opening a thought block. NOT a
                        # termination failure -- the answer is valid and is scored.
                        reasoning_skipped_count += 1
                    elif not gen_stats.get("found_reasoning_end", False):
                        reasoning_unterminated_count += 1

                parsed = parse_spans_from_tagged_output(generated, set(labels_for_constrained))
                total_predictions += parsed["span_count"]
                exact_copy_ok = validate_reconstruction(parsed["reconstructed_text"], input_text)

                if not exact_copy_ok:
                    wrong_text_count += 1
                    if eval_mode == "constrained":
                        print(f"\n\n===== Warning at seed {seed}, example {idx+1} =====")
                        print(f"Original:      {input_text[:120]!r}")
                        print(f"Reconstructed: {parsed['reconstructed_text'][:120]!r}")
                    # Reconstruction failed: the predicted offsets index the model's
                    # own text, not the input, so nothing can be credited.
                    pred_spans = []
                else:
                    pred_spans = [
                        {"start": e["start"], "end": e["end"], "label": e["label"]}
                        for e in parsed["entities"] if e["label"] in set(labels_for_constrained)
                    ]

                # How many predictions were insertion points. Tracked because it is the
                # behaviour this task's processor change exists to enable: if this stays
                # at 0 the model never used the empty-tag convention, which is a prompt
                # failure, not a constraint failure, and the two look identical in F1.
                empty_span_predictions += sum(1 for s in pred_spans if s["start"] == s["end"])

                # char_f1 -- label-agnostic, but insertion points DO count (module docstring).
                pred_chars = spans_to_char_set(pred_spans)
                cp, cr, cf = compute_character_f1(gold_chars, pred_chars)
                char_f1_per_ex.append(cf)
                char_p_per_ex.append(cp)
                char_r_per_ex.append(cr)

                # Semin et al.'s per-example overlap counts, hard and soft.
                hard_counts = compute_overlap_counts(pred_spans, gold_spans, hard_matching=True)
                soft_counts = compute_overlap_counts(pred_spans, gold_spans, hard_matching=False)
                hard_overlap += hard_counts["overlap_chars"]
                hard_predicted += hard_counts["predicted_chars"]
                hard_gold += hard_counts["gold_chars"]
                soft_overlap += soft_counts["overlap_chars"]
                soft_predicted += soft_counts["predicted_chars"]
                soft_gold += soft_counts["gold_chars"]

                ex_hard = f1_from_counts(
                    hard_counts["overlap_chars"], hard_counts["predicted_chars"], hard_counts["gold_chars"]
                )
                hard_f1_per_ex.append(ex_hard["f1"])
                hard_p_per_ex.append(ex_hard["precision"])
                hard_r_per_ex.append(ex_hard["recall"])

                log_jsonl(pred_fh, {
                    "key": f"{seed}:{idx}",
                    "dataset": "multigec",
                    "method": "constrained_gen",
                    "model": model_name,
                    "reasoning_enabled": reasoning_model,
                    "reasoning_marker_seen": gen_stats.get("reasoning_marker_seen", False),
                    "found_reasoning_end": gen_stats.get("found_reasoning_end", False),
                    "reasoning_skipped": gen_stats.get("reasoning_skipped", False),
                    "repetition_penalty": repetition_penalty,
                    "sampling_strategy": sampling_strategy,
                    "eval_mode": eval_mode,
                    "processor_class": config_label,
                    "seed": seed,
                    "example_idx": idx,
                    "example_id": example["example_id"],
                    "input_text": input_text,
                    "gold_spans": gold_spans,
                    "pred_spans": pred_spans,
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
                    "gold_empty_spans": sum(1 for s in gold_spans if s["start"] == s["end"]),
                    "pred_empty_spans": sum(1 for s in pred_spans if s["start"] == s["end"]),
                    "raw_output": generated,
                    "wrong_text": 0 if exact_copy_ok else 1,
                    "span_count": parsed["span_count"],
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
                        f"[{model_name} | {sampling_strategy} | {eval_mode} | {config_label}] "
                        f"seed {seed}, {idx+1}/{len(sampled)} "
                        f"charF1={statistics.mean(char_f1_per_ex):.4f} "
                        f"macroHardF1={statistics.mean(hard_f1_per_ex):.4f} "
                        f"hardF1={partial_hard['f1']:.4f} | "
                        f"wrong_text={wrong_text_count} spans={total_predictions} "
                        f"empty={empty_span_predictions} | "
                        f"elapsed={elapsed:.1f}m"
                    )

            # Compute micro-averaged Semin et al.'s metrics
            hard = f1_from_counts(hard_overlap, hard_predicted, hard_gold)
            soft = f1_from_counts(soft_overlap, soft_predicted, soft_gold)

            elapsed_min = (time.time() - start_time) / 60.0
            exp_metrics.append({
                "char_f1":        statistics.mean(char_f1_per_ex) if char_f1_per_ex else 0.0,
                "char_precision": statistics.mean(char_p_per_ex)  if char_p_per_ex  else 0.0,
                "char_recall":    statistics.mean(char_r_per_ex)  if char_r_per_ex  else 0.0,
                "macro_hard_f1":        statistics.mean(hard_f1_per_ex) if hard_f1_per_ex else 0.0,
                "macro_hard_precision": statistics.mean(hard_p_per_ex)  if hard_p_per_ex  else 0.0,
                "macro_hard_recall":    statistics.mean(hard_r_per_ex)  if hard_r_per_ex  else 0.0,
                "semin_hard_precision": hard["precision"],
                "semin_hard_recall": hard["recall"],
                "semin_hard_f1": hard["f1"],
                "semin_soft_precision": soft["precision"],
                "semin_soft_recall": soft["recall"],
                "semin_soft_f1": soft["f1"],
                "wrong_text_count": wrong_text_count,
                "wrong_text_rate": wrong_text_count / max(len(sampled), 1),
                "predicted_spans": total_predictions,
                "predicted_empty_spans": empty_span_predictions,
                "reasoning_unterminated_count": reasoning_unterminated_count,
                "reasoning_unterminated_rate": reasoning_unterminated_count / max(len(sampled), 1),
                "reasoning_skipped_count": reasoning_skipped_count,
                "reasoning_skipped_rate": reasoning_skipped_count / max(len(sampled), 1),
                "reasoning_tokens_avg": (sum(reasoning_token_counts) / len(reasoning_token_counts))
                                        if reasoning_token_counts else 0.0,
                "elapsed_minute": elapsed_min,
            })

        pred_fh.close()

        char_f1_mean, char_f1_std = mean_std([m["char_f1"]        for m in exp_metrics])
        char_p_mean,  char_p_std  = mean_std([m["char_precision"]  for m in exp_metrics])
        char_r_mean,  char_r_std  = mean_std([m["char_recall"]     for m in exp_metrics])
        mhard_p_mean, mhard_p_std = mean_std([m["macro_hard_precision"] for m in exp_metrics])
        mhard_r_mean, mhard_r_std = mean_std([m["macro_hard_recall"]    for m in exp_metrics])
        mhard_f1_mean, mhard_f1_std = mean_std([m["macro_hard_f1"]      for m in exp_metrics])
        hard_p_mean,  hard_p_std  = mean_std([m["semin_hard_precision"] for m in exp_metrics])
        hard_r_mean,  hard_r_std  = mean_std([m["semin_hard_recall"]    for m in exp_metrics])
        hard_f1_mean, hard_f1_std = mean_std([m["semin_hard_f1"]        for m in exp_metrics])
        soft_p_mean,  soft_p_std  = mean_std([m["semin_soft_precision"] for m in exp_metrics])
        soft_r_mean,  soft_r_std  = mean_std([m["semin_soft_recall"]    for m in exp_metrics])
        soft_f1_mean, soft_f1_std = mean_std([m["semin_soft_f1"]        for m in exp_metrics])
        wt_mean,      wt_std      = mean_std([m["wrong_text_count"] for m in exp_metrics])
        wt_rate_mean, wt_rate_std = mean_std([m["wrong_text_rate"]  for m in exp_metrics])
        ps_mean,      ps_std      = mean_std([m["predicted_spans"] for m in exp_metrics])
        pes_mean,     pes_std     = mean_std([m["predicted_empty_spans"] for m in exp_metrics])
        ru_mean,      ru_std      = mean_std([m["reasoning_unterminated_count"] for m in exp_metrics])
        ru_rate_mean, ru_rate_std = mean_std([m["reasoning_unterminated_rate"]  for m in exp_metrics])
        rs_mean,      rs_std      = mean_std([m["reasoning_skipped_count"] for m in exp_metrics])
        rs_rate_mean, rs_rate_std = mean_std([m["reasoning_skipped_rate"]  for m in exp_metrics])
        rt_mean,      rt_std      = mean_std([m["reasoning_tokens_avg"] for m in exp_metrics])
        elapsed_mean, elapsed_std = mean_std([m["elapsed_minute"]   for m in exp_metrics])

        results.append({
            "model":              model_name,
            "dataset":            "multigec",
            "reasoning_enabled":  reasoning_model,
            "reasoning_effort":   args.reasoning_effort if args.reasoning_effort else "n|a",
            "repetition_penalty": repetition_penalty,
            "sampling_strategy":  sampling_strategy,
            "do_sample":          DO_SAMPLE,
            "eval_mode":          eval_mode,
            "processor_class":    config_label,
            "batch_size":         BATCH_SIZE,
            "max_examples":       MAX_EXAMPLES,
            "max_new_tokens":     MAX_NEW_TOKENS,
            "seeds":              ",".join(str(s) for s in SEEDS),
            "n_iters":            len(SEEDS),
            "char_f1_report":           format_pm(to_pct(char_f1_mean), to_pct(char_f1_std)),
            "char_precision_report":    format_pm(to_pct(char_p_mean), to_pct(char_p_std)),
            "char_recall_report":       format_pm(to_pct(char_r_mean), to_pct(char_r_std)),
            "macro_hard_precision_report": format_pm(to_pct(mhard_p_mean), to_pct(mhard_p_std)),
            "macro_hard_recall_report":    format_pm(to_pct(mhard_r_mean), to_pct(mhard_r_std)),
            "macro_hard_f1_report":        format_pm(to_pct(mhard_f1_mean), to_pct(mhard_f1_std)),
            "semin_hard_precision_report": format_pm(to_pct(hard_p_mean), to_pct(hard_p_std)),
            "semin_hard_recall_report":    format_pm(to_pct(hard_r_mean), to_pct(hard_r_std)),
            "semin_hard_f1_report":        format_pm(to_pct(hard_f1_mean), to_pct(hard_f1_std)),
            "semin_soft_precision_report": format_pm(to_pct(soft_p_mean), to_pct(soft_p_std)),
            "semin_soft_recall_report":    format_pm(to_pct(soft_r_mean), to_pct(soft_r_std)),
            "semin_soft_f1_report":        format_pm(to_pct(soft_f1_mean), to_pct(soft_f1_std)),
            "wrong_text_count_avg":     round(wt_mean,  3),
            "wrong_text_count_std":     round(wt_std,   3),
            "wrong_text_rate_report":   format_pm(to_pct(wt_rate_mean), to_pct(wt_rate_std)),
            "predicted_spans_avg": round(ps_mean, 3),
            "predicted_spans_std": round(ps_std,  3),
            "predicted_empty_spans_avg": round(pes_mean, 3),
            "predicted_empty_spans_std": round(pes_std,  3),
            "reasoning_unterminated_count_avg": round(ru_mean, 3),
            "reasoning_unterminated_count_std": round(ru_std, 3),
            "reasoning_unterminated_rate_report": format_pm(to_pct(ru_rate_mean), to_pct(ru_rate_std)),
            "reasoning_skipped_count_avg": round(rs_mean, 3),
            "reasoning_skipped_rate_report": format_pm(to_pct(rs_rate_mean), to_pct(rs_rate_std)),
            "reasoning_tokens_avg_avg": round(rt_mean, 3),
            "reasoning_tokens_avg_std": round(rt_std, 3),
            "elapsed_minute_avg": round(elapsed_mean, 3),
            "elapsed_minute_std": round(elapsed_std,  3),
        })

# Save intermediate results to CSV after each model evaluation to avoid data loss in case of interruptions
intermediate_results_df = pd.DataFrame(results)
intermediate_results_path = f"{RESULTS_ROOT}/GEC/Constrained-Gen/Csv/multigec{shard_suffix()}_{model_name.split('/')[-1]}_{BATCH_SIZE}_BS_{config_tag()}_{sampling_strategy}.csv"
intermediate_results_txt_path = intermediate_results_path.replace("Csv", "Txt").replace(".csv", ".txt")

os.makedirs(os.path.dirname(intermediate_results_path), exist_ok=True)
os.makedirs(os.path.dirname(intermediate_results_txt_path), exist_ok=True)

intermediate_results_df.to_csv(intermediate_results_path, index=False)

with open(intermediate_results_txt_path, "w") as f:
    f.write(intermediate_results_df.to_string(index=False))

print(f"\nIntermediate results saved to {intermediate_results_path} and {intermediate_results_txt_path}")
