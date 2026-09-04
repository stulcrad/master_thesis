"""
Context-based (prompting) baseline for NER, on CoNLL-2003 or UniversalNER.

This is the BASELINE arm for the constrained-generation paper, and it is a mirror image of `evaluationNER_cons_gen.py`:
same model registry, same reasoning handling, same seeds, same sampling, same sharding/subsetting, same dataset objects,
same metrics, same CSV/JSONL shape. The only thing that differs is how the model is asked to produce spans -- a JSON
array of {entity, label, context} objects instead of inline markup over a verbatim copy of the input -- and therefore
what can go wrong.

Metrics
-------
Everything `evaluationNER_cons_gen.py` reports, so the columns pair:

- seqeval strict IOB2 entity P/R/F1/accuracy;
- Semin et al.'s pooled (micro) character-overlap F1, hard and soft, from their metrics.py;
- token budget and throughput: total / reasoning / answer tokens, generation seconds, tokens per second;
- reasoning termination counters.

Plus the failure modes that only exist on this side of the comparison, which is the reason this arm is run at all:

- `span_not_found` / `context_not_in_input` / `entity_not_in_context` -- the model named a span that is not in the
  input, or named a context snippet that is not. STRUCTURALLY IMPOSSIBLE under constrained generation.
- `invalid_label` -- the model used a label outside the tagset. Also structurally impossible under constrained
  generation.
- `format_invalid` -- the response did not parse as a JSON array. The counterpart of the constrained arm's wrong-text
  rate, and a weaker test: it asks only that the output be parseable, not that it reproduce the input.

UNER aggregation is unchanged from the constrained script: `--subset` always resolves to named treebanks, each with its
own JSONL and CSV row, and the headline is the MACRO average over the 18 -- a downstream step over the CSVs, not
something computed here.
"""
import argparse
import os
import random
import sys
import time
from collections import Counter

import evaluate
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from utils.utils_functions import (
    generate_markup, spans_to_bio_tags,
    mean_std, to_pct, format_pm,
    open_jsonl_writer, log_jsonl,
)
from utils.span_datasets import load_ner_dataset, ner_dataset_info, ner_subsets, bio_tags_to_char_spans
from utils.semin_metrics import compute_overlap_counts, f1_from_counts
from utils.context_matching_utils import (
    json_safe_parse, assign_char_spans_from_context, classify_generation_error, ERROR_TYPES,
)
from utils.system_prompts import SYSTEM_PROMPT_CONTEXT_BASE_CONLL, SYSTEM_PROMPT_CONTEXT_BASE_UNER
from utils.model_registry import get, resolve_sampling

# -------------------------
# Model configuration
# -------------------------
parser = argparse.ArgumentParser("Evaluate the context-based prompting baseline for NER.")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation.")
parser.add_argument("--model", required=True, type=str, help="Model name or ID to evaluate.")
parser.add_argument("--dataset", choices=["conll2003", "uner"], default="conll2003", help="Which NER benchmark to evaluate on.")
parser.add_argument("--subset", type=str, nargs="+", default=["all"],
                    help="Which named subset(s) of --dataset to run, or 'all'. For --dataset uner: a "
                         "treebank (e.g. en_ewt) -- 18 total. SEVERAL SUBSETS RUN IN ONE JOB, reusing the "
                         "loaded model -- the cluster caps submitted jobs at 200. Each subset still gets "
                         "its own JSONL and its own CSV row: for UNER, average the rows for the macro figure.")
parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True, help="Whether to enable reasoning in the model's prompt.")
parser.add_argument("--reasoning-effort", choices=['low', 'medium', 'high', 'xhigh'], default=None,
                    help="Reasoning effort. The union of both families' levels: Harmony/GPT-OSS takes "
                         "low|medium|high, Qwen3.8 takes low|medium|xhigh (xhigh is its default). "
                         "Which are actually valid is checked per model against the registry.")
parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty for the model.")
parser.add_argument("--temperature", type=float, default=None, help="Temperature for sampling.")  # None -> use preset
parser.add_argument("--top-p", type=float, default=None, help="Top-p for sampling.")
parser.add_argument("--top-k", type=int, default=None, help="Top-k for sampling.")
parser.add_argument("--min-p", type=float, default=None, help="Minimum probability for sampling.")
parser.add_argument("--max-examples", type=int, default=None,
                    help="Maximum number of examples to evaluate. None means the full test split.")
parser.add_argument("--max-new-tokens", type=int, default=16384,
                    help="Maximum number of new tokens to generate.")
parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="Seeds to run and average over.")
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
    torch_dtype="auto",
)

batch_size = args.batch_size
print(f"Batch size: {batch_size}")
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

EVAL_INTERVAL = 100

# Exact matching only -- see the module docstring. Kept as a named constant rather than a flag so it lands in the CSV
# and the JSONL and cannot be silently different between two runs.
MATCHING = "exact"

# Results root. Thesis runs live in Experiment_results/ and are FROZEN (cited in the thesis); publication runs go here
# so the two are trivially separable by eye. Overridable so the smoke test can write somewhere harmless; unset, it is
# the publication tree.
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "/home/stulcrad/master_thesis/Experiment_results_publication")

seqeval = evaluate.load("seqeval")

# labels/results_dir depend only on --dataset, not --subset, so this is free -- no data load needed to resolve them.
valid_labels_list, results_dir = ner_dataset_info(args.dataset)
valid_labels = set(valid_labels_list)

subsets_to_run = ner_subsets(args.dataset) if "all" in args.subset else list(args.subset)
print(f"Dataset: {args.dataset}, subsets to run ({len(subsets_to_run)}): {', '.join(subsets_to_run)}")
print(f"labels: {valid_labels_list}")

SYSTEM_PROMPT = SYSTEM_PROMPT_CONTEXT_BASE_UNER if args.dataset == "uner" else SYSTEM_PROMPT_CONTEXT_BASE_CONLL

# --shard i/N
shard_i, shard_N = None, None
if args.shard is not None:
    try:
        shard_i, shard_N = map(int, args.shard.split("/"))
        if not (0 <= shard_i < shard_N):
            raise ValueError
        if args.dataset != "conll2003":
            raise SystemExit(f"--shard is only supported for --dataset conll2003, got {args.dataset!r}")
    except ValueError:
        raise SystemExit(f"--shard must be of the form i/N with 0 <= i < N, got {args.shard!r}")
    print(f"Sharding dataset into {shard_N} shards, running only shard {shard_i}.")

PRED_DIR = f"{RESULTS_ROOT}/{results_dir}/Context-Based/Predictions"

results = []

sampling_strategy = "sampling" if DO_SAMPLE else "greedy"


def config_tag():
    parts = [f"think{int(args.enable_thinking)}"]
    if args.reasoning_effort: parts.append(f"effort_{args.reasoning_effort}")
    if args.repetition_penalty: parts.append(f"rep_{args.repetition_penalty}")
    if args.temperature is not None: parts.append(f"temp_{args.temperature}")
    return "_".join(parts)


def dataset_tag(subset):
    return f"{args.dataset}_{subset}"


def shard_suffix():
    return f"_shard{shard_i + 1}of{shard_N}" if shard_i is not None else ""


def run_tag():
    """Names the CSV for this JOB, which may cover several subsets."""
    if len(subsets_to_run) == 1:
        return f"{args.dataset}_{subsets_to_run[0]}{shard_suffix()}"
    return f"{args.dataset}_{len(subsets_to_run)}subsets_{subsets_to_run[0]}_to_{subsets_to_run[-1]}"


def save_results():
    """Write the CSV/TXT for everything finished so far.

    Called after EVERY subset, not just at the end: one job covers up to 18 (UNER) subsets, so a crash or a wall-clock
    kill partway through would otherwise throw away every completed subset with it.
    """
    df = pd.DataFrame(results)
    csv_path = (f"{RESULTS_ROOT}/{results_dir}/Context-Based/Csv/"
                f"{run_tag()}_{model_name.split('/')[-1]}_{batch_size}_BS_{config_tag()}_{sampling_strategy}.csv")
    txt_path = csv_path.replace("Csv", "Txt").replace(".csv", ".txt")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    with open(txt_path, "w") as f:
        f.write(df.to_string(index=False))
    return csv_path, txt_path


full_time_start = time.time()

for subset in subsets_to_run:
    dataset = load_ner_dataset(args.dataset, subset)[0]
    print(f"\n{'=' * 70}")
    print(f"=== dataset {args.dataset}, subset {subset}: {len(dataset):,} examples, shard {shard_suffix() or 'all'}")
    print(f"{'=' * 70}")

    exp_metrics = []
    print(
        f"\nEvaluating model={model_name}, dataset={dataset_tag(subset)}, shard={shard_suffix()}, "
        f"reasoning_enabled={reasoning_model}, reasoning_effort={args.reasoning_effort}, "
        f"repetition_penalty={repetition_penalty}, sampling_strategy={sampling_strategy}, matching={MATCHING}, "
        f"batch_size={batch_size}, max_examples={MAX_EXAMPLES}, max_new_tokens={MAX_NEW_TOKENS}, seeds={SEEDS}"
    )

    model_short = model_name.split("/")[-1]
    pred_fh = open_jsonl_writer(
        f"{PRED_DIR}/{dataset_tag(subset)}{shard_suffix()}_{model_short}_think_{args.enable_thinking}"
        f"_{sampling_strategy}_context_{MATCHING}_{config_tag()}_bs{batch_size}.jsonl"
    )

    for seed in SEEDS:
        # Without this the vendor presets (do_sample=True) make every run non-reproducible.
        set_seed(seed)

        # Sample BEFORE sharding, so the capped set is identical across arms with different shard counts.
        if MAX_EXAMPLES is None or MAX_EXAMPLES >= len(dataset):
            sampled_dataset = dataset
        else:
            sampled_dataset = random.Random(seed).sample(dataset, MAX_EXAMPLES)

        if shard_i is not None and shard_N is not None:
            total_examples = len(sampled_dataset)
            shard_size = (total_examples + shard_N - 1) // shard_N
            start_idx = shard_i * shard_size
            end_idx = min(start_idx + shard_size, total_examples)
            sampled_dataset = sampled_dataset[start_idx:end_idx]
            print(f"Sharding: {total_examples} examples -> {shard_N} shards of "
                  f"~{shard_size} each, running shard {shard_i}: {len(sampled_dataset)} examples")

        start_time = time.time()
        gold_sequences = []
        pred_sequences = []

        # Matching / format failure counters
        format_invalid_count = 0
        context_not_in_input_count = 0
        entity_not_in_context_count = 0
        invalid_label_count = 0
        located_entity_count = 0
        unlocated_entity_count = 0
        proposed_entity_count = 0
        total_predictions = 0
        unaligned_entity_count = 0
        error_type_counts = Counter()

        # Reasoning / token budget / throughput.
        reasoning_unterminated_count = 0
        reasoning_skipped_count = 0
        reasoning_token_counts = []
        total_output_tokens = 0
        total_reasoning_tokens = 0
        total_answer_tokens = 0
        total_generation_seconds = 0.0

        total_batches = (len(sampled_dataset) + batch_size - 1) // batch_size

        # Semin et al.'s metric is micro: sum these six counters over the run, then call f1_from_counts once at the end.
        hard_overlap = hard_predicted = hard_gold = 0
        soft_overlap = soft_predicted = soft_gold = 0

        for batch_idx in tqdm(range(total_batches), desc=f"seed {seed}", file=sys.stdout):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(sampled_dataset))
            batch = sampled_dataset[start_idx:end_idx]

            batch_tokens = []
            batch_gold_tags = []
            for example in batch:
                batch_tokens.extend(example["tokens"])
                batch_gold_tags.extend(example["tags"])

            input_text = " ".join(batch_tokens)
            # Gold character spans for Semin's metric, derived from the same tags seqeval scores, so the two metrics
            # share one gold.
            batch_gold_spans = bio_tags_to_char_spans(batch_tokens, batch_gold_tags)

            gen_stats = {}
            try:
                generated, num_output_tokens, generation_seconds = generate_markup(
                    model=model,
                    tokenizer=tokenizer,
                    processor=None,
                    eval_model="unconstrained",
                    input_text=input_text,
                    system_prompt=SYSTEM_PROMPT,
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
                # A generation failure is a data point, not a crash: it is scored as an empty prediction and logged.
                print(f"\n[error] batch {batch_idx}: {e}")
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
            )
            error_type = classify_generation_error(
                generated, json_parse_ok, len(pred_json), match_stats,
                hit_token_cap=num_output_tokens >= MAX_NEW_TOKENS,
            )
            error_type_counts[error_type] += 1

            # collect the counts for the run, to be aggregated at the end
            format_invalid_count += match_stats["format_invalid"]
            context_not_in_input_count += match_stats["context_not_in_input"]
            entity_not_in_context_count += match_stats["entity_not_in_context"]
            invalid_label_count += match_stats["invalid_label_count"]
            located_entity_count += match_stats["located_entities"]
            unlocated_entity_count += match_stats["unlocated_entities"]
            proposed_entity_count += match_stats["processed_entities"]
            total_predictions += len(pred_spans)

            # Token-level view for seqeval. The constrained script snaps the same way, from the same helper, so the two
            # F1s are computed by identical code.
            pred_tags, unalign_count = spans_to_bio_tags(
                tokens=batch_tokens,
                entities=pred_spans,
                valid_labels=valid_labels,
            )
            unaligned_entity_count += unalign_count

            gold_sequences.append(batch_gold_tags)
            pred_sequences.append(pred_tags)

            hard_counts = compute_overlap_counts(pred_spans, batch_gold_spans, hard_matching=True)
            soft_counts = compute_overlap_counts(pred_spans, batch_gold_spans, hard_matching=False)
            hard_overlap += hard_counts["overlap_chars"]
            hard_predicted += hard_counts["predicted_chars"]
            hard_gold += hard_counts["gold_chars"]
            soft_overlap += soft_counts["overlap_chars"]
            soft_predicted += soft_counts["predicted_chars"]
            soft_gold += soft_counts["gold_chars"]

            log_jsonl(pred_fh, {
                "key": f"{seed}:{batch_idx}",
                "dataset": dataset_tag(subset),
                "dataset_names": [ex["dataset_name"] for ex in batch],
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
                "batch_size": batch_size,
                "seed": seed,
                "batch_idx": batch_idx,
                "example_ids": [ex["example_id"] for ex in batch],
                "input_text": input_text,
                "gold_tags": batch_gold_tags,
                "pred_tags": pred_tags,
                "gold_spans": batch_gold_spans,
                "pred_spans": pred_spans,
                "pred_json": pred_json,
                "json_parse_ok": json_parse_ok,
                "match_stats": match_stats,
                "error_type": error_type,
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

            if (batch_idx + 1) % EVAL_INTERVAL == 0:
                partial = seqeval.compute(
                    predictions=pred_sequences, references=gold_sequences,
                    scheme="IOB2", mode="strict", zero_division=0,
                )
                partial_hard = f1_from_counts(hard_overlap, hard_predicted, hard_gold)
                elapsed = (time.time() - start_time) / 60.0
                tqdm.write(
                    f"[{model_name} | {dataset_tag(subset)} | {sampling_strategy} | context | bs={batch_size}] "
                    f"seed {seed}, batch {batch_idx + 1}/{total_batches} "
                    f"F1={partial['overall_f1']:.4f}, hardF1={partial_hard['f1']:.4f}, "
                    f"fmt_invalid={format_invalid_count}, ctx_miss={context_not_in_input_count}, "
                    f"ent_miss={entity_not_in_context_count}, bad_label={invalid_label_count}, "
                    f"elapsed={elapsed:.1f}m"
                )

        metrics = seqeval.compute(
            predictions=pred_sequences, references=gold_sequences,
            scheme="IOB2", mode="strict", zero_division=0,
        )
        hard = f1_from_counts(hard_overlap, hard_predicted, hard_gold)
        soft = f1_from_counts(soft_overlap, soft_predicted, soft_gold)

        elapsed_min = (time.time() - start_time) / 60.0
        seed_metrics = {
            "precision": metrics["overall_precision"],
            "recall": metrics["overall_recall"],
            "f1": metrics["overall_f1"],
            "accuracy": metrics["overall_accuracy"],
            "semin_hard_precision": hard["precision"],
            "semin_hard_recall": hard["recall"],
            "semin_hard_f1": hard["f1"],
            "semin_soft_precision": soft["precision"],
            "semin_soft_recall": soft["recall"],
            "semin_soft_f1": soft["f1"],
            "format_invalid_count": format_invalid_count,
            "format_invalid_rate": format_invalid_count / max(total_batches, 1),
            "context_not_in_input_count": context_not_in_input_count,
            "context_not_in_input_rate": context_not_in_input_count / max(proposed_entity_count, 1),
            "entity_not_in_context_count": entity_not_in_context_count,
            "entity_not_in_context_rate": entity_not_in_context_count / max(proposed_entity_count, 1),
            "span_not_found_count": unlocated_entity_count,
            "span_not_found_rate": unlocated_entity_count / max(proposed_entity_count, 1),
            "invalid_label_count": invalid_label_count,
            "invalid_label_rate": invalid_label_count / max(located_entity_count, 1),
            "unaligned_entity_count": unaligned_entity_count,
            "unaligned_entity_rate": unaligned_entity_count / max(total_predictions, 1),
            "proposed_entity_count": proposed_entity_count,
            "predicted_spans": total_predictions,
            "reasoning_unterminated_count": reasoning_unterminated_count,
            "reasoning_unterminated_rate": reasoning_unterminated_count / max(total_batches, 1),
            "reasoning_skipped_count": reasoning_skipped_count,
            "reasoning_skipped_rate": reasoning_skipped_count / max(total_batches, 1),
            "reasoning_tokens_avg": (sum(reasoning_token_counts) / len(reasoning_token_counts))
                                    if reasoning_token_counts else 0.0,
            "output_tokens_avg": total_output_tokens / max(total_batches, 1),
            "answer_tokens_avg": total_answer_tokens / max(total_batches, 1),
            "total_output_tokens": total_output_tokens,
            "total_reasoning_tokens": total_reasoning_tokens,
            "total_answer_tokens": total_answer_tokens,
            "generation_seconds_avg": total_generation_seconds / max(total_batches, 1),
            "tokens_per_second": total_output_tokens / max(total_generation_seconds, 1e-9),
            "elapsed_minute": elapsed_min,
        }
        # One rate per Semin error type, always all of them, so the CSV columns are stable across runs even when a given
        # failure never fires.
        for name in ERROR_TYPES:
            seed_metrics[f"err_{name}_rate"] = error_type_counts[name] / max(total_batches, 1)
        exp_metrics.append(seed_metrics)

    pred_fh.close()

    def agg(key):
        return mean_std([m[key] for m in exp_metrics])

    precision_mean, precision_std = agg("precision")
    recall_mean, recall_std = agg("recall")
    f1_mean, f1_std = agg("f1")
    accuracy_mean, accuracy_std = agg("accuracy")
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
    una_mean, una_std = agg("unaligned_entity_count")
    una_rate_mean, una_rate_std = agg("unaligned_entity_rate")
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
        "dataset": dataset_tag(subset),
        "method": "context_hf",
        "matching": MATCHING,
        "reasoning_enabled": reasoning_model,
        "reasoning_effort": args.reasoning_effort if args.reasoning_effort else "n|a",
        "repetition_penalty": repetition_penalty,
        "sampling_strategy": sampling_strategy,
        "do_sample": DO_SAMPLE,
        "eval_mode": "context",
        "batch_size": batch_size,
        "max_examples": MAX_EXAMPLES,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seeds": ",".join(str(s) for s in SEEDS),
        "n_iters": len(SEEDS),
        "precision_report": format_pm(to_pct(precision_mean), to_pct(precision_std)),
        "recall_report": format_pm(to_pct(recall_mean), to_pct(recall_std)),
        "f1_report": format_pm(to_pct(f1_mean), to_pct(f1_std)),
        "accuracy_report": format_pm(to_pct(accuracy_mean), to_pct(accuracy_std)),
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
        "unaligned_entity_count_avg": round(una_mean, 3),
        "unaligned_entity_count_std": round(una_std, 3),
        "unaligned_entity_rate_report": format_pm(to_pct(una_rate_mean), to_pct(una_rate_std)),
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

    # Persist after each subset so a wall-clock kill keeps completed subsets.
    csv_path, txt_path = save_results()
    print(f"\n[{subset}] results so far saved to {csv_path}")

print(f"\nTotal evaluation time: {(time.time() - full_time_start) / 60:.2f} minutes")

csv_path, txt_path = save_results()
print(f"\nDone: {len(subsets_to_run)} subset(s), {len(results)} result rows")
print(f"Results saved to {csv_path} and {txt_path}")

if args.dataset == "uner":
    print(
        "\n" + "=" * 78 + "\n"
        "REMINDER -- the semin_*_f1 numbers above are PER TREEBANK (this script\n"
        "never pools UNER into one run). Semin et al.'s headline figure is the\n"
        "MACRO average of the 18 per-treebank F1s -- a downstream step over every\n"
        f"uner_*.csv in {RESULTS_ROOT}/{results_dir}/Context-Based/Csv/ for this\n"
        "model/config, not something this script computes.\n"
        + "=" * 78
    )
