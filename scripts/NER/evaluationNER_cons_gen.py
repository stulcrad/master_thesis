"""Constrained generation evaluation for NER, on CoNLL-2003 or UniversalNER.

Semin et al. hard/soft F1   reported next to seqeval, from their metrics.py

Metrics
-------
seqeval strict IOB2 entity F1, as before, and Semin et al.'s pooled
character-overlap F1 (hard and soft) so our rows can sit beside their.

Note on UNER aggregation: with --uner-subset all, the Semin F1 below is pooled
over all 18 treebanks (micro). Their headline number is the MACRO average of
the 18 per-treebank F1s. Each prediction line records `dataset_names` and its
own overlap/predicted/gold counts, so the macro number is computed downstream
from the JSONL without re-running.
"""

import argparse
import os
import random
import sys
import time

import evaluate
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from utils.utils_functions import (
    generate_markup, validate_reconstruction,
    spans_to_bio_tags, parse_spans_from_tagged_output,
    mean_std, to_pct, format_pm,
    open_jsonl_writer, log_jsonl,
)
from utils.span_datasets import load_ner_dataset, bio_tags_to_char_spans
from utils.semin_metrics import compute_overlap_counts, f1_from_counts
from utils.TokTrie import build_toktrie_from_tokenizer
from utils.TrieSpanConstrainedProcessor import TrieSpanConstrainedProcessor
from utils.TrieSpanConstrainedProcessorTokenAware import TrieSpanConstrainedProcessorTokenAware
from utils.system_prompts import SYSTEM_PROMPT_CONSTR_GEN
from utils.model_reasoning_utils import reasoning_ended
from utils.model_registry import get, resolve_sampling

# -------------------------
# Model configuration
# -------------------------
parser = argparse.ArgumentParser("Evaluate constrained generation for NER.")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation.")
parser.add_argument("--model", required=True, type=str, help="Model name or ID to evaluate.")
parser.add_argument("--dataset", choices=["conll2003", "uner"], default="conll2003", help="Which NER benchmark to evaluate on.")
parser.add_argument("--uner-subset", type=str, default="all", help="UniversalNER treebank (e.g. en_ewt) or 'all' for all 18 (7,523 examples).")
parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True, help="Whether to enable reasoning in the model's prompt.")
parser.add_argument("--reasoning-effort", choices=['low', 'medium', 'high'], default=None, help="Reasoning effort level for the model (if it is supported).")
parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty for the model.")
parser.add_argument("--temperature", type=float, default=None, help="Temperature for sampling.") # None -> use preset
parser.add_argument("--top-p", type=float, default=None, help="Top-p for sampling.")
parser.add_argument("--top-k", type=int, default=None, help="Top-k for sampling.")
parser.add_argument("--min-p", type=float, default=None, help="Minimum probability for sampling.")
parser.add_argument("--max-examples", type=int, default=None, help="Maximum number of examples to evaluate. None means all examples, which is what Semin et al. ran.")
parser.add_argument("--max-new-tokens", type=int, default=16384, help="Maximum number of new tokens to generate. Semin et al. used max_completion_tokens=16384 for thinking runs.")
parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Seeds to run and average over. Semin et al. used 42 43 44 45 46.")


# -------------------------
# Parse arguments
# -------------------------
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
    raise SystemExit(f"{args.model} does not support reasoning effort (supports_reasoning_effort=False)")
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
sampling = resolve_sampling(spec, reasoning_model,
                            temperature=args.temperature, top_p=args.top_p,
                            top_k=args.top_k, min_p=args.min_p)
DO_SAMPLE = sampling.pop("do_sample", False)

SEEDS = args.seeds
MAX_EXAMPLES = args.max_examples
MAX_NEW_TOKENS = args.max_new_tokens
repetition_penalty = args.repetition_penalty

EVAL_INTERVAL = 10

# Evaluate both decoding modes in one run.
EVAL_MODES = ["unconstrained", "constrained"]

# Results root. Thesis runs live in Experiment_results/ and are FROZEN (cited in the
# thesis); publication runs go here so the two are trivially separable by eye.
RESULTS_ROOT = "/home/stulcrad/master_thesis/Experiment_results_publication"

# Processor class is only used for constrained mode.
# PROCESSOR_CLASSES = ["whole_sequence", "token_aware"]
PROCESSOR_CLASSES = ["token_aware"] # Use only token_aware for publication runs, since it is the better-performing one.

# Load the seqeval metric for span-level evaluation
seqeval = evaluate.load("seqeval")

# Dataset: a plain list of {"tokens", "tags", "dataset_name", "example_id"} dicts.
dataset, labels_for_constrained, results_dir = load_ner_dataset(args.dataset, args.uner_subset)
print(f"Dataset: {args.dataset}"
      f"{'/' + args.uner_subset if args.dataset == 'uner' else ''}"
      f" -- {len(dataset):,} examples, labels {labels_for_constrained}")

# Per-example predictions (JSONL, one line per generation) -- required for
# paired significance tests and post-hoc metrics without re-running.
PRED_DIR = f"{RESULTS_ROOT}/{results_dir}/Constrained-Gen/Predictions"

results = []

sampling_strategy = "sampling" if DO_SAMPLE else "greedy"

def config_tag():
    parts = [f"think{int(args.enable_thinking)}"]
    if args.reasoning_effort: parts.append(f"effort_{args.reasoning_effort}")
    if args.repetition_penalty: parts.append(f"rep_{args.repetition_penalty}")
    if args.temperature is not None: parts.append(f"temp_{args.temperature}")
    return "_".join(parts)

def dataset_tag():
    return args.dataset if args.dataset != "uner" else f"uner_{args.uner_subset}"

for eval_mode in EVAL_MODES:
    processor_class_options = PROCESSOR_CLASSES if eval_mode == "constrained" else [None]

    for processor_class in processor_class_options:
        exp_metrics = []
        config_label = processor_class if processor_class is not None else "n|a"
        print(
            f"\nEvaluating model={model_name}, dataset={dataset_tag()}, reasoning_enabled={reasoning_model}, reasoning_effort={args.reasoning_effort}, "
            f"repetition_penalty={repetition_penalty}, sampling_strategy={sampling_strategy}, eval_mode={eval_mode}, "
            f"processor_class={config_label}, batch_size={batch_size}, max_examples={MAX_EXAMPLES}, max_new_tokens={MAX_NEW_TOKENS}, seeds={SEEDS}"
        )

        model_short = model_name.split("/")[-1]
        pred_fh = open_jsonl_writer(
            f"{PRED_DIR}/{dataset_tag()}_{model_short}_think_{args.enable_thinking}_{sampling_strategy}_{eval_mode}_{config_tag()}_{config_label}_bs{batch_size}.jsonl"
        )

        for seed in SEEDS:
            # Seeds generation. Without this the vendor presets (do_sample=True)
            # make every run non-reproducible.
            set_seed(seed)

            if MAX_EXAMPLES is None or MAX_EXAMPLES >= len(dataset):
                sampled_dataset = dataset
            else:
                sampled_dataset = random.Random(seed).sample(dataset, MAX_EXAMPLES)

            start_time = time.time()
            gold_sequences = []
            pred_sequences = []
            wrong_text_count = 0
            reasoning_unterminated_count = 0
            reasoning_token_counts = []
            all_entities_wrongly_unaligned = 0
            unaligned_entity_count = 0
            total_predictions = 0
            total_batches = (len(sampled_dataset) + batch_size - 1) // batch_size

            # Semin et al.'s metric is micro: sum these six counters over the run,
            # then call f1_from_counts once at the end.
            hard_overlap = hard_predicted = hard_gold = 0
            soft_overlap = soft_predicted = soft_gold = 0

            toktrie = None
            if eval_mode == "constrained":
                toktrie = build_toktrie_from_tokenizer(tokenizer)

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
                # Gold character spans for Semin's metric, derived from the same
                # tags seqeval scores, so the two metrics share one gold.
                batch_gold_spans = bio_tags_to_char_spans(batch_tokens, batch_gold_tags)

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
                        )

                gen_stats = {}
                generated, num_output_tokens, generation_seconds = generate_markup(
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    eval_model=eval_mode,
                    input_text=input_text,
                    system_prompt=SYSTEM_PROMPT_CONSTR_GEN,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=DO_SAMPLE,
                    reasoning_model=reasoning_model,
                    reasoning_effort=args.reasoning_effort,
                    reasoning_end_marker=reasoning_end_marker,
                    stats_out=gen_stats,
                    repetition_penalty=repetition_penalty,
                    **sampling
                )
                # Reasoning/answer token split (0 / total for non-reasoning models).
                num_reasoning_tokens = gen_stats.get("num_reasoning_tokens", 0)
                num_answer_tokens = gen_stats.get("num_answer_tokens", num_output_tokens)

                if reasoning_model:
                    reasoning_token_counts.append(num_reasoning_tokens)
                    if not gen_stats.get("found_reasoning_end", False):
                        reasoning_unterminated_count += 1

                parsed = parse_spans_from_tagged_output(generated, set(labels_for_constrained))
                total_predictions += parsed["span_count"]
                exact_copy_ok = validate_reconstruction(parsed["reconstructed_text"], input_text)

                if not exact_copy_ok:
                    wrong_text_count += 1
                    if eval_mode == "constrained":
                        print(f"\n\n===== Warning at seed {seed}, batch {batch_idx + 1} =====")
                        print(f"Original text: \n{input_text}")
                        print(f"Reconstructed text: \n{parsed['reconstructed_text']}")
                        print(f"Generated markup: \n{generated}\n\n")
                        if reasoning_model:
                            print(f"Reasoning was on, so the maximum number of new generated tokens was hit\n")
                    pred_tags = ["O"] * len(batch_tokens)
                    # Reconstruction failed, so the predicted offsets index into the
                    # model's own text, not ours, and cannot be located in the input.
                    # Credit nothing, matching the all-O fallback for seqeval.
                    pred_spans = []
                    all_entities_wrongly_unaligned += parsed["span_count"]
                else:
                    pred_tags, unalign_count = spans_to_bio_tags(
                        tokens=batch_tokens,
                        entities=parsed["entities"],
                        valid_labels=set(labels_for_constrained),
                    )
                    unaligned_entity_count += unalign_count
                    all_entities_wrongly_unaligned += unalign_count
                    # Raw character spans, not the token-snapped ones: Semin et al.
                    # never touch tokenization, so snapping would change their metric.
                    pred_spans = [
                        {"start": e["start"], "end": e["end"], "label": e["label"]}
                        for e in parsed["entities"] if e["label"] in set(labels_for_constrained)
                    ]

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
                    "dataset": dataset_tag(),
                    "dataset_names": [ex["dataset_name"] for ex in batch],
                    "method": "constrained_gen",
                    "model": model_name,
                    "reasoning_enabled": reasoning_model,
                    "reasoning_marker_seen": gen_stats.get("reasoning_marker_seen", False),
                    "found_reasoning_end": gen_stats.get("found_reasoning_end", False),
                    "repetition_penalty": repetition_penalty,
                    "sampling_strategy": sampling_strategy,
                    "eval_mode": eval_mode,
                    "processor_class": config_label,
                    "batch_size": batch_size,
                    "seed": seed,
                    "batch_idx": batch_idx,
                    "example_ids": [ex["example_id"] for ex in batch],
                    "input_text": input_text,
                    "gold_tags": batch_gold_tags,
                    "pred_tags": pred_tags,
                    "gold_spans": batch_gold_spans,
                    "pred_spans": pred_spans,
                    "semin_hard_overlap": hard_counts["overlap_chars"],
                    "semin_hard_predicted": hard_counts["predicted_chars"],
                    "semin_hard_gold": hard_counts["gold_chars"],
                    "semin_soft_overlap": soft_counts["overlap_chars"],
                    "semin_soft_predicted": soft_counts["predicted_chars"],
                    "semin_soft_gold": soft_counts["gold_chars"],
                    "raw_output": generated,
                    "wrong_text": 0 if exact_copy_ok else 1,
                    "span_count": parsed["span_count"],
                    "num_output_tokens": num_output_tokens,
                    "num_reasoning_tokens": num_reasoning_tokens,
                    "num_answer_tokens": num_answer_tokens,
                    "reasoning_text": gen_stats.get("reasoning_text", ""),
                    "generation_seconds": generation_seconds,
                })

                if (batch_idx + 1) % EVAL_INTERVAL == 0:
                    partial = seqeval.compute(
                        predictions=pred_sequences,
                        references=gold_sequences,
                        scheme="IOB2",
                        mode="strict",
                        zero_division=0,
                    )
                    partial_hard = f1_from_counts(hard_overlap, hard_predicted, hard_gold)
                    elapsed = (time.time() - start_time) / 60.0
                    tqdm.write(
                        f"[{model_name} | {dataset_tag()} | {sampling_strategy} | {eval_mode} | {config_label} | bs={batch_size}] "
                        f"seed {seed}, batch {batch_idx + 1}/{total_batches} "
                        f"F1={partial['overall_f1']:.4f}, hardF1={partial_hard['f1']:.4f}, "
                        f"wrong_text={wrong_text_count}, unaligned_ent_count={unaligned_entity_count}, elapsed={elapsed:.1f}m"
                    )

            metrics = seqeval.compute(
                predictions=pred_sequences,
                references=gold_sequences,
                scheme="IOB2",
                mode="strict",
                zero_division=0,
            )
            hard = f1_from_counts(hard_overlap, hard_predicted, hard_gold)
            soft = f1_from_counts(soft_overlap, soft_predicted, soft_gold)

            elapsed_min = (time.time() - start_time) / 60.0
            exp_metrics.append({
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
                "wrong_text_count": wrong_text_count,
                "wrong_text_rate": wrong_text_count / max(total_batches, 1),
                "unaligned_entity_count": unaligned_entity_count,
                "unaligned_entity_rate": unaligned_entity_count / max(total_predictions, 1),
                "all_entities_wrongly_unaligned": all_entities_wrongly_unaligned,
                "all_entities_wrongly_unaligned_rate": all_entities_wrongly_unaligned / max(total_predictions, 1),
                "reasoning_unterminated_count": reasoning_unterminated_count,
                "reasoning_unterminated_rate": reasoning_unterminated_count / max(total_batches, 1),
                "reasoning_tokens_avg": (sum(reasoning_token_counts) / len(reasoning_token_counts))
                                        if reasoning_token_counts else 0.0,
                "elapsed_minute": elapsed_min,
            })

        pred_fh.close()

        precision_mean, precision_std = mean_std([m["precision"] for m in exp_metrics])
        recall_mean, recall_std = mean_std([m["recall"] for m in exp_metrics])
        f1_mean, f1_std = mean_std([m["f1"] for m in exp_metrics])
        accuracy_mean, accuracy_std = mean_std([m["accuracy"] for m in exp_metrics])
        hard_p_mean, hard_p_std = mean_std([m["semin_hard_precision"] for m in exp_metrics])
        hard_r_mean, hard_r_std = mean_std([m["semin_hard_recall"] for m in exp_metrics])
        hard_f1_mean, hard_f1_std = mean_std([m["semin_hard_f1"] for m in exp_metrics])
        soft_p_mean, soft_p_std = mean_std([m["semin_soft_precision"] for m in exp_metrics])
        soft_r_mean, soft_r_std = mean_std([m["semin_soft_recall"] for m in exp_metrics])
        soft_f1_mean, soft_f1_std = mean_std([m["semin_soft_f1"] for m in exp_metrics])
        wrong_text_count_mean, wrong_text_count_std = mean_std([m["wrong_text_count"] for m in exp_metrics])
        wrong_text_rate_mean, wrong_text_rate_std = mean_std([m["wrong_text_rate"] for m in exp_metrics])
        unaligned_entity_count_mean, unaligned_entity_count_std = mean_std([m["unaligned_entity_count"] for m in exp_metrics])
        unaligned_entity_rate_mean, unaligned_entity_rate_std = mean_std([m["unaligned_entity_rate"] for m in exp_metrics])
        all_entities_wrongly_unaligned_mean, all_entities_wrongly_unaligned_std = mean_std([m["all_entities_wrongly_unaligned"] for m in exp_metrics])
        all_entities_wrongly_unaligned_rate_mean, all_entities_wrongly_unaligned_rate_std = mean_std([m["all_entities_wrongly_unaligned_rate"] for m in exp_metrics])
        reasoning_unterminated_count_mean, reasoning_unterminated_count_std = mean_std([m["reasoning_unterminated_count"] for m in exp_metrics])
        reasoning_unterminated_rate_mean, reasoning_unterminated_rate_std = mean_std([m["reasoning_unterminated_rate"] for m in exp_metrics])
        reasoning_tokens_avg_mean, reasoning_tokens_avg_std = mean_std([m["reasoning_tokens_avg"] for m in exp_metrics])
        elapsed_mean, elapsed_std = mean_std([m["elapsed_minute"] for m in exp_metrics])

        results.append({
            "model": model_name,
            "dataset": dataset_tag(),
            "reasoning_enabled": reasoning_model,
            "reasoning_effort": args.reasoning_effort if args.reasoning_effort else "n|a",
            "repetition_penalty": repetition_penalty,
            "sampling_strategy": sampling_strategy,
            "do_sample": DO_SAMPLE,
            "eval_mode": eval_mode,
            "processor_class": config_label,
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
            "wrong_text_count_avg": round(wrong_text_count_mean, 3),
            "wrong_text_count_std": round(wrong_text_count_std, 3),
            "wrong_text_rate_report": format_pm(to_pct(wrong_text_rate_mean), to_pct(wrong_text_rate_std)),
            "unaligned_entity_count_avg": round(unaligned_entity_count_mean, 3),
            "unaligned_entity_count_std": round(unaligned_entity_count_std, 3),
            "unaligned_entity_rate_report": format_pm(to_pct(unaligned_entity_rate_mean), to_pct(unaligned_entity_rate_std)),
            "all_entities_wrongly_unaligned_avg": round(all_entities_wrongly_unaligned_mean, 3),
            "all_entities_wrongly_unaligned_std": round(all_entities_wrongly_unaligned_std, 3),
            "all_entities_wrongly_unaligned_rate_report": format_pm(to_pct(all_entities_wrongly_unaligned_rate_mean), to_pct(all_entities_wrongly_unaligned_rate_std)),
            "reasoning_unterminated_count_avg": round(reasoning_unterminated_count_mean, 3),
            "reasoning_unterminated_count_std": round(reasoning_unterminated_count_std, 3),
            "reasoning_unterminated_rate_report": format_pm(to_pct(reasoning_unterminated_rate_mean), to_pct(reasoning_unterminated_rate_std)),
            "reasoning_tokens_avg_avg": round(reasoning_tokens_avg_mean, 3),
            "reasoning_tokens_avg_std": round(reasoning_tokens_avg_std, 3),
            "elapsed_minute_avg": round(elapsed_mean, 3),
            "elapsed_minute_std": round(elapsed_std, 3),
        })

# Save intermediate results to CSV after each model evaluation to avoid data loss in case of interruptions
intermediate_results_df = pd.DataFrame(results)
intermediate_results_path = f"{RESULTS_ROOT}/{results_dir}/Constrained-Gen/Csv/{dataset_tag()}_{model_name.split('/')[-1]}_{batch_size}_BS_{config_tag()}_{sampling_strategy}.csv"
intermediate_results_txt_path = intermediate_results_path.replace("Csv", "Txt").replace(".csv", ".txt")

os.makedirs(os.path.dirname(intermediate_results_path), exist_ok=True)
os.makedirs(os.path.dirname(intermediate_results_txt_path), exist_ok=True)

intermediate_results_df.to_csv(intermediate_results_path, index=False)

with open(intermediate_results_txt_path, "w") as f:
    f.write(intermediate_results_df.to_string(index=False))

print(f"\nIntermediate results saved to {intermediate_results_path} and {intermediate_results_txt_path}")
