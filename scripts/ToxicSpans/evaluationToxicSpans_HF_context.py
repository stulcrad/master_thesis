"""
Context-based ToxicSpans evaluation using HuggingFace transformers (HF) for generation.

Logs per-example predictions to JSONL, then computes character-level F1 and saves to CSV.
  P = |pred ∩ gold| / |pred|,  R = |pred ∩ gold| / |gold|,  F1 = harmonic mean.
  Empty gold + empty pred → F1=1;  empty gold + non-empty pred → F1=0.
"""
import sys
import os
import time
import statistics
from typing import List

import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.system_prompts import SYSTEM_PROMPT_TOXIC_SPANS_MD
from utils.context_matching_utils import json_safe_parse, assign_spans_from_context
from utils.utils_functions import (
    generate_markup, tokenize_with_offsets, compute_character_f1,
    parse_position, example_to_tokens, chars_to_spans,
    mean_std, to_pct, format_pm, open_jsonl_writer, log_jsonl,
)

# Per-example predictions (JSONL, one line per generation) -- required for
# paired significance tests and post-hoc metrics without re-running.
PRED_DIR = "/home/stulcrad/master_thesis/Experiment_results/ToxicSpans/Context-Based/Predictions"

valid_labels = {"TOXIC"}

MAX_EXAMPLES = None
N_ITERS = 1
EVAL_INTERVAL = 100
BATCH_SIZE = 1  # fixed -- cannot batch across different posts
FUZZY_MODES = [False]
FUZZY_THRESHOLD = 0.6

DO_SAMPLE = False
TEMPERATURE = 0.2
MAX_NEW_TOKENS = 32578

MODEL_NAMES = ["google/gemma-3-4b-it", "Qwen/Qwen3-8B", "openai/gpt-oss-20b", "meta-llama/Llama-3.1-8B-Instruct"]

# Define system prompts to evaluate
prompts = {
    SYSTEM_PROMPT_TOXIC_SPANS_MD: "SYSTEM_PROMPT_TOXIC_SPANS_MD"
}

print("Loading heegyu/toxic-spans test split...")
dataset = load_dataset("heegyu/toxic-spans", split="test")
print(f"Examples in test split: {len(dataset)}")

print(f"Total examples sampled per run: {MAX_EXAMPLES}")
print(f"Number of iterations: {N_ITERS}")

all_results = []

for model_name in MODEL_NAMES:
    print(f"\nLoading model/tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )

    for FUZZY in FUZZY_MODES:
        print(f"\nFUZZY mode: {FUZZY}, FUZZY_THRESHOLD: {FUZZY_THRESHOLD}\n")

        for prompt in prompts.keys():
            print(f"\n===== Using system prompt: {prompts[prompt]} =====\n", flush=True)
            print(f"==== Evaluating model: {model_name} ====", flush=True)

            model_short = model_name.split("/")[-1]
            pred_fh = open_jsonl_writer(
                f"{PRED_DIR}/hf_context_{model_short}_{prompts[prompt]}_{'fuzzy' if FUZZY else 'exact'}.jsonl"
            )

            exp_metrics = []

            for exp_id in range(N_ITERS):
                print(f"\n--- Running experiment {exp_id + 1}/{N_ITERS} ---\n")

                if MAX_EXAMPLES is None:
                    sampled_dataset = dataset
                else:
                    sampled_dataset = dataset.shuffle(seed=42 + exp_id).select(range(MAX_EXAMPLES))

                system_prompt = prompt
                reasoning_effort = "low" if "gpt-oss" in model_name.lower() else None

                start_time = time.time()

                char_f1_per_post: List[float] = []
                char_p_per_post: List[float] = []
                char_r_per_post: List[float] = []
                context_not_in_input_count = 0
                entity_not_in_context_count = 0
                fuzzy_helped_count = 0
                exact_match_count = 0
                format_invalid_count = 0
                invalid_label_count = 0
                num_generations_count = 0
                total_predictions = 0

                for idx in tqdm(range(len(sampled_dataset)), file=sys.stdout,
                                 desc=f"exp {exp_id + 1}/{N_ITERS}"):
                    example = sampled_dataset[idx]
                    tokens = example_to_tokens(example['text_of_post'])

                    if not tokens:
                        gold_chars = set(parse_position(example["position"]))
                        cp, cr, cf = compute_character_f1(gold_chars, set())
                        char_f1_per_post.append(cf)
                        char_p_per_post.append(cp)
                        char_r_per_post.append(cr)
                        continue

                    num_generations_count += 1
                    post_text = " ".join(tokens)

                    # --- HF generation, replacing the Ollama chat-completion call ---
                    try:
                        content, num_output_tokens, generation_seconds = generate_markup(
                            model=model,
                            tokenizer=tokenizer,
                            processor=None,
                            eval_model="unconstrained",
                            input_text=post_text,
                            system_prompt=system_prompt,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=DO_SAMPLE,
                            temperature=TEMPERATURE,
                            reasoning_effort=reasoning_effort,
                        )
                        content = content.strip()
                        pred_json, json_parse_ok = json_safe_parse(content)
                        total_predictions += len(pred_json)
                    except Exception as e:
                        print(f"Error processing example {idx}: {e}")
                        content = ""
                        num_output_tokens, generation_seconds = 0, 0.0
                        pred_json, json_parse_ok = [], False

                    pred_tags, match_stats = assign_spans_from_context(
                        tokens,
                        pred_json,
                        fuzzy=FUZZY,
                        valid_labels=valid_labels,
                        fuzzy_threshold=FUZZY_THRESHOLD,
                        matching_type='anchor',
                        json_parse_ok=json_parse_ok,
                        return_stats=True,
                    )
                    context_not_in_input_count += match_stats['context_not_in_input']
                    entity_not_in_context_count += match_stats['entity_not_in_context']
                    fuzzy_helped_count += match_stats['fuzzy_helped']
                    exact_match_count += match_stats['exact_match']
                    format_invalid_count += match_stats['format_invalid']
                    invalid_label_count += match_stats['invalid_label_count']

                    # Character-level F1 (official metric)
                    _, orig_offsets = tokenize_with_offsets(example["text_of_post"])
                    pred_chars: set = set()
                    i_tok, n_tok = 0, len(pred_tags)
                    while i_tok < n_tok:
                        if pred_tags[i_tok] == "B-TOXIC":
                            span_start = orig_offsets[i_tok][0]
                            i_tok += 1
                            while i_tok < n_tok and pred_tags[i_tok] == "I-TOXIC":
                                i_tok += 1
                            pred_chars.update(range(span_start, orig_offsets[i_tok - 1][1]))
                        else:
                            i_tok += 1
                    gold_chars = set(parse_position(example["position"]))
                    cp, cr, cf = compute_character_f1(gold_chars, pred_chars)
                    char_f1_per_post.append(cf)
                    char_p_per_post.append(cp)
                    char_r_per_post.append(cr)

                    log_jsonl(pred_fh, {
                        "key": f"{42 + exp_id}:{idx}",
                        "dataset": "toxic_spans",
                        "method": "context_hf",
                        "model": model_name,
                        "system_prompt": prompts[prompt],
                        "fuzzy_mode": FUZZY,
                        "seed": 42 + exp_id,
                        "example_idx": idx,
                        "input_text": post_text,
                        "gold_spans": chars_to_spans(sorted(gold_chars)),
                        "pred_spans": chars_to_spans(sorted(pred_chars)),
                        "pred_tags": pred_tags,
                        "raw_output": content,
                        "match_stats": match_stats,
                        "char_precision": cp,
                        "char_recall": cr,
                        "char_f1": cf,
                        "num_output_tokens": num_output_tokens,
                        "generation_seconds": generation_seconds,
                    })

                    if (idx + 1) % EVAL_INTERVAL == 0:
                        elapsed = time.time() - start_time
                        tqdm.write(
                            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                            f"{model_name} {idx + 1}/{len(sampled_dataset)} "
                            f"charF1={statistics.mean(char_f1_per_post):.4f} "
                            f"charP={statistics.mean(char_p_per_post):.4f} "
                            f"charR={statistics.mean(char_r_per_post):.4f} | "
                            f"CtxMiss={context_not_in_input_count} "
                            f"EntMiss={entity_not_in_context_count} "
                            f"Fuzzy={fuzzy_helped_count} "
                            f"Exact={exact_match_count} "
                            f"FmtInvalid={format_invalid_count} | "
                            f"InvalidLabel={invalid_label_count} | "
                            f"elapsed={elapsed / 60:.1f}m",
                            file=sys.stdout,
                        )

                exp_duration = (time.time() - start_time) / 60.0
                exp_metrics.append({
                    "char_f1": statistics.mean(char_f1_per_post) if char_f1_per_post else 0.0,
                    "char_precision": statistics.mean(char_p_per_post) if char_p_per_post else 0.0,
                    "char_recall": statistics.mean(char_r_per_post) if char_r_per_post else 0.0,
                    "context_not_in_input": context_not_in_input_count,
                    "context_not_in_input_rate": context_not_in_input_count / max(total_predictions, 1),
                    "entity_not_in_context": entity_not_in_context_count,
                    "entity_not_in_context_rate": entity_not_in_context_count / max(total_predictions, 1),
                    "fuzzy_helped": fuzzy_helped_count,
                    "fuzzy_helped_rate": fuzzy_helped_count / max(total_predictions, 1),
                    "exact_match": exact_match_count,
                    "exact_match_rate": exact_match_count / max(total_predictions, 1),
                    "format_invalid": format_invalid_count,
                    "format_invalid_rate": format_invalid_count / max(num_generations_count, 1),
                    "invalid_label_count": invalid_label_count,
                    "invalid_label_rate": invalid_label_count / max(total_predictions, 1),
                    "elapsed_minute": exp_duration,
                })

            pred_fh.close()

            char_f1_mean, char_f1_std = mean_std([m["char_f1"] for m in exp_metrics])
            char_p_mean, char_p_std = mean_std([m["char_precision"] for m in exp_metrics])
            char_r_mean, char_r_std = mean_std([m["char_recall"] for m in exp_metrics])
            ctx_mean, ctx_std = mean_std([m["context_not_in_input"] for m in exp_metrics])
            ctx_rate_mean, ctx_rate_std = mean_std([m["context_not_in_input_rate"] for m in exp_metrics])
            ent_mean, ent_std = mean_std([m["entity_not_in_context"] for m in exp_metrics])
            ent_rate_mean, ent_rate_std = mean_std([m["entity_not_in_context_rate"] for m in exp_metrics])
            fuz_mean, fuz_std = mean_std([m["fuzzy_helped"] for m in exp_metrics])
            fuz_rate_mean, fuz_rate_std = mean_std([m["fuzzy_helped_rate"] for m in exp_metrics])
            exact_mean, exact_std = mean_std([m["exact_match"] for m in exp_metrics])
            exact_rate_mean, exact_rate_std = mean_std([m["exact_match_rate"] for m in exp_metrics])
            fmt_invalid_mean, fmt_invalid_std = mean_std([m["format_invalid"] for m in exp_metrics])
            fmt_invalid_rate_mean, fmt_invalid_rate_std = mean_std([m["format_invalid_rate"] for m in exp_metrics])
            invalid_label_mean, invalid_label_std = mean_std([m["invalid_label_count"] for m in exp_metrics])
            invalid_label_rate_mean, invalid_label_rate_std = mean_std([m["invalid_label_rate"] for m in exp_metrics])
            elapsed_mean, elapsed_std = mean_std([m["elapsed_minute"] for m in exp_metrics])

            all_results.append({
                "system_prompt": prompts[prompt],
                "model": model_name,
                "batch_size": BATCH_SIZE,
                "fuzzy_mode": FUZZY,
                "n_iters": N_ITERS,
                "char_f1_report": format_pm(to_pct(char_f1_mean), to_pct(char_f1_std)),
                "char_precision_report": format_pm(to_pct(char_p_mean), to_pct(char_p_std)),
                "char_recall_report": format_pm(to_pct(char_r_mean), to_pct(char_r_std)),
                "context_not_in_input_avg": round(ctx_mean, 3),
                "context_not_in_input_std": round(ctx_std, 3),
                "context_not_in_input_rate_report": format_pm(to_pct(ctx_rate_mean), to_pct(ctx_rate_std)),
                "entity_not_in_context_avg": round(ent_mean, 3),
                "entity_not_in_context_std": round(ent_std, 3),
                "entity_not_in_context_rate_report": format_pm(to_pct(ent_rate_mean), to_pct(ent_rate_std)),
                "fuzzy_helped_avg": round(fuz_mean, 3),
                "fuzzy_helped_std": round(fuz_std, 3),
                "fuzzy_helped_rate_report": format_pm(to_pct(fuz_rate_mean), to_pct(fuz_rate_std)),
                "exact_match_avg": round(exact_mean, 3),
                "exact_match_std": round(exact_std, 3),
                "exact_match_rate_report": format_pm(to_pct(exact_rate_mean), to_pct(exact_rate_std)),
                "format_invalid_avg": round(fmt_invalid_mean, 3),
                "format_invalid_std": round(fmt_invalid_std, 3),
                "format_invalid_rate_report": format_pm(to_pct(fmt_invalid_rate_mean), to_pct(fmt_invalid_rate_std)),
                "invalid_label_avg": round(invalid_label_mean, 3),
                "invalid_label_std": round(invalid_label_std, 3),
                "invalid_label_rate_report": format_pm(to_pct(invalid_label_rate_mean), to_pct(invalid_label_rate_std)),
                "elapsed_minute_avg": round(elapsed_mean, 3),
                "elapsed_minute_std": round(elapsed_std, 3),
            })

    # Free GPU memory before loading the next model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Save results to CSV
results_df = pd.DataFrame(all_results)

results_path = "/home/stulcrad/master_thesis/Experiment_results/ToxicSpans/Context-Based/Csv/HF_context_results_ToxicSpans.csv"
txt_path = results_path.replace(".csv", ".txt").replace("Csv", "Txt")

os.makedirs(os.path.dirname(results_path), exist_ok=True)
os.makedirs(os.path.dirname(txt_path), exist_ok=True)

results_df.to_csv(results_path, index=False)

with open(txt_path, "w") as f:
    f.write(results_df.to_string(index=False))

print(f"\nResults saved to {results_path} and {txt_path}")
