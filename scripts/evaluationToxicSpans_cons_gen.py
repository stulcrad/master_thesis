"""
Constrained generation evaluation on Toxic Spans dataset, with two types of constrained decoding:
- "whole_sequence"
- "token_aware"

Evaluation metric: character-level F1 averaged over examples.
    P = |pred ∩ gold| / |pred|, R = |pred ∩ gold| / |gold|, F1 = 2PR/(P+R)
    Empty gold + empty pred -> F1=1.0; empty gold + non-empty pred -> F1=0.0.
"""
import sys
import os
import time
import statistics
from typing import List

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.utils_functions import (
    generate_markup, validate_reconstruction, 
    spans_to_bio_tags, parse_spans_from_tagged_output,
    mean_std, to_pct, format_pm,
    compute_character_f1,
    parse_position, example_to_tokens,
)
from utils.TokTrie import build_toktrie_from_tokenizer
from utils.TrieSpanConstrainedProcessor import TrieSpanConstrainedProcessor
from utils.TrieSpanConstrainedProcessorTokenAware import TrieSpanConstrainedProcessorTokenAware
from utils.system_prompts import SYSTEM_PROMPT_CONSTR_GEN_TOXIC_SPANS

from huggingface_hub import login
login(token="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")

# -------------------------
# Evaluation configuration
# -------------------------
MAX_EXAMPLES = 150
N_ITERS = 5
EVAL_INTERVAL = 10
BATCH_SIZE = 1

MODEL_NAMES = ["google/gemma-3-4b-it", "Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B-Instruct"]

DO_SAMPLES = [False, True]
TEMPERATURE = 0.2
MAX_NEW_TOKENS = 32578

EVAL_MODES = ["unconstrained", "constrained"]
PROCESSOR_CLASSES = ["whole_sequence", "token_aware"]

# Single label for the constrained processor
labels_for_constrained = ["TOXIC"]

# -------------------------
# Load dataset
# -------------------------
print("Loading heegyu/toxic-spans test split...")
raw = load_dataset("heegyu/toxic-spans", split="test")
print(f"Examples in test split: {len(raw)}")

print(f"Max examples per iteration: {MAX_EXAMPLES}")

results = []

for model_name in MODEL_NAMES:
    print(f"\nLoading model/tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )

    batch_size = BATCH_SIZE
    print(f"Batch size: {batch_size}")

    for do_sample in DO_SAMPLES:
        sampling_strategy = 'sampling' if do_sample else 'greedy'

        for eval_mode in EVAL_MODES:
            processor_class_options = PROCESSOR_CLASSES if eval_mode == "constrained" else [None]

            for processor_class in processor_class_options:
                exp_metrics = []
                config_label = processor_class if processor_class is not None else "n/a"
                print(
                    f"\nEvaluating model={model_name}, strategy={sampling_strategy}, "
                    f"mode={eval_mode}, processor_class={config_label}, batch_size={batch_size}"
                )

                for exp_id in range(N_ITERS):
                    sampled = raw.shuffle(seed=42 + exp_id).select(range(MAX_EXAMPLES))

                    start_time = time.time()
                    # Metrics for counting generation errors
                    wrong_text_count = 0
                    all_entities_wrongly_unaligned = 0
                    unaligned_entity_count = 0
                    total_predictions = 0
                    # Metrics per post
                    char_f1_per_post: List[float] = []
                    char_p_per_post:  List[float] = []
                    char_r_per_post:  List[float] = []
                    
                    toktrie = None
                    if eval_mode == "constrained":
                        toktrie = build_toktrie_from_tokenizer(tokenizer)

                    for idx in tqdm(range(len(sampled)),
                                          desc=f"exp {exp_id+1}/{N_ITERS}", file=sys.stdout):
                        example = sampled[idx]
                        tokens = example_to_tokens(example['text_of_post'])

                        # Handle edge case of empty input text (all spaces or empty string)
                        if not tokens:
                            gold_chars = set(parse_position(example["position"]))
                            cp, cr, cf = compute_character_f1(gold_chars, set())
                            char_f1_per_post.append(cf)
                            char_p_per_post.append(cp)
                            char_r_per_post.append(cr)
                            continue

                        input_text = " ".join(tokens)

                        processor = None
                        if eval_mode == 'constrained':
                            if processor_class == 'token_aware':
                                processor = TrieSpanConstrainedProcessorTokenAware(
                                    labels_for_constrained, input_text,
                                    tokenizer, toktrie
                                )
                            else:
                                processor = TrieSpanConstrainedProcessor(
                                    labels_for_constrained,
                                    input_text,
                                    tokenizer,
                                    toktrie,
                                )

                        generated = generate_markup(
                            model=model,
                            tokenizer=tokenizer,
                            processor=processor,
                            eval_model=eval_mode,
                            input_text=input_text,
                            system_prompt=SYSTEM_PROMPT_CONSTR_GEN_TOXIC_SPANS,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=do_sample,
                            temperature=TEMPERATURE,
                        )
                        
                        parsed = parse_spans_from_tagged_output(generated, set(labels_for_constrained))
                        total_predictions += parsed['span_count']
                        exact_copy_ok = validate_reconstruction(parsed['reconstructed_text'], input_text)

                        if not exact_copy_ok:
                            wrong_text_count += 1
                            if eval_mode == 'constrained':
                                print(f"\n\n===== Warning in exp {exp_id+1}, example {idx+1} =====")
                                print(f"Original:      {input_text[:120]!r}")
                                print(f"Reconstructed: {parsed['reconstructed_text'][:120]!r}")
                            pred_tags = ["O"] * len(tokens)
                            all_entities_wrongly_unaligned += parsed["span_count"]
                        else:
                            pred_tags, unalign_count = spans_to_bio_tags(
                                tokens=tokens,
                                entities=parsed['entities'],
                                valid_labels=set(labels_for_constrained),
                            )
                            unaligned_entity_count += unalign_count
                            all_entities_wrongly_unaligned += unalign_count

                        pred_chars: set = set()
                        if exact_copy_ok:
                            for ent in parsed['entities']:
                                pred_chars.update(range(ent["start"], ent["end"]))
                        gold_chars = set(parse_position(example["position"]))
                        cp, cr, cf = compute_character_f1(gold_chars, pred_chars)
                        char_f1_per_post.append(cf)
                        char_p_per_post.append(cp)
                        char_r_per_post.append(cr)

                        if (idx + 1) % EVAL_INTERVAL == 0:
                            elapsed = (time.time() - start_time) / 60.0
                            tqdm.write(
                                f"[{model_name} | {sampling_strategy} | {eval_mode} | "
                                f"{config_label}] "
                                f"exp {exp_id+1}/{N_ITERS}, "
                                f"{idx+1}/{len(sampled)} "
                                f"charF1={statistics.mean(char_f1_per_post):.4f} "
                                f"charP={statistics.mean(char_p_per_post):.4f} "
                                f"charR={statistics.mean(char_r_per_post):.4f} | "
                                f"wrong_text={wrong_text_count} "
                                f"unaligned={unaligned_entity_count} | "
                                f"elapsed={elapsed:.1f}m"
                            )

                    elapsed_min = (time.time() - start_time) / 60.0
                    exp_metrics.append({
                        "char_f1":        statistics.mean(char_f1_per_post) if char_f1_per_post else 0.0,
                        "char_precision": statistics.mean(char_p_per_post)  if char_p_per_post  else 0.0,
                        "char_recall":    statistics.mean(char_r_per_post)  if char_r_per_post  else 0.0,
                        "wrong_text_count": wrong_text_count,
                        "wrong_text_rate": wrong_text_count / max(len(sampled), 1),
                        "unaligned_entity_count": unaligned_entity_count,
                        "unaligned_entity_rate": unaligned_entity_count / max(total_predictions, 1),
                        "all_entities_wrongly_unaligned": all_entities_wrongly_unaligned,
                        "all_entities_wrongly_unaligned_rate": all_entities_wrongly_unaligned / max(total_predictions, 1),
                        "elapsed_minute": elapsed_min,
                    })

                char_f1_mean, char_f1_std = mean_std([m["char_f1"]        for m in exp_metrics])
                char_p_mean,  char_p_std  = mean_std([m["char_precision"]  for m in exp_metrics])
                char_r_mean,  char_r_std  = mean_std([m["char_recall"]     for m in exp_metrics])
                wt_mean,      wt_std      = mean_std([m["wrong_text_count"] for m in exp_metrics])
                wt_rate_mean, wt_rate_std = mean_std([m["wrong_text_rate"]  for m in exp_metrics])
                ua_mean,      ua_std      = mean_std([m["unaligned_entity_count"] for m in exp_metrics])
                ua_rate_mean, ua_rate_std = mean_std([m["unaligned_entity_rate"]  for m in exp_metrics])
                aau_mean,     aau_std     = mean_std([m["all_entities_wrongly_unaligned"]      for m in exp_metrics])
                aau_rate_mean,aau_rate_std= mean_std([m["all_entities_wrongly_unaligned_rate"] for m in exp_metrics])
                elapsed_mean, elapsed_std = mean_std([m["elapsed_minute"]   for m in exp_metrics])

                results.append({
                    "model":              model_name,
                    "sampling_strategy":  sampling_strategy,
                    "do_sample":          do_sample,
                    "eval_mode":          eval_mode,
                    "processor_class":    config_label,
                    "batch_size":         BATCH_SIZE,
                    "n_iters":            N_ITERS,
                    "char_f1_pct":              round(to_pct(char_f1_mean), 2),
                    "char_f1_std_pct":          round(to_pct(char_f1_std),  2),
                    "char_f1_report":           format_pm(to_pct(char_f1_mean), to_pct(char_f1_std)),
                    "char_precision_pct":       round(to_pct(char_p_mean), 2),
                    "char_precision_std_pct":   round(to_pct(char_p_std),  2),
                    "char_precision_report":    format_pm(to_pct(char_p_mean), to_pct(char_p_std)),
                    "char_recall_pct":          round(to_pct(char_r_mean), 2),
                    "char_recall_std_pct":      round(to_pct(char_r_std),  2),
                    "char_recall_report":       format_pm(to_pct(char_r_mean), to_pct(char_r_std)),
                    "wrong_text_count_avg":     round(wt_mean,  3),
                    "wrong_text_count_std":     round(wt_std,   3),
                    "wrong_text_rate_pct":      round(to_pct(wt_rate_mean), 2),
                    "wrong_text_rate_std_pct":  round(to_pct(wt_rate_std),  2),
                    "wrong_text_rate_report":   format_pm(to_pct(wt_rate_mean), to_pct(wt_rate_std)),
                    "unaligned_entity_count_avg":    round(ua_mean,  3),
                    "unaligned_entity_count_std":    round(ua_std,   3),
                    "unaligned_entity_rate_pct":     round(to_pct(ua_rate_mean), 2),
                    "unaligned_entity_rate_std_pct": round(to_pct(ua_rate_std),  2),
                    "unaligned_entity_rate_report":  format_pm(to_pct(ua_rate_mean), to_pct(ua_rate_std)),
                    "all_entities_wrongly_unaligned_avg":     round(aau_mean,  3),
                    "all_entities_wrongly_unaligned_std":     round(aau_std,   3),
                    "all_entities_wrongly_unaligned_rate_pct":     round(to_pct(aau_rate_mean), 2),
                    "all_entities_wrongly_unaligned_rate_std_pct": round(to_pct(aau_rate_std),  2),
                    "all_entities_wrongly_unaligned_rate_report":  format_pm(to_pct(aau_rate_mean), to_pct(aau_rate_std)),
                    "elapsed_minute_avg": round(elapsed_mean, 3),
                    "elapsed_minute_std": round(elapsed_std,  3),
                })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

results_df = pd.DataFrame(results)

results_path = (
    f"/home/stulcrad/master_thesis/Experiment_results/ToxicSpans/Constrained-Gen/Csv/"
    f"hf_all_configs_eval_{BATCH_SIZE}_BS_toxic_spans.csv"
)
txt_path = results_path.replace("Csv", "Txt").replace(".csv", ".txt")

os.makedirs(os.path.dirname(results_path), exist_ok=True)
os.makedirs(os.path.dirname(txt_path), exist_ok=True)

results_df.to_csv(results_path, index=False)

with open(txt_path, "w") as f:
    f.write(results_df.to_string(index=False))

print(f"\nResults saved to {results_path} and {txt_path}")
