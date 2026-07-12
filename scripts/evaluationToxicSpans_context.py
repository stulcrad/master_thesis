"""
Context-based evaluation for span classification on the heegyu/toxic-spans dataset.

Evaluation metric: character-level F1 averaged over examples.
    P = |pred ∩ gold| / |pred|, R = |pred ∩ gold| / |gold|, F1 = 2PR/(P+R)
    Empty gold + empty pred -> F1=1.0; empty gold + non-empty pred -> F1=0.0.
"""
import sys
import os
import statistics
import time
from typing import List

from tqdm import tqdm
import pandas as pd
from openai import OpenAI
from datasets import load_dataset

from utils.system_prompts import (
    SYSTEM_PROMPT_TOXIC_SPANS,
    SYSTEM_PROMPT_TOXIC_SPANS_MD,
    SYSTEM_PROMPT_TOXIC_SPANS_MD_SHORT,
)
from utils.context_matching_utils import json_safe_parse, assign_spans_from_context
from utils.utils_functions import (
    tokenize_with_offsets, compute_character_f1,
    mean_std, to_pct, format_pm,
    parse_position, example_to_tokens,
)

# -------------------------
# Ollama client
# -------------------------
node_name = os.getenv("CLUSTER_NODE")
print(f"Detected cluster node: {node_name}")

client = OpenAI(
    base_url=f"http://{node_name}:9089/v1",
    api_key="ollama",
)

models = client.models.list()

print("Loading heegyu/toxic-spans test split...")
raw = load_dataset("heegyu/toxic-spans", split="test")
print(f"Examples in test split: {len(raw)}")

# -------------------------
# Experiment parameters
# -------------------------
N_ITERS = 5
EVAL_INTERVAL = 10
BATCH_SIZE = 1
FUZZY_MODES = [False, True]
FUZZY_THRESHOLD = 0.6

MAX_EXAMPLES = 150
print(f"Max examples: {MAX_EXAMPLES}, N_ITERS: {N_ITERS}, BATCH_SIZE: {BATCH_SIZE}")

prompts = {
    SYSTEM_PROMPT_TOXIC_SPANS:          "SYSTEM_PROMPT_TOXIC_SPANS",
    SYSTEM_PROMPT_TOXIC_SPANS_MD:       "SYSTEM_PROMPT_TOXIC_SPANS_MD",
    SYSTEM_PROMPT_TOXIC_SPANS_MD_SHORT: "SYSTEM_PROMPT_TOXIC_SPANS_MD_SHORT",
}

for FUZZY in FUZZY_MODES:
    print(f"\nFUZZY mode: {FUZZY}, threshold: {FUZZY_THRESHOLD}\n")

    txt_path = (
        f"/home/stulcrad/master_thesis/Experiment_results/ToxicSpans/Context-Based/Txt/"
        f"toxic_spans_context_fuzzy.txt"
        if FUZZY else
        f"/home/stulcrad/master_thesis/Experiment_results/ToxicSpans/Context-Based/Txt/"
        f"toxic_spans_context.txt"
    )
    csv_path = txt_path.replace("/Txt/", "/Csv/").replace(".txt", ".csv")

    print(f"Txt: {txt_path}\nCsv: {csv_path}")

    all_results = []

    for prompt in prompts.keys():
            print(f"\n===== Using system prompt: {prompts[prompt]} =====\n", flush=True)

            for model_name in ["gemma3:4b", "qwen3:8b", "gpt-oss:20b", "llama3.1:8b"]:
                print(f"\n==== Evaluating model: {model_name} ====", flush=True)

                exp_metrics = []

                for exp_id in range(N_ITERS):
                    print(f"\n--- exp {exp_id+1}/{N_ITERS} ---")

                    sampled = raw.shuffle(seed=42 + exp_id).select(range(MAX_EXAMPLES))

                    system_prompt = prompt
                    if model_name.startswith("qwen"):
                        system_prompt += "\n\\no_think"

                    start_time = time.time()
                    char_f1_per_post: List[float] = []
                    char_p_per_post:  List[float] = []
                    char_r_per_post:  List[float] = []
                    context_not_in_input_count = 0
                    entity_not_in_context_count = 0
                    fuzzy_helped_count = 0
                    exact_match_count = 0
                    format_invalid_count = 0
                    invalid_label_count = 0
                    num_generations_count = 0
                    total_predictions = 0

                    for idx in tqdm(range(len(sampled)), file=sys.stdout,
                                    desc=f"exp {exp_id+1}/{N_ITERS}"):
                        example = sampled[idx]
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

                        try:
                            req_kwargs = dict(
                                model=model_name,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user",   "content": post_text},
                                ],
                                temperature=0.2,
                            )
                            if model_name.startswith("gpt-oss"):
                                req_kwargs["reasoning_effort"] = "low"

                            response = client.chat.completions.create(**req_kwargs)
                            content = response.choices[0].message.content.strip()
                            pred_json, json_parse_ok = json_safe_parse(content)
                            total_predictions += len(pred_json)
                        except Exception as e:
                            print(f"Error at example {idx}: {e}")
                            pred_json, json_parse_ok = [], False

                        pred_tags, match_stats = assign_spans_from_context(
                            tokens,
                            pred_json,
                            fuzzy=FUZZY,
                            fuzzy_threshold=FUZZY_THRESHOLD,
                            matching_type="anchor",
                            json_parse_ok=json_parse_ok,
                            return_stats=True,
                        )

                        context_not_in_input_count  += match_stats["context_not_in_input"]
                        entity_not_in_context_count += match_stats["entity_not_in_context"]
                        fuzzy_helped_count          += match_stats["fuzzy_helped"]
                        exact_match_count           += match_stats["exact_match"]
                        format_invalid_count        += match_stats["format_invalid"]
                        invalid_label_count          += match_stats["invalid_label_count"]

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
                                pred_chars.update(range(span_start, orig_offsets[i_tok-1][1]))
                            else:
                                i_tok += 1
                        gold_chars = set(parse_position(example["position"]))
                        cp, cr, cf = compute_character_f1(gold_chars, pred_chars)
                        char_f1_per_post.append(cf)
                        char_p_per_post.append(cp)
                        char_r_per_post.append(cr)

                        if (idx + 1) % EVAL_INTERVAL == 0:
                            elapsed = time.time() - start_time
                            tqdm.write(
                                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                                f"{model_name} {idx+1}/{len(sampled)} "
                                f"charF1={statistics.mean(char_f1_per_post):.4f} "
                                f"charP={statistics.mean(char_p_per_post):.4f} "
                                f"charR={statistics.mean(char_r_per_post):.4f} | "
                                f"CtxMiss={context_not_in_input_count} "
                                f"EntMiss={entity_not_in_context_count} "
                                f"Fuzzy={fuzzy_helped_count} "
                                f"Exact={exact_match_count} "
                                f"FmtInvalid={format_invalid_count} | "
                                f"elapsed={elapsed/60:.1f}m",
                                file=sys.stdout,
                            )

                    exp_duration = (time.time() - start_time) / 60.0
                    exp_metrics.append({
                        "char_f1":        statistics.mean(char_f1_per_post) if char_f1_per_post else 0.0,
                        "char_precision": statistics.mean(char_p_per_post)  if char_p_per_post  else 0.0,
                        "char_recall":    statistics.mean(char_r_per_post)  if char_r_per_post  else 0.0,
                        "context_not_in_input":      context_not_in_input_count,
                        "context_not_in_input_rate": context_not_in_input_count / max(total_predictions, 1),
                        "entity_not_in_context":      entity_not_in_context_count,
                        "entity_not_in_context_rate": entity_not_in_context_count / max(total_predictions, 1),
                        "fuzzy_helped":      fuzzy_helped_count,
                        "fuzzy_helped_rate": fuzzy_helped_count / max(total_predictions, 1),
                        "exact_match":      exact_match_count,
                        "exact_match_rate": exact_match_count / max(total_predictions, 1),
                        "format_invalid":      format_invalid_count,
                        "format_invalid_rate": format_invalid_count / max(num_generations_count, 1),
                        "invalid_label_count":      invalid_label_count,
                        "invalid_label_rate": invalid_label_count / max(total_predictions, 1),
                        "elapsed_minute": exp_duration,
                    })
            
                char_f1_mean, char_f1_std = mean_std([m["char_f1"]        for m in exp_metrics])
                char_p_mean,  char_p_std  = mean_std([m["char_precision"]  for m in exp_metrics])
                char_r_mean,  char_r_std  = mean_std([m["char_recall"]     for m in exp_metrics])
                ctx_mean,     ctx_std     = mean_std([m["context_not_in_input"]      for m in exp_metrics])
                ctx_rate_mean,ctx_rate_std= mean_std([m["context_not_in_input_rate"] for m in exp_metrics])
                ent_mean,     ent_std     = mean_std([m["entity_not_in_context"]      for m in exp_metrics])
                ent_rate_mean,ent_rate_std= mean_std([m["entity_not_in_context_rate"] for m in exp_metrics])
                fuz_mean,     fuz_std     = mean_std([m["fuzzy_helped"]      for m in exp_metrics])
                fuz_rate_mean,fuz_rate_std= mean_std([m["fuzzy_helped_rate"] for m in exp_metrics])
                exact_mean,   exact_std   = mean_std([m["exact_match"]      for m in exp_metrics])
                exact_rate_mean,exact_rate_std = mean_std([m["exact_match_rate"] for m in exp_metrics])
                fmt_invalid_mean,      fmt_invalid_std      = mean_std([m["format_invalid"]      for m in exp_metrics])
                fmt_invalid_rate_mean, fmt_invalid_rate_std = mean_std([m["format_invalid_rate"] for m in exp_metrics])
                invalid_label_mean,      invalid_label_std      = mean_std([m["invalid_label_count"]      for m in exp_metrics])
                invalid_label_rate_mean, invalid_label_rate_std = mean_std([m["invalid_label_rate"] for m in exp_metrics])
                elapsed_mean, elapsed_std = mean_std([m["elapsed_minute"]   for m in exp_metrics])

                all_results.append({
                    "system_prompt": prompts[prompt],
                    "model":         model_name,
                    "fuzzy_mode":    FUZZY,
                    "n_iters":       N_ITERS,
                    "char_f1_pct":         round(to_pct(char_f1_mean), 2),
                    "char_f1_std_pct":     round(to_pct(char_f1_std),  2),
                    "char_f1_report":      format_pm(to_pct(char_f1_mean), to_pct(char_f1_std)),
                    "char_precision_pct":     round(to_pct(char_p_mean), 2),
                    "char_precision_std_pct": round(to_pct(char_p_std),  2),
                    "char_precision_report":  format_pm(to_pct(char_p_mean), to_pct(char_p_std)),
                    "char_recall_pct":     round(to_pct(char_r_mean), 2),
                    "char_recall_std_pct": round(to_pct(char_r_std),  2),
                    "char_recall_report":  format_pm(to_pct(char_r_mean), to_pct(char_r_std)),
                    "context_not_in_input_avg":      round(ctx_mean,  3),
                    "context_not_in_input_std":      round(ctx_std,   3),
                    "context_not_in_input_rate_pct": round(to_pct(ctx_rate_mean), 2),
                    "context_not_in_input_rate_std": round(to_pct(ctx_rate_std),  2),
                    "entity_not_in_context_avg":      round(ent_mean,  3),
                    "entity_not_in_context_std":      round(ent_std,   3),
                    "entity_not_in_context_rate_pct": round(to_pct(ent_rate_mean), 2),
                    "entity_not_in_context_rate_std": round(to_pct(ent_rate_std),  2),
                    "fuzzy_helped_avg":      round(fuz_mean,  3),
                    "fuzzy_helped_std":      round(fuz_std,   3),
                    "fuzzy_helped_rate_pct": round(to_pct(fuz_rate_mean), 2),
                    "fuzzy_helped_rate_std": round(to_pct(fuz_rate_std),  2),
                    "exact_match_avg":      round(exact_mean,  3),
                    "exact_match_std":      round(exact_std,   3),
                    "exact_match_rate_pct": round(to_pct(exact_rate_mean), 2),
                    "exact_match_rate_std": round(to_pct(exact_rate_std),  2),
                    "format_invalid_avg":      round(fmt_invalid_mean,  3),
                    "format_invalid_std":      round(fmt_invalid_std,   3),
                    "format_invalid_rate_pct": round(to_pct(fmt_invalid_rate_mean), 2),
                    "format_invalid_rate_std": round(to_pct(fmt_invalid_rate_std),  2),
                    "invalid_label_avg":      round(invalid_label_mean,  3),
                    "invalid_label_std":      round(invalid_label_std,   3),
                    "invalid_label_rate_pct": round(to_pct(invalid_label_rate_mean), 2),
                    "invalid_label_rate_std": round(to_pct(invalid_label_rate_std),  2),
                    "elapsed_minute_avg": round(elapsed_mean, 3),
                    "elapsed_minute_std": round(elapsed_std,  3),
                })

    df = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(txt_path, "w") as f:
        f.write(df.to_string(index=False))
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Results saved to {txt_path} and {csv_path}")
