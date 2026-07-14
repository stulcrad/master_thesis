"""
Grammar-constrained baseline for the context-based ToxicSpans approach (xgrammar).

Reuses generate_markup() completely unchanged: any HF-compatible LogitsProcessor
can be passed as `processor` with eval_model="constrained".

Evaluation metric: character-level F1 averaged over examples.
  P = |pred ∩ gold| / |pred|,  R = |pred ∩ gold| / |gold|,  F1 = harmonic mean.
  Empty gold + empty pred → F1=1;  empty gold + non-empty pred → F1=0.
"""
import sys
import os
import time
import json as json_module
import statistics
from typing import List

import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import xgrammar as xgr

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

VALID_LABELS = {"TOXIC"}

MAX_EXAMPLES = None
N_ITERS = 1
EVAL_INTERVAL = 100
BATCH_SIZE = 1  # fixed -- cannot batch across different posts
FUZZY_THRESHOLD = 0.6

DO_SAMPLE = False
TEMPERATURE = 0.2
MAX_NEW_TOKENS = 32578

MODEL_NAMES = ["google/gemma-3-4b-it", "Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B-Instruct"]

# Single prompt variant
SYSTEM_PROMPT = SYSTEM_PROMPT_TOXIC_SPANS_MD

# JSON schema: same {entity, label, context} shape the free-form prompts
# already ask for, with `label` constrained to an enum so format AND label
# validity are both guaranteed structurally, not merely requested via prompt.
ENTITY_LIST_SCHEMA = json_module.dumps({
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "entity": {"type": "string"},
            "label": {"type": "string", "enum": sorted(VALID_LABELS)},
            "context": {"type": "string"},
        },
        "required": ["entity", "label", "context"],
    },
})

print("Loading heegyu/toxic-spans test split...")
dataset = load_dataset("heegyu/toxic-spans", split="test")
print(f"Examples in test split: {len(dataset)}")

print(f"Total examples sampled per run: {MAX_EXAMPLES}")
print(f"Number of iterations: {N_ITERS}")

all_results = []

for model_name in MODEL_NAMES:
    print(f"\nLoading model/tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )

    # Compile the grammar ONCE per model/tokenizer. A fresh LogitsProcessor is
    # still required per generation: the compiled grammar is stateless and
    # reusable, but its matcher tracks position within a single generation.
    vocab_size = getattr(config, "vocab_size", None)
    if vocab_size is None:
        vocab_size = config.text_config.vocab_size
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled_grammar = grammar_compiler.compile_json_schema(ENTITY_LIST_SCHEMA)

    system_prompt = SYSTEM_PROMPT

    model_short = model_name.split("/")[-1]
    pred_fh = open_jsonl_writer(f"{PRED_DIR}/grammar_xgrammar_{model_short}.jsonl")

    exp_metrics = []

    for exp_id in range(N_ITERS):
        print(f"\n--- Running experiment {exp_id + 1}/{N_ITERS} ---\n")

        if MAX_EXAMPLES is None:
            sampled_dataset = dataset
        else:
            sampled_dataset = dataset.shuffle(seed=42 + exp_id).select(range(MAX_EXAMPLES))

        start_time = time.time()

        char_f1_per_post = {False: [], True: []}
        char_p_per_post = {False: [], True: []}
        char_r_per_post = {False: [], True: []}
        context_not_in_input_count = {False: 0, True: 0}
        entity_not_in_context_count = {False: 0, True: 0}
        exact_match_count = {False: 0, True: 0}
        fuzzy_helped_count = 0
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
                for fuzzy in (False, True):
                    char_f1_per_post[fuzzy].append(cf)
                    char_p_per_post[fuzzy].append(cp)
                    char_r_per_post[fuzzy].append(cr)
                continue

            num_generations_count += 1
            post_text = " ".join(tokens)

            # Fresh LogitsProcessor per generation: the compiled grammar is
            # reused, but a matcher tracks position within ONE generation and
            # must not carry state over from the previous example.
            xgr_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)

            try:
                content, num_output_tokens, generation_seconds = generate_markup(
                    model=model,
                    tokenizer=tokenizer,
                    processor=xgr_processor,
                    eval_model="constrained",
                    input_text=post_text,
                    system_prompt=system_prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=DO_SAMPLE,
                    temperature=TEMPERATURE,
                )
                content = content.strip()
                pred_json, json_parse_ok = json_safe_parse(content)
                total_predictions += len(pred_json)
            except Exception as e:
                print(f"Error processing example {idx}: {e}")
                content = ""
                num_output_tokens, generation_seconds = 0, 0.0
                pred_json, json_parse_ok = [], False

            gold_chars = set(parse_position(example["position"]))
            _, orig_offsets = tokenize_with_offsets(example["text_of_post"])

            # Compute BOTH exact and fuzzy stats from this single generation (a paired comparison)
            match_stats = None
            pred_chars_by_fuzzy = {}
            for fuzzy in (False, True):
                pred_tags, match_stats = assign_spans_from_context(
                    tokens,
                    pred_json,
                    fuzzy=fuzzy,
                    valid_labels=VALID_LABELS,
                    fuzzy_threshold=FUZZY_THRESHOLD,
                    matching_type='anchor',
                    json_parse_ok=json_parse_ok,
                    return_stats=True,
                )
                context_not_in_input_count[fuzzy] += match_stats['context_not_in_input']
                entity_not_in_context_count[fuzzy] += match_stats['entity_not_in_context']
                exact_match_count[fuzzy] += match_stats['exact_match']
                if fuzzy:
                    fuzzy_helped_count += match_stats['fuzzy_helped']

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
                pred_chars_by_fuzzy[fuzzy] = pred_chars

                cp, cr, cf = compute_character_f1(gold_chars, pred_chars)
                char_f1_per_post[fuzzy].append(cf)
                char_p_per_post[fuzzy].append(cp)
                char_r_per_post[fuzzy].append(cr)

            # format_invalid / invalid_label_count are computed before the
            # fuzzy/exact branch inside assign_spans_from_context, so they are
            # identical regardless of which `fuzzy` pass produced them.
            format_invalid_count += match_stats['format_invalid']
            invalid_label_count += match_stats['invalid_label_count']

            log_jsonl(pred_fh, {
                "key": f"{42 + exp_id}:{idx}",
                "dataset": "toxic_spans",
                "method": "xgrammar_json_schema",
                "model": model_name,
                "seed": 42 + exp_id,
                "example_idx": idx,
                "input_text": post_text,
                "gold_spans": chars_to_spans(sorted(gold_chars)),
                "pred_spans_exact": chars_to_spans(sorted(pred_chars_by_fuzzy[False])),
                "pred_spans_fuzzy": chars_to_spans(sorted(pred_chars_by_fuzzy[True])),
                "char_precision_exact": char_p_per_post[False][-1],
                "char_recall_exact": char_r_per_post[False][-1],
                "char_f1_exact": char_f1_per_post[False][-1],
                "char_precision_fuzzy": char_p_per_post[True][-1],
                "char_recall_fuzzy": char_r_per_post[True][-1],
                "char_f1_fuzzy": char_f1_per_post[True][-1],
                "raw_output": content,
                "format_invalid": match_stats['format_invalid'],
                "invalid_label_count": match_stats['invalid_label_count'],
                "num_output_tokens": num_output_tokens,
                "generation_seconds": generation_seconds,
            })

            if (idx + 1) % EVAL_INTERVAL == 0:
                elapsed = time.time() - start_time
                tqdm.write(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"{model_name} {idx + 1}/{len(sampled_dataset)} "
                    f"charF1(exact)={statistics.mean(char_f1_per_post[False]):.4f} | "
                    f"FmtInvalid={format_invalid_count}, InvalidLabel={invalid_label_count} | "
                    f"Elapsed: {elapsed / 60:.1f} min",
                    file=sys.stdout,
                )

        exp_duration = (time.time() - start_time) / 60.0

        for fuzzy in (False, True):
            exp_metrics.append({
                "fuzzy_mode": fuzzy,
                "char_f1": statistics.mean(char_f1_per_post[fuzzy]) if char_f1_per_post[fuzzy] else 0.0,
                "char_precision": statistics.mean(char_p_per_post[fuzzy]) if char_p_per_post[fuzzy] else 0.0,
                "char_recall": statistics.mean(char_r_per_post[fuzzy]) if char_r_per_post[fuzzy] else 0.0,
                "context_not_in_input": context_not_in_input_count[fuzzy],
                "context_not_in_input_rate": context_not_in_input_count[fuzzy] / max(total_predictions, 1),
                "entity_not_in_context": entity_not_in_context_count[fuzzy],
                "entity_not_in_context_rate": entity_not_in_context_count[fuzzy] / max(total_predictions, 1),
                "fuzzy_helped": fuzzy_helped_count if fuzzy else 0,
                "fuzzy_helped_rate": (fuzzy_helped_count / max(total_predictions, 1)) if fuzzy else 0.0,
                "exact_match": exact_match_count[fuzzy],
                "exact_match_rate": exact_match_count[fuzzy] / max(total_predictions, 1),
                "format_invalid": format_invalid_count,
                "format_invalid_rate": format_invalid_count / max(num_generations_count, 1),
                "invalid_label": invalid_label_count,
                "invalid_label_rate": invalid_label_count / max(total_predictions, 1),
                "elapsed_minute": exp_duration,
            })

    pred_fh.close()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for fuzzy in (False, True):
        rows = [m for m in exp_metrics if m["fuzzy_mode"] == fuzzy]

        char_f1_mean, char_f1_std = mean_std([m["char_f1"] for m in rows])
        char_p_mean, char_p_std = mean_std([m["char_precision"] for m in rows])
        char_r_mean, char_r_std = mean_std([m["char_recall"] for m in rows])
        ctx_mean, ctx_std = mean_std([m["context_not_in_input"] for m in rows])
        ctx_rate_mean, ctx_rate_std = mean_std([m["context_not_in_input_rate"] for m in rows])
        ent_mean, ent_std = mean_std([m["entity_not_in_context"] for m in rows])
        ent_rate_mean, ent_rate_std = mean_std([m["entity_not_in_context_rate"] for m in rows])
        fuz_mean, fuz_std = mean_std([m["fuzzy_helped"] for m in rows])
        fuz_rate_mean, fuz_rate_std = mean_std([m["fuzzy_helped_rate"] for m in rows])
        exact_mean, exact_std = mean_std([m["exact_match"] for m in rows])
        exact_rate_mean, exact_rate_std = mean_std([m["exact_match_rate"] for m in rows])
        fmt_invalid_mean, fmt_invalid_std = mean_std([m["format_invalid"] for m in rows])
        fmt_invalid_rate_mean, fmt_invalid_rate_std = mean_std([m["format_invalid_rate"] for m in rows])
        invalid_label_mean, invalid_label_std = mean_std([m["invalid_label"] for m in rows])
        invalid_label_rate_mean, invalid_label_rate_std = mean_std([m["invalid_label_rate"] for m in rows])
        elapsed_mean, elapsed_std = mean_std([m["elapsed_minute"] for m in rows])

        all_results.append({
            "model": model_name,
            "batch_size": BATCH_SIZE,
            "fuzzy_mode": fuzzy,
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

df = pd.DataFrame(all_results)

results_path = "/home/stulcrad/master_thesis/Experiment_results/ToxicSpans/Context-Based/Csv/grammar_baseline_xgrammar_toxic_spans.csv"
txt_path = results_path.replace(".csv", ".txt").replace("/Csv/", "/Txt/")

os.makedirs(os.path.dirname(results_path), exist_ok=True)
os.makedirs(os.path.dirname(txt_path), exist_ok=True)
df.to_csv(results_path, index=False)
with open(txt_path, "w") as f:
    f.write(df.to_string(index=False))
print(f"\nResults saved to {results_path} and {txt_path}")
