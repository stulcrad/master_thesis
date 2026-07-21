"""
Context-based NER evaluation using HuggingFace transformers (HF) for generation.

Logs per-example predictions to JSONL, then computes seqeval metrics and saves to CSV.
"""
import sys
import os
import time
import torch
import pandas as pd
import evaluate
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.system_prompts import SYSTEM_PROMPT_CONTEXT_MD
from utils.context_matching_utils import json_safe_parse, assign_spans_from_context
from utils.utils_functions import generate_markup, mean_std, to_pct, format_pm, open_jsonl_writer, log_jsonl

# Per-example predictions (JSONL, one line per generation) -- required for
# paired significance tests and post-hoc metrics without re-running.
PRED_DIR = "/home/stulcrad/master_thesis/Experiment_results/CoNLL/Context-Based/Predictions"

# Define label mappings
label2id = {
    'O': 0,
    'B-PER': 1,
    'I-PER': 2,
    'B-ORG': 3,
    'I-ORG': 4,
    'B-LOC': 5,
    'I-LOC': 6,
    'B-MISC': 7,
    'I-MISC': 8
}
id2label = {v: k for k, v in label2id.items()}

valid_labels = {"PER", "LOC", "ORG", "MISC"}


# Load seqeval for evaluation
seqeval = evaluate.load("seqeval")

MAX_EXAMPLES = None
N_ITERS = 1
EVAL_INTERVAL = 10
BATCH_SIZE = 32
FUZZY_MODES = [False]
FUZZY_THRESHOLD = 0.6

DO_SAMPLE = False
TEMPERATURE = 0.2
MAX_NEW_TOKENS = 32578 

# MODEL_NAMES = ["google/gemma-3-4b-it", "Qwen/Qwen3-8B", "openai/gpt-oss-20b", "meta-llama/Llama-3.1-8B-Instruct"]
MODEL_NAMES = ["openai/gpt-oss-20b"]

# Define system prompts to evaluate
prompts = {
    SYSTEM_PROMPT_CONTEXT_MD: "SYSTEM_PROMPT_CONTEXT_MD"
}

# Load CoNLL-2003 dataset
dataset = load_dataset("lhoestq/conll2003", split='test')

print(f"Total examples sampled per run: {MAX_EXAMPLES}")
print(f"Number of iterations: {N_ITERS}")
print(f"Batch size: {BATCH_SIZE}")

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
                if "qwen" in model_name.lower():
                    system_prompt += "\n\\no_think"
                reasoning_effort = "low" if "gpt-oss" in model_name.lower() else None

                start_time = time.time()

                true_entities, pred_entities = [], []
                context_not_in_input_count = 0
                entity_not_in_context_count = 0
                fuzzy_helped_count = 0
                exact_match_count = 0
                format_invalid_count = 0
                invalid_label_count = 0
                num_generations_count = 0
                total_predictions = 0

                num_batches = (len(sampled_dataset) + BATCH_SIZE - 1) // BATCH_SIZE
                for batch_idx in tqdm(range(num_batches), file=sys.stdout, desc=f"exp {exp_id + 1}/{N_ITERS}"):

                    batch = sampled_dataset.select(
                        range(batch_idx * BATCH_SIZE, min((batch_idx + 1) * BATCH_SIZE, len(sampled_dataset)))
                    )
                    if len(batch) == 0:
                        continue

                    num_generations_count += 1

                    batch_tokens = []
                    batch_gold_tags = []
                    for example in batch:
                        batch_tokens.extend(example['tokens'])
                        batch_gold_tags.extend([id2label[tid] for tid in example['ner_tags']])

                    full_text = " ".join(batch_tokens)

                    # --- HF generation, replacing the Ollama chat-completion call ---
                    # generate_markup() builds the same chat-template prompt used
                    # throughout the thesis, calls model.generate(), and returns only
                    # the newly generated tokens (output[len(prompt_tokens):]),
                    # exactly like generate_unconstrained_markup() does for the
                    # constrained-generation track.
                    try:
                        content, num_output_tokens, generation_seconds = generate_markup(
                            model=model,
                            tokenizer=tokenizer,
                            processor=None,
                            eval_model="unconstrained",
                            input_text=full_text,
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
                        print(f"Error processing batch: {e}")
                        content = ""
                        num_output_tokens, generation_seconds = 0, 0.0
                        pred_json, json_parse_ok = [], False

                    # Assign BIO tags based on predicted entities and contexts
                    pred_tags, match_stats = assign_spans_from_context(
                        batch_tokens,
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

                    log_jsonl(pred_fh, {
                        "key": f"{42 + exp_id}:{batch_idx}",
                        "dataset": "conll2003",
                        "method": "context_hf",
                        "model": model_name,
                        "system_prompt": prompts[prompt],
                        "fuzzy_mode": FUZZY,
                        "batch_size": BATCH_SIZE,
                        "seed": 42 + exp_id,
                        "batch_idx": batch_idx,
                        "input_text": full_text,
                        "gold_tags": batch_gold_tags,
                        "pred_tags": pred_tags,
                        "raw_output": content,
                        "match_stats": match_stats,
                        "num_output_tokens": num_output_tokens,
                        "generation_seconds": generation_seconds,
                    })

                    true_entities.append(batch_gold_tags)
                    pred_entities.append(pred_tags)

                    if (batch_idx + 1) % EVAL_INTERVAL == 0:
                        metrics_partial = seqeval.compute(
                            predictions=pred_entities, references=true_entities,
                            scheme="IOB2", mode="strict", zero_division=0
                        )
                        elapsed = time.time() - start_time
                        tqdm.write(
                            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                            f"{model_name} progress: {batch_idx + 1}/{num_batches} "
                            f"({(batch_idx + 1) / num_batches * 100:.2f}%) | "
                            f"F1={metrics_partial['overall_f1']:.4f}, "
                            f"P={metrics_partial['overall_precision']:.4f}, "
                            f"R={metrics_partial['overall_recall']:.4f} | "
                            f"Acc={metrics_partial['overall_accuracy']:.4f} | "
                            f"CtxMiss={context_not_in_input_count}, "
                            f"EntMissInCtx={entity_not_in_context_count}, "
                            f"FuzzyHelped={fuzzy_helped_count}, "
                            f"Exact={exact_match_count}, "
                            f"FmtInvalid={format_invalid_count} | "
                            f"InvalidLabel={invalid_label_count} | "
                            f"Elapsed: {elapsed / 60:.1f} min",
                            file=sys.stdout,
                        )

                metrics = seqeval.compute(
                    predictions=pred_entities, references=true_entities,
                    scheme="IOB2", mode="strict", zero_division=0
                )
                exp_duration = (time.time() - start_time) / 60.0
                exp_metrics.append({
                    "precision": metrics["overall_precision"],
                    "recall": metrics["overall_recall"],
                    "f1": metrics["overall_f1"],
                    "accuracy": metrics["overall_accuracy"],
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

            precision_mean, precision_std = mean_std([m["precision"] for m in exp_metrics])
            recall_mean, recall_std = mean_std([m["recall"] for m in exp_metrics])
            f1_mean, f1_std = mean_std([m["f1"] for m in exp_metrics])
            accuracy_mean, accuracy_std = mean_std([m["accuracy"] for m in exp_metrics])
            context_not_in_input_mean, context_not_in_input_std = mean_std([m["context_not_in_input"] for m in exp_metrics])
            context_not_in_input_rate_mean, context_not_in_input_rate_std = mean_std([m["context_not_in_input_rate"] for m in exp_metrics])
            entity_not_in_context_mean, entity_not_in_context_std = mean_std([m["entity_not_in_context"] for m in exp_metrics])
            entity_not_in_context_rate_mean, entity_not_in_context_rate_std = mean_std([m["entity_not_in_context_rate"] for m in exp_metrics])
            fuzzy_helped_mean, fuzzy_helped_std = mean_std([m["fuzzy_helped"] for m in exp_metrics])
            fuzzy_helped_rate_mean, fuzzy_helped_rate_std = mean_std([m["fuzzy_helped_rate"] for m in exp_metrics])
            exact_match_mean, exact_match_std = mean_std([m["exact_match"] for m in exp_metrics])
            exact_match_rate_mean, exact_match_rate_std = mean_std([m["exact_match_rate"] for m in exp_metrics])
            format_invalid_mean, format_invalid_std = mean_std([m["format_invalid"] for m in exp_metrics])
            format_invalid_rate_mean, format_invalid_rate_std = mean_std([m["format_invalid_rate"] for m in exp_metrics])
            invalid_label_mean, invalid_label_std = mean_std([m["invalid_label_count"] for m in exp_metrics])
            invalid_label_rate_mean, invalid_label_rate_std = mean_std([m["invalid_label_rate"] for m in exp_metrics])
            elapsed_mean, elapsed_std = mean_std([m["elapsed_minute"] for m in exp_metrics])

            all_results.append({
                "system_prompt": prompts[prompt],
                "model": model_name,
                "batch_size": BATCH_SIZE,
                "fuzzy_mode": FUZZY,
                "n_iters": N_ITERS,
                "precision_report": format_pm(to_pct(precision_mean), to_pct(precision_std)),
                "recall_report": format_pm(to_pct(recall_mean), to_pct(recall_std)),
                "f1_report": format_pm(to_pct(f1_mean), to_pct(f1_std)),
                "accuracy_report": format_pm(to_pct(accuracy_mean), to_pct(accuracy_std)),
                "context_not_in_input_avg": round(context_not_in_input_mean, 3),
                "context_not_in_input_std": round(context_not_in_input_std, 3),
                "context_not_in_input_rate_report": format_pm(to_pct(context_not_in_input_rate_mean), to_pct(context_not_in_input_rate_std)),
                "entity_not_in_context_avg": round(entity_not_in_context_mean, 3),
                "entity_not_in_context_std": round(entity_not_in_context_std, 3),
                "entity_not_in_context_rate_report": format_pm(to_pct(entity_not_in_context_rate_mean), to_pct(entity_not_in_context_rate_std)),
                "fuzzy_helped_avg": round(fuzzy_helped_mean, 3),
                "fuzzy_helped_std": round(fuzzy_helped_std, 3),
                "fuzzy_helped_rate_report": format_pm(to_pct(fuzzy_helped_rate_mean), to_pct(fuzzy_helped_rate_std)),
                "exact_match_avg": round(exact_match_mean, 3),
                "exact_match_std": round(exact_match_std, 3),
                "exact_match_rate_report": format_pm(to_pct(exact_match_rate_mean), to_pct(exact_match_rate_std)),
                "format_invalid_avg": round(format_invalid_mean, 3),
                "format_invalid_std": round(format_invalid_std, 3),
                "format_invalid_rate_report": format_pm(to_pct(format_invalid_rate_mean), to_pct(format_invalid_rate_std)),
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

results_path = f"/home/stulcrad/master_thesis/Experiment_results/CoNLL/Context-Based/Csv/HF_context_results_GPT_OSS.csv"
txt_path = results_path.replace(".csv", ".txt").replace("Csv", "Txt")

os.makedirs(os.path.dirname(results_path), exist_ok=True)
os.makedirs(os.path.dirname(txt_path), exist_ok=True)

results_df.to_csv(results_path, index=False)

with open(txt_path, "w") as f:
    f.write(results_df.to_string(index=False))

print(f"\nResults saved to {results_path} and {txt_path}")
