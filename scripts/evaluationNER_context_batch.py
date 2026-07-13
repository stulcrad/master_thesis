import sys, os, argparse
import evaluate
from tqdm import tqdm
import pandas as pd
from utils.system_prompts import SYSTEM_PROMPT_CONTEXT, SYSTEM_PROMPT_CONTEXT_MD, SYSTEM_PROMPT_CONTEXT_MD_SHORT
from openai import OpenAI
from datasets import load_dataset
import time
from utils.context_matching_utils import (
    json_safe_parse, assign_spans_from_context,
)
from utils.utils_functions import (
    mean_std, to_pct, format_pm
)

# Arguments and configuration
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
batch_size = parser.parse_args().batch_size

# Initialize Ollama client
node_name = os.getenv("CLUSTER_NODE")
print(f"Detected cluster node: {node_name}")

client = OpenAI(
    base_url = f"http://{node_name}:9069/v1",
    api_key = "ollama"
)

models = client.models.list()

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

# Load seqeval for evaluation
seqeval = evaluate.load("seqeval")

# Experiment parameters
# MAX_EXAMPLES = 1280
MAX_EXAMPLES = 250
# N_ITERS = 1
N_ITERS = 5
EVAL_INTERVAL = 10 # Log after every 10 iterations
if batch_size > 5:
    EVAL_INTERVAL = 5
BATCH_SIZES = [batch_size]
FUZZY_MODES = [False, True]
FUZZY_THRESHOLD = 0.6

# Define system prompts to evaluate
prompts = {
    SYSTEM_PROMPT_CONTEXT: "SYSTEM_PROMPT_CONTEXT",
    SYSTEM_PROMPT_CONTEXT_MD: "SYSTEM_PROMPT_CONTEXT_MD",
    SYSTEM_PROMPT_CONTEXT_MD_SHORT: "SYSTEM_PROMPT_CONTEXT_MD_SHORT",
}

# Load CoNLL-2003 dataset
dataset = load_dataset("lhoestq/conll2003", split='test')

# if MAX_EXAMPLES:
#     dataset = dataset.select(range(MAX_EXAMPLES))

print(f"Total examples in dataset: {MAX_EXAMPLES if MAX_EXAMPLES else len(dataset)}")
print(f"Number of iterations: {N_ITERS}")

for BATCH_SIZE in BATCH_SIZES:
    for FUZZY in FUZZY_MODES:     

        print(f"\nBATCH_SIZE: {BATCH_SIZE}")
        print(f"FUZZY mode: {FUZZY}, FUZZY_THRESHOLD: {FUZZY_THRESHOLD}\n")

        txt_path = f"/home/stulcrad/master_thesis/Experiment_results/CoNLL/Context-Based/Txt/ner_document_context_{BATCH_SIZE}_BATCHSZ_robust_prompt.txt" if not FUZZY else \
            f"/home/stulcrad/master_thesis/Experiment_results/CoNLL/Context-Based/Txt/ner_document_context_fuzzy_{BATCH_SIZE}_BATCHSZ_robust_prompt.txt"

        csv_path = txt_path.replace("/Txt/", "/Csv/").replace(".txt", ".csv")

        print(f"Text path: {txt_path}")
        print(f"Csv path: {csv_path}")

        all_results = []
        # Main evaluation loop, iterating over prompts and models
        for prompt in prompts.keys():
            print(f"\n===== Using system prompt: {prompts[prompt]} =====\n", flush=True)

            for model_name in ["gemma3:4b", "qwen3:8b", "gpt-oss:20b", "llama3.1:8b"]:
                print(f"\n==== Evaluating model: {model_name} ====", flush=True)
        
                exp_metrics = []

                # Repeat experiments for statistical significance
                for exp_id in range(N_ITERS):
                    print(f"\n--- Running experiment {exp_id+1}/{N_ITERS} ---\n")

                    # Random sample of dataset (without replacement)
                    sampled_dataset = dataset.shuffle(seed=42+exp_id).select(range(MAX_EXAMPLES))
        
                    # Low reasoning effort for faster responses
                    system_prompt = prompt
                    if model_name.startswith("qwen"):
                        system_prompt += "\n\\no_think"
                        pass

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

                    # Process dataset in batches
                    num_batches = (len(sampled_dataset) + BATCH_SIZE - 1) // BATCH_SIZE
                    for batch_idx in tqdm(range(num_batches), file=sys.stdout, desc=f"exp {exp_id + 1}/{N_ITERS}"):

                        # batch = dataset[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
                        batch = sampled_dataset.select(range(batch_idx * BATCH_SIZE, min((batch_idx + 1) * BATCH_SIZE, len(sampled_dataset))))

                        if len(batch) == 0:
                            continue

                        num_generations_count += 1

                        batch_tokens = []
                        batch_gold_tags = []
                        # Prepare full text and gold tags for the batch
                        for example in batch:
                            batch_tokens.extend(example['tokens'])
                            batch_gold_tags.extend([id2label[id] for id in example['ner_tags']])
                        # Join tokens to form full text input
                        full_text = " ".join(batch_tokens)
                        # Prepare and send request to the model
                        try:
                            req_kwargs = dict(model=model_name)
                            req_kwargs['messages'] = [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": full_text}
                            ]
                            req_kwargs['temperature'] = 0.2
                            if model_name.startswith("gpt-oss"):
                                req_kwargs['reasoning_effort'] = "low"

                            response = client.chat.completions.create(**req_kwargs)
                            content = response.choices[0].message.content.strip()
                            pred_json, json_parse_ok = json_safe_parse(content)
                            total_predictions += len(pred_json)
                        except Exception as e:
                            print(f"Error processing batch: {e}")
                            pred_json, json_parse_ok = [], False
                        # Assign BIO tags based on predicted entities and contexts
                        pred_tags, match_stats = assign_spans_from_context(
                            batch_tokens,
                            pred_json,
                            fuzzy=FUZZY,
                            valid_labels={"PER", "ORG", "LOC", "MISC"},
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

                        # Collect true and predicted entities for evaluation
                        true_entities.append(batch_gold_tags)
                        pred_entities.append(pred_tags)
                        # Periodic logging
                        if (batch_idx + 1) % EVAL_INTERVAL == 0:
                            metrics_partial = seqeval.compute(predictions=pred_entities, references=true_entities,
                                                            scheme="IOB2", mode="strict", zero_division=0)
                            elapsed = time.time() - start_time
                            tqdm.write(
                                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                                f"{model_name} progress: {batch_idx + 1}/{num_batches} "
                                f"({(batch_idx+1)/num_batches*100:.2f}%) | "
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
                                f"Elapsed: {elapsed/60:.1f} min",
                                file=sys.stdout,
                            )

                    # Compute metrics for this experiment
                    metrics = seqeval.compute(predictions=pred_entities, references=true_entities,
                                            scheme="IOB2", mode="strict", zero_division=0)
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
                        "elapsed_minute": exp_duration
                    })

                # Average metrics over experiments
                avg_precision = sum(m["precision"] for m in exp_metrics) / N_ITERS
                avg_recall =    sum(m["recall"] for m in exp_metrics) / N_ITERS
                avg_f1 =        sum(m["f1"] for m in exp_metrics) / N_ITERS
                avg_accuracy =  sum(m["accuracy"] for m in exp_metrics) / N_ITERS
                avg_context_not_in_input =       sum(m["context_not_in_input"] for m in exp_metrics) / N_ITERS
                avg_context_not_in_input_rate =  sum(m["context_not_in_input_rate"] for m in exp_metrics) / N_ITERS
                avg_entity_not_in_context =      sum(m["entity_not_in_context"] for m in exp_metrics) / N_ITERS
                avg_entity_not_in_context_rate = sum(m["entity_not_in_context_rate"] for m in exp_metrics) / N_ITERS
                avg_fuzzy_helped =      sum(m["fuzzy_helped"] for m in exp_metrics) / N_ITERS
                avg_fuzzy_helped_rate = sum(m["fuzzy_helped_rate"] for m in exp_metrics) / N_ITERS
                avg_exact_match =       sum(m["exact_match"] for m in exp_metrics) / N_ITERS
                avg_exact_match_rate =  sum(m["exact_match_rate"] for m in exp_metrics) / N_ITERS
                avg_elapsed =           sum(m["elapsed_minute"] for m in exp_metrics) / N_ITERS
        
                precision_mean, precision_std = mean_std([m["precision"] for m in exp_metrics])
                recall_mean, recall_std =       mean_std([m["recall"] for m in exp_metrics])
                f1_mean, f1_std =               mean_std([m["f1"] for m in exp_metrics])
                accuracy_mean, accuracy_std =   mean_std([m["accuracy"] for m in exp_metrics])
                context_not_in_input_mean, context_not_in_input_std =             mean_std([m["context_not_in_input"] for m in exp_metrics])
                context_not_in_input_rate_mean, context_not_in_input_rate_std =   mean_std([m["context_not_in_input_rate"] for m in exp_metrics])
                entity_not_in_context_mean, entity_not_in_context_std =           mean_std([m["entity_not_in_context"] for m in exp_metrics])
                entity_not_in_context_rate_mean, entity_not_in_context_rate_std = mean_std([m["entity_not_in_context_rate"] for m in exp_metrics])
                fuzzy_helped_mean, fuzzy_helped_std =           mean_std([m["fuzzy_helped"] for m in exp_metrics])
                fuzzy_helped_rate_mean, fuzzy_helped_rate_std = mean_std([m["fuzzy_helped_rate"] for m in exp_metrics])
                exact_match_mean, exact_match_std =             mean_std([m["exact_match"] for m in exp_metrics])
                exact_match_rate_mean, exact_match_rate_std =   mean_std([m["exact_match_rate"] for m in exp_metrics])
                format_invalid_mean, format_invalid_std =             mean_std([m["format_invalid"] for m in exp_metrics])
                format_invalid_rate_mean, format_invalid_rate_std =   mean_std([m["format_invalid_rate"] for m in exp_metrics])
                invalid_label_mean, invalid_label_std =             mean_std([m["invalid_label_count"] for m in exp_metrics])
                invalid_label_rate_mean, invalid_label_rate_std =   mean_std([m["invalid_label_rate"] for m in exp_metrics])
                elapsed_mean, elapsed_std =                     mean_std([m["elapsed_minute"] for m in exp_metrics])

                all_results.append({
                    "system_prompt": prompts[prompt],
                    "model": model_name,
                    "batch_size": BATCH_SIZE,
                    "fuzzy_mode": FUZZY,
                    "n_iters": N_ITERS,
                    "precision_pct":            round(to_pct(precision_mean), 2),
                    "precision_std_pct":        round(to_pct(precision_std), 2),
                    "precision_report":         format_pm(to_pct(precision_mean), to_pct(precision_std)),
                    "recall_pct":               round(to_pct(recall_mean), 2),
                    "recall_std_pct":           round(to_pct(recall_std), 2),
                    "recall_report":            format_pm(to_pct(recall_mean), to_pct(recall_std)),
                    "f1_pct":                   round(to_pct(f1_mean), 2),
                    "f1_std_pct":               round(to_pct(f1_std), 2),
                    "f1_report":                format_pm(to_pct(f1_mean), to_pct(f1_std)),
                    "accuracy_pct":             round(to_pct(accuracy_mean), 2),
                    "accuracy_std_pct":         round(to_pct(accuracy_std), 2),
                    "accuracy_report":          format_pm(to_pct(accuracy_mean), to_pct(accuracy_std)),
                    "context_not_in_input_avg": round(context_not_in_input_mean, 3),
                    "context_not_in_input_std": round(context_not_in_input_std, 3),
                    "context_not_in_input_rate_pct":        round(to_pct(context_not_in_input_rate_mean), 2),
                    "context_not_in_input_rate_std_pct":    round(to_pct(context_not_in_input_rate_std), 2),
                    "context_not_in_input_rate_report":     format_pm(to_pct(context_not_in_input_rate_mean), to_pct(context_not_in_input_rate_std)),
                    "entity_not_in_context_avg":            round(entity_not_in_context_mean, 3),
                    "entity_not_in_context_std":            round(entity_not_in_context_std, 3),
                    "entity_not_in_context_rate_pct":       round(to_pct(entity_not_in_context_rate_mean), 2),
                    "entity_not_in_context_rate_std_pct":   round(to_pct(entity_not_in_context_rate_std), 2),
                    "entity_not_in_context_rate_report":    format_pm(to_pct(entity_not_in_context_rate_mean), to_pct(entity_not_in_context_rate_std)),
                    "fuzzy_helped_avg":                     round(fuzzy_helped_mean, 3),
                    "fuzzy_helped_std":                     round(fuzzy_helped_std, 3),
                    "fuzzy_helped_rate_pct":                round(to_pct(fuzzy_helped_rate_mean), 2),
                    "fuzzy_helped_rate_std_pct":            round(to_pct(fuzzy_helped_rate_std), 2),
                    "fuzzy_helped_rate_report":             format_pm(to_pct(fuzzy_helped_rate_mean), to_pct(fuzzy_helped_rate_std)),
                    "exact_match_avg":          round(exact_match_mean, 3),
                    "exact_match_std":          round(exact_match_std, 3),
                    "exact_match_rate_pct":     round(to_pct(exact_match_rate_mean), 2),
                    "exact_match_rate_std_pct": round(to_pct(exact_match_rate_std), 2),
                    "exact_match_rate_report":  format_pm(to_pct(exact_match_rate_mean), to_pct(exact_match_rate_std)),
                    "format_invalid_avg":          round(format_invalid_mean, 3),
                    "format_invalid_std":          round(format_invalid_std, 3),
                    "format_invalid_rate_pct":     round(to_pct(format_invalid_rate_mean), 2),
                    "format_invalid_rate_std_pct": round(to_pct(format_invalid_rate_std), 2),
                    "format_invalid_rate_report":  format_pm(to_pct(format_invalid_rate_mean), to_pct(format_invalid_rate_std)),
                    "invalid_label_avg":          round(invalid_label_mean, 3),
                    "invalid_label_std":          round(invalid_label_std, 3),
                    "invalid_label_rate_pct":     round(to_pct(invalid_label_rate_mean), 2),
                    "invalid_label_rate_std_pct": round(to_pct(invalid_label_rate_std), 2),
                    "invalid_label_rate_report":  format_pm(to_pct(invalid_label_rate_mean), to_pct(invalid_label_rate_std)),
                    "elapsed_minute_avg":       round(elapsed_mean, 3),
                    "elapsed_minute_std":       round(elapsed_std, 3)
                })


        df = pd.DataFrame(all_results)

        import os
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        with open(txt_path, "w") as f:
            f.write(df.to_string(index=False))

        df.to_csv(csv_path, index=False, encoding="utf-8")

        print(f"Results saved to {txt_path} and {csv_path}") 
