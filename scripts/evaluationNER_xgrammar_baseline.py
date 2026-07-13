import sys
import time
import json as json_module
import torch
import pandas as pd
import evaluate
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import xgrammar as xgr

from utils.system_prompts import SYSTEM_PROMPT_CONTEXT
from utils.context_matching_utils import json_safe_parse, assign_spans_from_context
from utils.utils_functions import generate_markup, mean_std, to_pct, format_pm

# ---------------------------------------------------------------------------
# Grammar-constrained baseline for the context-based approach (xgrammar).
#
# Reuses generate_markup() / generate_constrained_markup() completely unchanged:
# any HF-compatible LogitsProcessor can be passed as `processor` with
# eval_model="constrained"
#
# Purpose: isolate "format enforcement" from "verbatim grounding". The JSON
# schema below constrains `label` to an ENUM of the valid classes, so BOTH
# format_invalid and invalid_label_rate should be ~0 by construction.
# context_not_in_input / entity_not_in_context are NOT constrained by any grammar
# (a grammar cannot express "this string must be a substring of some other, externally supplied
# text"), so those should still be nonzero whenever the model hallucinates
# content. That contrast is the whole point of this baseline.
# ---------------------------------------------------------------------------

id2label = {
    0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG',
    5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC',
}
VALID_LABELS = {"PER", "ORG", "LOC", "MISC"}

seqeval = evaluate.load("seqeval")

# Same sampling protocol as HF_context.ipynb, so the three context-based
# variants (Ollama / HF unconstrained / HF+xgrammar) are directly comparable
# on identical sampled sentences.
MAX_EXAMPLES = 1280
N_ITERS = 1
EVAL_INTERVAL = 10
BATCH_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
if BATCH_SIZE > 5:
    EVAL_INTERVAL = 5
FUZZY_THRESHOLD = 0.6

DO_SAMPLE = True
TEMPERATURE = 0.2
MAX_NEW_TOKENS = 32578

MODEL_NAMES = ['google/gemma-3-4b-it', 'Qwen/Qwen3-8B', 'meta-llama/Llama-3.1-8B-Instruct']

# Single prompt variant
SYSTEM_PROMPT = SYSTEM_PROMPT_CONTEXT

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

dataset = load_dataset("lhoestq/conll2003", split='test')

print(f"Total examples sampled per run: {MAX_EXAMPLES}")
print(f"Number of iterations: {N_ITERS}")
print(f"Batch size: {BATCH_SIZE}")

all_results = []

for model_name in MODEL_NAMES:
    print(f"\nLoading model/tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # device_map={"": 0},   # pin to a single GPU; a 4B model never needs sharding
        device_map="auto",
        torch_dtype="auto",
    )

    # Compile the grammar ONCE per model/tokenizer. A fresh LogitsProcessor is still required
    # per generation: the compiled grammar is stateless and reusable, but its matcher tracks position within a single generation.
    vocab_size = getattr(config, "vocab_size", None)
    if vocab_size is None:
        vocab_size = config.text_config.vocab_size
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled_grammar = grammar_compiler.compile_json_schema(ENTITY_LIST_SCHEMA)

    system_prompt = SYSTEM_PROMPT
    if "qwen" in model_name.lower():
        system_prompt += "\n\\no_think"

    exp_metrics = []

    for exp_id in range(N_ITERS):
        print(f"\n--- Running experiment {exp_id + 1}/{N_ITERS} ---\n")

        sampled_dataset = dataset.shuffle(seed=42 + exp_id).select(range(MAX_EXAMPLES))

        start_time = time.time()

        true_entities = []
        pred_entities_exact, pred_entities_fuzzy = [], []
        context_not_in_input_count = {False: 0, True: 0}
        entity_not_in_context_count = {False: 0, True: 0}
        exact_match_count = {False: 0, True: 0}
        fuzzy_helped_count = 0
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

            # Fresh LogitsProcessor per generation: the compiled grammar is
            # reused, but a matcher tracks position within ONE generation and
            # must not carry state over from the previous example.
            xgr_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)

            try:
                content = generate_markup(
                    model=model,
                    tokenizer=tokenizer,
                    processor=xgr_processor,
                    eval_model="constrained",
                    input_text=full_text,
                    system_prompt=system_prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=DO_SAMPLE,
                    temperature=TEMPERATURE,
                ).strip()
                pred_json, json_parse_ok = json_safe_parse(content)
                total_predictions += len(pred_json)
            except Exception as e:
                print(f"Error processing batch: {e}")
                pred_json, json_parse_ok = [], False

            # Compute BOTH exact and fuzzy stats from this single generation (a paired comparison)
            match_stats = None
            for fuzzy in (False, True):
                pred_tags, match_stats = assign_spans_from_context(
                    batch_tokens,
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
                (pred_entities_fuzzy if fuzzy else pred_entities_exact).append(pred_tags)

            # format_invalid / invalid_label_count are computed before the
            # fuzzy/exact branch inside assign_spans_from_context, so they are
            # identical regardless of which `fuzzy` pass produced them.
            format_invalid_count += match_stats['format_invalid']
            invalid_label_count += match_stats['invalid_label_count']

            true_entities.append(batch_gold_tags)

            if (batch_idx + 1) % EVAL_INTERVAL == 0:
                metrics_partial = seqeval.compute(
                    predictions=pred_entities_exact, references=true_entities,
                    scheme="IOB2", mode="strict", zero_division=0
                )
                elapsed = time.time() - start_time
                tqdm.write(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"{model_name} progress: {batch_idx + 1}/{num_batches} "
                    f"F1(exact)={metrics_partial['overall_f1']:.4f} | "
                    f"FmtInvalid={format_invalid_count}, InvalidLabel={invalid_label_count} | "
                    f"Elapsed: {elapsed / 60:.1f} min",
                    file=sys.stdout,
                )

        exp_duration = (time.time() - start_time) / 60.0

        for fuzzy in (False, True):
            metrics = seqeval.compute(
                predictions=(pred_entities_fuzzy if fuzzy else pred_entities_exact),
                references=true_entities,
                scheme="IOB2", mode="strict", zero_division=0
            )
            exp_metrics.append({
                "fuzzy_mode": fuzzy,
                "precision": metrics["overall_precision"],
                "recall": metrics["overall_recall"],
                "f1": metrics["overall_f1"],
                "accuracy": metrics["overall_accuracy"],
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

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for fuzzy in (False, True):
        rows = [m for m in exp_metrics if m["fuzzy_mode"] == fuzzy]

        precision_mean, precision_std = mean_std([m["precision"] for m in rows])
        recall_mean, recall_std =       mean_std([m["recall"] for m in rows])
        f1_mean, f1_std =               mean_std([m["f1"] for m in rows])
        accuracy_mean, accuracy_std =   mean_std([m["accuracy"] for m in rows])
        context_not_in_input_mean, context_not_in_input_std =               mean_std([m["context_not_in_input"] for m in rows])
        context_not_in_input_rate_mean, context_not_in_input_rate_std =     mean_std([m["context_not_in_input_rate"] for m in rows])
        entity_not_in_context_mean, entity_not_in_context_std =             mean_std([m["entity_not_in_context"] for m in rows])
        entity_not_in_context_rate_mean, entity_not_in_context_rate_std =   mean_std([m["entity_not_in_context_rate"] for m in rows])
        fuzzy_helped_mean, fuzzy_helped_std =           mean_std([m["fuzzy_helped"] for m in rows])
        fuzzy_helped_rate_mean, fuzzy_helped_rate_std = mean_std([m["fuzzy_helped_rate"] for m in rows])
        exact_match_mean, exact_match_std =             mean_std([m["exact_match"] for m in rows])
        exact_match_rate_mean, exact_match_rate_std =   mean_std([m["exact_match_rate"] for m in rows])
        format_invalid_mean, format_invalid_std =       mean_std([m["format_invalid"] for m in rows])
        format_invalid_rate_mean, format_invalid_rate_std = mean_std([m["format_invalid_rate"] for m in rows])
        invalid_label_mean, invalid_label_std =             mean_std([m["invalid_label"] for m in rows])
        invalid_label_rate_mean, invalid_label_rate_std =   mean_std([m["invalid_label_rate"] for m in rows])
        elapsed_mean, elapsed_std = mean_std([m["elapsed_minute"] for m in rows])

        all_results.append({
            "model": model_name,
            "batch_size": BATCH_SIZE,
            "fuzzy_mode": fuzzy,
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

df = pd.DataFrame(all_results)

results_path = f"/home/stulcrad/master_thesis/Experiment_results/CoNLL/Context-Based/Csv/grammar_baseline_xgrammar_{BATCH_SIZE}_BS_conll2003.csv"
txt_path = results_path.replace(".csv", ".txt").replace("/Csv/", "/Txt/")

import os
os.makedirs(os.path.dirname(results_path), exist_ok=True)
os.makedirs(os.path.dirname(txt_path), exist_ok=True)
df.to_csv(results_path, index=False)
with open(txt_path, "w") as f:
    f.write(df.to_string(index=False))
print(f"\nResults saved to {results_path} and {txt_path}")
