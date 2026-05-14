import sys
import os
import ast
import time
import statistics
from typing import List, Tuple
import re

import evaluate
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.utils_functions import (
    generate_markup, validate_reconstruction, 
    spans_to_bio_tags, parse_spans_from_tagged_output
)
from utils.TokTrie import build_toktrie_from_tokenizer
from utils.TrieSpanConstrainedProcessor import TrieSpanConstrainedProcessor
from utils.TrieSpanConstrainedProcessorTokenAware import TrieSpanConstrainedProcessorTokenAware
from utils.system_prompts import SYSTEM_PROMPT_CONSTR_GEN_TOXIC_SPANS

from huggingface_hub import login
login(token="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")

N_ITERS = 1
EVAL_INTERVAL = 10
BATCH_SIZE = 1

MODEL_NAMES = ["google/gemma-3-4b-it", "Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B-Instruct"]
# MODEL_NAMES = ["google/gemma-3-4b-it"]

DO_SAMPLES = [False, True]
# DO_SAMPLES = [True]
TEMPERATURE = 0.2
MAX_NEW_TOKENS = 32578

EVAL_MODES = ["unconstrained", "constrained"]
PROCESSOR_CLASSES = ["whole_sequence", "token_aware"]

labels_for_constrained = ["TOXIC"]

seqeval = evaluate.load("seqeval")

def parse_position(raw_position: str) -> List[int]:
    """
    Return a plain Python list of int char indices.
    """
    parsed = ast.literal_eval(raw_position)
    return [int(x) for x in parsed]
    
def chars_to_spans(char_indices: List[int]) -> List[Tuple[int, int]]:
    """
    Merge sorted individual char indices into (start, end) tuples.

    Example:
        [7, 8, 9, 10] -> [(7, 11)]
       [0,1,2,3,4,5,15,16,17] -> [(0, 6), (15, 18)]
    """
    if not char_indices:
        return []
    
    indices = sorted(set(char_indices))
    spans: List[Tuple[int, int]] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
          prev = idx
        else:
            spans.append((start, prev + 1))
            start = prev = idx
    spans.append((start, prev + 1))
    return spans

def tokenize_with_offsets(text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    tokens, offsets = [] , []
    i, n = 0, len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        j = i
        while j < n and not text[j].isspace():
            j += 1
        tokens.append(text[i:j])
        offsets.append((i, j))
        i = j
    return tokens, offsets

def spans_to_bio(
        token_offsets: List[Tuple[int, int]],
        spans: List[Tuple[int, int, str]]
) -> List[str]:
    bio_tags = ["O"] * len(token_offsets)
    for span_start, span_end, label in spans:
        covered = [
            i for i, (tok_start, tok_end) in enumerate(token_offsets)
            if max(tok_start, span_start) < min(tok_end, span_end)
        ]
        if not covered:
            continue
        bio_tags[covered[0]] = f"B-{label}"
        for i in covered[1:]:
            bio_tags[i] = f"I-{label}"
    return bio_tags

def example_to_tokens_and_tags(example) -> Tuple[List[str], List[str]]:
    text = example['text_of_post']
    positions = parse_position(example['position'])
    spans = [(s, e, "TOXIC") for s, e in chars_to_spans(positions)]
    tokens, offsets = tokenize_with_offsets(text)
    bio_tags = spans_to_bio(offsets, spans)
    return tokens, bio_tags

print("Loading heegyu/toxic-spans test split...")
raw = load_dataset("heegyu/toxic-spans", split="test")
print(f"Examples in test split: {len(raw)}")

MAX_EXAMPLES = 50
print(f"Max examples per iteration: {MAX_EXAMPLES}")

def mean_std(values):
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def to_pct(v): return v * 100.0
def format_pm(m, s): return f"{m:.2f} ± {s:.2f}"

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
                    gold_sequences: List[List[str]] = []
                    pred_sequences: List[List[str]] = []
                    wrong_text_count = 0
                    all_entities_wrongly_unaligned = 0
                    unaligned_entity_count = 0
                    total_predictions = 0
                    total_batches = (len(sampled) + batch_size - 1) // batch_size

                    toktrie = None
                    if eval_mode == "constrained":
                        toktrie = build_toktrie_from_tokenizer(tokenizer)

                    for batch_idx in tqdm(range(total_batches),
                                          desc=f"exp {exp_id+1}/{N_ITERS}", file=sys.stdout):
                        start_idx = batch_idx * batch_size
                        end_idx = min((batch_idx + 1) * batch_size, len(sampled))
                        batch = sampled.select(range(start_idx, end_idx))

                        batch_tokens: List[str] = []
                        batch_gold_tags: List[str] = []
                        for example in batch:
                            toks, tags = example_to_tokens_and_tags(example)
                            batch_tokens.extend(toks)
                            batch_gold_tags.extend(tags)

                        input_text = " ".join(batch_tokens)

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
                                print(f"\n\n===== Warning in exp {exp_id+1}, batch {batch_idx+1} =====")
                                print(f"Original:      {input_text[:120]!r}")
                                print(f"Reconstructed: {parsed['reconstructed_text'][:120]!r}")
                            pred_tags = ["O"] * len(batch_tokens)
                            all_entities_wrongly_unaligned += parsed["span_count"]
                        else:
                            pred_tags, unalign_count = spans_to_bio_tags(
                                tokens=batch_tokens,
                                entities=parsed['entities'],
                                valid_labels=set(labels_for_constrained),
                            )
                            unaligned_entity_count += unalign_count
                            all_entities_wrongly_unaligned += unalign_count

                        gold_sequences.append(batch_gold_tags)
                        pred_sequences.append(pred_tags)

                        print(f"\nOriginal text: \n{input_text}\n")
                        print(f"Generated markup: \n{generated}\n")
                        print(f"Parsed entities: {parsed['entities']}\n")
                        print(f"Predicted tags: \n{pred_tags}\n")
                        print(f"Gold tags: \n{batch_gold_tags}\n")

                        if (batch_idx + 1) % EVAL_INTERVAL == 0:
                            partial = seqeval.compute(
                                predictions=pred_sequences,
                                references=gold_sequences,
                                scheme="IOB2",
                                mode="strict",
                                zero_division=0,
                            )
                            elapsed = (time.time() - start_time) / 60.0
                            tqdm.write(
                                f"[{model_name} | {sampling_strategy} | {eval_mode} | {config_label} | bs={batch_size}] "
                                f"exp {exp_id+1}/{N_ITERS}, batch {batch_idx+1}/{total_batches} "
                                f"F1={partial['overall_f1']:.4f}, wrong_text={wrong_text_count}, "
                                f"unaligned_ent_count={unaligned_entity_count}, elapsed={elapsed:.1f}m"
                            )

                    metrics = seqeval.compute(
                        predictions=pred_sequences,
                        references=gold_sequences,
                        scheme="IOB2",
                        mode="strict",
                        zero_division=0,
                    )

                    elapsed_min = (time.time() - start_time) / 60.0
                    exp_metrics.append({
                        "precision": metrics["overall_precision"],
                        "recall": metrics["overall_recall"],
                        "f1": metrics["overall_f1"],
                        "accuracy": metrics["overall_accuracy"],
                        "wrong_text_count": wrong_text_count,
                        "wrong_text_rate": wrong_text_count / max(total_batches, 1),
                        "unaligned_entity_count": unaligned_entity_count,
                        "unaligned_entity_rate": unaligned_entity_count / max(total_predictions, 1),
                        "all_entities_wrongly_unaligned": all_entities_wrongly_unaligned,
                        "all_entities_wrongly_unaligned_rate": all_entities_wrongly_unaligned / max(total_predictions, 1),
                        "elapsed_minute": elapsed_min,
                    })

                precision_mean, precision_std = mean_std([m["precision"] for m in exp_metrics])
                recall_mean, recall_std = mean_std([m["recall"] for m in exp_metrics])
                f1_mean, f1_std = mean_std([m["f1"] for m in exp_metrics])
                accuracy_mean, accuracy_std = mean_std([m["accuracy"] for m in exp_metrics])
                wt_mean, wt_std = mean_std([m["wrong_text_count"] for m in exp_metrics])
                wt_rate_mean, wt_rate_std = mean_std([m["wrong_text_rate"] for m in exp_metrics])
                ua_mean, ua_std = mean_std([m["unaligned_entity_count"] for m in exp_metrics])
                ua_rate_mean, ua_rate_std = mean_std([m["unaligned_entity_rate"] for m in exp_metrics])
                aau_mean, aau_std = mean_std([m["all_entities_wrongly_unaligned"] for m in exp_metrics])
                aau_rate_mean, aau_rate_std = mean_std([m["all_entities_wrongly_unaligned_rate"] for m in exp_metrics])
                elapsed_mean, elapsed_std = mean_std([m["elapsed_minute"] for m in exp_metrics])

                results.append({
                    "model": model_name,
                    "sampling_strategy": sampling_strategy,
                    "do_sample": do_sample,
                    "eval_mode": eval_mode,
                    "processor_class": config_label,
                    "batch_size": batch_size,
                    "n_iters": N_ITERS,
                    "precision_pct": round(to_pct(precision_mean), 2),
                    "precision_std_pct": round(to_pct(precision_std), 2),
                    "precision_report": format_pm(to_pct(precision_mean), to_pct(precision_std)),
                    "recall_pct": round(to_pct(recall_mean), 2),
                    "recall_std_pct": round(to_pct(recall_std), 2),
                    "recall_report": format_pm(to_pct(recall_mean), to_pct(recall_std)),
                    "f1_pct": round(to_pct(f1_mean), 2),
                    "f1_std_pct": round(to_pct(f1_std), 2),
                    "f1_report": format_pm(to_pct(f1_mean), to_pct(f1_std)),
                    "accuracy_pct": round(to_pct(accuracy_mean), 2),
                    "accuracy_std_pct": round(to_pct(accuracy_std), 2),
                    "accuracy_report": format_pm(to_pct(accuracy_mean), to_pct(accuracy_std)),
                    "wrong_text_count_avg": round(wt_mean, 3),
                    "wrong_text_count_std": round(wt_std, 3),
                    "wrong_text_rate_pct": round(to_pct(wt_rate_mean), 2),
                    "wrong_text_rate_std_pct": round(to_pct(wt_rate_std), 2),
                    "wrong_text_rate_report": format_pm(to_pct(wt_rate_mean), to_pct(wt_rate_std)),
                    "unaligned_entity_count_avg": round(ua_mean, 3),
                    "unaligned_entity_count_std": round(ua_std, 3),
                    "unaligned_entity_rate_pct": round(to_pct(ua_rate_mean), 2),
                    "unaligned_entity_rate_std_pct": round(to_pct(ua_rate_std), 2),
                    "unaligned_entity_rate_report": format_pm(to_pct(ua_rate_mean), to_pct(ua_rate_std)),
                    "all_entities_wrongly_unaligned_avg": round(aau_mean, 3),
                    "all_entities_wrongly_unaligned_std": round(aau_std, 3),
                    "all_entities_wrongly_unaligned_rate_pct": round(to_pct(aau_rate_mean), 2),
                    "all_entities_wrongly_unaligned_rate_std_pct": round(to_pct(aau_rate_std), 2),
                    "all_entities_wrongly_unaligned_rate_report": format_pm(to_pct(aau_rate_mean), to_pct(aau_rate_std)),
                    "elapsed_minute_avg": round(elapsed_mean, 3),
                    "elapsed_minute_std": round(elapsed_std, 3),
                })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

results_df = pd.DataFrame(results)

results_path = (
    f"/home/stulcrad/master_thesis/NER_results/ToxicSpans/Constrained-Gen/Csv/"
    f"hf_all_configs_eval_{BATCH_SIZE}_BS_toxic_spans.csv"
)
txt_path = results_path.replace("Csv", "Txt").replace(".csv", ".txt")

os.makedirs(os.path.dirname(results_path), exist_ok=True)
os.makedirs(os.path.dirname(txt_path), exist_ok=True)

results_df.to_csv(results_path, index=False)

with open(txt_path, "w") as f:
    f.write(results_df.to_string(index=False))

print(f"\nResults saved to {results_path} and {txt_path}")
