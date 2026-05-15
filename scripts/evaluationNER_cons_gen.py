import sys
import time
import statistics
from typing import List, Tuple
import pandas as pd
import evaluate
from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.utils_functions import (
    generate_markup, validate_reconstruction, 
    spans_to_bio_tags, parse_spans_from_tagged_output,
    mean_std, to_pct, format_pm
)
from utils.TokTrie import build_toktrie_from_tokenizer
from utils.TrieSpanConstrainedProcessor import TrieSpanConstrainedProcessor
from utils.TrieSpanConstrainedProcessorTokenAware import TrieSpanConstrainedProcessorTokenAware

from utils.system_prompts import SYSTEM_PROMPT_CONSTR_GEN

# -------------------------
# Evaluation configuration
# -------------------------
# MAX_EXAMPLES = 1280
# N_ITERS = 1
MAX_EXAMPLES = 250
N_ITERS = 5
EVAL_INTERVAL = 10
# Single batch size per run. You can override from CLI:
# python evaluationNER_cons_gen.py 5
BATCH_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
if BATCH_SIZE > 5:
    EVAL_INTERVAL = 5

MODEL_NAMES = ['google/gemma-3-4b-it', 'Qwen/Qwen3-8B', 'meta-llama/Llama-3.1-8B-Instruct']

DO_SAMPLES = [False, True]
TEMPERATURE = 0.2
MAX_NEW_TOKENS = 32578

# Evaluate both decoding modes in one run.
EVAL_MODES = ["unconstrained", "constrained"]

# Processor class is only used for constrained mode.
PROCESSOR_CLASSES = ["whole_sequence", "token_aware"]

# Load the seqeval metric for span-level evaluation
seqeval = evaluate.load("seqeval")

dataset = load_dataset("lhoestq/conll2003", split="test")

results = []

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

labels_for_constrained = ["PER", "LOC", "ORG", "MISC"]

for model_name in MODEL_NAMES:
    print(f"\nLoading model/tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        # quantization_config=quantization_config,
    )

    batch_size = BATCH_SIZE
    print(f"Batch size: {batch_size}")

    for do_sample in DO_SAMPLES:
        sampling_strategy = "sampling" if do_sample else "greedy"

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
                    sampled_dataset = dataset.shuffle(seed=42 + exp_id).select(range(MAX_EXAMPLES))

                    start_time = time.time()
                    gold_sequences: List[List[str]] = []
                    pred_sequences: List[List[str]] = []
                    wrong_text_count = 0
                    all_entities_wrongly_unaligned = 0
                    unaligned_entity_count = 0
                    total_predictions = 0
                    total_batches = (len(sampled_dataset) + batch_size - 1) // batch_size

                    toktrie = None
                    if eval_mode == "constrained":
                        toktrie = build_toktrie_from_tokenizer(tokenizer)

                    for batch_idx in tqdm(range(total_batches), desc=f"exp {exp_id + 1}/{N_ITERS}", file=sys.stdout):
                        start_idx = batch_idx * batch_size
                        end_idx = min((batch_idx + 1) * batch_size, len(sampled_dataset))
                        batch = sampled_dataset.select(range(start_idx, end_idx))

                        batch_tokens = []
                        batch_gold_tags = []
                        for example in batch:
                            batch_tokens.extend(example["tokens"])
                            batch_gold_tags.extend([id2label[tag_id] for tag_id in example["ner_tags"]])

                        input_text = " ".join(batch_tokens)
                        processor = None
                        if eval_mode == "constrained":
                            if processor_class == "token_aware":
                                processor = TrieSpanConstrainedProcessorTokenAware(
                                    labels_for_constrained,
                                    input_text,
                                    tokenizer,
                                    toktrie,
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
                            system_prompt=SYSTEM_PROMPT_CONSTR_GEN,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=do_sample,
                            temperature=TEMPERATURE,
                        )

                        parsed = parse_spans_from_tagged_output(generated, set(labels_for_constrained))
                        total_predictions += parsed["span_count"]
                        exact_copy_ok = validate_reconstruction(parsed["reconstructed_text"], input_text)

                        if not exact_copy_ok:
                            wrong_text_count += 1
                            if eval_mode == "constrained":
                                print(f"\n\n===== Warning in exp {exp_id + 1}, batch {batch_idx + 1} =====")
                                print(f"Original text: \n{input_text}")
                                print(f"Reconstructed text: \n{parsed['reconstructed_text']}")
                                print(f"Generated markup: \n{generated}\n\n")
                            pred_tags = ["O"] * len(batch_tokens)
                            all_entities_wrongly_unaligned += parsed["span_count"]
                        else:
                            pred_tags, unalign_count = spans_to_bio_tags(
                                tokens=batch_tokens,
                                entities=parsed["entities"],
                                valid_labels=set(labels_for_constrained),
                            )
                            unaligned_entity_count += unalign_count
                            all_entities_wrongly_unaligned += unalign_count

                        gold_sequences.append(batch_gold_tags)
                        pred_sequences.append(pred_tags)

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
                                f"exp {exp_id + 1}/{N_ITERS}, batch {batch_idx + 1}/{total_batches} "
                                f"F1={partial['overall_f1']:.4f}, wrong_text={wrong_text_count}, unaligned_ent_count={unaligned_entity_count}, elapsed={elapsed:.1f}m"
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
                wrong_text_count_mean, wrong_text_count_std = mean_std([m["wrong_text_count"] for m in exp_metrics])
                wrong_text_rate_mean, wrong_text_rate_std = mean_std([m["wrong_text_rate"] for m in exp_metrics])
                unaligned_entity_count_mean, unaligned_entity_count_std = mean_std([m["unaligned_entity_count"] for m in exp_metrics])
                unaligned_entity_rate_mean, unaligned_entity_rate_std = mean_std([m["unaligned_entity_rate"] for m in exp_metrics])
                all_entities_wrongly_unaligned_mean, all_entities_wrongly_unaligned_std = mean_std([m["all_entities_wrongly_unaligned"] for m in exp_metrics])
                all_entities_wrongly_unaligned_rate_mean, all_entities_wrongly_unaligned_rate_std = mean_std([m["all_entities_wrongly_unaligned_rate"] for m in exp_metrics])
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
                    "wrong_text_count_avg": round(wrong_text_count_mean, 3),
                    "wrong_text_count_std": round(wrong_text_count_std, 3),
                    "wrong_text_rate_pct": round(to_pct(wrong_text_rate_mean), 2),
                    "wrong_text_rate_std_pct": round(to_pct(wrong_text_rate_std), 2),
                    "wrong_text_rate_report": format_pm(to_pct(wrong_text_rate_mean), to_pct(wrong_text_rate_std)),
                    "unaligned_entity_count_avg": round(unaligned_entity_count_mean, 3),
                    "unaligned_entity_count_std": round(unaligned_entity_count_std, 3),
                    "unaligned_entity_rate_pct": round(to_pct(unaligned_entity_rate_mean), 2),
                    "unaligned_entity_rate_std_pct": round(to_pct(unaligned_entity_rate_std), 2),
                    "unaligned_entity_rate_report": format_pm(to_pct(unaligned_entity_rate_mean), to_pct(unaligned_entity_rate_std)),
                    "all_entities_wrongly_unaligned_avg": round(all_entities_wrongly_unaligned_mean, 3),
                    "all_entities_wrongly_unaligned_std": round(all_entities_wrongly_unaligned_std, 3),
                    "all_entities_wrongly_unaligned_rate_pct": round(to_pct(all_entities_wrongly_unaligned_rate_mean), 2),
                    "all_entities_wrongly_unaligned_rate_std_pct": round(to_pct(all_entities_wrongly_unaligned_rate_std), 2),
                    "all_entities_wrongly_unaligned_rate_report": format_pm(to_pct(all_entities_wrongly_unaligned_rate_mean), to_pct(all_entities_wrongly_unaligned_rate_std)),
                    "elapsed_minute_avg": round(elapsed_mean, 3),
                    "elapsed_minute_std": round(elapsed_std, 3),
                })

    # Free GPU memory before loading next model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Save results to a DataFrame and CSV
results_df = pd.DataFrame(results)

# Optional persistence
results_path = f"/home/stulcrad/master_thesis/Experiment_results/CoNLL/Constrained-Gen/Csv/hf_all_configs_eval_{BATCH_SIZE}_BS_conll2003.csv"
txt_path = results_path.replace("Csv", "Txt").replace(".csv", ".txt")

import os
os.makedirs(os.path.dirname(results_path), exist_ok=True)
os.makedirs(os.path.dirname(txt_path), exist_ok=True)

results_df.to_csv(results_path, index=False)

with open(txt_path, "w") as f:
    f.write(results_df.to_string(index=False))

print(f"\nResults saved to {results_path} and {txt_path}")
