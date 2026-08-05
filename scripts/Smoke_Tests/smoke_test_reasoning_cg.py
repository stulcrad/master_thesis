import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.system_prompts import SYSTEM_PROMPT_CONSTR_GEN, SYSTEM_PROMPT_CONSTR_GEN_LEGALQA_TEMPLATE, SYSTEM_PROMPT_CONSTR_GEN_TOXIC_SPANS
from utils.utils_functions import (
    generate_markup, validate_reconstruction,
    parse_spans_from_tagged_output, example_to_tokens,
)
from utils.TokTrie import build_toktrie_from_tokenizer
from utils.TrieSpanConstrainedProcessor import TrieSpanConstrainedProcessor
from utils.TrieSpanConstrainedProcessorTokenAware import TrieSpanConstrainedProcessorTokenAware
from utils.model_reasoning_utils import reasoning_ended

from argparse import ArgumentParser

# Arguments and configuration
parser = ArgumentParser()
parser.add_argument("--model", type=str, default='openai/gpt-oss-20b', help="Model ID to use for reasoning tests.")
model_id = parser.parse_args().model

N_SMOKE_EXAMPLES = 5
SMOKE_MAX_NEW_TOKENS = 6000

def smoke_test_examples(dataset_name, examples, labels_for_constrained,
                         reasoning_model, reasoning_end_marker, toktrie,
                         tokenizer, model, max_new_tokens=SMOKE_MAX_NEW_TOKENS):
    """Run unconstrained + both constrained processor classes over a handful of
    (input_text, system_prompt) examples. Returns a list of diagnostic row dicts;
    nothing is written to disk."""
    rows = []
    for idx, (input_text, system_prompt) in enumerate(examples):
        for eval_mode in ["unconstrained", "constrained"]:
            processor_class_options = ["whole_sequence", "token_aware"] if eval_mode == "constrained" else [None]
            for processor_class in processor_class_options:
                processor = None
                if eval_mode == "constrained":
                    cls = TrieSpanConstrainedProcessorTokenAware if processor_class == "token_aware" else TrieSpanConstrainedProcessor
                    processor = cls(
                        labels_for_constrained, input_text, tokenizer, toktrie,
                        reasoning_model=reasoning_model,
                        reasoning_end_marker=reasoning_end_marker,
                        reasoning_ended=reasoning_ended,
                        model_eos_token_id=model.generation_config.eos_token_id,
                        tokenizer_eos_token_id=tokenizer.eos_token_id,
                    )
                    
                repetition_penalty = None
                if "Qwen" in model_id:
                    repetition_penalty = 1.3 if reasoning_model else None
                if "gemma" in model_id:
                    repetition_penalty = 1.15 if reasoning_model else None

                gen_stats = {}
                generated, num_output_tokens, generation_seconds = generate_markup(
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    eval_model=eval_mode,
                    input_text=input_text,
                    system_prompt=system_prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=True if reasoning_model else False,
                    temperature=0.2,
                    reasoning_model=reasoning_model,
                    reasoning_effort='low',
                    reasoning_end_marker=reasoning_end_marker,
                    stats_out=gen_stats,
                    repetition_penalty=repetition_penalty
                )

                parsed = parse_spans_from_tagged_output(generated, set(labels_for_constrained))
                exact_copy_ok = validate_reconstruction(parsed["reconstructed_text"], input_text)
                n_reason = gen_stats.get("num_reasoning_tokens", 0)
                n_answer = gen_stats.get("num_answer_tokens", num_output_tokens)

                print(f"\nOriginal input:\n {input_text}")
                print(f"\nReasoning output:\n {gen_stats.get('reasoning_text', 'NO REASONING OUTPUT')}")
                print(f"\nAnswer output:\n {gen_stats.get('answer_text', 'NO ANSWER OUTPUT')}")
                print(f"\nReconstructed text:\n {parsed['reconstructed_text']}")
                print(f"\nParsed spans:\n {parsed['entities']}")

                rows.append({
                    "dataset": dataset_name,
                    "example_idx": idx,
                    "eval_mode": eval_mode,
                    "processor_class": processor_class or "n|a",
                    "wrong_text": 0 if exact_copy_ok else 1,
                    "span_count": parsed["span_count"],
                    "num_output_tokens": num_output_tokens,
                    "num_reasoning_tokens": n_reason,
                    "num_answer_tokens": n_answer,
                    "tokens_sum_ok": (n_reason + n_answer) == num_output_tokens,
                    "found_reasoning_end": gen_stats.get("found_reasoning_end"),
                    "reasoning_text": gen_stats.get("reasoning_text"),
                    "answer_text": gen_stats.get("answer_text"),
                    "generation_seconds": generation_seconds,
                })
    return rows

def run_conll_2003(tokenizer, model, reasoning_model, reasoning_end_marker):
    ner_raw = load_dataset("lhoestq/conll2003", split="test").shuffle(seed=42).select(range(N_SMOKE_EXAMPLES))
    ner_labels = ["PER", "LOC", "ORG", "MISC"]
    ner_examples = [(" ".join(ex["tokens"]), SYSTEM_PROMPT_CONSTR_GEN) for ex in ner_raw]

    ner_toktrie = build_toktrie_from_tokenizer(tokenizer)
    ner_rows = smoke_test_examples(
        "NER", ner_examples, ner_labels,
        reasoning_model, reasoning_end_marker, ner_toktrie,
        tokenizer, model,
    )
    ner_smoke_df = pd.DataFrame(ner_rows)
    return ner_smoke_df

def run_toxicspans(tokenizer, model, reasoning_model, reasoning_end_marker):
    toxic_raw = load_dataset("heegyu/toxic-spans", split="test").shuffle(seed=42).select(range(N_SMOKE_EXAMPLES * 3))
    toxic_labels = ["TOXIC"]
    toxic_examples = []
    for ex in toxic_raw:
        tokens = example_to_tokens(ex['text_of_post'])
        if not tokens:
            continue
        toxic_examples.append((" ".join(tokens), SYSTEM_PROMPT_CONSTR_GEN_TOXIC_SPANS))
        if len(toxic_examples) == N_SMOKE_EXAMPLES:
            break

    toxic_tokrie = build_toktrie_from_tokenizer(tokenizer)
    toxic_rows = smoke_test_examples(
        "ToxicSpans", toxic_examples, toxic_labels,
        reasoning_model, reasoning_end_marker, toxic_tokrie,
        tokenizer, model,
    )
    toxic_smoke_df = pd.DataFrame(toxic_rows)
    return toxic_smoke_df

def run_legalqa(tokenizer, model, reasoning_model, reasoning_end_marker):
    legalqa_raw = load_dataset("isaacus/LegalQAEval", split="test").shuffle(seed=42).select(range(N_SMOKE_EXAMPLES))
    legalqa_labels = ["ANSWER"]
    legalqa_examples = []
    for ex in legalqa_raw:
        tokens = example_to_tokens(ex["text"])
        input_text = " ".join(tokens)
        system_prompt = SYSTEM_PROMPT_CONSTR_GEN_LEGALQA_TEMPLATE.format(question=ex["question"])
        legalqa_examples.append((input_text, system_prompt))

    legalqa_toktrie = build_toktrie_from_tokenizer(tokenizer)
    legalqa_rows = smoke_test_examples(
        "LegalQA", legalqa_examples, legalqa_labels,
        reasoning_model, reasoning_end_marker, legalqa_toktrie, tokenizer, model,
    )
    legalqa_smoke_df = pd.DataFrame(legalqa_rows)
    return legalqa_smoke_df

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype='auto'
        )

    REASONING_END_MARKER_STR = {
        'google/gemma-4-E4B-it': "<channel|>",
        'google/gemma-4-26B-A4B-it': "<channel|>",
        'google/gemma-4-31B-it': "<channel|>",
        'Qwen/Qwen3.5-4B': "</think>",
        'Qwen/Qwen3.5-9B': "</think>",
        'Qwen/Qwen3.5-27B': "</think>",
        'Qwen/Qwen3.6-35B-A3B': "</think>",
        'Qwen/Qwen3.6-27B': "</think>",
        'openai/gpt-oss-20b': "<|channel|>final<|message|>",
        'openai/gpt-oss-120b': "<|channel|>final<|message|>",
    }
    reasoning_model = model_id in REASONING_END_MARKER_STR
    reasoning_end_marker = (
        tokenizer(REASONING_END_MARKER_STR[model_id], add_special_tokens=False).input_ids
        if reasoning_model else None
    )

    print(f"model_id={model_id}  reasoning_model={reasoning_model}  reasoning_end_marker={reasoning_end_marker}")

    ner_smoke_df = run_conll_2003(tokenizer, model, reasoning_model, reasoning_end_marker)
    toxic_smoke_df = run_toxicspans(tokenizer, model, reasoning_model, reasoning_end_marker)
    legalqa_smoke_df = run_legalqa(tokenizer, model, reasoning_model, reasoning_end_marker)

    smoke_test_df = pd.concat([ner_smoke_df, toxic_smoke_df, legalqa_smoke_df], ignore_index=True)

    node_name = os.environ.get("SLURMD_NODENAME", os.uname().nodename)
    save_path = f"/home/stulcrad/master_thesis/Experiment_results/Smoke_Tests/CG/smoke_test_results_{model_id.replace('/', '_')}_{node_name}.csv"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    smoke_test_df.to_csv(save_path, index=False)
