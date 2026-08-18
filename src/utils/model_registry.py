"""Single source of truth for the model set used across every evaluation script.

Why this exists
---------------
Unifies the model list so that all evaluation scripts share the same set, and so that the SLURM array
index is defined in one place instead of being duplicated in bash. Also centralizes the reasoning
phase markers, which are model-specific and must be kept in sync with the reasoning aware processor.

What lives here vs. on the command line
---------------------------------------
- **Capabilities** (properties of the model) live here: does it reason, what marks the end of its
  reasoning phase, can that phase be switched off, does its chat template accept `reasoning_effort`.
- **Run knobs** (properties of the experiment) are CLI arguments: `repetition_penalty`,
  `reasoning_effort`'s value, `enable_thinking`.
- **Sampling presets** are the exception: they are vendor recommendations that differ per model
  *and* per thinking mode, so keeping them here avoids retyping `--top-p/--top-k` on every sbatch
  line. CLI arguments override them; resolution order is CLI -> preset -> the model's own
  generation_config.json.

Two model sets
--------------
- `MODELS`   -- the publication run set. Its order defines the SLURM array index.
- `LEGACY_MODELS` -- the three thesis models. Already evaluated (results live in the
  frozen `Experiment_results/`), so they are NOT re-run and are deliberately absent from
  the array. They stay registered so `get()` still resolves them.

All runs target A100/H200; V100 is not used (too slow on the larger models, and the
Qwen3.5+ linear-attention kernels do not build for sm_70).

SLURM array indexing
--------------------
Batch scripts should read the list from here instead of duplicating it in bash:

    MODEL_NAME=$(python -m utils.model_registry --id "$SLURM_ARRAY_TASK_ID")
"""
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SamplingPreset:
    """Vendor-recommended decoding settings for one model in one thinking mode.

    Every field defaults to None, meaning "do not pass this kwarg to generate()", which
    lets the model's own generation_config.json apply. `presence_penalty` and
    `frequency_penalty` are deliberately absent: HF `generate()` does not implement them
    (verified on transformers 5.14.1), so a vendor recommendation that includes them
    cannot be reproduced on this stack -- it would be silently dropped with a
    "generation flags are not valid and may be ignored" warning.
    """
    do_sample: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None

    def as_kwargs(self) -> Dict[str, Any]:
        """Only the fields that were actually set, ready to splat into generate()."""
        return {k: v for k, v in asdict(self).items() if v is not None}


# -- Vendor presets, from the model cards ------------------------------------------------
# Gemma 4 publishes a single recommendation (no separate thinking/instruct split).
_GEMMA_SAMPLING = SamplingPreset(do_sample=True, temperature=1.0, top_p=0.95, top_k=64)

# Qwen publishes two, selected by thinking mode. Both also recommend
# repetition_penalty=1.0 -- which our smoke tests contradict (1.3 was needed to stop the
# self-verification loop), so repetition_penalty stays a CLI knob and is swept, not fixed.
# The instruct preset's presence_penalty=1.5 is unsupported by HF generate() and dropped.
_QWEN_SAMPLING_THINKING = SamplingPreset(do_sample=True, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
_QWEN_SAMPLING_INSTRUCT = SamplingPreset(do_sample=True, temperature=0.7, top_p=0.80, top_k=20, min_p=0.0)

_GPT_OSS_SAMPLING = SamplingPreset(do_sample=True, temperature=1.0, top_p=1.0, top_k=0)


@dataclass(frozen=True)
class ModelSpec:
    """Everything the evaluation harness needs to know about one model.

    Fields
    ------
    - model_id: HuggingFace repo id, as passed to `from_pretrained`.

    - reasoning_end_marker: the STRING marking the reasoning->answer boundary. It is
      tokenized per-model at runtime and matched by token IDs, never by string search --
      several of these are special tokens and vanish under `skip_special_tokens=True`.
      None means the model has no reasoning mode; `reasoning` is derived from it.
    
    - reasoning_start_marker: the STRING that OPENS the reasoning block, but only for
      families where the MODEL has to emit it. Leave None when the chat template
      already opens the block (Qwen3.5+ ends its generation prompt with `<think>\n`)
      or when the end marker doubles as the answer opener (Harmony's
      `<|channel|>final<|message|>`). 
      This is per MODEL, not per family: Qwen3-8B ends its prompt at
      `<|im_start|>assistant\n` with no `<think>`, so it DOES need one, while
      Qwen3.5 must not have one. Verify with the diagnostic cells in
      Notebooks/CG_publication/reasoning_aware_CG.ipynb before setting it.
      Gemma-4's E2B/E4B prompt ends at `<|turn>model\n` and the model must emit `<|channel>`
      itself; they sometimes skip the thought block entirely and answer
      directly. Without this marker that case is indistinguishable from truncation,
      and a perfectly good answer gets discarded as reasoning (see `_split_reasoning_answer`).
    
    - reasoning_off_supported: whether the reasoning phase can actually be disabled, for
      the reasoning-ON/OFF ablation. Default False -- only set True once verified for that
      model.
    
    - reasoning_effort_levels: the values this model's chat template accepts for
      `reasoning_effort`, or () when it takes none. This must be data rather than a
      boolean because the families disagree: GPT-OSS/Harmony uses low/medium/high,
      Qwen3.8 uses low/medium/**xhigh** (xhigh is its default). A single
      low|medium|high choice list would silently reject Qwen's default and let
      `high` through to a model that does not know it.
      `supports_reasoning_effort` is derived from this so the two cannot drift.
    
    - sampling_thinking / sampling_instruct: vendor presets, selected by thinking mode.
    
    - notes: free text; includes which hardware the model has actually been run on.
    """
    model_id: str
    reasoning_end_marker: Optional[str] = None
    reasoning_start_marker: Optional[str] = None
    reasoning_off_supported: bool = False
    reasoning_effort_levels: Tuple[str, ...] = ()
    sampling_thinking: SamplingPreset = SamplingPreset()
    sampling_instruct: SamplingPreset = SamplingPreset()
    notes: str = ""

    @property
    def supports_reasoning_effort(self) -> bool:
        """Derived, so the flag and the accepted values cannot disagree."""
        return bool(self.reasoning_effort_levels)

    @property
    def reasoning(self) -> bool:
        """A model reasons iff it has an end-of-reasoning marker. Derived so the two
        facts cannot drift apart."""
        return self.reasoning_end_marker is not None

    @property
    def short_name(self) -> str:
        """Filename-friendly name, matching `model_name.split('/')[-1]` used everywhere."""
        return self.model_id.split("/")[-1]

    def sampling(self, enable_thinking: bool) -> SamplingPreset:
        """The vendor preset for the given thinking mode."""
        return self.sampling_thinking if enable_thinking else self.sampling_instruct


# Marker strings, grouped so a typo in one family cannot silently affect another.
# All three were confirmed against real generations in the smoke tests.
_QWEN_MARKER = "</think>"
_GEMMA4_MARKER = "<channel|>"
_HARMONY_MARKER = "<|channel|>final<|message|>"

# Opening marker -- only needed where the MODEL emits it (see ModelSpec docs).
# Verified by rendering each chat template (Notebooks/CG_publication/reasoning_aware_CG.ipynb,
# "reasoning_start_marker diagnostic" cells):
#
#   gemma-4     prompt ends `<|turn>model\n`             -> model emits <|channel>
#   Qwen3-8B    prompt ends `<|im_start|>assistant\n`    -> model emits <think>
#   Qwen3.5+    prompt ends `<|im_start|>assistant\n<think>\n` -> template opens it
#   Harmony     end marker IS the answer opener          -> cannot be skipped
#
# So this is NOT a per-family constant: Qwen3-8B needs a start marker while
# Qwen3.5 must not have one. Check per model before setting it.
_GEMMA4_START = "<|channel>"
_QWEN_START = "<think>"


#: Publication run set. Index == SLURM_ARRAY_TASK_ID.
#:
#: Cut from 10 models to 5 on 2026-08-18. Reasoning effort turned out to be the
#: knob that matters, so the repetition-penalty sweep is dropped and the model set
#: is chosen to cover the three families that expose it (or deliberately do not).
MODELS: List[ModelSpec] = [
    # -- Gemma 4 -------------------------------------------------------------------------
    # Neither Gemma-4 model exposes reasoning_effort, so they carry the ON/OFF axis.
    ModelSpec(
        model_id="google/gemma-4-E2B-it",
        reasoning_end_marker=_GEMMA4_MARKER,
        reasoning_start_marker=_GEMMA4_START,
        reasoning_off_supported=True,
        sampling_thinking=_GEMMA_SAMPLING,
        sampling_instruct=_GEMMA_SAMPLING,
        notes="5.1B total / 2.3B effective. Cheap enough to run BOTH reasoning arms. "
              "Like E4B it must emit <|channel> itself and sometimes skips the thought "
              "block entirely -- hence reasoning_start_marker.",
    ),
    ModelSpec(
        model_id="google/gemma-4-31B-it",
        reasoning_end_marker=_GEMMA4_MARKER,
        reasoning_start_marker=_GEMMA4_START,
        reasoning_off_supported=True,
        sampling_thinking=_GEMMA_SAMPLING,
        sampling_instruct=_GEMMA_SAMPLING,
        notes="Run reasoning-OFF only: this is the row that sits beside Semin et al.'s "
              "Gemma-4-31B column (they disabled reasoning by default).",
    ),

    # -- Qwen 3.8 ------------------------------------------------------------------------
    # First Qwen generation with reasoning_effort.
    ModelSpec(
        model_id="Qwen/Qwen3.8-27B",
        reasoning_end_marker=_QWEN_MARKER,
        # Verified in Notebooks/CG_publication/reasoning_aware_CG.ipynb: prompt ends at
        # `<|im_start|>assistant<think>\n`, so the cannot have a start marker.
        reasoning_start_marker=None,
        reasoning_off_supported=True,
        reasoning_effort_levels=("low", "medium", "xhigh"),
        sampling_thinking=_QWEN_SAMPLING_THINKING,
        sampling_instruct=_QWEN_SAMPLING_INSTRUCT,
        notes="27B dense, BF16 (~54GB weights). Thinking CAN be disabled here, unlike "
              "the 2.4T variant, so an OFF arm is available if wanted.",
    ),

    # -- GPT-OSS / Harmony ---------------------------------------------------------------
    # Reasoning cannot be disabled: Harmony always emits a reasoning channel, so
    # reasoning_effort is the only verbosity control.
    ModelSpec(
        model_id="openai/gpt-oss-20b",
        sampling_thinking=_GPT_OSS_SAMPLING,
        reasoning_end_marker=_HARMONY_MARKER,
        reasoning_off_supported=False,
        reasoning_effort_levels=("low", "medium", "high"),
        notes="Smoke-tested on H200. Falls back to bf16 on A100.",
    ),
    ModelSpec(
        model_id="openai/gpt-oss-120b",
        sampling_thinking=_GPT_OSS_SAMPLING,
        reasoning_end_marker=_HARMONY_MARKER,
        reasoning_off_supported=False,
        reasoning_effort_levels=("low", "medium", "high"),
        notes="Smoke-tested on H200; H200 in practice.",
    ),

    # -- Qwen 3 (the matched-comparison row) ---------------------------------------------
    # Appended LAST on purpose: adding it anywhere else would renumber the SLURM
    # array indices that the submit scripts hardcode.
    ModelSpec(
        model_id="Qwen/Qwen3-8B",
        reasoning_end_marker=_QWEN_MARKER,
        # Verified: with enable_thinking=True the prompt ends at
        # `<|im_start|>assistant\n` with NO <think>, so the model emits the opener
        # itself and can skip the block.
        reasoning_start_marker=_QWEN_START,
        # Verified: enable_thinking=False changes the prompt and pre-fills an empty
        # `<think>\n\n</think>` block, which is Qwen3's documented way of disabling.
        reasoning_off_supported=True,
        sampling_thinking=_QWEN_SAMPLING_THINKING,
        sampling_instruct=_QWEN_SAMPLING_INSTRUCT,
        notes="Semin et al.'s main open model. Run REASONING-OFF to sit beside their "
              "Qwen3-8B column; they disabled reasoning by default and reported "
              "Qwen3-8B-Think only in Appendix B.3. Also a thesis model, so its frozen "
              "results live in Experiment_results/.",
    ),
]


#: Registered so `get()` resolves them, but deliberately OUT of the run set.
RETIRED_MODELS: List[ModelSpec] = [
    # Publication candidates dropped on 2026-08-18: too slow for the seed budget
    # (Qwen3.5-4B needed ~5h for 15 examples x 3 seeds) or superseded by Qwen3.8.
    ModelSpec(model_id="google/gemma-4-E4B-it", reasoning_end_marker=_GEMMA4_MARKER,
              reasoning_start_marker=_GEMMA4_START, reasoning_off_supported=True,
              sampling_thinking=_GEMMA_SAMPLING, sampling_instruct=_GEMMA_SAMPLING,
              notes="Dropped from the run set; pilot results exist in Experiment_results_publication."),
    ModelSpec(model_id="google/gemma-4-26B-A4B-it", reasoning_end_marker=_GEMMA4_MARKER,
              reasoning_start_marker=_GEMMA4_START, reasoning_off_supported=True,
              sampling_thinking=_GEMMA_SAMPLING, sampling_instruct=_GEMMA_SAMPLING,
              notes="Dropped: >3h for 15 examples x 3 seeds."),
    ModelSpec(model_id="Qwen/Qwen3.5-4B", reasoning_end_marker=_QWEN_MARKER,
              reasoning_off_supported=True, sampling_thinking=_QWEN_SAMPLING_THINKING,
              sampling_instruct=_QWEN_SAMPLING_INSTRUCT, notes="Dropped: ~5h for 15 examples x 3 seeds."),
    ModelSpec(model_id="Qwen/Qwen3.5-9B", reasoning_end_marker=_QWEN_MARKER,
              reasoning_off_supported=True, sampling_thinking=_QWEN_SAMPLING_THINKING,
              sampling_instruct=_QWEN_SAMPLING_INSTRUCT, notes="Dropped: superseded by Qwen3.8."),
    ModelSpec(model_id="Qwen/Qwen3.5-27B", reasoning_end_marker=_QWEN_MARKER,
              reasoning_off_supported=True, sampling_thinking=_QWEN_SAMPLING_THINKING,
              sampling_instruct=_QWEN_SAMPLING_INSTRUCT, notes="Dropped: superseded by Qwen3.8-27B."),
    ModelSpec(model_id="Qwen/Qwen3.6-27B", reasoning_end_marker=_QWEN_MARKER,
              reasoning_off_supported=True, sampling_thinking=_QWEN_SAMPLING_THINKING,
              sampling_instruct=_QWEN_SAMPLING_INSTRUCT, notes="Dropped: superseded by Qwen3.8-27B."),
    ModelSpec(model_id="Qwen/Qwen3.6-35B-A3B", reasoning_end_marker=_QWEN_MARKER,
              reasoning_off_supported=True, sampling_thinking=_QWEN_SAMPLING_THINKING,
              sampling_instruct=_QWEN_SAMPLING_INSTRUCT, notes="Dropped: superseded by Qwen3.8."),
]


#: Thesis models -- already evaluated, results frozen in `Experiment_results/`.
#: Not re-run, so deliberately excluded from `MODELS` and the SLURM array.
#:
#: Qwen/Qwen3-8B is NOT here any more: it was promoted into MODELS for the matched
#: Semin comparison. Keeping a copy in this list would shadow it.
LEGACY_MODELS: List[ModelSpec] = [
    ModelSpec(
        model_id="google/gemma-3-4b-it",
        notes="Thesis model. No reasoning mode.",
    ),
    ModelSpec(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        notes="Thesis model. No reasoning mode.",
    ),
]


MODEL_IDS: List[str] = [m.model_id for m in MODELS]
#: Lookups resolve both sets, so `get()` still works for thesis models.
BY_ID: Dict[str, ModelSpec] = {m.model_id: m for m in (*MODELS, *RETIRED_MODELS, *LEGACY_MODELS)}


def get(model_id: str) -> ModelSpec:
    """Look up a model, raising if it is not registered.

    Deliberately does NOT fall back to a default: an unregistered model would otherwise
    be treated as non-reasoning, which silently produces degraded-but-plausible results
    instead of an error. Register the model here first.
    """
    try:
        return BY_ID[model_id]
    except KeyError:
        raise SystemExit(
            f"[model_registry] Unregistered model: {model_id!r}\n"
            f"Add a ModelSpec for it in src/utils/model_registry.py.\n"
            f"Run set:\n  " + "\n  ".join(MODEL_IDS) +
            f"\nLegacy (thesis, not re-run):\n  " +
            "\n  ".join(m.model_id for m in LEGACY_MODELS)
        )


def by_index(index: int) -> ModelSpec:
    """Look up a run-set model by its SLURM array index, raising on out-of-range.

    Guards the bash failure mode where an oversized `--array` upper bound yields an
    empty `${MODELS[$SLURM_ARRAY_TASK_ID]}` that silently expands to nothing.
    """
    if not 0 <= index < len(MODELS):
        raise SystemExit(
            f"[model_registry] Array index {index} out of range "
            f"(valid: 0-{len(MODELS) - 1}, i.e. --array=0-{len(MODELS) - 1})."
        )
    return MODELS[index]


def resolve_sampling(spec: ModelSpec, enable_thinking: bool, **overrides: Any) -> Dict[str, Any]:
    """Merge a model's vendor preset with explicit CLI overrides.

    Resolution order is CLI override -> vendor preset -> omitted (so the model's own
    generation_config.json applies). Overrides that are None are ignored rather than
    blanking the preset, so `--top-p` left unset does not wipe out the recommended value.
    """
    resolved = spec.sampling(enable_thinking).as_kwargs()
    resolved.update({k: v for k, v in overrides.items() if v is not None})
    return resolved


def _main() -> None:
    """CLI so SLURM scripts can read the model list instead of duplicating it."""
    import argparse

    parser = argparse.ArgumentParser(description="Query the model registry.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--id", type=int, metavar="INDEX",
                       help="print the run-set model id at this array index")
    group.add_argument("--list", action="store_true",
                       help="print the run set with its capabilities")
    group.add_argument("--array", action="store_true",
                       help="print the full --array spec for the run set, e.g. 0-9")
    args = parser.parse_args()

    if args.id is not None:
        print(by_index(args.id).model_id)
    elif args.list:
        print(f"{'idx':<4}{'model':<34}{'reason':<8}{'off?':<6}{'effort?':<9}sampling(thinking)")
        for i, m in enumerate(MODELS):
            print(f"{i:<4}{m.model_id:<34}"
                  f"{'yes' if m.reasoning else 'no':<8}"
                  f"{'yes' if m.reasoning_off_supported else 'no':<6}"
                  f"{'yes' if m.supports_reasoning_effort else 'no':<9}"
                  f"{m.sampling_thinking.as_kwargs() or 'model default'}")
    elif args.array:
        print(f"0-{len(MODELS) - 1}")


if __name__ == "__main__":
    _main()
