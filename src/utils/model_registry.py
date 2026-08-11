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
from typing import Any, Dict, List, Optional


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
    - reasoning_off_supported: whether the reasoning phase can actually be disabled, for
      the reasoning-ON/OFF ablation. Default False -- only set True once verified for that
      model.
    - supports_reasoning_effort: whether the chat template accepts a `reasoning_effort`
      kwarg. True only for the GPT-OSS/Harmony family; passing it elsewhere is an error.
    - sampling_thinking / sampling_instruct: vendor presets, selected by thinking mode.
    - notes: free text; includes which hardware the model has actually been run on.
    """
    model_id: str
    reasoning_end_marker: Optional[str] = None
    reasoning_off_supported: bool = False
    supports_reasoning_effort: bool = False
    sampling_thinking: SamplingPreset = SamplingPreset()
    sampling_instruct: SamplingPreset = SamplingPreset()
    notes: str = ""

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


#: Publication run set. Index == SLURM_ARRAY_TASK_ID.
MODELS: List[ModelSpec] = [
    # -- Gemma 4 -----------------------------------------------------------------------
    ModelSpec(
        model_id="google/gemma-4-E4B-it",
        reasoning_end_marker=_GEMMA4_MARKER,
        reasoning_off_supported=True,
        sampling_thinking=_GEMMA_SAMPLING,
        sampling_instruct=_GEMMA_SAMPLING,
        notes="Smoke-tested on A100 (g12).",
    ),
    ModelSpec(
        model_id="google/gemma-4-26B-A4B-it",
        reasoning_end_marker=_GEMMA4_MARKER,
        reasoning_off_supported=True,
        sampling_thinking=_GEMMA_SAMPLING,
        sampling_instruct=_GEMMA_SAMPLING,
        notes="Smoke-tested on A100 (g12).",
    ),
    ModelSpec(
        model_id="google/gemma-4-31B-it",
        reasoning_end_marker=_GEMMA4_MARKER,
        reasoning_off_supported=True,
        sampling_thinking=_GEMMA_SAMPLING,
        sampling_instruct=_GEMMA_SAMPLING,
        notes="Smoke-tested on A100 (g12).",
    ),

    # -- Qwen 3.5 / 3.6 ------------------------------------------------------------------
    # Linear-attention kernels (flash-linear-attention + causal-conv1d); our causal-conv1d
    # build targets sm_75/80/87/90/100/120.
    ModelSpec(
        model_id="Qwen/Qwen3.5-4B",
        reasoning_end_marker=_QWEN_MARKER,
        reasoning_off_supported=True,
        sampling_thinking=_QWEN_SAMPLING_THINKING,
        sampling_instruct=_QWEN_SAMPLING_INSTRUCT,
        notes="Smoke-tested on A100 (g08).",
    ),
    ModelSpec(
        model_id="Qwen/Qwen3.5-9B",
        reasoning_end_marker=_QWEN_MARKER,
        reasoning_off_supported=True,
        sampling_thinking=_QWEN_SAMPLING_THINKING,
        sampling_instruct=_QWEN_SAMPLING_INSTRUCT,
        notes="Smoke-tested on A100 (g06).",
    ),
    ModelSpec(
        model_id="Qwen/Qwen3.5-27B",
        reasoning_end_marker=_QWEN_MARKER,
        reasoning_off_supported=True,
        sampling_thinking=_QWEN_SAMPLING_THINKING,
        sampling_instruct=_QWEN_SAMPLING_INSTRUCT,
    ),
    ModelSpec(
        model_id="Qwen/Qwen3.6-27B",
        reasoning_end_marker=_QWEN_MARKER,
        reasoning_off_supported=True,
        sampling_thinking=_QWEN_SAMPLING_THINKING,
        sampling_instruct=_QWEN_SAMPLING_INSTRUCT,
        notes="Smoke-tested on A100 (g12).",
    ),
    ModelSpec(
        model_id="Qwen/Qwen3.6-35B-A3B",
        reasoning_end_marker=_QWEN_MARKER,
        reasoning_off_supported=True,
        sampling_thinking=_QWEN_SAMPLING_THINKING,
        sampling_instruct=_QWEN_SAMPLING_INSTRUCT,
        notes="Smoke-tested on A100 (g12).",
    ),

    # -- GPT-OSS / Harmony ---------------------------------------------------------------
    # Reasoning cannot be disabled: Harmony always emits a reasoning channel, and
    # `reasoning_effort` (low/medium/high) is the only verbosity control -- hence
    # reasoning_off_supported=False and supports_reasoning_effort=True.
    # No sampling preset: gpt-oss ships its own generation_config.json and we do not
    # override it, which is the closest thing to "the vendor recommendation" here.
    ModelSpec(
        model_id="openai/gpt-oss-20b",
        reasoning_end_marker=_HARMONY_MARKER,
        reasoning_off_supported=False,
        supports_reasoning_effort=True,
        notes="Smoke-tested on H200. Falls back to bf16 on A100.",
    ),
    ModelSpec(
        model_id="openai/gpt-oss-120b",
        reasoning_end_marker=_HARMONY_MARKER,
        reasoning_off_supported=False,
        supports_reasoning_effort=True,
        notes="Smoke-tested on H200; H200 in practice.",
    ),
]


#: Thesis models -- already evaluated, results frozen in `Experiment_results/`.
#: Not re-run, so deliberately excluded from `MODELS` and the SLURM array.
LEGACY_MODELS: List[ModelSpec] = [
    ModelSpec(
        model_id="google/gemma-3-4b-it",
        notes="Thesis model. No reasoning mode.",
    ),
    ModelSpec(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        notes="Thesis model. No reasoning mode.",
    ),
    ModelSpec(
        model_id="Qwen/Qwen3-8B",
        reasoning_end_marker=_QWEN_MARKER,
        sampling_thinking=_QWEN_SAMPLING_THINKING,
        sampling_instruct=_QWEN_SAMPLING_INSTRUCT,
        notes="Thesis model. Ran with no reasoning in thesis.",
    ),
]


MODEL_IDS: List[str] = [m.model_id for m in MODELS]
#: Lookups resolve both sets, so `get()` still works for thesis models.
BY_ID: Dict[str, ModelSpec] = {m.model_id: m for m in (*MODELS, *LEGACY_MODELS)}


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
