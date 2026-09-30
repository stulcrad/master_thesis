"""Grammar-constrained (xgrammar) variant of the context-based baseline.

The context-based prompts ask for a JSON array of {entity, label, context} objects. The grammar below enforces exactly
that shape and restricts `label` to the task's label set, so a malformed response and an invalid label are both
impossible by construction. What it cannot enforce is that `entity` and `context` occur in the input -- the failure
the trie processor removes and this baseline is run to measure.
"""
import json

# xgrammar is imported inside the functions: the plain context-based runs share these scripts and must not need it.


def compile_entity_grammar(tokenizer, model, labels):
    """Compile the JSON-schema grammar ONCE per model. The compiled grammar is stateless and reusable."""
    schema = json.dumps({
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "entity": {"type": "string"},
                "label": {"type": "string", "enum": sorted(labels)},
                "context": {"type": "string"},
            },
            "required": ["entity", "label", "context"],
        },
    })
    import xgrammar as xgr

    vocab_size = getattr(model.config, "vocab_size", None)
    if vocab_size is None:
        vocab_size = model.config.text_config.vocab_size
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    return xgr.GrammarCompiler(tokenizer_info).compile_json_schema(schema)


def new_grammar_processor(compiled_grammar):
    """A fresh LogitsProcessor per generation: its matcher tracks the position within ONE generation."""
    import xgrammar as xgr

    return xgr.contrib.hf.LogitsProcessor(compiled_grammar)
