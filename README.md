# Span Classification for Decoder-Only Transformer Models

Source code for the master's thesis *Span Classification for Decoder-Only Transformer Models* (Bc. Radek Štulc, CTU Prague, 2026).

The repository contains two fully evaluated approaches for zero-shot span localization with generative LLMs — **context-based** and **constrained generation** — plus an exploratory **position-based** track that serves as a negative result. All three approaches are evaluated on CoNLL-2003 (NER), ToxicSpans (toxic span detection), and LegalQAEval (extractive legal QA).

---

## Repository layout

```
master_thesis/
├── src/utils/                          # Core library
│   ├── TokTrie.py                      # Byte-level token trie
│   ├── TrieSpanConstrainedProcessor.py         # Whole-sequence logits processor
│   ├── TrieSpanConstrainedProcessorTokenAware.py  # Token-aware logits processor
│   ├── context_matching_utils.py       # Context-based alignment utilities
│   ├── utils_functions.py              # Shared generation and evaluation helpers
│   ├── system_prompts.py               # All system prompt constants
│   └── ChatUI.py                       # Interactive Jupyter chat widget (Ollama)
├── scripts/                            # Batch evaluation scripts (HPC / CLI)
│   ├── evaluationNER_context_batch.py
│   ├── evaluationNER_cons_gen.py
│   ├── evaluationToxicSpans_context.py
│   ├── evaluationToxicSpans_cons_gen.py
│   ├── evaluationLegalQA_context.py
│   └── evaluationLegalQA_cons_gen.py
├── Notebooks/                          # Jupyter notebooks
│   ├── CoNLL_context.ipynb
│   ├── CoNLL_cons_gen.ipynb
│   ├── toxic_spans_context.ipynb
│   ├── toxic_spans_cons_gen.ipynb
│   ├── LegalQA_context.ipynb
│   ├── LegalQA_cons_gen.ipynb
│   ├── Trie_Notebook.ipynb
│   ├── Experiment_analysis.ipynb
│   └── Empty_prediction_ablation.ipynb
├── Experiment_results/                 # Raw CSV + TXT outputs
├── requirements.txt
└── README.md
```

---

## Setup

**Python 3.12** is recommended (the cluster environment uses Python 3.12.3).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Constrained generation approach

No additional setup is required beyond the Python environment above. Model weights are loaded directly from Hugging Face via the `transformers` library.

On first run, the models (Gemma-3 4B-IT, Qwen3 8B, Llama-3.1 8B Instruct) are downloaded automatically to the Hugging Face cache. A Hugging Face account token may be required for gated models:

```python
from huggingface_hub import login
login()   # or set HF_TOKEN env variable
```

**Run a notebook:**
```bash
jupyter notebook Notebooks/CoNLL_cons_gen.ipynb
```

**Run a batch script:**
```bash
python scripts/evaluationNER_cons_gen.py
```

---

## Context-based approach

The context-based approach communicates with models through an OpenAI-compatible chat API. The recommended local backend is **Ollama**.

1. Download and install Ollama from [ollama.com](https://ollama.com).
2. Pull the models you want to evaluate:
   ```bash
   ollama pull gemma3:4b
   ollama pull llama3.1:8b
   ollama pull qwen3:8b
   ```
3. Start the Ollama server (it starts automatically on most platforms, or run `ollama serve`).
4. The notebooks and scripts default to `http://localhost:11434` — no further configuration needed.

**Run a notebook:**
```bash
jupyter notebook Notebooks/CoNLL_context.ipynb
```

**Run a batch script:**
```bash
python scripts/evaluationNER_context_batch.py
```

---

## Experiment analysis

`Notebooks/Experiment_analysis.ipynb` loads the aggregated CSV files from `Experiment_results/` and reproduces all plots and tables reported in the thesis. No model inference is required — it runs entirely on the saved results.

`Notebooks/Empty_prediction_ablation.ipynb` computes the char-F1 achieved by always predicting the empty set on ToxicSpans and LegalQAEval.

---

## Adding `src` to the Python path

The scripts and notebooks import from `src/utils/`. If you run them from the repository root, add `src` to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
jupyter notebook
```

Or, from inside a notebook:

```python
import sys
sys.path.insert(0, "../src")
```
