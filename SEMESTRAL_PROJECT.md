# SEMESTRAL_REPORT

# Semestral Project: Text Segment Classification for Transformer Decoder-Only Architectures - Report of Work

**Author:** Radek Stulc

**Date:** January 2026

**Project:** Master’s Thesis - Text Segment Classification Using Generative Models

---

## Executive Summary

Text segment classification using only decoder-only LLMs is much more challenging than it sounds. Not only does it require finding and correctly classifying what is required, but in large documents, we also need to precisely locate the classified segment within the original text. For now, encoder-only models dominate this field, as they have no problem assigning the correct labels to the segments and showing where exactly the segments are located. However, training of these models requires data, time, and budget, and the models are often not robust enough to handle new labels and require further fine-tuning.

For this reason, I want to explore the possibilities of using generative LLMs for this exact task, and we chose Named Entity Recognition as a starting problem, since it is an easy problem that can be solved acceptably with generative LLMs, but I took a deeper look on how to connect labeled entities with the original text, supposing the entities appear more times in the input text with different context.

---

## 1. State of the Art in Solving NER with Generative LLMs

[https://arxiv.org/abs/2401.10825](https://arxiv.org/abs/2401.10825)

### 1.1 Traditional Approaches

Named Entity Recognition has traditionally been dominated by:
- **Fine-tuned encoder models** (BERT, RoBERTa), which excel at token-level classification
- **CRF-based approaches** for capturing dependencies between entity labels
- **Bi-LSTM architectures** for sequence modeling

These approaches require:
- Large amounts of labeled training data
- Task-specific fine-tuning
- Difficulty adapting to new entity types without retraining

### 1.2 Generative LLM Approaches

Recent advances in Large Language Models have opened new possibilities:

**Advantages:**
- Zero-shot or few-shot learning capabilities
- Natural language understanding without fine-tuning
- Flexibility to handle new entity types through prompt engineering
- Ability to leverage context understanding

**Challenges:**
- Precise span alignment with input text
- Handling multiple occurrences of the same entity
- Computational cost and latency
- Consistency in output format

### 1.3 The iNERD Approach

[https://arxiv.org/abs/2308.07791](https://arxiv.org/abs/2308.07791)

A particularly interesting approach is constrained decoding for NER tasks, which:
- Guides the LLM generation to produce valid entity spans
- Ensures output format consistency
- Improves precision by constraining the search space

However, this approach still faces challenges with entity localization when entities appear multiple times with different contexts in long documents.

---

## 2. Problem Statement

The core challenge addressed in this semestral project is:

**How can we accurately map entities extracted by generative LLMs back to their precise locations in the input text, especially when entities appear multiple times with different surrounding contexts?**

### 2.1 Key Challenges

1. **Multiple Occurrences:** An entity like “Charles” may appear multiple times in a document with different contexts (e.g., as “Charles IV” vs “Charles I” vs “Charles of Luxembourg”)
2. **Span Alignment:** The LLM output must be precisely aligned with token boundaries in the original text
3. **Context Disambiguation:** When an entity appears multiple times, we need to correctly identify each occurrence
4. **Fuzzy Matching:** LLM outputs may have slight variations (punctuation, spacing) compared to the original text

---

## 3. Methodology

### 3.1 Context-Based Entity Extraction

Instead of asking the LLM to simply extract entities, I designed a prompt that requires the model to:
1. Extract each entity
2. Provide its label (PER, LOC, ORG, MISC, etc.)
3. Include a short surrounding context (4-8 words)

**JSON Output Format:**

```json
[
    { "entity": "Barack Obama", "label": "PER", "context": "Barack Obama was born" },
    { "entity": "Hawaii", "label": "LOC", "context": "born in Hawaii." },
    { "entity": "Barack", "label": "PER", "context": "Barack was american." }
]
```

This approach allows us to:
- Distinguish between multiple occurrences of the same entity
- Precisely locate each entity in the original text
- Handle partial entity mentions (e.g., “Charles” vs “Charles IV”)

### 3.2 Initial Attempts: Token/Character Position-Based Approaches

Before settling on the context-based approach, I experimented with having the LLM directly output token positions, character offsets, or word counts along with the entities. This seemed like a more direct solution that would avoid the need for fuzzy matching altogether.

**Attempted Approaches:**

1. **Token Position Output (**[`SYSTEM_PROMPT_TOKENS_TEXT`](https://www.notion.so/SEMESTRAL_REPORT-2df6ee137c588015aeb0eed7c35b777d?pvs=21)**, [`SYSTEM_PROMPT_TOKENS_JSON`](https://www.notion.so/SEMESTRAL_REPORT-2df6ee137c588015aeb0eed7c35b777d?pvs=21))**
    - Asked LLM to first tokenize the text
    - Then output entity spans using token indices (start/end)
    - Example format: 
    `Radek Stulc - PERSON - 4-5`
    or
    `{"entity": "Barack Obama", "label": "PER", "start": 0, "end": 1}`
2. **Character Offset Output**
    - Requested character-level positions in the original text
    - Format: `{"entity": "Barack Obama", "label": "PER", "start": 0, "end": 11}`
3. **Word Count-Based Output**
    - Attempted to have LLM count words from the beginning of text
    - Used word indices instead of token indices

**Why These Approaches Failed:**

The experiments revealed several critical problems:

1. **Inconsistent Tokenization:** LLMs tokenize text differently than standard tokenizers, and the tokenization can vary between calls, even with the same model.
2. **Counting Errors:** Models frequently made off-by-one errors or miscounted positions, especially in longer texts or with punctuation.
3. **Complex Reasoning Overhead:** The model had to perform two tasks simultaneously:
    - Identify named entities
    - Count positions accurately
    
    This dual requirement significantly degraded both tasks’ performance.
    
4. **Verbose Outputs:** Models often included explanatory text or reasoning steps despite being instructed not to, making JSON parsing unreliable.
5. **Punctuation Ambiguity:** Uncertainty about whether to include trailing punctuation or parentheses led to inconsistent boundaries.
6. **Performance Issues:** The added complexity led to longer inference times, and often the model failed to finish within time limits or looped indefinitely.

All of these issues culminated in unreliable outputs, questionable performance, looping reasoning chains, or even timeouts and random outputs, as seen in the chat logs.

**Example of Failure:**

![random_story_example.png](random_story_example.png)

![example_of_looping.png](example_of_looping.png)

**Key Insight:** LLMs excel at pattern recognition and natural language understanding, but struggle with precise numerical counting tasks. This led to the development of the context-based approach, which plays to the model’s strengths.

### 3.3 System Prompts

After abandoning position-based approaches, I developed and tested three context-based system prompt variants:

### [SYSTEM_PROMPT_CONTEXT](https://www.notion.so/SEMESTRAL_REPORT-2df6ee137c588015aeb0eed7c35b777d?pvs=21)

- Detailed instructions with examples
- Explicit rules about nested entities
- Clear JSON format specification

### [SYSTEM_PROMPT_CONTEXT_MD](https://www.notion.so/SEMESTRAL_REPORT-2df6ee137c588015aeb0eed7c35b777d?pvs=21)

- Markdown-formatted for better readability
- Hierarchical structure with priority rules
- More extensive examples

### [SYSTEM_PROMPT_CONTEXT_MD_SHORT](https://www.notion.so/SEMESTRAL_REPORT-2df6ee137c588015aeb0eed7c35b777d?pvs=21)

- Condensed version for faster processing
- Essential instructions only
- Optimized for smaller models

### 3.4 Entity-to-Token Alignment Algorithm

The core innovation is the [`assign_entities_from_context()`](https://www.notion.so/SEMESTRAL_REPORT-2df6ee137c588015aeb0eed7c35b777d?pvs=21) function, which:

1. **Context Localization:** Finds the provided context in the full text
2. **Entity Localization:** Locates the entity within the matched context
3. **BIO Tag Assignment:** Assigns Begin-Inside-Outside tags to tokens

**Pseudocode:**

```python
def assign_entities_from_context(full_text_tokens, entities, fuzzy, fuzzy_threshold=0.7, matching_type="anchor"):
    """Return BIO tags for tokens based on entities with context."""
    tags = ['O'] * len(full_text_tokens)
    full_text_str = " ".join(full_text_tokens)

    for ent in entities:
        entity_text  = ent["entity"], entity_label = ent["label"], context_text = ent["context"]

        if fuzzy:
            # 1) Locate context in full text using fuzzy matching
            ctx_start, ctx_end, _ = find_best_fuzzy_match(
                context_text, full_text_str, threshold=fuzzy_threshold, matching_type=matching_type)

            matched_ctx = full_text_str[ctx_start:ctx_end]
            # 2) Locate entity inside the matched context
            ent_start = matched_ctx.lower().find(entity_text.lower())
            if ent_start == -1:
                # The entity not found exactly, try fuzzy matching
                ent_start, _, _ = find_best_fuzzy_match(
                    entity_text, matched_ctx, threshold=fuzzy_threshold, matching_type=matching_type)
        else:
            # Exact search fallback
            ctx_start = full_text_str.lower().find(context_text.lower())
            ent_start = context_text.lower().find(entity_text.lower())

        # 3) Map to absolute character offsets
        ent_char_start = ctx_start + ent_start
        ent_char_end   = ent_char_start + len(entity_text)

        # 4) Assign BIO tags to overlapping tokens
        char_idx = 0
        for i, tok in enumerate(full_text_tokens):
            tags[i] = assign_bio_tag(tok) # Assign bio tag according to the token

    return tags
```

### 3.5 [Fuzzy Matching Enhancement](https://www.notion.so/SEMESTRAL_REPORT-2df6ee137c588015aeb0eed7c35b777d?pvs=21)

To handle slight variations in LLM outputs, I implemented fuzzy string matching using RapidFuzz:

**Key Features:**
- Partial ratio matching with a sliding window
- Full ratio matching for accuracy
- Anchor-based optimization for efficiency
- Configurable similarity threshold (default 0.6-0.75)
- Fallback to exact matching when possible

I used, in experiments, only the anchor-based matching for speed.

**Pseudocode:**

```python
def find_best_fuzzy_match(context, full_text, threshold=0.7, matching_type="anchor"):
    """Return (start, end, score) of the best match or (None, None, 0)."""

    # Normalize inputs for case-insensitive matching
    ctx = context.lower(); text = full_text.lower(); n = len(ctx)

    # 1) Fast exit: exact substring
    pos = text.find(ctx)
    if pos != -1:
        return (pos, pos + n, 1.0)

    best = (None, None, 0.0) # start, end, ratio

    # Anchor-based matching
    # Really fast filtering using context start as anchor
    # Has worse finding rate but is O(n) instead of O(n^2)
    if matching_type == "anchor":
        # Use start of context as anchor
        anchors = context[:6] if n > 10 else context[:3]
        candidates = positions_from_anchors(anchors, text) # compute where full context could start
        candidates = fallback_grid_if_empty(candidates, n, len(text)) # If no anchors found, use regular sliding window
        for start in candidates:
            # Window is always +- 15 chars around context
            window = take_window(text, start, n, 15)
            score = max(partial_ratio(ctx, window), token_sort_ratio(ctx, window)) / 100
            if score >= threshold and score > best[2]:
                span_start, span_end, span_score = best_subspan(window, ctx)  # local refine loop
                best = (start + span_start, start + span_end, max(score, span_score))

    # Sliding window partial ratio
    elif matching_type == "partial":
        # Slide medium windows (n-5 .. n+10) with partial_ratio, stepping to stay fast
        step = max(2, n // 10)
        for w in range(max(5, n - 5), n + 11):
            for i in range(0, len(text) - w + 1, step):
                window = text[i:i + w]
                score = partial_ratio(ctx, window) / 100
                if score >= threshold and score > best[2]:
                    best = (i, i + w, score)

    # Full ratio sliding window
    elif matching_type == "full":
        # Brute-force nearby window sizes with full ratio (more accurate, slower)
        for w in range(max(1, n - 5), n + 10):
            for i in range(0, len(text) - w + 1):
                window = text[i:i + w]
                score = ratio(ctx, window) / 100
                if score > best[2]:
                    best = (i, i + w, score)

    return best if best[2] >= threshold else (None, None, 0.0)
```

---

## 4. Experimental Setup

### 4.1 Datasets

**CoNLL-2003:**
- Standard NER benchmark dataset
- 4 entity types: PER, LOC, ORG, MISC
- Test split: 3,453 examples
- Used 200 examples per experiment for faster iteration

**DocRed (Document-level Relation Extraction Dataset):**
- Multi-paragraph documents
- More complex entity types: PER, LOC, ORG, NUM, TIME, MISC
- 64 examples per experiment
- Validation split used: 998 examples
- System prompts differ only in label definitions

### 4.2 Models Evaluated

All models served via Ollama with OpenAI-compatible API:

1. **gemma3:4b** - Compact model for efficiency testing
2. **qwen3:8b** - Medium-sized model with reasoning capabilities
3. **gpt-oss:20b** - Larger model with advanced reasoning
4. **llama3.1:8b** - Popular open-source baseline

### 4.3 Experimental Variables

**Batch Sizes:** 1, 5, 10, 32 for CoNLL; 1, 2, 4 for DocRed
- Testing document-level context handling
- Multiple examples concatenated into single prompt

**Fuzzy Matching:** Enabled/Disabled
- Threshold: 0.6 for fuzzy matching
- Comparison with exact string matching

**System Prompts:** 3 variants
- Testing prompt engineering impact

**Iterations:** 3 repetitions
- Statistical significance and variance measurement

**Reasoning Configuration:**
- **Low reasoning effort** enabled for gpt-oss models (`reasoning_effort="low"`)
- **Reasoning disabled** for Qwen models (using `\no_think` directive)
- **Rationale:** Reduces inference time and prevents models from overthinking simple pattern-matching tasks
- **Temperature:** 0.2 for all models to ensure consistent outputs

The reasoning configuration is crucial:
This configuration prevents the models from engaging in verbose reasoning chains (which were problematic in the token-counting approach) and focuses them on direct pattern matching and entity extraction.

### 4.4 Evaluation Metrics

Using `seqeval` library with strict IOB2 scheme:

- **Precision:** % of predicted entities that are correct
- **Recall:** % of gold entities that were found
- **F1 Score:** Harmonic mean of precision and recall
- **Accuracy:** Token-level accuracy
- **Runtime:** Elapsed time in minutes

---

## 5. Results

### 5.1 Example Of Overall Performance (CoNLL-2003, Batch Size 5, Exact Matching)

| System Prompt | Model | Precision | Recall | F1 | Accuracy | Time (min) |
| --- | --- | --- | --- | --- | --- | --- |
| CONTEXT | gemma3:4b | 0.624 | 0.472 | 0.536 | 0.881 | 4.67 |
| CONTEXT | qwen3:8b | 0.700 | 0.518 | 0.595 | 0.904 | 4.95 |
| CONTEXT | gpt-oss:20b | **0.634** | **0.667** | **0.650** | **0.914** | 6.01 |
| CONTEXT | llama3.1:8b | **0.732** | 0.527 | 0.612 | 0.908 | **1.96** |
| CONTEXT_MD | gemma3:4b | 0.591 | 0.419 | 0.490 | 0.864 | 2.04 |
| CONTEXT_MD | qwen3:8b | 0.697 | 0.555 | 0.618 | 0.911 | 5.18 |
| CONTEXT_MD | gpt-oss:20b | 0.633 | **0.699** | **0.664** | 0.916 | 6.65 |
| CONTEXT_MD | llama3.1:8b | **0.721** | 0.475 | 0.572 | 0.905 | **1.87** |
| CONTEXT_MD_SHORT | gemma3:4b | 0.631 | 0.415 | 0.501 | 0.873 | 1.97 |
| CONTEXT_MD_SHORT | qwen3:8b | 0.696 | 0.545 | 0.612 | 0.911 | 2.23 |
| CONTEXT_MD_SHORT | gpt-oss:20b | 0.643 | **0.680** | **0.661** | 0.919 | 7.07 |
| CONTEXT_MD_SHORT | llama3.1:8b | **0.743** | 0.491 | 0.591 | 0.909 | **1.81** |

**Key Findings:**

1. **Best F1 Score:** gpt-oss:20b with CONTEXT_MD prompt achieved 0.664 F1
2. **Best Precision:** llama3.1:8b consistently achieved >0.72 precision
3. **Best Recall:** gpt-oss:20b models achieved ~0.68-0.70 recall
4. **Fastest:** llama3.1:8b (~2 minutes vs 5-7 minutes for others)

### 5.2 Visualizations and Analysis

1. **Per model F1 across all batch sizes and prompts**
    1. F1 scores across batch sizes
    2. Separated by prompt type
    3. Model comparison and prompt comparison
    4. All prompts have similar performance, so for the next plots, I decided to average over all prompts to gain even more statistical significance

![per_model_f1_vs_batch_size_prompts.png](per_model_f1_vs_batch_size_prompts.png)

1. **Per model F1 vs Batch Size**
    1. Fuzzy vs Exact matching comparison
    2. It can be seen that fuzzy matching helps a little; there is a possibility that on more complex problems, the fuzzy matching will help much more

![per_model_f1_vs_batch_size.png](per_model_f1_vs_batch_size.png)

1. **Runtime vs Batch Size**
    1. Computational cost analysis
    2. Model efficiency comparison
    3. gpt-oss:20b is naturally the slowest

![per_model_runtime_vs_batch_size.png](per_model_runtime_vs_batch_size.png)

1. **Average Metrics Per Model vs Batch Size**
    1. Analysis of all metrics
    2. Interestingly precision rose across batch sizes, recall, and F1 naturally fell

![avg_metrics_per_model_vs_batch_size.png](avg_metrics_per_model_vs_batch_size.png)

### 5.3 Impact of Fuzzy Matching

Fuzzy matching showed improvements in entity alignment according to F1 scores. A more thorough analysis is required to see the exact effects of fuzzy matching, for example, count how many entities were correctly matched only with fuzzy matching enabled, or in how many cases fuzzy matching corrected misalignments.

Fuzzy matching even reduced the runtime with 5 samples.

### 5.4 Batch Size Impact

Different batch sizes showed interesting trade-offs:
- **Batch Size 1:** Most accurate individual predictions
- **Batch Size 5:** Good balance of speed and accuracy
- **Batch Size 10+:** Faster but potential context confusion

### 5.5 Model Comparison

**gpt-oss:20b:**
- Best overall F1 score
- Excellent recall
- Slower inference time
- Good at finding entities but less precise

**llama3.1:8b:**
- Highest precision
- Fastest inference
- Lower recall (more conservative)
- Best cost-performance trade-off

**qwen3:8b:**
- Balanced performance
- Moderate speed
- Consistent across prompts

**gemma3:4b:**
- Smallest and fastest
- Lower accuracy
- Good for prototyping

---

## 6. Implementation Details

### 6.1 Project Structure

```
master_thesis/
├── chats/                                      # Saved conversation logs
├── NER_results/
│   ├── CoNLL/
│   │   ├── Csv/                                # Structured results
│   │   ├── Txt/                                # Human-readable results
│   │   └── Plots/                              # Visualizations
│   └── DocRed/
│       ├── Csv/
│       └── Txt/
├── Notebooks/
│   ├── test_ollama.ipynb                       # Development & analysis
│   ├── DocRed_tests.ipynb                      # DocRed experiments
│   └── test_constrained_decoding.ipynb         # Constrained decoding tests
│
├── screenshots/                                # Screenshots of chats only for report
├── scripts/
│   ├── evaluationNER_context_batch.py          # CoNLL-2003 evaluation
│   └── evaluationNER_DocRed_context_batch.py   # DocRed evaluation
├── src/utils/
│   ├── system_prompts.py                       # Prompt templates
│   ├── context_matching_utils.py               # Alignment algorithms
│   └── ChatUI.py                               # Interactive testing interface
│
├── slurm/                                      # SLURM batch scripts
│   ├── jupyter/                                # Jupyter on cluster
│   ├── ollama/                                 # Ollama serving
│   ├── scripts/                                # Evaluation jobs
└───┴── vllm/                                   # vLLM serving (alternative)
```

### 6.2 Key Components

### System Prompts (`system_prompts.py`)

Multiple prompt variants tested:

- SYSTEM_PROMPT_TOKENS_TEXT: Original token-based approach without json
- SYSTEM_PROMPT_TOKENS_JSON: Original token-based approach
- SYSTEM_PROMPT_CONTEXT: Context-enhanced prompts
- SYSTEM_PROMPT_CONTEXT_MD: Markdown formatted
- SYSTEM_PROMPT_CONTEXT_MD_SHORT: Optimized short version
- SYSTEM_PROMPT_DOCRED: Document-level entity extraction

### Context Matching Utils (`context_matching_utils.py`)

Core algorithms:
- `json_safe_parse()`: Robust JSON extraction from LLM output
- `find_best_fuzzy_match()`: Fuzzy string matching with RapidFuzz
- `assign_entities_from_context()`: Entity-to-token alignment
- `docred_to_bio_tags()`: DocRed format conversion

### Interactive Chat UI (`ChatUI.py`)

- Real-time testing interface
- Conversation saving/loading
- Support for multiple models
- Context injection

---

## 7. Challenges and Solutions

### 7.1 Failed Positional Approaches

As detailed in Section 3.2, the initial attempts to have LLMs output exact token positions or character offsets proved unsuccessful. The chat logs in the `chats/` directory document these experiments.

They can be loaded using `ChatUI.py` in `test_ollama.ipynb` notebook.

Visual documentation in `screenshots/` provides concrete examples:
- Infinite reasoning loops when counting positions
- Inconsistent outputs on identical inputs
- Occasional success on trivial examples that didn’t scale
- Dramatic improvement when switching to a context-based approach on large texts

These failures led to the fundamental design decision to use context snippets instead of positions, which proved far more reliable and played to the LLM’s strengths in pattern matching rather than precise counting. Additionally, configuring models with low reasoning effort prevented them from overthinking the task, resulting in faster and more consistent outputs.

### 7.2 Entity Localization

**Challenge:** Entities appearing multiple times with different forms
**Solution:** Context-based disambiguation with fuzzy matching

Example:

```
Input: "Charles was born in Prague. Charles became King of Bohemia."
LLM Output:
- {"entity": "Charles", "label": "PER", "context": "Charles was born"}
- {"entity": "Charles", "label": "PER", "context": "Charles became King"}
```

The algorithm successfully distinguishes these two different references.

### 7.3 Output Format Consistency

**Challenge:** LLMs sometimes include explanations or malformed JSON
**Solution:** `json_safe_parse()` extracts JSON array from any text

```python
def json_safe_parse(text):
    start = text.find("[")
    end = text.rfind("]") + 1
    return json.loads(text[start:end]) if start != -1 else []
    
```

### 7.4 Performance vs Accuracy Trade-off

**Challenge:** Larger models are more accurate but slower
**Solution:** Batch processing and model selection based on use case

- Development/Testing: gemma3:4b (fast)
- Production: llama3.1:8b (balanced)
- High-Accuracy: gpt-oss:20b (best F1)

### 7.5 Fuzzy Matching Performance

**Challenge:** Naive fuzzy matching is too slow for long documents
**Solution:** Anchor-based search with windowing

Optimization reduced matching time from O(n²) to O(n) with anchor filtering.

---

## 8. Notebooks and Interactive Development

### 8.1 test_ollama.ipynb

Main development notebook containing:
- Model connection and testing
- Prompt engineering experiments
- Fuzzy matching algorithm development
- Interactive ChatUI for manual testing
- Full evaluation pipeline
- Results visualization

Key experiments documented:
- Testing with Charles IV Wikipedia text
- Fuzzy matching threshold tuning
- Context window size optimization

### 8.2 DocRed_tests.ipynb

Document-level NER experiments:
- Multi-paragraph document handling
- Complex entity type evaluation
- Cross-sentence entity resolution

### 8.3 test_constrained_decoding.ipynb

Exploring constrained decoding approaches:
- Grammar-based output control
- Format enforcement

---

## 9. Conclusions

### 9.1 Main Achievements

1. **Successfully demonstrated** that generative LLMs can perform NER with proper context-based entity localization
2. **Developed a novel approach** for handling multiple entity occurrences through context disambiguation
3. **Implemented efficient fuzzy matching** that improves entity alignment without a significant performance penalty
4. **Evaluated multiple models** and prompt strategies, identifying optimal configurations

### 9.2 Key Insights

1. **Context is crucial:** Including the surrounding context in the output dramatically improves entity localization
2. **Reasoning effort matters:** Disabling or reducing reasoning effort prevents models from overthinking simple tasks:
    - Faster inference (2-7 minutes vs potential timeouts)
    - More consistent outputs
    - Less verbose, cleaner JSON
    - Prevents infinite reasoning loops (as seen in `screenshots/example_of_looping.png`)
3. **Model selection matters:** Different models excel at different aspects:
    - gpt-oss:20b: Best recall
    - llama3.1:8b: Best precision and speed
    - Trade-offs depend on the use case
4. **Fuzzy matching is essential:** Real-world LLM outputs have variations that exact matching cannot handle
5. **Batch processing trade-offs:** Larger batches are faster, but may reduce accuracy for complex documents

### 9.3 Limitations

1. **Computational Cost:** LLM inference is significantly slower than fine-tuned encoder models
2. **Consistency:** Some variation across runs despite low temperature
3. **Complex Entities:** Nested entities and overlapping mentions remain challenging
4. **Language Support:** Primarily tested on English; multilingual performance unknown

### 9.4 Future Work

1. **Constrained Decoding:** Integrate grammar-based generation for guaranteed format compliance
2. **Few-Shot Learning:** Explore dynamic example selection for prompt engineering
3. **Relation Extraction:** Extend context-based approach to entity relations
4. **Model Optimization:** Fine-tune smaller models on context-based outputs
5. **Real-time Applications:** Optimize for streaming and incremental processing
6. **Multilingual Evaluation:** Test on Czech, German, and other languages

---

## 10. References

### Papers Reviewed

1. **Recent Advances in NER**
    - [https://arxiv.org/abs/2401.10825](https://arxiv.org/abs/2401.10825)
    - Survey of SOTA approaches
    - Comparison of encoder vs decoder models
    - Zero-shot and few-shot techniques
2. **iNERD Paper**
    - [https://arxiv.org/abs/2308.07791](https://arxiv.org/abs/2308.07791)
    - Constrained decoding for NER
    - Format enforcement strategies
    - Integration with generative models

### Tools and Libraries

- **Ollama:** LLM serving framework
- **RapidFuzz:** High-performance fuzzy string matching
- **Hugging Face Datasets:** Dataset loading and processing
- **seqeval:** Sequence labeling evaluation
- **OpenAI Python Client:** API interface

---

## Appendix A: System Prompt Examples

### [SYSTEM_PROMPT_TOKENS_TEXT](https://www.notion.so/SEMESTRAL_REPORT-2df6ee137c588015aeb0eed7c35b777d?pvs=21)

```
You are an expert at named entity recognition and I want you to with the given message input tokenize the text and find all named entities in the text.
The given labels for named entities are: 
PERSON: names of people, can be names in different languages, nicknames, usernames, fictional characters, titles with names (like Dr. Smith), etc.
LOCATION: names of cities, countries, landmarks, geographical features, addresses, etc.
ORGANIZATION: names of companies, institutions, agencies, teams, etc.
MISC: everything else that does not fit into the previous categories, but can be considered a named entity, like events, works of art, nationalities, religions, etc.
I want you to list all entities and number their span where they are in the text according to the given tokens, tokens are everything you see in the sentence,
it can be words, punctuation marks, special characters, etc. 
Keep the original token only split the text token by token like a tokenizer would, so "I'm" is split into ['I', ''', 'am'], so it is three tokens,
and ""What is wrong with Robert?" he said", the sentence has tokens: ['"', 'What', 'is', 'Wrong', 'with', 'Robert', '?', '"', 'he', 'said'], so the sentence has 10 tokens.
Example:
Input: I'm Radek Stulc, I was born in Prague, and I am currently studying at CTU.
Tokens: ['I', ''', 'm', 'Radek', 'Stulc', ',', 'I', 'was', 'born', 'in', 'Prague', ',', 'and', 'I', 'am', 'currently', 'studying', 'at', 'CTU', '.']
Output:
Radek Stulc - PERSON - 4-5
Prague - LOCATION - 11
CTU - ORGANIZATION - 19

Do not overthink, do not add any explanations, do not add anything else, just tokenize the given text, find all named entities and list them with their type and span.

```

### [SYSTEM_PROMPT_TOKENS_JSON](https://www.notion.so/SEMESTRAL_REPORT-2df6ee137c588015aeb0eed7c35b777d?pvs=21)

```
You are an expert at named entity recognition. Given an input text, tokenize it and extract all named entities along with their types and token positions.
Do not extract nested entities, only the outermost ones.

The possible labels for named entities are:
PER: names of people (different languages, nicknames, usernames, fictional characters, titles with names, etc.)
LOC: names of cities, countries, landmarks, geographical features, addresses, etc.
ORG: names of companies, institutions, agencies, teams, etc.
MISC: everything else that does not fit into the previous categories, but can still be considered a named entity (events, works of art, nationalities, religions, etc.)

Tokenization rules:
Split the tokens as you see fit, but output the tokenized input text as well for verification.

Output format:
First, output the tokenized input text as a list of strings.
Then output the named entities and their label and span as an array of JSON objects like this:
[
    { "entity": "ENTITY", "label": "ENTITY_LABEL", "start": TOKEN_START, "end": TOKEN_END },
    ...
]

Example:
Input text: "Barack Obama was born in Hawaii."
Output:
["Barack", "Obama", "was", "born", "in", "Hawaii", "."]
[
    { "entity": "Barack Obama", "label": "PER", "start": 0, "end": 1 },
    { "entity": "Hawaii", "label": "LOC", "start": 5, "end": 5 }
]

If there are no named entities, output an empty JSON array [].
IMPORTANT: Do not add any explanations, just output the tokenized text as list and the JSON array.
```

### [SYSTEM_PROMPT_CONTEXT](https://www.notion.so/SEMESTRAL_REPORT-2df6ee137c588015aeb0eed7c35b777d?pvs=21)

```
You are an expert at named entity recognition. Given an input text, extract all named entities along with their types and surrounding context.
Do not extract nested entities, only the outermost ones.

IMPORTANT: If an entity appears multiple times, but with different surrounding context, extract each occurrence separately.

The possible labels for named entities are:
PER: names of people (different languages, nicknames, usernames, fictional characters, titles with names, etc.)
LOC: names of cities, countries, landmarks, geographical features, addresses, etc.
ORG: names of companies, institutions, agencies, teams, etc.
MISC: everything else that could be considered a named entity (languages, nationalities, etc.).

The order of labeling is PER, LOC, ORG, MISC.

Output format:
Return a JSON array of objects:
[
    { "entity": "ENTITY", "label": "ENTITY_LABEL", "context": "SURROUNDING_CONTEXT" },
    ...
]

Rules:
- "entity" must exactly match the original substring from the input text.
- "label" must be one of the specified entity labels.
- "context" must be a short snippet (4-8 words) from the input text that contains the entity and a few neighboring words.
- The entity must be included in the context snippet
- If the entity is at the beginning or end of the text, use only the available neighboring words.
- If there are no named entities, output an empty JSON array [].

Example:
Input text: "Barack Obama was born in Hawaii. Barack was american."
Output:
[
    { "entity": "Barack Obama", "label": "PER", "context": "Barack Obama was born" },
    { "entity": "Hawaii", "label": "LOC", "context": "born in Hawaii." },
    { "entity": "Barack", "label": "PER", "context": "Barack was american." }
]

IMPORTANT: Only output the JSON array. DO NOT add any explanations. Follow the format exactly.
```

### [SYSTEM_PROMPT_CONTEXT_MD](https://www.notion.so/SEMESTRAL_REPORT-2df6ee137c588015aeb0eed7c35b777d?pvs=21)

```
### Role
You are an expert in **Named Entity Recognition (NER)**.  
Your task is to extract all named entities from a given text, along with their **types** and **short surrounding context**.

---

### Entity Label Definitions
- **PER** — people's names (real or fictional, including titles, nicknames, usernames, etc.)
- **LOC** — cities, countries, landmarks, geographical features, or addresses
- **ORG** — companies, institutions, agencies, or teams
- **MISC** — everything else that can be considered a named entity (events, works of art, nationalities, religions, languages, etc.)

*Priority rule:* `PER > LOC > ORG > MISC`

---

### Extraction Rules
1. Extract **only the outermost entities** — do not include nested entities.
2. If an entity appears multiple times in different contexts, extract **each occurrence separately**.
3. For each entity, include a **short context snippet (4-8 words)** containing the entity and nearby words.
4. The entity **must** be part of the context snippet.
5. The `"entity"` text must **exactly match** the substring from the input.
6. If no entities are found, output an empty array: `[]`.

---

### Output Format
Return **only** a JSON array in this format:
```json
[
    { "entity": "ENTITY", "label": "ENTITY_LABEL", "context": "CONTEXT_SNIPPET" },
    ...
]
```

No explanations, no markdown, no extra text — only valid JSON.

---

### Example
**Input:**
Barack Obama was born in Hawaii. Barack was american.

**Output:**
```json
[
    { "entity": "Barack Obama", "label": "PER", "context": "Barack Obama was born" },
    { "entity": "Hawaii", "label": "LOC", "context": "born in Hawaii." },
    { "entity": "Barack", "label": "PER", "context": "Barack was american." }
]
```
```
```

### [SYSTEM_PROMPT_CONTEXT_MD_SHORT](https://www.notion.so/SEMESTRAL_REPORT-2df6ee137c588015aeb0eed7c35b777d?pvs=21)

```
You are an expert in named entity recognition (NER).  
Extract all named entities from the input text with their **type** and **short surrounding context**.

**Entity labels:**
- PER - names of people (real or fictional, nicknames, usernames, titles, etc.)
- LOC - cities, countries, landmarks, geographical features, addresses
- ORG - companies, institutions, agencies, teams
- MISC - other named entities (events, works of art, nationalities, religions, languages, etc.)

**Rules:**
- Extract only **outermost** entities (no nesting).
- If an entity appears multiple times in different contexts, list each separately.
- `"entity"` must exactly match the substring from the text.
- `"context"` = 4-8 words around the entity (use fewer if at sentence edges).
- Entity must be included in the context snippet.
- If no entities exist, output `[]`.

*Priority rule:* `PER > LOC > ORG > MISC`

**Output format (JSON only):**
```json
[
  { "entity": "ENTITY_TEXT", "label": "ENTITY_LABEL", "context": "CONTEXT_SNIPPET" },
  ...
]
```
```

**Example**
Input: Barack Obama was born in Hawaii. Barack was American.  
Output:
```json
[
  { "entity": "Barack Obama", "label": "PER", "context": "Barack Obama was born" },
  { "entity": "Hawaii", "label": "LOC", "context": "born in Hawaii." },
  { "entity": "Barack", "label": "PER", "context": "Barack was American." }
]
```

Only output the JSON array. No explanations, markdown, or extra text.
```
```


## Appendix B: Code Snippets

### [Entity Alignment Function](https://www.notion.so/SEMESTRAL_REPORT-2df6ee137c588015aeb0eed7c35b777d?pvs=21)

```python
def assign_entities_from_context(full_text_tokens: list, entities: list,
                                  fuzzy: bool, fuzzy_threshold: float = 0.75,
                                  matching_type:str = 'anchor') -> list:
    """
    Assign BIO tags to tokens based on extracted entities with context.

    Args:
        full_text_tokens (list): List of tokens from the full text.
        entities (list): List of dicts from LLM output {"entity": ..., "label": ..., "context": ...}
        fuzzy (bool): Whether to use fuzzy matching for context.
        fuzzy_threshold (float): Similarity threshold for fuzzy matching.
        matching_type (str): Type of fuzzy matching to use: 'anchor', 'partial', 'full'.
        
    Returns:
        list: BIO tags corresponding to full_text_tokens.
    """
    tags = ['O'] * len(full_text_tokens)
    full_text_str = " ".join(full_text_tokens)

    for ent in entities:
        if 'entity' not in ent or 'label' not in ent or 'context' not in ent:
            continue

        entity_text = ent['entity']
        entity_label = ent['label']
        context_text = ent['context']

        if fuzzy:
            context_start, context_end, similarity = find_best_fuzzy_match(
                context_text, full_text_str, threshold=fuzzy_threshold,
                matching_type=matching_type
            )
            if context_start is None:
                continue

            matched_context = full_text_str[context_start:context_end]
            entity_start_in_context = matched_context.lower().find(entity_text.lower())

            if entity_start_in_context == -1:
                entity_start_in_context, _, _ = find_best_fuzzy_match(
                    entity_text, matched_context, fuzzy_threshold,
                    matching_type=matching_type
                )
                if entity_start_in_context is None:
                    continue
        else:
            context_start = full_text_str.lower().find(context_text.lower())
            if context_start == -1:
                continue

            entity_start_in_context = context_text.lower().find(entity_text.lower())
            if entity_start_in_context == -1:
                continue

        # Map to character positions
        entity_char_start = context_start + entity_start_in_context
        entity_char_end = entity_char_start + len(entity_text)

        # Assign BIO tags to overlapping tokens
        char_idx = 0
        for i, tok in enumerate(full_text_tokens):
            tok_start = char_idx
            tok_end = tok_start + len(tok)

            if tok_end > entity_char_start and tok_start < entity_char_end:
                is_first = i == 0 or tags[i-1] == "O" or not tags[i-1].endswith(entity_label)
                tag_prefix = "B-" if is_first else "I-"
                tags[i] = f"{tag_prefix}{entity_label}"

            char_idx += len(tok) + 1

    return tags
```

### [Fuzzy Matching Function](https://www.notion.so/SEMESTRAL_REPORT-2df6ee137c588015aeb0eed7c35b777d?pvs=21)

```python
def find_best_fuzzy_match(context:str, full_text:str, threshold: float = 0.7, matching_type:str = 'anchor') -> tuple:
    """
    Find the best fuzzy match of context in full_text using sliding window.
    
    Args:
        context (str): The context string to match.
        full_text (str): The full text string to search within.
        threshold (float): Minimum similarity ratio to consider a match.
        matching_type (str): Type of fuzzy matching to use: 'anchor', 'partial', 'full'.
        
    Returns:
        (start_pos, end_pos, similarity) if match found above threshold, else (None, None, 0.0).
    """
    context_lower = context.lower()
    full_text_lower = full_text.lower()
    context_len = len(context_lower)

    # First try exact match
    exact_start = full_text_lower.find(context_lower)
    if exact_start != -1:
        return (exact_start, exact_start + context_len, 1.0)
    
    # Fuzzy match with sliding window
    best_ratio = 0.0
    best_start = None
    best_end = None

    if matching_type == 'anchor':   
        # ======== Partial ratio with anchor filtering ========
        anchor_len = 6 if context_len > 10 else 3
        anchor = context_lower[:anchor_len]
        # Find all positions of anchor in full_text
        anchor_starts = []
        pos = full_text_lower.find(anchor)

        if pos == -1:
            for i in range(len(full_text_lower) - anchor_len + 1):
                window = full_text_lower[i:i + anchor_len]
                ratio = fuzz.ratio(anchor, window) / 100.0
                if ratio >= threshold:
                    anchor_starts.append(i)
        else:
            while pos != -1:
                anchor_starts.append(pos)
                pos = full_text_lower.find(anchor, pos + 1)
        # If no anchors found, use regular sliding window
        if not anchor_starts:
            anchor_starts = range(0, len(full_text_lower) - context_len + 1, max(1, context_len // 3))

        for astart in anchor_starts:
            # Check +-15 characters around context
            win_start = max(0, astart - 15)
            win_end = min(len(full_text_lower), astart + context_len + 15)
            window = full_text_lower[win_start:win_end]
            # Compute partial ratio with score cutoff, to find promising windows
            ratio = fuzz.partial_ratio(context_lower, window, score_cutoff=int(threshold*100)) / 100.0
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                # Find best matching substring within window
                best_local_pos = -1
                best_local_score = 0
                # Go through all possible substrings of context length
                for i in range(0, len(window) - context_len + 1):
                    chunk = window[i:i + context_len]
                    # Find ratio with score cutoff to speed up
                    chunk_ratio = fuzz.ratio(context_lower, chunk, score_cutoff=int(best_local_score*100)) / 100.0
                    if chunk_ratio > best_local_score:
                        best_local_score = chunk_ratio
                        best_local_pos = i

                if best_local_pos != -1:
                    best_start = win_start + best_local_pos
                    best_end = best_start + context_len

    if matching_type == 'partial':
        # === Partial ratio with steps around window size ===
        # Slide in steps to reduce number of comparisons
        step = max(2, context_len // 10)

        # Allow window bigger than context to catch punctuation/spacing differences
        min_w = max(5, context_len - 5)
        max_w = context_len + 10

        for window_size in range(min_w, max_w + 1):
            for i in range(0, len(full_text_lower) - window_size + 1, step):
                window = full_text_lower[i:i + window_size]
                ratio = fuzz.partial_ratio(context_lower, window, score_cutoff=int(threshold*100)) / 100.0
                if ratio > best_ratio and ratio >= threshold:
                    best_ratio = ratio
                    best_start = i
                    best_end = i + window_size
        
    if matching_type == 'full':
        # === Full ratio with windows ===
        # Try different window sizes around the context length
        for window_size in range(max(1, context_len - 5), context_len + 10):
            for i in range(len(full_text_lower) - window_size + 1):
                window = full_text_lower[i:i + window_size]
                ratio = fuzz.ratio(context_lower, window) / 100.0
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_start = i
                    best_end = i + window_size

    if best_ratio >= threshold:
        return (best_start, best_end, best_ratio)
    else:
        return (None, None, 0.0)
```

---

## Appendix C: Experimental Configuration

### CoNLL-2003 Evaluation Parameters

- **Examples:** 200 per iteration
- **Iterations:** 3
- **Batch Sizes:** [1, 5, 10, 32]
- **Fuzzy Modes:** [False, True]
- **Fuzzy Threshold:** 0.6
- **Temperature:** 0.2
- **System Prompts:** 3 variants

### DocRed Evaluation Parameters

- **Examples:** 64 per iteration
- **Iterations:** 3
- **Batch Sizes:** [1, 2, 4]
- **Fuzzy Modes:** [False, True]
- **Fuzzy Threshold:** 0.6
- **Temperature:** 0.2
- **System Prompts:** 2 DocRed-specific variants

---