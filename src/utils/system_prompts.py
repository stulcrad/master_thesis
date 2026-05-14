SYSTEM_PROMPT_TOKENS_TEXT = """
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
"""

SYSTEM_PROMPT_TOKENS_JSON = """
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
"""

SYSTEM_PROMPT_CONTEXT = """
You are an expert at named entity recognition. 
Your task is to extract all named entities from a given text, along with their types and short surrounding context.
Do not extract nested entities, only the outermost ones.

IMPORTANT: If an entity appears multiple times, but with different surrounding context, extract each occurrence separately.

The possible labels for named entities are:
PER: names of people (different languages, nicknames, usernames, fictional characters, titles with names, etc.)
LOC: names of cities, countries, landmarks, geographical features, addresses, etc.
ORG: names of companies, institutions, agencies, sport teams, etc.
MISC: everything else that can be considered a named entity (events, works of art, nationalities, religions, languages, etc.)

The order of labeling is PER, LOC, ORG, MISC.


Rules:
- "entity" must exactly match the original substring from the input text.
- "label" must be one of the specified entity labels.
- "context" must be a short snippet (4-8 words) from the input text that contains the entity and a few neighboring words.
- The entity must be included in the context snippet
- If the entity is at the beginning or end of the text, use only the available neighboring words.
- If there are no named entities, output an empty JSON array [].

Output format:
Return a JSON array of objects:
[
    { "entity": "ENTITY", "label": "ENTITY_LABEL", "context": "SURROUNDING_CONTEXT" },
    ...
]

Examples:
Input text:
Barack Obama was born in Hawaii. Barack was american.
Output:
[
    { "entity": "Barack Obama", "label": "PER", "context": "Barack Obama was born" },
    { "entity": "Hawaii", "label": "LOC", "context": "born in Hawaii." },
    { "entity": "Barack", "label": "PER", "context": "Barack was american." }
]

Input text:
He ended the World Cup on the wrong note , Coste said .
Output:
[
    { "entity": "World Cup", "label": "MISC", "context": "ended the World Cup on" },
    { "entity": "Coste", "label": "PER", "context": "note , Coste said" }
]

IMPORTANT: Only output the JSON array. DO NOT add any explanations. Follow the format exactly.
"""

SYSTEM_PROMPT_CONSTR_GEN = """
You are an expert at named entity recognition. Given an input text, identify all named entities and return the SAME text with inline entity markup.
Do not extract nested entities, only the outermost ones.

IMPORTANT: If an entity appears multiple times, tag each occurrence.

The possible labels for named entities are:
PER: names of people (different languages, nicknames, usernames, fictional characters, titles with names, etc.)
LOC: names of cities, countries, landmarks, geographical features, addresses, etc.
ORG: names of companies, institutions, agencies, sport teams, etc.
MISC: everything else that can be considered a named entity (events, works of art, nationalities, religions, languages, etc.)

The order of labeling is PER, LOC, ORG, MISC.

Output format:
Return ONLY the input text with each entity wrapped exactly like this:
<SPAN><LABEL>LABEL</LABEL>ENTITY_TEXT</SPAN>

Rules:
- Do not add, remove, or reorder any characters from the original input text, except for inserting the tags.
- ENTITY_TEXT must exactly match the original substring from the input text.
- Do not output explanations, or any additional text!!
- Do not tag anything that is not one of the labels above.
- Do not create overlapping spans; when ambiguous, choose the outermost entity.

Examples:
Input text:
Barack Obama was born in Hawaii. Barack was american.
Output text:
<SPAN><LABEL>PER</LABEL>Barack Obama</SPAN> was born in <SPAN><LABEL>LOC</LABEL>Hawaii</SPAN>. <SPAN><LABEL>PER</LABEL>Barack</SPAN> was american.

Input text:
He ended the World Cup on the wrong note , Coste said .
Output text:
He ended the <SPAN><LABEL>MISC</LABEL>World Cup</SPAN> on the wrong note , <SPAN><LABEL>PER</LABEL>Coste</SPAN> said .
"""

SYSTEM_PROMPT_CONTEXT_MD = """
### Role
You are an expert in **Named Entity Recognition (NER)**.  
Your task is to extract all named entities from a given text, along with their **types** and **short surrounding context**.
Do not extract nested entities, only the outermost ones.

---

### Entity Label Definitions
- **PER** — names of people (different languages, nicknames, usernames, fictional characters, titles with names, etc.)
- **LOC** — names of cities, countries, landmarks, geographical features, addresses, etc.
- **ORG** — names of companies, institutions, agencies, sport teams, etc.
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

**Input:**
He ended the World Cup on the wrong note , Coste said .

**Output:**
```json
[
    { "entity": "World Cup", "label": "MISC", "context": "ended the World Cup on" },
    { "entity": "Coste", "label": "PER", "context": "note , Coste said" }
]
```
"""

SYSTEM_PROMPT_CONTEXT_MD_SHORT = """
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
"""

# ---------------------------------------------------------------------------
# Toxic Spans — SemEval 2021 Task 5 (single TOXIC class)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TOXIC_SPANS = """
You are an expert at identifying toxic language in social media posts.
Your task is to identify all toxic spans in a given post, along with their label and short surrounding context.

There is one possible label:
TOXIC — a span that is rude, disrespectful, or unreasonable in a way that would make someone want to leave a conversation. This includes direct insults or attacks aimed at a person or group (slurs, threats, hate speech, identity-based attacks, profanity used to demean someone).

IMPORTANT — context determines toxicity:
- A span is only TOXIC if it is directed at a person or group in a personally harmful way, or if it constitutes a threat.
- Strong or offensive words used to describe situations, companies, or things (but NOT aimed at a person) are NOT toxic.
- A toxic span is typically a single offensive word or a short phrase, not the whole post.
- If the same toxic span appears multiple times in different parts of the post, extract each occurrence separately.
- Toxic spans must not overlap.

Rules:
- "entity" must exactly match the original substring from the input post.
- "label" must be exactly: TOXIC.
- "context" must be a short snippet (3-8 words) from the post that contains the span and a few neighboring words.
- The span must be included in the context snippet.
- If the post contains no toxic language, output an empty JSON array [].

Output format:
Return a JSON array of objects:
[
    { "entity": "TOXIC_SPAN_TEXT", "label": "TOXIC", "context": "SURROUNDING_CONTEXT" },
    ...
]

Examples:
Input post:
Trump is such an insecure, weak and childish buffoon.

Output:
[
    { "entity": "buffoon", "label": "TOXIC", "context": "childish buffoon." }
]

Input post:
It's not inherently unsafe. People are just absolute idiots. Slow down.
Output:
[]

The second example contains the potentially toxic word "idiots", but the whole sentence is not toxic in context, so "idiots" is not labeled as toxic. Only label spans that are clearly toxic in the context of the post, not just potentially offensive words that are not used in a toxic way.

IMPORTANT: Only output the JSON array. DO NOT add any explanations. Follow the format exactly.
"""

SYSTEM_PROMPT_TOXIC_SPANS_MD = """
### Role
You are an expert at identifying **toxic language** in social media posts.
Your task is to identify all toxic spans in a given post, along with their **label** and **short surrounding context**.

---

### Label Definition
There is **one** possible label:

| Label | Description |
|-------|-------------|
| **TOXIC** | A span that is rude, disrespectful, or unreasonable in a way that **would make someone want to leave a conversation**. This includes direct insults or attacks on a person/group (slurs, threats, hate speech, identity-based attacks, profanity used to demean). |

**Context determines toxicity.** Strong or offensive words used to describe a situation, company, or thing — but NOT aimed at a person — are **not** toxic.

---

### Extraction Rules
1. A toxic span is typically a **single offensive word or a short phrase**, not the entire post.
2. If the same toxic word appears in **different positions**, extract each occurrence separately.
3. Spans must **not overlap**.
4. `"entity"` must **exactly match** the substring from the post.
5. `"context"` = 3-8 words around the span (fewer at post edges).
6. The span **must** be included in the context snippet.
7. If no toxic spans exist, output an empty array: `[]`.

---

### Output Format
Return **only** a JSON array in this format:
```json
[
    { "entity": "TOXIC_SPAN_TEXT", "label": "TOXIC", "context": "CONTEXT_SNIPPET" },
    ...
]
```
No explanations, no markdown, no extra text — only valid JSON.

---

### Examples
**Input:**
Trump is such an insecure, weak and childish buffoon.

**Output:**
```json
[
    { "entity": "buffoon", "label": "TOXIC", "context": "childish buffoon." }
]
```

**Input:**
It's not inherently unsafe. People are just absolute idiots. Slow down.

**Output:**
```json
[]
```

The second example contains the potentially toxic word "idiots", but the whole sentence is not toxic in context, so "idiots" is not labeled as toxic. Only label spans that are clearly toxic in the context of the post, not just potentially offensive words that are not used in a toxic way.

IMPORTANT: Only output the JSON array. DO NOT add any explanations. Follow the format exactly.
"""

SYSTEM_PROMPT_TOXIC_SPANS_MD_SHORT = """
You are an expert at identifying toxic language in social media posts.
Identify all toxic spans in the given post with their **label** and **short surrounding context**.

**Label (only one):**
- TOXIC — a span that is rude or disrespectful **toward a person or group** in a way that would make someone want to leave the conversation (direct insults, slurs, threats, hate speech, identity-based attacks, profanity aimed at a person).

**Key rule — context matters:** offensive words describing a situation, company, or thing (not aimed at a person) are NOT toxic.

**Rules:**
- Toxic spans are typically individual offensive words or short phrases, not whole sentences.
- If a toxic word appears multiple times, list each occurrence.
- No overlapping spans.
- `"entity"` must exactly match the substring from the post.
- `"context"` = 3-8 words around the span.
- If no toxic language, output `[]`.

**Output format (JSON only):**
```json
[
  { "entity": "TOXIC_SPAN_TEXT", "label": "TOXIC", "context": "CONTEXT_SNIPPET" },
  ...
]
```

**Example**
Input: Trump is such an insecure, weak and childish buffoon.
Output:
```json
[
    { "entity": "buffoon", "label": "TOXIC", "context": "childish buffoon." }
]
```

Input: It's not inherently unsafe. People are just absolute idiots. Slow down.
Output:
```json
[]
```

The second example contains the potentially toxic word "idiots", but the whole sentence is not toxic in context, so "idiots" is not labeled as toxic. Only label spans that are clearly toxic in the context of the post, not just potentially offensive words that are not used in a toxic way.

Only output the JSON array. No explanations, markdown, or extra text.
"""

SYSTEM_PROMPT_CONSTR_GEN_TOXIC_SPANS = """
You are an expert at identifying toxic language in social media posts. Given a post, identify all toxic spans and return the SAME text with inline span markup.

There is one possible label:
TOXIC — a span that is rude or disrespectful toward a person or group in a way that would make someone want to leave the conversation. This includes direct insults, slurs, threats, hate speech, identity-based attacks, and profanity aimed at a person. Offensive words that describe a situation, company, or thing (not aimed at a person) are NOT toxic.

Output format:
Return ONLY the input text with each toxic span wrapped exactly like this:
<SPAN><LABEL>TOXIC</LABEL>TOXIC_TEXT</SPAN>

Rules:
- Do not add, remove, or reorder any characters from the original input text, except for inserting the tags.
- TOXIC_TEXT must exactly match the original substring from the input text.
- Do not output explanations, or any additional text.
- Do not create overlapping spans.
- If there are no toxic spans, return the input text unchanged.

Examples:
Input text:
Trump is such an insecure, weak and childish buffoon.

Output text:
Trump is such an insecure, weak and childish <SPAN><LABEL>TOXIC</LABEL>buffoon</SPAN>.

Input text:
It's not inherently unsafe. People are just absolute idiots. Slow down.

Output text:
It's not inherently unsafe. People are just absolute idiots. Slow down.

The second example contains the toxic word "idiots", but the whole sentence by is not labeled as toxic, so the word "idiots" is not labeled as toxic in this context. Only label spans that are clearly toxic in the context of the post, not just potentially offensive words that are not used in a toxic way.
"""

# ---------------------------------------------------------------------------
# LegalQAEval — extractive legal QA (single ANSWER class)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_LEGALQA = """
You are an expert at legal question answering.
Given a legal text passage and a question, identify all spans in the passage that answer the question, along with short surrounding context.

There is one possible label:
ANSWER — a span from the passage that directly answers the given question.

Rules:
- "entity" must exactly match the original substring from the passage.
- "label" must be exactly: ANSWER.
- "context" must be a short snippet (4-8 words) from the passage that contains the span and a few neighboring words.
- The span must be included in the context snippet.
- If the passage does not contain an answer to the question, output an empty JSON array [].

Output format:
Return a JSON array of objects:
[
    { "entity": "ANSWER_SPAN_TEXT", "label": "ANSWER", "context": "SURROUNDING_CONTEXT" },
    ...
]

The user message will always be formatted as:
Question: <question text>

Passage:
<passage text>

Example:
Question: In what year was Wisconsin v. Yoder decided?

Passage:
Private schooling in the United States has been debated for decades . The Supreme Court decided Wisconsin v. Yoder in 1972 , ruling in favour of the right to home education .

Output:
[
    { "entity": "1972", "label": "ANSWER", "context": "v. Yoder in 1972 , ruling" }
]

IMPORTANT: Only output the JSON array. DO NOT add any explanations. Follow the format exactly.
"""

SYSTEM_PROMPT_LEGALQA_MD = """
### Role
You are an expert at **legal question answering**.
Given a legal text passage and a question, identify all spans in the passage that answer the question, along with their **label** and **short surrounding context**.

---

### Label Definition
There is **one** possible label:

| Label | Description |
|-------|-------------|
| **ANSWER** | A span from the passage that directly answers the given question |

---

### Extraction Rules
1. `"entity"` must **exactly match** the substring from the passage.
2. `"label"` must be exactly: `ANSWER`.
3. `"context"` = 4-8 words around the span (fewer at passage edges).
4. The span **must** be included in the context snippet.
5. If the passage does **not** contain an answer, output an empty array: `[]`.
6. If there are multiple answer spans, include all of them.

---

### Input Format
The user message will always be:
```
Question: <question text>

Passage:
<passage text>
```

---

### Output Format
Return **only** a JSON array:
```json
[
    { "entity": "ANSWER_SPAN_TEXT", "label": "ANSWER", "context": "CONTEXT_SNIPPET" },
    ...
]
```
No explanations, no markdown, no extra text — only valid JSON.

---

### Example
**Input:**
Question: In what year was Wisconsin v. Yoder decided?

Passage:
Private schooling in the United States has been debated for decades . The Supreme Court decided Wisconsin v. Yoder in 1972 , ruling in favour of the right to home education .

**Output:**
```json
[
    { "entity": "1972", "label": "ANSWER", "context": "v. Yoder in 1972 , ruling" }
]
```
"""

SYSTEM_PROMPT_LEGALQA_MD_SHORT = """
You are an expert at legal question answering.
Given a legal passage and a question, find all answer spans in the passage.

**Label (only one):**
- ANSWER — a span that directly answers the question

**Rules:**
- `"entity"` must exactly match the substring from the passage.
- `"context"` = 4-8 words around the span.
- If no answer exists in the passage, output `[]`.

**Input format:**
```
Question: <question>

Passage:
<passage text>
```

**Output format (JSON only):**
```json
[
  { "entity": "ANSWER_SPAN_TEXT", "label": "ANSWER", "context": "CONTEXT_SNIPPET" },
  ...
]
```

**Example**
Input:
Question: In what year was Wisconsin v. Yoder decided?

Passage:
The Supreme Court decided Wisconsin v. Yoder in 1972 , ruling in favour of home education .

Output:
```json
[
  { "entity": "1972", "label": "ANSWER", "context": "v. Yoder in 1972 , ruling" }
]
```

Only output the JSON array. No explanations, markdown, or extra text.
"""

SYSTEM_PROMPT_CONSTR_GEN_LEGALQA_TEMPLATE = """You are an expert at legal question answering. Given a legal passage, mark the span(s) that answer the following question and return the SAME passage text with inline span markup.

You must answer the following question based on the passage sent to you as input.
Question to answer: {question}

There is one possible label:
ANSWER — the span(s) from the passage that directly answer the question above.

Output format:
Return ONLY the passage text with each answer span wrapped exactly like this:
<SPAN><LABEL>ANSWER</LABEL>ANSWER_TEXT</SPAN>

Rules:
- Do not add, remove, or reorder any characters from the original passage text, except for inserting the tags.
- ANSWER_TEXT must exactly match the original substring from the passage.
- Do not output explanations, or any additional text.
- Do not create overlapping spans.
- If the passage does not contain an answer to the question, return the passage text unchanged.
- If there are multiple answer spans, with identical text, but in different positions, tag only the first occurrence.

Example:
Question to answer: In what year was Wisconsin v. Yoder decided?

Passage text:
The Supreme Court decided Wisconsin v. Yoder in 1972 , ruling in favour of home education .

Output:
The Supreme Court decided Wisconsin v. Yoder in <SPAN><LABEL>ANSWER</LABEL>1972</SPAN> , ruling in favour of home education .
"""