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

SYSTEM_PROMPT_DOCRED = """
You are an expert at named entity recognition. Given an input text, extract all named entities along with their types and surrounding context.
Do not extract nested entities, only the outermost ones.

IMPORTANT: If an entity appears multiple times, but with different surrounding context, extract each occurrence separately.

The possible labels for named entities are:
PER: names of people (different languages, nicknames, usernames, fictional characters, titles with names, etc.)
LOC: names of cities, countries, landmarks, geographical features, addresses, etc.
ORG: names of companies, institutions, agencies, teams, etc.
NUM: numerical expressions (counts, quantities, ages, etc.) with units (if applicable).
TIME: temporal expressions (dates, times, durations, years, etc.).
MISC: everythin else that could be considered a named entity (languages, nationalities, etc.).

The order of labeling is PER, LOC, ORG, NUM, TIME, MISC.

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
Input text: "John Doe , a software engineer at OpenAI , moved to San Francisco on September 6 , 2020 . He is 30 years old ."
Output:
```json
[
    { "entity": "John Doe", "label": "PER", "context": "John Doe , a software engineer" },
    { "entity": "OpenAI", "label": "ORG", "context": "software engineer at OpenAI , moved" },
    { "entity": "San Francisco", "label": "LOC", "context": "moved to San Francisco on September" },
    { "entity": "September 6 , 2020", "label": "TIME", "context": "on September 6 , 2020" },
    { "entity": "30 years", "label": "NUM", "context": "He is 30 years old ." }
]
```

IMPORTANT: Only output the JSON array. DO NOT add any explanations. Follow the format exactly.
"""

SYSTEM_PROMPT_DOCRED_MD = """
### Role
You are an expert in **Named Entity Recognition (NER)**.  
Your task is to extract all named entities from a given text, along with their **types** and **short surrounding context**.

### Entity Label Definitions
- **PER** — people's names (real or fictional, including titles, nicknames, usernames, etc.)
- **LOC** — cities, countries, landmarks, geographical features, or addresses
- **ORG** — companies, institutions, agencies, or teams
- **NUM** — numerical expressions (counts, quantities, ages, etc.) with units (if applicable)
- **TIME** — temporal expressions (dates, times, durations, years, etc.)
- **MISC** — everything else that can be considered a named entity (events, works of art, nationalities, religions, languages, etc.)

*Priority rule:* `PER > LOC > ORG > NUM > TIME > MISC`

### Extraction Rules
1. Extract **only the outermost entities** — do not include nested entities.
2. If an entity appears multiple times in different contexts, extract **each occurrence separately**.
3. For each entity, include a **short context snippet (4-8 words)** containing the entity and nearby words.
4. The entity **must** be part of the context snippet.
5. The `"entity"` text must **exactly match** the substring from the input.
6. If no entities are found, output an empty array: `[]`.

### Output Format
Return **only** a JSON array in this format:
```json
[
    { "entity": "ENTITY", "label": "ENTITY_LABEL", "context": "CONTEXT_SNIPPET" },
    ...
]
```

No explanations, no markdown, no extra text — only valid JSON.

### Example
**Input:**
John Doe , a software engineer at OpenAI , moved to San Francisco on September 6 , 2020 . He is 30 years old .
**Output:**
```json
[
    { "entity": "John Doe", "label": "PER", "context": "John Doe , a software engineer" },
    { "entity": "OpenAI", "label": "ORG", "context": "software engineer at OpenAI , moved" },
    { "entity": "San Francisco", "label": "LOC", "context": "moved to San Francisco on September" },
    { "entity": "September 6 , 2020", "label": "TIME", "context": "on September 6 , 2020" },
    { "entity": "30 years", "label": "NUM", "context": "He is 30 years old ." }
]
```
"""