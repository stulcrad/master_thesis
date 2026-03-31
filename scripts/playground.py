from huggingface_hub import login
login(token="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")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, BitsAndBytesConfig, Mxfp4Config
from utils.system_prompts import SYSTEM_PROMPT_CONSTR_GEN

# model_id = 'meta-llama/Llama-3.1-8B-Instruct'
# model_id = 'Qwen/Qwen3-8B'
model_id = 'google/gemma-3-4b-it'
# model_id = 'openai/gpt-oss-20b'
# model_id = 'unsloth/gpt-oss-20b-GGUF'

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
# quantization_config = Mxfp4Config()
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", torch_dtype="auto", 
    quantization_config=quantization_config
)
# model = AutoModelForCausalLM.from_pretrained(model_id)
# model.eval()

class FSMConstrainedProcessor(LogitsProcessor):
    def __init__(self, labels: list[str], input_text: str = "", 
                 tokenizer: AutoTokenizer = tokenizer):
        # Initialize with the tokenizer and model to access the vocabulary
        self.tokenizer = tokenizer
        self.labels = labels

        # Build a mapping from surface forms to token IDs
        self.surface_to_ids = {}
        vocab = tokenizer.get_vocab()
        # Get the correct word start prefix for the tokenizer (e.g., "Ġ" for GPT-2)
        self.word_start_prefix = self._get_word_start_prefix(tokenizer)

        for token, token_id in vocab.items():
            # Remove leading prefix which indicates the start of a new word in many tokenizers
            if token.startswith(self.word_start_prefix):
                surface = token[len(self.word_start_prefix):]
            else:
                surface = token
            # Map the surface form to the token ID
            if surface not in self.surface_to_ids:
                self.surface_to_ids[surface] = set()
            self.surface_to_ids[surface].add(token_id)

        # Tokenize input text
        self.input_token_ids = tokenizer.encode(input_text, add_special_tokens=False)
        self.allowed_tokens = set()
        # Add all token IDs corresponding to the surface forms in the input text into the allowed tokens
        for token_id in self.input_token_ids:
            token = tokenizer.convert_ids_to_tokens(token_id)
            if token.startswith(self.word_start_prefix):
                surface = token[len(self.word_start_prefix):]
            else:
                surface = token

            self.allowed_tokens.update(self.surface_to_ids[surface])

        # Lastly, chech that for all allowed tokens, if the token is prefixed,
        # that without the prefix the token is tokenized in the same way
        # If not, add into the allowed tokens also the tokenized version without the prefix
        self.diff_words_map = {} # A mapping from token IDs that are prefixed and have a diff tokenized version without the prefix
        for token_id in list(self.allowed_tokens):
            token = self.tokenizer.convert_ids_to_tokens(token_id)
            if token.startswith(self.word_start_prefix):
                tokenized_without_prefix = self.tokenizer.tokenize(token[1:])
                tokenized_without_prefix_ids = self.tokenizer.convert_tokens_to_ids(tokenized_without_prefix)
                if tokenized_without_prefix_ids != [token_id]:
                    # If the tokenized version without the prefix is different, add it to the allowed tokens
                    # and also save the mapping for check that the input and generated output are the same surface form
                    self.diff_words_map[token_id] = tokenized_without_prefix_ids
                    self.allowed_tokens.update(tokenized_without_prefix_ids)

        # Also allow token for the end of sequence
        # EOS tokens, for some reason some models use different tokens to indicate
        # end of sequence other then tokenizer.eos_token_id
        self.eos_token_ids = [self.tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        possible_eos_tokens = ["<end_of_turn>", "<|im_end|>", "<|eot_id|>"]

        for eos_token in possible_eos_tokens:
            if eos_token in tokenizer.get_vocab():
                self.eos_token_ids.append(tokenizer.convert_tokens_to_ids(eos_token))
                break

        # self.allowed_tokens.update(self.eos_token_ids)

        # FSM state
        self.state = "OUTSIDE"
        # 
        self.sequence = None
        self.tag_pointer = 0
        self.first_word_in_span = True
        self.input_length = len(self.input_token_ids)

        self.input_coverage_pointer = 0
        self._diff_sub_pointer = 0

        self.span_open = self.tokenizer.encode("<SPAN>", add_special_tokens=False)
        # Make sure the first part of the span open tokens is prefixed with the word start prefix
        if not self.tokenizer.convert_ids_to_tokens(self.span_open[0]).startswith(self.word_start_prefix):
            prefixed_span_open_token = self.word_start_prefix + self.tokenizer.convert_ids_to_tokens(self.span_open[0])
            self.span_open.append(tokenizer.convert_tokens_to_ids(prefixed_span_open_token))
        else:
            self.span_open.append(self.span_open[0][1])

        self.span_close = self.tokenizer.encode("</SPAN>", add_special_tokens=False)
        self.label_open = self.tokenizer.encode("<LABEL>", add_special_tokens=False)
        self.label_close = self.tokenizer.encode("</LABEL>", add_special_tokens=False)
        # all other tokens should not be prefixed with the word start prefix
        for seq in [self.span_close, self.label_open, self.label_close]:
            if self.tokenizer.convert_ids_to_tokens(seq[0]).startswith(self.word_start_prefix):
                seq[0] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.convert_ids_to_tokens(seq[0])[len(self.word_start_prefix):])

        self.label_token_ids = {
            label: self.tokenizer.encode(label, add_special_tokens=False) 
            for label in labels
        }
        # Same for label tokens - they should not be prefixed with the word start prefix
        for seq in self.label_token_ids.values():
            if self.tokenizer.convert_ids_to_tokens(seq[0]).startswith(self.word_start_prefix):
                seq[0] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.convert_ids_to_tokens(seq[0])[len(self.word_start_prefix):])
        self.allowed_label_tokens = set(token_id for seq in self.label_token_ids.values() for token_id in seq)

    def _advance_input_coverage(self, last_token_id: int) -> None:
        """Advance the input coverage pointer if the given token matches the current expected input token."""
        if self.input_coverage_pointer >= self.input_length:
            return

        expected_token_id = self.input_token_ids[self.input_coverage_pointer]

        # Direct match (prefixed or not)
        if last_token_id == expected_token_id:
            self.input_coverage_pointer += 1
            self._diff_sub_pointer = 0
            return

        # Check if the expected token has a diff_words_map entry and last_token_id starts that sequence
        if expected_token_id in self.diff_words_map:
            unprefixed_seq = self.diff_words_map[expected_token_id]

            if last_token_id == unprefixed_seq[self._diff_sub_pointer]:
                self._diff_sub_pointer += 1
                # If we have matched the entire unprefixed sequence, advance the main pointer and reset the sub pointer
                if self._diff_sub_pointer == len(unprefixed_seq):
                    self.input_coverage_pointer += 1
                    self._diff_sub_pointer = 0
                return

    def _all_input_covered(self) -> bool:
        return self.input_coverage_pointer >= self.input_length

    def _get_word_start_prefix(self, tokenizer) -> str:
        # tokenize "a b" to always have prefix before the 'b' character
        tokens = tokenizer.tokenize("a b")
        second_token = tokens[1] # will be something like "▁b" or "Ġb"

        if len(second_token) > 1:
            prefix = second_token[0] # the prefix character
            return prefix
        
        return ""
        
    def _mask_except(self, scores: torch.FloatTensor, allowed_tokens: set[int]) -> None:
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask[:, list(allowed_tokens)] = False
        scores.masked_fill_(mask, -float("inf"))

    def ends_with(self, input_ids_list: list[int], token_seq: list[int]) -> bool:
        if len(input_ids_list) < len(token_seq):
            return False
        return input_ids_list[-len(token_seq):] == token_seq

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # last_token = input_ids[0, -1].item()
        input_ids_list = input_ids[0].tolist()
        last_token_id = input_ids_list[-1]
        second_last_token_id = input_ids_list[-2] if len(input_ids_list) > 1 else None

        self._advance_input_coverage(last_token_id)

        # --- STATE TRANSITIONS ---
        if self.state == "OUTSIDE":
            if last_token_id == self.span_open[0] or last_token_id == self.span_open[3]:
                self.state = "MAYBE_SPAN_OPEN"

        elif self.state == "MAYBE_SPAN_OPEN":
            if (second_last_token_id == self.span_open[0] and last_token_id == self.span_open[1]) \
                or (second_last_token_id == self.span_open[3] and last_token_id == self.span_open[1]):
                self.state = "SPAN_OPEN_LAST_TOKEN"

        elif self.state == "SPAN_OPEN_LAST_TOKEN":
            if last_token_id == self.span_open[2]:
                self.tag_pointer = 0
                self.state = "LABEL_OPEN"

        elif self.state == "LABEL_OPEN":
            if last_token_id == self.label_open[-1]:
                self.state = "LABEL_VALUE"

        elif self.state == "LABEL_VALUE":
            if any(self.ends_with(input_ids_list, seq)
                   for seq in self.label_token_ids.values()):
                self.tag_pointer = 0
                self.state = "LABEL_CLOSE"

        elif self.state == "LABEL_CLOSE":
            if last_token_id == self.label_close[-1]:
                self.state = "SPAN_TEXT"
                self.first_word_in_span = True

        elif self.state == "SPAN_TEXT":
            if last_token_id == self.span_close[0]:
                self.state = "MAYBE_SPAN_CLOSE"

        elif self.state == "MAYBE_SPAN_CLOSE":
            if second_last_token_id == self.span_close[0] and last_token_id == self.span_close[1]:
                self.state = "SPAN_CLOSE"

        elif self.state == "SPAN_CLOSE":
            if last_token_id == self.span_close[2]:
                self.state = "OUTSIDE"

        # --- CONSTRAINTS ---
        def add_eos_if_done(allowed_tokens: set[int]) -> set:
            if self._all_input_covered():
                allowed_tokens.update(self.eos_token_ids)
            return allowed_tokens

        if self.state == "OUTSIDE":
            # Only allow tokens from the input text or span open tokens
            allowed_tokens = self.allowed_tokens.copy()
            allowed_tokens.add(self.span_open[0])
            allowed_tokens.add(self.span_open[3])
            allowed_tokens = add_eos_if_done(allowed_tokens)
            self._mask_except(scores, allowed_tokens)

        elif self.state == "MAYBE_SPAN_OPEN":
            # Only allow tokens from the input text or the next span_open token
            allowed_tokens = self.allowed_tokens.copy()
            allowed_tokens.add(self.span_open[1]) # only allow the second token of the span open sequence
            self._mask_except(scores, allowed_tokens)

        elif self.state == "SPAN_OPEN_LAST_TOKEN":
            # Only allow last token of the span open sequence
            allowed_tokens = set()
            allowed_tokens.add(self.span_open[2])
            self._mask_except(scores, allowed_tokens)

        elif self.state == "LABEL_OPEN":
            allowed_tokens = set()
            allowed_tokens.add(self.label_open[self.tag_pointer]) # only allow the next token in the label open sequence
            self._mask_except(scores, allowed_tokens)
            self.tag_pointer += 1

        elif self.state == "LABEL_VALUE":
            # Only allow label value tokens
            allowed_tokens = self.allowed_label_tokens
            self._mask_except(scores, allowed_tokens)

        elif self.state == "LABEL_CLOSE":
            # Only allow label close tokens
            allowed_tokens = set()
            allowed_tokens.add(self.label_close[self.tag_pointer]) # only allow the next token in the
            self._mask_except(scores, allowed_tokens)
            self.tag_pointer += 1

        elif self.state == "SPAN_TEXT":
            # Only allow tokens from the input text or span close tokens
            allowed_tokens = self.allowed_tokens.copy()
            if self.first_word_in_span:
                # If it is the first word in the span, it cannot have a new word start prefix
                for token_id in list(allowed_tokens):
                    token = self.tokenizer.convert_ids_to_tokens(token_id)
                    if token.startswith(self.word_start_prefix):
                        allowed_tokens.remove(token_id)
                self.first_word_in_span = False
            allowed_tokens.add(self.span_close[0]) # only allow the first token of the span close sequence
            self._mask_except(scores, allowed_tokens)

        elif self.state == "MAYBE_SPAN_CLOSE":
            # Same as with MAYBE_SPAN_OPEN
            allowed_tokens = self.allowed_tokens.copy()
            allowed_tokens.add(self.span_close[1])
            self._mask_except(scores, allowed_tokens)

        elif self.state == "SPAN_CLOSE":
            allowed_tokens = set()
            allowed_tokens.add(self.span_close[2])
            self._mask_except(scores, allowed_tokens)

        # # Show which token will be generated next (for debugging)
        # next_token_id = torch.argmax(scores, dim=-1).item()
        # print(next_token_id)
        # print(f"State: {self.state}, Next token: {self.tokenizer.decode(next_token_id)}")
        # scores_ordered = torch.argsort(scores, dim=-1, descending=True)
        # second_most_likely_token_id = scores_ordered[0, 1].item()
        # print(f"Second most likely token: {self.tokenizer.decode(second_most_likely_token_id)}")


        return scores
    
    
labels = ["PER", "LOC", "ORG", "MISC"]

input_text = "Radek was born in Prague."

processor = FSMConstrainedProcessor(labels, input_text, tokenizer)

def generate_from_chat_models(input_text: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_CONSTR_GEN},
        {"role": "user", "content": input_text}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        reasoning_effort='low'
    )

    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    outputs_constrained = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.2,
        do_sample=True,
        logits_processor=[processor],
    )

    outputs_unconstrained = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.2,
        do_sample=True,
    )

    output_ids_constrained = outputs_constrained[0][len(inputs.input_ids[0]):].tolist()
    output_ids_unconstrained = outputs_unconstrained[0][len(inputs.input_ids[0]):].tolist() 

    print(tokenizer.batch_decode(output_ids_constrained, skip_special_tokens=True)[0])
    return output_ids_constrained, output_ids_unconstrained

output_ids_constrained, output_ids_unconstrained = generate_from_chat_models(input_text)
