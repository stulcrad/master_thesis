from transformers import LogitsProcessor, AutoTokenizer
from typing import Optional, Set
import torch
from utils.TokTrie import TokTrie, build_toktrie_from_tokenizer

class TrieSpanConstrainedProcessor(LogitsProcessor):
    """
    Constrained generation processor for span classification with generative LLMs.

    Behavior
    --------
    1. Outside tags, generation must copy the input text exactly (byte-wise) using the token trie
    to find all possible prefix tokens for the remaining byte suffix of the now generated token.
    2. The model may open a span and emit <SPAN><LABEL>...</LABEL>...</SPAN>.
    3. Text inside the spans is still constrained to copy the input text exactly.

    This gives probabilistic token choices while guaranteeing that removing tags reconstructs the 
    original input text exactly.
    """
    def __init__(self, labels: list[str],  input_text: str, tokenizer: AutoTokenizer,
                 toktrie: Optional[TokTrie] = None):
        """
        Initialize the processor.
        
        Args
        ----
        - labels: list[str] -> list of all possible span labels, e.g. ["PER", "LOC", "ORG"] for NER
        - input_text: str -> the input text to be classified; used for building the token trie constraints
        - tokenizer: AutoTokenizer -> the tokenizer corresponding to the model being used; needed to build the token trie
        - toktrie: TokTrie -> pre-built token trie for the tokenizer; if not provided, it will be built from the tokenizer
        """
        # Store the labels for constructing the control tokens for opening spans.
        self.labels = labels


        # Store the tokenizer and token trie for constraint logic.
        self.tokenizer = tokenizer
        self.toktrie = toktrie if toktrie is not None else build_toktrie_from_tokenizer(tokenizer)

        # Store the input text and its byte representation for tracking how much of the input has been copied so far.
        self.input_text = input_text
        self.input_tokens = tokenizer.encode(input_text, add_special_tokens=False)
        # Some input bytes may not be in the vocab, so we need to first encode the input text to tokens,
        # then convert those tokens back to bytes and concatenate to get the full byte string of the input text.
        self.input_bytes = b""
        for token_id in self.input_tokens:
            token_bytes = self.toktrie.token_id_to_bytes.get(token_id)
            if token_bytes is not None:
                self.input_bytes += token_bytes
        # self.input_bytes = input_text.encode("utf-8")
        self.input_pos = 0 # to track the current byte position in the input text
        
        # Runtime generation bookkeeping.
        self.STATE = "OUTSIDE"
        self.seq_pos = 0  # used only for SPAN_CLOSE sequencing
        self.prev_len = 0 # track the len of the generated seq

        # Per-label position tracking for TAG_BLOCK disambiguation.
        # Maps label -> current position within that label's open block token sequence.
        # Blocks whose token doesn't match the emitted token are dropped immediately,
        # preventing tokens from eliminated blocks from polluting the allowed set.
        self.live_blocks: Optional[dict] = None

        # Tracks whether at least one copy token was emitted in the current span body.
        self.span_text_has_content = False

        # Structural tokens for the constrained generation format
        self.SPAN_CLOSE = self.tokenizer.encode("</SPAN>", add_special_tokens=False)

        # Pre-encode the label tokens for quick access during generation (with and without space)
        self.label_open_blocks = {
            label: tokenizer.encode(f" <SPAN><LABEL>{label}</LABEL>", add_special_tokens=False)
            for label in labels
        }
        self.label_open_blocks_nospace = {
            label: tokenizer.encode(f"<SPAN><LABEL>{label}</LABEL>", add_special_tokens=False)
            for label in labels
        }
        self.selected_label = None  # set when entering SPAN_TEXT, not during TAG_BLOCK
        self._active_blocks = self.label_open_blocks  # which variant (space / no-space) is active

        # Set of token IDs that can end the generation (e.g. EOS tokens)
        self.eos_token_ids: Set[int] = set()
        if self.tokenizer.eos_token_id is not None:
            self.eos_token_ids.add(self.tokenizer.eos_token_id)
        for tok in ["<end_of_turn>", "<|im_end|>", "<|eot_id|>"]:
            tok_id = tokenizer.convert_tokens_to_ids(tok)
            if tok_id is not None and tok_id != tokenizer.unk_token_id:
                self.eos_token_ids.add(tok_id)

    def reset(self):
        """"
        Reset the processor state for a new generation sequence.
        """
        self.STATE = "OUTSIDE"
        self.seq_pos = 0
        self.input_pos = 0
        self.selected_label = None
        self.live_blocks = None
        self.span_text_has_content = False
        self.prev_len = 0
        self._active_blocks = None

    def _mask_except(self, scores: torch.FloatTensor, allowed_tokens: Set[int]) -> torch.FloatTensor:
        """
        Mask the scores to only allow the specified token IDs in allowed_tokens, setting all other token scores to -inf.
        """
        if not allowed_tokens:
            # Avoid all -inf rows, which break sampling (nan/inf probabilities).
            return scores
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask[:, list(allowed_tokens)] = False
        scores = scores.masked_fill(mask, -float("inf"))
        return scores
    
    def _remaining_bytes(self) -> bytes:
        """
        Get the remaining byte suffix of the input text that has not been copied yet, 
        starting from the current input position.
        """
        return self.input_bytes[self.input_pos:]
    
    def _allowed_copy_tokens(self) -> Set[int]:
        """
        Get the set of token IDs that can be emitted to copy the next part of the input text, 
        based on the remaining byte suffix and the token trie.
        """
        return self.toktrie.prefix_search(self._remaining_bytes())

    def _prefer_literal_angle_bracket(self) -> bool:
        """
        If the source text currently starts with '<', prevent starting a control tag at this step.
        This avoids confusing literal '<' in text with control token '<SPAN>'.
        """
        return self._remaining_bytes().startswith(b"<")
    
    def _all_input_consumed(self) -> bool:
        """
        Check if all input bytes have been consumed (i.e. copied) so far, based on the current input position.
        """
        return self.input_pos >= len(self.input_bytes)
    
    def _consume_copy_token(self, token_id: int) -> bool:
        """
        Consume a copy token and advance the input position 
        if the given token ID corresponds to a token that can copy the next part of the input text.
        """
        if self._all_input_consumed():
            return False
        token_bytes = self.toktrie.token_id_to_bytes.get(token_id)
        if not token_bytes:
            return False
        if self._remaining_bytes().startswith(token_bytes):
            self.input_pos += len(token_bytes)
            return True
        return False
    
    def _advance_state(self, token_id: int) -> None:
        """
        Advance the FSM state based on the emitted token ID, updating the current state, sequence position, selected label, 
        and span text content flag as needed according to the constrained generation logic.
        """
        if self.STATE == "OUTSIDE":
            # First check if the last emitted token is a copy token, and if so, consume it and advance the input position accordingly.
            if self._consume_copy_token(token_id):
                return
            # Enter atomic tag block once any block-start token is emitted
            space_match = any(block and token_id == block[0] for block in self.label_open_blocks.values())
            nospace_match = any(block and token_id == block[0] for block in self.label_open_blocks_nospace.values())
            if space_match:
                if self._remaining_bytes().startswith(b" "):
                    self.input_pos += 1
                else:
                    print(f"Warning: space-prefixed block chosen but no space at input_pos {self.input_pos}")
                self.STATE = "TAG_BLOCK"
                self.live_blocks = {label: 1 for label in self.labels}
                self.selected_label = None
                self._active_blocks = self.label_open_blocks
                return
            if nospace_match:
                self.STATE = "TAG_BLOCK"
                self.live_blocks = {label: 1 for label in self.labels}
                self.selected_label = None
                self._active_blocks = self.label_open_blocks_nospace
                return
            return

        if self.STATE == "TAG_BLOCK":
            # Advance each live block if its next expected token matches the emitted token,
            # and drop blocks that do not match. This prevents tokens from eliminated blocks
            # (e.g. "ISC" from the MISC block after ">" was chosen instead of ">M") from
            # appearing in the allowed set at subsequent steps.
            new_live = {}
            for label, pos in self.live_blocks.items():
                block = self._active_blocks[label]
                if pos < len(block) and block[pos] == token_id:
                    new_pos = pos + 1
                    if new_pos == len(block):
                        # This label's open block is fully emitted: transition to SPAN_TEXT.
                        self.STATE = "SPAN_TEXT"
                        self.selected_label = label
                        self.live_blocks = None
                        self.seq_pos = 0
                        self.span_text_has_content = False
                        return
                    new_live[label] = new_pos
            self.live_blocks = new_live
            return
        
        if self.STATE == "SPAN_TEXT":
            # First check if the last emitted token is a copy token, and if so, consume it and advance the input position accordingly.
            if self._consume_copy_token(token_id):
                self.span_text_has_content = True
                return
            # If the emitted token is not a copy token, it can only be the start of the closing tag
            if token_id == self.SPAN_CLOSE[0]:
                self.STATE = "SPAN_CLOSE"
                self.seq_pos = 1
                return
            return
        
        if self.STATE == "SPAN_CLOSE":
            # Advance through the span close sequence until it is complete, then return to OUTSIDE state
            if token_id == self.SPAN_CLOSE[self.seq_pos]:
                self.seq_pos += 1
                if self.seq_pos == len(self.SPAN_CLOSE):
                    self.STATE = "OUTSIDE"
                    self.seq_pos = 0
                    self.span_text_has_content = False
                return
            return

    def _allowed_tokens(self) -> Set[int]:
        """
        Get the set of allowed token IDs for the next generation step based on the current FSM state and seq position, 
        using the token trie to find valid copy tokens for the remaining input bytes, and allowing the appropriate special tokens
        """
        if self.STATE == "OUTSIDE":
            # Allow all tokens that can copy the next part of the input text
            allowed = self._allowed_copy_tokens()
            # Additionally, allow the tokens that can start any of the label blocks, unless the next part of the input text starts with '<'
            if not self._prefer_literal_angle_bracket():
                if self._remaining_bytes().startswith(b" "):
                    allowed.update(tok[0] for tok in self.label_open_blocks.values())
                    self._active_blocks = self.label_open_blocks # pre-set the active blocks so _advance_state sees the correct variant
                else:
                    allowed.update(tok[0] for tok in self.label_open_blocks_nospace.values())
                    self._active_blocks = self.label_open_blocks_nospace # pre-set the active blocks so _advance_state sees the correct variant
            # If all input has been consumed, allow only EOS tokens to end the generation.
            if self._all_input_consumed():
                allowed = self.eos_token_ids
            return allowed
        
        if self.STATE == "TAG_BLOCK":
            # Only allow the next token from blocks that are still live (consistent with
            # tokens emitted so far). Eliminated blocks are already absent from live_blocks.
            allowed = set()
            for label, pos in self.live_blocks.items():
                block = self._active_blocks[label]
                if pos < len(block):
                    allowed.add(block[pos])
            return allowed
        
        if self.STATE == "SPAN_TEXT":
            # We allow all tokens that can copy the next part of the input text
            allowed = self._allowed_copy_tokens()
            if self.span_text_has_content:
                # Span close at index 0 is '</', which is a very specific token
                allowed.add(self.SPAN_CLOSE[0])
            return allowed
        
        if self.STATE == "SPAN_CLOSE":
            # Just continue the sequence
            return {self.SPAN_CLOSE[self.seq_pos]}
        return set()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply the constrained generation logic to the scores.
        """
        # Get the last generated token ID from input_ids and advance the FSM state
        last_token_id = int(input_ids[0, -1])
        curr_len = input_ids.shape[1]
        if self.prev_len > 0 and curr_len > self.prev_len:
            # Only advance the state if a new token has been generated (i.e. input_ids has increased in length)
            self._advance_state(last_token_id)
        self.prev_len = curr_len

        allowed_tokens = self._allowed_tokens()
        if not allowed_tokens:
            # Dead-end in FSM/tokenization alignment: terminate safely instead of crashing sampling.
            if self.eos_token_ids:
                allowed_tokens = set(self.eos_token_ids)
        scores = self._mask_except(scores, allowed_tokens)
        
        return scores
