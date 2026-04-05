from transformers import LogitsProcessor, AutoTokenizer
from typing import List, Set, Optional
import torch
from utils.TokTrie import TokTrie, build_toktrie_from_tokenizer

class TrieSpanConstrainedProcessorTokenAware(LogitsProcessor):
    """
    Token-aware constrained generation processor for span classification with generative LLMs.

    Copy behavior:
    -------------
    1. Constrained copy is performed inside each tokenized token (not over full input bytes).
    2. For the current input token remainder (e.g. b"ade"), allowed copy tokens are all trie prefixes
       (e.g. b"a", b"ad", b"ade").
    3. If model emits b"a", we stay in the same input token and continue with b"de".

    Tagging behavior:
    -----------------
    Model may emit <SPAN><LABEL>...</LABEL> ... </SPAN> while preserving exact token-wise copying.

    """

    def __init__(self, labels: list[str],  input_text: str, tokenizer: AutoTokenizer,
                 toktrie: Optional[TokTrie] = None):
        # Store the labels for constructing the control tokens for opening spans.
        self.labels = labels

        # Store the tokenizer and token trie for constraint logic.
        self.tokenizer = tokenizer
        self.toktrie = toktrie if toktrie is not None else build_toktrie_from_tokenizer(tokenizer)

        # Store the input text and its token-level representation for tracking how much of the input has been copied so far.
        self.input_text = input_text

        # Token-level copy source: we track progress as (token index, byte offset inside that token).
        self.input_token_ids = tokenizer.encode(input_text, add_special_tokens=False)
        self.input_token_bytes: List[bytes] = [
            self.toktrie.token_id_to_bytes[tok_id]
            for tok_id in self.input_token_ids
        ]
        # Pointers to track how much of the input has been copied so far at the token level.
        self.input_token_ptr = 0
        self.input_token_byte_ptr = 0

        # Runtime generation bookkeeping.
        self.STATE = "OUTSIDE"
        self.seq_pos = 0 # to track which token in the current control block should come next
        self.prev_len = 0 # track the len of the generated seq

        # Tracks whether at least one copy token was emitted in the current span body.
        self.span_text_has_content = False

        # Structural token sequences.
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
        self.selected_label = None # to track which label block we are currently generating
        self._active_blocks = self.label_open_blocks # to track which set of label blocks we are generating (with or without space)

        # End tokens accepted once all input tokens are fully consumed.
        self.eos_token_ids: Set[int] = set()
        if tokenizer.eos_token_id is not None:
            self.eos_token_ids.add(tokenizer.eos_token_id)
        for tok in ["<end_of_turn>", "<|im_end|>", "<|eot_id|>"] :
            tok_id = tokenizer.convert_tokens_to_ids(tok)
            if tok_id is not None and tok_id != tokenizer.unk_token_id:
                self.eos_token_ids.add(tok_id)

    def reset(self):
        """
        Reset the processor state for a new generation sequence.
        """
        self.STATE = "OUTSIDE"
        self.seq_pos = 0
        self.input_token_ptr = 0
        self.input_token_byte_ptr = 0
        self.selected_label = None
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

    def _all_input_consumed(self) -> bool:
        """
        Check if all input tokens have been fully consumed (i.e. copied) so far, 
        based on the current token pointer and byte pointer.
        """
        return self.input_token_ptr >= len(self.input_token_bytes)

    def _normalize_token_cursor(self) -> None:
        """Advance to the next input token when current token bytes are exhausted."""
        while not self._all_input_consumed():
            cur_len = len(self.input_token_bytes[self.input_token_ptr])
            if self.input_token_byte_ptr < cur_len:
                break
            self.input_token_ptr += 1
            self.input_token_byte_ptr = 0
    
    def _current_remaining_bytes(self) -> bytes:
        """
        Get the remaining byte suffix of the current input token that has not been copied yet, 
        based on the current token pointer and byte pointer.
        """
        self._normalize_token_cursor()
        if self._all_input_consumed():
            return b""
        cur = self.input_token_bytes[self.input_token_ptr]
        return cur[self.input_token_byte_ptr:]
    
    def _allowed_copy_tokens(self) -> Set[int]:
        """
        Get the set of token IDs that can be emitted to copy the next part of the input text, 
        based on the remaining byte suffix of the current input token and the token trie.
        """
        remaining = self._current_remaining_bytes()
        if not remaining:
            return set()
        return self.toktrie.prefix_search(remaining)

    def _prefer_literal_angle_bracket(self) -> bool:
        """
        If the source text currently starts with '<', prevent starting a control tag at this step.
        This avoids confusing literal '<' in text with control token '<SPAN>'.
        """
        return self._current_remaining_bytes().startswith(b"<")

    def _consume_copy_token(self, token_id: int) -> bool:
        """
        Consume a copy token and advance the input token pointer and byte pointer 
        if the given token ID corresponds to a token that can copy the next part of the input text.
        """
        if self._all_input_consumed():
            return False
        token_bytes = self.toktrie.token_id_to_bytes.get(token_id)
        if not token_bytes:
            return False
        remaining = self._current_remaining_bytes()
        if not remaining.startswith(token_bytes):
            return False

        # Advance the byte pointer by the length of the consumed token bytes
        self.input_token_byte_ptr += len(token_bytes)
        cur_len = len(self.input_token_bytes[self.input_token_ptr])

        # If the current input token was fully consumed, move to next token.
        if self.input_token_byte_ptr >= cur_len:
            self.input_token_ptr += 1
            self.input_token_byte_ptr = 0
        return True

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
                if self._current_remaining_bytes().startswith(b" "):
                    self.input_token_byte_ptr += 1
                    self._normalize_token_cursor()
                else:
                    print(f"Warning: space-prefixed block chosen but no space at current input token {self.input_token_ptr} byte offset {self.input_token_byte_ptr}")
                self.STATE = "TAG_BLOCK"
                self.seq_pos = 1
                self.selected_label = None
                self._active_blocks = self.label_open_blocks
                return
            if nospace_match:
                self.STATE = "TAG_BLOCK"
                self.seq_pos = 1
                self.selected_label = None
                self._active_blocks = self.label_open_blocks_nospace
                return
            return

        if self.STATE == "TAG_BLOCK":
            # If the last emitted token matches exactly one token in all label blocks, we know that this block must be
            # extended to the end of the block.
            if self.selected_label is None:
                matching_blocks = {
                    label: block for label, block in self._active_blocks.items()
                    if block and self.seq_pos < len(block) and token_id == block[self.seq_pos]
                }
                if len(matching_blocks) == 1:
                    # Now we know which label block we are generating
                    self.selected_label = next(iter(matching_blocks.keys()))
                    self.seq_pos += 1
                    return
                else:
                    # More matching blocks, so just advance the seq position
                    self.seq_pos += 1
            else:
                # Should happen always since we only allow the next token in the selected block, but just in case we add this check
                if token_id == self._active_blocks[self.selected_label][self.seq_pos]:
                    self.seq_pos += 1
                    if self.seq_pos == len(self._active_blocks[self.selected_label]):
                        # We have reached the end of the selected block, so we start generating the span text
                        self.STATE = "SPAN_TEXT"
                        self.selected_label = None
                        self.seq_pos = 0
                        # Spans must copy at least one token of content, so we can avoid loops
                        self.span_text_has_content = False
                    return
                else:
                    # This should not happen in practice since we only allow the next token in the selected block,
                    # but we add this check for safety to avoid index errors.
                    print(f"Warning: {token_id} does not match expected token in selected block {self.selected_label} at position {self.seq_pos}")
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
            if self.seq_pos < len(self.SPAN_CLOSE) and token_id == self.SPAN_CLOSE[self.seq_pos]:
                self.seq_pos += 1
                if self.seq_pos == len(self.SPAN_CLOSE):
                    self.STATE = "OUTSIDE"
                    self.seq_pos = 0
                    self.span_text_has_content = False
                return

    def _allowed_tokens(self) -> Set[int]:
        """
        Get the set of allowed token IDs for the next generation step based on the current FSM state and seq position,
        using the token trie to find valid copy tokens for the remaining input bytes, and allowing the appropriate special tokens
        """
        if self.STATE == "OUTSIDE":
            # Allow all tokens that can copy the next part of the input text based on the current token-level position
            allowed = self._allowed_copy_tokens()
            # Additionally, allow the tokens that can start any of the label blocks, unless the next part of the input text starts with '<'
            if not self._prefer_literal_angle_bracket():
                if self._current_remaining_bytes().startswith(b" "):
                    allowed.update(tok[0] for tok in self.label_open_blocks.values())
                    self._active_blocks = self.label_open_blocks # pre-set the active blocks so _advance_state sees the correct variant
                else:
                    allowed.update(tok[0] for tok in self.label_open_blocks_nospace.values())
                    self._active_blocks = self.label_open_blocks_nospace # pre-set the active blocks so _advance_state sees the correct variant
            # If all input has been consumed, allow only EOS tokens to end the generation.
            if self._all_input_consumed():
                allowed = set(self.eos_token_ids)
            return allowed

        if self.STATE == "TAG_BLOCK":
            allowed = set()
            # If we have not yet disambiguated which label block we are generating, we allow any token that can be the next token in any of the label blocks
            if self.selected_label is None:
                for block in self._active_blocks.values():
                    if block and self.seq_pos < len(block):
                        allowed.add(block[self.seq_pos])
            else:
                # Now we know which label block we are generating, so only allow the next token in that specific block.
                if self.seq_pos < len(self._active_blocks[self.selected_label]):
                    allowed.add(self._active_blocks[self.selected_label][self.seq_pos])
            return allowed

        if self.STATE == "SPAN_TEXT":
            # We allow all tokens that can copy the next part of the input text
            allowed = self._allowed_copy_tokens()
            if self.span_text_has_content:
                # Span close at index 0 is '</', which is a very specific token
                allowed.add(self.SPAN_CLOSE[0])
            return allowed

        if self.STATE == "SPAN_CLOSE":
            # Just continue the span close sequence until it is complete
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
            self._advance_state(last_token_id)
        self.prev_len = curr_len

        allowed_tokens = self._allowed_tokens()
        if not allowed_tokens:
            # Dead-end in FSM/tokenization alignment: terminate safely instead of crashing sampling.
            if self.eos_token_ids:
                allowed_tokens = set(self.eos_token_ids)
        scores = self._mask_except(scores, allowed_tokens)
        
        return scores
    