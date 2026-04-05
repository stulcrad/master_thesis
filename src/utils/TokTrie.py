from dataclasses import dataclass, field
from typing import Dict, List, Set
from transformers import AutoTokenizer

@dataclass
class TokTrieNode:
    """
    Node in a byte-level token trie.

    Contains
    --------
    - children: Dict[int, int] -> key: byte (int), value: index of child node in TokTrie.nodes
    - token_ids: List[int] -> list of token IDs whose byte string matches the path to this node
    """
    children: Dict[int, int] = field(default_factory=dict)
    token_ids: List[int] = field(default_factory=list)

class TokTrie:
    """
    Byte-level token trie inspired by llguidance toktrie.

    Each path from root to a terminal stores token IDs whose decoded byte string
    matches that exact path. This enables fast lookup of all tokens that are a prefix of a target byte suffix.
    """
    def __init__(self):
        """
        Initialize an empty trie.

        Contains
        --------
        - nodes: List[TokTrieNode] -> list of trie nodes; index 0 is the root
        - token_id_to_bytes: Dict[int, bytes] -> mapping from token ID to its byte string
        """
        self.nodes: List[TokTrieNode] = [TokTrieNode()]
        self.token_id_to_bytes: Dict[int, bytes] = {}

    def insert(self, token_bytes: bytes, token_id: int) -> None:
        """
        Insert a token into the trie.

        Args
        ---
        - token_bytes: bytes -> the byte string of the token to insert
        - token_id: int -> the token ID corresponding to token_bytes
        """
        node_idx = 0 # root index
        self.token_id_to_bytes[token_id] = token_bytes

        # Traverse the trie according to the bytes in token_bytes, creating new nodes as needed
        for b in token_bytes:
            child_idx = self.nodes[node_idx].children.get(b, None)
            if child_idx is None:
                # If no child for byte b, create a new node and link it
                child_idx = len(self.nodes) # new node index
                self.nodes[node_idx].children[b] = child_idx # link from parent to child
                self.nodes.append(TokTrieNode()) # add new node to the trie
            node_idx = child_idx # move to child node

        # at the terminal node -> store the token ID
        self.nodes[node_idx].token_ids.append(token_id)

    def prefix_search(self, remaining_bytes: bytes) -> Set[int]:
        """
        Find all token IDs in the trie that are a prefix of the given byte suffix.

        Args
        ----
        - remaining_bytes: bytes -> the byte string suffix for which we want to find prefix token IDs

        Example
        --------
        - At input we have remaining_bytes = b'Input'.
        - We start at the root and follow the path for bytes corresponding to 'I', 'n', 'p', 'u', 't'.
        - At each node along the path, we collect any token IDs stored at that node.
        - As a result, we find all token IDs whose byte string is a prefix of b'Input', such as tokens for 'I', 'In', and 'Input'.
        """
        node_idx = 0 # root index
        out: Set[int] = set() # to store token IDs that are a prefix of remaining_bytes

        # Traverse the trie according to the bytes in remaining_bytes, collecting token IDs along the path
        for b in remaining_bytes:
            child_idx = self.nodes[node_idx].children.get(b)
            if child_idx is None:
                # This should not happen in practice since we only search for prefixes that exist in the trie,
                # but we add this check for safety to avoid KeyError.
                break

            node_idx = child_idx # move to child node
            if self.nodes[node_idx].token_ids:
                # Update the output set with token IDs stored at this node, since they are a prefix of remaining_bytes
                out.update(self.nodes[node_idx].token_ids)

        return out

def build_toktrie_from_tokenizer(tokenizer: AutoTokenizer) -> TokTrie:
    """
    Build a toktrie structure from the given tokenizer and his vocabulary

    Args
    ----
    - tokenizer: AutoTokenizer -> the tokenizer from which to build the toktrie
    """
    toktrie = TokTrie()

    vocab = tokenizer.get_vocab()

    # For each token in the tokenizer's vocabulary, convert it to its byte string and insert it into the trie.
    for token, token_id in vocab.items():
        try:
            # surface = tokenizer.convert_tokens_to_string([token]) # convert token to surface form (string)
            surface = tokenizer.decode([token_id], clean_up_tokenization_spaces=False) # decode token ID to surface form (string)
        except Exception:
            # Fallback for tokenizers that may fail conversion on some special tokens.
            continue

        # Encode the surface string to bytes using UTF-8 encoding, which is the standard encoding for tokenizers.
        # This is done because the different tokens with or without 
        token_bytes = surface.encode("utf-8")
        toktrie.insert(token_bytes, token_id)

    return toktrie
