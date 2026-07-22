"""
Every model has different special tags to signal the model is thinking. This file contains functions
specific for each model to determine if the model is reasoning or not, print only the output of the model, and
other model specific utilities.
"""
import torch
from typing import List

def extract_harmony_final_channel(text: str) -> str:
    """Isolate GPT-OSS's harmony 'final' channel from its raw decoded output.

    Harmony wraps the answer as `...<|channel|>final<|message|>ANSWER<|return|>`, extract
    this answer to skip any preceding 'analysis' channel content.
    """
    marker = "<|channel|>final<|message|>"
    idx = text.rfind(marker)
    if idx == -1:
        return text
    tail = text[idx + len(marker):]
    for end_marker in ("<|return|>", "<|end|>", "<|call|>", "<|start|>"):
        end_idx = tail.find(end_marker)
        if end_idx != -1:
            tail = tail[:end_idx]
    return tail.strip()

def reasoning_ended(input_ids: torch.LongTensor, reasoning_end_marker: torch.LongTensor, found_reasoning_end: bool) -> bool:
    """
    Determine if the model has finished reasoning based on the presence of a reasoning marker in the text.

    Args:
        text (str): The text output from the model.
        reasoning_end_marker (str): The marker indicating that reasoning has ended.
        found_reasoning_end (bool): A flag indicating if the reasoning end marker has been found in previous checks.
    """
    if found_reasoning_end:
        return True

    # Check if the reasoning end marker is present in the input_ids
    if reasoning_end_marker is not None and reasoning_end_marker.numel() > 0:
        # Convert input_ids to a list for easier searching
        input_ids_list = input_ids.squeeze().tolist()
        reasoning_end_marker_list = reasoning_end_marker.squeeze().tolist()

        # Check if the reasoning end marker is a subsequence of input_ids
        for i in range(len(input_ids_list) - len(reasoning_end_marker_list) + 1):
            if input_ids_list[i:i + len(reasoning_end_marker_list)] == reasoning_end_marker_list:
                return True

    return False

    # if found_reasoning_end:
    #         return True
    #     if reasoning_end_marker in text:
    #         found_reasoning_end = True
    #         return True
    #     return False
    