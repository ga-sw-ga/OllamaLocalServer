from transformers import AutoTokenizer

# Path to the tokenizer directory
TOKENIZER_PATH = r"C:\Users\parsa.rahmaty\.cache\llamatokenizer"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)

def decode_context(token_ids):
    """
    Decodes a list of token IDs into a human-readable string.

    :param token_ids: List of token IDs (integers)
    :return: Decoded string
    """
    return tokenizer.decode(token_ids, skip_special_tokens=True)

# Example usage
tokens = [128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18]
decoded_text = decode_context(tokens)
print(decoded_text)
