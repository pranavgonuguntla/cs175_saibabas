import re
import unicodedata
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MAX_CHARS, MAX_WORD_TOKENS, MIN_WORD_TOKENS, MAX_WORD_TOKENS_FILTER


def normalize_text(text: str) -> str:
    """
    Minimal, safe normalization:
    - Unicode normalize (NFKC)
    - Standardize whitespace
    - Strip leading/trailing spaces
    NOTE: We intentionally do NOT lowercase by default (can be toggled later).
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clip_by_chars(text: str, max_chars: int = MAX_CHARS) -> str:
    """Clip very long reviews by character count (fast, tokenizer-independent)."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip()  # avoid cutting mid-word if possible


def count_words_nltk(text: str) -> int:
    """Count word tokens using NLTK (used for quick stats / filtering)."""
    return len(nltk.word_tokenize(text))


def clip_by_word_tokens(text: str, max_tokens: int = MAX_WORD_TOKENS) -> str:
    """
    Clip by NLTK word tokens as a proxy length control.
    This is temporary until we finalize the model tokenizer (likely subword/BPE).
    """
    tokens = nltk.word_tokenize(text)
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])


def preprocess_review(text: str) -> str:
    """One-stop minimal preprocessing for Yelp review text."""
    text = normalize_text(text)
    text = clip_by_chars(text, MAX_CHARS)
    text = clip_by_word_tokens(text, MAX_WORD_TOKENS)
    return text


def passes_length_filter(
    text: str,
    min_tokens: int = MIN_WORD_TOKENS,
    max_tokens: int = MAX_WORD_TOKENS_FILTER
) -> bool:
    """
    Returns True if a preprocessed review has an acceptable length.
    Used to remove reviews that are too short or
    too long for diffusion training, before keyword extraction.
    """
    n_tokens = count_words_nltk(text)
    return min_tokens <= n_tokens <= max_tokens
