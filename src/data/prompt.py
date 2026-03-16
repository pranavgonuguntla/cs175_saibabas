"""
data/prompt.py — Prompt construction from extracted keywords and sentiment label.

Converts a list of KeyBERT keywords and an optional Yelp polarity label
(0=negative, 1=positive) into a natural-language conditioning string used by
both the MDLM diffusion model and the GPT-2 baseline.

Default (natural_style=True) format:
    "positive great food friendly staff"
    "negative dry service korean bbq"

Structured format (natural_style=False, used for debugging):
    "PROMPT: sentiment=positive; keywords=great food, friendly, staff"

The same build_prompt() call is used during:
  - Dataset construction (YelpPromptedDataset.__getitem__)
  - Interactive generation (generate.py)
  - Evaluation sampling (evaluate.py, plot_guidance.py)
  - Sanity checks (baseline/sanity_check.py)

This ensures the conditioning format is identical at train and inference time.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Optional
from config import LABEL_TO_SENTIMENT, NULL_PROMPT
from data.preprocessing import normalize_text, preprocess_review


def clean_keyword(kw: str) -> str:
    """Light cleanup to keep prompts readable and consistent."""
    kw = normalize_text(kw)
    kw = kw.strip(" ,;:|")
    return kw


def build_prompt(
    keywords: List[str],
    label: Optional[int] = None,
    max_keywords: int = 6,
    include_sentiment: bool = True,
    natural_style: bool = True,
) -> str:
    """
    Build a prompt string used to condition generation.
    Examples:
      - "negative taste korean bbq service"
      - "PROMPT: negative; keywords=bbq, dry, salty"
    """
    kws = [clean_keyword(k) for k in keywords if k and isinstance(k, str)]
    # Deduplicate while preserving order
    seen = set()
    kws = [k for k in kws if not (k in seen or seen.add(k))]
    kws = kws[:max_keywords]

    sentiment = None
    if include_sentiment and label is not None and label in LABEL_TO_SENTIMENT:
        sentiment = LABEL_TO_SENTIMENT[label]

    if natural_style:
        parts = []
        if sentiment is not None:
            parts.append(sentiment)
        parts.extend(kws)
        return " ".join(parts).strip()
    else:
        # More explicit structured format if you prefer
        sent_part = f"sentiment={sentiment}" if sentiment is not None else "sentiment=<NULL>"
        kw_part = ", ".join(kws) if kws else "<NULL>"
        return f"PROMPT: {sent_part}; keywords={kw_part}"


def format_training_text(prompt: str, review_text: str) -> str:
    """
    Optional helper to create a single serialized string for debugging/logging.
    (SEDD pipelines may use separate fields; this is just convenient for inspection.)
    """
    prompt = normalize_text(prompt)
    review_text = preprocess_review(review_text)
    return f"PROMPT: {prompt}\nREVIEW: {review_text}"
